#!/usr/bin/env python
"""Rebuild the corrected 5 kb phase-1 Hi-C target store.

This is a script extraction of ``notebooks/corrected_target_processing.ipynb``.
It intentionally consumes the frozen membership/context tables rather than
re-deriving cell labels from external scHiCAR reference files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import gzip
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

# cooltools imports matplotlib even though this script does not plot.
_matplotlib_cache = (
    Path(tempfile.gettempdir()) / f"hi-scooby-matplotlib-{os.getuid()}"
)
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
_xdg_cache = Path(tempfile.gettempdir()) / f"hi-scooby-cache-{os.getuid()}"
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected
import h5py
from numcodecs import Blosc
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr


AUTOSOMES = tuple(f"chr{index}" for index in range(1, 20))
BIN_SIZE = 5_000
TARGET_BP = 1_000_000
TARGET_BINS = TARGET_BP // BIN_SIZE
MASKED_DIAGONALS = 4
COARSEGRAIN_CUTOFF = 2
COARSEGRAIN_MAX_LEVELS = 5
COARSEGRAIN_MIN_SHAPE = 8
EXPECTED_EDGE_RATIO = 1.001
EXPECTED_SPLIT_COUNTS = {
    "train": 2_753,
    "validation": 473,
    "test": 443,
}
EXPECTED_CONTEXTS = 20
EXPECTED_PAIR_FILES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild corrected adaptive-coarse-grained phase-1 targets."
        )
    )
    parser.add_argument(
        "--filtered-pairs",
        type=Path,
        default=Path("data/processed/phase1_hic/filtered_pairs"),
        help="Directory containing the 40 filtered cis-UU pairs files.",
    )
    parser.add_argument(
        "--genome-fai",
        type=Path,
        default=Path("data/external/mm10_genome/mm10.fa.fai"),
    )
    parser.add_argument(
        "--tiles",
        type=Path,
        default=Path("data/processed/multiome/tiles.parquet"),
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path(
            "data/processed/multiome_rna/phase1_membership.parquet"
        ),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path(
            "data/processed/multiome_rna/target_contexts.parquet"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/multiome_rna"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing generated targets/intermediates after a "
            "successful rebuild."
        ),
    )
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required inputs:\n{formatted}")


def _load_contract(
    tiles_path: Path,
    membership_path: Path,
    contexts_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/8] Validating tile, membership, and context tables")
    tiles = pd.read_parquet(tiles_path).reset_index(drop=True)
    membership = pd.read_parquet(membership_path).reset_index(drop=True)
    contexts = (
        pd.read_parquet(contexts_path)
        .sort_values("context_index")
        .reset_index(drop=True)
    )

    split_counts = tiles["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected tile split counts: {split_counts}")
    if not (tiles["target_end"] - tiles["target_start"]).eq(TARGET_BP).all():
        raise ValueError("Every target interval must span exactly 1 Mb")
    if not tiles["target_start"].mod(BIN_SIZE).eq(0).all():
        raise ValueError("Every target interval must be 5 kb aligned")
    if tiles["tile_id"].duplicated().any():
        raise ValueError("tile_id must be unique")
    if tiles["tile_index"].duplicated().any():
        raise ValueError("tile_index must be unique")

    if len(contexts) != EXPECTED_CONTEXTS:
        raise ValueError(
            f"Expected {EXPECTED_CONTEXTS} target contexts; found "
            f"{len(contexts)}"
        )
    if not np.array_equal(
        contexts["context_index"].to_numpy(),
        np.arange(len(contexts)),
    ):
        raise ValueError("context_index must be contiguous and row aligned")
    if membership.duplicated(["library_id", "dna_barcode"]).any():
        raise ValueError("Membership contains duplicate DNA cell keys")
    if not set(membership["context_index"]).issubset(
        set(contexts["context_index"])
    ):
        raise ValueError("Membership references an unknown context")

    observed = (
        membership.groupby(
            ["context_index", "cell_type"],
            as_index=False,
            sort=True,
        )
        .agg(
            n_cells=("cell_id", "nunique"),
            valid_pairs=("valid_pairs", "sum"),
        )
        .sort_values("context_index")
        .reset_index(drop=True)
    )
    expected = contexts[
        [
            "context_index",
            "cell_type",
            "n_cells",
            "valid_pairs",
        ]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
        check_like=False,
    )

    print(
        f"      {len(tiles):,} tiles; {len(membership):,} cells; "
        f"{len(contexts)} contexts"
    )
    return tiles, membership, contexts


def _write_autosomal_chromsizes(
    genome_fai: Path,
    output_path: Path,
) -> None:
    fai = pd.read_csv(
        genome_fai,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["chrom", "length"],
    ).set_index("chrom")
    missing = [chrom for chrom in AUTOSOMES if chrom not in fai.index]
    if missing:
        raise ValueError(f"Genome index is missing autosomes: {missing}")
    chromsizes = fai.loc[list(AUTOSOMES), ["length"]].reset_index()
    chromsizes.to_csv(output_path, sep="\t", header=False, index=False)


def _library_id(path: Path) -> str:
    match = re.search(r"_DNA_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse library number from {path.name}")
    return f"DNA{int(match.group(1)):02d}"


def _route_pairs(
    pair_files: list[Path],
    membership: pd.DataFrame,
    contexts: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Path], Path]:
    print("[2/8] Routing filtered contacts by target context")
    output_dir.mkdir(parents=True)
    cell_type_by_key = {
        (str(row.library_id), str(row.dna_barcode)): str(row.cell_type)
        for row in membership.itertuples(index=False)
    }
    context_paths = {
        str(cell_type): output_dir / f"{cell_type}.pairs.gz"
        for cell_type in contexts["cell_type"]
    }
    all_cells_path = output_dir / "all_cells.pairs.gz"
    routed_counts: defaultdict[str, int] = defaultdict(int)

    with ExitStack() as stack:
        context_handles = {
            cell_type: stack.enter_context(gzip.open(path, "wt"))
            for cell_type, path in context_paths.items()
        }
        all_cells_handle = stack.enter_context(
            gzip.open(all_cells_path, "wt")
        )
        for pairs_path in tqdm(pair_files, unit="library"):
            library_id = _library_id(pairs_path)
            with gzip.open(pairs_path, "rt") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 9:
                        raise ValueError(
                            f"Malformed pair at {pairs_path.name}:"
                            f"{line_number}"
                        )
                    dna_barcode = fields[0].split(":", 1)[0]
                    cell_type = cell_type_by_key.get(
                        (library_id, dna_barcode)
                    )
                    if cell_type is None:
                        continue
                    context_handles[cell_type].write(line)
                    all_cells_handle.write(line)
                    routed_counts[cell_type] += 1

    expected_counts = contexts.set_index("cell_type")[
        "valid_pairs"
    ].astype(int).to_dict()
    if dict(routed_counts) != expected_counts:
        differences = {
            cell_type: (
                routed_counts.get(cell_type, 0),
                expected_counts[cell_type],
            )
            for cell_type in expected_counts
            if routed_counts.get(cell_type, 0) != expected_counts[cell_type]
        }
        raise ValueError(
            "Routed contact counts do not match membership totals: "
            f"{differences}"
        )
    print(f"      Routed {sum(routed_counts.values()):,} contacts")
    return context_paths, all_cells_path


def _cload_cooler(
    chromsizes_path: Path,
    pairs_path: Path,
    cooler_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "cooler",
        "cload",
        "pairs",
        "--zero-based",
        "--chrom1",
        "2",
        "--pos1",
        "3",
        "--chrom2",
        "4",
        "--pos2",
        "5",
        f"{chromsizes_path}:{BIN_SIZE}",
        str(pairs_path),
        str(cooler_path),
    ]
    subprocess.run(command, check=True)


def _build_coolers(
    context_paths: dict[str, Path],
    all_cells_pairs_path: Path,
    contexts: pd.DataFrame,
    chromsizes_path: Path,
    cooler_dir: Path,
) -> dict[str, Path]:
    print("[3/8] Building 5 kb pseudobulk Coolers")
    cooler_dir.mkdir(parents=True)
    cooler_paths = {
        cell_type: cooler_dir / f"{cell_type}.5kb.cool"
        for cell_type in context_paths
    }
    all_cells_cooler_path = cooler_dir / "all_cells.5kb.cool"
    builds = [
        *[
            (cell_type, context_paths[cell_type], cooler_paths[cell_type])
            for cell_type in contexts["cell_type"].astype(str)
        ],
        ("all_cells", all_cells_pairs_path, all_cells_cooler_path),
    ]
    for _, pairs_path, cooler_path in tqdm(builds, unit="cooler"):
        _cload_cooler(chromsizes_path, pairs_path, cooler_path)

    print("[4/8] Fitting and copying shared all-cell visibility weights")
    all_cells = cooler.Cooler(str(all_cells_cooler_path))
    shared_weights, stats = cooler.balance_cooler(
        all_cells,
        cis_only=True,
        ignore_diags=0,
        min_nnz=10,
        mad_max=5,
        tol=1e-5,
        store=True,
    )
    if not bool(np.all(stats["converged"])):
        raise RuntimeError("All-cell ICE balancing did not converge")

    shared_bins = all_cells.bins()[:][["chrom", "start", "end"]]
    with h5py.File(all_cells_cooler_path, "r") as handle:
        weight_attrs = dict(handle["bins/weight"].attrs)

    for cell_type in tqdm(contexts["cell_type"], unit="context"):
        path = cooler_paths[str(cell_type)]
        context_cooler = cooler.Cooler(str(path))
        if not context_cooler.bins()[:][
            ["chrom", "start", "end"]
        ].equals(shared_bins):
            raise ValueError(f"Cooler bin mismatch for {cell_type}")
        with h5py.File(path, "r+") as handle:
            bins_group = handle["bins"]
            if "weight" in bins_group:
                del bins_group["weight"]
            dataset = bins_group.create_dataset(
                "weight",
                data=shared_weights,
            )
            for key, value in weight_attrs.items():
                dataset.attrs[key] = value

    expected_counts = contexts.set_index("cell_type")["valid_pairs"]
    for cell_type, expected_count in expected_counts.items():
        observed_count = int(
            cooler.Cooler(
                str(cooler_paths[str(cell_type)])
            ).pixels()[:]["count"].sum()
        )
        if observed_count != int(expected_count):
            raise ValueError(
                f"{cell_type}: expected {expected_count:,} contacts; "
                f"Cooler contains {observed_count:,}"
            )
    all_count = int(all_cells.pixels()[:]["count"].sum())
    if all_count != int(expected_counts.sum()):
        raise ValueError(
            f"All-cell Cooler contains {all_count:,} contacts; "
            f"expected {int(expected_counts.sum()):,}"
        )

    print(
        f"      ICE finite weights: "
        f"{np.isfinite(shared_weights).sum():,}/{len(shared_weights):,}"
    )
    cooler_paths["all_cells"] = all_cells_cooler_path
    return cooler_paths


def _compute_coarse_maps(
    cell_type: str,
    cooler_path: Path,
    n_cells: int,
    tiles: pd.DataFrame,
) -> np.ndarray:
    context_cooler = cooler.Cooler(str(cooler_path))
    raw_selector = context_cooler.matrix(balance=False, sparse=True)
    bin_table = context_cooler.bins()[:][["chrom", "weight"]]
    weights_by_chrom = {
        str(chrom): frame["weight"].to_numpy(dtype=np.float32)
        for chrom, frame in bin_table.groupby(
            "chrom",
            observed=True,
            sort=False,
        )
    }
    raw_by_chrom: dict[str, object] = {}
    maps = np.empty(
        (len(tiles), TARGET_BINS, TARGET_BINS),
        dtype=np.float32,
    )

    for tile_row, tile in enumerate(
        tqdm(
            tiles.itertuples(index=False),
            total=len(tiles),
            desc=f"{cell_type} coarse-grain",
            unit="tile",
            leave=False,
        )
    ):
        chrom = str(tile.chrom)
        if chrom not in raw_by_chrom:
            raw_by_chrom[chrom] = (
                raw_selector.fetch(chrom).tocsr().astype(np.float32)
            )
        start = int(tile.target_start) // BIN_SIZE
        end = start + TARGET_BINS
        raw_sum = raw_by_chrom[chrom][start:end, start:end].toarray()
        if raw_sum.shape != (TARGET_BINS, TARGET_BINS):
            raise ValueError(
                f"{tile.tile_id}: fetched raw map has shape {raw_sum.shape}"
            )
        weights = weights_by_chrom[chrom][start:end]
        if weights.shape != (TARGET_BINS,):
            raise ValueError(
                f"{tile.tile_id}: fetched {len(weights)} visibility weights"
            )
        observed = (
            raw_sum
            * weights[:, None]
            * weights[None, :]
            / float(n_cells)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            coarse = adaptive_coarsegrain(
                observed,
                raw_sum,
                cutoff=COARSEGRAIN_CUTOFF,
                max_levels=COARSEGRAIN_MAX_LEVELS,
                min_shape=COARSEGRAIN_MIN_SHAPE,
            )
        maps[tile_row] = np.asarray(coarse, dtype=np.float32)
    return maps


def _fit_train_expected(
    coarse_maps: np.ndarray,
    train_tile_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_pixels = np.zeros(TARGET_BINS, dtype=np.float64)
    n_pixels = np.zeros(TARGET_BINS, dtype=np.int64)
    exact_edges = np.arange(TARGET_BINS + 1)

    for coarse in tqdm(
        coarse_maps[train_tile_mask],
        desc="Fit train expectation",
        unit="tile",
        leave=False,
    ):
        fit_mask = np.isfinite(coarse) & (coarse != 0)
        _, distance_edges, tile_sums, tile_counts = observed_over_expected(
            coarse,
            mask=fit_mask,
            dist_bin_edge_ratio=EXPECTED_EDGE_RATIO,
        )
        if not np.array_equal(distance_edges, exact_edges):
            raise ValueError("Expected bins do not match exact diagonals")
        sum_pixels += tile_sums
        n_pixels += tile_counts

    expected = np.full(TARGET_BINS, np.nan, dtype=np.float32)
    has_data = n_pixels > 0
    expected[has_data] = (
        sum_pixels[has_data] / n_pixels[has_data]
    ).astype(np.float32)
    if not np.any(np.isfinite(expected) & (expected > 0)):
        raise ValueError("No positive expected values were fitted")
    return expected, sum_pixels, n_pixels


def _normalize_targets(
    coarse_maps: np.ndarray,
    expected: np.ndarray,
    pixel_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    positive_expected = expected[
        np.isfinite(expected) & (expected > 0)
    ]
    epsilon = float(positive_expected.min())
    expected_matrix = expected[pixel_distance]
    valid_mask = (
        np.isfinite(coarse_maps)
        & np.isfinite(expected_matrix)[None, :, :]
        & (expected_matrix[None, :, :] > 0)
        & (pixel_distance[None, :, :] >= MASKED_DIAGONALS)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        targets = np.log(
            (coarse_maps + epsilon)
            / (expected_matrix[None, :, :] + epsilon)
        )
    targets = (
        (targets + targets.transpose(0, 2, 1)) * 0.5
    ).astype(np.float32)
    targets[~valid_mask] = np.nan

    if not np.array_equal(np.isfinite(targets), valid_mask):
        raise ValueError("Finite target pixels do not match the valid mask")
    if not np.array_equal(
        targets,
        targets.transpose(0, 2, 1),
        equal_nan=True,
    ):
        raise ValueError("Normalized targets are not symmetric")
    return targets, valid_mask, epsilon


def _create_target_store(
    output_path: Path,
    contexts: pd.DataFrame,
    tiles: pd.DataFrame,
    tiles_path: Path,
) -> tuple[zarr.Array, zarr.Array]:
    compressor = Blosc(
        cname="lz4",
        clevel=1,
        shuffle=Blosc.SHUFFLE,
    )
    root = zarr.open_group(str(output_path), mode="w")
    context_ids = contexts["cell_type"].astype(str).tolist()
    root.attrs.update(
        {
            "dimension_order": ["context", "tile", "bin1", "bin2"],
            "context_ids": context_ids,
            "tiles_table": str(tiles_path),
            "bin_size_bp": BIN_SIZE,
            "masked_diagonals": MASKED_DIAGONALS,
            "expectation_tiles": "train",
            "target_transform": "signed_log_observed_expected",
        }
    )
    shape = (
        len(context_ids),
        len(tiles),
        TARGET_BINS,
        TARGET_BINS,
    )
    target_array = root.create_dataset(
        "targets",
        shape=shape,
        chunks=(1, 64, TARGET_BINS, TARGET_BINS),
        dtype="float16",
        fill_value=np.nan,
        compressor=compressor,
    )
    mask_array = root.create_dataset(
        "valid_mask",
        shape=shape,
        chunks=target_array.chunks,
        dtype="bool",
        fill_value=False,
        compressor=compressor,
    )
    return target_array, mask_array


def _build_targets(
    target_store_path: Path,
    expected_path: Path,
    cooler_paths: dict[str, Path],
    tiles: pd.DataFrame,
    contexts: pd.DataFrame,
    tiles_path: Path,
) -> None:
    print("[5/8] Creating corrected target store")
    target_array, mask_array = _create_target_store(
        target_store_path,
        contexts,
        tiles,
        tiles_path,
    )
    train_tile_mask = tiles["split"].eq("train").to_numpy()
    pixel_distance = np.abs(
        np.arange(TARGET_BINS)[:, None]
        - np.arange(TARGET_BINS)[None, :]
    )
    expected_rows: list[dict[str, object]] = []

    print("[6/8] Coarse-graining and normalizing each context")
    for context in tqdm(
        contexts.itertuples(index=False),
        total=len(contexts),
        desc="Target contexts",
        unit="context",
    ):
        cell_type = str(context.cell_type)
        coarse_maps = _compute_coarse_maps(
            cell_type,
            cooler_paths[cell_type],
            int(context.n_cells),
            tiles,
        )
        expected, sums, counts = _fit_train_expected(
            coarse_maps,
            train_tile_mask,
        )
        targets, valid_mask, epsilon = _normalize_targets(
            coarse_maps,
            expected,
            pixel_distance,
        )
        stored_targets = targets.astype(np.float16)
        if not np.array_equal(
            np.isfinite(stored_targets),
            valid_mask,
        ):
            raise ValueError(
                f"float16 conversion changed the mask for {cell_type}"
            )
        target_array[int(context.context_index)] = stored_targets
        mask_array[int(context.context_index)] = valid_mask
        expected_rows.extend(
            {
                "context_index": int(context.context_index),
                "cell_type": cell_type,
                "dist": distance,
                "distance_bp": distance * BIN_SIZE,
                "sum_pixels": sums[distance],
                "n_pixels": counts[distance],
                "expected_per_cell": expected[distance],
                "eps": epsilon,
            }
            for distance in range(TARGET_BINS)
        )
        del coarse_maps, targets, stored_targets, valid_mask

    pd.DataFrame(expected_rows).to_parquet(expected_path, index=False)


def _validate_target_store(
    target_store_path: Path,
    expected_path: Path,
    contexts: pd.DataFrame,
    tiles: pd.DataFrame,
) -> None:
    print("[7/8] Reopening and validating generated targets")
    root = zarr.open_group(str(target_store_path), mode="r")
    expected_shape = (
        len(contexts),
        len(tiles),
        TARGET_BINS,
        TARGET_BINS,
    )
    if root["targets"].shape != expected_shape:
        raise ValueError(
            f"Unexpected target shape: {root['targets'].shape}"
        )
    if root["valid_mask"].shape != expected_shape:
        raise ValueError(
            f"Unexpected mask shape: {root['valid_mask'].shape}"
        )
    if list(root.attrs["context_ids"]) != contexts[
        "cell_type"
    ].astype(str).tolist():
        raise ValueError("Stored context order does not match contexts table")

    diagonal = np.arange(TARGET_BINS)
    sample_tiles = sorted(
        {
            0,
            len(tiles) // 2,
            len(tiles) - 1,
        }
    )
    for context_index in range(len(contexts)):
        for tile_index in sample_tiles:
            target = root["targets"][context_index, tile_index]
            valid = root["valid_mask"][context_index, tile_index]
            if not np.array_equal(np.isfinite(target), valid):
                raise ValueError("Reopened target/mask disagreement")
            if not np.array_equal(target, target.T, equal_nan=True):
                raise ValueError("Reopened target is not symmetric")
            for offset in range(MASKED_DIAGONALS):
                if valid[diagonal[: TARGET_BINS - offset], diagonal[offset:]].any():
                    raise ValueError("A masked diagonal contains valid pixels")

    expected = pd.read_parquet(expected_path)
    if len(expected) != len(contexts) * TARGET_BINS:
        raise ValueError(
            f"Expected table has {len(expected):,} rows; expected "
            f"{len(contexts) * TARGET_BINS:,}"
        )


def _publish(
    temporary_root: Path,
    output_root: Path,
    overwrite: bool,
) -> None:
    publications = _publication_paths(temporary_root, output_root)
    existing = [destination for _, destination in publications if destination.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to replace existing outputs without --overwrite:\n"
            f"{formatted}"
        )

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for source, destination in publications:
            if destination.exists():
                backup = temporary_root / f".backup-{destination.name}"
                os.replace(destination, backup)
                backups.append((backup, destination))
            os.replace(source, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for backup, destination in reversed(backups):
            os.replace(backup, destination)
        raise
    for backup, _ in backups:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _publication_paths(
    temporary_root: Path,
    output_root: Path,
) -> list[tuple[Path, Path]]:
    return [
        (
            temporary_root / "hic_pairs_by_cell_type",
            output_root / "hic_pairs_by_cell_type",
        ),
        (
            temporary_root / "hic_coolers",
            output_root / "hic_coolers",
        ),
        (
            temporary_root / "hic_targets.zarr",
            output_root / "hic_targets.zarr",
        ),
        (
            temporary_root
            / "hic_expected_train_after_coarsegrain.5kb.parquet",
            output_root
            / "hic_expected_train_after_coarsegrain.5kb.parquet",
        ),
    ]


def main() -> int:
    args = parse_args()
    filtered_pairs_dir = _resolved(args.filtered_pairs)
    genome_fai = _resolved(args.genome_fai)
    tiles_path = _resolved(args.tiles)
    membership_path = _resolved(args.membership)
    contexts_path = _resolved(args.contexts)
    output_root = _resolved(args.output_root)
    _require_paths(
        [
            filtered_pairs_dir,
            genome_fai,
            tiles_path,
            membership_path,
            contexts_path,
        ]
    )
    pair_files = sorted(
        filtered_pairs_dir.glob("*.cis_uu_autosomes.pairs.gz")
    )
    if len(pair_files) != EXPECTED_PAIR_FILES:
        raise ValueError(
            f"Expected {EXPECTED_PAIR_FILES} filtered pairs files; "
            f"found {len(pair_files)}"
        )

    existing_outputs = [
        destination
        for _, destination in _publication_paths(
            Path("__not_created__"),
            output_root,
        )
        if destination.exists()
    ]
    if existing_outputs and not args.overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Generated outputs already exist. Re-run with --overwrite "
            "to replace them after a successful private rebuild:\n"
            f"{formatted}"
        )

    tiles, membership, contexts = _load_contract(
        tiles_path,
        membership_path,
        contexts_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        output_root / f".phase1-target-build.partial-{os.getpid()}"
    )
    if temporary_root.exists():
        raise FileExistsError(
            f"Private build directory already exists: {temporary_root}"
        )
    temporary_root.mkdir()

    try:
        chromsizes_path = temporary_root / "mm10.autosomes.chrom.sizes"
        _write_autosomal_chromsizes(genome_fai, chromsizes_path)
        context_paths, all_cells_pairs_path = _route_pairs(
            pair_files,
            membership,
            contexts,
            temporary_root / "hic_pairs_by_cell_type",
        )
        cooler_paths = _build_coolers(
            context_paths,
            all_cells_pairs_path,
            contexts,
            chromsizes_path,
            temporary_root / "hic_coolers",
        )
        target_store_path = temporary_root / "hic_targets.zarr"
        expected_path = (
            temporary_root
            / "hic_expected_train_after_coarsegrain.5kb.parquet"
        )
        _build_targets(
            target_store_path,
            expected_path,
            cooler_paths,
            tiles,
            contexts,
            tiles_path,
        )
        _validate_target_store(
            target_store_path,
            expected_path,
            contexts,
            tiles,
        )
        print("[8/8] Publishing complete target build")
        _publish(temporary_root, output_root, args.overwrite)
    except Exception:
        print(
            f"Build failed; removing private partial output: "
            f"{temporary_root}",
            file=sys.stderr,
        )
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"Phase-1 targets: {output_root / 'hic_targets.zarr'}")
    print(
        "Train expectation: "
        f"{output_root / 'hic_expected_train_after_coarsegrain.5kb.parquet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
