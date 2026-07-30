"""One-pass AlphaGenome orchestration for Hi-Scooby inference."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from hi_scooby import __version__
from hi_scooby.inference.smooth import SmoothPredictor
from hi_scooby.inference.sparse import SparsePredictor
from hi_scooby.io import (
    ContactMapOutput,
    SPARSE_BINS,
    SPARSE_COUNT_FIELDS,
    SPARSE_FLOAT_FIELDS,
    build_output_tile_table,
)
from hi_scooby.resources import ResourceRegistry, load_resources
from hi_scooby.rna import RNAInputSummary, inspect_wide_rna


EmbeddingProvider = Callable[[pd.Series], np.ndarray]

SPARSE_PAIR_INPUT_COLUMNS = (
    "pair_id",
    "chrom",
    "bin_i",
    "bin_j",
    "distance_bp",
    "distance_bin",
    "tile_row",
    "exposure",
    "distance_offset",
)


def cached_embedding_provider(cache_directory: str | Path) -> EmbeddingProvider:
    """Return a development provider backed by historical NPY embeddings."""
    cache = Path(cache_directory).expanduser().resolve()
    if not cache.is_dir():
        raise FileNotFoundError(
            f"Embedding cache directory does not exist: {cache}"
        )

    def provide(tile: pd.Series) -> np.ndarray:
        stored_name = Path(str(tile["embedding_path"])).name
        candidates = (
            cache / stored_name,
            cache / f"{tile['chrom']}_{int(tile['input_start'])}.npy",
        )
        path = next((value for value in candidates if value.is_file()), None)
        if path is None:
            checked = "\n".join(f"  - {value}" for value in candidates)
            raise FileNotFoundError(
                f"No cached embedding for {tile['tile_id']}.\n"
                f"Checked:\n{checked}"
            )
        return np.load(path, allow_pickle=False)

    return provide


def _live_embedding_provider(
    resources: ResourceRegistry,
) -> EmbeddingProvider:
    # AlphaGenome imports TensorFlow/JAX and performs substantial framework
    # initialization. Keep it out of cached and sparse-only startup paths.
    from hi_scooby.alphagenome import load_pair_embedder

    fasta = resources.resolve("mm10_fasta")
    resources.resolve("mm10_fasta_index")
    resources.resolve("alphagenome_window_manifest")
    embedder = load_pair_embedder(fasta)

    def provide(tile: pd.Series) -> np.ndarray:
        return embedder.embed_interval(
            str(tile["chrom"]),
            int(tile["input_start"]),
            int(tile["input_end"]),
        )

    return provide


def _rna_python_command(resources: ResourceRegistry) -> list[str]:
    override = os.environ.get("HI_SCOOBY_RNA_PYTHON")
    if override:
        executable = Path(override).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"HI_SCOOBY_RNA_PYTHON is not a file: {executable}"
            )
        return [str(executable)]

    conda = shutil.which("conda")
    if conda is not None:
        return [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            "hi-scooby-rna",
            "python",
        ]

    raise RuntimeError(
        "Smooth inference requires the historical RNA-SCVI environment. "
        "Create environment-rna.yml as the named environment "
        "'hi-scooby-rna', or set HI_SCOOBY_RNA_PYTHON to its Python "
        "executable."
    )


def encode_rna_centroids(
    rna_counts: Path,
    resources: ResourceRegistry,
) -> pd.DataFrame:
    """Run the frozen RNA encoder in its isolated historical environment."""
    model_path = resources.resolve("rna_scvi_14_model")
    command = _rna_python_command(resources)

    with tempfile.TemporaryDirectory(
        prefix="hi_scooby_rna_",
    ) as directory:
        output = Path(directory) / "centroids.parquet"
        worker_command = [
            *command,
            "-m",
            "hi_scooby.rna_worker",
            "--input",
            str(rna_counts),
            "--model",
            str(model_path),
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        # This is ``<repo>/src`` in an editable/source checkout and the
        # containing site-packages directory in an installed wheel.
        source = Path(__file__).resolve().parents[2]
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{source}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(source)
        )
        print(
            "[RNA-SCVI] Starting isolated historical-environment inference",
            flush=True,
        )
        subprocess.run(
            worker_command,
            check=True,
            env=environment,
        )
        if not output.is_file():
            raise RuntimeError("RNA-SCVI worker produced no centroid table")
        return pd.read_parquet(output)


def _select_smooth_contexts(
    centroids: pd.DataFrame,
    context_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"cell_type", "n_cells", "embedding"}
    if missing := required - set(centroids.columns):
        raise ValueError(
            f"RNA centroid table lacks columns: {sorted(missing)}"
        )
    if centroids["cell_type"].astype(str).duplicated().any():
        raise ValueError("RNA centroid table contains duplicate cell types")

    indexed = centroids.copy()
    indexed["cell_type"] = indexed["cell_type"].astype(str)
    indexed = indexed.set_index("cell_type")
    missing_contexts = [
        context for context in context_ids if context not in indexed.index
    ]
    if missing_contexts:
        raise ValueError(
            "RNA input lacks trained smooth contexts: "
            f"{missing_contexts}"
        )

    selected = indexed.loc[list(context_ids)].reset_index()
    embeddings = np.stack(selected["embedding"]).astype(np.float32)
    if embeddings.shape != (len(context_ids), 14):
        raise ValueError(
            f"Smooth RNA embeddings have shape {embeddings.shape}; "
            f"expected {(len(context_ids), 14)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Smooth RNA embeddings contain non-finite values")

    selected.insert(
        0,
        "context_index",
        np.arange(len(selected), dtype=np.int16),
    )
    selected["smooth_output"] = True

    omitted = sorted(set(indexed.index) - set(context_ids))
    if omitted:
        print(
            "[smooth] Omitting cell types outside the 20 trained target "
            f"contexts: {', '.join(omitted)}",
            flush=True,
        )
    return selected, embeddings


def _input_cell_types(summary: RNAInputSummary) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_type": list(summary.cell_type_counts),
            "n_cells": list(summary.cell_type_counts.values()),
            "smooth_output": False,
        }
    )


def _load_sparse_pairs(
    path: Path,
    *,
    selected_tile_count: int,
    full_tile_count: int,
) -> pd.DataFrame:
    filters = (
        None
        if selected_tile_count == full_tile_count
        else [("tile_row", "<", selected_tile_count)]
    )
    table = pq.read_table(
        path,
        columns=list(SPARSE_PAIR_INPUT_COLUMNS),
        filters=filters,
    )
    pairs = table.to_pandas().sort_values(
        "pair_id",
        kind="stable",
    ).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("Selected tiles contain no canonical sparse pairs")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("Canonical sparse pair IDs are duplicated")
    if (
        np.any(pairs["tile_row"].to_numpy(np.int64) < 0)
        or np.any(
            pairs["tile_row"].to_numpy(np.int64)
            >= selected_tile_count
        )
    ):
        raise ValueError("Canonical pair owner is outside selected tiles")

    original_ids = pairs["pair_id"].to_numpy(np.int64)
    expected_ids = np.arange(len(pairs), dtype=np.int64)
    if selected_tile_count == full_tile_count:
        if not np.array_equal(original_ids, expected_ids):
            raise ValueError(
                "Full canonical sparse pair IDs are not row aligned"
            )
    else:
        pairs["pair_id"] = expected_ids

    pairs["chrom"] = pairs["chrom"].astype("category")
    return pairs


def _owner_pair_positions(
    pairs: pd.DataFrame,
    tile_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tile_row = pairs["tile_row"].to_numpy(np.int64)
    order = np.argsort(tile_row, kind="stable")
    counts = np.bincount(tile_row, minlength=tile_count)
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        )
    )
    return order, offsets


def _canonical_pair_frame(
    owner_pairs: pd.DataFrame,
    prediction,
    tile: pd.Series,
) -> pd.DataFrame:
    frame = prediction.to_frame().rename(
        columns={
            "row": "owner_row",
            "column": "owner_column",
        }
    )
    frame.insert(1, "tile_row", int(tile["tile_index"]))
    frame.insert(2, "tile_id", str(tile["tile_id"]))
    frame.insert(3, "chrom", str(tile["chrom"]))
    frame.insert(
        4,
        "bin_i",
        owner_pairs["bin_i"].to_numpy(np.int64),
    )
    frame.insert(
        5,
        "bin_j",
        owner_pairs["bin_j"].to_numpy(np.int64),
    )
    return frame


def _allocate_sparse_predictions(pair_count: int) -> dict[str, np.ndarray]:
    return {
        "expected_contacts_per_million": np.full(
            pair_count, np.nan, np.float32
        ),
        "expected_count": np.full(pair_count, np.nan, np.float32),
        "nb2_dispersion": np.full(pair_count, np.nan, np.float32),
        "residual_score": np.full(pair_count, np.nan, np.float32),
        "predictive_lower": np.full(pair_count, -1, np.int64),
        "predictive_upper": np.full(pair_count, -1, np.int64),
        "simulated_count": np.full(pair_count, -1, np.int64),
    }


def _record_sparse_prediction(
    storage: dict[str, np.ndarray],
    prediction,
) -> None:
    pair_ids = prediction.pair_id.astype(np.int64, copy=False)
    for name, output in storage.items():
        output[pair_ids] = np.asarray(getattr(prediction, name))


def _validate_sparse_prediction_storage(
    storage: dict[str, np.ndarray],
) -> None:
    for name in SPARSE_FLOAT_FIELDS:
        if name == "tile_band_probability":
            continue
        if not np.isfinite(storage[name]).all():
            raise RuntimeError(
                f"Canonical sparse field {name} is incomplete"
            )
    for name in SPARSE_COUNT_FIELDS:
        if np.any(storage[name] < 0):
            raise RuntimeError(
                f"Canonical sparse count field {name} is incomplete"
            )


def _chromosome_pair_index(
    pairs: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for chromosome, group in pairs.groupby(
        "chrom",
        observed=True,
        sort=False,
    ):
        left = group["bin_i"].to_numpy(np.int64) // 10_000
        right = group["bin_j"].to_numpy(np.int64) // 10_000
        stride = int(max(left.max(initial=0), right.max(initial=0))) + 1
        keys = left * stride + right
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        if np.any(np.diff(sorted_keys) == 0):
            raise ValueError(
                f"Canonical sparse coordinate keys duplicate on {chromosome}"
            )
        pair_ids = group["pair_id"].to_numpy(np.int64)[order]
        output[str(chromosome)] = (sorted_keys, pair_ids, stride)
    return output


def _sparse_pair_grid(
    tile: pd.Series,
    index: dict[str, tuple[np.ndarray, np.ndarray, int]],
) -> np.ndarray:
    matrix = np.full((SPARSE_BINS, SPARSE_BINS), -1, np.int64)
    chromosome = str(tile["chrom"])
    if chromosome not in index:
        return matrix

    sorted_keys, sorted_ids, stride = index[chromosome]
    start_bin = int(tile["sparse_target_start"]) // 10_000
    for distance in range(25, 100):
        local_i = np.arange(SPARSE_BINS - distance, dtype=np.int64)
        local_j = local_i + distance
        candidate_keys = (
            (start_bin + local_i) * stride
            + start_bin
            + local_j
        )
        positions = np.searchsorted(sorted_keys, candidate_keys)
        within = positions < len(sorted_keys)
        safe = np.minimum(positions, len(sorted_keys) - 1)
        present = within & (sorted_keys[safe] == candidate_keys)
        if present.any():
            pair_ids = sorted_ids[safe[present]]
            rows = local_i[present]
            columns = local_j[present]
            matrix[rows, columns] = pair_ids
            matrix[columns, rows] = pair_ids
    return matrix


def _sparse_tile_values(
    pair_grid: np.ndarray,
    pairs: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    valid = pair_grid >= 0
    safe = np.maximum(pair_grid, 0)
    output: dict[str, np.ndarray] = {
        "pair_id": pair_grid,
        "valid_mask": valid,
    }

    for name in SPARSE_FLOAT_FIELDS:
        values = np.full(pair_grid.shape, np.nan, np.float32)
        if name != "tile_band_probability":
            values[valid] = predictions[name][safe[valid]]
        output[name] = values
    for name in SPARSE_COUNT_FIELDS:
        values = np.full(pair_grid.shape, -1, np.int64)
        values[valid] = predictions[name][safe[valid]]
        output[name] = values

    distance = pairs["distance_bp"].to_numpy(np.int64)
    upper = np.triu(np.ones(pair_grid.shape, dtype=bool), k=1)
    expected = output["expected_contacts_per_million"]
    probability = output["tile_band_probability"]
    for minimum, maximum in ((250_000, 500_000), (500_000, 1_000_000)):
        selected = (
            valid
            & (distance[safe] >= minimum)
            & (distance[safe] < maximum)
        )
        total = float(expected[selected & upper].sum())
        if selected.any():
            if total <= 0.0:
                raise RuntimeError(
                    "Sparse visualization tile has a nonpositive band total"
                )
            probability[selected] = expected[selected] / total
    if valid.any() and not np.isfinite(probability[valid]).all():
        raise RuntimeError(
            "Sparse visualization probability is incomplete"
        )
    return output


def run_inference(
    rna_counts: str | Path,
    output_path: str | Path,
    *,
    mode: str = "both",
    contact_depth: int = 1_000_000,
    seed: int = 0,
    device: str | None = None,
    tile_limit: int | None = None,
    resources: ResourceRegistry | None = None,
    centroids: pd.DataFrame | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> Path:
    """Run smooth, sparse, or shared-pass Hi-Scooby inference."""
    if mode not in {"smooth", "sparse", "both"}:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if contact_depth <= 0:
        raise ValueError("contact_depth must be positive")
    if tile_limit is not None and tile_limit <= 0:
        raise ValueError("tile_limit must be positive")

    selected_modes = (
        ("smooth", "sparse")
        if mode == "both"
        else (mode,)
    )
    resources = resources or load_resources()
    rna_summary = inspect_wide_rna(rna_counts)
    rna_path = rna_summary.path

    phase1_tiles_path = resources.resolve("phase1_tiles")
    sparse_tiles_path = resources.resolve("sparse_tiles")
    phase1_tiles = pd.read_parquet(phase1_tiles_path).reset_index(drop=True)
    sparse_tiles = pd.read_parquet(sparse_tiles_path).reset_index(drop=True)
    full_tile_count = len(phase1_tiles)
    if len(sparse_tiles) != full_tile_count:
        raise ValueError("Smooth and sparse tile counts differ")
    selected_tile_count = (
        full_tile_count
        if tile_limit is None
        else min(int(tile_limit), full_tile_count)
    )
    phase1_tiles = phase1_tiles.iloc[:selected_tile_count].reset_index(
        drop=True
    )
    sparse_tiles = sparse_tiles.iloc[:selected_tile_count].reset_index(
        drop=True
    )
    row_indices = np.arange(selected_tile_count, dtype=np.int64)
    phase1_tiles["tile_index"] = row_indices
    sparse_tiles["tile_index"] = row_indices
    output_tiles = build_output_tile_table(
        phase1_tiles,
        sparse_tiles,
    )

    smooth_predictor = None
    smooth_embeddings = None
    if "smooth" in selected_modes:
        smooth_predictor = SmoothPredictor.load(
            resources,
            device=device,
        )
        if centroids is None:
            centroids = encode_rna_centroids(rna_path, resources)
        cell_types, smooth_embeddings = _select_smooth_contexts(
            centroids,
            smooth_predictor.context_ids,
        )
    else:
        cell_types = _input_cell_types(rna_summary)

    sparse_predictor = None
    sparse_pairs = None
    owner_order = None
    owner_offsets = None
    sparse_storage = None
    if "sparse" in selected_modes:
        resources.resolve("sparse_distance_offsets")
        sparse_predictor = SparsePredictor.load(
            resources,
            device=device,
        )
        sparse_pairs_path = resources.resolve("sparse_canonical_pairs")
        print("[sparse] Loading canonical pair geometry", flush=True)
        sparse_pairs = _load_sparse_pairs(
            sparse_pairs_path,
            selected_tile_count=selected_tile_count,
            full_tile_count=full_tile_count,
        )
        owner_order, owner_offsets = _owner_pair_positions(
            sparse_pairs,
            selected_tile_count,
        )
        sparse_storage = _allocate_sparse_predictions(len(sparse_pairs))

    if embedding_provider is None:
        embedding_provider = _live_embedding_provider(resources)

    with ContactMapOutput(
        output_path,
        modes=selected_modes,
        tile_count=selected_tile_count,
        smooth_context_ids=(
            smooth_predictor.context_ids
            if smooth_predictor is not None
            else ()
        ),
        contact_depth=contact_depth,
        sparse_pair_count=(
            len(sparse_pairs)
            if sparse_pairs is not None
            else None
        ),
    ) as writer:
        writer.write_tiles(output_tiles)
        writer.write_cell_types(cell_types)

        iterator = output_tiles.iterrows()
        for tile_index, tile in tqdm(
            iterator,
            total=selected_tile_count,
            desc="Sequence windows and contact heads",
            unit="tile",
            dynamic_ncols=True,
        ):
            phase1_tile = phase1_tiles.iloc[tile_index]
            pair_embedding = embedding_provider(phase1_tile)

            if smooth_predictor is not None:
                smooth_maps = smooth_predictor.predict_tile(
                    pair_embedding,
                    smooth_embeddings,
                    input_start=int(phase1_tile["input_start"]),
                    target_start=int(phase1_tile["target_start"]),
                )
                writer.write_smooth_tile(tile_index, smooth_maps)

            if sparse_predictor is not None:
                start = int(owner_offsets[tile_index])
                stop = int(owner_offsets[tile_index + 1])
                positions = owner_order[start:stop]
                if len(positions):
                    owner_pairs = sparse_pairs.iloc[positions]
                    sparse_tile = sparse_tiles.iloc[tile_index]
                    prediction = sparse_predictor.predict_owner_tile(
                        pair_embedding,
                        owner_pairs,
                        tile_row=tile_index,
                        input_start=int(sparse_tile["input_start"]),
                        target_start=int(sparse_tile["target_start"]),
                        contact_depth=contact_depth,
                        seed=seed,
                    )
                    writer.write_sparse_pairs(
                        _canonical_pair_frame(
                            owner_pairs,
                            prediction,
                            sparse_tile,
                        )
                    )
                    _record_sparse_prediction(
                        sparse_storage,
                        prediction,
                    )

            del pair_embedding

        if sparse_predictor is not None:
            _validate_sparse_prediction_storage(sparse_storage)
            index = _chromosome_pair_index(sparse_pairs)
            for tile_index, tile in tqdm(
                output_tiles.iterrows(),
                total=selected_tile_count,
                desc="Render sparse visualization maps",
                unit="tile",
                dynamic_ncols=True,
            ):
                pair_grid = _sparse_pair_grid(tile, index)
                writer.write_sparse_tile(
                    tile_index,
                    _sparse_tile_values(
                        pair_grid,
                        sparse_pairs,
                        sparse_storage,
                    ),
                )

        manifest: dict[str, Any] = {
            "package_version": __version__,
            "input": {
                "path": str(rna_path),
                "bytes": rna_path.stat().st_size,
                "cells": rna_summary.n_cells,
                "genes": rna_summary.n_genes,
                "libraries": len(rna_summary.library_ids),
                "cell_type_counts": rna_summary.cell_type_counts,
            },
            "mode": mode,
            "development_tile_limit": tile_limit,
            "seed": int(seed),
            "smooth_context_ids": (
                list(smooth_predictor.context_ids)
                if smooth_predictor is not None
                else []
            ),
            "sparse_model_status": (
                sparse_predictor.model_status
                if sparse_predictor is not None
                else None
            ),
        }
        writer.write_manifest(manifest)
        return writer.finalize()
