from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import cooler
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from tqdm.auto import tqdm
import zarr
from numcodecs import Blosc

from .common import (
    atomic_json,
    configured_distance_bands,
    distance_range_bp,
    enforce_or_warn,
    resolution_contract,
    source_record,
    update_manifest,
)
from .metrics import (
    build_top_contact_groups,
    fixed_distance_oe_scores,
    grouped_top_contact_metrics,
)
from .power import PowerParquetWriter, power_rows


LIBRARY_PATTERN = re.compile(r"_DNA_(\d+)_")
RECORD_DTYPE = np.dtype([("cell", "<u2"), ("key", "<u8")])


def _legacy_balanced_splits(
    cells: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Depth-balanced, library-stratified whole-cell splits."""
    assignments = np.full((len(cells), n_splits), -1, np.int8)
    summaries: list[dict[str, Any]] = []
    depth = cells["valid_pairs"].to_numpy(np.int64)
    for split in range(n_splits):
        rng = np.random.default_rng(seed + split)
        total_depth = np.zeros(2, np.int64)
        for _, group in cells.groupby("library_id", observed=True, sort=True):
            ordered = group.sort_values(
                ["valid_pairs", "cell_id"],
                ascending=[False, True],
                kind="stable",
            ).index.to_numpy()
            pairs = [ordered[i : i + 2] for i in range(0, len(ordered) - 1, 2)]
            rng.shuffle(pairs)
            for pair in pairs:
                left, right = map(int, pair)
                normal = abs(int(total_depth[0] + depth[left] - total_depth[1] - depth[right]))
                swapped = abs(int(total_depth[0] + depth[right] - total_depth[1] - depth[left]))
                if swapped < normal or (swapped == normal and rng.integers(2)):
                    left, right = right, left
                assignments[left, split] = 0
                assignments[right, split] = 1
                total_depth += depth[[left, right]]
            if len(ordered) % 2:
                cell = int(ordered[-1])
                half = int(total_depth[1] < total_depth[0])
                assignments[cell, split] = half
                total_depth[half] += depth[cell]
        unassigned = np.flatnonzero(assignments[:, split] < 0)
        for cell in unassigned:
            half = int(total_depth[1] < total_depth[0])
            assignments[cell, split] = half
            total_depth[half] += depth[cell]
        summaries.append(
            {
                "split": split,
                "half_a_cells": int(np.sum(assignments[:, split] == 0)),
                "half_b_cells": int(np.sum(assignments[:, split] == 1)),
                "half_a_valid_pairs": int(depth[assignments[:, split] == 0].sum()),
                "half_b_valid_pairs": int(depth[assignments[:, split] == 1].sum()),
            }
        )
    return assignments, summaries


def balanced_splits(
    cells: pd.DataFrame, n_splits: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Delegate exactly to the established pseudoreplicate splitter."""
    import sys
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from run_noise_experiment import make_balanced_splits
    return make_balanced_splits(cells, n_splits, seed)


def _load_pair_arrays(
    pairs_path: Path,
    cooler_path: Path,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    pair = pq.read_table(
        pairs_path,
        columns=[
            "pair_id", "chrom", "bin_i", "bin_j", "tile_row",
            "distance_band", "distance_bin", "exposure", "split",
        ],
    ).to_pandas()
    contact_map = cooler.Cooler(str(cooler_path))
    if contact_map.binsize is None:
        raise ValueError("Evidence mapping requires a fixed-bin Cooler")
    bin_size = int(contact_map.binsize)
    if bin_size != int(config["bin_size_bp"]):
        raise ValueError(
            f"Evidence Cooler binsize {bin_size} does not match configured "
            f"{config['bin_size_bp']}"
        )
    offsets = {
        str(chrom): int(contact_map.extent(str(chrom))[0])
        for chrom in contact_map.chromnames
    }
    n_genome_bins = int(contact_map.info["nbins"])
    chrom = pair["chrom"].astype(str).to_numpy()
    bin_i = pair["bin_i"].to_numpy(np.int64)
    bin_j = pair["bin_j"].to_numpy(np.int64)
    left = np.fromiter(
        (offsets[c] + int(position) // bin_size for c, position in zip(chrom, bin_i, strict=True)),
        dtype=np.int64,
        count=len(pair),
    )
    right = np.fromiter(
        (offsets[c] + int(position) // bin_size for c, position in zip(chrom, bin_j, strict=True)),
        dtype=np.int64,
        count=len(pair),
    )
    key = left.astype(np.uint64) * np.uint64(n_genome_bins) + right.astype(np.uint64)
    if np.any(key[1:] <= key[:-1]):
        order = np.argsort(key, kind="stable")
    else:
        order = np.arange(len(key))
    band_lookup = {
        str(row["id"]): index
        for index, row in enumerate(configured_distance_bands(config))
    }
    unknown_bands = set(pair["distance_band"].astype(str)) - set(band_lookup)
    if unknown_bands:
        raise ValueError(f"Pair table contains unknown bands: {sorted(unknown_bands)}")
    return {
        "key": key[order],
        "pair_id": pair["pair_id"].to_numpy(np.int64)[order],
        "tile_row": pair["tile_row"].to_numpy(np.int32),
        "band": pair["distance_band"].astype(str).map(band_lookup).to_numpy(np.uint8),
        "band_label": pair["distance_band"].astype(str).to_numpy(),
        "distance_bin": pair["distance_bin"].to_numpy(np.int16),
        "exposure": pair["exposure"].to_numpy(np.float32),
        "bin_i": bin_i.astype(np.int32),
        "bin_j": bin_j.astype(np.int32),
        "chrom": chrom,
        "split": pair["split"].astype(str).to_numpy(),
        "n_genome_bins": np.asarray([n_genome_bins], dtype=np.int64),
        "bin_size": np.asarray([bin_size], dtype=np.int64),
        "chrom_offsets": offsets,
    }


def _stream_records(
    pairs_dir: Path,
    membership: pd.DataFrame,
    context_indices: list[int],
    pair_arrays: dict[str, np.ndarray],
    temporary: Path,
    *,
    minimum_distance_bp: int,
    maximum_distance_bp_exclusive: int,
) -> tuple[dict[int, Path], dict[str, int]]:
    lookups: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for context in context_indices:
        cells = membership.loc[membership["context_index"].eq(context)].sort_values(
            "cell_id", kind="stable"
        ).reset_index(drop=True)
        for local, row in enumerate(cells.itertuples(index=False)):
            lookups[str(row.library_id)][str(row.dna_barcode)] = (context, local)
    paths = {context: temporary / f"context_{context:03d}.bin" for context in context_indices}
    buffers: dict[int, list[tuple[int, int]]] = {context: [] for context in context_indices}
    key = pair_arrays["key"]
    offsets = pair_arrays["chrom_offsets"]
    n_bins = int(pair_arrays["n_genome_bins"][0])
    bin_size = int(pair_arrays["bin_size"][0])
    stats = {
        "input_data_rows": 0,
        "malformed_rows": 0,
        "unmatched_library_or_barcode": 0,
        "non_cis_rows": 0,
        "unknown_chromosome_rows": 0,
        "outside_distance_rows": 0,
        "candidate_key_misses": 0,
        "candidate_events_written": 0,
        "membership_expected_valid_pairs": int(membership["valid_pairs"].sum()),
        "matched_membership_rows": 0,
    }
    matched_by_context = {
        context: np.zeros(
            int(membership["context_index"].eq(context).sum()), dtype=np.int64
        )
        for context in context_indices
    }

    def flush(handles: dict[int, Any]) -> None:
        for context, rows in buffers.items():
            if not rows:
                continue
            array = np.asarray(rows, dtype=RECORD_DTYPE)
            positions = np.searchsorted(key, array["key"])
            valid = positions < len(key)
            safe = np.minimum(positions, len(key) - 1)
            valid &= key[safe] == array["key"]
            stats["candidate_key_misses"] += int((~valid).sum())
            stats["candidate_events_written"] += int(valid.sum())
            array[valid].tofile(handles[context])
            rows.clear()

    with ExitStack() as stack:
        handles = {context: stack.enter_context(path.open("wb")) for context, path in paths.items()}
        buffered = 0
        pair_paths = sorted(pairs_dir.glob("*.cis_uu_autosomes.pairs.gz"))
        if not pair_paths:
            raise FileNotFoundError(f"No filtered raw-pair files under {pairs_dir}")
        for path in tqdm(pair_paths, desc="Streaming raw contacts"):
            match = LIBRARY_PATTERN.search(path.name)
            if match is None:
                continue
            library = f"DNA{int(match.group(1)):02d}"
            barcode_lookup = lookups.get(library)
            if not barcode_lookup:
                continue
            with gzip.open(path, "rb") as handle:
                for line in handle:
                    if line.startswith(b"#"):
                        continue
                    stats["input_data_rows"] += 1
                    fields = line.rstrip().split(b"\t")
                    if len(fields) < 5:
                        stats["malformed_rows"] += 1
                        continue
                    barcode = fields[0].split(b":", 1)[0].decode()
                    assignment = barcode_lookup.get(barcode)
                    if assignment is None:
                        stats["unmatched_library_or_barcode"] += 1
                        continue
                    context, cell = assignment
                    stats["matched_membership_rows"] += 1
                    matched_by_context[context][cell] += 1
                    chrom1, chrom2 = fields[1].decode(), fields[3].decode()
                    if chrom1 != chrom2 or chrom1 not in offsets:
                        if chrom1 != chrom2:
                            stats["non_cis_rows"] += 1
                        else:
                            stats["unknown_chromosome_rows"] += 1
                        continue
                    local_i = int(fields[2]) // bin_size
                    local_j = int(fields[4]) // bin_size
                    if local_j < local_i:
                        local_i, local_j = local_j, local_i
                    distance = (local_j - local_i) * bin_size
                    if (
                        distance < minimum_distance_bp
                        or distance >= maximum_distance_bp_exclusive
                    ):
                        stats["outside_distance_rows"] += 1
                        continue
                    left = offsets[chrom1] + local_i
                    right = offsets[chrom1] + local_j
                    buffers[context].append((cell, left * n_bins + right))
                    buffered += 1
                    if buffered >= 500_000:
                        flush(handles)
                        buffered = 0
        flush(handles)
    membership_mismatches = 0
    for context in context_indices:
        expected = membership.loc[
            membership["context_index"].eq(context)
        ].sort_values("cell_id", kind="stable")["valid_pairs"].to_numpy(np.int64)
        membership_mismatches += int(np.sum(matched_by_context[context] != expected))
    stats["membership_cells_with_count_mismatch"] = membership_mismatches
    stats["membership_count_conservation"] = bool(
        membership_mismatches == 0
        and stats["matched_membership_rows"]
        == stats["membership_expected_valid_pairs"]
    )
    if not stats["membership_count_conservation"]:
        raise RuntimeError(
            "Raw pair rows do not reproduce per-cell membership valid_pairs totals"
        )
    return paths, stats


def _cell_pair_matrix(
    record_path: Path,
    n_cells: int,
    sorted_keys: np.ndarray,
    sorted_pair_id: np.ndarray,
    pair_count: int,
) -> sparse.csr_matrix:
    records = np.fromfile(record_path, dtype=RECORD_DTYPE)
    if not len(records):
        return sparse.csr_matrix((n_cells, pair_count), dtype=np.uint32)
    positions = np.searchsorted(sorted_keys, records["key"])
    safe = np.minimum(positions, len(sorted_keys) - 1)
    valid = (positions < len(sorted_keys)) & (sorted_keys[safe] == records["key"])
    columns = sorted_pair_id[safe[valid]]
    matrix = sparse.coo_matrix(
        (
            np.ones(valid.sum(), dtype=np.uint32),
            (records["cell"][valid].astype(np.int64), columns),
        ),
        shape=(n_cells, pair_count),
        dtype=np.uint32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def _cooler_candidate_event_total(
    cooler_path: Path,
    sorted_candidate_keys: np.ndarray,
    n_genome_bins: int,
    *,
    chunk_rows: int = 1_000_000,
) -> int:
    """Independently audit candidate mass from raw Cooler pixels."""
    contact_map = cooler.Cooler(str(cooler_path))
    pixels = contact_map.pixels(join=False)
    total = 0
    for start in tqdm(
        range(0, int(contact_map.info["nnz"]), chunk_rows),
        desc="Audit candidate mass against Cooler",
        unit="pixel",
        unit_scale=True,
    ):
        frame = pixels[start : start + chunk_rows]
        keys = (
            frame["bin1_id"].to_numpy(np.uint64) * np.uint64(n_genome_bins)
            + frame["bin2_id"].to_numpy(np.uint64)
        )
        positions = np.searchsorted(sorted_candidate_keys, keys)
        valid = positions < len(sorted_candidate_keys)
        safe = np.minimum(positions, len(sorted_candidate_keys) - 1)
        valid &= sorted_candidate_keys[safe] == keys
        total += int(frame["count"].to_numpy(np.uint64)[valid].sum())
    return total


def _ranks_and_support(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    tile_row: np.ndarray,
    distance_bin: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    *,
    top_fraction: float = 0.02,
    bin_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rank_a = np.zeros(len(counts_a), np.float32)
    rank_b = np.zeros(len(counts_b), np.float32)
    support = np.zeros(len(counts_a), np.float32)
    group_width = int(distance_bin.max()) + 1
    group = tile_row.astype(np.int64) * group_width + distance_bin.astype(np.int64)
    order = np.argsort(group, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, group[order][1:] != group[order][:-1], True])
    tie_groups = 0
    defined_groups = 0

    def positive_midrank(
        counts: np.ndarray,
        indices: np.ndarray,
        output: np.ndarray,
    ) -> np.ndarray:
        order = np.argsort(-counts[indices], kind="stable")
        positive = order[counts[indices[order]] > 0]
        if not len(positive):
            return positive
        ordered_counts = counts[indices[positive]]
        boundaries = np.flatnonzero(
            np.r_[True, ordered_counts[1:] != ordered_counts[:-1], True]
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            # All members of a count plateau get the same midpoint rank. This
            # keeps support weights invariant to pair_id/stable-sort order.
            midpoint = (start + stop - 1) / 2.0
            percentile = 1.0 - midpoint / max(len(indices) - 1, 1)
            output[indices[positive[start:stop]]] = percentile
        return positive

    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        indices = order[begin:end]
        size = len(indices)
        if size < 2:
            continue
        defined_groups += 1
        k = max(1, int(np.ceil(size * top_fraction)))
        positive_a = positive_midrank(counts_a, indices, rank_a)
        positive_b = positive_midrank(counts_b, indices, rank_b)
        if not len(positive_a) or not len(positive_b):
            continue
        k_a, k_b = min(k, len(positive_a)), min(k, len(positive_b))
        threshold_a = counts_a[indices[positive_a[k_a - 1]]]
        threshold_b = counts_b[indices[positive_b[k_b - 1]]]
        top_a = indices[positive_a[counts_a[indices[positive_a]] >= threshold_a]]
        top_b = indices[positive_b[counts_b[indices[positive_b]] >= threshold_b]]
        if len(top_a) > k_a or len(top_b) > k_b:
            tie_groups += 1
        set_a = {(int(bin_i[x]) // bin_size, int(bin_j[x]) // bin_size): x for x in top_a}
        set_b = {(int(bin_i[x]) // bin_size, int(bin_j[x]) // bin_size): x for x in top_b}
        for source, reference, source_rank, reference_rank in (
            (set_a, set_b, rank_a, rank_b),
            (set_b, set_a, rank_b, rank_a),
        ):
            for (left, right), index in source.items():
                matched = [
                    reference[(left + di, right + dj)]
                    for di in (-1, 0, 1)
                    for dj in (-1, 0, 1)
                    if (left + di, right + dj) in reference
                ]
                for partner in matched:
                    weight = min(float(source_rank[index]), float(reference_rank[partner]))
                    support[index] = max(support[index], weight)
                    support[partner] = max(support[partner], weight)
    return rank_a, rank_b, support, {
        "defined_groups": defined_groups,
        "cutoff_tie_groups": tie_groups,
        "cutoff_tie_rate": tie_groups / defined_groups if defined_groups else None,
    }


def _combine_support(
    primary_support: np.ndarray,
    reproduced_splits: np.ndarray,
    split_count: int,
) -> np.ndarray:
    if split_count <= 0:
        raise ValueError("split_count must be positive")
    reproducibility = np.asarray(reproduced_splits, np.float32) / float(split_count)
    primary = np.asarray(primary_support, np.float32)
    if primary.shape != reproducibility.shape:
        raise ValueError("Primary support and reproducibility shapes differ")
    return np.maximum(
        primary * (0.5 + 0.5 * reproducibility), reproducibility
    ).astype(np.float32)


def export_evidence(
    config: dict[str, Any],
    pairs_path: Path,
    *,
    splits: int,
    seed: int,
) -> dict[str, Any]:
    output_root = Path(config["outputs"]["data_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    pair_arrays = _load_pair_arrays(
        pairs_path, Path(config["paths"]["cooler"]), config
    )
    minimum_distance, maximum_distance_exclusive = distance_range_bp(config)
    pair_count = len(pair_arrays["pair_id"])
    membership = pd.read_parquet(config["paths"]["membership"])
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    context_indices = contexts["context_index"].astype(int).tolist()
    assignment_frames = []
    assignments_by_context: dict[int, np.ndarray] = {}
    balance_rows = []
    for context in context_indices:
        cells = membership.loc[membership["context_index"].eq(context)].sort_values(
            "cell_id", kind="stable"
        ).reset_index(drop=True)
        assignment, summaries = balanced_splits(
            cells, splits, seed + 10_000 * context
        )
        assignments_by_context[context] = assignment
        for split in range(splits):
            frame = cells[["context_index", "cell_type", "cell_id", "library_id", "dna_barcode"]].copy()
            frame["split"] = split
            frame["half"] = np.where(assignment[:, split] == 0, "A", "B")
            assignment_frames.append(frame)
        balance_rows.extend({"context_index": context, **row} for row in summaries)
    assignments_path = output_root / "pseudoreplicate_assignments.parquet"
    pd.concat(assignment_frames, ignore_index=True).to_parquet(
        assignments_path, index=False, compression="zstd"
    )

    temp = tempfile.TemporaryDirectory(prefix="raw-contact-evidence-", dir="/tmp")
    try:
        record_paths, stream_stats = _stream_records(
            Path(config["paths"]["filtered_pairs"]),
            membership,
            context_indices,
            pair_arrays,
            Path(temp.name),
            minimum_distance_bp=minimum_distance,
            maximum_distance_bp_exclusive=maximum_distance_exclusive,
        )
        store_path = output_root / "pseudoreplicate_evidence.zarr"
        root = zarr.open_group(str(store_path), mode="w")
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        shape = (len(context_indices), pair_count)
        chunks = (1, min(1_000_000, pair_count))
        arrays = {
            "counts_a": root.create_dataset("counts_a", shape=shape, chunks=chunks, dtype="u4", compressor=compressor),
            "counts_b": root.create_dataset("counts_b", shape=shape, chunks=chunks, dtype="u4", compressor=compressor),
            "full_count": root.create_dataset("full_count", shape=shape, chunks=chunks, dtype="u4", compressor=compressor),
            "valid_mask": root.create_dataset("valid_mask", shape=shape, chunks=chunks, dtype="bool", compressor=compressor, fill_value=True),
            "rank_a": root.create_dataset("rank_a", shape=shape, chunks=chunks, dtype="f4", compressor=compressor),
            "rank_b": root.create_dataset("rank_b", shape=shape, chunks=chunks, dtype="f4", compressor=compressor),
            "support_weight": root.create_dataset("support_weight", shape=shape, chunks=chunks, dtype="f4", compressor=compressor),
        }
        primary = int(config["primary_pseudoreplicate_split"])
        training_mask = pair_arrays["split"] == "train"
        validation_mask = pair_arrays["split"] == "validation"
        validation_pairs = pd.DataFrame(
            {
                "chrom": pair_arrays["chrom"][validation_mask],
                "bin_i": pair_arrays["bin_i"][validation_mask],
                "bin_j": pair_arrays["bin_j"][validation_mask],
                "tile_row": pair_arrays["tile_row"][validation_mask],
                "distance_band": pair_arrays["band_label"][validation_mask],
            }
        )
        ceiling_groups = build_top_contact_groups(
            validation_pairs["tile_row"].to_numpy(),
            validation_pairs["distance_band"].to_numpy(),
        )
        band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
        ceiling_values: dict[str, list[float]] = {
            f"{band}:top{fraction:g}": []
            for band in band_ids
            for fraction in (0.01, 0.02)
        }
        ceiling_neighborhood_cache: dict[tuple[str, int, int], np.ndarray] = {}
        power_fractions = (
            float(config.get("power", {}).get("primary_top_fraction", 0.01)),
            *tuple(
                float(value)
                for value in config.get("power", {}).get(
                    "diagnostic_top_fractions", (0.02, 0.05)
                )
            ),
        )
        validation_count = int(validation_mask.sum())
        pooled_power_a = np.zeros((splits, validation_count), np.uint64)
        pooled_power_b = np.zeros_like(pooled_power_a)
        configured_pools = {
            str(row["id"]): set(map(str, row["members"]))
            for row in config.get("contexts", {}).get("pools", [])
        }
        pool_power_a = {
            pool_id: np.zeros_like(pooled_power_a)
            for pool_id in configured_pools
        }
        pool_power_b = {
            pool_id: np.zeros_like(pooled_power_b)
            for pool_id in configured_pools
        }
        power_writer = PowerParquetWriter(output_root / "power_groups.parquet")
        context_reports = []
        materialized_event_total = 0
        for output_context, context in tqdm(
            enumerate(context_indices),
            total=len(context_indices),
            desc="Materialize pseudoreplicates",
            unit="context",
        ):
            cells = membership.loc[membership["context_index"].eq(context)].sort_values(
                "cell_id", kind="stable"
            ).reset_index(drop=True)
            matrix = _cell_pair_matrix(
                record_paths[context],
                len(cells),
                pair_arrays["key"],
                pair_arrays["pair_id"],
                pair_count,
            )
            full = np.asarray(matrix.sum(axis=0)).ravel().astype(np.uint32)
            materialized_event_total += int(full.sum())
            reproduced = np.zeros(pair_count, np.uint8)
            counts_a = counts_b = None
            assignments = assignments_by_context[context]
            cell_type = str(
                contexts.loc[
                    contexts["context_index"].eq(context), "cell_type"
                ].iloc[0]
            )
            exposure = np.clip(pair_arrays["exposure"], 1e-12, None)
            reference = (full / exposure) / max(len(cells), 1)
            distances = pair_arrays["distance_bin"].astype(np.int64)
            size = int(distances.max()) + 1
            training_distances = distances[training_mask]
            candidate_counts = np.bincount(training_distances, minlength=size)
            contact_sums = np.bincount(
                training_distances,
                weights=reference[training_mask],
                minlength=size,
            )
            expected = np.full(size, np.nan, np.float64)
            represented = candidate_counts > 0
            expected[represented] = contact_sums[represented] / candidate_counts[represented]
            positive_expected = expected[np.isfinite(expected) & (expected > 0)]
            if not len(positive_expected):
                raise RuntimeError(
                    f"Context {context} has no positive training distance curve"
                )
            epsilon = float(positive_expected.min())
            validation_expected = expected[distances[validation_mask]]
            for split in range(splits):
                assignment = assignments[:, split]
                split_a = np.asarray(
                    matrix[assignment == 0].sum(axis=0)
                ).ravel().astype(np.uint32)
                split_b = np.asarray(
                    matrix[assignment == 1].sum(axis=0)
                ).ravel().astype(np.uint32)
                if not np.array_equal(split_a.astype(np.uint64) + split_b, full):
                    raise RuntimeError(
                        f"Pseudoreplicate counts do not recombine for split {split}"
                    )
                reproduced += ((split_a > 0) & (split_b > 0)).astype(np.uint8)
                cells_a = max(int(np.sum(assignment == 0)), 1)
                cells_b = max(int(np.sum(assignment == 1)), 1)
                a_validation = (split_a[validation_mask] / exposure[validation_mask]) / cells_a
                b_validation = (split_b[validation_mask] / exposure[validation_mask]) / cells_b
                with np.errstate(divide="ignore", invalid="ignore"):
                    a_validation = np.log(
                        (a_validation + epsilon) / (validation_expected + epsilon)
                    )
                    b_validation = np.log(
                        (b_validation + epsilon) / (validation_expected + epsilon)
                    )
                invalid_expected = (
                    ~np.isfinite(validation_expected) | (validation_expected <= 0)
                )
                a_validation[invalid_expected] = np.nan
                b_validation[invalid_expected] = np.nan
                top_rows = grouped_top_contact_metrics(
                    validation_pairs["chrom"].to_numpy(),
                    validation_pairs["bin_i"].to_numpy(),
                    validation_pairs["bin_j"].to_numpy(),
                    validation_pairs["tile_row"].to_numpy(),
                    validation_pairs["distance_band"].to_numpy(),
                    a_validation,
                    b_validation,
                    fractions=(0.01, 0.02),
                    tolerances=(1,),
                    neighborhood_size_cache=ceiling_neighborhood_cache,
                    groups=ceiling_groups,
                    tie_mode=(
                        "hard_cutoff"
                        if str(config["evaluation"]["primary_tie_mode"])
                        == "positive_fractional"
                        else str(config["evaluation"]["primary_tie_mode"])
                    ),
                    bin_size_bp=int(config["bin_size_bp"]),
                )
                for row in top_rows:
                    value = row["enrichment_over_chance"]
                    if value is not None and np.isfinite(value):
                        key = f'{row["band"]}:top{row["top_fraction"]:g}'
                        ceiling_values[key].append(float(value))
                if split == primary:
                    counts_a, counts_b = split_a, split_b
                validation_a = split_a[validation_mask]
                validation_b = split_b[validation_mask]
                pooled_power_a[split] += validation_a.astype(np.uint64)
                pooled_power_b[split] += validation_b.astype(np.uint64)
                for pool_id, members in configured_pools.items():
                    if cell_type in members:
                        pool_power_a[pool_id][split] += validation_a.astype(np.uint64)
                        pool_power_b[pool_id][split] += validation_b.astype(np.uint64)
                power_writer.write(
                    power_rows(
                        validation_pairs,
                        validation_a,
                        validation_b,
                        output_id=cell_type,
                        output_type="context",
                        split_index=split,
                        fractions=power_fractions,
                        groups=ceiling_groups,
                    )
                )
            if counts_a is None or counts_b is None:
                raise RuntimeError("Primary pseudoreplicate split was not materialized")
            rank_a, rank_b, primary_support, tie_report = _ranks_and_support(
                counts_a / np.clip(pair_arrays["exposure"], 1e-12, None),
                counts_b / np.clip(pair_arrays["exposure"], 1e-12, None),
                pair_arrays["tile_row"],
                pair_arrays["distance_bin"],
                pair_arrays["bin_i"],
                pair_arrays["bin_j"],
                bin_size=int(config["bin_size_bp"]),
            )
            reproducibility = reproduced.astype(np.float32) / float(splits)
            support = _combine_support(primary_support, reproduced, splits)
            arrays["counts_a"][output_context] = counts_a
            arrays["counts_b"][output_context] = counts_b
            arrays["full_count"][output_context] = full
            arrays["valid_mask"][output_context] = True
            arrays["rank_a"][output_context] = rank_a
            arrays["rank_b"][output_context] = rank_b
            arrays["support_weight"][output_context] = support
            context_reports.append(
                {
                    "context_index": context,
                    "cell_type": cell_type,
                    "full_events_in_universe": int(full.sum()),
                    "supported_pairs": int(np.sum(support > 0)),
                    "pairs_reproduced_in_any_split": int(np.sum(reproduced > 0)),
                    "pairs_reproduced_in_all_splits": int(np.sum(reproduced == splits)),
                    "mean_reproducibility_of_supported_pairs": float(
                        reproducibility[support > 0].mean()
                    ) if np.any(support > 0) else None,
                    **tie_report,
                }
            )
        for split in range(splits):
            power_writer.write(
                power_rows(
                    validation_pairs,
                    pooled_power_a[split],
                    pooled_power_b[split],
                    output_id="shared",
                    output_type="shared",
                    split_index=split,
                    fractions=power_fractions,
                    groups=ceiling_groups,
                )
            )
            for pool_id in configured_pools:
                power_writer.write(
                    power_rows(
                        validation_pairs,
                        pool_power_a[pool_id][split],
                        pool_power_b[pool_id][split],
                        output_id=pool_id,
                        output_type="pool",
                        split_index=split,
                        fractions=power_fractions,
                        groups=ceiling_groups,
                    )
                )
        power_writer.close()
        root.attrs.update(
            {
                "schema_version": 3,
                "context_ids": contexts["cell_type"].astype(str).tolist(),
                "context_indices": context_indices,
                "pair_count": pair_count,
                "primary_split": primary,
                "seed": seed,
                "unobserved_semantics": "unlabeled_opportunity",
                "support_definition": "maximum of all-split exact-pair reproducibility and reproducibility-weighted primary exposure-corrected exact-distance top-2pct neighborhood support",
                "support_splits": splits,
                "resolution": resolution_contract(config),
                "power_audit": "positive_fractional_all_splits",
            }
        )
    finally:
        temp.cleanup()
    cooler_candidate_events = _cooler_candidate_event_total(
        Path(config["paths"]["cooler"]),
        pair_arrays["key"],
        int(pair_arrays["n_genome_bins"][0]),
    )
    cooler_candidate_conservation = cooler_candidate_events == materialized_event_total
    if not cooler_candidate_conservation:
        raise RuntimeError(
            "Raw streamed candidate events do not match independent Cooler pixels: "
            f"{materialized_event_total} != {cooler_candidate_events}"
        )
    report = {
        "pairs": str(pairs_path),
        "store": str(output_root / "pseudoreplicate_evidence.zarr"),
        "assignments": str(assignments_path),
        "pair_count": pair_count,
        "contexts": context_reports,
        "balance": balance_rows,
        "count_conservation": True,
        "membership_count_conservation": bool(
            stream_stats["membership_count_conservation"]
        ),
        "cooler_pixels_consulted_for_targets": False,
        "cooler_pixels_consulted_for_conservation_audit": True,
        "cooler_candidate_events": cooler_candidate_events,
        "cooler_candidate_count_conservation": cooler_candidate_conservation,
        "cooler_balance_weights_consulted": False,
        "cooler_bin_index_source": source_record(Path(config["paths"]["cooler"])),
        "stream_exclusion_counts": stream_stats,
        "materialized_candidate_events": materialized_event_total,
        "stream_to_matrix_conservation": (
            materialized_event_total == stream_stats["candidate_events_written"]
        ),
        "canonical_ceiling": {
            "split_count": splits,
            "context_split_rows": {
                key: len(values) for key, values in ceiling_values.items()
            },
            "values": {
                key: (float(np.median(values)) if values else None)
                for key, values in ceiling_values.items()
            },
            "tie_mode": str(config["evaluation"]["primary_tie_mode"]),
            "candidate_representation": "unique canonical validation pairs",
        },
        "power_groups": str(output_root / "power_groups.parquet"),
        "power_tie_mode": "positive_fractional",
        "power_zero_padding": False,
        "resolution": resolution_contract(config),
        "warnings": [],
    }
    if not report["stream_to_matrix_conservation"]:
        raise RuntimeError(
            "Streamed candidate events do not equal materialized matrix events"
        )
    atomic_json(output_root / "evidence_export_report.json", report)
    update_manifest(
        output_root,
        "evidence",
        {
            **report,
            "membership_source": source_record(Path(config["paths"]["membership"])),
            "filtered_pairs_source": source_record(Path(config["paths"]["filtered_pairs"])),
        },
    )
    return report


def _recompute_primary_enrichment(config: dict[str, Any], root) -> dict[str, float | None]:
    pairs = pd.read_parquet(
        Path(config["outputs"]["data_root"]) / "canonical_pairs.parquet",
        columns=[
            "chrom", "bin_i", "bin_j", "distance_bin", "distance_band",
            "tile_row", "split", "exposure",
        ],
    )
    validation = pairs["split"].eq("validation").to_numpy()
    training = pairs["split"].eq("train").to_numpy()
    validation_pairs = pairs.loc[validation].reset_index(drop=True)
    primary_split = int(root.attrs["primary_split"])
    assignment_path = (
        Path(config["outputs"]["data_root"]) / "pseudoreplicate_assignments.parquet"
    )
    assignments = pd.read_parquet(
        assignment_path, columns=["context_index", "split", "half"]
    )
    assignments = assignments.loc[assignments["split"].eq(primary_split)]
    half_cell_counts = assignments.groupby(
        ["context_index", "half"], observed=True
    ).size()
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    output: dict[str, list[float]] = {
        f"{band}:top{fraction:g}": []
        for band in band_ids for fraction in (0.01, 0.02)
    }
    distance = pairs["distance_bin"].to_numpy(np.int16)
    neighborhood_size_cache: dict[tuple[str, int, int], np.ndarray] = {}
    groups = build_top_contact_groups(
        validation_pairs["tile_row"].to_numpy(),
        validation_pairs["distance_band"].to_numpy(),
    )
    inverse_exposure = 1.0 / np.clip(pairs["exposure"].to_numpy(np.float64), 1e-12, None)
    for context in tqdm(
        range(root["counts_a"].shape[0]),
        desc="Validate pseudoreplicate enrichment",
        unit="context",
    ):
        context_index = int(root.attrs["context_indices"][context])
        cells_a = int(half_cell_counts.loc[(context_index, "A")])
        cells_b = int(half_cell_counts.loc[(context_index, "B")])
        if cells_a <= 0 or cells_b <= 0:
            raise ValueError(f"Context {context_index} has an empty pseudoreplicate half")
        raw_a = np.asarray(root["counts_a"][context], np.float64) * inverse_exposure
        raw_b = np.asarray(root["counts_b"][context], np.float64) * inverse_exposure
        a = raw_a / cells_a
        b = raw_b / cells_b
        fixed_reference = (raw_a + raw_b) / (cells_a + cells_b)
        a = fixed_distance_oe_scores(a, fixed_reference, distance, training)
        b = fixed_distance_oe_scores(b, fixed_reference, distance, training)
        top_rows = grouped_top_contact_metrics(
            validation_pairs["chrom"].to_numpy(),
            validation_pairs["bin_i"].to_numpy(),
            validation_pairs["bin_j"].to_numpy(),
            validation_pairs["tile_row"].to_numpy(),
            validation_pairs["distance_band"].to_numpy(),
            a[validation],
            b[validation],
            fractions=(0.01, 0.02),
            tolerances=(1,),
            neighborhood_size_cache=neighborhood_size_cache,
            groups=groups,
            progress_desc=(
                f"Pseudoreplicate top contacts {context + 1}/"
                f"{root['counts_a'].shape[0]}"
            ),
            tie_mode=(
                "hard_cutoff"
                if str(config["evaluation"]["primary_tie_mode"])
                == "positive_fractional"
                else str(config["evaluation"]["primary_tie_mode"])
            ),
            bin_size_bp=int(config["bin_size_bp"]),
        )
        for row in top_rows:
            key = f'{row["band"]}:top{row["top_fraction"]:g}'
            value = row["enrichment_over_chance"]
            if value is not None and np.isfinite(value):
                output[key].append(float(value))
    return {key: (float(np.median(values)) if values else None) for key, values in output.items()}


def validate_evidence(
    config: dict[str, Any],
    reference: Path,
    output: Path,
) -> dict[str, Any]:
    output_root = Path(config["outputs"]["data_root"])
    root = zarr.open_group(str(output_root / "pseudoreplicate_evidence.zarr"), mode="r")
    warnings: list[dict[str, Any]] = []
    conserved = True
    for context in tqdm(
        range(root["full_count"].shape[0]),
        desc="Verify count conservation",
        unit="context",
    ):
        conserved &= bool(
            np.array_equal(
                root["counts_a"][context].astype(np.uint64)
                + root["counts_b"][context].astype(np.uint64),
                root["full_count"][context].astype(np.uint64),
            )
        )
    if not conserved:
        raise RuntimeError("Count conservation failed")
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    expected_keys = [
        (band, fraction)
        for band in band_ids
        for fraction in (0.01, 0.02)
    ]
    evidence_report_path = output_root / "evidence_export_report.json"
    with evidence_report_path.open() as handle:
        evidence_report = json.load(handle)
    canonical_ceiling = evidence_report.get("canonical_ceiling", {})
    canonical_values = canonical_ceiling.get("values", {})
    if int(canonical_ceiling.get("split_count", 0)) != int(
        config["pseudoreplicate_splits"]
    ):
        raise RuntimeError("Canonical ceiling was not computed over all configured splits")
    fractional_contract = (
        str(config["evaluation"]["primary_tie_mode"]) == "positive_fractional"
    )
    if (
        not fractional_contract
        and any(
            canonical_values.get(f"{band}:top{fraction:g}") is None
            for band, fraction in expected_keys
        )
    ):
        raise RuntimeError("Canonical all-split ceiling is incomplete")
    if fractional_contract and not (
        output_root / "power_groups.parquet"
    ).is_file():
        raise RuntimeError("Positive-fractional power audit is missing")
    reference_values: dict[str, Any] = {}
    if reference.exists() and int(config["bin_size_bp"]) == 5_000:
        with reference.open() as handle:
            payload = json.load(handle)
        rows = payload.get("target_topk_summary_across_contexts", [])
        for band, fraction in expected_keys:
            matches = [
                row for row in rows
                if row.get("representation") == "raw"
                and row.get("normalization") == "fixed"
                and row.get("band") == band
                and float(row.get("top_fraction", -1)) == fraction
                and row.get("selection_mode") == "hard_topk"
                and int(row.get("match_tolerance_bins", -1)) == 1
            ]
            key = f"{band}:top{fraction:g}"
            reference_values[key] = (
                matches[0].get("enrichment_over_chance", {}).get("median")
                if matches else None
            )
            enforce_or_warn(
                bool(matches),
                warnings,
                "REFERENCE_METRIC_MISSING",
                f"Published reference row is missing for {key}",
                strict=False,
            )
    else:
        enforce_or_warn(
            False,
            warnings,
            "REFERENCE_FILE_MISSING",
            f"Reference top-contact result does not exist: {reference}",
            strict=False,
        )
    report = {
        "count_conservation": conserved,
        "metric_definition": {
            "candidate_representation": "unique canonical pairs assigned to one tile",
            "normalization": "shared full-target training-distance fixed static-exposure-corrected observed/expected",
            "chance": "tile-specific geometry-aware hypergeometric probability",
            "aggregation": "median across tiles within each context/split, then median across all context-split rows",
        },
        "historical_reproduction_established": None,
        "historical_reproduction_note": "Legacy values use overlapping tile-expanded hard-top-K geometry and are diagnostic only; the acceptance gate uses the all-split canonical ceiling.",
        "reference_values": reference_values,
        "canonical_all_split_ceiling": canonical_values,
        "canonical_ceiling_split_count": canonical_ceiling["split_count"],
        "primary_power_audit": (
            str(output_root / "power_groups.parquet")
            if fractional_contract
            else None
        ),
        "resolution": resolution_contract(config),
        "warnings": warnings,
    }
    atomic_json(output, report)
    return report
