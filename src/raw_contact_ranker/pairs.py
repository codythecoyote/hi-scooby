from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import (
    atomic_json,
    distance_band,
    distance_range_bp,
    resolution_contract,
    source_record,
    update_manifest,
)


def _candidate_count(tile_count: int, min_bin: int, max_bin: int, bins: int) -> int:
    return tile_count * sum(bins - distance for distance in range(min_bin, max_bin + 1))


def _purge_cross_split_anchors(
    selected: np.ndarray,
    chrom_code: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    tile_row: np.ndarray,
    tile_splits: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Remove pairs touching a genomic bin represented in multiple splits."""
    if not len(selected):
        return selected, 0, 0
    _, split_code = np.unique(tile_splits.astype(str), return_inverse=True)
    owner = split_code[tile_row[selected]].astype(np.int16)
    stride = int(bin_j[selected].max()) + 1
    left_key = chrom_code[selected].astype(np.int64) * stride + bin_i[selected]
    right_key = chrom_code[selected].astype(np.int64) * stride + bin_j[selected]
    keys = np.concatenate([left_key, right_key])
    owners = np.concatenate([owner, owner])
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_owners = owners[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    minimum_owner = np.minimum.reduceat(sorted_owners, starts)
    maximum_owner = np.maximum.reduceat(sorted_owners, starts)
    conflicting = sorted_keys[starts][minimum_owner != maximum_owner]
    del keys, owners, order, sorted_keys, sorted_owners
    if not len(conflicting):
        return selected, 0, 0
    retained = ~np.isin(left_key, conflicting) & ~np.isin(right_key, conflicting)
    return selected[retained], int(len(conflicting)), int((~retained).sum())


def export_canonical_pairs(
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    bin_size = int(config["bin_size_bp"])
    minimum, maximum_exclusive = distance_range_bp(config)
    maximum = maximum_exclusive - bin_size
    min_bin = minimum // bin_size
    max_bin = maximum // bin_size
    tiles_path = Path(config["paths"]["tiles"])
    tiles = pd.read_parquet(tiles_path).reset_index(drop=True)
    required = {
        "tile_id", "chrom", "target_start", "target_end", "split",
    }
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"Tile table lacks columns: {sorted(missing)}")
    lengths = tiles["target_end"].to_numpy(np.int64) - tiles["target_start"].to_numpy(np.int64)
    if not np.all(lengths == 1_000_000):
        raise ValueError("All target tiles must span exactly 1 Mb")
    target_bins = 1_000_000 // bin_size
    n_candidates = _candidate_count(len(tiles), min_bin, max_bin, target_bins)
    chrom_names = sorted(tiles["chrom"].astype(str).unique(), key=lambda x: int(x[3:]))
    chrom_lookup = {chrom: index for index, chrom in enumerate(chrom_names)}

    chrom_code = np.empty(n_candidates, dtype=np.uint8)
    bin_i = np.empty(n_candidates, dtype=np.int32)
    bin_j = np.empty(n_candidates, dtype=np.int32)
    tile_row = np.empty(n_candidates, dtype=np.int32)
    boundary_margin = np.empty(n_candidates, dtype=np.int32)
    cursor = 0
    for row, tile in enumerate(
        tqdm(tiles.itertuples(index=False), total=len(tiles), desc="Enumerate pairs", unit="tile")
    ):
        start_bin = int(tile.target_start) // bin_size
        for distance in range(min_bin, max_bin + 1):
            count = target_bins - distance
            selected = slice(cursor, cursor + count)
            left = start_bin + np.arange(count, dtype=np.int32)
            chrom_code[selected] = chrom_lookup[str(tile.chrom)]
            bin_i[selected] = left
            bin_j[selected] = left + distance
            tile_row[selected] = row
            left_margin = np.arange(count, dtype=np.int32)
            right_margin = target_bins - (left_margin + distance + 1)
            boundary_margin[selected] = np.minimum(left_margin, right_margin)
            cursor += count
    if cursor != n_candidates:
        raise RuntimeError("Candidate preallocation and generation disagree")

    # Last key is primary. Pair coordinates ascend; preferred representation has
    # greatest boundary margin, with tile row (and therefore tile_id order after
    # a stable source sort) as deterministic final tie-break.
    tile_id_order = pd.Series(tiles["tile_id"].astype(str)).rank(method="dense").to_numpy(np.int32)
    order = np.lexsort(
        (
            tile_id_order[tile_row],
            -boundary_margin,
            bin_j,
            bin_i,
            chrom_code,
        )
    )
    ordered_chrom = chrom_code[order]
    ordered_i = bin_i[order]
    ordered_j = bin_j[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (
        (ordered_chrom[1:] != ordered_chrom[:-1])
        | (ordered_i[1:] != ordered_i[:-1])
        | (ordered_j[1:] != ordered_j[:-1])
    )
    selected = order[first]
    del order, ordered_chrom, ordered_i, ordered_j, first

    # Restore genomic ordering. This also makes genome_key monotonic when the
    # cooler chromosome order is canonical.
    selected = selected[
        np.lexsort((bin_j[selected], bin_i[selected], chrom_code[selected]))
    ]
    canonical_pair_count = len(selected)
    selected, conflicting_anchor_bins, split_buffer_pairs_removed = (
        _purge_cross_split_anchors(
            selected,
            chrom_code,
            bin_i,
            bin_j,
            tile_row,
            tiles["split"].astype(str).to_numpy(),
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "canonical_pairs.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    chunk_size = 1_000_000
    chrom_offsets: dict[str, int] = {}
    # The key is independent of Cooler row numbering and remains stable.
    for chrom in chrom_names:
        chrom_offsets[chrom] = int(tiles.loc[tiles["chrom"].eq(chrom), "target_start"].min()) // bin_size
    try:
        for start in tqdm(
            range(0, len(selected), chunk_size),
            total=(len(selected) + chunk_size - 1) // chunk_size,
            desc="Write canonical pairs",
            unit="batch",
        ):
            idx = selected[start : start + chunk_size]
            left = bin_i[idx].astype(np.int64)
            right = bin_j[idx].astype(np.int64)
            distances = (right - left) * bin_size
            rows = tile_row[idx]
            frame = pd.DataFrame(
                {
                    "pair_id": np.arange(start, start + len(idx), dtype=np.int64),
                    "chrom": np.asarray(chrom_names, dtype=object)[chrom_code[idx]],
                    "bin_i": left * bin_size,
                    "bin_j": right * bin_size,
                    "distance_bp": distances.astype(np.int32),
                    "distance_bin": (right - left).astype(np.int16),
                    "distance_band": distance_band(distances, config),
                    "tile_id": tiles["tile_id"].astype(str).to_numpy()[rows],
                    "tile_row": rows,
                    "split": tiles["split"].astype(str).to_numpy()[rows],
                    "boundary_margin_bins": boundary_margin[idx].astype(np.int16),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["chrom", "distance_band", "tile_id", "split"],
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)

    warnings: list[dict[str, Any]] = []
    report = {
        "output": str(output),
        "source_representations": int(n_candidates),
        "canonical_pairs": int(len(selected)),
        "duplicates_removed": int(n_candidates - canonical_pair_count),
        "conflicting_anchor_bins_identified": conflicting_anchor_bins,
        "split_buffer_pairs_removed": split_buffer_pairs_removed,
        "cross_split_anchor_overlap_after_purge": 0,
        "bin_size_bp": bin_size,
        "minimum_distance_bp": minimum,
        "maximum_distance_bp": maximum,
        "maximum_distance_bp_exclusive": maximum_exclusive,
        "resolution": resolution_contract(config),
        "tie_break": "maximum minimum anchor-to-tile-boundary margin, then tile_id",
        "split_counts": {},
        "warnings": warnings,
    }
    parquet = pq.ParquetFile(output)
    split_counts: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=["split"]):
        values, counts = np.unique(batch.column(0).to_numpy(zero_copy_only=False), return_counts=True)
        for value, count in zip(values, counts, strict=True):
            split_counts[str(value)] = split_counts.get(str(value), 0) + int(count)
    report["split_counts"] = split_counts
    update_manifest(
        output_root,
        "pair_export",
        {
            **report,
            "tiles": source_record(tiles_path),
            "config": source_record(Path(config["_config_path"])),
        },
    )
    atomic_json(output_root / "pair_export_report.json", report)
    return report
