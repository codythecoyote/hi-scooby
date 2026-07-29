from __future__ import annotations

from pathlib import Path
from typing import Any

import cooler
import pandas as pd
from tqdm.auto import tqdm

from .common import atomic_json, resolution_contract, source_record


TARGET_SPAN_BP = 1_000_000


def aligned_target_start(
    input_start: int,
    input_end: int,
    *,
    bin_size_bp: int,
) -> int:
    input_span = int(input_end) - int(input_start)
    if input_span < TARGET_SPAN_BP:
        raise ValueError("AlphaGenome input window is shorter than 1 Mb")
    ideal = int(input_start) + (input_span - TARGET_SPAN_BP) // 2
    return ((ideal + bin_size_bp // 2) // bin_size_bp) * bin_size_bp


def build_aligned_tiles(
    source_tiles: Path,
    output_tiles: Path,
    *,
    bin_size_bp: int,
) -> dict[str, Any]:
    tiles = pd.read_parquet(source_tiles).reset_index(drop=True)
    required = {
        "tile_id",
        "chrom",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
        "split",
        "embedding_path",
    }
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"Tile table lacks required columns: {sorted(missing)}")
    original_identity = tiles[
        ["tile_id", "chrom", "input_start", "input_end", "split", "embedding_path"]
    ].copy()
    starts = [
        aligned_target_start(
            int(row.input_start),
            int(row.input_end),
            bin_size_bp=bin_size_bp,
        )
        for row in tqdm(
            tiles.itertuples(index=False),
            total=len(tiles),
            desc=f"Align {bin_size_bp // 1_000} kb targets",
            unit="tile",
        )
    ]
    changed = int((tiles["target_start"].astype("int64") != starts).sum())
    tiles["target_start"] = starts
    tiles["target_end"] = tiles["target_start"] + TARGET_SPAN_BP
    if not tiles["target_start"].mod(bin_size_bp).eq(0).all():
        raise RuntimeError("One or more target intervals are not resolution aligned")
    if not (
        (tiles["target_start"] >= tiles["input_start"])
        & (tiles["target_end"] <= tiles["input_end"])
    ).all():
        raise RuntimeError("One or more aligned targets leave their input window")
    if not original_identity.equals(
        tiles[
            [
                "tile_id",
                "chrom",
                "input_start",
                "input_end",
                "split",
                "embedding_path",
            ]
        ]
    ):
        raise RuntimeError("Resolution alignment changed immutable tile identity")
    output_tiles.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_tiles.with_suffix(output_tiles.suffix + ".tmp")
    tiles.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_tiles)
    return {
        "rows": int(len(tiles)),
        "changed_target_starts": changed,
        "split_counts": {
            str(key): int(value)
            for key, value in tiles["split"].value_counts().items()
        },
        "bin_size_bp": int(bin_size_bp),
        "source": source_record(source_tiles),
        "output": source_record(output_tiles),
    }


def coarsen_count_cooler(
    source: Path,
    output: Path,
    *,
    target_bin_size_bp: int,
    chunksize: int = 10_000_000,
) -> dict[str, Any]:
    contact_map = cooler.Cooler(str(source))
    source_bin_size = contact_map.binsize
    if source_bin_size is None or target_bin_size_bp % int(source_bin_size):
        raise ValueError("Source Cooler binsize must divide target_bin_size_bp")
    factor = target_bin_size_bp // int(source_bin_size)
    if factor < 1:
        raise ValueError("Target Cooler cannot be finer than its source")

    def count_total(value: cooler.Cooler, *, description: str) -> int:
        pixels = value.pixels(join=False)
        total = 0
        starts = range(0, int(value.info["nnz"]), chunksize)
        for start in tqdm(
            starts,
            total=len(starts),
            desc=description,
            unit="pixel-chunk",
        ):
            total += int(
                pixels[start : start + chunksize]["count"]
                .astype("uint64")
                .sum()
            )
        return total

    source_total = count_total(
        contact_map, description="Audit source Cooler counts"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    cooler.coarsen_cooler(
        str(source),
        str(temporary),
        factor,
        chunksize=chunksize,
        columns=["count"],
        agg={"count": "sum"},
    )
    target = cooler.Cooler(str(temporary))
    if int(target.binsize or 0) != target_bin_size_bp:
        raise RuntimeError("Coarsened Cooler has the wrong binsize")
    target_total = count_total(
        target, description="Audit coarsened Cooler counts"
    )
    if source_total != target_total:
        raise RuntimeError(
            f"Cooler count conservation failed: {source_total} != {target_total}"
        )
    temporary.replace(output)
    return {
        "source_bin_size_bp": int(source_bin_size),
        "target_bin_size_bp": int(target_bin_size_bp),
        "factor": int(factor),
        "source_count": source_total,
        "target_count": target_total,
        "count_conservation": True,
        "balance_weights_consulted": False,
        "source": source_record(source),
        "output": source_record(output),
    }


def prepare_resolution_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    tiles_report = build_aligned_tiles(
        Path(paths["source_tiles"]),
        Path(paths["tiles"]),
        bin_size_bp=int(config["bin_size_bp"]),
    )
    cooler_report = coarsen_count_cooler(
        Path(paths["source_cooler"]),
        Path(paths["cooler"]),
        target_bin_size_bp=int(config["bin_size_bp"]),
    )
    report = {
        "schema_version": 1,
        "resolution": resolution_contract(config),
        "tiles": tiles_report,
        "cooler": cooler_report,
    }
    output = Path(paths["tiles"]).parent / "resolution_input_report.json"
    atomic_json(output, report)
    return report
