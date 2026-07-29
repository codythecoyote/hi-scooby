from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

from .common import atomic_json, selected_zarr_row, source_record, update_manifest


def _fit_monotone_distance_offset(
    distance_bin: np.ndarray,
    exposure: np.ndarray,
    training: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a decreasing, exposure-adjusted rate using training pairs only."""
    distance_bin = np.asarray(distance_bin, np.int64)
    exposure = np.asarray(exposure, np.float64)
    training = np.asarray(training, bool)
    counts = np.asarray(counts, np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(distance_bin):
        raise ValueError("counts must have shape [contexts, pairs]")
    valid = training & np.isfinite(exposure) & (exposure > 0)
    distances = np.unique(distance_bin[valid])
    if len(distances) < 2:
        raise ValueError("At least two training distances are required")
    event_sum = counts[:, valid].sum(axis=0)
    valid_distance = distance_bin[valid]
    valid_exposure = exposure[valid]
    context_count = counts.shape[0]
    rows = []
    for distance in distances:
        selected = valid_distance == distance
        events = float(event_sum[selected].sum())
        opportunity = float(valid_exposure[selected].sum() * context_count)
        # Jeffreys smoothing is negligible at production scale but keeps every
        # represented distance finite in small fixtures.
        raw_rate = (events + 0.5) / (opportunity + 0.5)
        rows.append((int(distance), events, opportunity, math.log(raw_rate)))
    curve = pd.DataFrame(
        rows, columns=["distance_bin", "event_count", "exposure_opportunity", "raw_log_rate"]
    )
    fitter = IsotonicRegression(increasing=False, out_of_bounds="clip")
    curve["distance_offset"] = fitter.fit_transform(
        curve["distance_bin"].to_numpy(float),
        curve["raw_log_rate"].to_numpy(float),
        sample_weight=np.maximum(curve["exposure_opportunity"].to_numpy(float), 1.0),
    )
    weights = np.maximum(
        curve["exposure_opportunity"].to_numpy(np.float64), 1.0
    )
    rates = np.exp(curve["distance_offset"].to_numpy(np.float64))
    normalizer = math.log(float(np.average(rates, weights=weights)))
    curve["distance_offset"] = curve["distance_offset"] - normalizer
    curve["normalization_log_rate"] = normalizer
    lookup = dict(zip(curve["distance_bin"], curve["distance_offset"], strict=True))
    offsets = np.asarray([lookup.get(int(value), np.nan) for value in distance_bin])
    if not np.isfinite(offsets).all():
        raise ValueError("Distance offset is undefined for one or more candidate distances")
    return offsets.astype(np.float32), curve


def _write_offset_column(path: Path, offsets: np.ndarray) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != len(offsets):
        raise ValueError("Pair table and distance offsets have different lengths")
    temporary = path.with_suffix(".offsets.parquet.tmp")
    writer: pq.ParquetWriter | None = None
    cursor = 0
    try:
        for batch in tqdm(parquet.iter_batches(batch_size=500_000), desc="Write fixed offsets"):
            frame = batch.to_pandas()
            stop = cursor + len(frame)
            frame["distance_offset"] = offsets[cursor:stop]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            cursor = stop
    finally:
        if writer is not None:
            writer.close()
    if cursor != len(offsets):
        raise RuntimeError("Did not write every distance offset")
    temporary.replace(path)


def fit_distance_offsets(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["outputs"]["data_root"])
    pairs_path = output_root / "canonical_pairs.parquet"
    pairs = pd.read_parquet(
        pairs_path, columns=["pair_id", "distance_bin", "split", "exposure"]
    ).sort_values("pair_id", kind="stable")
    expected_ids = np.arange(len(pairs), dtype=np.int64)
    if not np.array_equal(pairs["pair_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("pair_id must remain contiguous before fitting offsets")
    evidence_path = output_root / "pseudoreplicate_evidence.zarr"
    evidence = zarr.open_group(str(evidence_path), mode="r")
    training_ids = pairs.loc[
        pairs["split"].eq("train"), "pair_id"
    ].to_numpy(np.int64)
    counts = np.stack(
        [
            selected_zarr_row(
                evidence["full_count"],
                context,
                training_ids,
                pair_count=len(pairs),
                dtype=np.float64,
            )
            for context in range(evidence["full_count"].shape[0])
        ],
        axis=0,
    )
    offsets, curve = _fit_monotone_distance_offset(
        pairs["distance_bin"].to_numpy(),
        pairs["exposure"].to_numpy(),
        pairs["split"].eq("train").to_numpy(),
        counts,
    )
    _write_offset_column(pairs_path, offsets)
    curve_path = output_root / "distance_offset_curve.parquet"
    curve.to_parquet(curve_path, index=False, compression="zstd")
    report = {
        "schema_version": 2,
        "method": "training_only_exposure_adjusted_isotonic_decreasing",
        "normalization": (
            "training_exposure_weighted_mean_exp_distance_offset_equals_one"
        ),
        "training_pairs": int(pairs["split"].eq("train").sum()),
        "contexts": int(counts.shape[0]),
        "distances": int(len(curve)),
        "monotone_nonincreasing": bool(
            np.all(np.diff(curve["distance_offset"].to_numpy(float)) <= 1e-12)
        ),
        "curve": str(curve_path),
        "evidence": source_record(evidence_path),
    }
    atomic_json(output_root / "distance_offset_report.json", report)
    update_manifest(output_root, "distance_offsets", report)
    return report
