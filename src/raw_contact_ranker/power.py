from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import atomic_json, configured_distance_bands
from .metrics import build_top_contact_groups


POWER_GROUP_SCHEMA = pa.schema(
    [
        pa.field("output_id", pa.string()),
        pa.field("output_type", pa.string()),
        pa.field("split_index", pa.int64()),
        pa.field("chrom", pa.string()),
        pa.field("tile_row", pa.int64()),
        pa.field("band_id", pa.string()),
        pa.field("top_fraction", pa.float64()),
        pa.field("candidate_count", pa.int64()),
        pa.field("k", pa.int64()),
        pa.field("positive_a", pa.int64()),
        pa.field("positive_b", pa.int64()),
        pa.field("forced_zero_fraction_a", pa.float64()),
        pa.field("forced_zero_fraction_b", pa.float64()),
        pa.field("cutoff_a", pa.float64()),
        pa.field("cutoff_b", pa.float64()),
        pa.field("cutoff_tie_size_a", pa.int64()),
        pa.field("cutoff_tie_size_b", pa.int64()),
        pa.field("cutoff_weight_a", pa.float64()),
        pa.field("cutoff_weight_b", pa.float64()),
        pa.field("evaluable", pa.bool_()),
        pa.field("overlap", pa.float64()),
        pa.field("chance", pa.float64()),
        pa.field("overlap_excess", pa.float64()),
        pa.field("enrichment_over_chance", pa.float64()),
        pa.field("reason", pa.string()),
    ]
)


@dataclass(frozen=True)
class FractionalTop:
    weights: np.ndarray
    k: int
    positive_count: int
    cutoff: float
    cutoff_tie_size: int
    cutoff_weight: float


def fractional_top(
    values: np.ndarray,
    k: int,
    *,
    eligible: np.ndarray | None = None,
) -> FractionalTop | None:
    """Select exactly K eligible units with fractional, row-order-invariant ties."""
    values = np.asarray(values)
    if values.ndim != 1 or k < 1 or k > len(values):
        raise ValueError("Invalid positive-fractional top-K inputs")
    eligible_mask = (
        np.asarray(eligible, bool)
        if eligible is not None
        else values > 0
    )
    if eligible_mask.shape != values.shape:
        raise ValueError("Eligibility mask does not align with values")
    positive = np.flatnonzero(eligible_mask & np.isfinite(values))
    if len(positive) < k:
        return None
    positive_values = values[positive]
    order = np.argsort(-positive_values, kind="stable")
    cutoff = float(positive_values[order[k - 1]])
    above = positive[positive_values > cutoff]
    tied = positive[positive_values == cutoff]
    remaining = k - len(above)
    weight = remaining / len(tied)
    output = np.zeros(len(values), np.float64)
    output[above] = 1.0
    output[tied] = weight
    if not np.isclose(output.sum(), k):
        raise RuntimeError("Fractional top-K weights do not sum to K")
    return FractionalTop(
        weights=output,
        k=k,
        positive_count=int(len(positive)),
        cutoff=cutoff,
        cutoff_tie_size=int(len(tied)),
        cutoff_weight=float(weight),
    )


def positive_fractional_top(counts: np.ndarray, k: int) -> FractionalTop | None:
    """Select exactly K positive counts with fractional ties."""
    values = np.asarray(counts)
    return fractional_top(values, k, eligible=values > 0)


def power_rows(
    pairs: pd.DataFrame,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    *,
    output_id: str,
    output_type: str,
    split_index: int,
    fractions: Iterable[float],
    groups: list[tuple[str, int, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    required = {"chrom", "tile_row", "distance_band"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Power pairs lack columns: {sorted(missing)}")
    a = np.asarray(counts_a)
    b = np.asarray(counts_b)
    if a.shape != (len(pairs),) or b.shape != (len(pairs),):
        raise ValueError("Power count arrays must align to pairs")
    candidate_groups = groups or build_top_contact_groups(
        pairs["tile_row"].to_numpy(),
        pairs["distance_band"].to_numpy(),
    )
    chromosomes = pairs["chrom"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for band, tile_row, indices in candidate_groups:
        candidate_count = int(len(indices))
        chrom = str(chromosomes[indices[0]])
        for fraction_value in fractions:
            fraction = float(fraction_value)
            k = max(1, int(math.ceil(candidate_count * fraction)))
            top_a = positive_fractional_top(a[indices], k)
            top_b = positive_fractional_top(b[indices], k)
            evaluable = top_a is not None and top_b is not None
            overlap = chance = excess = enrichment = None
            if evaluable:
                assert top_a is not None and top_b is not None
                overlap = float(np.dot(top_a.weights, top_b.weights) / k)
                chance = float(k / candidate_count)
                excess = overlap - chance
                enrichment = overlap / chance if chance > 0 else None
            rows.append(
                {
                    "output_id": output_id,
                    "output_type": output_type,
                    "split_index": int(split_index),
                    "chrom": chrom,
                    "tile_row": int(tile_row),
                    "band_id": str(band),
                    "top_fraction": fraction,
                    "candidate_count": candidate_count,
                    "k": k,
                    "positive_a": int(np.count_nonzero(a[indices] > 0)),
                    "positive_b": int(np.count_nonzero(b[indices] > 0)),
                    "forced_zero_fraction_a": float(
                        max(0, k - np.count_nonzero(a[indices] > 0)) / k
                    ),
                    "forced_zero_fraction_b": float(
                        max(0, k - np.count_nonzero(b[indices] > 0)) / k
                    ),
                    "cutoff_a": top_a.cutoff if top_a is not None else None,
                    "cutoff_b": top_b.cutoff if top_b is not None else None,
                    "cutoff_tie_size_a": (
                        top_a.cutoff_tie_size if top_a is not None else None
                    ),
                    "cutoff_tie_size_b": (
                        top_b.cutoff_tie_size if top_b is not None else None
                    ),
                    "cutoff_weight_a": (
                        top_a.cutoff_weight if top_a is not None else None
                    ),
                    "cutoff_weight_b": (
                        top_b.cutoff_weight if top_b is not None else None
                    ),
                    "evaluable": bool(evaluable),
                    "overlap": overlap,
                    "chance": chance,
                    "overlap_excess": excess,
                    "enrichment_over_chance": enrichment,
                    "reason": None if evaluable else "positive_capacity_below_k",
                }
            )
    return rows


class PowerParquetWriter:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.temporary = output.with_suffix(output.suffix + ".tmp")
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary.exists():
            self.temporary.unlink()

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        # A whole context can be unevaluable, leaving every optional metric
        # as None in one batch. An explicit schema prevents Arrow from
        # inferring `null` and rejecting a later numeric batch (or vice versa).
        table = pa.Table.from_pylist(rows, schema=POWER_GROUP_SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary,
                POWER_GROUP_SCHEMA,
                compression="zstd",
                use_dictionary=["output_id", "output_type", "chrom", "band_id"],
            )
        self.writer.write_table(table)
        self.row_count += len(rows)

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError("Power audit produced no rows")
        self.writer.close()
        self.writer = None
        self.temporary.replace(self.output)

    def __enter__(self) -> "PowerParquetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if exc_type is None:
            self.temporary.replace(self.output)
        elif self.temporary.exists():
            self.temporary.unlink()


def _aggregate_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    k_total = float(frame["k"].sum())
    evaluable = frame["evaluable"].astype(bool)
    evaluable_k = float(frame.loc[evaluable, "k"].sum())
    if not evaluable.any():
        return {
            "rows": int(len(frame)),
            "evaluable_rows": 0,
            "k_weighted_evaluable_fraction": 0.0,
            "chromosomes": 0,
            "overlap": None,
            "chance": None,
            "overlap_excess": None,
            "enrichment_over_chance": None,
        }
    weights = frame.loc[evaluable, "k"].to_numpy(float)
    overlap = float(
        np.average(frame.loc[evaluable, "overlap"].to_numpy(float), weights=weights)
    )
    chance = float(
        np.average(frame.loc[evaluable, "chance"].to_numpy(float), weights=weights)
    )
    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(evaluable.sum()),
        "k_weighted_evaluable_fraction": evaluable_k / k_total if k_total else 0.0,
        "chromosomes": int(frame.loc[evaluable, "chrom"].nunique()),
        "overlap": overlap,
        "chance": chance,
        "overlap_excess": overlap - chance,
        "enrichment_over_chance": overlap / chance if chance > 0 else None,
    }


def chromosome_bootstrap_excess(
    frame: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | None]:
    evaluable = frame.loc[frame["evaluable"].astype(bool)].copy()
    chromosomes = np.asarray(sorted(evaluable["chrom"].unique()), dtype=object)
    if not len(chromosomes):
        return {"replicates": 0, "lower": None, "median": None, "upper": None}
    by_chrom = {
        str(chrom): evaluable.loc[evaluable["chrom"].eq(chrom)]
        for chrom in chromosomes
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    for index in range(replicates):
        sample = rng.choice(chromosomes, size=len(chromosomes), replace=True)
        selected = pd.concat([by_chrom[str(chrom)] for chrom in sample], ignore_index=True)
        values[index] = float(_aggregate_metrics(selected)["overlap_excess"])
    alpha = 1.0 - confidence_level
    return {
        "replicates": int(replicates),
        "lower": float(np.quantile(values, alpha)),
        "median": float(np.median(values)),
        "upper": float(np.quantile(values, confidence_level)),
    }


def summarize_power_audit(
    config: dict[str, Any],
    power_path: Path,
    *,
    output: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(power_path)
    primary_fraction = float(config["power"]["primary_top_fraction"])
    summary_rows = []
    grouped = frame.groupby(
        ["output_id", "output_type", "band_id", "top_fraction"],
        observed=True,
        sort=True,
    )
    for keys, group in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Summarize power groups",
        unit="summary",
    ):
        metrics = _aggregate_metrics(group)
        bootstrap = (
            chromosome_bootstrap_excess(
                group,
                replicates=int(config["power"]["bootstrap_replicates"]),
                confidence_level=float(config["power"]["confidence_level"]),
                seed=int(config["seed"]),
            )
            if np.isclose(float(keys[3]), primary_fraction)
            else None
        )
        summary_rows.append(
            {
                "output_id": str(keys[0]),
                "output_type": str(keys[1]),
                "band_id": str(keys[2]),
                "top_fraction": float(keys[3]),
                **metrics,
                "overlap_excess_bootstrap": bootstrap,
            }
        )
    pooled = [
        row
        for row in summary_rows
        if row["output_id"] == "shared"
        and row["output_type"] == "shared"
        and np.isclose(row["top_fraction"], primary_fraction)
    ]
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    minimum_coverage = float(config["power"]["minimum_evaluable_k_fraction"])
    minimum_chromosomes = int(config["power"]["minimum_validation_chromosomes"])
    minimum_enrichment = float(config["power"]["minimum_ceiling_enrichment"])
    checks: dict[str, bool] = {}
    for band in band_ids:
        matches = [row for row in pooled if row["band_id"] == band]
        row = matches[0] if len(matches) == 1 else None
        prefix = band.replace("-", "_")
        checks[f"{prefix}_complete"] = row is not None
        checks[f"{prefix}_coverage"] = bool(
            row is not None
            and float(row["k_weighted_evaluable_fraction"]) >= minimum_coverage
        )
        checks[f"{prefix}_chromosomes"] = bool(
            row is not None and int(row["chromosomes"]) >= minimum_chromosomes
        )
        checks[f"{prefix}_enrichment"] = bool(
            row is not None
            and row["enrichment_over_chance"] is not None
            and float(row["enrichment_over_chance"]) >= minimum_enrichment
        )
        checks[f"{prefix}_bootstrap"] = bool(
            row is not None
            and row["overlap_excess_bootstrap"] is not None
            and row["overlap_excess_bootstrap"]["lower"] is not None
            and float(row["overlap_excess_bootstrap"]["lower"]) > 0
        )
    report = {
        "schema_version": 1,
        "power_path": str(power_path),
        "primary_top_fraction": primary_fraction,
        "summaries": summary_rows,
        "checks": checks,
        "eligible": bool(checks) and all(checks.values()),
        "tie_mode": "positive_fractional",
        "zero_padding": False,
    }
    atomic_json(output, report)
    return report
