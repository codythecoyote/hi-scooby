from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    configured_distance_bands,
    selected_zarr_row,
    sha256_file,
)
from .metrics import build_top_contact_groups
from .power import fractional_top


def fit_gamma_poisson_prior(
    counts: np.ndarray,
    baseline_mean: np.ndarray,
) -> float:
    y = np.asarray(counts, np.float64)
    mu = np.asarray(baseline_mean, np.float64)
    valid = np.isfinite(mu) & (mu > 0) & np.isfinite(y) & (y >= 0)
    if not valid.any():
        raise ValueError("Gamma-Poisson prior has no valid observations")
    y = y[valid]
    mu = mu[valid]

    def objective(log_shape: float) -> float:
        shape = math.exp(log_shape)
        likelihood = (
            gammaln(y + shape)
            - gammaln(shape)
            - gammaln(y + 1.0)
            + shape * (math.log(shape) - np.log(shape + mu))
            + y * (np.log(mu) - np.log(shape + mu))
        )
        return -float(likelihood.sum())

    result = minimize_scalar(objective, bounds=(-6.0, 12.0), method="bounded")
    if not result.success:
        raise RuntimeError("Gamma-Poisson prior fit failed")
    return float(math.exp(result.x))


def gamma_poisson_log_enrichment(
    counts: np.ndarray,
    baseline_mean: np.ndarray,
    shape: float,
) -> np.ndarray:
    if shape <= 0 or not np.isfinite(shape):
        raise ValueError("Gamma-Poisson shape must be positive")
    y = np.asarray(counts, np.float64)
    mu = np.asarray(baseline_mean, np.float64)
    return (np.log(y + shape) - np.log(mu + shape)).astype(np.float64)


def _fixed_baseline_mean(
    pairs: pd.DataFrame,
    counts: np.ndarray,
) -> tuple[np.ndarray, float]:
    raw = (
        np.clip(pairs["exposure"].to_numpy(np.float64), 1e-12, None)
        * np.exp(pairs["distance_offset"].to_numpy(np.float64))
    )
    training = pairs["split"].eq("train").to_numpy()
    denominator = float(raw[training].sum())
    scale = float(np.asarray(counts, np.float64)[training].sum()) / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not calibrate the training-only fixed baseline")
    return raw * scale, scale


def _bootstrap(
    frame: pd.DataFrame,
    *,
    value_fn,
    replicates: int,
    confidence_level: float,
    seed: int,
    progress_desc: str | None = None,
) -> dict[str, float | int | None]:
    chromosomes = np.asarray(sorted(frame["chrom"].astype(str).unique()), object)
    if not len(chromosomes):
        return {"replicates": 0, "lower": None, "median": None, "upper": None}
    groups = {
        str(chrom): frame.loc[frame["chrom"].astype(str).eq(str(chrom))]
        for chrom in chromosomes
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in tqdm(
        range(replicates),
        desc=progress_desc,
        unit="replicate",
        disable=progress_desc is None,
        leave=False,
    ):
        sample = rng.choice(chromosomes, len(chromosomes), replace=True)
        selected = pd.concat(
            [groups[str(chrom)] for chrom in sample], ignore_index=True
        )
        value = value_fn(selected)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"replicates": 0, "lower": None, "median": None, "upper": None}
    alpha = 1.0 - confidence_level
    array = np.asarray(values)
    return {
        "replicates": len(values),
        "lower": float(np.quantile(array, alpha)),
        "median": float(np.median(array)),
        "upper": float(np.quantile(array, confidence_level)),
    }


def _topology_rows(
    pairs: pd.DataFrame,
    counts: np.ndarray,
    residual: np.ndarray,
    exact_probability: np.ndarray,
    target_score: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = build_top_contact_groups(
        pairs["tile_row"].to_numpy(),
        pairs["distance_band"].to_numpy(),
    )
    top_rows = []
    exact_rows = []
    exposure = np.clip(pairs["exposure"].to_numpy(np.float64), 1e-12, None)
    distance = pairs["distance_bin"].to_numpy(np.int64)
    exact_code = pairs["tile_row"].to_numpy(np.int64) * (
        int(distance.max()) + 1
    ) + distance
    exact_codes = np.unique(exact_code)
    for code in tqdm(
        exact_codes,
        desc="Score exact-distance groups",
        unit="group",
    ):
        indices = np.flatnonzero(exact_code == code)
        events = float(counts[indices].sum())
        if events <= 0:
            continue
        probability = np.clip(exact_probability[indices], 1e-300, None)
        baseline_probability = exposure[indices] / exposure[indices].sum()
        exact_rows.append(
            {
                "chrom": str(pairs.iloc[indices[0]]["chrom"]),
                "band_id": str(pairs.iloc[indices[0]]["distance_band"]),
                "events": events,
                "log_likelihood_gain": float(
                    np.sum(
                        counts[indices]
                        * (np.log(probability) - np.log(baseline_probability))
                    )
                ),
            }
        )
    for band, tile_row, indices in tqdm(
        groups,
        desc="Score top-contact groups",
        unit="group",
    ):
        size = len(indices)
        k = max(1, int(math.ceil(size * top_fraction)))
        target = fractional_top(
            target_score[indices],
            k,
            eligible=counts[indices] > 0,
        )
        model = fractional_top(
            residual[indices],
            k,
            eligible=np.ones(size, bool),
        )
        if target is None or model is None:
            continue
        overlap = float(np.dot(target.weights, model.weights) / k)
        chance = k / size
        top_rows.append(
            {
                "chrom": str(pairs.iloc[indices[0]]["chrom"]),
                "tile_row": int(tile_row),
                "band_id": str(band),
                "k": k,
                "candidate_count": size,
                "overlap": overlap,
                "chance": chance,
                "overlap_excess": overlap - chance,
                "enrichment_over_chance": overlap / chance,
            }
        )
    return pd.DataFrame(top_rows), pd.DataFrame(exact_rows)


def _aggregate_top(frame: pd.DataFrame) -> dict[str, float] | None:
    if frame.empty:
        return None
    weights = frame["k"].to_numpy(float)
    overlap = float(np.average(frame["overlap"], weights=weights))
    chance = float(np.average(frame["chance"], weights=weights))
    return {
        "overlap": overlap,
        "chance": chance,
        "overlap_excess": overlap - chance,
        "enrichment_over_chance": overlap / chance,
    }


def _aggregate_likelihood(frame: pd.DataFrame) -> float | None:
    events = float(frame["events"].sum())
    return float(frame["log_likelihood_gain"].sum() / events) if events else None


def _replicate_top_rows(
    pairs: pd.DataFrame,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    *,
    top_fraction: float,
    progress_desc: str = "Score pseudoreplicate top contacts",
) -> pd.DataFrame:
    rows = []
    groups = build_top_contact_groups(
        pairs["tile_row"].to_numpy(),
        pairs["distance_band"].to_numpy(),
    )
    for band, tile_row, indices in tqdm(
        groups,
        desc=progress_desc,
        unit="group",
    ):
        size = len(indices)
        k = max(1, int(math.ceil(size * top_fraction)))
        top_a = fractional_top(
            counts_a[indices], k, eligible=counts_a[indices] > 0
        )
        top_b = fractional_top(
            counts_b[indices], k, eligible=counts_b[indices] > 0
        )
        if top_a is None or top_b is None:
            continue
        overlap = float(np.dot(top_a.weights, top_b.weights) / k)
        chance = k / size
        rows.append(
            {
                "chrom": str(pairs.iloc[indices[0]]["chrom"]),
                "tile_row": int(tile_row),
                "band_id": str(band),
                "k": k,
                "candidate_count": size,
                "overlap": overlap,
                "chance": chance,
                "overlap_excess": overlap - chance,
                "enrichment_over_chance": overlap / chance,
            }
        )
    return pd.DataFrame(rows)


def evaluate_shared_topology(
    config: dict[str, Any],
    *,
    prediction_path: Path,
    output: Path,
    split: str = "validation",
    freeze_test: bool = False,
    frozen_release: Path | None = None,
    test_lock: Path | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Shared topology gates are defined for validation or test")
    if split == "test" and not freeze_test:
        raise ValueError("Test topology evaluation requires freeze_test=True")
    if split == "test":
        if frozen_release is None or test_lock is None:
            raise ValueError(
                "Test topology evaluation requires frozen release and lock"
            )
        from .release import verify_test_lock

        verify_test_lock(frozen_release, test_lock)
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    authorized = pairs["split"].isin(["train", split]).to_numpy()
    authorized_ids = pairs.loc[authorized, "pair_id"].to_numpy(np.int64)
    split_ids = pairs.loc[
        pairs["split"].eq(split), "pair_id"
    ].to_numpy(np.int64)
    counts = np.zeros(len(pairs), np.uint64)
    half_a = np.zeros(len(pairs), np.uint64)
    half_b = np.zeros(len(pairs), np.uint64)
    for context in range(evidence["full_count"].shape[0]):
        counts += selected_zarr_row(
            evidence["full_count"],
            context,
            authorized_ids,
            pair_count=len(pairs),
            dtype=np.uint64,
        )
        half_a += selected_zarr_row(
            evidence["counts_a"],
            context,
            split_ids,
            pair_count=len(pairs),
            dtype=np.uint64,
        )
        half_b += selected_zarr_row(
            evidence["counts_b"],
            context,
            split_ids,
            pair_count=len(pairs),
            dtype=np.uint64,
        )
    baseline_mean, scale = _fixed_baseline_mean(pairs, counts)
    training = pairs["split"].eq("train").to_numpy()
    shape = fit_gamma_poisson_prior(counts[training], baseline_mean[training])
    target = gamma_poisson_log_enrichment(counts, baseline_mean, shape)
    predictions = pd.read_parquet(prediction_path).sort_values(
        "pair_id", kind="stable"
    )
    selected_split = pairs["split"].eq(split).to_numpy()
    selected_pairs = pairs.loc[selected_split].reset_index(drop=True)
    expected_ids = selected_pairs["pair_id"].to_numpy(np.int64)
    if not np.array_equal(
        predictions["pair_id"].to_numpy(np.int64), expected_ids
    ):
        raise ValueError(f"Topology predictions do not align with {split} pairs")
    residual = predictions["shared_residual_score"].to_numpy(np.float64)
    probability = predictions[
        "probability_within_owner_tile_exact_distance"
    ].to_numpy(np.float64)
    top_rows, likelihood_rows = _topology_rows(
        selected_pairs,
        counts[selected_split].astype(np.float64),
        residual,
        probability,
        target[selected_split],
        top_fraction=float(config["power"]["primary_top_fraction"]),
    )
    bands = [str(row["id"]) for row in configured_distance_bands(config)]
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["seed"])
    band_reports = {}
    power_report = None
    if split == "validation":
        power_report_path = (
            Path(config["outputs"]["results_root"]) / "power_gate.json"
        )
        with power_report_path.open() as handle:
            power_report = json.load(handle)
    for band in bands:
        top_band = top_rows.loc[top_rows["band_id"].eq(band)]
        ll_band = likelihood_rows.loc[likelihood_rows["band_id"].eq(band)]
        top_summary = _aggregate_top(top_band)
        top_bootstrap = _bootstrap(
            top_band,
            value_fn=lambda frame: (
                (_aggregate_top(frame) or {}).get("enrichment_over_chance")
            ),
            replicates=replicates,
            confidence_level=confidence,
            seed=seed,
            progress_desc=f"Bootstrap {band} top-contact enrichment",
        )
        ll_bootstrap = _bootstrap(
            ll_band,
            value_fn=_aggregate_likelihood,
            replicates=replicates,
            confidence_level=confidence,
            seed=seed + 1,
            progress_desc=f"Bootstrap {band} likelihood gain",
        )
        if power_report is not None:
            ceiling_matches = [
                row
                for row in power_report["summaries"]
                if row["output_id"] == "shared"
                and row["band_id"] == band
                and np.isclose(
                    float(row["top_fraction"]),
                    float(config["power"]["primary_top_fraction"]),
                )
            ]
            ceiling = ceiling_matches[0] if len(ceiling_matches) == 1 else None
            ceiling_excess = (
                float(ceiling["overlap_excess"]) if ceiling is not None else None
            )
        else:
            # The test split is touched exactly once. Its reliability ceiling
            # is computed from the already frozen primary pseudoreplicate,
            # without refitting any target or model component.
            ceiling_rows = _replicate_top_rows(
                selected_pairs,
                half_a[selected_split].astype(np.float64),
                half_b[selected_split].astype(np.float64),
                top_fraction=float(config["power"]["primary_top_fraction"]),
                progress_desc=f"Score {band} pseudoreplicate ceiling",
            )
            ceiling_band = ceiling_rows.loc[
                ceiling_rows["band_id"].eq(band)
            ]
            ceiling_summary = _aggregate_top(ceiling_band)
            ceiling_excess = (
                ceiling_summary["overlap_excess"]
                if ceiling_summary is not None
                else None
            )
        recovery_fraction = (
            float(top_summary["overlap_excess"]) / ceiling_excess
            if top_summary is not None
            and ceiling_excess is not None
            and ceiling_excess > 0
            else None
        )
        band_reports[band] = {
            "top1": top_summary,
            "top1_bootstrap_enrichment": top_bootstrap,
            "likelihood_gain_per_event": _aggregate_likelihood(ll_band),
            "likelihood_bootstrap": ll_bootstrap,
            "pseudoreplicate_ceiling_excess": ceiling_excess,
            "ceiling_excess_recovery_fraction": recovery_fraction,
        }
    exact_reports = {}
    distances = selected_pairs["distance_bp"].to_numpy(np.int64)
    selected_target = target[selected_split]
    a_selected = half_a[selected_split].astype(np.float64)
    b_selected = half_b[selected_split].astype(np.float64)
    for distance_bp in config["evaluation"]["exact_distances_bp"]:
        selected = distances == int(distance_bp)
        rows = []
        for chrom in sorted(selected_pairs.loc[selected, "chrom"].astype(str).unique()):
            local = selected & selected_pairs["chrom"].astype(str).eq(chrom).to_numpy()
            if local.sum() < 3:
                continue
            model_value = spearmanr(
                residual[local], selected_target[local]
            ).statistic
            reliability = spearmanr(
                a_selected[local], b_selected[local]
            ).statistic
            if np.isfinite(model_value) and np.isfinite(reliability):
                rows.append(
                    {
                        "chrom": chrom,
                        "candidate_count": int(local.sum()),
                        "model": float(model_value),
                        "reliability": float(reliability),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            exact_reports[str(distance_bp)] = {
                "model": None,
                "reliability": None,
                "reliability_ceiling": None,
                "reliability_fraction": None,
                "bootstrap": {"lower": None},
            }
            continue
        weights = frame["candidate_count"].to_numpy(float)
        model_value = float(np.average(frame["model"], weights=weights))
        reliability = float(np.average(frame["reliability"], weights=weights))
        reliability_ceiling = math.sqrt(max(reliability, 0.0))
        bootstrap = _bootstrap(
            frame,
            value_fn=lambda sample: float(
                np.average(sample["model"], weights=sample["candidate_count"])
            ),
            replicates=replicates,
            confidence_level=confidence,
            seed=seed + int(distance_bp),
            progress_desc=f"Bootstrap {distance_bp // 1_000} kb correlation",
        )
        exact_reports[str(distance_bp)] = {
            "model": model_value,
            "reliability": reliability,
            "reliability_ceiling": reliability_ceiling,
            "reliability_fraction": (
                model_value / reliability_ceiling
                if reliability_ceiling > 0
                else None
            ),
            "bootstrap": bootstrap,
        }
    minimum_enrichment = float(config["evaluation"]["minimum_top1_enrichment"])
    minimum_ceiling = float(
        config["evaluation"]["minimum_ceiling_excess_fraction"]
    )
    minimum_reliability = float(
        config["evaluation"]["minimum_reliability_fraction"]
    )
    checks = {}
    for band, metrics in band_reports.items():
        prefix = band.replace("-", "_")
        checks[f"{prefix}_likelihood"] = bool(
            metrics["likelihood_bootstrap"]["lower"] is not None
            and metrics["likelihood_bootstrap"]["lower"] > 0
        )
        checks[f"{prefix}_top1_point"] = bool(
            metrics["top1"] is not None
            and metrics["top1"]["enrichment_over_chance"] >= minimum_enrichment
        )
        checks[f"{prefix}_top1_bootstrap"] = bool(
            metrics["top1_bootstrap_enrichment"]["lower"] is not None
            and metrics["top1_bootstrap_enrichment"]["lower"] > 1
        )
        checks[f"{prefix}_ceiling_fraction"] = bool(
            metrics["ceiling_excess_recovery_fraction"] is not None
            and metrics["ceiling_excess_recovery_fraction"] >= minimum_ceiling
        )
    for distance_bp, metrics in exact_reports.items():
        checks[f"distance_{distance_bp}_positive"] = bool(
            metrics["bootstrap"]["lower"] is not None
            and metrics["bootstrap"]["lower"] > 0
        )
        checks[f"distance_{distance_bp}_reliability_fraction"] = bool(
            metrics["reliability_fraction"] is not None
            and metrics["reliability_fraction"] >= minimum_reliability
        )
    report = {
        "schema_version": 1,
        "split": split,
        "gamma_poisson_shape": shape,
        "baseline_scale": scale,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "bands": band_reports,
        "exact_distances": exact_reports,
        "checks": checks,
        "promoted": bool(checks) and all(checks.values()),
        "test_accessed": split == "test",
        "frozen_release_sha256": (
            sha256_file(frozen_release)
            if frozen_release is not None
            else None
        ),
    }
    atomic_json(output, report)
    top_rows.to_parquet(
        output.with_suffix(".top_groups.parquet"),
        index=False,
        compression="zstd",
    )
    likelihood_rows.to_parquet(
        output.with_suffix(".likelihood_groups.parquet"),
        index=False,
        compression="zstd",
    )
    return report
