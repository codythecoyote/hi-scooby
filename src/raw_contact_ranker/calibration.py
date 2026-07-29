from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    configured_distance_bands,
    selected_zarr_row,
    sha256_file,
)


def nb2_logpmf(
    counts: np.ndarray,
    mean: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    y = np.asarray(counts, np.float64)
    mu = np.asarray(mean, np.float64)
    if dispersion <= 0 or np.any(mu <= 0) or np.any(y < 0):
        raise ValueError("Invalid NB2 parameters")
    size = 1.0 / dispersion
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * (np.log(size) - np.log(size + mu))
        + y * (np.log(mu) - np.log(size + mu))
    )


def fit_nb2_offset(
    counts: np.ndarray,
    log_offset: np.ndarray,
    *,
    fixed_dispersion: float | None = None,
    prior_mean: float | None = None,
    prior_sd: float | None = None,
) -> dict[str, float]:
    y = np.asarray(counts, np.float64)
    offset = np.asarray(log_offset, np.float64)
    valid = np.isfinite(offset) & np.isfinite(y) & (y >= 0)
    y = y[valid]
    offset = offset[valid]
    if not len(y):
        raise ValueError("NB2 calibration has no valid observations")
    initial_alpha = np.log((y.sum() + 0.5) / (np.exp(offset).sum() + 0.5))

    if (prior_mean is None) != (prior_sd is None):
        raise ValueError("NB2 intercept prior mean and SD must be paired")
    if prior_sd is not None and prior_sd <= 0:
        raise ValueError("NB2 intercept prior SD must be positive")

    def objective(parameters: np.ndarray) -> float:
        alpha = float(parameters[0])
        dispersion = (
            float(fixed_dispersion)
            if fixed_dispersion is not None
            else float(np.exp(parameters[1]))
        )
        mean = np.exp(np.clip(offset + alpha, -40, 30))
        value = -float(nb2_logpmf(y, mean, dispersion).sum())
        if prior_mean is not None and prior_sd is not None:
            value += 0.5 * ((alpha - prior_mean) / prior_sd) ** 2
        return value

    if fixed_dispersion is not None and fixed_dispersion <= 0:
        raise ValueError("Fixed NB2 dispersion must be positive")
    result = minimize(
        objective,
        x0=(
            np.asarray([initial_alpha])
            if fixed_dispersion is not None
            else np.asarray([initial_alpha, -4.0])
        ),
        method="L-BFGS-B",
        bounds=(
            [(-30.0, 30.0)]
            if fixed_dispersion is not None
            else [(-30.0, 30.0), (-12.0, 8.0)]
        ),
    )
    if not result.success:
        raise RuntimeError(f"NB2 calibration failed: {result.message}")
    alpha = float(result.x[0])
    dispersion = (
        float(fixed_dispersion)
        if fixed_dispersion is not None
        else float(np.exp(result.x[1]))
    )
    mean = np.exp(np.clip(offset + alpha, -40, 30))
    return {
        "alpha": alpha,
        "dispersion": dispersion,
        "training_log_likelihood": float(
            nb2_logpmf(y, mean, dispersion).sum()
        ),
        "intercept_shrunk": prior_mean is not None,
    }


def calibration_slope(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    bins: int = 10,
) -> float | None:
    y = np.asarray(observed, np.float64)
    mu = np.asarray(predicted, np.float64)
    if len(y) < bins or np.any(mu <= 0):
        return None
    quantile = pd.qcut(mu, q=min(bins, len(np.unique(mu))), duplicates="drop")
    frame = pd.DataFrame({"observed": y, "predicted": mu, "bin": quantile})
    summary = frame.groupby("bin", observed=True).mean()
    if len(summary) < 3:
        return None
    epsilon = max(float(summary["predicted"].min()), 1e-12)
    x = np.log(summary["predicted"].to_numpy() + epsilon)
    z = np.log(summary["observed"].to_numpy() + epsilon)
    return float(np.polyfit(x, z, 1)[0])


def _output_counts_and_depths(
    config: dict[str, Any],
    evidence,
    *,
    pair_ids: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    context_names = contexts["cell_type"].astype(str).tolist()
    pair_count = int(evidence["full_count"].shape[1])
    ids = (
        np.arange(pair_count, dtype=np.int64)
        if pair_ids is None
        else np.asarray(pair_ids, np.int64)
    )
    context_counts = {}
    for index, name in tqdm(
        enumerate(context_names),
        total=len(context_names),
        desc="Load output calibration counts",
        unit="context",
    ):
        context_counts[name] = selected_zarr_row(
            evidence["full_count"],
            index,
            ids,
            pair_count=pair_count,
            dtype=np.uint64,
        )
    context_depth = {
        str(row.cell_type): int(row.valid_pairs)
        for row in contexts.itertuples(index=False)
    }
    outputs: dict[str, dict[str, Any]] = {
        "shared": {
            "output_type": "shared",
            "members": context_names,
            "counts": np.sum(
                np.stack(list(context_counts.values()), axis=0),
                axis=0,
                dtype=np.uint64,
            ),
            "depth": int(sum(context_depth.values())),
        }
    }
    for name in context_names:
        outputs[name] = {
            "output_type": "context",
            "members": [name],
            "counts": context_counts[name],
            "depth": context_depth[name],
        }
    for pool in config.get("contexts", {}).get("pools", []):
        members = [str(value) for value in pool["members"]]
        outputs[str(pool["id"])] = {
            "output_type": "pool",
            "members": members,
            "counts": np.sum(
                np.stack([context_counts[name] for name in members], axis=0),
                axis=0,
                dtype=np.uint64,
            ),
            "depth": int(sum(context_depth[name] for name in members)),
        }
    return outputs


def _chromosome_bootstrap_gain(
    rows: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | None]:
    chromosomes = np.asarray(sorted(rows["chrom"].astype(str).unique()), object)
    if not len(chromosomes):
        return {"replicates": 0, "lower": None, "median": None, "upper": None}
    grouped = {
        str(chrom): rows.loc[rows["chrom"].astype(str).eq(str(chrom))]
        for chrom in chromosomes
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sample = rng.choice(chromosomes, len(chromosomes), replace=True)
        selected = pd.concat(
            [grouped[str(chrom)] for chrom in sample], ignore_index=True
        )
        values.append(
            float(
                (
                    selected["model_log_likelihood"]
                    - selected["baseline_log_likelihood"]
                ).sum()
                / max(float(selected["events"].sum()), 1.0)
            )
        )
    alpha = 1.0 - confidence_level
    array = np.asarray(values)
    return {
        "replicates": len(values),
        "lower": float(np.quantile(array, alpha)),
        "median": float(np.median(array)),
        "upper": float(np.quantile(array, confidence_level)),
    }


def fit_rate_calibration(
    config: dict[str, Any],
    *,
    train_predictions: Path,
    validation_predictions: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    authorized_ids = pairs.loc[
        pairs["split"].isin(["train", "validation"]), "pair_id"
    ].to_numpy(np.int64)
    outputs = _output_counts_and_depths(
        config, evidence, pair_ids=authorized_ids
    )
    train = pd.read_parquet(train_predictions).sort_values("pair_id", kind="stable")
    validation = pd.read_parquet(validation_predictions).sort_values(
        "pair_id", kind="stable"
    )
    train_pairs = pairs.loc[pairs["split"].eq("train")].reset_index(drop=True)
    validation_pairs = pairs.loc[pairs["split"].eq("validation")].reset_index(
        drop=True
    )
    if not np.array_equal(train["pair_id"], train_pairs["pair_id"]):
        raise ValueError("Training predictions do not align with canonical pairs")
    if not np.array_equal(validation["pair_id"], validation_pairs["pair_id"]):
        raise ValueError("Validation predictions do not align with canonical pairs")
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    parameters: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    shrinkage_sd = float(
        config["calibration"].get("intercept_shrinkage_sd", 2.0)
    )
    for output_id, output_data in tqdm(
        outputs.items(),
        total=len(outputs),
        desc="Fit and validate NB2 outputs",
        unit="output",
    ):
        counts = np.asarray(output_data["counts"], np.float64)
        depth = int(output_data["depth"])
        unit_parameters = {}
        unit_metrics = {}
        for band in band_ids:
            train_mask = train_pairs["distance_band"].eq(band).to_numpy()
            validation_mask = validation_pairs["distance_band"].eq(band).to_numpy()
            train_ids = train_pairs.loc[train_mask, "pair_id"].to_numpy(np.int64)
            validation_ids = validation_pairs.loc[
                validation_mask, "pair_id"
            ].to_numpy(np.int64)
            train_base = (
                np.log(depth / float(config["calibration"]["reference_depth"]))
                + np.log(
                    np.clip(
                        train_pairs.loc[train_mask, "exposure"].to_numpy(float),
                        1e-12,
                        None,
                    )
                )
                + train_pairs.loc[train_mask, "distance_offset"].to_numpy(float)
            )
            validation_base = (
                np.log(depth / float(config["calibration"]["reference_depth"]))
                + np.log(
                    np.clip(
                        validation_pairs.loc[
                            validation_mask, "exposure"
                        ].to_numpy(float),
                        1e-12,
                        None,
                    )
                )
                + validation_pairs.loc[
                    validation_mask, "distance_offset"
                ].to_numpy(float)
            )
            train_residual = train.loc[
                train["pair_id"].isin(train_ids), "shared_residual_score"
            ].to_numpy(float)
            validation_residual = validation.loc[
                validation["pair_id"].isin(validation_ids),
                "shared_residual_score",
            ].to_numpy(float)
            if output_id == "shared":
                model_fit = fit_nb2_offset(
                    counts[train_ids],
                    train_base + train_residual,
                )
                baseline_fit = fit_nb2_offset(counts[train_ids], train_base)
            else:
                shared_band = parameters["shared"]["bands"][band]
                model_fit = fit_nb2_offset(
                    counts[train_ids],
                    train_base + train_residual,
                    fixed_dispersion=float(
                        shared_band["model"]["dispersion"]
                    ),
                    prior_mean=float(shared_band["model"]["alpha"]),
                    prior_sd=shrinkage_sd,
                )
                baseline_fit = fit_nb2_offset(
                    counts[train_ids],
                    train_base,
                    fixed_dispersion=float(
                        shared_band["baseline"]["dispersion"]
                    ),
                    prior_mean=float(shared_band["baseline"]["alpha"]),
                    prior_sd=shrinkage_sd,
                )
            model_mean = np.exp(
                np.clip(
                    validation_base
                    + validation_residual
                    + model_fit["alpha"],
                    -40,
                    30,
                )
            )
            baseline_mean = np.exp(
                np.clip(validation_base + baseline_fit["alpha"], -40, 30)
            )
            observed = counts[validation_ids]
            model_ll = nb2_logpmf(
                observed, model_mean, model_fit["dispersion"]
            )
            baseline_ll = nb2_logpmf(
                observed, baseline_mean, baseline_fit["dispersion"]
            )
            chrom = validation_pairs.loc[
                validation_mask, "chrom"
            ].astype(str).to_numpy()
            rows = pd.DataFrame(
                {
                    "chrom": chrom,
                    "events": observed,
                    "model_log_likelihood": model_ll,
                    "baseline_log_likelihood": baseline_ll,
                }
            )
            bootstrap = _chromosome_bootstrap_gain(
                rows,
                replicates=bootstrap_replicates,
                confidence_level=confidence,
                seed=int(config["seed"]),
            )
            ratio = float(observed.sum() / model_mean.sum())
            slope = calibration_slope(observed, model_mean)
            unit_parameters[band] = {
                "model": model_fit,
                "baseline": baseline_fit,
                "depth": depth,
            }
            unit_metrics[band] = {
                "observed_predicted_ratio": ratio,
                "calibration_slope": slope,
                "nb_log_likelihood_gain_per_event": float(
                    (model_ll.sum() - baseline_ll.sum())
                    / max(observed.sum(), 1.0)
                ),
                "chromosome_bootstrap_gain_per_event": bootstrap,
            }
        parameters[output_id] = {
            "output_type": output_data["output_type"],
            "members": output_data["members"],
            "bands": unit_parameters,
        }
        metrics[output_id] = unit_metrics
    ratio_min, ratio_max = map(
        float, config["calibration"]["observed_predicted_ratio"]
    )
    slope_min, slope_max = map(float, config["calibration"]["calibration_slope"])
    unit_checks = {}
    unit_accepted = {}
    for output_id, output_metrics in metrics.items():
        checks = {}
        for band in band_ids:
            row = output_metrics[band]
            checks[f"{band}_deviance"] = bool(
                row["chromosome_bootstrap_gain_per_event"]["lower"] is not None
                and row["chromosome_bootstrap_gain_per_event"]["lower"] > 0
            )
            checks[f"{band}_total_ratio"] = bool(
                ratio_min <= row["observed_predicted_ratio"] <= ratio_max
            )
            checks[f"{band}_slope"] = bool(
                row["calibration_slope"] is not None
                and slope_min <= row["calibration_slope"] <= slope_max
            )
        unit_checks[output_id] = checks
        unit_accepted[output_id] = bool(checks) and all(checks.values())
    checks = unit_checks["shared"]
    report = {
        "schema_version": 1,
        "distribution": "nb2",
        "intercept_shrinkage": {
            "reference": "shared_band_intercept",
            "prior_sd": shrinkage_sd,
        },
        "dispersion_scope": "one_shared_parameter_per_distance_band",
        "reference_depth": int(config["calibration"]["reference_depth"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_predictions": str(train_predictions),
        "training_predictions_sha256": sha256_file(train_predictions),
        "validation_predictions": str(validation_predictions),
        "validation_predictions_sha256": sha256_file(validation_predictions),
        "parameters": parameters,
        "validation_metrics": metrics,
        "checks": checks,
        "unit_checks": unit_checks,
        "unit_accepted": unit_accepted,
        "accepted": bool(checks) and all(checks.values()),
        "rank_scores_modified": False,
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report


def evaluate_frozen_calibration(
    config: dict[str, Any],
    *,
    predictions: Path,
    calibration_path: Path,
    rollout_path: Path,
    frozen_release: Path,
    test_lock: Path,
    output: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate, but never refit, a frozen NB2 calibration."""
    if split != "test":
        raise ValueError("Frozen calibration evaluation is reserved for test")
    from .release import verify_test_lock

    frozen = verify_test_lock(frozen_release, test_lock)
    if (
        frozen["inputs"]["calibration_gate"]["sha256"]
        != sha256_file(calibration_path)
        or frozen["inputs"]["rollout"]["sha256"] != sha256_file(rollout_path)
    ):
        raise RuntimeError("Test calibration inputs changed after release freeze")
    with calibration_path.open() as handle:
        calibration = json.load(handle)
    with rollout_path.open() as handle:
        rollout = json.load(handle)
    if not calibration.get("accepted"):
        raise RuntimeError("Unaccepted validation calibration cannot touch test")
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    selected_pairs = pairs.loc[pairs["split"].eq(split)].reset_index(drop=True)
    prediction = pd.read_parquet(predictions).sort_values(
        "pair_id", kind="stable"
    )
    if not np.array_equal(
        prediction["pair_id"].to_numpy(np.int64),
        selected_pairs["pair_id"].to_numpy(np.int64),
    ):
        raise ValueError("Frozen calibration predictions do not align with test")
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    outputs = _output_counts_and_depths(
        config,
        evidence,
        pair_ids=selected_pairs["pair_id"].to_numpy(np.int64),
    )
    reference_depth = float(calibration["reference_depth"])
    residual = prediction["shared_residual_score"].to_numpy(np.float64)
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    source_rows: dict[str, dict[str, Any]] = {}
    for row in rollout["outputs"]:
        source = str(row["calibration_source"])
        if source in source_rows:
            prior = source_rows[source]
            if (
                prior.get("context_delta_path") != row.get("context_delta_path")
                or prior.get("context_delta_column")
                != row.get("context_delta_column")
            ):
                raise RuntimeError(
                    f"Rollout uses inconsistent topology for calibration source {source}"
                )
        else:
            source_rows[source] = row
    required_sources = sorted(source_rows)
    delta_cache: dict[tuple[str, str], np.ndarray] = {}

    def source_delta(output_id: str) -> np.ndarray:
        row = source_rows[output_id]
        path = row.get("context_delta_path")
        column = row.get("context_delta_column")
        if path is None and column is None:
            return np.zeros(len(selected_pairs), np.float64)
        if not path or not column:
            raise ValueError("Incomplete rollout context-delta reference")
        key = (str(path), str(column))
        if key not in delta_cache:
            frame = pd.read_parquet(
                path, columns=["pair_id", str(column)]
            ).sort_values("pair_id", kind="stable")
            selected_frame = frame.loc[
                frame["pair_id"].isin(
                    selected_pairs["pair_id"].to_numpy(np.int64)
                )
            ]
            if not np.array_equal(
                selected_frame["pair_id"].to_numpy(np.int64),
                selected_pairs["pair_id"].to_numpy(np.int64),
            ):
                raise ValueError("Context delta does not cover test pairs")
            delta_cache[key] = selected_frame[str(column)].to_numpy(np.float64)
        return delta_cache[key]

    metrics: dict[str, Any] = {}
    for output_id in tqdm(
        required_sources,
        desc="Evaluate frozen NB2 outputs",
        unit="output",
    ):
        output_data = outputs[output_id]
        if output_id not in calibration["parameters"]:
            raise RuntimeError(f"Calibration lacks output {output_id}")
        counts = np.asarray(output_data["counts"], np.float64)
        depth = float(output_data["depth"])
        delta = source_delta(output_id)
        unit_metrics = {}
        for band, band_parameters in calibration["parameters"][output_id][
            "bands"
        ].items():
            selected = selected_pairs["distance_band"].eq(band).to_numpy()
            pair_ids = selected_pairs.loc[selected, "pair_id"].to_numpy(np.int64)
            base = (
                math.log(depth / reference_depth)
                + np.log(
                    np.clip(
                        selected_pairs.loc[selected, "exposure"].to_numpy(float),
                        1e-12,
                        None,
                    )
                )
                + selected_pairs.loc[
                    selected, "distance_offset"
                ].to_numpy(float)
            )
            model_fit = band_parameters["model"]
            baseline_fit = band_parameters["baseline"]
            model_mean = np.exp(
                np.clip(
                    base
                    + residual[selected]
                    + delta[selected]
                    + float(model_fit["alpha"]),
                    -40,
                    30,
                )
            )
            baseline_mean = np.exp(
                np.clip(base + float(baseline_fit["alpha"]), -40, 30)
            )
            observed = counts[pair_ids]
            model_ll = nb2_logpmf(
                observed, model_mean, float(model_fit["dispersion"])
            )
            baseline_ll = nb2_logpmf(
                observed, baseline_mean, float(baseline_fit["dispersion"])
            )
            rows = pd.DataFrame(
                {
                    "chrom": selected_pairs.loc[
                        selected, "chrom"
                    ].astype(str).to_numpy(),
                    "events": observed,
                    "model_log_likelihood": model_ll,
                    "baseline_log_likelihood": baseline_ll,
                }
            )
            unit_metrics[band] = {
                "observed_predicted_ratio": float(
                    observed.sum() / model_mean.sum()
                ),
                "calibration_slope": calibration_slope(observed, model_mean),
                "nb_log_likelihood_gain_per_event": float(
                    (model_ll.sum() - baseline_ll.sum())
                    / max(float(observed.sum()), 1.0)
                ),
                "chromosome_bootstrap_gain_per_event": (
                    _chromosome_bootstrap_gain(
                        rows,
                        replicates=replicates,
                        confidence_level=confidence,
                        seed=int(config["seed"]),
                    )
                ),
            }
        metrics[output_id] = unit_metrics
    ratio_min, ratio_max = map(
        float, config["calibration"]["observed_predicted_ratio"]
    )
    slope_min, slope_max = map(float, config["calibration"]["calibration_slope"])
    unit_checks = {}
    unit_accepted = {}
    for output_id in required_sources:
        checks = {}
        for band, row in metrics[output_id].items():
            checks[f"{band}_deviance"] = bool(
                row["chromosome_bootstrap_gain_per_event"]["lower"] is not None
                and row["chromosome_bootstrap_gain_per_event"]["lower"] > 0
            )
            checks[f"{band}_total_ratio"] = bool(
                ratio_min <= row["observed_predicted_ratio"] <= ratio_max
            )
            checks[f"{band}_slope"] = bool(
                row["calibration_slope"] is not None
                and slope_min <= row["calibration_slope"] <= slope_max
            )
        unit_checks[output_id] = checks
        unit_accepted[output_id] = bool(checks) and all(checks.values())
    checks = {
        f"{output_id}:{name}": passed
        for output_id, values in unit_checks.items()
        for name, passed in values.items()
    }
    report = {
        "schema_version": 1,
        "split": split,
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "rollout": str(rollout_path),
        "rollout_sha256": sha256_file(rollout_path),
        "frozen_release_sha256": sha256_file(frozen_release),
        "prediction_path": str(predictions),
        "prediction_sha256": sha256_file(predictions),
        "validation_checkpoint_sha256": calibration["checkpoint_sha256"],
        "metrics": metrics,
        "checks": checks,
        "unit_checks": unit_checks,
        "unit_accepted": unit_accepted,
        "accepted": bool(checks) and all(checks.values()),
        "parameters_refitted": False,
        "test_accessed": True,
    }
    atomic_json(output, report)
    return report


def calibrate_context_extension(
    config: dict[str, Any],
    *,
    shared_calibration_path: Path,
    context_gate_path: Path,
    shared_train_predictions: Path,
    shared_validation_predictions: Path,
    calibration_output: Path,
    gate_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recalibrate accepted context ranks without changing their ordering."""
    with shared_calibration_path.open() as handle:
        calibration = json.load(handle)
    with context_gate_path.open() as handle:
        context_gate = json.load(handle)
    if not calibration.get("accepted") or context_gate.get("test_accessed"):
        raise RuntimeError("Context calibration requires accepted validation inputs")
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    train_pairs = pairs.loc[pairs["split"].eq("train")].reset_index(drop=True)
    validation_pairs = pairs.loc[
        pairs["split"].eq("validation")
    ].reset_index(drop=True)
    shared_train = pd.read_parquet(shared_train_predictions).sort_values(
        "pair_id", kind="stable"
    )
    shared_validation = pd.read_parquet(
        shared_validation_predictions
    ).sort_values("pair_id", kind="stable")
    if not np.array_equal(
        shared_train["pair_id"].to_numpy(np.int64),
        train_pairs["pair_id"].to_numpy(np.int64),
    ) or not np.array_equal(
        shared_validation["pair_id"].to_numpy(np.int64),
        validation_pairs["pair_id"].to_numpy(np.int64),
    ):
        raise ValueError("Shared predictions do not align for context calibration")
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    authorized_ids = pairs.loc[
        pairs["split"].isin(["train", "validation"]), "pair_id"
    ].to_numpy(np.int64)
    output_data = _output_counts_and_depths(
        config, evidence, pair_ids=authorized_ids
    )
    ratio_min, ratio_max = map(
        float, config["calibration"]["observed_predicted_ratio"]
    )
    slope_min, slope_max = map(float, config["calibration"]["calibration_slope"])
    reference_depth = float(calibration["reference_depth"])
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    calibrated_outputs = {}
    extensions = context_gate["outputs"]
    for output_id, extension in tqdm(
        extensions.items(),
        total=len(extensions),
        desc="Calibrate context extensions",
        unit="output",
    ):
        if not extension.get("topology_accepted"):
            calibrated_outputs[output_id] = {
                **extension,
                "accepted": False,
                "calibration_failure": "topology_not_accepted",
            }
            continue
        train_delta_frame = pd.read_parquet(
            extension["training_prediction_path"],
            columns=["pair_id", extension["prediction_column"]],
        ).sort_values("pair_id", kind="stable")
        validation_delta_frame = pd.read_parquet(
            extension["validation_prediction_path"],
            columns=["pair_id", extension["prediction_column"]],
        ).sort_values("pair_id", kind="stable")
        if not np.array_equal(
            train_delta_frame["pair_id"].to_numpy(np.int64),
            train_pairs["pair_id"].to_numpy(np.int64),
        ) or not np.array_equal(
            validation_delta_frame["pair_id"].to_numpy(np.int64),
            validation_pairs["pair_id"].to_numpy(np.int64),
        ):
            raise ValueError(f"Context delta does not align for {output_id}")
        train_delta = train_delta_frame[
            extension["prediction_column"]
        ].to_numpy(float)
        validation_delta = validation_delta_frame[
            extension["prediction_column"]
        ].to_numpy(float)
        counts = np.asarray(output_data[output_id]["counts"], np.float64)
        depth = float(output_data[output_id]["depth"])
        band_metrics = {}
        passed = True
        for band_index, band in enumerate(band_ids):
            train_mask = train_pairs["distance_band"].eq(band).to_numpy()
            validation_mask = validation_pairs["distance_band"].eq(band).to_numpy()
            train_ids = train_pairs.loc[
                train_mask, "pair_id"
            ].to_numpy(np.int64)
            validation_ids = validation_pairs.loc[
                validation_mask, "pair_id"
            ].to_numpy(np.int64)
            train_base = (
                math.log(depth / reference_depth)
                + np.log(
                    np.clip(
                        train_pairs.loc[
                            train_mask, "exposure"
                        ].to_numpy(float),
                        1e-12,
                        None,
                    )
                )
                + train_pairs.loc[
                    train_mask, "distance_offset"
                ].to_numpy(float)
            )
            validation_base = (
                math.log(depth / reference_depth)
                + np.log(
                    np.clip(
                        validation_pairs.loc[
                            validation_mask, "exposure"
                        ].to_numpy(float),
                        1e-12,
                        None,
                    )
                )
                + validation_pairs.loc[
                    validation_mask, "distance_offset"
                ].to_numpy(float)
            )
            train_shared = shared_train.loc[
                train_mask, "shared_residual_score"
            ].to_numpy(float)
            validation_shared = shared_validation.loc[
                validation_mask, "shared_residual_score"
            ].to_numpy(float)
            old_fit = calibration["parameters"][output_id]["bands"][band][
                "model"
            ]
            distance_fit = calibration["parameters"][output_id]["bands"][band][
                "baseline"
            ]
            context_fit = fit_nb2_offset(
                counts[train_ids],
                train_base + train_shared + train_delta[train_mask],
                fixed_dispersion=float(old_fit["dispersion"]),
                prior_mean=float(old_fit["alpha"]),
                prior_sd=float(
                    config["calibration"].get("intercept_shrinkage_sd", 2.0)
                ),
            )
            context_mean = np.exp(
                np.clip(
                    validation_base
                    + validation_shared
                    + validation_delta[validation_mask]
                    + context_fit["alpha"],
                    -40,
                    30,
                )
            )
            old_mean = np.exp(
                np.clip(
                    validation_base
                    + validation_shared
                    + float(old_fit["alpha"]),
                    -40,
                    30,
                )
            )
            distance_mean = np.exp(
                np.clip(
                    validation_base + float(distance_fit["alpha"]),
                    -40,
                    30,
                )
            )
            observed = counts[validation_ids]
            context_ll = nb2_logpmf(
                observed, context_mean, context_fit["dispersion"]
            )
            old_ll = nb2_logpmf(
                observed, old_mean, float(old_fit["dispersion"])
            )
            distance_ll = nb2_logpmf(
                observed,
                distance_mean,
                float(distance_fit["dispersion"]),
            )
            rows_shared = pd.DataFrame(
                {
                    "chrom": validation_pairs.loc[
                        validation_mask, "chrom"
                    ].astype(str).to_numpy(),
                    "events": observed,
                    "model_log_likelihood": context_ll,
                    "baseline_log_likelihood": old_ll,
                }
            )
            rows_distance = rows_shared.copy()
            rows_distance["baseline_log_likelihood"] = distance_ll
            bootstrap_shared = _chromosome_bootstrap_gain(
                rows_shared,
                replicates=replicates,
                confidence_level=confidence,
                seed=int(config["seed"]) + band_index,
            )
            bootstrap_distance = _chromosome_bootstrap_gain(
                rows_distance,
                replicates=replicates,
                confidence_level=confidence,
                seed=int(config["seed"]) + 1_000 + band_index,
            )
            ratio = float(observed.sum() / context_mean.sum())
            slope = calibration_slope(observed, context_mean)
            checks = {
                "gain_over_shared": bootstrap_shared["lower"] is not None
                and bootstrap_shared["lower"] > 0,
                "gain_over_distance": bootstrap_distance["lower"] is not None
                and bootstrap_distance["lower"] > 0,
                "ratio": ratio_min <= ratio <= ratio_max,
                "slope": slope is not None and slope_min <= slope <= slope_max,
            }
            passed &= all(checks.values())
            band_metrics[band] = {
                "fit": context_fit,
                "observed_predicted_ratio": ratio,
                "calibration_slope": slope,
                "gain_over_shared_topology": bootstrap_shared,
                "gain_over_distance_exposure": bootstrap_distance,
                "checks": checks,
            }
        calibrated_outputs[output_id] = {
            **extension,
            "accepted": bool(passed),
            "requires_rate_recalibration": False,
            "calibration": band_metrics,
            "calibration_failure": None if passed else "context_rate_gate_failed",
        }
        if passed:
            for band in band_ids:
                calibration["parameters"][output_id]["bands"][band]["model"] = (
                    band_metrics[band]["fit"]
                )
        calibration.setdefault("unit_accepted", {})[output_id] = bool(passed)
    gate_report = {
        **context_gate,
        "schema_version": 2,
        "uncalibrated_context_gate": {
            "path": str(context_gate_path),
            "sha256": sha256_file(context_gate_path),
        },
        "outputs": calibrated_outputs,
        "rank_scores_modified_by_calibration": False,
        "test_accessed": False,
    }
    calibration["schema_version"] = 2
    calibration["context_extension_gate"] = {
        "path": str(gate_output),
    }
    calibration["shared_calibration"] = {
        "path": str(shared_calibration_path),
        "sha256": sha256_file(shared_calibration_path),
    }
    calibration["context_extension_attempted"] = True
    calibration["test_accessed"] = False
    atomic_json(gate_output, gate_report)
    calibration["context_extension_gate"]["sha256"] = sha256_file(gate_output)
    atomic_json(calibration_output, calibration)
    return calibration, gate_report
