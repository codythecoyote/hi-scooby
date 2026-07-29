from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from .common import atomic_json


FEATURE_ORDER = ("exposure_only", "annotations", "alphagenome", "combined")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open() as handle:
        return json.load(handle)


def select_shared_feature_set(
    config: dict[str, Any],
    metric_paths: Iterable[Path],
    *,
    output: Path,
) -> dict[str, Any]:
    """Apply the prespecified five-fold, one-standard-error selection rule."""
    folds = int(config["model"]["chromosome_folds"])
    by_feature: dict[str, dict[int, dict[str, Any]]] = {}
    baseline_by_fold: dict[int, float] = {}
    for path in metric_paths:
        report = _read_json(Path(path))
        feature = str(report["feature_set"])
        fold = int(report["fold"])
        if fold < 0 or fold >= folds:
            raise ValueError(f"Invalid chromosome fold in {path}: {fold}")
        history = report.get("history", [])
        if not history:
            raise ValueError(f"No epoch history in {path}")
        best = min(
            history,
            key=lambda row: float(
                row["validation_negative_log_likelihood_per_event"]
            ),
        )
        loss = float(best["validation_negative_log_likelihood_per_event"])
        gain = best.get("validation_gain_per_event")
        if gain is None or not np.isfinite(float(gain)):
            raise ValueError(f"Missing validation gain in {path}")
        baseline_loss = loss + float(gain)
        if fold in baseline_by_fold and not math.isclose(
            baseline_by_fold[fold], baseline_loss, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError("Exposure-only baseline differs across feature fits")
        baseline_by_fold[fold] = baseline_loss
        feature_rows = by_feature.setdefault(feature, {})
        if fold in feature_rows:
            raise ValueError(f"Duplicate feature/fold metrics: {feature}/{fold}")
        feature_rows[fold] = {
            "loss": loss,
            "best_epoch_count": int(best["epoch"]) + 1,
            "path": str(Path(path).resolve()),
        }
    expected_model_features = set(FEATURE_ORDER[1:])
    if set(by_feature) != expected_model_features:
        raise ValueError(
            "Feature ladder is incomplete: "
            f"expected={sorted(expected_model_features)}, "
            f"actual={sorted(by_feature)}"
        )
    expected_folds = set(range(folds))
    if set(baseline_by_fold) != expected_folds or any(
        set(rows) != expected_folds for rows in by_feature.values()
    ):
        raise ValueError("Every feature candidate must have every chromosome fold")

    candidates: dict[str, dict[str, Any]] = {}
    baseline_losses = np.asarray(
        [baseline_by_fold[fold] for fold in range(folds)], np.float64
    )
    candidates["exposure_only"] = {
        "fold_losses": baseline_losses.tolist(),
        "mean_loss": float(baseline_losses.mean()),
        "standard_error": float(
            baseline_losses.std(ddof=1) / math.sqrt(folds)
        ),
        "epoch_counts": [],
    }
    for feature in FEATURE_ORDER[1:]:
        rows = by_feature[feature]
        losses = np.asarray(
            [rows[fold]["loss"] for fold in range(folds)], np.float64
        )
        candidates[feature] = {
            "fold_losses": losses.tolist(),
            "mean_loss": float(losses.mean()),
            "standard_error": float(losses.std(ddof=1) / math.sqrt(folds)),
            "epoch_counts": [
                int(rows[fold]["best_epoch_count"]) for fold in range(folds)
            ],
        }
    best_feature = min(
        FEATURE_ORDER, key=lambda feature: candidates[feature]["mean_loss"]
    )
    threshold = (
        float(candidates[best_feature]["mean_loss"])
        + float(candidates[best_feature]["standard_error"])
    )
    selected = next(
        feature
        for feature in FEATURE_ORDER
        if float(candidates[feature]["mean_loss"]) <= threshold
    )
    epoch_count = (
        int(round(median(candidates[selected]["epoch_counts"])))
        if selected != "exposure_only"
        else None
    )
    report = {
        "schema_version": 1,
        "folds": folds,
        "selection_rule": "simplest_within_one_standard_error_of_best",
        "feature_order": list(FEATURE_ORDER),
        "candidates": candidates,
        "best_feature_set": best_feature,
        "one_standard_error_threshold": threshold,
        "selected_feature_set": selected,
        "selected_epoch_count": epoch_count,
        "topology_training_authorized": selected != "exposure_only",
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report
