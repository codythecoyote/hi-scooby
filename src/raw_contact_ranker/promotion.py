from __future__ import annotations

from typing import Any

import numpy as np


def assess_epoch3_promotion(
    training: dict[str, Any],
    metrics: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    """Require early evidence that the full model is learning local recovery."""
    history = training.get("history", [])
    rank_losses = [
        float(row["validation_rank_loss"])
        for row in history
        if row.get("validation_rank_loss") is not None
    ]
    best_epoch = training.get("best_epoch")
    if (
        len(rank_losses) >= 2
        and rank_losses[0] > 0
        and isinstance(best_epoch, int)
        and 0 <= best_epoch < len(rank_losses)
    ):
        selected_rank = rank_losses[best_epoch]
        rank_improvement = (rank_losses[0] - selected_rank) / rank_losses[0]
    else:
        selected_rank = None
        rank_improvement = float("-inf")

    top1: dict[str, list[float]] = {"250-500": [], "500-995": []}
    exact: dict[str, dict[str, list[float]]] = {
        distance: {"pearson": [], "spearman": []}
        for distance in ("250000", "500000", "750000")
    }
    likelihoods: list[float] = []
    for context in metrics.get("contexts", []):
        likelihood = context.get("conditional_log_likelihood_per_event")
        if likelihood is not None and np.isfinite(likelihood):
            likelihoods.append(float(likelihood))
        for row in context.get("top_contact", []):
            if (
                int(row.get("match_tolerance_bins", -1)) == 1
                and np.isclose(float(row.get("top_fraction", -1)), 0.01)
                and row.get("band") in top1
                and row.get("enrichment_over_chance") is not None
            ):
                top1[str(row["band"])].append(
                    float(row["enrichment_over_chance"])
                )
        for distance, bundle in context.get("exact_distance", {}).items():
            if str(distance) not in exact or not bundle.get("defined"):
                continue
            for name in ("pearson", "spearman"):
                value = bundle.get(name)
                if value is not None and np.isfinite(value):
                    exact[str(distance)][name].append(float(value))

    top1_medians = {
        band: float(np.median(values)) if values else None
        for band, values in top1.items()
    }
    exact_medians = {}
    for distance, values_by_metric in exact.items():
        metric_medians = [
            float(np.median(values))
            for values in values_by_metric.values()
            if values
        ]
        exact_medians[distance] = max(metric_medians) if metric_medians else None
    baseline_values = [
        float(value)
        for value in baselines.get(
            "conditional_log_likelihood_per_event", {}
        ).values()
        if value is not None and np.isfinite(value)
    ]
    model_likelihood = float(np.mean(likelihoods)) if likelihoods else None
    likelihood_gain = (
        model_likelihood - max(baseline_values)
        if model_likelihood is not None and baseline_values
        else None
    )
    acceptance = metrics.get("acceptance_summary", {})
    top_bootstrap = acceptance.get(
        "top_contact_chromosome_bootstrap_gain_over_chance", {}
    )
    top1_bootstrap_lowers = {
        band: top_bootstrap.get(
            f"{band}:top0.01:tolerance1", {}
        ).get("lower")
        for band in top1
    }
    likelihood_bootstrap_lower = acceptance.get(
        "likelihood_chromosome_bootstrap_gain_per_event", {}
    ).get("lower")
    checks = {
        "three_epochs_completed": len(history) == 3,
        "validation_rank_loss_improved_two_percent": rank_improvement >= 0.02,
        "top1_250_500_above_chance": (
            top1_medians["250-500"] is not None
            and top1_medians["250-500"] > 1.0
        ),
        "top1_500_995_above_chance": (
            top1_medians["500-995"] is not None
            and top1_medians["500-995"] > 1.0
        ),
        "top1_chromosome_bootstrap_lowers_are_positive": all(
            value is not None and value > 0
            for value in top1_bootstrap_lowers.values()
        ),
        "all_exact_distance_correlations_show_early_signal": all(
            value is not None and value >= 0.06
            for value in exact_medians.values()
        ),
        "likelihood_still_improves_fixed_baseline": (
            likelihood_gain is not None
            and likelihood_gain > 0
            and likelihood_bootstrap_lower is not None
            and likelihood_bootstrap_lower > 0
        ),
        "learned_score_normalizer_audit_passed": bool(
            acceptance.get("learned_score_normalizer_audit_pass")
        ),
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "validation_rank_relative_improvement": 0.02,
            "top1_enrichment": 1.0,
            "exact_distance_best_correlation": 0.06,
            "likelihood_gain": 0.0,
        },
        "observed": {
            "first_validation_rank_loss": rank_losses[0] if rank_losses else None,
            "selected_checkpoint_epoch": best_epoch,
            "selected_checkpoint_validation_rank_loss": selected_rank,
            "validation_rank_relative_improvement": (
                rank_improvement if np.isfinite(rank_improvement) else None
            ),
            "top1_enrichment_medians": top1_medians,
            "top1_chromosome_bootstrap_lowers": top1_bootstrap_lowers,
            "exact_distance_best_correlation_medians": exact_medians,
            "model_likelihood": model_likelihood,
            "likelihood_gain": likelihood_gain,
            "likelihood_chromosome_bootstrap_lower": likelihood_bootstrap_lower,
        },
    }
