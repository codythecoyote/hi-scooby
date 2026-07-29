#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from _bootstrap import default_config
from raw_contact_ranker.common import atomic_json, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--data-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with args.metrics.open() as handle:
        metrics = json.load(handle)
    with args.baselines.open() as handle:
        baselines = json.load(handle)
    with args.data_validation.open() as handle:
        data_validation = json.load(handle)
    enrichments: dict[tuple[str, float], list[float]] = {}
    correlations: dict[tuple[str, str], list[float]] = {}
    auprc_gains: dict[str, list[float]] = {}
    for context in metrics["contexts"]:
        for row in context["top_contact"]:
            if row["match_tolerance_bins"] == 1:
                value = row["enrichment_over_chance"]
                if value is not None:
                    enrichments.setdefault((row["band"], row["top_fraction"]), []).append(value)
        for distance, bundle in context["exact_distance"].items():
            if bundle.get("defined"):
                for name in ("pearson", "spearman"):
                    value = bundle.get(name)
                    if value is not None and np.isfinite(value):
                        correlations.setdefault((distance, name), []).append(float(value))
                if bundle.get("auprc_defined"):
                    auprc_gains.setdefault(distance, []).append(
                        float(bundle["auprc"] - bundle["support_prevalence"])
                    )
    model_likelihood = float(np.mean([
        row["conditional_log_likelihood_per_event"] for row in metrics["contexts"]
    ]))
    baseline_likelihoods = baselines.get("conditional_log_likelihood_per_event", {})
    baseline_values = [value for value in baseline_likelihoods.values() if value is not None]
    acceptance = metrics.get("acceptance_summary", {})
    baseline_present = bool(baselines.get("pair_level_baseline_likelihood_present"))
    bootstrap_pass = acceptance.get("chromosome_bootstrap_lower_bound_pass")
    anchor_pass = acceptance.get("anchor_stratified_gain_pass")
    fractions = tuple(float(value) for value in config["evaluation"]["top_fractions"])
    ceiling = data_validation.get("canonical_all_split_ceiling", {})
    top1_thresholds = {
        band: (
            0.20 * float(ceiling[f"{band}:top0.01"])
            if ceiling.get(f"{band}:top0.01") is not None else np.inf
        )
        for band in ("250-500", "500-995")
    }
    checks = {
        "top1_250_500_recovery": np.median(enrichments.get(("250-500", 0.01), [-np.inf])) >= top1_thresholds["250-500"],
        "top1_500_995_recovery": np.median(enrichments.get(("500-995", 0.01), [-np.inf])) >= top1_thresholds["500-995"],
        "all_top_fractions_above_chance": all(
            np.median(enrichments.get((band, fraction), [-np.inf])) > 1.0
            for band in ("250-500", "500-995")
            for fraction in fractions
        ),
        "exact_distance_correlation": all(
            (
                np.median(correlations.get((str(distance), "pearson"), [-np.inf])) >= 0.20
                or np.median(correlations.get((str(distance), "spearman"), [-np.inf])) >= 0.20
            )
            for distance in (250000, 500000, 750000)
        ),
        "exact_distance_auprc_gain": all(
            np.median(auprc_gains.get(str(distance), [-np.inf])) > 0
            for distance in (250000, 500000, 750000)
        ),
        "baseline_likelihood_improvement": baseline_present and bool(baseline_values) and model_likelihood > max(baseline_values),
        "chromosome_bootstrap_lower_bound": bool(bootstrap_pass),
        "likelihood_chromosome_bootstrap_lower_bound": bool(
            acceptance.get("likelihood_chromosome_bootstrap_lower_bound_pass")
        ),
        "anchor_stratified_gain": bool(anchor_pass),
        "learned_score_normalizer_audit": bool(
            acceptance.get("learned_score_normalizer_audit_pass")
        ),
        "acceptance_evidence_complete": bool(acceptance.get("complete")),
    }
    accepted = all(checks.values())
    report = {
        "accepted": accepted,
        "checks": {key: bool(value) for key, value in checks.items()},
        "acceptance_summary_present": bool(acceptance),
        "pair_level_baseline_likelihood_present": baseline_present,
        "canonical_top1_recovery_thresholds": top1_thresholds,
        "warning": (
            None
            if accepted
            else "Validation acceptance failed; test evaluation must not proceed."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(f"Model gate accepted: {accepted}")
    if not accepted:
        sys.exit(2)


if __name__ == "__main__":
    main()
