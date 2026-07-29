from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.special import logsumexp
from tqdm.auto import tqdm

from .common import atomic_json, enforce_or_warn, source_record, update_manifest
from .data import (
    ANCHOR_LABEL_COLUMNS,
    PairData,
    anchor_stratum_roles,
    model_from_checkpoint,
)
from .metrics import (
    build_top_contact_groups,
    chromosome_bootstrap,
    fixed_distance_oe_scores,
    grouped_top_contact_metrics,
    metric_bundle,
)


def _likelihood_groups(
    chrom: np.ndarray, distance_band: np.ndarray
) -> list[tuple[str, str, np.ndarray]]:
    frame = pd.DataFrame(
        {"chrom": np.asarray(chrom).astype(str), "band": np.asarray(distance_band).astype(str)}
    )
    return [
        (str(key[0]), str(key[1]), np.asarray(indices, np.int64))
        for key, indices in frame.groupby(["chrom", "band"], sort=False).indices.items()
    ]


def _conditional_likelihood(
    log_rate: np.ndarray,
    counts: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
) -> tuple[float, float, list[dict[str, Any]], np.ndarray]:
    total_ll = 0.0
    total_events = 0.0
    rows = []
    probability = np.empty(len(log_rate), np.float64)
    for chrom, band, indices in groups:
        eta = np.asarray(log_rate[indices], np.float64)
        y = np.asarray(counts[indices], np.float64)
        normalizer = float(logsumexp(eta))
        events = float(y.sum())
        likelihood = float(np.sum(y * eta) - events * normalizer)
        probability[indices] = np.exp(eta - normalizer)
        total_ll += likelihood
        total_events += events
        rows.append(
            {
                "chrom": chrom,
                "distance_band": band,
                "events": events,
                "conditional_log_likelihood": likelihood,
                "conditional_log_likelihood_per_event": (
                    likelihood / events if events > 0 else None
                ),
            }
        )
    return total_ll, total_events, rows, probability


def _sampled_normalizer_error(
    opportunity_scores: np.ndarray,
    sampled_scores: np.ndarray,
    proposal_probabilities: np.ndarray,
) -> float:
    opportunity_scores = np.asarray(opportunity_scores, np.float64)
    sampled_scores = np.asarray(sampled_scores, np.float64)
    proposal_probabilities = np.asarray(proposal_probabilities, np.float64)
    if (
        not len(opportunity_scores)
        or sampled_scores.shape != proposal_probabilities.shape
        or np.any(proposal_probabilities <= 0)
    ):
        raise ValueError("Invalid learned-normalizer audit inputs")
    exact = float(logsumexp(opportunity_scores))
    estimated = float(
        logsumexp(sampled_scores - np.log(proposal_probabilities))
        - np.log(len(sampled_scores))
    )
    return abs(estimated - exact)


def _summarize_anchor_strata(
    correlations: dict[str, dict[str, list[float]]],
    auprc_gains: dict[str, list[float]],
    required_labels: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, float | None], bool]:
    correlation_summary = {
        label: {
            metric: (float(np.median(values)) if values else None)
            for metric, values in by_metric.items()
        }
        for label, by_metric in correlations.items()
    }
    auprc_summary = {
        label: (float(np.median(values)) if values else None)
        for label, values in auprc_gains.items()
    }
    required = required_labels or tuple(correlation_summary)
    unknown = set(required) - set(correlation_summary)
    if not required or unknown:
        raise ValueError(f"Required anchor strata are empty or unknown: {sorted(unknown)}")
    passed = all(
        (
            correlation_summary[label].get("pearson") is not None
            and correlation_summary[label]["pearson"] > 0
        )
        or (
            correlation_summary[label].get("spearman") is not None
            and correlation_summary[label]["spearman"] > 0
        )
        for label in required
    ) and all(
        auprc_summary[label] is not None and auprc_summary[label] > 0
        for label in required
    )
    return correlation_summary, auprc_summary, bool(passed)


def build_evaluation_manifest(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["outputs"]["data_root"])
    pairs_path = output_root / "canonical_pairs.parquet"
    columns = [
        "pair_id", "chrom", "bin_i", "bin_j", "distance_bp", "distance_bin",
        "distance_band", "tile_id", "tile_row", "split", "anchor_class",
        "exposure", "distance_offset", *ANCHOR_LABEL_COLUMNS,
    ]
    pairs = pd.read_parquet(pairs_path, columns=columns)
    manifest = pairs.loc[pairs["split"].eq("validation")].copy()
    output = output_root / "evaluation_manifest.parquet"
    manifest.to_parquet(output, index=False, compression="zstd")
    report = {
        "schema_version": 2,
        "output": str(output),
        "candidate_pairs": len(manifest),
        "split": "validation",
        "likelihood_conditioning": "chromosome_distance_band",
        "top_fractions": config["evaluation"]["top_fractions"],
        "match_tolerances_bins": config["evaluation"]["match_tolerances_bins"],
        "exact_distances_bp": config["evaluation"]["exact_distances_bp"],
        "test_pairs_touched": 0,
        "source": source_record(pairs_path),
    }
    atomic_json(output_root / "evaluation_manifest_report.json", report)
    update_manifest(output_root, "evaluation_manifest", report)
    return report


def evaluate_baselines(
    config: dict[str, Any], manifest_path: Path, output: Path
) -> dict[str, Any]:
    manifest = pd.read_parquet(manifest_path)
    pair_ids = manifest["pair_id"].to_numpy(np.int64)
    fixed_log_rate = (
        np.log(np.clip(manifest["exposure"].to_numpy(float), 1e-12, None))
        + manifest["distance_offset"].to_numpy(float)
    )
    predictions = manifest[["pair_id"]].copy()
    predictions["fixed_offset_log_rate"] = fixed_log_rate.astype(np.float32)
    predictions["linear_score"] = np.nan
    predictions["film_score"] = np.nan
    prediction_path = manifest_path.parent / "baseline_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    evidence = zarr.open_group(
        str(manifest_path.parent / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    groups = _likelihood_groups(
        manifest["chrom"].to_numpy(), manifest["distance_band"].to_numpy()
    )
    values = []
    by_context = []
    for context in range(evidence["full_count"].shape[0]):
        counts = np.asarray(evidence["full_count"][context, pair_ids], np.float64)
        likelihood, events, _, _ = _conditional_likelihood(fixed_log_rate, counts, groups)
        per_event = likelihood / events if events > 0 else None
        by_context.append({"context_index": context, "value": per_event})
        if per_event is not None:
            values.append(per_event)
    warnings: list[dict[str, Any]] = []
    reference = Path(config["paths"]["reference_topk"])
    historical: dict[str, Any] | None = None
    if reference.exists():
        with reference.open() as handle:
            payload = json.load(handle)
        historical = {
            "source": str(reference),
            "model_topk_summary": payload.get("model_topk_summary_across_contexts", []),
        }
    else:
        enforce_or_warn(
            False, warnings, "BASELINE_REFERENCE_MISSING",
            f"Existing linear/FiLM top-contact result is absent: {reference}",
            strict=False,
        )
    report = {
        "schema_version": 2,
        "prediction_path": str(prediction_path),
        "candidate_pairs": len(predictions),
        "pair_level_baseline_likelihood_present": bool(values),
        "fixed_offset_predictions_finite": bool(np.isfinite(fixed_log_rate).all()),
        "historical_legacy_baselines": historical,
        "conditional_log_likelihood_per_event": {
            "fixed_offset": float(np.mean(values)) if values else None,
            "linear": None,
            "film": None,
        },
        "fixed_offset_by_context": by_context,
        "warnings": warnings,
    }
    atomic_json(output, report)
    return report


def _score_pairs(
    model,
    data: PairData,
    pair_ids: np.ndarray,
    device: torch.device,
    context_embeddings: torch.Tensor | None,
    context: int | None,
    batch_size: int = 16_384,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_rate, residual, deltas = [], [], []
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(pair_ids), batch_size),
            total=math.ceil(len(pair_ids) / batch_size),
            desc=progress_desc or "Score pairs",
            unit="batch", leave=False, mininterval=5.0,
        ):
            ids = pair_ids[start : start + batch_size]
            inputs = data.model_inputs(ids, device)
            if context_embeddings is not None and context is None:
                raise ValueError("A context index is required for context-conditioned scores")
            context_index = (
                torch.full((len(ids),), context, dtype=torch.long, device=device)
                if context_embeddings is not None else None
            )
            result = model(
                **inputs, context_embedding=context_embeddings, context_index=context_index
            )
            delta = result["context_delta"]
            log_rate.append(result["log_rate"].cpu().numpy())
            deltas.append(delta.cpu().numpy())
            residual.append((result["residual_score"] + delta).cpu().numpy())
    return np.concatenate(log_rate), np.concatenate(residual), np.concatenate(deltas)


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    *,
    output: Path,
    freeze_test: bool = False,
) -> dict[str, Any]:
    if split == "test" and not freeze_test:
        raise ValueError("Test evaluation requires --freeze-test")
    output_root = Path(config["outputs"]["data_root"])
    data = PairData(
        output_root / "canonical_pairs.parquet",
        output_root / "pair_features.zarr",
        preload_features=bool(config["model"].get("preload_features", True)),
        progress=True,
    )
    if data.preloaded_bytes:
        tqdm.write(f"RAM feature/covariate cache: {data.preloaded_bytes / 2**30:.2f} GiB")
    pair_ids = np.flatnonzero(data.pairs["split"].eq(split).to_numpy())
    rows = data.pairs.iloc[pair_ids].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, device)
    checkpoint_stage = str(checkpoint.get("stage", ""))
    context_embeddings = None
    if checkpoint_stage in {"rna", "full"}:
        contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
            "context_index", kind="stable"
        )
        centroids = pd.read_parquet(config["paths"]["centroids"])
        ordered_centroids = contexts[["cell_type", "context_index"]].merge(
            centroids[["cell_type", "embedding"]],
            on="cell_type", how="left", sort=False, validate="one_to_one",
        ).sort_values("context_index", kind="stable")
        if ordered_centroids["embedding"].isna().any():
            raise ValueError("One or more target contexts lack an RNA centroid")
        context_embeddings = torch.as_tensor(
            np.stack(
                ordered_centroids["embedding"].map(
                    lambda x: np.asarray(x, np.float32)
                )
            ),
            device=device,
        )
    elif checkpoint_stage != "sequence":
        raise ValueError(f"Unsupported checkpoint stage: {checkpoint_stage!r}")
    evidence = zarr.open_group(str(output_root / "pseudoreplicate_evidence.zarr"), mode="r")
    all_distances = data.pairs["distance_bin"].to_numpy(np.int64)
    all_exposure = data.pairs["exposure"].to_numpy(np.float64)
    training_pairs = data.pairs["split"].eq("train").to_numpy()
    fixed_log_rate = (
        np.log(np.clip(rows["exposure"].to_numpy(float), 1e-12, None))
        + rows["distance_offset"].to_numpy(float)
    )
    likelihood_groups = _likelihood_groups(
        rows["chrom"].to_numpy(), rows["distance_band"].to_numpy()
    )
    likelihood_group_indices = {
        (chrom, band): indices for chrom, band, indices in likelihood_groups
    }
    validation_controls = None
    local_pair_index = None
    normalizer_errors: list[float] = []
    if split == "validation":
        validation_controls = pd.read_parquet(
            output_root / "validation_sampled_control_groups.parquet",
            columns=[
                "positive_pair_id", "context_id", "event_control_pair_ids",
                "event_proposal_probabilities",
            ],
        )
        local_pair_index = np.full(len(data.pairs), -1, np.int64)
        local_pair_index[pair_ids] = np.arange(len(pair_ids), dtype=np.int64)
    top_groups = build_top_contact_groups(
        rows["tile_row"].to_numpy(), rows["distance_band"].to_numpy()
    )
    top_groups_by_chromosome: dict[str, list[tuple[str, int, np.ndarray]]] = {}
    chrom_values = rows["chrom"].astype(str).to_numpy()
    for group in top_groups:
        chromosome = str(chrom_values[group[2][0]])
        top_groups_by_chromosome.setdefault(chromosome, []).append(group)
    neighborhood_size_cache: dict[tuple[str, int, int], np.ndarray] = {}
    context_metrics = []
    prediction_frames = []
    bootstrap_rows = []
    top_bootstrap_rows = []
    anchor_values: dict[str, dict[str, list[float]]] = {
        name: {"pearson": [], "spearman": []}
        for name in ANCHOR_LABEL_COLUMNS
    }
    anchor_auprc_gains: dict[str, list[float]] = {
        name: [] for name in ANCHOR_LABEL_COLUMNS
    }
    minimum_candidates = int(config["evaluation"]["minimum_candidates"])
    minimum_supported = int(config["evaluation"]["minimum_supported_candidates"])
    required_anchor_strata, descriptive_anchor_strata = anchor_stratum_roles(config)
    tie_mode = str(config["evaluation"]["primary_tie_mode"])
    sequence_score_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for context in tqdm(
        range(evidence["full_count"].shape[0]),
        desc=f"Evaluate {split} contexts", unit="context",
    ):
        if sequence_score_cache is None or context_embeddings is not None:
            scores = _score_pairs(
                model, data, pair_ids, device, context_embeddings, context,
                progress_desc=(
                    f"Score {split} context {context + 1}/"
                    f"{evidence['full_count'].shape[0]}"
                ),
            )
            if context_embeddings is None:
                sequence_score_cache = scores
        else:
            scores = sequence_score_cache
        log_rate, residual, context_delta = scores
        full_count = np.asarray(evidence["full_count"][context], np.float64)
        exposure_corrected = full_count / np.clip(all_exposure, 1e-12, None)
        target_score = fixed_distance_oe_scores(
            exposure_corrected, exposure_corrected, all_distances, training_pairs
        )[pair_ids]
        target_count = full_count[pair_ids].astype(np.float32)
        support = np.asarray(evidence["support_weight"][context, pair_ids], np.float32) > 0
        bundle = metric_bundle(
            residual, target_score, support,
            minimum_candidates=minimum_candidates,
            minimum_supported_candidates=minimum_supported,
        )
        model_ll, total_events, model_chrom, probability = _conditional_likelihood(
            log_rate, target_count, likelihood_groups
        )
        if validation_controls is not None and local_pair_index is not None:
            audit_rows = validation_controls.loc[
                validation_controls["context_id"].eq(context)
            ].head(16)
            for audit_row in audit_rows.itertuples(index=False):
                positive_local = int(local_pair_index[int(audit_row.positive_pair_id)])
                control_local = local_pair_index[
                    np.asarray(audit_row.event_control_pair_ids, np.int64)
                ]
                if positive_local < 0 or np.any(control_local < 0):
                    raise RuntimeError(
                        "Validation normalizer controls fall outside validation manifest"
                    )
                key = (
                    str(rows.iloc[positive_local]["chrom"]),
                    str(rows.iloc[positive_local]["distance_band"]),
                )
                population = likelihood_group_indices[key]
                opportunities = population[population != positive_local]
                normalizer_errors.append(
                    _sampled_normalizer_error(
                        log_rate[opportunities],
                        log_rate[control_local],
                        np.asarray(
                            audit_row.event_proposal_probabilities, np.float64
                        ),
                    )
                )
        baseline_ll, _, baseline_chrom, _ = _conditional_likelihood(
            fixed_log_rate, target_count, likelihood_groups
        )
        for model_row, baseline_row in zip(model_chrom, baseline_chrom, strict=True):
            events = model_row["events"]
            gain = model_row["conditional_log_likelihood"] - baseline_row["conditional_log_likelihood"]
            bootstrap_rows.append(
                {
                    "chrom": model_row["chrom"],
                    "context_index": context,
                    "distance_band": model_row["distance_band"],
                    "gain_per_event": gain / events if events > 0 else None,
                }
            )
        top_rows = grouped_top_contact_metrics(
            rows["chrom"].to_numpy(), rows["bin_i"].to_numpy(), rows["bin_j"].to_numpy(),
            rows["tile_row"].to_numpy(), rows["distance_band"].to_numpy(),
            residual, target_score,
            fractions=tuple(float(v) for v in config["evaluation"]["top_fractions"]),
            tolerances=tuple(int(v) for v in config["evaluation"]["match_tolerances_bins"]),
            neighborhood_size_cache=neighborhood_size_cache,
            groups=top_groups,
            progress_desc=f"Top contacts {split} context {context + 1}",
            tie_mode=tie_mode,
        )
        hard_rows = grouped_top_contact_metrics(
            rows["chrom"].to_numpy(), rows["bin_i"].to_numpy(), rows["bin_j"].to_numpy(),
            rows["tile_row"].to_numpy(), rows["distance_band"].to_numpy(),
            residual, target_score,
            fractions=tuple(float(v) for v in config["evaluation"]["top_fractions"]),
            tolerances=tuple(int(v) for v in config["evaluation"]["match_tolerances_bins"]),
            neighborhood_size_cache=neighborhood_size_cache,
            groups=top_groups,
            tie_mode="hard_cutoff",
        )
        chromosome_top_rows = []
        for chromosome, chromosome_groups in top_groups_by_chromosome.items():
            selected_rows = grouped_top_contact_metrics(
                rows["chrom"].to_numpy(), rows["bin_i"].to_numpy(),
                rows["bin_j"].to_numpy(), rows["tile_row"].to_numpy(),
                rows["distance_band"].to_numpy(), residual, target_score,
                fractions=tuple(
                    float(v) for v in config["evaluation"]["top_fractions"]
                ),
                tolerances=tuple(
                    int(v) for v in config["evaluation"]["match_tolerances_bins"]
                ),
                neighborhood_size_cache=neighborhood_size_cache,
                groups=chromosome_groups,
                tie_mode=tie_mode,
            )
            for row in selected_rows:
                row["chrom"] = chromosome
                row["context_index"] = context
                row["gain_over_chance"] = row["enrichment_over_chance"] - 1.0
                top_bootstrap_rows.append(row.copy())
            chromosome_top_rows.extend(selected_rows)
        for collection in (top_rows, hard_rows):
            for row in collection:
                model_tied = int(row.pop("score_a_cutoff_tied_tiles"))
                target_tied = int(row.pop("score_b_cutoff_tied_tiles"))
                tile_count = int(row["candidate_tile_count"])
                row["model_cutoff_tied"] = bool(model_tied)
                row["target_cutoff_tied"] = bool(target_tied)
                row["model_cutoff_tied_tiles"] = model_tied
                row["target_cutoff_tied_tiles"] = target_tied
                row["model_cutoff_tie_rate"] = model_tied / max(tile_count, 1)
                row["target_cutoff_tie_rate"] = target_tied / max(tile_count, 1)
        exact = {}
        for distance in config["evaluation"]["exact_distances_bp"]:
            mask = rows["distance_bp"].eq(int(distance)).to_numpy()
            exact[str(distance)] = metric_bundle(
                residual[mask], target_score[mask], support[mask],
                minimum_candidates=minimum_candidates,
                minimum_supported_candidates=minimum_supported,
            )
        anchor = {}
        for label in ANCHOR_LABEL_COLUMNS:
            mask = rows[label].to_numpy(bool)
            anchor[label] = metric_bundle(
                residual[mask], target_score[mask], support[mask],
                minimum_candidates=minimum_candidates,
                minimum_supported_candidates=minimum_supported,
            )
            candidate = anchor[label]
            if candidate.get("defined"):
                for metric in ("pearson", "spearman"):
                    value = candidate.get(metric)
                    if value is not None and np.isfinite(value):
                        anchor_values[label][metric].append(float(value))
                if candidate.get("auprc_defined"):
                    anchor_auprc_gains[label].append(
                        float(candidate["auprc"] - candidate["support_prevalence"])
                    )
        context_metrics.append(
            {
                "context_index": context,
                "metrics": bundle,
                "conditional_log_likelihood": model_ll,
                "conditional_log_likelihood_per_event": model_ll / max(total_events, 1.0),
                "baseline_conditional_log_likelihood_per_event": baseline_ll / max(total_events, 1.0),
                "conditional_likelihood_by_chromosome": model_chrom,
                "exact_distance": exact,
                "anchor_strata": anchor,
                "top_contact": top_rows,
                "top_contact_by_chromosome": chromosome_top_rows,
                "top_contact_hard_cutoff_sensitivity": hard_rows,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "context_index": context,
                    "log_rate": log_rate.astype(np.float32),
                    "fixed_offset_log_rate": fixed_log_rate.astype(np.float32),
                    "residual_score": residual.astype(np.float32),
                    "context_delta": context_delta.astype(np.float32),
                    "probability_within_chromosome_band": probability.astype(np.float32),
                }
            )
        )
    likelihood_bootstrap = chromosome_bootstrap(
        bootstrap_rows, "gain_per_event",
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    top_bootstrap: dict[str, dict[str, float | None]] = {}
    top_bootstrap_passes = []
    for band in ("250-500", "500-995"):
        for fraction in (float(v) for v in config["evaluation"]["top_fractions"]):
            selected_rows = [
                row for row in top_bootstrap_rows
                if row["band"] == band
                and row["top_fraction"] == fraction
                and row["match_tolerance_bins"] == 1
            ]
            interval = chromosome_bootstrap(
                selected_rows,
                "gain_over_chance",
                replicates=int(config["evaluation"]["bootstrap_replicates"]),
                seed=int(config["seed"]) + int(fraction * 10_000),
            )
            key = f"{band}:top{fraction:g}:tolerance1"
            top_bootstrap[key] = interval
            top_bootstrap_passes.append(
                interval["lower"] is not None and interval["lower"] > 0
            )
    anchor_summary, anchor_auprc_summary, anchor_pass = _summarize_anchor_strata(
        anchor_values, anchor_auprc_gains, required_anchor_strata
    )
    tolerance = float(config["sampling"]["exact_normalizer_tolerance"])
    learned_normalizer_audit = {
        "events": len(normalizer_errors),
        "median_absolute_log_error": (
            float(np.median(normalizer_errors)) if normalizer_errors else None
        ),
        "p95_absolute_log_error": (
            float(np.quantile(normalizer_errors, 0.95))
            if normalizer_errors else None
        ),
        "tolerance": tolerance,
        "passed": bool(
            normalizer_errors and np.median(normalizer_errors) <= tolerance
        ),
    }
    evidence_complete = bool(
        likelihood_bootstrap["lower"] is not None
        and top_bootstrap
        and all(interval["lower"] is not None for interval in top_bootstrap.values())
        and all(
            any(value is not None for value in anchor_summary[label].values())
            for label in required_anchor_strata
        )
        and all(
            anchor_auprc_summary[label] is not None
            for label in required_anchor_strata
        )
        and learned_normalizer_audit["events"] > 0
    )
    acceptance_summary = {
        "complete": evidence_complete,
        "likelihood_conditioning": "chromosome_distance_band",
        "pair_level_baseline_likelihood_present": True,
        "likelihood_chromosome_bootstrap_gain_per_event": likelihood_bootstrap,
        "top_contact_chromosome_bootstrap_gain_over_chance": top_bootstrap,
        "chromosome_bootstrap_lower_bound_pass": bool(
            top_bootstrap_passes and all(top_bootstrap_passes)
        ),
        "likelihood_chromosome_bootstrap_lower_bound_pass": bool(
            likelihood_bootstrap["lower"] is not None
            and likelihood_bootstrap["lower"] > 0
        ),
        "anchor_stratified_median_correlation": anchor_summary,
        "anchor_stratified_median_auprc_gain_over_prevalence": anchor_auprc_summary,
        "required_anchor_strata": list(required_anchor_strata),
        "descriptive_anchor_strata": list(
            descriptive_anchor_strata
        ),
        "anchor_stratified_gain_pass": anchor_pass,
        "learned_score_normalizer_audit": learned_normalizer_audit,
        "learned_score_normalizer_audit_pass": learned_normalizer_audit["passed"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.parent / f"{split}_topk_predictions.parquet"
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        predictions_path, index=False, compression="zstd"
    )
    undefined = []
    for row in context_metrics:
        for distance, metric in row["exact_distance"].items():
            if not metric.get("defined"):
                undefined.append(
                    {"context_index": row["context_index"], "distance": distance, "metric": "bundle"}
                )
            elif not metric.get("auprc_defined", False):
                undefined.append(
                    {"context_index": row["context_index"], "distance": distance, "metric": "auprc"}
                )
    report = {
        "schema_version": 2,
        "claim": "scHiCAR_contact_propensity",
        "checkpoint": source_record(checkpoint_path),
        "split": split,
        "test_frozen": bool(split == "test" and freeze_test),
        "candidate_pairs": len(pair_ids),
        "contexts": context_metrics,
        "predictions": str(predictions_path),
        "undefined_metrics": undefined,
        "learned_score_normalizer_audit": learned_normalizer_audit,
        "acceptance_summary": acceptance_summary,
    }
    atomic_json(output, report)
    if split == "test":
        results_root = Path(config["outputs"]["results_root"])
        results_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, results_root / "final_model.pt")
        shutil.copy2(predictions_path, results_root / "topk_predictions.parquet")
        pd.DataFrame(
            [
                {
                    "metric": "conditional_likelihood_gain_per_event",
                    "lower_95": likelihood_bootstrap["lower"],
                    "estimate": likelihood_bootstrap["median"],
                    "upper_95": likelihood_bootstrap["upper"],
                    "defined": likelihood_bootstrap["lower"] is not None,
                }
            ]
        ).to_parquet(
            results_root / "bootstrap_intervals.parquet", index=False, compression="zstd"
        )
        (results_root / "figures").mkdir(exist_ok=True)
        _write_test_report(results_root / "test_report.md", report)
    return report


def _write_test_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Raw-contact residual ranker test report", "",
        "Output interpretation: `scHiCAR_contact_propensity`; this is neither an individual-cell conformation nor a causal enhancer–promoter call.",
        "", f"- Checkpoint SHA-256: `{report['checkpoint']['sha256']}`",
        f"- Candidate pairs: {report['candidate_pairs']:,}",
        f"- Test contract frozen: {report['test_frozen']}",
        f"- Undefined metric entries: {len(report['undefined_metrics'])}",
        "", "Full likelihood, exact-distance, top-contact, tie sensitivity, context, and anchor metadata are stored in `test_metrics.json`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def evaluate_rna_gate(
    config: dict[str, Any],
    sequence_checkpoint: Path,
    rna_checkpoint: Path,
    split: str,
    output: Path,
) -> dict[str, Any]:
    sequence = evaluate_checkpoint(
        config, sequence_checkpoint, split, output=output.parent / "sequence_comparison.json"
    )
    rna = evaluate_checkpoint(
        config, rna_checkpoint, split, output=output.parent / "rna_comparison.json"
    )
    seq_values = np.asarray(
        [row["conditional_log_likelihood_per_event"] for row in sequence["contexts"]]
    )
    rna_values = np.asarray(
        [row["conditional_log_likelihood_per_event"] for row in rna["contexts"]]
    )
    likelihood_gain = float(np.mean(rna_values - seq_values))
    report = {
        "sequence_checkpoint": str(sequence_checkpoint),
        "rna_checkpoint": str(rna_checkpoint),
        "split": split,
        "conditional_likelihood_gain_per_event": likelihood_gain,
        "ceiling_normalized_recovery_relative_gain": None,
        "paired_chromosome_bootstrap_ci": None,
        "context_swap_accuracy_gain": None,
        "rna_permutation_gain": None,
        "rna_supported": False,
        "production_checkpoint": str(sequence_checkpoint),
        "warnings": [
            {
                "code": "RNA_GATE_INCOMPLETE",
                "message": "RNA is conservatively rejected until top-contact, paired-bootstrap, context-swap, and permutation gates are jointly established.",
                "project_breaking": False,
            }
        ],
    }
    atomic_json(output, report)
    return report
