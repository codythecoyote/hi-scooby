#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from _bootstrap import default_config  # importing also inserts src/ into sys.path

from raw_contact_ranker.common import atomic_json, load_config
from raw_contact_ranker.data import ANCHOR_LABEL_COLUMNS, anchor_stratum_roles
from raw_contact_ranker.metrics import supported_context_coverage
from raw_contact_ranker.provenance import verify_preparation_contract


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required preparation report is missing: {path}")
    with path.open() as handle:
        return json.load(handle)


def _finite_above(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and bool(
        np.isfinite(value) and float(value) > threshold
    )


def verify(config: dict, output: Path) -> dict:
    data_root = Path(config["outputs"]["data_root"])
    results_root = Path(config["outputs"]["results_root"])
    pairs_path = data_root / "canonical_pairs.parquet"
    contract = verify_preparation_contract(
        config,
        data_root / "preparation_contract.json",
        verify_sources=False,
    )
    pairs = pq.ParquetFile(pairs_path)
    required_columns = {
        "pair_id", "chrom", "bin_i", "bin_j", "distance_band", "split",
        "exposure", "distance_offset", "exposure_clipped", *ANCHOR_LABEL_COLUMNS,
    }
    missing = required_columns - set(pairs.schema_arrow.names)
    if missing:
        raise ValueError(f"Canonical pair schema lacks v2 fields: {sorted(missing)}")
    pair_count = pairs.metadata.num_rows
    pair_export = _json(data_root / "pair_export_report.json")
    annotation = _json(data_root / "annotation_report.json")
    evidence_report = _json(data_root / "evidence_export_report.json")
    data_validation = _json(results_root / "data_validation.json")
    offset = _json(data_root / "distance_offset_report.json")
    sampling = _json(data_root / "sampling_report.json")
    train_controls = pq.ParquetFile(data_root / "sampled_control_groups.parquet")
    validation_controls = pq.ParquetFile(
        data_root / "validation_sampled_control_groups.parquet"
    )
    baseline = _json(results_root / "baselines.json")
    manifest = _json(data_root / "evaluation_manifest_report.json")
    feature = _json(data_root / "feature_extraction_report.json")
    feature_store_path = data_root / "pair_features.zarr"
    if not feature_store_path.is_dir():
        raise FileNotFoundError(f"Required feature store is missing: {feature_store_path}")
    feature_store = zarr.open_group(str(feature_store_path), mode="r")
    evidence = zarr.open_group(str(data_root / "pseudoreplicate_evidence.zarr"), mode="r")
    multi_labels = annotation.get("multi_label_counts", {})
    annotation_sources = annotation.get("sources", {})
    contract_sources = contract.get("sources", {})
    ctcf_fraction = float(multi_labels.get("anchor_has_ctcf", 0)) / max(pair_count, 1)
    validation_label_frame = pq.read_table(
        pairs_path,
        columns=["pair_id", "split", *ANCHOR_LABEL_COLUMNS],
    ).to_pandas()
    validation_label_frame = validation_label_frame.loc[
        validation_label_frame["split"].eq("validation")
    ]
    validation_pair_ids = validation_label_frame["pair_id"].to_numpy(np.int64)
    validation_label_masks = {
        label: validation_label_frame[label].to_numpy(bool)
        for label in ANCHOR_LABEL_COLUMNS
    }
    validation_label_counts = {
        label: int(mask.sum()) for label, mask in validation_label_masks.items()
    }
    required_anchor_strata, descriptive_anchor_strata = anchor_stratum_roles(config)
    supported_by_context_and_label = {
        label: [] for label in ANCHOR_LABEL_COLUMNS
    }
    for context in range(evidence["support_weight"].shape[0]):
        supported = (
            np.asarray(
                evidence["support_weight"][context, validation_pair_ids],
                np.float32,
            )
            > 0
        )
        for label, mask in validation_label_masks.items():
            supported_by_context_and_label[label].append(
                int(np.sum(supported & mask))
            )
    minimum_supported = int(config["evaluation"]["minimum_supported_candidates"])
    minimum_context_fraction = float(
        config["evaluation"]["minimum_supported_context_fraction"]
    )
    supported_context_counts, required_supported_contexts = (
        supported_context_coverage(
            supported_by_context_and_label,
            minimum_supported_candidates=minimum_supported,
            minimum_context_fraction=minimum_context_fraction,
        )
    )
    ceiling = data_validation.get("canonical_all_split_ceiling", {})
    ceiling_rows = evidence_report.get("canonical_ceiling", {}).get(
        "context_split_rows", {}
    )
    ceiling_keys = {
        f"{band}:top{fraction:g}"
        for band in ("250-500", "500-995")
        for fraction in (0.01, 0.02)
    }
    checks = {
        "schema_version_2": int(config.get("schema_version", 0)) == 2,
        "provenance_contract_verified": int(contract.get("schema_version", 0)) == 2,
        "strict_stop_conditions_enabled": bool(config.get("strict_stop_conditions")),
        "positive_pair_count": pair_count > 0,
        "cross_split_anchor_overlap_absent": int(
            pair_export.get("cross_split_anchor_overlap_after_purge", -1)
        ) == 0,
        "annotation_pair_count_matches": int(annotation.get("pair_count", -1)) == pair_count,
        "contact_counts_not_used_for_exposure": annotation.get("contact_counts_consulted") is False,
        "cooler_weights_not_used_for_exposure": annotation.get("cooler_balance_weights_consulted") is False,
        "ctcf_threshold_matches_config": float(
            annotation.get("ctcf_score_threshold", np.nan)
        ) == float(config["annotations"]["ctcf_score_threshold"]),
        "annotation_static_sources_match_contract": all(
            annotation_sources.get(key) == contract_sources.get(key)
            for key in ("gene_annotation", "ccre_registry", "fasta", "mappability")
        ),
        "ctcf_calls_not_collapsed": 0.001 < ctcf_fraction < 0.5,
        "promoter_stratum_nonempty": int(multi_labels.get("anchor_has_promoter", 0)) > 0,
        "ccre_stratum_nonempty": int(multi_labels.get("anchor_has_ccre", 0)) > 0,
        "all_validation_anchor_strata_have_enough_candidates": all(
            validation_label_counts[label]
            >= int(config["evaluation"]["minimum_candidates"])
            for label in required_anchor_strata
        ),
        "all_validation_anchor_strata_have_context_support": all(
            supported_context_counts[label] >= required_supported_contexts
            for label in required_anchor_strata
        ),
        "evidence_pair_count_matches": int(evidence_report.get("pair_count", -1)) == pair_count,
        "evidence_shape_matches": evidence["full_count"].shape[1] == pair_count,
        "evidence_schema_version_2": int(evidence.attrs.get("schema_version", 0)) == 2,
        "count_conservation": bool(evidence_report.get("count_conservation")),
        "membership_count_conservation": bool(
            evidence_report.get("membership_count_conservation")
        ),
        "cooler_pixels_not_used_as_targets": evidence_report.get(
            "cooler_pixels_consulted_for_targets"
        ) is False,
        "cooler_candidate_count_conservation": bool(
            evidence_report.get("cooler_candidate_count_conservation")
        ),
        "cooler_weights_not_used_in_evidence": evidence_report.get(
            "cooler_balance_weights_consulted"
        ) is False,
        "data_validation_count_conservation": bool(
            data_validation.get("count_conservation")
        ),
        "stream_to_matrix_conservation": bool(
            evidence_report.get("stream_to_matrix_conservation")
        ),
        "distance_offset_monotone": bool(offset.get("monotone_nonincreasing")),
        "sampling_full_support": bool(sampling.get("full_support_guaranteed")),
        "training_controls_nonempty": train_controls.metadata.num_rows > 0,
        "validation_controls_nonempty": validation_controls.metadata.num_rows > 0,
        "training_control_count_matches_report": train_controls.metadata.num_rows
        == int(sampling.get("events", -1)),
        "validation_control_count_matches_report": validation_controls.metadata.num_rows
        == int(sampling.get("validation_events", -1)),
        "normalizer_audit_passed": bool(sampling.get("normalizer_audit", {}).get("passed")),
        "canonical_ceiling_uses_all_splits": int(
            data_validation.get("canonical_ceiling_split_count", 0)
        ) == int(config["pseudoreplicate_splits"]),
        "canonical_ceiling_complete_and_above_chance": all(
            _finite_above(ceiling.get(key), 1.0)
            for key in ceiling_keys
        ),
        "canonical_ceiling_has_every_context_split": all(
            int(ceiling_rows.get(key, -1))
            == int(config["pseudoreplicate_splits"])
            * int(evidence["full_count"].shape[0])
            for key in ceiling_keys
        ),
        "pair_level_baseline_present": bool(baseline.get("pair_level_baseline_likelihood_present")),
        "fixed_baseline_finite": np.isfinite(
            baseline.get("conditional_log_likelihood_per_event", {}).get("fixed_offset", np.nan)
        ),
        "validation_manifest_nonempty": int(manifest.get("candidate_pairs", 0)) > 0,
        "test_pairs_untouched": int(manifest.get("test_pairs_touched", -1)) == 0,
        "feature_pair_count_matches": int(feature.get("pair_count", -1)) == pair_count,
        "feature_store_shape_matches": all(
            name in feature_store
            and feature_store[name].shape
            == (pair_count, int(config["features"]["pair_channels"]))
            for name in (
                "pair_embedding", "anchor_i_embedding", "anchor_j_embedding"
            )
        ),
    }
    report = {
        "schema_version": 2,
        "prepared": bool(all(checks.values())),
        "pair_count": pair_count,
        "contexts": int(evidence["full_count"].shape[0]),
        "checks": {name: bool(value) for name, value in checks.items()},
        "normalizer_audit": sampling.get("normalizer_audit"),
        "annotation_summary": {
            "ctcf_pair_fraction": ctcf_fraction,
            "multi_label_counts": multi_labels,
            "required_anchor_strata": list(required_anchor_strata),
            "descriptive_anchor_strata": list(
                descriptive_anchor_strata
            ),
            "validation_multi_label_counts": validation_label_counts,
            "minimum_supported_validation_pairs_by_label_across_contexts": {
                label: int(min(values))
                for label, values in supported_by_context_and_label.items()
            },
            "median_supported_validation_pairs_by_label_across_contexts": {
                label: float(np.median(values))
                for label, values in supported_by_context_and_label.items()
            },
            "contexts_meeting_minimum_support_by_label": supported_context_counts,
            "required_supported_contexts": required_supported_contexts,
            "minimum_supported_context_fraction": minimum_context_fraction,
        },
        "paths": {"data_root": str(data_root), "results_root": str(results_root)},
    }
    atomic_json(output, report)
    if not report["prepared"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Preparation verification failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or Path(config["outputs"]["results_root"]) / "preparation_verification.json"
    report = verify(config, output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
