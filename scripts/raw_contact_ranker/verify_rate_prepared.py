#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr

from _bootstrap import default_config
from raw_contact_ranker.common import (
    atomic_json,
    distance_range_bp,
    load_config,
    resolution_contract,
)
from raw_contact_ranker.provenance import verify_preparation_contract


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required preparation artifact is missing: {path}")
    with path.open() as handle:
        return json.load(handle)


def verify(config: dict, output: Path) -> dict:
    data_root = Path(config["outputs"]["data_root"])
    results_root = Path(config["outputs"]["results_root"])
    contract = verify_preparation_contract(
        config,
        data_root / "preparation_contract.json",
        verify_sources=False,
    )
    resolution_report = _read(
        Path(config["paths"]["tiles"]).parent
        / "resolution_input_report.json"
    )
    pair_report = _read(data_root / "pair_export_report.json")
    annotation = _read(data_root / "annotation_report.json")
    evidence_report = _read(data_root / "evidence_export_report.json")
    validation = _read(results_root / "data_validation.json")
    power = _read(results_root / "power_gate.json")
    offsets = _read(data_root / "distance_offset_report.json")
    features = _read(data_root / "feature_extraction_report.json")
    manifest = _read(data_root / "feature_manifest.json")
    pairs_path = data_root / "canonical_pairs.parquet"
    pair_file = pq.ParquetFile(pairs_path)
    pairs = pd.read_parquet(
        pairs_path,
        columns=[
            "pair_id",
            "bin_i",
            "bin_j",
            "distance_bp",
            "distance_band",
            "distance_bin",
            "tile_row",
            "split",
        ],
    )
    expected_ids = np.arange(len(pairs), dtype=np.int64)
    minimum, maximum_exclusive = distance_range_bp(config)
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    feature_store = zarr.open_group(
        str(data_root / "pair_features.zarr"), mode="r"
    )
    recombination = all(
        np.array_equal(
            np.asarray(evidence["counts_a"][index], np.uint64)
            + np.asarray(evidence["counts_b"][index], np.uint64),
            np.asarray(evidence["full_count"][index], np.uint64),
        )
        for index in range(evidence["full_count"].shape[0])
    )
    forbidden_test_artifacts = [
        str(path)
        for path in results_root.glob("*test*")
        if path.is_file()
    ]
    checks = {
        "schema_version_3": int(config.get("schema_version", 0)) == 3,
        "provenance_schema_version_3": int(contract.get("schema_version", 0))
        == 3,
        "strict_stop_conditions": config.get("strict_stop_conditions") is True,
        "resolution_input_matches": resolution_report.get("resolution")
        == resolution_contract(config),
        "tile_split_counts_preserved": resolution_report["tiles"].get(
            "split_counts"
        )
        == {"train": 2753, "validation": 473, "test": 443},
        "cooler_coarsened_by_two": resolution_report["cooler"].get("factor")
        == 2,
        "cooler_count_conserved": resolution_report["cooler"].get(
            "count_conservation"
        )
        is True,
        "cooler_weights_ignored": resolution_report["cooler"].get(
            "balance_weights_consulted"
        )
        is False,
        "pair_count_positive": pair_file.metadata.num_rows > 0,
        "pair_ids_aligned": np.array_equal(
            pairs["pair_id"].to_numpy(np.int64), expected_ids
        ),
        "pair_anchors_10kb_aligned": bool(
            np.all(pairs["bin_i"].to_numpy(np.int64) % 10_000 == 0)
            and np.all(pairs["bin_j"].to_numpy(np.int64) % 10_000 == 0)
        ),
        "pair_distances_half_open": bool(
            pairs["distance_bp"].between(
                minimum, maximum_exclusive, inclusive="left"
            ).all()
        ),
        "maximum_represented_distance_990kb": int(
            pairs["distance_bp"].max()
        )
        == 990_000,
        "exact_group_size_at_most_75": int(
            pairs.groupby(
                ["tile_row", "distance_bin"], observed=True
            ).size().max()
        )
        <= 75,
        "cross_split_anchor_overlap_absent": pair_report.get(
            "cross_split_anchor_overlap_after_purge"
        )
        == 0,
        "annotation_pair_count_matches": int(
            annotation.get("pair_count", -1)
        )
        == len(pairs),
        "targets_streamed_from_raw": evidence_report.get(
            "cooler_pixels_consulted_for_targets"
        )
        is False,
        "stream_counts_conserved": evidence_report.get(
            "stream_to_matrix_conservation"
        )
        is True,
        "cooler_candidate_counts_conserved": evidence_report.get(
            "cooler_candidate_count_conservation"
        )
        is True,
        "evidence_balance_weights_ignored": evidence_report.get(
            "cooler_balance_weights_consulted"
        )
        is False,
        "all_primary_halves_recombine": recombination,
        "evidence_schema_version_3": int(
            evidence.attrs.get("schema_version", 0)
        )
        == 3,
        "evidence_resolution_matches": evidence.attrs.get("resolution")
        == resolution_contract(config),
        "power_gate_eligible": power.get("eligible") is True,
        "power_positive_fractional": power.get("tie_mode")
        == "positive_fractional",
        "power_zero_padding_disabled": power.get("zero_padding") is False,
        "distance_fit_training_only": offsets.get("method")
        == "training_only_exposure_adjusted_isotonic_decreasing",
        "distance_curve_monotone": offsets.get("monotone_nonincreasing")
        is True,
        "feature_pair_count_matches": int(features.get("pair_count", -1))
        == len(pairs),
        "feature_geometry_100_bins": int(
            feature_store.attrs.get("target_bins", 0)
        )
        == 100,
        "feature_overlap_width_at_most_six": int(
            feature_store.attrs.get(
                "maximum_native_overlaps_per_target_bin", 99
            )
        )
        <= 6,
        "feature_manifest_present": "features" in manifest
        and "distance_offsets" in manifest,
        "validation_count_conserved": validation.get("count_conservation")
        is True,
        "test_artifacts_absent": not forbidden_test_artifacts,
    }
    report = {
        "schema_version": 1,
        "prepared": bool(checks) and all(checks.values()),
        "pair_count": len(pairs),
        "checks": checks,
        "forbidden_test_artifacts": forbidden_test_artifacts,
        "resolution": resolution_contract(config),
    }
    atomic_json(output, report)
    if not report["prepared"]:
        raise RuntimeError(
            "10 kb preparation verification failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(load_config(args.config), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
