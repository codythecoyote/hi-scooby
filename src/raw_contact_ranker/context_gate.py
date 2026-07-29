from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    configured_distance_bands,
    selected_zarr_row,
    sha256_file,
)
from .context_head import benjamini_hochberg
from .metrics import build_top_contact_groups
from .power import fractional_top
from .topology import (
    _fixed_baseline_mean,
    fit_gamma_poisson_prior,
    gamma_poisson_log_enrichment,
)


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _conditional_group_rows(
    pairs: pd.DataFrame,
    counts: np.ndarray,
    shared: np.ndarray,
    candidate_delta: np.ndarray,
    baseline_delta: np.ndarray,
    *,
    band: str,
) -> pd.DataFrame:
    exposure = np.log(
        np.clip(pairs["exposure"].to_numpy(np.float64), 1e-12, None)
    )
    distance = pairs["distance_bin"].to_numpy(np.int64)
    width = int(distance.max(initial=0)) + 1
    code = pairs["tile_row"].to_numpy(np.int64) * width + distance
    selected_band = pairs["distance_band"].astype(str).eq(band).to_numpy()
    rows = []
    for group in np.unique(code[selected_band]):
        selected = selected_band & (code == group)
        events = float(counts[selected].sum())
        if events <= 0:
            continue

        def likelihood(delta: np.ndarray) -> float:
            score = shared[selected] + delta[selected]
            score = score - score.mean()
            logits = exposure[selected] + score
            maximum = float(logits.max())
            log_normalizer = maximum + math.log(
                float(np.exp(logits - maximum).sum())
            )
            return float(np.sum(counts[selected] * (logits - log_normalizer)))

        rows.append(
            {
                "chrom": str(pairs.loc[selected, "chrom"].iloc[0]),
                "weight": events,
                "value": (
                    likelihood(candidate_delta) - likelihood(baseline_delta)
                )
                / events,
            }
        )
    return pd.DataFrame(rows)


def _top_improvement_rows(
    pairs: pd.DataFrame,
    counts: np.ndarray,
    target: np.ndarray,
    shared: np.ndarray,
    candidate_delta: np.ndarray,
    baseline_delta: np.ndarray,
    *,
    band: str,
    fraction: float,
) -> pd.DataFrame:
    rows = []
    for group_band, tile_row, indices in build_top_contact_groups(
        pairs["tile_row"].to_numpy(),
        pairs["distance_band"].to_numpy(),
    ):
        if str(group_band) != band:
            continue
        size = len(indices)
        k = max(1, int(math.ceil(size * fraction)))
        observed = fractional_top(
            target[indices], k, eligible=counts[indices] > 0
        )
        candidate = fractional_top(
            shared[indices] + candidate_delta[indices],
            k,
            eligible=np.ones(size, bool),
        )
        baseline = fractional_top(
            shared[indices] + baseline_delta[indices],
            k,
            eligible=np.ones(size, bool),
        )
        if observed is None or candidate is None or baseline is None:
            continue
        rows.append(
            {
                "chrom": str(pairs.iloc[indices[0]]["chrom"]),
                "weight": k,
                "value": float(
                    (
                        np.dot(observed.weights, candidate.weights)
                        - np.dot(observed.weights, baseline.weights)
                    )
                    / k
                ),
                "tile_row": int(tile_row),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_positive(
    rows: pd.DataFrame,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | None]:
    if rows.empty:
        return {
            "point": None,
            "lower": None,
            "median": None,
            "upper": None,
            "p_one_sided": None,
            "replicates": 0,
        }

    def aggregate(frame: pd.DataFrame) -> float:
        return float(
            np.average(
                frame["value"].to_numpy(float),
                weights=frame["weight"].to_numpy(float),
            )
        )

    chromosomes = np.asarray(sorted(rows["chrom"].unique()), object)
    grouped = {
        str(chrom): rows.loc[rows["chrom"].eq(chrom)]
        for chrom in chromosomes
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    for index in range(replicates):
        sample = rng.choice(chromosomes, len(chromosomes), replace=True)
        values[index] = aggregate(
            pd.concat([grouped[str(chrom)] for chrom in sample])
        )
    alpha = 1.0 - confidence
    return {
        "point": aggregate(rows),
        "lower": float(np.quantile(values, alpha)),
        "median": float(np.median(values)),
        "upper": float(np.quantile(values, confidence)),
        "p_one_sided": float((1 + np.count_nonzero(values <= 0)) / (replicates + 1)),
        "replicates": replicates,
    }


def _load_mode(
    report_path: Path,
    expected_pair_ids: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    report = _read(report_path)
    prediction_path = Path(report["prediction_paths"]["validation"])
    prediction = pd.read_parquet(prediction_path).sort_values(
        "pair_id", kind="stable"
    )
    if not np.array_equal(
        prediction["pair_id"].to_numpy(np.int64), expected_pair_ids
    ):
        raise ValueError(f"Context predictions do not align: {prediction_path}")
    columns = [
        f"context_delta_{index:03d}" for index in range(len(report["units"]))
    ]
    return report, prediction[columns].to_numpy(np.float64)


def evaluate_context_extension(
    config: dict[str, Any],
    *,
    shared_predictions: Path,
    onehot_report: Path,
    rna_report: Path,
    permuted_rna_report: Path,
    output: Path,
) -> dict[str, Any]:
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    validation = pairs["split"].eq("validation").to_numpy()
    validation_pairs = pairs.loc[validation].reset_index(drop=True)
    train = pairs["split"].eq("train").to_numpy()
    shared_frame = pd.read_parquet(shared_predictions).sort_values(
        "pair_id", kind="stable"
    )
    if not np.array_equal(
        shared_frame["pair_id"].to_numpy(np.int64),
        validation_pairs["pair_id"].to_numpy(np.int64),
    ):
        raise ValueError("Shared predictions do not align with validation")
    shared = shared_frame["shared_residual_score"].to_numpy(np.float64)
    mode_reports = {}
    mode_delta = {}
    for mode, path in (
        ("onehot", onehot_report),
        ("rna", rna_report),
        ("rna_permuted", permuted_rna_report),
    ):
        report, delta = _load_mode(
            path, validation_pairs["pair_id"].to_numpy(np.int64)
        )
        mode_reports[mode] = report
        mode_delta[mode] = delta
    unit_ids = [
        str(row["output_id"]) for row in mode_reports["onehot"]["units"]
    ]
    if any(
        [str(row["output_id"]) for row in mode_reports[mode]["units"]]
        != unit_ids
        for mode in ("rna", "rna_permuted")
    ):
        raise RuntimeError("Context modes were trained on different outputs")
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    context_index = {
        str(name): index
        for index, name in enumerate(contexts["cell_type"].astype(str))
    }
    authorized_ids = pairs.loc[
        pairs["split"].isin(["train", "validation"]), "pair_id"
    ].to_numpy(np.int64)
    context_counts = {}
    for name, index in tqdm(
        context_index.items(),
        total=len(context_index),
        desc="Load validation context counts",
        unit="context",
    ):
        context_counts[name] = selected_zarr_row(
            evidence["full_count"],
            index,
            authorized_ids,
            pair_count=len(pairs),
            dtype=np.uint64,
        )
    units = mode_reports["onehot"]["units"]
    counts_by_unit = []
    targets_by_unit = []
    for unit in tqdm(
        units,
        desc="Build validation context targets",
        unit="output",
    ):
        counts = np.sum(
            np.stack(
                [
                    context_counts[name]
                    for name in unit["members"]
                ]
            ),
            axis=0,
            dtype=np.uint64,
        )
        baseline, _ = _fixed_baseline_mean(pairs, counts)
        shape = fit_gamma_poisson_prior(counts[train], baseline[train])
        counts_by_unit.append(counts)
        targets_by_unit.append(
            gamma_poisson_log_enrichment(counts, baseline, shape)
        )
    bands = [str(row["id"]) for row in configured_distance_bands(config)]
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    fraction = float(config["power"]["primary_top_fraction"])
    comparisons: dict[str, Any] = {}
    primary_p: dict[str, float] = {}

    def comparison(
        unit_index: int,
        candidate: np.ndarray,
        baseline: np.ndarray,
        label: str,
    ) -> dict[str, Any]:
        unit_report = {}
        counts = counts_by_unit[unit_index][validation].astype(np.float64)
        target = targets_by_unit[unit_index][validation]
        for band_index, band in enumerate(bands):
            ll_rows = _conditional_group_rows(
                validation_pairs,
                counts,
                shared,
                candidate,
                baseline,
                band=band,
            )
            top_rows = _top_improvement_rows(
                validation_pairs,
                counts,
                target,
                shared,
                candidate,
                baseline,
                band=band,
                fraction=fraction,
            )
            ll = _bootstrap_positive(
                ll_rows,
                replicates=replicates,
                confidence=confidence,
                seed=int(config["seed"]) + unit_index * 101 + band_index,
            )
            top = _bootstrap_positive(
                top_rows,
                replicates=replicates,
                confidence=confidence,
                seed=int(config["seed"]) + 10_000 + unit_index * 101 + band_index,
            )
            unit_report[band] = {"likelihood": ll, "top_contact": top}
            if label in {"onehot_vs_shared", "rna_vs_shared"}:
                for metric_name, metric in (
                    ("likelihood", ll),
                    ("top_contact", top),
                ):
                    if metric["p_one_sided"] is not None:
                        primary_p[
                            f"{label}:{unit_ids[unit_index]}:{band}:{metric_name}"
                        ] = float(metric["p_one_sided"])
        return unit_report

    for unit_index, output_id in enumerate(
        tqdm(
            unit_ids,
            desc="Gate validation context heads",
            unit="output",
        )
    ):
        zero = np.zeros(len(validation_pairs), np.float64)
        next_index = (unit_index + 1) % len(unit_ids)
        comparisons[output_id] = {
            "onehot_vs_shared": comparison(
                unit_index,
                mode_delta["onehot"][:, unit_index],
                zero,
                "onehot_vs_shared",
            ),
            "rna_vs_shared": comparison(
                unit_index,
                mode_delta["rna"][:, unit_index],
                zero,
                "rna_vs_shared",
            ),
            "rna_vs_onehot": comparison(
                unit_index,
                mode_delta["rna"][:, unit_index],
                mode_delta["onehot"][:, unit_index],
                "rna_vs_onehot",
            ),
            "rna_vs_swapped": comparison(
                unit_index,
                mode_delta["rna"][:, unit_index],
                mode_delta["rna"][:, next_index],
                "rna_vs_swapped",
            ),
            "rna_vs_permuted": comparison(
                unit_index,
                mode_delta["rna"][:, unit_index],
                mode_delta["rna_permuted"][:, unit_index],
                "rna_vs_permuted",
            ),
        }
    adjusted = benjamini_hochberg(primary_p)
    fdr = float(config["contexts"]["fdr"])

    def primary_passed(output_id: str, label: str) -> bool:
        return all(
            adjusted.get(
                f"{label}:{output_id}:{band}:{metric}", 1.0
            )
            <= fdr
            and comparisons[output_id][label][band][metric]["point"] is not None
            and comparisons[output_id][label][band][metric]["point"] > 0
            for band in bands
            for metric in ("likelihood", "top_contact")
        )

    def extra_passed(output_id: str, label: str) -> bool:
        return all(
            comparisons[output_id][label][band][metric]["lower"] is not None
            and comparisons[output_id][label][band][metric]["lower"] > 0
            for band in bands
            for metric in ("likelihood", "top_contact")
        )

    release_path = (
        Path(config["outputs"]["results_root"])
        / "context_predictions.all.parquet"
    )
    outputs = {}
    for unit_index, unit in enumerate(units):
        output_id = str(unit["output_id"])
        onehot_ok = primary_passed(output_id, "onehot_vs_shared")
        rna_ok = (
            primary_passed(output_id, "rna_vs_shared")
            and extra_passed(output_id, "rna_vs_onehot")
            and extra_passed(output_id, "rna_vs_swapped")
            and extra_passed(output_id, "rna_vs_permuted")
        )
        mode = "rna" if rna_ok else ("onehot" if onehot_ok else None)
        selected_report = mode_reports[mode] if mode else None
        outputs[output_id] = {
            "output_type": unit["output_type"],
            "members": unit["members"],
            "topology_accepted": mode is not None,
            "accepted": False,
            "selected_mode": mode,
            "claim_scope": (
                "rna_conditioned"
                if mode == "rna"
                else ("label_conditioned" if mode == "onehot" else None)
            ),
            "checkpoint": (
                selected_report["checkpoint"] if selected_report else None
            ),
            "checkpoint_sha256": (
                selected_report["checkpoint_sha256"]
                if selected_report
                else None
            ),
            "validation_prediction_path": (
                selected_report["prediction_paths"]["validation"]
                if selected_report
                else None
            ),
            "training_prediction_path": (
                selected_report["prediction_paths"]["train"]
                if selected_report
                else None
            ),
            "prediction_path": str(release_path),
            "prediction_column": f"context_delta_{unit_index:03d}",
            "failure_reason": (
                None if mode else "context_topology_failed_fdr_or_ablation_gates"
            ),
            "requires_rate_recalibration": mode is not None,
        }
    report = {
        "schema_version": 1,
        "fdr": fdr,
        "adjusted_p_values": adjusted,
        "comparisons": comparisons,
        "outputs": outputs,
        "mode_reports": {
            mode: {
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_sha256": mode_reports[mode]["checkpoint_sha256"],
            }
            for mode, path in (
                ("onehot", onehot_report),
                ("rna", rna_report),
                ("rna_permuted", permuted_rna_report),
            )
        },
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report


def evaluate_context_test_extension(
    config: dict[str, Any],
    *,
    validation_context_gate: Path,
    shared_test_predictions: Path,
    onehot_test_predictions: Path,
    rna_test_predictions: Path,
    permuted_rna_test_predictions: Path,
    frozen_release: Path,
    test_lock: Path,
    output: Path,
) -> dict[str, Any]:
    """Apply the frozen context comparisons once on the untouched test split."""
    from .release import verify_test_lock

    verify_test_lock(frozen_release, test_lock)
    validation_gate = _read(validation_context_gate)
    accepted_validation = {
        output_id: row
        for output_id, row in validation_gate["outputs"].items()
        if row.get("accepted") is True
    }
    if not accepted_validation:
        raise RuntimeError("No validation-accepted context output needs a test gate")
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    )
    test = pairs["split"].eq("test").to_numpy()
    train = pairs["split"].eq("train").to_numpy()
    test_pairs = pairs.loc[test].reset_index(drop=True)
    expected_ids = test_pairs["pair_id"].to_numpy(np.int64)
    shared_frame = pd.read_parquet(shared_test_predictions).sort_values(
        "pair_id", kind="stable"
    )
    if not np.array_equal(
        shared_frame["pair_id"].to_numpy(np.int64), expected_ids
    ):
        raise ValueError("Shared context-test predictions do not align")
    shared = shared_frame["shared_residual_score"].to_numpy(np.float64)
    mode_paths = {
        "onehot": onehot_test_predictions,
        "rna": rna_test_predictions,
        "rna_permuted": permuted_rna_test_predictions,
    }
    mode_delta = {}
    training_reports = {}
    for mode, prediction_path in mode_paths.items():
        training_path = Path(
            validation_gate["mode_reports"][mode]["path"]
        )
        if (
            sha256_file(training_path)
            != validation_gate["mode_reports"][mode]["sha256"]
        ):
            raise RuntimeError(f"{mode} training report changed after freeze")
        training_report = _read(training_path)
        training_reports[mode] = training_report
        frame = pd.read_parquet(prediction_path).sort_values(
            "pair_id", kind="stable"
        )
        if not np.array_equal(
            frame["pair_id"].to_numpy(np.int64), expected_ids
        ):
            raise ValueError(f"{mode} test predictions do not align")
        columns = [
            f"context_delta_{index:03d}"
            for index in range(len(training_report["units"]))
        ]
        mode_delta[mode] = frame[columns].to_numpy(np.float64)
        test_report_path = prediction_path.with_suffix(".json")
        test_report = _read(test_report_path)
        if (
            test_report.get("checkpoint_sha256")
            != validation_gate["mode_reports"][mode]["checkpoint_sha256"]
        ):
            raise RuntimeError(f"{mode} context checkpoint changed before test")
    units = training_reports["onehot"]["units"]
    unit_ids = [str(row["output_id"]) for row in units]
    if any(
        [str(row["output_id"]) for row in training_reports[mode]["units"]]
        != unit_ids
        for mode in ("rna", "rna_permuted")
    ):
        raise RuntimeError("Context test modes use different output order")
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    context_index = {
        str(name): index
        for index, name in enumerate(contexts["cell_type"].astype(str))
    }
    authorized_ids = pairs.loc[
        pairs["split"].isin(["train", "test"]), "pair_id"
    ].to_numpy(np.int64)
    context_counts = {}
    for name, index in tqdm(
        context_index.items(),
        total=len(context_index),
        desc="Load test context counts",
        unit="context",
    ):
        context_counts[name] = selected_zarr_row(
            evidence["full_count"],
            index,
            authorized_ids,
            pair_count=len(pairs),
            dtype=np.uint64,
        )
    counts_by_unit = []
    target_by_unit = []
    for unit in tqdm(
        units,
        desc="Build test context targets",
        unit="output",
    ):
        counts = np.sum(
            np.stack(
                [
                    context_counts[name]
                    for name in unit["members"]
                ]
            ),
            axis=0,
            dtype=np.uint64,
        )
        baseline, _ = _fixed_baseline_mean(pairs, counts)
        shape = fit_gamma_poisson_prior(counts[train], baseline[train])
        counts_by_unit.append(counts)
        target_by_unit.append(
            gamma_poisson_log_enrichment(counts, baseline, shape)
        )
    bands = [str(row["id"]) for row in configured_distance_bands(config)]
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"].get("confidence_level", 0.975))
    fraction = float(config["power"]["primary_top_fraction"])

    def compare(
        unit_index: int,
        candidate: np.ndarray,
        baseline: np.ndarray,
        seed_offset: int,
    ) -> dict[str, Any]:
        result = {}
        counts = counts_by_unit[unit_index][test].astype(np.float64)
        target = target_by_unit[unit_index][test]
        for band_index, band in enumerate(bands):
            likelihood = _bootstrap_positive(
                _conditional_group_rows(
                    test_pairs,
                    counts,
                    shared,
                    candidate,
                    baseline,
                    band=band,
                ),
                replicates=replicates,
                confidence=confidence,
                seed=(
                    int(config["seed"])
                    + seed_offset
                    + unit_index * 101
                    + band_index
                ),
            )
            top = _bootstrap_positive(
                _top_improvement_rows(
                    test_pairs,
                    counts,
                    target,
                    shared,
                    candidate,
                    baseline,
                    band=band,
                    fraction=fraction,
                ),
                replicates=replicates,
                confidence=confidence,
                seed=(
                    int(config["seed"])
                    + 50_000
                    + seed_offset
                    + unit_index * 101
                    + band_index
                ),
            )
            result[band] = {
                "likelihood": likelihood,
                "top_contact": top,
            }
        return result

    def passes(comparison: dict[str, Any]) -> bool:
        return all(
            comparison[band][metric]["lower"] is not None
            and comparison[band][metric]["lower"] > 0
            for band in bands
            for metric in ("likelihood", "top_contact")
        )

    comparisons = {}
    outputs = {}
    validation_outputs = validation_gate["outputs"]
    for output_id, validation_row in tqdm(
        validation_outputs.items(),
        total=len(validation_outputs),
        desc="Gate frozen context heads on test",
        unit="output",
    ):
        if validation_row.get("accepted") is not True:
            outputs[output_id] = {
                **validation_row,
                "final_test_accepted": False,
            }
            continue
        unit_index = unit_ids.index(output_id)
        mode = str(validation_row["selected_mode"])
        zero = np.zeros(len(test_pairs), np.float64)
        primary = compare(
            unit_index,
            mode_delta[mode][:, unit_index],
            zero,
            0,
        )
        output_comparisons = {"selected_vs_shared": primary}
        accepted = passes(primary)
        if mode == "rna":
            next_index = (unit_index + 1) % len(unit_ids)
            extras = {
                "rna_vs_onehot": compare(
                    unit_index,
                    mode_delta["rna"][:, unit_index],
                    mode_delta["onehot"][:, unit_index],
                    10_000,
                ),
                "rna_vs_swapped": compare(
                    unit_index,
                    mode_delta["rna"][:, unit_index],
                    mode_delta["rna"][:, next_index],
                    20_000,
                ),
                "rna_vs_permuted": compare(
                    unit_index,
                    mode_delta["rna"][:, unit_index],
                    mode_delta["rna_permuted"][:, unit_index],
                    30_000,
                ),
            }
            output_comparisons.update(extras)
            accepted &= all(passes(value) for value in extras.values())
        comparisons[output_id] = output_comparisons
        outputs[output_id] = {
            **validation_row,
            "final_test_accepted": bool(accepted),
        }
    required = [
        output_id
        for output_id, row in validation_gate["outputs"].items()
        if row.get("accepted") is True
    ]
    checks = {
        output_id: outputs[output_id]["final_test_accepted"]
        for output_id in required
    }
    report = {
        "schema_version": 1,
        "split": "test",
        "validation_context_gate": str(validation_context_gate),
        "validation_context_gate_sha256": sha256_file(
            validation_context_gate
        ),
        "shared_predictions_sha256": sha256_file(shared_test_predictions),
        "mode_prediction_sha256": {
            mode: sha256_file(path) for mode, path in mode_paths.items()
        },
        "comparisons": comparisons,
        "outputs": outputs,
        "checks": checks,
        "accepted": bool(checks) and all(checks.values()),
        "test_accessed": True,
        "retuning_permitted": False,
    }
    atomic_json(output, report)
    return report
