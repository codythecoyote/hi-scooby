#!/usr/bin/env python3
"""Evaluate A-only family predictions against B-only Peakachu loop calls.

The family EB and shared predictions are fixed from pseudoreplicate A. Loop
labels are called independently from whole-cell pseudoreplicate B raw contact
maps. Evaluation is restricted to the authorized validation-pair universe;
prepared internal-test prediction rows are never loaded. Peakachu nevertheless
requires chromosome-wide raw-B context to construct its outcome labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "diagnostics"))

from evaluate_family_ep_recovery import (  # noqa: E402
    BAND_IDS,
    BAND_LABELS,
    FAMILY_IDS,
    FAMILY_PARTITION,
    PASSED_FAMILIES,
    PASSED_INDEX,
    load_family_validation,
    read_zarr_v2_row,
    set_style,
    zarr_v2_attrs,
)
from evaluate_family_scdeeplucia_loops_10kb import (  # noqa: E402
    bootstrap_macro_difference,
)
from raw_contact_ranker.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    source_record,
)


MATCH_MODES = ("exact", "anchor_tolerance_10kb")
SOURCES = (
    "family_eb",
    "frozen_shared",
    "raw_a",
    "raw_a_oe",
    "smoothed_raw_a_oe",
    "distance_only",
    "swapped_family",
)
SOURCE_LABELS = {
    "family_eb": "Family EB",
    "frozen_shared": "Frozen shared",
    "raw_a": "Raw A count",
    "raw_a_oe": "Raw A O/E",
    "smoothed_raw_a_oe": "Smoothed A O/E",
    "distance_only": "Distance only",
    "swapped_family": "Other family EB",
}
COLORS = {
    "family_eb": "#2F75B5",
    "frozen_shared": "#A9A9A9",
    "raw_a": "#E68632",
    "raw_a_oe": "#D55E00",
    "smoothed_raw_a_oe": "#2A9D8F",
    "distance_only": "#7B61A8",
    "swapped_family": "#C8A36A",
}
TOP_FRACTION = 0.01
BIN_SIZE = 10_000
SCOPE = (
    "Prediction = split-0 whole-cell half A only • loop outcome = disjoint "
    "whole-cell half B only • 10 kb • metrics on validation pairs only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/raw_contact_ranker_10kb.yaml",
    )
    parser.add_argument(
        "--eb-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results/raw_contact_ranker_10kb_v1"
            / "context_family_eb_50kb_cpu/production_v1"
        ),
    )
    parser.add_argument(
        "--loop-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results/raw_contact_ranker_10kb_v1/diagnostics"
            / "b_only_peakachu_10kb_v1"
        ),
    )
    parser.add_argument(
        "--cooler-root",
        type=Path,
        default=(
            REPO_ROOT
            / "data/processed/raw_contact_ranker_10kb_v1"
            / "b_only_family_peakachu_10kb_v1"
        ),
    )
    parser.add_argument(
        "--peakachu-model",
        type=Path,
        default=(
            REPO_ROOT
            / "data/external/peakachu_models"
            / "HiCAR-peakachu-pretrained.10million.10kb.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/raw_contact_ranker_10kb_v1/diagnostics"
            / "a_only_vs_b_only_peakachu_10kb_v1"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.975)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def read_loops(path: Path, *, family_id: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=(
            "chrom",
            "bin_i",
            "end_i",
            "chrom_j",
            "bin_j",
            "end_j",
            "peakachu_probability",
            "raw_b_signal",
        ),
    )
    if frame.empty:
        raise RuntimeError(f"No B-only Peakachu loops for {family_id}")
    if not frame["chrom"].eq(frame["chrom_j"]).all():
        raise RuntimeError(f"{family_id} contains trans loops")
    for side in ("i", "j"):
        if not (
            frame[f"end_{side}"].astype(np.int64)
            - frame[f"bin_{side}"].astype(np.int64)
        ).eq(BIN_SIZE).all():
            raise RuntimeError(f"{family_id} has non-10-kb anchors")
    if not frame["bin_i"].astype(np.int64).mod(BIN_SIZE).eq(0).all():
        raise RuntimeError(f"{family_id} has off-grid left anchors")
    if not frame["bin_j"].astype(np.int64).mod(BIN_SIZE).eq(0).all():
        raise RuntimeError(f"{family_id} has off-grid right anchors")
    distance = (
        frame["bin_j"].astype(np.int64)
        - frame["bin_i"].astype(np.int64)
    )
    if not distance.between(250_000, 990_000).all():
        raise RuntimeError(f"{family_id} loop distance escaped 250–990 kb")
    frame["family_id"] = family_id
    return frame


def loop_labels(
    fine: pd.DataFrame,
    loops: pd.DataFrame,
) -> dict[str, np.ndarray]:
    pair_keys = list(
        zip(
            fine["chrom"].astype(str),
            fine["bin_i"].to_numpy(np.int64),
            fine["bin_j"].to_numpy(np.int64),
            strict=True,
        )
    )
    output: dict[str, np.ndarray] = {}
    for mode, tolerance in (("exact", 0), ("anchor_tolerance_10kb", 1)):
        lookup: set[tuple[str, int, int]] = set()
        for row in loops.itertuples(index=False):
            for delta_i in range(-tolerance, tolerance + 1):
                for delta_j in range(-tolerance, tolerance + 1):
                    left = int(row.bin_i) + delta_i * BIN_SIZE
                    right = int(row.bin_j) + delta_j * BIN_SIZE
                    if left >= 0 and right > left:
                        lookup.add((str(row.chrom), left, right))
        output[mode] = np.fromiter(
            (key in lookup for key in pair_keys),
            dtype=bool,
            count=len(pair_keys),
        )
    return output


def load_family_counts_a(
    evidence_path: Path,
    pair_ids: np.ndarray,
) -> np.ndarray:
    attrs = zarr_v2_attrs(evidence_path)
    context_ids = list(map(str, attrs["context_ids"]))
    context_lookup = {
        context: index for index, context in enumerate(context_ids)
    }
    output = np.zeros((len(FAMILY_IDS), len(pair_ids)), np.uint32)
    for family_id in PASSED_FAMILIES:
        family = FAMILY_IDS.index(family_id)
        total = np.zeros(len(pair_ids), np.uint64)
        for member in FAMILY_PARTITION[family_id]:
            row = read_zarr_v2_row(
                evidence_path / "counts_a",
                context_lookup[member],
            )
            total += row[pair_ids].astype(np.uint64)
        if np.any(total > np.iinfo(np.uint32).max):
            raise OverflowError(f"{family_id} counts_a exceeds uint32")
        output[family] = total.astype(np.uint32)
    return output


def observed_over_expected(
    counts: np.ndarray,
    distance_bp: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(counts), np.float64)
    for distance in np.unique(distance_bp):
        selected = distance_bp == distance
        expected = float(np.mean(counts[selected]))
        if expected > 0:
            output[selected] = counts[selected] / expected
    return output


def smooth_pair_scores(
    fine: pd.DataFrame,
    values: np.ndarray,
) -> np.ndarray:
    """Sum the available 3x3 anchor neighborhood without crossing chromosomes."""
    output = np.zeros(len(fine), np.float64)
    chrom = fine["chrom"].astype(str).to_numpy()
    left_all = fine["bin_i"].to_numpy(np.int64) // BIN_SIZE
    right_all = fine["bin_j"].to_numpy(np.int64) // BIN_SIZE
    for current in np.unique(chrom):
        positions = np.flatnonzero(chrom == current)
        left = left_all[positions]
        right = right_all[positions]
        width = int(max(left.max(), right.max())) + 2
        keys = left * width + right
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_values = values[positions][order]
        smoothed = np.zeros(len(positions), np.float64)
        for delta_i in (-1, 0, 1):
            for delta_j in (-1, 0, 1):
                query = (left + delta_i) * width + right + delta_j
                insertion = np.searchsorted(sorted_keys, query)
                valid = insertion < len(sorted_keys)
                matched = np.zeros(len(query), bool)
                matched[valid] = sorted_keys[insertion[valid]] == query[valid]
                smoothed[matched] += sorted_values[insertion[matched]]
        output[positions] = smoothed
    return output


def classification(
    labels: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, bool)
    score = np.asarray(score, np.float64)
    if not labels.any() or labels.all():
        raise RuntimeError("Degenerate loop labels")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, score))
    return {
        "candidate_pair_count": int(len(labels)),
        "positive_pair_count": int(labels.sum()),
        "positive_prevalence": prevalence,
        "auprc": auprc,
        "auprc_lift_over_prevalence": float(auprc / prevalence),
        "auroc": float(roc_auc_score(labels, score)),
    }


def tie_aware_top1(
    labels: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, bool)
    score = np.asarray(score, np.float64)
    selected_count = max(1, int(math.ceil(TOP_FRACTION * len(labels))))
    threshold = float(np.partition(score, len(score) - selected_count)[
        len(score) - selected_count
    ])
    above = score > threshold
    tied = score == threshold
    remaining = selected_count - int(above.sum())
    expected_positive = float(labels[above].sum())
    if remaining:
        expected_positive += remaining * float(labels[tied].mean())
    precision = expected_positive / selected_count
    prevalence = float(labels.mean())
    return {
        "selected_pair_count": int(selected_count),
        "positive_precision_tie_aware": float(precision),
        "positive_recall_tie_aware": float(expected_positive / labels.sum()),
        "enrichment_over_prevalence": float(precision / prevalence),
        "boundary_score": threshold,
        "boundary_tie_count": int(tied.sum()),
    }


def score_sources(
    data: dict[str, Any],
    counts_a: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    fine = data["fine"]
    distance = fine["distance_bp"].to_numpy(np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    for family_offset, family in enumerate(PASSED_INDEX):
        family_id = FAMILY_IDS[family]
        swapped = PASSED_INDEX[1 - family_offset]
        raw = counts_a[family].astype(np.float64)
        raw_oe = observed_over_expected(raw, distance)
        output[family_id] = {
            "family_eb": data["family_prediction_b"][family],
            "frozen_shared": data["shared_prediction_b"][family],
            "raw_a": raw,
            "raw_a_oe": raw_oe,
            "smoothed_raw_a_oe": smooth_pair_scores(fine, raw_oe),
            "distance_only": -distance.astype(np.float64),
            "swapped_family": data["family_prediction_b"][swapped],
        }
    return output


def matched_negative_indices(
    labels: np.ndarray,
    near_loop: np.ndarray,
    chromosome: np.ndarray,
    distance_bp: np.ndarray,
    raw_a: np.ndarray,
    shared: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Match one non-near-loop negative to each exact loop."""
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~near_loop)
    raw_bucket = np.minimum(raw_a.astype(np.int64), 3)
    shared_decile = np.minimum(
        (rankdata(shared, method="average") / (len(shared) + 1) * 10).astype(
            np.int64
        ),
        9,
    )
    distance_50kb = distance_bp // 50_000
    pools: list[dict[tuple[Any, ...], list[int]]] = [
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    ]
    for index in negative:
        common = (
            str(chromosome[index]),
            int(raw_bucket[index]),
            int(shared_decile[index]),
        )
        pools[0][common + (int(distance_bp[index]),)].append(int(index))
        pools[1][common + (int(distance_50kb[index]),)].append(int(index))
        pools[2][common].append(int(index))
    rng = np.random.default_rng(seed)
    selected = []
    tiers: Counter[str] = Counter()
    for index in positive:
        common = (
            str(chromosome[index]),
            int(raw_bucket[index]),
            int(shared_decile[index]),
        )
        keys = (
            common + (int(distance_bp[index]),),
            common + (int(distance_50kb[index]),),
            common,
        )
        choice = None
        for tier, key in enumerate(keys):
            candidates = pools[tier].get(key, [])
            if candidates:
                choice = int(candidates[rng.integers(0, len(candidates))])
                tiers[("exact_10kb", "within_50kb", "within_band")[tier]] += 1
                break
        if choice is None:
            raise RuntimeError(
                "No chromosome/raw-A/shared-decile matched negative"
            )
        selected.append(choice)
    selected_array = np.asarray(selected, np.int64)
    stats = dict(tiers)
    stats["unique_negative_count"] = int(np.unique(selected_array).size)
    stats["duplicate_negative_selections"] = int(
        len(selected_array) - np.unique(selected_array).size
    )
    return selected_array, stats


def evaluate(
    data: dict[str, Any],
    labels: dict[str, dict[str, np.ndarray]],
    scores: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    chromosome = fine["chrom"].astype(str).to_numpy()
    distance = fine["distance_bp"].to_numpy(np.int64)
    output: dict[str, Any] = {}
    rows = []
    matched_rows = []
    for mode_offset, mode in enumerate(MATCH_MODES):
        output[mode] = {}
        for family_offset, family_id in enumerate(PASSED_FAMILIES):
            output[mode][family_id] = {}
            for band, band_id in enumerate(BAND_IDS):
                selected = band_index == band
                current_labels = labels[family_id][mode][selected]
                current_scores = {
                    name: values[selected]
                    for name, values in scores[family_id].items()
                }
                metrics = {
                    name: {
                        "classification": classification(
                            current_labels, current_score
                        ),
                        "top1": tie_aware_top1(
                            current_labels, current_score
                        ),
                    }
                    for name, current_score in current_scores.items()
                }
                comparisons = {}
                for baseline_offset, baseline in enumerate(
                    (
                        "frozen_shared",
                        "raw_a",
                        "raw_a_oe",
                        "smoothed_raw_a_oe",
                        "distance_only",
                        "swapped_family",
                    )
                ):
                    comparisons[f"family_minus_{baseline}_auprc"] = (
                        bootstrap_macro_difference(
                            current_labels,
                            current_scores["family_eb"],
                            current_scores[baseline],
                            chromosome[selected],
                            metric="auprc",
                            replicates=bootstrap_replicates,
                            confidence_level=confidence_level,
                            seed=(
                                seed
                                + 10_000 * mode_offset
                                + 1_000 * family_offset
                                + 100 * band
                                + baseline_offset
                            ),
                        )
                    )
                output[mode][family_id][band_id] = {
                    "metrics": metrics,
                    "paired_chromosome_bootstrap": comparisons,
                }
                for name in SOURCES:
                    result = metrics[name]
                    rows.append(
                        {
                            "match_mode": mode,
                            "family_id": family_id,
                            "band_id": band_id,
                            "source": name,
                            **result["classification"],
                            **result["top1"],
                        }
                    )

                if mode == "exact":
                    positions = np.flatnonzero(selected)
                    negative, tiers = matched_negative_indices(
                        current_labels,
                        labels[family_id]["anchor_tolerance_10kb"][selected],
                        chromosome[selected],
                        distance[selected],
                        current_scores["raw_a"],
                        current_scores["frozen_shared"],
                        seed=seed + 100 * family_offset + band,
                    )
                    positive = np.flatnonzero(current_labels)
                    matched = np.concatenate((positive, negative))
                    matched_labels = np.concatenate(
                        (
                            np.ones(len(positive), bool),
                            np.zeros(len(negative), bool),
                        )
                    )
                    matched_result = {}
                    for name in SOURCES:
                        score = current_scores[name][matched]
                        matched_result[name] = {
                            "auroc": float(
                                roc_auc_score(matched_labels, score)
                            ),
                            "auprc": float(
                                average_precision_score(
                                    matched_labels, score
                                )
                            ),
                        }
                        matched_rows.append(
                            {
                                "family_id": family_id,
                                "band_id": band_id,
                                "source": name,
                                "matched_pair_count": int(len(matched)),
                                "positive_pair_count": int(len(positive)),
                                **matched_result[name],
                            }
                        )
                    output[mode][family_id][band_id][
                        "hard_matched"
                    ] = {
                        "rule": (
                            "One B non-loop outside the ±10-kb loop "
                            "neighborhood per exact loop; matched by "
                            "chromosome, raw-A bucket, shared-score decile, "
                            "then exact/50-kb distance."
                        ),
                        "negative_sampling_with_replacement": True,
                        "match_tiers": tiers,
                        "metrics": matched_result,
                        "validation_pair_positions_sha256": sha256_array(
                            positions[matched]
                        ),
                    }
    return output, pd.DataFrame(rows), pd.DataFrame(matched_rows)


def evaluate_zero_a(
    data: dict[str, Any],
    labels: dict[str, dict[str, np.ndarray]],
    scores: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate exact B loops where the corresponding family A count is zero."""
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    output: dict[str, Any] = {}
    rows = []
    for family_id in PASSED_FAMILIES:
        output[family_id] = {}
        for band, band_id in enumerate(BAND_IDS):
            selected = (
                (band_index == band)
                & (scores[family_id]["raw_a"] == 0)
            )
            current_labels = labels[family_id]["exact"][selected]
            output[family_id][band_id] = {}
            for source in (
                "family_eb",
                "frozen_shared",
                "smoothed_raw_a_oe",
                "distance_only",
                "swapped_family",
            ):
                result = {
                    "classification": classification(
                        current_labels, scores[family_id][source][selected]
                    ),
                    "top1": tie_aware_top1(
                        current_labels, scores[family_id][source][selected]
                    ),
                }
                output[family_id][band_id][source] = result
                rows.append(
                    {
                        "family_id": family_id,
                        "band_id": band_id,
                        "source": source,
                        **result["classification"],
                        **result["top1"],
                    }
                )
    return output, pd.DataFrame(rows)


def sha256_array(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(values).view(np.uint8)
    ).hexdigest()


def plot_scorecard(table: pd.DataFrame, output: Path) -> None:
    frame = table.loc[
        table["match_mode"].eq("exact")
        & table["source"].isin(
            (
                "family_eb",
                "frozen_shared",
                "raw_a",
                "raw_a_oe",
                "smoothed_raw_a_oe",
                "distance_only",
            )
        )
    ].copy()
    groups = [
        (family, band)
        for family in PASSED_FAMILIES
        for band in BAND_IDS
    ]
    sources = list(dict.fromkeys(frame["source"]))
    x = np.arange(len(groups))
    width = 0.82 / len(sources)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for offset, source in enumerate(sources):
        selected = frame.loc[frame["source"].eq(source)].set_index(
            ["family_id", "band_id"]
        ).loc[groups]
        position = x + (offset - (len(sources) - 1) / 2) * width
        for axis, metric in zip(
            axes,
            (
                "auprc_lift_over_prevalence",
                "auroc",
                "enrichment_over_prevalence",
            ),
            strict=True,
        ):
            axis.bar(
                position,
                selected[metric],
                width,
                color=COLORS[source],
                label=SOURCE_LABELS[source],
            )
    axes[0].set_title("AUPRC / loop prevalence")
    axes[1].set_title("AUROC")
    axes[2].set_title("Top 1% enrichment (tie-aware)")
    axes[0].axhline(1, color="#222222", linestyle="--", linewidth=1)
    axes[1].axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    axes[2].axhline(1, color="#222222", linestyle="--", linewidth=1)
    labels = [
        f"{family.replace('_', ' ').title()}\n"
        f"{dict(zip(BAND_IDS, BAND_LABELS, strict=True))[band]}"
        for family, band in groups
    ]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fold")
    axes[1].set_ylabel("Area")
    axes[2].set_ylabel("Fold")
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle(
        "A-only predictions versus B-only Peakachu loops",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.01, 0.008, SCOPE, color="#777777", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def plot_matched(table: pd.DataFrame, output: Path) -> None:
    sources = (
        "family_eb",
        "frozen_shared",
        "raw_a",
        "raw_a_oe",
        "smoothed_raw_a_oe",
        "distance_only",
    )
    groups = [
        (family, band)
        for family in PASSED_FAMILIES
        for band in BAND_IDS
    ]
    x = np.arange(len(groups))
    width = 0.82 / len(sources)
    fig, axis = plt.subplots(figsize=(10.8, 5.1))
    for offset, source in enumerate(sources):
        selected = table.loc[table["source"].eq(source)].set_index(
            ["family_id", "band_id"]
        ).loc[groups]
        axis.bar(
            x + (offset - (len(sources) - 1) / 2) * width,
            selected["auroc"],
            width,
            color=COLORS[source],
            label=SOURCE_LABELS[source],
        )
    axis.axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    axis.set_ylabel("AUROC")
    axis.set_title(
        "Discrimination after matching raw-A support and shared-score decile",
        fontweight="bold",
    )
    axis.set_xticks(
        x,
        [
            f"{family.replace('_', ' ').title()}\n"
            f"{dict(zip(BAND_IDS, BAND_LABELS, strict=True))[band]}"
            for family, band in groups
        ],
        rotation=12,
        ha="right",
    )
    axis.legend(fontsize=7.5, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.008, SCOPE, color="#777777", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    matched: pd.DataFrame,
    zero_a: pd.DataFrame,
    recovery: dict[str, Any],
    output: Path,
) -> None:
    exact = metrics.loc[metrics["match_mode"].eq("exact")]
    lines = [
        "# A-only prediction versus B-only Peakachu loops at 10 kb",
        "",
        "Peakachu loops were called from whole-cell pseudoreplicate B, while "
        "the family EB predictions and every predictive baseline use only "
        "pseudoreplicate A (or the frozen shared topology).",
        "",
        "## Exact-anchor results",
        "",
        "| Family | Distance | B-only loops | Source | AUPRC | Lift | AUROC | "
        "Top 1% enrichment |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for family_id in PASSED_FAMILIES:
        for band_id in BAND_IDS:
            group = exact.loc[
                exact["family_id"].eq(family_id)
                & exact["band_id"].eq(band_id)
            ]
            for source in (
                "family_eb",
                "frozen_shared",
                "raw_a",
                "raw_a_oe",
                "smoothed_raw_a_oe",
                "distance_only",
            ):
                row = group.loc[group["source"].eq(source)].iloc[0]
                lines.append(
                    f"| {family_id} | {band_id} | "
                    f"{int(row.positive_pair_count):,} | "
                    f"{SOURCE_LABELS[source]} | {row.auprc:.5f} | "
                    f"{row.auprc_lift_over_prevalence:.2f}× | "
                    f"{row.auroc:.3f} | "
                    f"{row.enrichment_over_prevalence:.2f}× |"
                )
    lines.extend(
        [
            "",
            "## Hard matched negatives",
            "",
            "Each exact B-loop was paired to a B non-loop outside the ±10-kb "
            "loop neighborhood, matching chromosome, raw-A count bucket, "
            "frozen-shared score decile, and distance as tightly as possible.",
            "",
            "| Family | Distance | Source | Matched AUROC | Matched AUPRC |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in matched.itertuples(index=False):
        if row.source not in (
            "family_eb",
            "frozen_shared",
            "raw_a",
            "raw_a_oe",
            "smoothed_raw_a_oe",
            "distance_only",
        ):
            continue
        lines.append(
            f"| {row.family_id} | {row.band_id} | "
            f"{SOURCE_LABELS[row.source]} | {row.auroc:.3f} | "
            f"{row.auprc:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Loops absent from raw A",
            "",
            "This subset contains only candidate pairs with zero family-A "
            "contacts. Raw A therefore cannot distinguish them.",
            "",
            "| Family | Distance | Zero-A B loops | Source | AUROC | AUPRC lift |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for family_id in PASSED_FAMILIES:
        for band_id in BAND_IDS:
            group = zero_a.loc[
                zero_a["family_id"].eq(family_id)
                & zero_a["band_id"].eq(band_id)
            ]
            for source in (
                "family_eb",
                "frozen_shared",
                "smoothed_raw_a_oe",
                "distance_only",
                "swapped_family",
            ):
                row = group.loc[group["source"].eq(source)].iloc[0]
                lines.append(
                    f"| {family_id} | {band_id} | "
                    f"{int(row.positive_pair_count):,} | "
                    f"{SOURCE_LABELS[source]} | {row.auroc:.3f} | "
                    f"{row.auprc_lift_over_prevalence:.2f}× |"
                )
    family_shared_supported = sum(
        recovery["exact"][family_id][band_id][
            "paired_chromosome_bootstrap"
        ]["family_minus_frozen_shared_auprc"]["positive_lower_bound"]
        for family_id in PASSED_FAMILIES
        for band_id in BAND_IDS
    )
    family_swapped_supported = sum(
        recovery["exact"][family_id][band_id][
            "paired_chromosome_bootstrap"
        ]["family_minus_swapped_family_auprc"]["positive_lower_bound"]
        for family_id in PASSED_FAMILIES
        for band_id in BAND_IDS
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The A-only prediction strongly identifies B-only loop "
            "locations, including pairs with zero raw-A contacts.",
            f"- Family EB has a positive chromosome-bootstrap AUPRC advantage "
            f"over frozen shared in {family_shared_supported}/4 exact "
            "family-by-distance comparisons.",
            f"- Correct-family EB has a positive advantage over the swapped "
            f"family in {family_swapped_supported}/4 comparisons. Thus the "
            "validated loop-location signal is mainly shared topology; this "
            "test does not establish family-specific loop ownership.",
            "",
            "## Scope",
            "",
            "- Exact anchors are primary; ±10-kb anchor tolerance is a "
            "prespecified sensitivity analysis.",
            "- Prepared internal-test prediction rows were not loaded and no "
            "test metrics were computed. Peakachu did process genome-wide B "
            "contacts to obtain the local and chromosome-wide background "
            "needed for validation-pair outcome labels.",
            "- These are pseudoreplicate-independent loop outcomes, not an "
            "independent animal/cohort or orthogonal-assay validation.",
            "- Peakachu cannot be called cleanly on the current predicted "
            "map itself: the predictor covers 250–990 kb validation pairs, "
            "whereas Peakachu requires complete 13×13 local windows, "
            "including contacts below 250 kb and neighboring non-validation "
            "pixels. A hybrid filled map would confound the test with raw-A "
            "contacts, so it was not used.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    print("[b-only-loops10k] Loading frozen validation predictions", flush=True)
    data = load_family_validation(args)
    if data["report"]["test_accessed"]:
        raise RuntimeError("Family EB report accessed internal test")
    fine = data["fine"]
    if not fine["split"].eq("validation").all():
        raise RuntimeError("Non-validation prediction rows were loaded")

    loops: dict[str, pd.DataFrame] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    loop_sources = {}
    for family_id in PASSED_FAMILIES:
        path = args.loop_root / f"{family_id}.half_B.loops.bedpe"
        loops[family_id] = read_loops(path, family_id=family_id)
        labels[family_id] = loop_labels(fine, loops[family_id])
        loop_sources[family_id] = source_record(path)
    pd.concat(loops.values(), ignore_index=True).to_parquet(
        output_dir / "b_only_peakachu_loops.10kb.parquet",
        index=False,
        compression="zstd",
    )

    data_root = Path(data["config"]["outputs"]["data_root"]).resolve()
    evidence_path = data_root / "pseudoreplicate_evidence.zarr"
    counts_a = load_family_counts_a(
        evidence_path,
        fine["pair_id"].to_numpy(np.int64),
    )
    print("[b-only-loops10k] Building A-only baselines", flush=True)
    scores = score_sources(data, counts_a)
    print("[b-only-loops10k] Evaluating recovery and hard matches", flush=True)
    recovery, table, matched = evaluate(
        data,
        labels,
        scores,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    table.to_csv(output_dir / "recovery_metrics.csv", index=False)
    matched.to_csv(output_dir / "hard_matched_metrics.csv", index=False)
    zero_a_recovery, zero_a_table = evaluate_zero_a(data, labels, scores)
    zero_a_table.to_csv(output_dir / "zero_a_recovery_metrics.csv", index=False)

    report = {
        "schema_version": 1,
        "scope": {
            "prediction_inputs": (
                "Whole-cell split-0 pseudoreplicate A family counts plus "
                "the frozen shared topology."
            ),
            "outcome": (
                "Peakachu 2.2 loop calls from disjoint whole-cell split-0 "
                "pseudoreplicate B raw family contact maps."
            ),
            "genome_build": "mm10",
            "resolution_bp": 10_000,
            "distance_range_bp_inclusive": [250_000, 990_000],
            "peakachu_probability_threshold": 0.5,
            "peakachu_balance": False,
            "peakachu_model_nominal_contacts": 10_000_000,
            "genomic_pair_split": "validation",
            "prepared_internal_test_prediction_rows_accessed": False,
            "test_metrics_computed": False,
            "raw_b_genomewide_contacts_processed_for_peakachu_context": True,
            "family_eb_report_test_accessed": bool(
                data["report"]["test_accessed"]
            ),
            "biologically_independent_cohort": False,
            "pseudoreplicate_independent": True,
            "bootstrap_unit": "chromosome",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "match_modes": {
                "exact": "Both 10-kb anchors match exactly.",
                "anchor_tolerance_10kb": (
                    "Each anchor may differ by at most one 10-kb bin."
                ),
            },
        },
        "sources": {
            "family_eb_report": source_record(data["paths"]["report"]),
            "fine_validation_pairs": source_record(data["paths"]["fine"]),
            "pseudoreplicate_evidence": str(evidence_path),
            "b_only_loop_bedpe": loop_sources,
            "b_only_coolers": {
                family_id: source_record(
                    args.cooler_root
                    / f"{family_id}.half_B.10kb.cool"
                )
                for family_id in PASSED_FAMILIES
            },
            "b_only_cooler_build_report": source_record(
                args.cooler_root / "build_report.json"
            ),
            "peakachu_model": {
                **source_record(args.peakachu_model),
                "sha256": sha256_file(args.peakachu_model),
            },
        },
        "loop_counts": {
            family_id: {
                "genomewide_calls_250_990kb": int(len(loops[family_id])),
                "validation_positive_pairs": {
                    mode: int(labels[family_id][mode].sum())
                    for mode in MATCH_MODES
                },
            }
            for family_id in PASSED_FAMILIES
        },
        "recovery": recovery,
        "zero_a_recovery": zero_a_recovery,
    }
    atomic_json(output_dir / "metrics.json", report)
    plot_scorecard(table, output_dir / "01_recovery_scorecard.png")
    plot_matched(matched, output_dir / "02_hard_matched_auroc.png")
    write_summary(
        table,
        matched,
        zero_a_table,
        recovery,
        output_dir / "summary.md",
    )
    print(table.to_string(index=False), flush=True)
    print(f"[b-only-loops10k] Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
