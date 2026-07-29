#!/usr/bin/env python3
"""Evaluate 10 kb family maps against published high-depth scHiCAR loops.

The published scDeepLUCIA catalog is a model-assisted call set derived from
the same 1.62-million-cell mouse frontal-cortex scHiCAR assay collection used
to construct the target counts. This diagnostic maps its 5 kb BEDPE anchors
to the canonical 10 kb pair universe and evaluates only the already-authorized
validation chromosomes. Internal test rows are never loaded.

Absolute recovery asks whether a family map ranks the published loops above
other valid pairs. Incremental recovery compares the family EB map with the
frozen pooled/shared topology at the same family depth. Family identity asks
whether the IT-minus-corticofugal score distinguishes high-depth loops with
greater normalized support in the corresponding published family catalogs.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "diagnostics"))

from evaluate_family_ep_recovery import (  # noqa: E402
    BAND_IDS,
    BAND_LABELS,
    FAMILY_IDS,
    PASSED_FAMILIES,
    PASSED_INDEX,
    TOP_FRACTIONS,
    load_family_validation,
    set_style,
)
from raw_contact_ranker.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    source_record,
)


CATALOG_MEMBERS = {
    "cortical_IT": (
        "L23IT_1",
        "L23IT_2",
        "L23IT_3",
        "L45IT",
        "L5IT",
        "L6IT",
    ),
    "corticofugal": ("L5ET", "L6CT", "PT"),
}
CATALOG_DOI = "10.5281/zenodo.18196031"
MATCH_MODES = ("exact", "anchor_tolerance_10kb")
BLUE = "#2F75B5"
ORANGE = "#E68632"
GREEN = "#2A9D8F"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
BLACK = "#222222"
SCOPE = (
    "Published high-depth scDeepLUCIA calls • 5 kb anchors collapsed to "
    "10 kb • validation chromosomes only • internal test untouched"
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
            / "data/external"
            / "schicar_scdeeplucia_loops_zenodo_18196031"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/raw_contact_ranker_10kb_v1"
            / "context_family_eb_grid_10kb_cpu/diagnostic_v1"
            / "scdeeplucia_loop_recovery_10kb_v1"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.975)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.008, SCOPE, color=GRAY, fontsize=7.2, ha="left")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[family-loops10k] Wrote {path.name}", flush=True)


def read_bedpe(path: Path, *, member: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=(
            "chrom_i",
            "start_i",
            "end_i",
            "chrom_j",
            "start_j",
            "end_j",
        ),
        dtype={
            "chrom_i": str,
            "start_i": np.int64,
            "end_i": np.int64,
            "chrom_j": str,
            "start_j": np.int64,
            "end_j": np.int64,
        },
    )
    if frame.empty:
        raise RuntimeError(f"{path} is empty")
    if not frame["chrom_i"].eq(frame["chrom_j"]).all():
        raise RuntimeError(f"{path} contains trans calls")
    for side in ("i", "j"):
        width = frame[f"end_{side}"] - frame[f"start_{side}"]
        if not width.eq(5_000).all():
            raise RuntimeError(f"{path} has non-5 kb anchors")
        if not frame[f"start_{side}"].mod(5_000).eq(0).all():
            raise RuntimeError(f"{path} has off-grid anchors")

    left = np.minimum(
        frame["start_i"].to_numpy(np.int64),
        frame["start_j"].to_numpy(np.int64),
    )
    right = np.maximum(
        frame["start_i"].to_numpy(np.int64),
        frame["start_j"].to_numpy(np.int64),
    )
    output = pd.DataFrame(
        {
            "chrom": frame["chrom_i"].astype(str),
            "bin_i": (left // 10_000) * 10_000,
            "bin_j": (right // 10_000) * 10_000,
            "member": member,
        }
    )
    distance = output["bin_j"] - output["bin_i"]
    output = output.loc[
        distance.ge(250_000) & distance.lt(1_000_000)
    ].drop_duplicates(["chrom", "bin_i", "bin_j", "member"])
    return output.reset_index(drop=True)


def load_catalogs(
    loop_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    rows = []
    source_stats: dict[str, dict[str, Any]] = {}
    source_records: dict[str, Any] = {}
    for family_id, members in CATALOG_MEMBERS.items():
        member_rows = []
        for member in members:
            path = loop_root / f"{member}.bedpe"
            if not path.is_file():
                raise FileNotFoundError(path)
            mapped = read_bedpe(path, member=member)
            member_rows.append(mapped)
            source_records[member] = source_record(path)
            source_stats[member] = {
                "source_rows": int(sum(1 for _ in path.open())),
                "mapped_unique_10kb_rows": int(len(mapped)),
                "sha256": sha256_file(path),
            }
        combined = pd.concat(member_rows, ignore_index=True)
        grouped = (
            combined.groupby(
                ["chrom", "bin_i", "bin_j"],
                observed=True,
                sort=True,
            )["member"]
            .agg(
                member_support="nunique",
                source_members=lambda values: ",".join(
                    sorted(set(map(str, values)))
                ),
            )
            .reset_index()
        )
        grouped["family_id"] = family_id
        grouped["family_member_count"] = len(members)
        grouped["member_fraction"] = (
            grouped["member_support"] / len(members)
        )
        rows.append(grouped)
    catalog = pd.concat(rows, ignore_index=True)
    catalog = catalog[
        [
            "family_id",
            "chrom",
            "bin_i",
            "bin_j",
            "member_support",
            "family_member_count",
            "member_fraction",
            "source_members",
        ]
    ].sort_values(
        ["family_id", "chrom", "bin_i", "bin_j"], kind="stable"
    )
    return catalog.reset_index(drop=True), source_records, source_stats


def support_lookup(
    catalog: pd.DataFrame,
    *,
    tolerance_bins: int,
) -> dict[tuple[str, int, int], float]:
    lookup: dict[tuple[str, int, int], float] = {}
    for row in catalog.itertuples(index=False):
        for delta_i in range(-tolerance_bins, tolerance_bins + 1):
            for delta_j in range(-tolerance_bins, tolerance_bins + 1):
                bin_i = int(row.bin_i) + 10_000 * delta_i
                bin_j = int(row.bin_j) + 10_000 * delta_j
                if bin_i < 0 or bin_j <= bin_i:
                    continue
                key = (str(row.chrom), bin_i, bin_j)
                lookup[key] = max(
                    lookup.get(key, 0.0), float(row.member_fraction)
                )
    return lookup


def map_support_to_pairs(
    fine: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    pair_keys = list(
        zip(
            fine["chrom"].astype(str),
            fine["bin_i"].to_numpy(np.int64),
            fine["bin_j"].to_numpy(np.int64),
            strict=True,
        )
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for mode, tolerance in (
        ("exact", 0),
        ("anchor_tolerance_10kb", 1),
    ):
        output[mode] = {}
        for family_id in PASSED_FAMILIES:
            selected = catalog.loc[catalog["family_id"].eq(family_id)]
            lookup = support_lookup(selected, tolerance_bins=tolerance)
            output[mode][family_id] = np.fromiter(
                (lookup.get(key, 0.0) for key in pair_keys),
                dtype=np.float64,
                count=len(pair_keys),
            )
    return output


def classification_metrics(
    labels: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, bool)
    score = np.asarray(score, np.float64)
    if not labels.any() or labels.all():
        raise RuntimeError("Loop labels are degenerate")
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


def topk_metrics(
    labels: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, float | int]]:
    labels = np.asarray(labels, bool)
    order = np.argsort(np.asarray(score, np.float64), kind="stable")[::-1]
    prevalence = float(labels.mean())
    output = []
    for fraction in TOP_FRACTIONS:
        selected_count = max(1, int(math.ceil(fraction * len(order))))
        selected = labels[order[:selected_count]]
        realized = selected_count / len(order)
        precision = float(selected.mean())
        recall = float(selected.sum() / labels.sum())
        output.append(
            {
                "requested_top_fraction": float(fraction),
                "selected_pair_count": int(selected_count),
                "selected_pair_fraction": float(realized),
                "positive_precision": precision,
                "positive_recall": recall,
                "enrichment_over_prevalence": float(
                    precision / prevalence
                ),
            }
        )
    return output


def bootstrap_macro_difference(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    chromosome: np.ndarray,
    *,
    metric: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, bool)
    chromosome = np.asarray(chromosome).astype(str)
    per_chromosome = []
    for chrom in sorted(np.unique(chromosome)):
        selected = chromosome == chrom
        current = labels[selected]
        if not current.any() or current.all():
            continue
        if metric == "auprc":
            value_a = average_precision_score(current, score_a[selected])
            value_b = average_precision_score(current, score_b[selected])
        elif metric == "auroc":
            value_a = roc_auc_score(current, score_a[selected])
            value_b = roc_auc_score(current, score_b[selected])
        else:
            raise ValueError(f"Unknown bootstrap metric: {metric}")
        per_chromosome.append(
            {
                "chrom": chrom,
                "source_a": float(value_a),
                "source_b": float(value_b),
                "difference": float(value_a - value_b),
            }
        )
    if len(per_chromosome) < 2:
        raise RuntimeError("Fewer than two evaluable chromosomes")
    differences = np.asarray(
        [row["difference"] for row in per_chromosome], np.float64
    )
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(
            0,
            len(differences),
            size=(replicates, len(differences)),
        )
    ].mean(axis=1)
    alpha = 1.0 - confidence_level
    return {
        "metric": metric,
        "macro_difference": float(differences.mean()),
        "chromosomes": int(len(differences)),
        "per_chromosome": per_chromosome,
        "bootstrap": {
            "replicates": int(replicates),
            "lower": float(np.quantile(samples, alpha)),
            "median": float(np.median(samples)),
            "upper": float(np.quantile(samples, confidence_level)),
        },
        "positive_lower_bound": bool(
            differences.mean() > 0
            and np.quantile(samples, alpha) > 0
        ),
    }


def recovery_metrics(
    data: dict[str, Any],
    support: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    chromosome = fine["chrom"].astype(str).to_numpy()
    output: dict[str, Any] = {}
    for mode_index, mode in enumerate(MATCH_MODES):
        output[mode] = {}
        for family_offset, family in enumerate(PASSED_INDEX):
            family_id = FAMILY_IDS[family]
            swapped_family = PASSED_INDEX[1 - family_offset]
            labels_all = support[mode][family_id] > 0
            output[mode][family_id] = {}
            for band, band_id in enumerate(BAND_IDS):
                selected = band_index == band
                labels = labels_all[selected]
                family_score = data["family_prediction_b"][family, selected]
                shared_score = data["shared_prediction_b"][family, selected]
                swapped_score = data["family_prediction_b"][
                    swapped_family, selected
                ]
                sources = {
                    "family_eb": family_score,
                    "frozen_shared": shared_score,
                    "swapped_family": swapped_score,
                }
                classification = {
                    name: classification_metrics(labels, score)
                    for name, score in sources.items()
                }
                topk = {
                    name: topk_metrics(labels, score)
                    for name, score in sources.items()
                }
                family_shared = bootstrap_macro_difference(
                    labels,
                    family_score,
                    shared_score,
                    chromosome[selected],
                    metric="auprc",
                    replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    seed=(
                        seed
                        + 1_000 * mode_index
                        + 100 * family_offset
                        + 10 * band
                    ),
                )
                family_swapped = bootstrap_macro_difference(
                    labels,
                    family_score,
                    swapped_score,
                    chromosome[selected],
                    metric="auprc",
                    replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    seed=(
                        seed
                        + 10_000
                        + 1_000 * mode_index
                        + 100 * family_offset
                        + 10 * band
                    ),
                )
                family_result = classification["family_eb"]
                shared_result = classification["frozen_shared"]
                output[mode][family_id][band_id] = {
                    "classification": classification,
                    "topk": topk,
                    "paired_chromosome_bootstrap": {
                        "family_minus_shared_auprc": family_shared,
                        "family_minus_swapped_auprc": family_swapped,
                    },
                    "absolute_recovery_supported": bool(
                        family_result["auprc"]
                        > family_result["positive_prevalence"]
                        and topk["family_eb"][0][
                            "enrichment_over_prevalence"
                        ]
                        > 1
                    ),
                    "incremental_over_shared_supported": bool(
                        family_result["auprc"] > shared_result["auprc"]
                        and family_shared["positive_lower_bound"]
                    ),
                }
    return output


def identity_metrics(
    data: dict[str, Any],
    support: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    chromosome = fine["chrom"].astype(str).to_numpy()
    delta = np.log(
        np.clip(
            data["family_probability"][PASSED_INDEX[0]],
            1e-300,
            None,
        )
    ) - np.log(
        np.clip(
            data["family_probability"][PASSED_INDEX[1]],
            1e-300,
            None,
        )
    )
    output: dict[str, Any] = {}
    for mode_index, mode in enumerate(MATCH_MODES):
        it_support = support[mode]["cortical_IT"]
        cf_support = support[mode]["corticofugal"]
        eligible = (it_support != cf_support) & (
            (it_support > 0) | (cf_support > 0)
        )
        labels_all = it_support > cf_support
        exact_exclusive = (it_support > 0) ^ (cf_support > 0)
        output[mode] = {
            "definition": (
                "Among pairs supported by either catalog with unequal "
                "member-normalized support, positive means cortical IT "
                "support exceeds corticofugal support."
            ),
            "strict_exclusive_pair_count": int(exact_exclusive.sum()),
            "bands": {},
        }
        for band, band_id in enumerate(BAND_IDS):
            selected = eligible & (band_index == band)
            labels = labels_all[selected]
            scores = delta[selected]
            if not labels.any() or labels.all():
                raise RuntimeError(f"Degenerate identity labels for {band_id}")
            classification = classification_metrics(labels, scores)
            direction = scores > 0
            bootstrap = bootstrap_macro_difference(
                labels,
                scores,
                np.zeros_like(scores),
                chromosome[selected],
                metric="auroc",
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 20_000 + 1_000 * mode_index + 10 * band,
            )
            # Subtract the constant-score AUROC explicitly to expose the
            # biologically relevant distance from chance.
            bootstrap["chance_auroc"] = 0.5
            output[mode]["bands"][band_id] = {
                "classification": classification,
                "it_positive_pair_count": int(labels.sum()),
                "corticofugal_positive_pair_count": int((~labels).sum()),
                "direction_accuracy": float(np.mean(direction == labels)),
                "balanced_direction_accuracy": float(
                    balanced_accuracy_score(labels, direction)
                ),
                "it_minus_corticofugal_probability_score": {
                    "minimum": float(scores.min()),
                    "median": float(np.median(scores)),
                    "maximum": float(scores.max()),
                },
                "auroc_minus_chance_chromosome_bootstrap": bootstrap,
                "identity_supported": bool(
                    classification["auroc"] > 0.5
                    and bootstrap["positive_lower_bound"]
                ),
            }
    return output


def flatten_recovery(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode, families in metrics.items():
        for family_id, bands in families.items():
            for band_id, result in bands.items():
                row: dict[str, Any] = {
                    "match_mode": mode,
                    "family_id": family_id,
                    "band_id": band_id,
                }
                for source in (
                    "family_eb",
                    "frozen_shared",
                    "swapped_family",
                ):
                    values = result["classification"][source]
                    for metric in (
                        "positive_pair_count",
                        "positive_prevalence",
                        "auprc",
                        "auprc_lift_over_prevalence",
                        "auroc",
                    ):
                        row[f"{source}_{metric}"] = values[metric]
                    row[f"{source}_top1_enrichment"] = result["topk"][
                        source
                    ][0]["enrichment_over_prevalence"]
                paired = result["paired_chromosome_bootstrap"]
                row["family_minus_shared_macro_auprc"] = paired[
                    "family_minus_shared_auprc"
                ]["macro_difference"]
                row["family_minus_shared_ci_lower"] = paired[
                    "family_minus_shared_auprc"
                ]["bootstrap"]["lower"]
                row["family_minus_shared_ci_upper"] = paired[
                    "family_minus_shared_auprc"
                ]["bootstrap"]["upper"]
                row["absolute_recovery_supported"] = result[
                    "absolute_recovery_supported"
                ]
                row["incremental_over_shared_supported"] = result[
                    "incremental_over_shared_supported"
                ]
                rows.append(row)
    return pd.DataFrame(rows)


def flatten_identity(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode, result in metrics.items():
        for band_id, values in result["bands"].items():
            classification = values["classification"]
            bootstrap = values[
                "auroc_minus_chance_chromosome_bootstrap"
            ]
            rows.append(
                {
                    "match_mode": mode,
                    "band_id": band_id,
                    "eligible_pairs": classification[
                        "candidate_pair_count"
                    ],
                    "it_positive_pairs": values[
                        "it_positive_pair_count"
                    ],
                    "corticofugal_positive_pairs": values[
                        "corticofugal_positive_pair_count"
                    ],
                    "auroc": classification["auroc"],
                    "auprc": classification["auprc"],
                    "direction_accuracy": values["direction_accuracy"],
                    "balanced_direction_accuracy": values[
                        "balanced_direction_accuracy"
                    ],
                    "macro_auroc_minus_chance": bootstrap[
                        "macro_difference"
                    ],
                    "bootstrap_ci_lower": bootstrap["bootstrap"]["lower"],
                    "bootstrap_ci_upper": bootstrap["bootstrap"]["upper"],
                    "identity_supported": values["identity_supported"],
                }
            )
    return pd.DataFrame(rows)


def plot_scorecard(table: pd.DataFrame, *, output: Path) -> None:
    frame = table.loc[table["match_mode"].eq("exact")].copy()
    frame["label"] = (
        frame["family_id"].map(
            {
                "cortical_IT": "Cortical IT",
                "corticofugal": "Corticofugal",
            }
        )
        + "\n"
        + frame["band_id"].map(
            dict(zip(BAND_IDS, BAND_LABELS, strict=True))
        )
    )
    x = np.arange(len(frame))
    width = 0.26
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9))
    axes[0].bar(
        x - width,
        frame["family_eb_auprc_lift_over_prevalence"],
        width,
        color=BLUE,
        label="Family EB",
    )
    axes[0].bar(
        x,
        frame["frozen_shared_auprc_lift_over_prevalence"],
        width,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        label="Frozen pooled/shared",
    )
    axes[0].bar(
        x + width,
        frame["swapped_family_auprc_lift_over_prevalence"],
        width,
        color=ORANGE,
        label="Swapped family",
    )
    axes[0].axhline(1, color=BLACK, linestyle="--", linewidth=1)
    axes[0].set_title("Published-loop classification")
    axes[0].set_ylabel("AUPRC / loop prevalence")
    axes[0].legend(fontsize=7.5)

    axes[1].bar(
        x - width / 2,
        frame["family_eb_top1_enrichment"],
        width,
        color=BLUE,
        label="Family EB",
    )
    axes[1].bar(
        x + width / 2,
        frame["frozen_shared_top1_enrichment"],
        width,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        label="Frozen pooled/shared",
    )
    axes[1].axhline(1, color=BLACK, linestyle="--", linewidth=1)
    axes[1].set_title("Top 1% predicted pairs")
    axes[1].set_ylabel("Loop enrichment over prevalence")

    point = frame["family_minus_shared_macro_auprc"].to_numpy()
    lower = frame["family_minus_shared_ci_lower"].to_numpy()
    upper = frame["family_minus_shared_ci_upper"].to_numpy()
    axes[2].errorbar(
        x,
        point,
        yerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        color=GREEN,
        ecolor=GREEN,
        capsize=4,
        markersize=7,
    )
    axes[2].axhline(0, color=BLACK, linestyle="--", linewidth=1)
    axes[2].set_title("Family EB gain over shared")
    axes[2].set_ylabel("Macro chromosome AUPRC difference")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(frame["label"], rotation=20, ha="right")
    fig.suptitle(
        "10 kb recovery of published high-depth scHiCAR loop calls",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(bottom=0.26, wspace=0.32)
    save_figure(fig, output)


def plot_precision_recall(
    data: dict[str, Any],
    support: dict[str, dict[str, np.ndarray]],
    *,
    output: Path,
) -> None:
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for family_offset, family in enumerate(PASSED_INDEX):
        family_id = FAMILY_IDS[family]
        labels_all = support["exact"][family_id] > 0
        swapped_family = PASSED_INDEX[1 - family_offset]
        for band, band_id in enumerate(BAND_IDS):
            selected = band_index == band
            labels = labels_all[selected]
            sources = (
                (
                    data["family_prediction_b"][family, selected],
                    BLUE,
                    "Family EB",
                ),
                (
                    data["shared_prediction_b"][family, selected],
                    GRAY,
                    "Frozen pooled/shared",
                ),
                (
                    data["family_prediction_b"][
                        swapped_family, selected
                    ],
                    ORANGE,
                    "Swapped family",
                ),
            )
            axis = axes[family_offset, band]
            for score, color, label in sources:
                precision, recall, _ = precision_recall_curve(labels, score)
                axis.plot(
                    recall,
                    precision,
                    color=color,
                    linewidth=2,
                    label=label,
                )
            axis.axhline(
                labels.mean(),
                color=BLACK,
                linestyle="--",
                linewidth=1,
                label="Prevalence",
            )
            axis.set_title(
                f"{family_id.replace('_', ' ').title()} • "
                f"{BAND_LABELS[band]}"
            )
            axis.set_xlabel("Recall of published 10 kb loop pairs")
            axis.set_ylabel("Precision")
            axis.set_xlim(0, 1)
            axis.set_ylim(bottom=0)
            axis.legend(fontsize=7.5)
    fig.suptitle(
        "Published-loop precision–recall on validation chromosomes",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.10, hspace=0.34, wspace=0.25)
    save_figure(fig, output)


def plot_identity(table: pd.DataFrame, *, output: Path) -> None:
    labels = []
    x = []
    y = []
    lower = []
    upper = []
    colors = []
    for mode_index, mode in enumerate(MATCH_MODES):
        for band_index, band_id in enumerate(BAND_IDS):
            row = table.loc[
                table["match_mode"].eq(mode)
                & table["band_id"].eq(band_id)
            ].iloc[0]
            x.append(2 * mode_index + band_index)
            y.append(row["macro_auroc_minus_chance"])
            lower.append(row["bootstrap_ci_lower"])
            upper.append(row["bootstrap_ci_upper"])
            labels.append(
                ("Exact" if mode == "exact" else "±10 kb")
                + "\n"
                + BAND_LABELS[band_index]
            )
            colors.append(BLUE if band_index == 0 else GREEN)
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].bar(x_array, y_array, color=colors, width=0.72)
    axes[0].errorbar(
        x_array,
        y_array,
        yerr=np.vstack(
            [y_array - np.asarray(lower), np.asarray(upper) - y_array]
        ),
        fmt="none",
        ecolor=BLACK,
        capsize=4,
    )
    axes[0].axhline(0, color=BLACK, linestyle="--", linewidth=1)
    axes[0].set_xticks(x_array, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Macro AUROC − 0.5")
    axes[0].set_title("IT versus corticofugal loop identity")

    exact = table.loc[table["match_mode"].eq("exact")].copy()
    positions = np.arange(len(exact))
    axes[1].bar(
        positions - 0.18,
        exact["direction_accuracy"],
        0.36,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        label="Raw direction accuracy",
    )
    axes[1].bar(
        positions + 0.18,
        exact["balanced_direction_accuracy"],
        0.36,
        color=ORANGE,
        label="Balanced direction accuracy",
    )
    axes[1].axhline(0.5, color=BLACK, linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, BAND_LABELS)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Sign of IT-minus-corticofugal score")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Family identity is tested independently of absolute loop recovery",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.30)
    save_figure(fig, output)


def plot_topk(
    recovery: dict[str, Any],
    *,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), sharex=True)
    for family_index, family_id in enumerate(PASSED_FAMILIES):
        for band_index, band_id in enumerate(BAND_IDS):
            axis = axes[family_index, band_index]
            result = recovery["exact"][family_id][band_id]
            for source, color, label in (
                ("family_eb", BLUE, "Family EB"),
                ("frozen_shared", GRAY, "Frozen pooled/shared"),
                ("swapped_family", ORANGE, "Swapped family"),
            ):
                records = result["topk"][source]
                axis.plot(
                    [100 * row["selected_pair_fraction"] for row in records],
                    [
                        row["enrichment_over_prevalence"]
                        for row in records
                    ],
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=label,
                )
            axis.axhline(1, color=BLACK, linewidth=1, linestyle="--")
            axis.set_title(
                f"{family_id.replace('_', ' ').title()} • "
                f"{BAND_LABELS[band_index]}"
            )
            axis.set_xlabel("Top-scoring valid pairs (%)")
            axis.set_ylabel("Published-loop enrichment")
            axis.legend(fontsize=7.5)
    fig.suptitle(
        "Published loops concentrate among top 10 kb predictions",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.10, hspace=0.34, wspace=0.25)
    save_figure(fig, output)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if not 0.5 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie between 0.5 and 1")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    print("[family-loops10k] Loading safeguarded validation maps", flush=True)
    data = load_family_validation(args)
    if data["report"]["test_accessed"]:
        raise RuntimeError("Internal test split was accessed")
    if not data["fine"]["split"].eq("validation").all():
        raise RuntimeError("Non-validation pairs entered loop evaluation")

    print("[family-loops10k] Loading published loop catalogs", flush=True)
    loop_root = args.loop_root.expanduser().resolve()
    catalog, source_records, source_stats = load_catalogs(loop_root)
    catalog.to_parquet(
        output_dir / "published_loops.collapsed_10kb.parquet",
        index=False,
        compression="zstd",
    )
    support = map_support_to_pairs(data["fine"], catalog)
    print("[family-loops10k] Computing recovery metrics", flush=True)
    recovery = recovery_metrics(
        data,
        support,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    identity = identity_metrics(
        data,
        support,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    recovery_table = flatten_recovery(recovery)
    identity_table = flatten_identity(identity)
    recovery_table.to_csv(output_dir / "recovery_metrics.csv", index=False)
    identity_table.to_csv(output_dir / "identity_metrics.csv", index=False)

    validation_catalog_counts = {
        mode: {
            family_id: int((support[mode][family_id] > 0).sum())
            for family_id in PASSED_FAMILIES
        }
        for mode in MATCH_MODES
    }
    report = {
        "schema_version": 1,
        "scope": {
            "prediction": (
                "Family EB expected 10 kb scHiCAR contact map fixed from "
                "validation pseudoreplicate A and frozen pooled "
                "AlphaGenome topology."
            ),
            "external_outcome": (
                "Published scDeepLUCIA 5 kb loop calls from the same "
                "high-depth 1.62-million-cell mouse frontal-cortex assay "
                "collection used to construct the target counts."
            ),
            "external_outcome_caveat": (
                "scDeepLUCIA calls are model-assisted experimental-data "
                "calls from the same underlying scHiCAR samples, not an "
                "independent cohort or assay-independent physical-loop "
                "ground truth."
            ),
            "independent_biological_validation": False,
            "shared_source_accessions": (
                "GSM8260434–GSM8260473 within GSE305889/GSE267126"
            ),
            "genome_build": "mm10",
            "source_bin_size_bp": 5_000,
            "evaluation_bin_size_bp": 10_000,
            "collapse_rule": (
                "Each 5 kb anchor start is floored to its containing "
                "10 kb bin; duplicate collapsed pairs are removed within "
                "each source cell type."
            ),
            "match_modes": {
                "exact": "Both collapsed 10 kb anchors match exactly.",
                "anchor_tolerance_10kb": (
                    "Each anchor may differ by at most one 10 kb bin."
                ),
            },
            "distance_range_bp": [250_000, 1_000_000],
            "distance_maximum_exclusive": True,
            "families": {
                key: list(value) for key, value in CATALOG_MEMBERS.items()
            },
            "genomic_split": "validation",
            "test_accessed": False,
            "bootstrap_unit": "chromosome",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_interval_quantiles": [
                float(1 - args.confidence_level),
                float(args.confidence_level),
            ],
            "primary_interpretation": (
                "10 kb recovery of a published high-depth loop catalog. "
                "This does not establish 5 kb localization, causal "
                "enhancer assignment, independent-cohort generalization, "
                "or assay-independent loop truth."
            ),
        },
        "external_catalog": {
            "doi": CATALOG_DOI,
            "license": "CC-BY-4.0",
            "source_files": source_records,
            "source_file_stats": source_stats,
            "collapsed_10kb_rows": {
                family_id: int(
                    catalog["family_id"].eq(family_id).sum()
                )
                for family_id in PASSED_FAMILIES
            },
            "validation_positive_pairs": validation_catalog_counts,
        },
        "internal_sources": {
            "family_eb_report": source_record(data["paths"]["report"]),
            "fine_pair_table": source_record(data["paths"]["fine"]),
        },
        "recovery": recovery,
        "family_identity": identity,
    }
    atomic_json(output_dir / "metrics.json", report)

    plot_scorecard(
        recovery_table,
        output=output_dir / "01_loop_recovery_scorecard.png",
    )
    plot_precision_recall(
        data,
        support,
        output=output_dir / "02_loop_precision_recall.png",
    )
    plot_identity(
        identity_table,
        output=output_dir / "03_family_identity.png",
    )
    plot_topk(
        recovery,
        output=output_dir / "04_loop_topk_enrichment.png",
    )
    print(recovery_table.to_string(index=False), flush=True)
    print(identity_table.to_string(index=False), flush=True)
    print(f"[family-loops10k] Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
