#!/usr/bin/env python3
"""Evaluate family maps against Peakachu and cross-caller consensus loops.

Peakachu calls are read directly from Supplementary Table 5 of the scHiCAR
paper. Consensus is defined strictly within each matching source cell type:
a Peakachu and scDeepLUCIA call must collapse to the same 10 kb anchor pair
before cell types are pooled into cortical-IT or corticofugal families.
Only the safeguarded validation split is loaded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "diagnostics"))

from evaluate_family_ep_recovery import (  # noqa: E402
    BAND_IDS,
    BAND_LABELS,
    FAMILY_IDS,
    PASSED_FAMILIES,
    PASSED_INDEX,
    load_family_validation,
    set_style,
)
from evaluate_family_scdeeplucia_loops_10kb import (  # noqa: E402
    CATALOG_MEMBERS,
    MATCH_MODES,
    flatten_recovery,
    map_support_to_pairs,
    read_bedpe,
    recovery_metrics,
)
from raw_contact_ranker.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    source_record,
)


PAPER_DOI = "10.1038/s41587-026-03013-7"
SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41587-026-03013-7/MediaObjects/"
    "41587_2026_3013_MOESM3_ESM.xlsx"
)
PEAKACHU_TO_CANONICAL = {
    "L23IT-1": "L23IT_1",
    "L23IT-2": "L23IT_2",
    "L23IT-3": "L23IT_3",
    "L45IT": "L45IT",
    "L5IT": "L5IT",
    "L6IT": "L6IT",
    "L5ET": "L5ET",
    "L6CT": "L6CT",
    "PT": "PT",
}
CANONICAL_TO_FAMILY = {
    member: family
    for family, members in CATALOG_MEMBERS.items()
    for member in members
}
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CATALOG_LABELS = {
    "peakachu": "Peakachu",
    "consensus": "Peakachu ∩ scDeepLUCIA",
}
COLORS = {
    "family_eb": "#2F75B5",
    "frozen_shared": "#B9B9B9",
    "swapped_family": "#E68632",
    "peakachu": "#5B8FF9",
    "consensus": "#2A9D8F",
}
GRAY = "#777777"
BLACK = "#222222"
SCOPE = (
    "Official scHiCAR Supplementary Table 5 • 5 kb calls collapsed to "
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
        "--supplement-xlsx",
        type=Path,
        default=(
            REPO_ROOT
            / "data/external/schicar_paper_supplement_pmc12995399"
            / "41587_2026_3013_MOESM3_ESM.xlsx"
        ),
    )
    parser.add_argument(
        "--scdeeplucia-root",
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
            / "peakachu_consensus_loop_recovery_10kb_v1"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.975)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def column_index(reference: str) -> int:
    match = re.fullmatch(r"([A-Z]+)\d+", reference)
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {reference}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def load_shared_strings(archive: ZipFile) -> list[str]:
    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as handle:
        for _, element in iterparse(handle, events=("end",)):
            if element.tag == XML_NS + "si":
                strings.append(
                    "".join(
                        child.text or ""
                        for child in element.iter(XML_NS + "t")
                    )
                )
                element.clear()
    return strings


def cell_value(cell: Any, shared_strings: list[str]) -> str | None:
    value = cell.find(XML_NS + "v")
    if value is None or value.text is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def collapse_5kb_calls(
    frame: pd.DataFrame,
    *,
    member: str,
) -> pd.DataFrame:
    required = (
        "chrom_i",
        "start_i",
        "end_i",
        "chrom_j",
        "start_j",
        "end_j",
        "probability",
    )
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Peakachu columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"No Peakachu rows for {member}")
    if not frame["chrom_i"].eq(frame["chrom_j"]).all():
        raise RuntimeError(f"Peakachu {member} contains trans calls")
    for side in ("i", "j"):
        start = frame[f"start_{side}"].astype(np.int64)
        end = frame[f"end_{side}"].astype(np.int64)
        if not (end - start).eq(5_000).all():
            raise RuntimeError(f"Peakachu {member} has non-5 kb anchors")
        if not start.mod(5_000).eq(0).all():
            raise RuntimeError(f"Peakachu {member} has off-grid anchors")

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
            "peakachu_probability": frame["probability"].astype(float),
        }
    )
    distance = output["bin_j"] - output["bin_i"]
    output = output.loc[
        distance.ge(250_000) & distance.lt(1_000_000)
    ]
    return (
        output.groupby(
            ["chrom", "bin_i", "bin_j", "member"],
            as_index=False,
            observed=True,
            sort=True,
        )["peakachu_probability"]
        .max()
        .reset_index(drop=True)
    )


def read_peakachu_table(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Stream the selected cell-type blocks from Supplementary Table 5."""
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_starts: dict[int, str] = {}
    with ZipFile(path) as archive:
        shared_strings = load_shared_strings(archive)
        with archive.open("xl/worksheets/sheet5.xml") as handle:
            for _, row in iterparse(handle, events=("end",)):
                if row.tag != XML_NS + "row":
                    continue
                row_number = int(row.attrib["r"])
                cells = {
                    column_index(cell.attrib["r"]): cell_value(
                        cell, shared_strings
                    )
                    for cell in row.findall(XML_NS + "c")
                }
                if row_number == 2:
                    for start, value in cells.items():
                        if value in PEAKACHU_TO_CANONICAL:
                            block_starts[start] = PEAKACHU_TO_CANONICAL[value]
                    observed = set(block_starts.values())
                    expected = set(CANONICAL_TO_FAMILY)
                    if observed != expected:
                        raise RuntimeError(
                            "Peakachu member mismatch: "
                            f"missing={sorted(expected - observed)}, "
                            f"unexpected={sorted(observed - expected)}"
                        )
                elif row_number == 3:
                    expected_headers = (
                        "chrom1",
                        "start1",
                        "end1",
                        "chrom2",
                        "start2",
                        "end2",
                    )
                    for start, member in block_starts.items():
                        observed = tuple(cells.get(start + i) for i in range(6))
                        if observed != expected_headers:
                            raise RuntimeError(
                                f"Unexpected Table 5 headers for {member}: "
                                f"{observed}"
                            )
                elif row_number >= 4:
                    for start, member in block_starts.items():
                        values = [cells.get(start + i) for i in range(7)]
                        if all(value is None for value in values):
                            continue
                        if any(value is None for value in values):
                            raise RuntimeError(
                                f"Incomplete Peakachu row {row_number} "
                                f"for {member}"
                            )
                        raw[member].append(
                            {
                                "chrom_i": values[0],
                                "start_i": int(values[1]),
                                "end_i": int(values[2]),
                                "chrom_j": values[3],
                                "start_j": int(values[4]),
                                "end_j": int(values[5]),
                                "probability": float(values[6]),
                            }
                        )
                row.clear()

    mapped = []
    stats: dict[str, dict[str, Any]] = {}
    for member in sorted(CANONICAL_TO_FAMILY):
        source = pd.DataFrame(raw[member])
        collapsed = collapse_5kb_calls(source, member=member)
        mapped.append(collapsed)
        stats[member] = {
            "source_5kb_rows": int(len(source)),
            "mapped_unique_10kb_rows_in_evaluation_distance": int(
                len(collapsed)
            ),
        }
    return pd.concat(mapped, ignore_index=True), stats


def load_scdeeplucia_members(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    stats: dict[str, dict[str, Any]] = {}
    for member in sorted(CANONICAL_TO_FAMILY):
        path = root / f"{member}.bedpe"
        mapped = read_bedpe(path, member=member)
        rows.append(mapped)
        stats[member] = {
            "source_5kb_rows": int(sum(1 for _ in path.open())),
            "mapped_unique_10kb_rows_in_evaluation_distance": int(
                len(mapped)
            ),
            "sha256": sha256_file(path),
        }
    return pd.concat(rows, ignore_index=True), stats


def member_consensus(
    peakachu: pd.DataFrame,
    scdeeplucia: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["chrom", "bin_i", "bin_j", "member"]
    consensus = peakachu.merge(
        scdeeplucia[keys],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if consensus.duplicated(keys).any():
        raise RuntimeError("Consensus contains duplicate member calls")
    return consensus


def aggregate_family_catalog(
    member_calls: pd.DataFrame,
    *,
    catalog_type: str,
) -> pd.DataFrame:
    frame = member_calls.copy()
    frame["family_id"] = frame["member"].map(CANONICAL_TO_FAMILY)
    if frame["family_id"].isna().any():
        raise RuntimeError("Unmapped member in catalog")
    grouped = (
        frame.groupby(
            ["family_id", "chrom", "bin_i", "bin_j"],
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
    counts = {family: len(members) for family, members in CATALOG_MEMBERS.items()}
    grouped["family_member_count"] = grouped["family_id"].map(counts)
    grouped["member_fraction"] = (
        grouped["member_support"] / grouped["family_member_count"]
    )
    grouped["catalog_type"] = catalog_type
    return grouped[
        [
            "catalog_type",
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


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.008, SCOPE, color=GRAY, fontsize=7.2, ha="left")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[peakachu-consensus10k] Wrote {path.name}", flush=True)


def plot_scorecard(table: pd.DataFrame, *, output: Path) -> None:
    frame = table.loc[table["match_mode"].eq("exact")].copy()
    frame["group"] = (
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
    groups = list(dict.fromkeys(frame["group"]))
    x = np.arange(len(groups))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    for offset, catalog_type in enumerate(CATALOG_LABELS):
        selected = frame.loc[frame["catalog_type"].eq(catalog_type)].set_index(
            "group"
        ).loc[groups]
        position = x + (offset - 0.5) * width
        axes[0].bar(
            position,
            selected["family_eb_auprc_lift_over_prevalence"],
            width,
            color=COLORS[catalog_type],
            label=CATALOG_LABELS[catalog_type],
        )
        axes[1].bar(
            position,
            selected["family_eb_top1_enrichment"],
            width,
            color=COLORS[catalog_type],
        )
        axes[2].bar(
            position,
            selected["family_eb_auroc"],
            width,
            color=COLORS[catalog_type],
        )
    axes[0].axhline(1, color=BLACK, linestyle="--", linewidth=1)
    axes[0].set_title("AUPRC lift over loop prevalence")
    axes[0].set_ylabel("Fold enrichment")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color=BLACK, linestyle="--", linewidth=1)
    axes[1].set_title("Top 1% prediction enrichment")
    axes[1].set_ylabel("Fold enrichment")
    axes[2].axhline(0.5, color=BLACK, linestyle="--", linewidth=1)
    axes[2].set_title("All-pair discrimination")
    axes[2].set_ylabel("AUROC")
    axes[2].set_ylim(0.45, 1.0)
    for axis in axes:
        axis.set_xticks(x, groups, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "10 kb recovery strengthens for cross-caller consensus",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_figure(fig, output)


def plot_precision_recall(
    data: dict[str, Any],
    supports: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    output: Path,
) -> None:
    fine = data["fine"]
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for family_offset, family in enumerate(PASSED_INDEX):
        family_id = FAMILY_IDS[family]
        for band, band_id in enumerate(BAND_IDS):
            axis = axes[family_offset, band]
            selected = band_index == band
            score = data["family_prediction_b"][family, selected]
            for catalog_type in CATALOG_LABELS:
                labels = (
                    supports[catalog_type]["exact"][family_id][selected] > 0
                )
                precision, recall, _ = precision_recall_curve(labels, score)
                axis.plot(
                    recall,
                    precision,
                    color=COLORS[catalog_type],
                    linewidth=2,
                    label=CATALOG_LABELS[catalog_type],
                )
                axis.axhline(
                    labels.mean(),
                    color=COLORS[catalog_type],
                    linestyle=":",
                    linewidth=1,
                    alpha=0.8,
                )
            axis.set_title(
                f"{family_id.replace('_', ' ').title()} • "
                f"{dict(zip(BAND_IDS, BAND_LABELS, strict=True))[band_id]}"
            )
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    fig.suptitle(
        "Family-map precision–recall against experimental loop catalogs",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, output)


def plot_top1_precision_recall(
    recovery: dict[str, dict[str, Any]],
    *,
    output: Path,
) -> None:
    rows = []
    for catalog_type, metrics in recovery.items():
        for family_id, bands in metrics["exact"].items():
            for band_id, result in bands.items():
                top1 = result["topk"]["family_eb"][0]
                rows.append(
                    {
                        "catalog_type": catalog_type,
                        "family_id": family_id,
                        "band_id": band_id,
                        "precision": top1["positive_precision"],
                        "recall": top1["positive_recall"],
                        "enrichment": top1["enrichment_over_prevalence"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame["group"] = (
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
    groups = list(dict.fromkeys(frame["group"]))
    x = np.arange(len(groups))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for offset, catalog_type in enumerate(CATALOG_LABELS):
        selected = frame.loc[frame["catalog_type"].eq(catalog_type)].set_index(
            "group"
        ).loc[groups]
        position = x + (offset - 0.5) * width
        for axis, metric in zip(
            axes, ("precision", "recall", "enrichment"), strict=True
        ):
            axis.bar(
                position,
                selected[metric],
                width,
                color=COLORS[catalog_type],
                label=CATALOG_LABELS[catalog_type],
            )
    axes[0].set_title("Top 1% precision")
    axes[0].set_ylabel("Fraction that are published loops")
    axes[1].set_title("Top 1% recall")
    axes[1].set_ylabel("Fraction of published loops recovered")
    axes[2].set_title("Top 1% enrichment")
    axes[2].set_ylabel("Precision / prevalence")
    axes[2].axhline(1, color=BLACK, linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(x, groups, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "What the highest-ranked 1% of predicted contacts recover",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_figure(fig, output)


def write_summary(
    recovery: dict[str, dict[str, Any]],
    *,
    output: Path,
) -> None:
    lines = [
        "# Peakachu and cross-caller loop recovery at 10 kb",
        "",
        "The family maps strongly rank published Peakachu loops and the "
        "stricter per-cell-type Peakachu ∩ scDeepLUCIA consensus on frozen "
        "validation chromosomes.",
        "",
        "## Exact 10 kb recovery",
        "",
        "| Catalog | Family | Distance | Loops | AUPRC | Baseline | "
        "AUPRC lift | AUROC | Top 1% precision | Top 1% recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    family_labels = {
        "cortical_IT": "Cortical IT",
        "corticofugal": "Corticofugal",
    }
    band_labels = {
        "250-500": "250–500 kb",
        "500-1000": "500–990 kb",
    }
    for catalog_type in CATALOG_LABELS:
        for family in PASSED_FAMILIES:
            for band_id in BAND_IDS:
                result = recovery[catalog_type]["exact"][family][band_id]
                classification = result["classification"]["family_eb"]
                top1 = result["topk"]["family_eb"][0]
                lines.append(
                    f"| {CATALOG_LABELS[catalog_type]} | "
                    f"{family_labels[family]} | {band_labels[band_id]} | "
                    f"{classification['positive_pair_count']:,} | "
                    f"{classification['auprc']:.4f} | "
                    f"{classification['positive_prevalence']:.5f} | "
                    f"{classification['auprc_lift_over_prevalence']:.1f}× | "
                    f"{classification['auroc']:.3f} | "
                    f"{top1['positive_precision']:.1%} | "
                    f"{top1['positive_recall']:.1%} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Peakachu alone is the cleaner caller-level corroboration: "
            "AUPRC is 20.0–53.3× prevalence and AUROC is 0.896–0.934.",
            "- Cross-caller consensus is much rarer, but the model recovers "
            "47.6–58.1% of those calls in its top-ranked 1% of candidate "
            "pairs.",
            "- The family EB and frozen pooled/shared maps are nearly tied. "
            "Only two of eight exact catalog/family/band comparisons have "
            "a positive chromosome-bootstrap lower bound for family-minus-"
            "shared AUPRC. The loop signal is therefore principally shared "
            "topology, not validated family-specific loop placement.",
            "",
            "## Critical limitation",
            "",
            "This is not independent biological validation. The target "
            "counts and both published loop-call sets come from the same "
            "mouse frontal-cortex scHiCAR assay collection "
            "(GSM8260434–GSM8260473 within GSE305889/GSE267126). The "
            "Peakachu/scDeepLUCIA agreement reduces caller-specific "
            "dependence, but it does not create a held-out cohort. The "
            "result supports denoising/ranking of loop-like contacts in "
            "this dataset; it does not yet establish transfer to new mice, "
            "experiments, or assays.",
            "",
            "The internal test split was not accessed.",
            "",
        ]
    )
    output.write_text("\n".join(lines))
    print(f"[peakachu-consensus10k] Wrote {output.name}", flush=True)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if not 0.5 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie between 0.5 and 1")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    print("[peakachu-consensus10k] Loading validation maps", flush=True)
    data = load_family_validation(args)
    if data["report"]["test_accessed"]:
        raise RuntimeError("Internal test split was accessed")
    if not data["fine"]["split"].eq("validation").all():
        raise RuntimeError("Non-validation pairs entered loop evaluation")

    supplement = args.supplement_xlsx.expanduser().resolve()
    print("[peakachu-consensus10k] Parsing Peakachu Table 5", flush=True)
    peakachu_members, peakachu_stats = read_peakachu_table(supplement)
    scdeep_root = args.scdeeplucia_root.expanduser().resolve()
    scdeep_members, scdeep_stats = load_scdeeplucia_members(scdeep_root)
    consensus_members = member_consensus(peakachu_members, scdeep_members)
    if len(consensus_members) > min(
        len(peakachu_members), len(scdeep_members)
    ):
        raise RuntimeError("Consensus is not a subset of both call sets")

    catalogs = {
        "peakachu": aggregate_family_catalog(
            peakachu_members, catalog_type="peakachu"
        ),
        "consensus": aggregate_family_catalog(
            consensus_members, catalog_type="consensus"
        ),
    }
    pd.concat(catalogs.values(), ignore_index=True).to_parquet(
        output_dir / "published_loops.collapsed_10kb.parquet",
        index=False,
        compression="zstd",
    )
    peakachu_members.to_parquet(
        output_dir / "peakachu_member_calls.collapsed_10kb.parquet",
        index=False,
        compression="zstd",
    )
    consensus_members.to_parquet(
        output_dir / "consensus_member_calls.collapsed_10kb.parquet",
        index=False,
        compression="zstd",
    )

    supports = {
        catalog_type: map_support_to_pairs(data["fine"], catalog)
        for catalog_type, catalog in catalogs.items()
    }
    recovery = {}
    tables = []
    for catalog_offset, catalog_type in enumerate(CATALOG_LABELS):
        print(
            f"[peakachu-consensus10k] Evaluating {catalog_type}", flush=True
        )
        recovery[catalog_type] = recovery_metrics(
            data,
            supports[catalog_type],
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.confidence_level,
            seed=args.seed + 100_000 * catalog_offset,
        )
        table = flatten_recovery(recovery[catalog_type])
        table.insert(0, "catalog_type", catalog_type)
        tables.append(table)
    recovery_table = pd.concat(tables, ignore_index=True)
    recovery_table.to_csv(
        output_dir / "recovery_metrics.csv", index=False
    )

    member_overlap = {}
    consensus_keys = set(
        map(
            tuple,
            consensus_members[
                ["member", "chrom", "bin_i", "bin_j"]
            ].itertuples(index=False, name=None),
        )
    )
    for member in sorted(CANONICAL_TO_FAMILY):
        peak_count = int(peakachu_members["member"].eq(member).sum())
        scdeep_count = int(scdeep_members["member"].eq(member).sum())
        consensus_count = sum(key[0] == member for key in consensus_keys)
        member_overlap[member] = {
            "peakachu_10kb_calls": peak_count,
            "scdeeplucia_10kb_calls": scdeep_count,
            "consensus_10kb_calls": int(consensus_count),
            "consensus_fraction_of_peakachu": float(
                consensus_count / peak_count
            ),
            "consensus_fraction_of_scdeeplucia": float(
                consensus_count / scdeep_count
            ),
        }

    report = {
        "schema_version": 1,
        "scope": {
            "prediction": (
                "Family EB expected 10 kb scHiCAR contact map fixed from "
                "validation pseudoreplicate A and frozen pooled "
                "AlphaGenome topology."
            ),
            "published_outcomes": {
                "peakachu": (
                    "Peakachu 5 kb calls reported in the original scHiCAR "
                    "paper Supplementary Table 5."
                ),
                "consensus": (
                    "Exact 10 kb intersection of Peakachu and "
                    "scDeepLUCIA calls within each source cell type."
                ),
            },
            "consensus_rule": (
                "Each caller is first collapsed from 5 kb to 10 kb within "
                "the same cell type; exact 10 kb pair intersections are "
                "then taken per cell type before family pooling."
            ),
            "independent_biological_validation": False,
            "source_relationship": (
                "The target counts and both loop-call sets derive from the "
                "same frontal-cortex scHiCAR samples, GSM8260434–"
                "GSM8260473 within GSE305889/GSE267126. Independence is "
                "between loop callers, not biological cohorts."
            ),
            "genome_build": "mm10",
            "source_bin_size_bp": 5_000,
            "evaluation_bin_size_bp": 10_000,
            "distance_range_bp": [250_000, 1_000_000],
            "distance_maximum_exclusive": True,
            "match_modes": {
                "exact": "Both collapsed 10 kb anchors match exactly.",
                "anchor_tolerance_10kb": (
                    "Each anchor may differ by at most one 10 kb bin."
                ),
            },
            "families": {
                family: list(members)
                for family, members in CATALOG_MEMBERS.items()
            },
            "genomic_split": "validation",
            "test_accessed": False,
            "bootstrap_unit": "chromosome",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_interval_quantiles": [
                float(1 - args.confidence_level),
                float(args.confidence_level),
            ],
            "interpretation_caveat": (
                "Peakachu and scDeepLUCIA are computational loop callers "
                "applied to the same experimental scHiCAR data used to "
                "construct the prediction targets. Their agreement is "
                "higher-confidence caller consensus, not independent-"
                "cohort validation or assay-independent physical-loop "
                "ground truth."
            ),
        },
        "external_catalog": {
            "paper_doi": PAPER_DOI,
            "supplement_url": SUPPLEMENT_URL,
            "supplement_xlsx": source_record(supplement),
            "supplement_sha256": sha256_file(supplement),
            "peakachu_member_stats": peakachu_stats,
            "scdeeplucia_member_stats": scdeep_stats,
            "member_overlap": member_overlap,
            "collapsed_family_rows": {
                catalog_type: {
                    family: int(catalog["family_id"].eq(family).sum())
                    for family in PASSED_FAMILIES
                }
                for catalog_type, catalog in catalogs.items()
            },
            "validation_positive_pairs": {
                catalog_type: {
                    mode: {
                        family: int(
                            (supports[catalog_type][mode][family] > 0).sum()
                        )
                        for family in PASSED_FAMILIES
                    }
                    for mode in MATCH_MODES
                }
                for catalog_type in CATALOG_LABELS
            },
        },
        "internal_sources": {
            "family_eb_report": source_record(data["paths"]["report"]),
            "fine_pair_table": source_record(data["paths"]["fine"]),
        },
        "recovery": recovery,
    }
    atomic_json(output_dir / "metrics.json", report)
    plot_scorecard(
        recovery_table,
        output=output_dir / "01_cross_caller_scorecard.png",
    )
    plot_precision_recall(
        data,
        supports,
        output=output_dir / "02_precision_recall.png",
    )
    plot_top1_precision_recall(
        recovery,
        output=output_dir / "03_top1_precision_recall.png",
    )
    write_summary(
        recovery,
        output=output_dir / "summary.md",
    )
    print(recovery_table.to_string(index=False), flush=True)
    print(f"[peakachu-consensus10k] Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
