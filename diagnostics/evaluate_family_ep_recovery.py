#!/usr/bin/env python3
"""Evaluate held-out enhancer--promoter-like contact recovery by family maps.

The family prediction is fixed from validation pseudoreplicate A and the
frozen pooled AlphaGenome topology.  The outcome is validation
pseudoreplicate B.  Test rows are never read.

This diagnostic treats a 10 kb pair as an enhancer--promoter-like candidate
when exactly one anchor contains a GENCODE TSS and the other contains an
ENCODE dELS or pELS cCRE but no TSS.  These are candidate loop-like contacts,
not causal enhancer assignments or base-pair-resolved loops.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import types
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numcodecs import get_codec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "diagnostics"))

# This read-only diagnostic does not use the trainable torch model.  Register
# the package path directly so it can run in the lightweight CPU analysis
# environment without importing raw_contact_ranker.__init__ and torch.
if "raw_contact_ranker" not in sys.modules:
    package = types.ModuleType("raw_contact_ranker")
    package.__path__ = [str(REPO_ROOT / "src/raw_contact_ranker")]
    sys.modules["raw_contact_ranker"] = package

from raw_contact_ranker.common import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    source_record,
)
from raw_contact_ranker.context_family_eb import FAMILY_PARTITION  # noqa: E402


BAND_IDS = ("250-500", "500-1000")
BAND_LABELS = ("250–500 kb", "500–990 kb")
FAMILY_IDS = tuple(FAMILY_PARTITION)
PASSED_FAMILIES = ("cortical_IT", "corticofugal")
PASSED_INDEX = tuple(FAMILY_IDS.index(value) for value in PASSED_FAMILIES)
BLUE = "#2F75B5"
ORANGE = "#E68632"
GREEN = "#2A9D8F"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
BLACK = "#222222"
TOP_FRACTIONS = (0.01, 0.02, 0.05)
SUBCLASSES = ("ELS", "dELS", "pELS")
PRIMARY_SUBCLASS = "ELS"
MINIMUM_PROMOTER_CANDIDATES = 20
SCOPE = (
    "Validation A-informed family EB prediction • outcome = independent "
    "pseudoreplicate B • strict TSS–ELS 10 kb candidates • frozen "
    "pooled/shared comparator • test untouched"
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
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/raw_contact_ranker_10kb_v1"
            / "context_family_eb_grid_10kb_cpu/diagnostic_v1"
            / "enhancer_promoter_recovery_v1"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.975)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.008, SCOPE, color=GRAY, fontsize=7.2, ha="left")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[family-ep] Wrote {path.name}", flush=True)


def zarr_v2_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads((path / ".zarray").read_text())
    if int(metadata["zarr_format"]) != 2:
        raise RuntimeError(f"{path} is not a Zarr v2 array")
    if metadata.get("filters") not in (None, []):
        raise RuntimeError(f"{path} uses unsupported Zarr filters")
    return metadata


def decode_zarr_v2_chunk(
    path: Path,
    metadata: dict[str, Any],
    *,
    chunk_shape: tuple[int, ...],
) -> np.ndarray:
    dtype = np.dtype(metadata["dtype"])
    if not path.exists():
        return np.full(
            chunk_shape,
            metadata.get("fill_value", 0),
            dtype=dtype,
        )
    payload = path.read_bytes()
    compressor = metadata.get("compressor")
    if compressor is not None:
        payload = get_codec(compressor).decode(payload)
    expected = int(np.prod(chunk_shape))
    values = np.frombuffer(payload, dtype=dtype)
    if len(values) != expected:
        raise RuntimeError(
            f"Decoded {path} has {len(values)} rather than {expected} values"
        )
    return values.reshape(chunk_shape, order=metadata.get("order", "C"))


def read_zarr_v2_1d(path: Path) -> np.ndarray:
    metadata = zarr_v2_metadata(path)
    shape = tuple(map(int, metadata["shape"]))
    chunks = tuple(map(int, metadata["chunks"]))
    if len(shape) != 1 or len(chunks) != 1:
        raise RuntimeError(f"{path} is not one-dimensional")
    output = np.empty(shape[0], dtype=np.dtype(metadata["dtype"]))
    for chunk_index, start in enumerate(range(0, shape[0], chunks[0])):
        stop = min(start + chunks[0], shape[0])
        decoded = decode_zarr_v2_chunk(
            path / str(chunk_index),
            metadata,
            chunk_shape=(chunks[0],),
        )
        output[start:stop] = decoded[: stop - start]
    return output


def read_zarr_v2_row(path: Path, row: int) -> np.ndarray:
    metadata = zarr_v2_metadata(path)
    shape = tuple(map(int, metadata["shape"]))
    chunks = tuple(map(int, metadata["chunks"]))
    if (
        len(shape) != 2
        or len(chunks) != 2
        or chunks[0] != 1
        or row < 0
        or row >= shape[0]
    ):
        raise RuntimeError(f"{path} lacks supported row-major chunks")
    output = np.empty(shape[1], dtype=np.dtype(metadata["dtype"]))
    for chunk_index, start in enumerate(range(0, shape[1], chunks[1])):
        stop = min(start + chunks[1], shape[1])
        decoded = decode_zarr_v2_chunk(
            path / f"{row}.{chunk_index}",
            metadata,
            chunk_shape=(1, chunks[1]),
        )[0]
        output[start:stop] = decoded[: stop - start]
    return output


def zarr_v2_attrs(path: Path) -> dict[str, Any]:
    return json.loads((path / ".zattrs").read_text())


def half_depths(
    evidence_report: dict[str, Any],
    *,
    primary_split: int,
) -> tuple[np.ndarray, np.ndarray]:
    context_names = {
        int(row["context_index"]): str(row["cell_type"])
        for row in evidence_report["contexts"]
    }
    by_half: dict[str, dict[str, int]] = {"A": {}, "B": {}}
    for row in evidence_report["balance"]:
        if int(row["split"]) != primary_split:
            continue
        context = context_names[int(row["context_index"])]
        for half in ("A", "B"):
            matches = [
                value
                for value in row["halves"]
                if str(value["half"]) == half
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Context {context} lacks exactly one half {half}"
                )
            by_half[half][context] = int(matches[0]["valid_pairs"])
    expected = {
        member
        for members in FAMILY_PARTITION.values()
        for member in members
    }
    if set(by_half["A"]) != expected or set(by_half["B"]) != expected:
        raise RuntimeError("Pseudoreplicate depths do not cover all contexts")
    arrays = []
    for half in ("A", "B"):
        arrays.append(
            np.asarray(
                [
                    sum(by_half[half][member] for member in members)
                    for members in FAMILY_PARTITION.values()
                ],
                np.float64,
            )
        )
    return arrays[0], arrays[1]


def load_family_counts_b(
    evidence_path: Path,
    pair_ids: np.ndarray,
) -> np.ndarray:
    attrs = zarr_v2_attrs(evidence_path)
    context_ids = list(map(str, attrs["context_ids"]))
    context_lookup = {
        context: index for index, context in enumerate(context_ids)
    }
    expected = {
        member
        for members in FAMILY_PARTITION.values()
        for member in members
    }
    if set(context_lookup) != expected:
        raise RuntimeError("Evidence contexts differ from family partition")
    output = np.zeros((len(FAMILY_IDS), len(pair_ids)), np.uint32)
    for family_id in PASSED_FAMILIES:
        family = FAMILY_IDS.index(family_id)
        total = np.zeros(len(pair_ids), np.uint64)
        for member in FAMILY_PARTITION[family_id]:
            row = read_zarr_v2_row(
                evidence_path / "counts_b",
                context_lookup[member],
            )
            total += row[pair_ids].astype(np.uint64)
        if np.any(total > np.iinfo(np.uint32).max):
            raise OverflowError(f"{family_id} counts_b exceeds uint32")
        output[family] = total.astype(np.uint32)
    return output


def load_family_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Load the same authorized arrays without the incompatible Zarr v3 API."""
    config = load_config(args.config)
    eb_root = args.eb_root.expanduser().resolve()
    fine_path = eb_root / "fine_pairs.authorized.multires_10kb.parquet"
    coarse_path = eb_root / "validation_coarse_pairs.50kb.parquet"
    diagnostic_path = eb_root / "validation_diagnostics.50kb.zarr"
    fine_store_path = eb_root / "family_targets.multires_10kb.zarr"
    report_path = eb_root / "report.json"
    data_root = Path(config["outputs"]["data_root"]).resolve()
    evidence_path = data_root / "pseudoreplicate_evidence.zarr"
    evidence_report_path = data_root / "evidence_export_report.json"

    report = json.loads(report_path.read_text())
    evidence_report = json.loads(evidence_report_path.read_text())
    if report["test_accessed"]:
        raise RuntimeError("Family EB report accessed the test split")
    for family_id in PASSED_FAMILIES:
        if not report["validation"]["families"][family_id][
            "both_band_shared_gain_ci_gate"
        ]:
            raise RuntimeError(f"{family_id} is no longer a two-band pass")

    fine = pd.read_parquet(
        fine_path,
        filters=[("split", "==", "validation")],
    ).sort_values("pair_id", kind="stable").reset_index(drop=True)
    coarse = pd.read_parquet(coarse_path).sort_values(
        "validation_row", kind="stable"
    ).reset_index(drop=True)
    if fine.empty or not fine["split"].eq("validation").all():
        raise RuntimeError("Fine table is not isolated to validation")
    diagnostic_attrs = zarr_v2_attrs(diagnostic_path)
    fine_store_attrs = zarr_v2_attrs(fine_store_path)
    evidence_attrs = zarr_v2_attrs(evidence_path)
    if list(map(str, diagnostic_attrs["family_ids"])) != list(FAMILY_IDS):
        raise RuntimeError("Family order changed")
    if sha256_file(coarse_path) != diagnostic_attrs["row_table_sha256"]:
        raise RuntimeError("Coarse diagnostics changed row table")
    if sha256_file(fine_path) != fine_store_attrs["row_table_sha256"]:
        raise RuntimeError("Fine topology store changed row table")

    pair_ids = fine["pair_id"].to_numpy(np.int64)
    coarse_ids = coarse["coarse_pair_id"].to_numpy(np.int64)
    dense = np.full(int(coarse_ids.max()) + 1, -1, np.int64)
    dense[coarse_ids] = np.arange(len(coarse), dtype=np.int64)
    fine_coarse_position = dense[
        fine["coarse_pair_id"].to_numpy(np.int64)
    ]
    if np.any(fine_coarse_position < 0):
        raise RuntimeError("Fine rows reference a missing coarse row")

    posterior = np.vstack(
        [
            read_zarr_v2_row(
                diagnostic_path / "posterior_probability_from_counts_a",
                family,
            )
            for family in range(len(FAMILY_IDS))
        ]
    ).astype(np.float64)
    shared_coarse = read_zarr_v2_1d(
        diagnostic_path / "shared_prior_probability"
    ).astype(np.float64)
    counts_a_coarse = np.vstack(
        [
            read_zarr_v2_row(diagnostic_path / "counts_a", family)
            for family in range(len(FAMILY_IDS))
        ]
    ).astype(np.float64)
    group_index = coarse["normalization_group_id"].to_numpy(np.int64)
    fine_group = group_index[fine_coarse_position]
    within_all = read_zarr_v2_1d(
        fine_store_path / "within_coarse_shared_probability"
    ).astype(np.float64)
    within = within_all[fine["target_row"].to_numpy(np.int64)]
    del within_all
    family_probability = (
        posterior[:, fine_coarse_position] * within[None, :]
    )
    shared_probability = shared_coarse[fine_coarse_position] * within
    group_count = int(group_index.max()) + 1
    group_total_a = np.vstack(
        [
            np.bincount(
                group_index,
                weights=counts_a_coarse[family],
                minlength=group_count,
            )
            for family in range(len(FAMILY_IDS))
        ]
    )
    family_prediction_a = family_probability * group_total_a[:, fine_group]
    shared_prediction_a = (
        shared_probability[None, :] * group_total_a[:, fine_group]
    )
    primary_split = int(evidence_attrs["primary_split"])
    if primary_split != 0:
        raise RuntimeError("Unexpected primary pseudoreplicate split")
    depth_a, depth_b = half_depths(
        evidence_report,
        primary_split=primary_split,
    )
    family_prediction_b = family_prediction_a * (depth_b / depth_a)[:, None]
    shared_prediction_b = shared_prediction_a * (depth_b / depth_a)[:, None]

    # Held-out B is loaded only after all predictions have been fixed.
    observed_b = load_family_counts_b(evidence_path, pair_ids)
    counts_b_coarse = np.vstack(
        [
            read_zarr_v2_row(diagnostic_path / "counts_b", family)
            for family in range(len(FAMILY_IDS))
        ]
    )
    for family in PASSED_INDEX:
        aggregate_b = np.bincount(
            fine_coarse_position,
            weights=observed_b[family].astype(np.float64),
            minlength=len(coarse),
        ).astype(np.uint64)
        if not np.array_equal(
            aggregate_b,
            counts_b_coarse[family].astype(np.uint64),
        ):
            raise RuntimeError(
                f"Fine counts_b do not reproduce {FAMILY_IDS[family]}"
            )
    return {
        "config": config,
        "fine": fine,
        "observed_b": observed_b,
        "family_prediction_b": family_prediction_b,
        "shared_prediction_b": shared_prediction_b,
        "family_probability": family_probability,
        "shared_probability": shared_probability,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "report": report,
        "paths": {
            "fine": fine_path,
            "coarse": coarse_path,
            "report": report_path,
            "evidence_report": evidence_report_path,
        },
    }


def enhancer_bins(
    path: Path,
    *,
    bin_size: int,
) -> dict[str, set[tuple[str, int]]]:
    """Return dELS and pELS occupied genomic bins."""
    output = {"dELS": set(), "pELS": set()}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise RuntimeError(f"Malformed cCRE BED row: {line.rstrip()}")
            chrom, start, end, label = (
                str(fields[0]),
                int(fields[1]),
                int(fields[2]),
                str(fields[9]),
            )
            if label not in output or end <= start:
                continue
            first = start // bin_size
            last = (end - 1) // bin_size
            for index in range(first, last + 1):
                output[label].add((chrom, index * bin_size))
    return output


def pair_annotations(
    fine: pd.DataFrame,
    *,
    canonical_path: Path,
    ccre_path: Path,
    bin_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    annotations = pd.read_parquet(
        canonical_path,
        columns=["pair_id", "bin_i_tss", "bin_j_tss"],
        filters=[("split", "==", "validation")],
    )
    joined = fine[["pair_id"]].merge(
        annotations,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined[["bin_i_tss", "bin_j_tss"]].isna().any().any():
        raise RuntimeError("Fine validation pairs lack canonical TSS annotations")
    tss_i = joined["bin_i_tss"].to_numpy(bool)
    tss_j = joined["bin_j_tss"].to_numpy(bool)
    single_tss = tss_i ^ tss_j

    bins = enhancer_bins(ccre_path, bin_size=bin_size)
    chrom = fine["chrom"].astype(str).to_numpy()
    bin_i = fine["bin_i"].to_numpy(np.int64)
    bin_j = fine["bin_j"].to_numpy(np.int64)
    masks: dict[str, np.ndarray] = {}
    for label in ("dELS", "pELS"):
        occupied = bins[label]
        enhancer_i = np.fromiter(
            (
                (str(chromosome), int(start)) in occupied
                for chromosome, start in zip(chrom, bin_i, strict=True)
            ),
            dtype=bool,
            count=len(fine),
        )
        enhancer_j = np.fromiter(
            (
                (str(chromosome), int(start)) in occupied
                for chromosome, start in zip(chrom, bin_j, strict=True)
            ),
            dtype=bool,
            count=len(fine),
        )
        # The enhancer-side bin must not also contain a TSS.
        masks[label] = (
            (tss_i & ~tss_j & enhancer_j)
            | (tss_j & ~tss_i & enhancer_i)
        )
    masks["ELS"] = masks["dELS"] | masks["pELS"]
    promoter_start = np.where(tss_i & ~tss_j, bin_i, bin_j)
    promoter_key = np.asarray(
        [
            f"{chromosome}:{start}"
            for chromosome, start in zip(
                chrom, promoter_start, strict=True
            )
        ],
        dtype=object,
    )
    if np.any(masks["ELS"] & ~single_tss):
        raise RuntimeError("Strict ELS candidate does not have one TSS anchor")
    return masks, promoter_key


def top_contact_enrichment(
    observed: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, float]]:
    observed = np.asarray(observed, np.float64)
    score = np.asarray(score, np.float64)
    if observed.sum() <= 0:
        raise RuntimeError("Candidate set has no held-out contacts")
    order = np.argsort(score, kind="stable")[::-1]
    records = []
    for fraction in TOP_FRACTIONS:
        selected_count = max(1, int(np.ceil(fraction * len(order))))
        realized = selected_count / len(order)
        captured = float(observed[order[:selected_count]].sum() / observed.sum())
        records.append(
            {
                "requested_top_fraction": float(fraction),
                "selected_pair_count": int(selected_count),
                "selected_pair_fraction": float(realized),
                "heldout_contact_fraction": captured,
                "enrichment_over_random": float(captured / realized),
            }
        )
    return records


def promoter_macro_ap(
    labels: np.ndarray,
    score: np.ndarray,
    promoter: np.ndarray,
) -> dict[str, float | int | None]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, bool),
            "score": np.asarray(score, np.float64),
            "promoter": np.asarray(promoter, object),
        }
    )
    values = []
    eligible_candidates = 0
    for _, group in frame.groupby("promoter", sort=False):
        group_labels = group["label"].to_numpy(bool)
        if (
            len(group) < MINIMUM_PROMOTER_CANDIDATES
            or not group_labels.any()
            or group_labels.all()
        ):
            continue
        values.append(
            average_precision_score(
                group_labels,
                group["score"].to_numpy(np.float64),
            )
        )
        eligible_candidates += len(group)
    return {
        "minimum_candidates_per_promoter": MINIMUM_PROMOTER_CANDIDATES,
        "eligible_promoters": int(len(values)),
        "eligible_candidates": int(eligible_candidates),
        "macro_auprc": float(np.mean(values)) if values else None,
        "median_promoter_auprc": (
            float(np.median(values)) if values else None
        ),
    }


def likelihood_gain(
    observed: np.ndarray,
    family_probability: np.ndarray,
    shared_probability: np.ndarray,
    chromosome: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    observed = np.asarray(observed, np.float64)
    probability_ratio = np.log(
        np.clip(np.asarray(family_probability, np.float64), 1e-300, 1.0)
    ) - np.log(
        np.clip(np.asarray(shared_probability, np.float64), 1e-300, 1.0)
    )
    difference = observed * probability_ratio
    event_total = float(observed.sum())
    if event_total <= 0:
        raise RuntimeError("Likelihood gain has no held-out events")
    chromosome = np.asarray(chromosome).astype(str)
    chromosomes = np.asarray(
        sorted(np.unique(chromosome[observed > 0])),
        dtype=object,
    )
    sufficient = {
        str(value): (
            float(difference[chromosome == value].sum()),
            float(observed[chromosome == value].sum()),
        )
        for value in chromosomes
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    for index in range(replicates):
        sample = rng.choice(chromosomes, len(chromosomes), replace=True)
        numerator = sum(sufficient[str(value)][0] for value in sample)
        denominator = sum(sufficient[str(value)][1] for value in sample)
        values[index] = numerator / denominator
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, confidence_level))
    point = float(difference.sum() / event_total)
    return {
        "heldout_contacts": int(event_total),
        "gain_per_contact_nats": point,
        "relative_probability_per_contact": float(np.exp(point)),
        "chromosome_bootstrap": {
            "replicates": int(replicates),
            "chromosomes": int(len(chromosomes)),
            "lower": lower,
            "median": float(np.median(values)),
            "upper": upper,
        },
        "positive_gain_with_lower_bound_above_zero": bool(
            point > 0 and lower > 0
        ),
    }


def classification_metrics(
    observed: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(observed) > 0
    if not labels.any() or labels.all():
        raise RuntimeError("Candidate labels are degenerate")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, score))
    return {
        "positive_pair_count": int(labels.sum()),
        "positive_prevalence": prevalence,
        "auprc": auprc,
        "auprc_lift_over_prevalence": float(auprc / prevalence),
        "auroc": float(roc_auc_score(labels, score)),
    }


def compute_metrics(
    data: dict[str, Any],
    masks: dict[str, np.ndarray],
    promoter_key: np.ndarray,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    fine = data["fine"]
    chromosome = fine["chrom"].astype(str).to_numpy()
    band_index = fine[
        "normalization_fine_distance_band_index"
    ].to_numpy(np.int64)
    output: dict[str, Any] = {}
    for family_offset, family in enumerate(PASSED_INDEX):
        family_id = FAMILY_IDS[family]
        output[family_id] = {
            "members": list(FAMILY_PARTITION[family_id]),
            "bands": {},
        }
        for band, band_id in enumerate(BAND_IDS):
            in_band = band_index == band
            all_observed = data["observed_b"][family, in_band].astype(
                np.float64
            )
            all_family = data["family_prediction_b"][family, in_band]
            all_shared = data["shared_prediction_b"][family, in_band]
            band_result: dict[str, Any] = {}
            for subtype_index, subtype in enumerate(SUBCLASSES):
                selected = in_band & masks[subtype]
                observed = data["observed_b"][family, selected].astype(
                    np.float64
                )
                family_score = data["family_prediction_b"][family, selected]
                shared_score = data["shared_prediction_b"][family, selected]
                labels = observed > 0
                if len(observed) < 100 or not labels.any() or labels.all():
                    raise RuntimeError(
                        f"Unevaluable {family_id} {band_id} {subtype}"
                    )
                family_classification = classification_metrics(
                    observed, family_score
                )
                shared_classification = classification_metrics(
                    observed, shared_score
                )
                family_top = top_contact_enrichment(
                    observed, family_score
                )
                shared_top = top_contact_enrichment(
                    observed, shared_score
                )
                gain = likelihood_gain(
                    observed,
                    data["family_probability"][family, selected],
                    data["shared_probability"][selected],
                    chromosome[selected],
                    replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    seed=(
                        seed
                        + 100 * family_offset
                        + 10 * band
                        + subtype_index
                    ),
                )
                candidate_fraction = float(selected.sum() / in_band.sum())
                observed_fraction = float(
                    observed.sum() / all_observed.sum()
                )
                family_fraction = float(
                    family_score.sum() / all_family.sum()
                )
                shared_fraction = float(
                    shared_score.sum() / all_shared.sum()
                )
                band_result[subtype] = {
                    "candidate_pair_count": int(selected.sum()),
                    "candidate_pair_fraction_of_band": candidate_fraction,
                    "heldout_positive_pair_count": int(labels.sum()),
                    "heldout_contacts": int(observed.sum()),
                    "classification": {
                        "family_eb": family_classification,
                        "frozen_shared": shared_classification,
                        "family_minus_shared_auprc": float(
                            family_classification["auprc"]
                            - shared_classification["auprc"]
                        ),
                    },
                    "top_contact_enrichment": {
                        "family_eb": family_top,
                        "frozen_shared": shared_top,
                    },
                    "promoter_macro_classification": {
                        "family_eb": promoter_macro_ap(
                            labels,
                            family_score,
                            promoter_key[selected],
                        ),
                        "frozen_shared": promoter_macro_ap(
                            labels,
                            shared_score,
                            promoter_key[selected],
                        ),
                    },
                    "family_vs_shared_likelihood": gain,
                    "candidate_mass_allocation": {
                        "heldout_contact_fraction": observed_fraction,
                        "family_predicted_fraction": family_fraction,
                        "shared_predicted_fraction": shared_fraction,
                        "heldout_enrichment_over_pair_fraction": float(
                            observed_fraction / candidate_fraction
                        ),
                        "family_enrichment_over_pair_fraction": float(
                            family_fraction / candidate_fraction
                        ),
                        "shared_enrichment_over_pair_fraction": float(
                            shared_fraction / candidate_fraction
                        ),
                    },
                    "absolute_recovery_supported": bool(
                        family_classification["auprc"] > family_classification[
                            "positive_prevalence"
                        ]
                        and family_top[0]["enrichment_over_random"] > 1
                    ),
                    "incremental_family_gain_supported": bool(
                        gain[
                            "positive_gain_with_lower_bound_above_zero"
                        ]
                        and family_classification["auprc"]
                        > shared_classification["auprc"]
                        and family_top[0]["enrichment_over_random"]
                        > shared_top[0]["enrichment_over_random"]
                    ),
                }
            output[family_id]["bands"][band_id] = band_result
    return output


def flatten_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for family_id, family in metrics.items():
        for band_id, band in family["bands"].items():
            for subtype, result in band.items():
                family_class = result["classification"]["family_eb"]
                shared_class = result["classification"]["frozen_shared"]
                gain = result["family_vs_shared_likelihood"]
                family_top = result["top_contact_enrichment"]["family_eb"]
                shared_top = result["top_contact_enrichment"]["frozen_shared"]
                macro = result["promoter_macro_classification"]
                rows.append(
                    {
                        "family_id": family_id,
                        "band_id": band_id,
                        "subclass": subtype,
                        "candidate_pairs": result["candidate_pair_count"],
                        "heldout_positive_pairs": result[
                            "heldout_positive_pair_count"
                        ],
                        "heldout_contacts": result["heldout_contacts"],
                        "prevalence": family_class["positive_prevalence"],
                        "family_auprc": family_class["auprc"],
                        "shared_auprc": shared_class["auprc"],
                        "family_auprc_lift": family_class[
                            "auprc_lift_over_prevalence"
                        ],
                        "shared_auprc_lift": shared_class[
                            "auprc_lift_over_prevalence"
                        ],
                        "family_auroc": family_class["auroc"],
                        "shared_auroc": shared_class["auroc"],
                        "family_top1_enrichment": family_top[0][
                            "enrichment_over_random"
                        ],
                        "shared_top1_enrichment": shared_top[0][
                            "enrichment_over_random"
                        ],
                        "family_promoter_macro_auprc": macro["family_eb"][
                            "macro_auprc"
                        ],
                        "shared_promoter_macro_auprc": macro[
                            "frozen_shared"
                        ]["macro_auprc"],
                        "ll_gain_nats_per_contact": gain[
                            "gain_per_contact_nats"
                        ],
                        "ll_gain_ci_lower": gain[
                            "chromosome_bootstrap"
                        ]["lower"],
                        "ll_gain_ci_upper": gain[
                            "chromosome_bootstrap"
                        ]["upper"],
                        "relative_probability_per_contact": gain[
                            "relative_probability_per_contact"
                        ],
                        "absolute_recovery_supported": result[
                            "absolute_recovery_supported"
                        ],
                        "incremental_family_gain_supported": result[
                            "incremental_family_gain_supported"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def plot_scorecard(table: pd.DataFrame, *, output: Path) -> None:
    primary = table.loc[table["subclass"].eq(PRIMARY_SUBCLASS)].copy()
    primary["label"] = (
        primary["family_id"].map(
            {
                "cortical_IT": "Cortical IT",
                "corticofugal": "Corticofugal",
            }
        )
        + "\n"
        + primary["band_id"].map(
            dict(zip(BAND_IDS, BAND_LABELS, strict=True))
        )
    )
    x = np.arange(len(primary))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    ax = axes[0]
    ax.bar(
        x - width / 2,
        primary["family_auprc"],
        width,
        color=BLUE,
        label="Family EB",
    )
    ax.bar(
        x + width / 2,
        primary["shared_auprc"],
        width,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        label="Frozen pooled/shared",
    )
    ax.scatter(
        x,
        primary["prevalence"],
        marker="_",
        s=180,
        linewidth=2,
        color=BLACK,
        label="Random/prevalence",
        zorder=4,
    )
    ax.set_title("Held-out contact classification")
    ax.set_ylabel("Area under precision–recall curve")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(
        x - width / 2,
        primary["family_top1_enrichment"],
        width,
        color=BLUE,
        label="Family EB",
    )
    ax.bar(
        x + width / 2,
        primary["shared_top1_enrichment"],
        width,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        label="Frozen pooled/shared",
    )
    ax.axhline(1, color=BLACK, linewidth=1, linestyle="--")
    ax.set_title("Top 1% recovery")
    ax.set_ylabel("Held-out contact-mass enrichment")

    ax = axes[2]
    point = primary["ll_gain_nats_per_contact"].to_numpy()
    lower = primary["ll_gain_ci_lower"].to_numpy()
    upper = primary["ll_gain_ci_upper"].to_numpy()
    ax.errorbar(
        x,
        point,
        yerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        color=GREEN,
        ecolor=GREEN,
        capsize=4,
        markersize=7,
    )
    ax.axhline(0, color=BLACK, linewidth=1, linestyle="--")
    ax.set_title("Family-specific gain over pooled")
    ax.set_ylabel("Log-likelihood gain / held-out contact (nats)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(primary["label"], rotation=20, ha="right")
    fig.suptitle(
        "Recovery of strict promoter–ELS candidate contacts",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(bottom=0.25, wspace=0.32)
    save_figure(fig, output)


def plot_precision_recall(
    data: dict[str, Any],
    mask: np.ndarray,
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
        for band, band_id in enumerate(BAND_IDS):
            ax = axes[family_offset, band]
            selected = mask & (band_index == band)
            observed = data["observed_b"][family, selected]
            labels = observed > 0
            family_score = data["family_prediction_b"][family, selected]
            shared_score = data["shared_prediction_b"][family, selected]
            for score, color, label in (
                (family_score, BLUE, "Family EB"),
                (shared_score, GRAY, "Frozen pooled/shared"),
            ):
                precision, recall, _ = precision_recall_curve(labels, score)
                ax.plot(recall, precision, color=color, linewidth=2, label=label)
            ax.axhline(
                labels.mean(),
                color=BLACK,
                linestyle="--",
                linewidth=1,
                label="Prevalence",
            )
            ax.set_title(
                f"{family_id.replace('_', ' ').title()} • "
                f"{BAND_LABELS[band]}"
            )
            ax.set_xlabel("Recall of B-positive candidate pairs")
            ax.set_ylabel("Precision")
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Held-out promoter–ELS contact precision–recall",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.10, hspace=0.34, wspace=0.25)
    save_figure(fig, output)


def plot_topk(metrics: dict[str, Any], *, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), sharex=True)
    for family_index, family_id in enumerate(PASSED_FAMILIES):
        for band_index, band_id in enumerate(BAND_IDS):
            ax = axes[family_index, band_index]
            result = metrics[family_id]["bands"][band_id][PRIMARY_SUBCLASS]
            for source, color, label in (
                ("family_eb", BLUE, "Family EB"),
                ("frozen_shared", GRAY, "Frozen pooled/shared"),
            ):
                records = result["top_contact_enrichment"][source]
                ax.plot(
                    [100 * row["selected_pair_fraction"] for row in records],
                    [row["enrichment_over_random"] for row in records],
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=label,
                )
            ax.axhline(1, color=BLACK, linewidth=1, linestyle="--")
            ax.set_title(
                f"{family_id.replace('_', ' ').title()} • "
                f"{BAND_LABELS[band_index]}"
            )
            ax.set_xlabel("Top-scoring candidate pairs (%)")
            ax.set_ylabel("Held-out contact-mass enrichment")
            ax.legend(fontsize=8)
    fig.suptitle(
        "Promoter–ELS contacts concentrate in top predicted pairs",
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

    print("[family-ep] Loading safeguarded family validation arrays", flush=True)
    data = load_family_validation(args)
    print("[family-ep] Validation arrays loaded", flush=True)
    config = data["config"]
    data_root = Path(config["outputs"]["data_root"]).resolve()
    canonical_path = data_root / "canonical_pairs.parquet"
    ccre_path = Path(config["paths"]["ccre_registry"]).resolve()
    masks, promoter_key = pair_annotations(
        data["fine"],
        canonical_path=canonical_path,
        ccre_path=ccre_path,
        bin_size=int(config["bin_size_bp"]),
    )
    print(
        "[family-ep] Joined TSS and cCRE-subclass annotations",
        flush=True,
    )
    metrics = compute_metrics(
        data,
        masks,
        promoter_key,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print("[family-ep] Recovery metrics computed", flush=True)
    table = flatten_metrics(metrics)
    table.to_csv(output_dir / "metrics.csv", index=False)
    report = {
        "schema_version": 1,
        "scope": {
            "prediction_input": (
                "validation pseudoreplicate A plus frozen pooled "
                "AlphaGenome topology"
            ),
            "outcome": "independent validation pseudoreplicate B",
            "test_accessed": False,
            "bin_size_bp": int(config["bin_size_bp"]),
            "candidate_definition": (
                "Exactly one anchor has a GENCODE TSS; the other has an "
                "ENCODE dELS or pELS cCRE and has no TSS."
            ),
            "interpretation": (
                "Candidate enhancer-promoter-like contact recovery; not "
                "causal enhancer assignment or base-pair-resolved loop proof."
            ),
            "primary_subclass": PRIMARY_SUBCLASS,
            "distance_bands": list(BAND_IDS),
            "families": list(PASSED_FAMILIES),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_interval_quantiles": [
                float(1 - args.confidence_level),
                float(args.confidence_level),
            ],
        },
        "sources": {
            "canonical_pairs": source_record(canonical_path),
            "ccre_registry": source_record(ccre_path),
            "family_eb_report": source_record(data["paths"]["report"]),
        },
        "metrics": metrics,
    }
    atomic_json(output_dir / "metrics.json", report)
    plot_scorecard(table, output=output_dir / "01_ep_recovery_scorecard.png")
    plot_precision_recall(
        data,
        masks[PRIMARY_SUBCLASS],
        output=output_dir / "02_ep_precision_recall.png",
    )
    plot_topk(metrics, output=output_dir / "03_ep_topk_enrichment.png")
    print(table.to_string(index=False), flush=True)
    print(f"[family-ep] Wrote {output_dir / 'metrics.json'}", flush=True)
    print(f"[family-ep] Wrote {output_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
