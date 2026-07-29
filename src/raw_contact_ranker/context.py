from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    configured_distance_bands,
    selected_zarr_values,
)
from .metrics import build_top_contact_groups
from .power import fractional_top


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def output_passes_power(
    config: dict[str, Any],
    power_report: dict[str, Any],
    output_id: str,
) -> bool:
    primary = float(config["power"]["primary_top_fraction"])
    bands = [str(row["id"]) for row in configured_distance_bands(config)]
    matching = {
        str(row["band_id"]): row
        for row in power_report["summaries"]
        if str(row["output_id"]) == output_id
        and np.isclose(float(row["top_fraction"]), primary)
    }
    for band in bands:
        row = matching.get(band)
        if row is None:
            return False
        bootstrap = row.get("overlap_excess_bootstrap") or {}
        if not (
            float(row["k_weighted_evaluable_fraction"])
            >= float(config["power"]["minimum_evaluable_k_fraction"])
            and int(row["chromosomes"])
            >= int(config["power"]["minimum_validation_chromosomes"])
            and row["enrichment_over_chance"] is not None
            and float(row["enrichment_over_chance"])
            >= float(config["power"]["minimum_ceiling_enrichment"])
            and bootstrap.get("lower") is not None
            and float(bootstrap["lower"]) > 0
        ):
            return False
    return True


def _thin_counts(
    counts: np.ndarray,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source = np.asarray(counts, np.int64)
    source_total = int(source.sum())
    if total < 0 or total > source_total:
        raise ValueError("Invalid depth-matching total")
    if total == source_total:
        return source.copy()
    if total == 0:
        return np.zeros_like(source)
    return rng.multivariate_hypergeometric(source, total).astype(np.int64)


def _top_overlap(
    pairs: pd.DataFrame,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    *,
    band: str,
    fraction: float,
) -> dict[str, float | int | None]:
    rows = []
    for group_band, _, indices in build_top_contact_groups(
        pairs["tile_row"].to_numpy(),
        pairs["distance_band"].to_numpy(),
    ):
        if str(group_band) != band:
            continue
        size = len(indices)
        k = max(1, int(math.ceil(size * fraction)))
        top_a = fractional_top(
            counts_a[indices], k, eligible=counts_a[indices] > 0
        )
        top_b = fractional_top(
            counts_b[indices], k, eligible=counts_b[indices] > 0
        )
        if top_a is None or top_b is None:
            continue
        overlap = float(np.dot(top_a.weights, top_b.weights) / k)
        rows.append((k, overlap, k / size))
    if not rows:
        return {
            "groups": 0,
            "overlap": None,
            "chance": None,
            "overlap_excess": None,
            "enrichment_over_chance": None,
        }
    weights = np.asarray([row[0] for row in rows], float)
    overlap = float(np.average([row[1] for row in rows], weights=weights))
    chance = float(np.average([row[2] for row in rows], weights=weights))
    return {
        "groups": len(rows),
        "overlap": overlap,
        "chance": chance,
        "overlap_excess": overlap - chance,
        "enrichment_over_chance": overlap / chance,
    }


def evaluate_context_concordance(
    config: dict[str, Any],
    *,
    power_gate: Path,
    output: Path,
) -> dict[str, Any]:
    """Gate within-output signal against a depth-matched other-context control."""
    power = _read(power_gate)
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(
        data_root / "canonical_pairs.parquet",
        columns=["pair_id", "tile_row", "distance_band", "split"],
    ).sort_values("pair_id", kind="stable")
    validation = pairs["split"].eq("validation").to_numpy()
    validation_pairs = pairs.loc[validation].reset_index(drop=True)
    validation_ids = pairs.loc[
        validation, "pair_id"
    ].to_numpy(np.int64)
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    names = contexts["cell_type"].astype(str).tolist()
    context_a = {}
    context_b = {}
    for index, name in tqdm(
        enumerate(names),
        total=len(names),
        desc="Load concordance pseudoreplicates",
        unit="context",
    ):
        context_a[name] = selected_zarr_values(
            evidence["counts_a"],
            index,
            validation_ids,
            dtype=np.uint64,
        )
        context_b[name] = selected_zarr_values(
            evidence["counts_b"],
            index,
            validation_ids,
            dtype=np.uint64,
        )
    units: dict[str, dict[str, Any]] = {
        name: {
            "output_type": "context",
            "members": [name],
            "a": context_a[name],
            "b": context_b[name],
        }
        for name in names
    }
    for pool in config.get("contexts", {}).get("pools", []):
        members = list(map(str, pool["members"]))
        units[str(pool["id"])] = {
            "output_type": "pool",
            "members": members,
            "a": np.sum(
                np.stack([context_a[name] for name in members]),
                axis=0,
                dtype=np.uint64,
            ),
            "b": np.sum(
                np.stack([context_b[name] for name in members]),
                axis=0,
                dtype=np.uint64,
            ),
        }
    bands = [str(row["id"]) for row in configured_distance_bands(config)]
    fraction = float(config["power"]["primary_top_fraction"])
    rng = np.random.default_rng(int(config["seed"]) + 91_771)
    reports = {}
    for output_id, unit in tqdm(
        units.items(),
        total=len(units),
        desc="Evaluate context concordance",
        unit="output",
    ):
        eligible = output_passes_power(config, power, output_id)
        other_members = [name for name in names if name not in unit["members"]]
        other = np.sum(
            np.stack([context_b[name] for name in other_members]),
            axis=0,
            dtype=np.uint64,
        )
        band_reports = {}
        for band in bands:
            selected = validation_pairs["distance_band"].eq(band).to_numpy()
            same_a = np.asarray(unit["a"], np.uint64).copy()
            same_b = np.asarray(unit["b"], np.uint64).copy()
            comparison_a = np.zeros_like(same_a)
            comparison_b = np.zeros_like(same_b)
            depth = min(
                int(same_a[selected].sum()),
                int(other[selected].sum()),
            )
            comparison_a[selected] = _thin_counts(
                same_a[selected], depth, rng
            )
            comparison_b[selected] = _thin_counts(
                other[selected], depth, rng
            )
            same = _top_overlap(
                validation_pairs,
                same_a,
                same_b,
                band=band,
                fraction=fraction,
            )
            cross = _top_overlap(
                validation_pairs,
                comparison_a,
                comparison_b,
                band=band,
                fraction=fraction,
            )
            band_reports[band] = {
                "same_context": same,
                "depth_matched_cross_context": cross,
                "matched_depth": depth,
                "passed": bool(
                    same["overlap_excess"] is not None
                    and cross["overlap_excess"] is not None
                    and float(same["overlap_excess"])
                    > float(cross["overlap_excess"])
                ),
            }
        reports[output_id] = {
            "output_type": unit["output_type"],
            "members": unit["members"],
            "power_eligible": eligible,
            "bands": band_reports,
            "passed": bool(eligible)
            and all(row["passed"] for row in band_reports.values()),
        }
    report = {
        "schema_version": 1,
        "comparison": "same_output_vs_depth_matched_other_contexts",
        "top_fraction": fraction,
        "outputs": reports,
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report


def build_context_rollout(
    config: dict[str, Any],
    *,
    power_gate: Path,
    concordance_gate: Path,
    calibration_gate: Path,
    extension_gate: Path | None,
    output: Path,
) -> dict[str, Any]:
    """Resolve every public output to an accepted topology and calibration."""
    power = _read(power_gate)
    concordance = _read(concordance_gate)
    calibration = _read(calibration_gate)
    if calibration.get("accepted") is not True:
        raise RuntimeError("Shared calibration is not accepted")
    extension = _read(extension_gate) if extension_gate is not None else None
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    names = contexts["cell_type"].astype(str).tolist()
    pools = list(config.get("contexts", {}).get("pools", []))
    pool_power = {
        str(pool["id"]): output_passes_power(
            config, power, str(pool["id"])
        )
        for pool in pools
    }
    pool_calibration = {
        str(pool["id"]): calibration.get("unit_accepted", {}).get(
            str(pool["id"])
        )
        is True
        for pool in pools
    }
    extension_outputs = (extension or {}).get("outputs", {})
    rows: list[dict[str, Any]] = [
        {
            "output_id": "shared",
            "output_type": "shared",
            "members": names,
            "topology_source": "shared",
            "calibration_source": "shared",
            "context_delta_path": None,
            "context_delta_column": None,
            "claim_scope": "shared",
        }
    ]
    for pool in pools:
        pool_id = str(pool["id"])
        if not (pool_power[pool_id] and pool_calibration[pool_id]):
            continue
        extension_row = extension_outputs.get(pool_id, {})
        accepted = extension_row.get("accepted") is True
        rows.append(
            {
                "output_id": pool_id,
                "output_type": "pool",
                "members": list(map(str, pool["members"])),
                "topology_source": pool_id if accepted else "shared",
                "calibration_source": pool_id,
                "context_delta_path": (
                    extension_row.get("prediction_path") if accepted else None
                ),
                "context_delta_column": (
                    extension_row.get("prediction_column") if accepted else None
                ),
                "claim_scope": (
                    extension_row.get("claim_scope") if accepted else "shared"
                ),
            }
        )
    unmatched = set(
        map(str, config.get("contexts", {}).get("unpooled_shared_fallback", []))
    )
    for name in names:
        extension_row = extension_outputs.get(name, {})
        accepted = extension_row.get("accepted") is True
        if accepted:
            rows.append(
                {
                    "output_id": name,
                    "output_type": "context",
                    "members": [name],
                    "topology_source": name,
                    "calibration_source": name,
                    "context_delta_path": extension_row["prediction_path"],
                    "context_delta_column": extension_row["prediction_column"],
                    "claim_scope": extension_row["claim_scope"],
                }
            )
            continue
        fallback_pool = next(
            (
                str(pool["id"])
                for pool in pools
                if name in set(map(str, pool["members"]))
                and pool_power[str(pool["id"])]
                and pool_calibration[str(pool["id"])]
            ),
            None,
        )
        direct_power = output_passes_power(config, power, name)
        prerequisite = concordance["outputs"].get(name, {}).get("passed") is True
        if name in unmatched and not direct_power:
            fallback_pool = None
        fallback_extension = extension_outputs.get(fallback_pool, {})
        fallback_extension_accepted = (
            fallback_pool is not None
            and fallback_extension.get("accepted") is True
        )
        rows.append(
            {
                "output_id": name,
                "output_type": "context",
                "members": [name],
                "topology_source": fallback_pool or "shared",
                "calibration_source": fallback_pool or "shared",
                "context_delta_path": (
                    fallback_extension.get("prediction_path")
                    if fallback_extension_accepted
                    else None
                ),
                "context_delta_column": (
                    fallback_extension.get("prediction_column")
                    if fallback_extension_accepted
                    else None
                ),
                "claim_scope": "pooled" if fallback_pool else "shared",
                "context_extension_eligible": bool(
                    direct_power and prerequisite
                ),
                "context_extension_failure": (
                    extension_row.get("failure_reason")
                    if extension_row
                    else "not_accepted_or_not_attempted"
                ),
            }
        )
    report = {
        "schema_version": 1,
        "outputs": rows,
        "fallback_pool_order": [str(pool["id"]) for pool in pools],
        "extension_gate": str(extension_gate) if extension_gate else None,
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report
