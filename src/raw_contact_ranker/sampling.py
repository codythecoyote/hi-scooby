from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from tqdm.auto import tqdm

from .common import atomic_json, update_manifest


def exposure_deciles(pairs: pd.DataFrame) -> np.ndarray:
    train = pairs["split"].eq("train")
    finite = np.isfinite(pairs.loc[train, "exposure"]) & (pairs.loc[train, "exposure"] > 0)
    if finite.sum() < 10:
        raise ValueError("Too few finite training exposure values")
    boundaries = np.quantile(
        np.log(pairs.loc[train, "exposure"].to_numpy(float)[finite]),
        np.linspace(0.1, 0.9, 9),
    )
    values = np.log(np.clip(pairs["exposure"].to_numpy(float), 1e-12, None))
    return np.searchsorted(boundaries, values, side="right").astype(np.uint8)


def _draw_excluding(
    rng: np.random.Generator,
    pool: np.ndarray,
    excluded: int,
    count: int,
) -> np.ndarray:
    """Uniformly draw with replacement from a sorted pool, excluding one id."""
    if count == 0:
        return np.empty(0, np.int64)
    position = int(np.searchsorted(pool, excluded))
    present = position < len(pool) and int(pool[position]) == int(excluded)
    available = len(pool) - int(present)
    if available <= 0:
        raise ValueError("Control population contains only the focal event")
    draws = rng.integers(0, available, size=count)
    if present:
        draws = draws + (draws >= position)
    return pool[draws]


def draw_event_controls(
    rng: np.random.Generator,
    broad_pool: np.ndarray,
    matched_pool: np.ndarray,
    positive: int,
    *,
    broad_count: int,
    matched_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a full-support broad/matched mixture and return exact mixture q."""
    # Both pools are constructed from the focal row's group and are sorted by
    # pair_id, so the focal row is present exactly once.
    broad_available = len(broad_pool) - 1
    matched_available = len(matched_pool) - 1
    if broad_available <= 0:
        raise ValueError("Broad likelihood population has no non-focal candidates")
    use_matched = matched_count > 0 and matched_available > 0
    if use_matched:
        broad = _draw_excluding(rng, broad_pool, positive, broad_count)
        matched = _draw_excluding(rng, matched_pool, positive, matched_count)
        selected = np.concatenate([broad, matched])
        broad_weight = broad_count / len(selected)
        matched_weight = matched_count / len(selected)
        proposal = np.full(len(selected), broad_weight / broad_available, np.float64)
        positions = np.searchsorted(matched_pool, selected)
        in_matched = positions < len(matched_pool)
        in_matched[in_matched] &= matched_pool[positions[in_matched]] == selected[in_matched]
        proposal[in_matched] += matched_weight / matched_available
        component = np.concatenate(
            [np.zeros(broad_count, np.uint8), np.ones(matched_count, np.uint8)]
        )
    else:
        selected = _draw_excluding(
            rng, broad_pool, positive, broad_count + matched_count
        )
        proposal = np.full(len(selected), 1.0 / broad_available, np.float64)
        component = np.zeros(len(selected), np.uint8)
    if np.any(proposal <= 0) or np.any(~np.isfinite(proposal)):
        raise FloatingPointError("Event proposal has invalid probability")
    return selected, proposal, component


def _group_pools(
    pairs: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    groups = pairs.groupby(columns, observed=True, sort=False)
    group_ids = groups.ngroup().to_numpy(np.int32)
    pools = {
        int(group_ids[index[0]]): pairs.iloc[index]["pair_id"].to_numpy(np.int64)
        for index in groups.indices.values()
    }
    return group_ids, pools


def _local_zero_pool(
    pool: np.ndarray,
    full_count: np.ndarray,
) -> np.ndarray:
    """Return unlabeled controls from the exact evaluation ranking group."""
    return pool[full_count[pool] == 0]


def sample_controls(
    config: dict[str, Any],
    *,
    controls_per_event: int,
    seed: int,
) -> dict[str, Any]:
    output_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(
        output_root / "canonical_pairs.parquet",
        columns=[
            "pair_id", "chrom", "distance_bin", "distance_band", "tile_row", "split",
            "anchor_class", "exposure",
            "distance_offset",
        ],
    ).sort_values("pair_id", kind="stable").reset_index(drop=True)
    expected = np.arange(len(pairs), dtype=np.int64)
    if not np.array_equal(pairs["pair_id"].to_numpy(np.int64), expected):
        raise ValueError("pair_id must be contiguous before control sampling")
    if config["sampling"].get("likelihood_conditioning") != "chromosome_distance_band":
        raise ValueError("Only chromosome_distance_band likelihood conditioning is supported")
    broad_count = int(config["sampling"]["broad_event_controls"])
    matched_count = int(config["sampling"]["matched_event_controls"])
    rank_count = int(config["sampling"]["rank_controls_per_event"])
    if broad_count + matched_count != controls_per_event:
        raise ValueError("Broad and matched event counts must sum to controls_per_event")
    if broad_count <= 0 or matched_count < 0 or rank_count <= 0:
        raise ValueError(
            "A positive broad component is required for full support, matched "
            "controls must be nonnegative, and rank controls must be positive"
        )
    pairs["exposure_decile"] = exposure_deciles(pairs)
    broad_ids, broad_pools = _group_pools(
        pairs, ["chrom", "distance_band", "split"]
    )
    matched_ids, matched_pools = _group_pools(
        pairs,
        ["chrom", "distance_bin", "exposure_decile", "anchor_class", "split"],
    )
    local_rank_ids, local_rank_pools = _group_pools(
        pairs, ["tile_row", "distance_band", "split"]
    )
    evidence = zarr.open_group(
        str(output_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    outputs = {
        "train": output_root / "sampled_control_groups.parquet",
        "validation": output_root / "validation_sampled_control_groups.parquet",
    }
    temporaries = {
        split: path.with_suffix(".parquet.tmp") for split, path in outputs.items()
    }
    writers: dict[str, pq.ParquetWriter | None] = {
        split: None for split in outputs
    }
    rng = np.random.default_rng(seed)
    written_events = {split: 0 for split in outputs}
    matched_fallback_events = {split: 0 for split in outputs}
    rank_fallback_events = {split: 0 for split in outputs}
    normalizer_errors: list[float] = []
    fixed_scores = (
        np.log(np.clip(pairs["exposure"].to_numpy(float), 1e-12, None))
        + pairs["distance_offset"].to_numpy(float)
    )
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in outputs}
    validation_cap = int(config["sampling"]["validation_events_per_context"])

    def flush(split: str) -> None:
        if not rows[split]:
            return
        table = pa.Table.from_pylist(rows[split])
        if writers[split] is None:
            writers[split] = pq.ParquetWriter(
                temporaries[split], table.schema, compression="zstd"
            )
        writers[split].write_table(table)
        rows[split] = []

    split_masks = {
        split: pairs["split"].eq(split).to_numpy() for split in outputs
    }
    try:
        context_count = evidence["support_weight"].shape[0]
        for context in tqdm(range(context_count), desc="Sample control contexts", unit="context"):
            support = np.asarray(evidence["support_weight"][context], np.float32)
            full_count = np.asarray(evidence["full_count"][context], np.uint32)
            local_rank_zero = {
                key: _local_zero_pool(pool, full_count)
                for key, pool in local_rank_pools.items()
            }
            for optimization_split in outputs:
                positive_ids = np.flatnonzero(
                    (full_count > 0) & split_masks[optimization_split]
                )
                if (
                    optimization_split == "validation"
                    and len(positive_ids) > validation_cap
                ):
                    positive_ids = np.sort(
                        rng.choice(positive_ids, size=validation_cap, replace=False)
                    )
                for positive in tqdm(
                    positive_ids,
                    desc=(
                        f"Sample {optimization_split} events "
                        f"{context + 1}/{context_count}"
                    ),
                    unit="event",
                    leave=False,
                    mininterval=5.0,
                ):
                    broad_pool = broad_pools[int(broad_ids[positive])]
                    matched_pool = matched_pools[int(matched_ids[positive])]
                    if len(matched_pool) <= 1:
                        matched_fallback_events[optimization_split] += 1
                    event_ids, proposal, components = draw_event_controls(
                        rng,
                        broad_pool,
                        matched_pool,
                        int(positive),
                        broad_count=broad_count,
                        matched_count=matched_count,
                    )
                    if optimization_split == "train" and len(normalizer_errors) < 64:
                        position = int(np.searchsorted(broad_pool, positive))
                        opportunity_ids = np.concatenate(
                            [broad_pool[:position], broad_pool[position + 1 :]]
                        )
                        exact_scores = fixed_scores[opportunity_ids]
                        exact_max = float(np.max(exact_scores))
                        exact = float(
                            np.log(np.exp(exact_scores - exact_max).sum()) + exact_max
                        )
                        sampled_scores = fixed_scores[event_ids] - np.log(proposal)
                        sampled_max = float(np.max(sampled_scores))
                        estimate = float(
                            np.log(np.exp(sampled_scores - sampled_max).mean())
                            + sampled_max
                        )
                        normalizer_errors.append(abs(estimate - exact))
                    rank_pool = local_rank_zero[int(local_rank_ids[positive])]
                    if not len(rank_pool):
                        rank_fallback_events[optimization_split] += 1
                    if not len(rank_pool):
                        # Never substitute a chromosome-wide control: doing so
                        # changes the training estimand relative to top-contact
                        # evaluation. Retain the valid likelihood event and
                        # disable only this row's local rank term.
                        rank_ids = np.full(rank_count, int(positive), np.int64)
                        rank_weight = 0.0
                    else:
                        rank_ids = rng.choice(
                            rank_pool, size=rank_count, replace=True
                        )
                        rank_weight = float(support[positive])
                    rows[optimization_split].append(
                        {
                            "positive_pair_id": int(positive),
                            "context_id": int(context),
                            "support_weight": rank_weight,
                            "event_count": int(full_count[positive]),
                            "event_control_pair_ids": event_ids.tolist(),
                            "event_proposal_probabilities": proposal.tolist(),
                            "event_proposal_components": components.tolist(),
                            "rank_control_pair_ids": rank_ids.tolist(),
                        }
                    )
                    written_events[optimization_split] += 1
                    if len(rows[optimization_split]) >= 2_000:
                        flush(optimization_split)
        for optimization_split in outputs:
            flush(optimization_split)
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()
    missing_outputs = [split for split, writer in writers.items() if writer is None]
    if missing_outputs:
        raise RuntimeError(
            f"No supported events produced controls for splits: {missing_outputs}"
        )
    for split in outputs:
        temporaries[split].replace(outputs[split])
    tolerance = float(config["sampling"]["exact_normalizer_tolerance"])
    normalizer_median_error = float(np.median(normalizer_errors))
    normalizer_pass = bool(normalizer_median_error <= tolerance)
    report = {
        "schema_version": 2,
        "events": written_events["train"],
        "validation_events": written_events["validation"],
        "event_controls": written_events["train"] * controls_per_event,
        "rank_controls": written_events["train"] * rank_count,
        "validation_event_controls": (
            written_events["validation"] * controls_per_event
        ),
        "validation_rank_controls": written_events["validation"] * rank_count,
        "controls_per_event": controls_per_event,
        "rank_controls_per_event": rank_count,
        "likelihood_conditioning": "chromosome_distance_band",
        "event_proposal": "broad_full_support_plus_exact_matched_mixture",
        "rank_proposal": "same_tile_distance_band_zero_count",
        "full_support_guaranteed": True,
        "normalizer_audit": {
            "events": len(normalizer_errors),
            "median_absolute_log_error": normalizer_median_error,
            "p95_absolute_log_error": float(np.quantile(normalizer_errors, 0.95)),
            "tolerance": tolerance,
            "passed": normalizer_pass,
        },
        "matched_fallback_events": matched_fallback_events,
        "rank_fallback_events": rank_fallback_events,
        "grouped_output": str(outputs["train"]),
        "validation_grouped_output": str(outputs["validation"]),
        "label_semantics": "rank controls are unlabeled opportunities",
        "warnings": [],
    }
    atomic_json(output_root / "sampling_report.json", report)
    update_manifest(output_root, "sampling", report)
    if not normalizer_pass:
        for path in outputs.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(
            "Sampled normalizer audit exceeded tolerance: "
            f"{normalizer_median_error:.4f} > {tolerance:.4f}"
        )
    return report


def resample_local_rank_controls(
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Replace only rank controls in an existing prepared v2 dataset.

    Event controls and proposal probabilities are copied unchanged at the
    logical table level. Supported rank rows receive zero-support and lower-
    support controls from the same tile, broad distance band, and optimization
    split used by top-contact evaluation.
    """
    output_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(
        output_root / "canonical_pairs.parquet",
        columns=["pair_id", "tile_row", "distance_band", "split"],
    ).sort_values("pair_id", kind="stable").reset_index(drop=True)
    expected = np.arange(len(pairs), dtype=np.int64)
    if not np.array_equal(pairs["pair_id"].to_numpy(np.int64), expected):
        raise ValueError("pair_id must be contiguous before local rank resampling")
    local_ids, local_pools = _group_pools(
        pairs, ["tile_row", "distance_band", "split"]
    )
    evidence = zarr.open_group(
        str(output_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    rank_count = int(config["sampling"]["rank_controls_per_event"])
    sources = {
        "train": output_root / "sampled_control_groups.parquet",
        "validation": output_root / "validation_sampled_control_groups.parquet",
    }
    outputs = {
        "train": output_root / "sampled_local_control_groups.parquet",
        "validation": output_root / "validation_sampled_local_control_groups.parquet",
    }
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "schema_version": 3,
        "rank_proposal": (
            "same_tile_distance_band_zero_and_lower_replicate_support"
        ),
        "rank_controls_per_event": rank_count,
        "seed": int(seed),
        "splits": {},
    }
    report_path = output_root / "local_rank_resampling_report.json"
    # A failed refresh must never leave an older success report authorizing a
    # partially replaced pair of control files on the next submission.
    report_path.unlink(missing_ok=True)

    for split, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing prepared control table: {source}")
        output = outputs[split]
        temporary = output.with_suffix(".parquet.tmp")
        parquet = pq.ParquetFile(source)
        writer: pq.ParquetWriter | None = None
        row_count = 0
        active_context: int | None = None
        active_support: np.ndarray | None = None
        rank_pool_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        changed_rows = 0
        disabled_rows = 0
        try:
            for batch in tqdm(
                parquet.iter_batches(batch_size=8_192),
                total=(parquet.metadata.num_rows + 8_191) // 8_192,
                desc=f"Local rank controls ({split})",
                unit="batch",
            ):
                table = pa.Table.from_batches([batch])
                names = table.schema.names
                positive = np.asarray(
                    table["positive_pair_id"].combine_chunks().to_numpy(),
                    dtype=np.int64,
                )
                contexts = np.asarray(
                    table["context_id"].combine_chunks().to_numpy(),
                    dtype=np.int64,
                )
                support = np.asarray(
                    table["support_weight"].combine_chunks().to_numpy(),
                    dtype=np.float32,
                ).copy()
                rank_column = table["rank_control_pair_ids"].combine_chunks()
                rank_ids = np.asarray(
                    rank_column.flatten().to_numpy(), dtype=np.int64
                ).reshape(len(table), rank_count).copy()
                for context in np.unique(contexts):
                    context = int(context)
                    if active_context != context:
                        active_context = context
                        active_support = np.asarray(
                            evidence["support_weight"][context], np.float32
                        )
                        rank_pool_cache = {}
                    assert active_support is not None
                    context_rows = np.flatnonzero(
                        (contexts == context) & (support > 0)
                    )
                    if not len(context_rows):
                        continue
                    groups = local_ids[positive[context_rows]]
                    for group_id in np.unique(groups):
                        selected_rows = context_rows[groups == group_id]
                        group_id = int(group_id)
                        cached = rank_pool_cache.get(group_id)
                        if cached is None:
                            local_pool = local_pools[group_id]
                            local_support = active_support[local_pool]
                            zero_pool = local_pool[local_support <= 0]
                            lower_pool = local_pool[local_support > 0]
                            lower_order = np.argsort(
                                active_support[lower_pool], kind="stable"
                            )
                            lower_pool = lower_pool[lower_order]
                            lower_support = active_support[lower_pool]
                            cached = (zero_pool, lower_pool, lower_support)
                            rank_pool_cache[group_id] = cached
                        zero_pool, lower_pool, lower_support = cached
                        focal_support = support[selected_rows]
                        lower_limits = np.searchsorted(
                            lower_support, focal_support, side="left"
                        )
                        no_control = (len(zero_pool) == 0) & (lower_limits == 0)
                        if np.any(no_control):
                            disabled = selected_rows[no_control]
                            support[disabled] = 0.0
                            rank_ids[disabled] = positive[disabled, None]
                            disabled_rows += len(disabled)
                        usable_rows = selected_rows[~no_control]
                        if not len(usable_rows):
                            continue
                        usable_limits = lower_limits[~no_control]
                        zero_width = rank_count // 2 if len(zero_pool) else 0
                        hard_width = rank_count - zero_width
                        if zero_width:
                            zero_draws = rng.integers(
                                0,
                                len(zero_pool),
                                size=(len(usable_rows), zero_width),
                            )
                            rank_ids[usable_rows, :zero_width] = zero_pool[zero_draws]
                        has_lower = usable_limits > 0
                        if hard_width and np.any(has_lower):
                            random = rng.random((int(has_lower.sum()), hard_width))
                            indices = (
                                random * usable_limits[has_lower, None]
                            ).astype(np.int64)
                            rank_ids[
                                usable_rows[has_lower], zero_width:
                            ] = lower_pool[indices]
                        if hard_width and np.any(~has_lower):
                            if not len(zero_pool):
                                raise RuntimeError(
                                    "Local rank sampler has no lower-support control"
                                )
                            zero_draws = rng.integers(
                                0,
                                len(zero_pool),
                                size=(int((~has_lower).sum()), hard_width),
                            )
                            rank_ids[
                                usable_rows[~has_lower], zero_width:
                            ] = zero_pool[zero_draws]
                        if not zero_width:
                            # With no zero pool every usable row has a lower
                            # positive pool, which fills the complete matrix.
                            if np.any(~has_lower):
                                raise RuntimeError(
                                    "Usable local rows lack lower-support controls"
                                )
                        controls = rank_ids[usable_rows]
                        if np.any(local_ids[controls] != group_id):
                            raise RuntimeError(
                                "A local rank control escaped its tile/band group"
                            )
                        if np.any(
                            active_support[controls]
                            >= support[usable_rows, None]
                        ):
                            raise RuntimeError(
                                "A local rank control does not have lower support"
                            )
                        changed_rows += len(usable_rows)
                values = pa.array(rank_ids.reshape(-1), type=pa.int64())
                offsets = pa.array(
                    np.arange(
                        0,
                        (len(rank_ids) + 1) * rank_count,
                        rank_count,
                        dtype=np.int32,
                    )
                )
                local_lists = pa.ListArray.from_arrays(offsets, values)
                table = table.set_column(
                    names.index("rank_control_pair_ids"),
                    "rank_control_pair_ids",
                    local_lists,
                )
                table = table.set_column(
                    names.index("support_weight"),
                    "support_weight",
                    pa.array(support, type=pa.float32()),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                row_count += len(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise RuntimeError(f"No control rows found in {source}")
        temporary.replace(output)
        report["splits"][split] = {
            "source": str(source),
            "output": str(output),
            "rows": row_count,
            "locally_resampled_supported_rows": changed_rows,
            "disabled_no_lower_support_rows": disabled_rows,
        }
    atomic_json(report_path, report)
    return report


def exact_vs_sampled_normalizer(
    scores: np.ndarray,
    proposal: np.ndarray,
    sampled_indices: np.ndarray,
) -> tuple[float, float]:
    shifted = scores - scores.max()
    exact = float(np.log(np.exp(shifted).sum()) + scores.max())
    sampled = scores[sampled_indices] - np.log(proposal)
    estimate = float(np.log(np.exp(sampled - sampled.max()).mean()) + sampled.max())
    return exact, estimate
