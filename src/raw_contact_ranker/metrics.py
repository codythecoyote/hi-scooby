from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
from scipy.stats import hypergeom, pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm


def supported_context_coverage(
    counts_by_label: dict[str, list[int]],
    *,
    minimum_supported_candidates: int,
    minimum_context_fraction: float,
) -> tuple[dict[str, int], int]:
    if not 0 < minimum_context_fraction <= 1:
        raise ValueError("minimum_context_fraction must be in (0, 1]")
    lengths = {len(counts) for counts in counts_by_label.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Every stratum must have support counts for every context")
    context_count = next(iter(lengths))
    required = max(1, math.ceil(context_count * minimum_context_fraction))
    passing = {
        label: int(np.sum(np.asarray(counts) >= minimum_supported_candidates))
        for label, counts in counts_by_label.items()
    }
    return passing, required


def _select_top_from_order(
    values: np.ndarray,
    fraction: float,
    order: np.ndarray,
    tie_mode: str = "hard_cutoff",
) -> tuple[np.ndarray, float]:
    if not len(order):
        return order, float("nan")
    count = max(1, int(np.ceil(len(order) * fraction)))
    tied = (
        count < len(order)
        and values[order[count - 1]] == values[order[count]]
    )
    if tie_mode == "inclusive_cutoff" and tied:
        cutoff = values[order[count - 1]]
        selected = order[values[order] >= cutoff]
    elif tie_mode == "hard_cutoff":
        selected = order[:count]
    elif tie_mode == "inclusive_cutoff":
        selected = order[:count]
    else:
        raise ValueError(f"Unsupported tie mode: {tie_mode}")
    return selected, float(tied)


def select_top(
    values: np.ndarray, fraction: float, tie_mode: str = "hard_cutoff"
) -> tuple[np.ndarray, float]:
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(-values[finite], kind="stable")]
    return _select_top_from_order(values, fraction, order, tie_mode)


def neighborhood_overlap(
    chrom: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    tolerance_bins: int,
    bin_size_bp: int = 5_000,
) -> float:
    if not len(selected_a) or not len(selected_b):
        return float("nan")
    reference = defaultdict(set)
    for idx in selected_b:
        reference[str(chrom[idx])].add(
            (int(bin_i[idx]) // bin_size_bp, int(bin_j[idx]) // bin_size_bp)
        )
    hits = 0
    for idx in selected_a:
        left = int(bin_i[idx]) // bin_size_bp
        right = int(bin_j[idx]) // bin_size_bp
        candidates = reference[str(chrom[idx])]
        found = any(
            (left + di, right + dj) in candidates
            for di in range(-tolerance_bins, tolerance_bins + 1)
            for dj in range(-tolerance_bins, tolerance_bins + 1)
        )
        hits += int(found)
    return hits / len(selected_a)


def candidate_neighborhood_sizes(
    chrom: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    tolerance_bins: int,
    bin_size_bp: int = 5_000,
) -> np.ndarray:
    """Count valid candidate matches around every candidate pair."""
    if tolerance_bins == 0:
        return np.ones(len(chrom), dtype=np.int16)

    from scipy.ndimage import convolve

    chrom_values = np.asarray(chrom).astype(str)
    left_values = np.asarray(bin_i, dtype=np.int64) // bin_size_bp
    right_values = np.asarray(bin_j, dtype=np.int64) // bin_size_bp
    sizes = np.empty(len(chrom_values), dtype=np.int16)
    kernel = np.ones(
        (2 * tolerance_bins + 1, 2 * tolerance_bins + 1), dtype=np.int16
    )
    for chromosome in np.unique(chrom_values):
        indices = np.flatnonzero(chrom_values == chromosome)
        left = left_values[indices] - left_values[indices].min()
        right = right_values[indices] - right_values[indices].min()
        coordinates = np.column_stack([left, right])
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("Candidate pair coordinates must be unique within a tile")
        grid = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int16)
        grid[left, right] = 1
        neighborhood_counts = convolve(grid, kernel, mode="constant", cval=0)
        sizes[indices] = neighborhood_counts[left, right]
    return sizes


def neighborhood_match_chance(
    chrom: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    tolerance_bins: int,
    bin_size_bp: int = 5_000,
    *,
    neighborhood_sizes: np.ndarray | None = None,
) -> float:
    """Exact null match probability for two random fixed-size top sets.

    For each selected query, the hypergeometric survival probability is the
    chance that a random reference set contains at least one valid candidate
    in its coordinate neighborhood. The two directions are averaged to match
    the symmetric observed-overlap definition.
    """
    if not len(selected_a) or not len(selected_b) or not len(chrom):
        return float("nan")
    sizes = (
        candidate_neighborhood_sizes(
            chrom, bin_i, bin_j, tolerance_bins, bin_size_bp
        )
        if neighborhood_sizes is None
        else np.asarray(neighborhood_sizes)
    )
    population_size = len(chrom)

    def directional(query: np.ndarray, reference_size: int) -> float:
        probability = hypergeom.sf(
            0,
            population_size,
            sizes[query],
            reference_size,
        )
        return float(np.mean(probability))

    return float(
        np.mean(
            [
                directional(selected_a, len(selected_b)),
                directional(selected_b, len(selected_a)),
            ]
        )
    )


def fixed_distance_oe_scores(
    counts: np.ndarray,
    reference_counts: np.ndarray,
    distance_bin: np.ndarray,
    training: np.ndarray,
) -> np.ndarray:
    """Apply one shared training-derived fixed observed/expected transform."""
    counts = np.asarray(counts, dtype=np.float64)
    reference = np.asarray(reference_counts, dtype=np.float64)
    distances = np.asarray(distance_bin, dtype=np.int64)
    training = np.asarray(training, dtype=bool)
    if not (len(counts) == len(reference) == len(distances) == len(training)):
        raise ValueError("Fixed O/E inputs must have equal lengths")
    if not training.any():
        raise ValueError("Fixed O/E requires training candidates")
    size = int(distances.max()) + 1
    training_distances = distances[training]
    candidate_counts = np.bincount(training_distances, minlength=size)
    contact_sums = np.bincount(
        training_distances,
        weights=reference[training],
        minlength=size,
    )
    expected = np.full(size, np.nan, dtype=np.float64)
    represented = candidate_counts > 0
    expected[represented] = contact_sums[represented] / candidate_counts[represented]
    positive = expected[np.isfinite(expected) & (expected > 0)]
    if not len(positive):
        raise ValueError("Fixed O/E training curve is entirely zero")
    epsilon = float(positive.min())
    selected_expected = expected[distances]
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.log((counts + epsilon) / (selected_expected + epsilon))
    scores[~np.isfinite(selected_expected) | (selected_expected <= 0)] = np.nan
    return scores.astype(np.float32)


def build_top_contact_groups(
    tile_row: np.ndarray,
    distance_band: np.ndarray,
) -> list[tuple[str, int, np.ndarray]]:
    """Index every band/tile group in one O(N log N) pass."""
    tile_values = np.asarray(tile_row)
    band_values = np.asarray(distance_band).astype(str)
    if len(tile_values) != len(band_values):
        raise ValueError("Tile and distance-band arrays must have equal lengths")
    if not len(tile_values):
        return []
    _, band_codes = np.unique(band_values, return_inverse=True)
    order = np.lexsort((tile_values, band_codes))
    ordered_band = band_codes[order]
    ordered_tile = tile_values[order]
    boundaries = np.flatnonzero(
        (ordered_band[1:] != ordered_band[:-1])
        | (ordered_tile[1:] != ordered_tile[:-1])
    ) + 1
    groups = []
    for indices in np.split(order, boundaries):
        groups.append(
            (
                str(band_values[indices[0]]),
                int(tile_values[indices[0]]),
                indices,
            )
        )
    return groups


def grouped_top_contact_metrics(
    chrom: np.ndarray,
    bin_i: np.ndarray,
    bin_j: np.ndarray,
    tile_row: np.ndarray,
    distance_band: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    fractions: list[float] | tuple[float, ...],
    tolerances: list[int] | tuple[int, ...],
    bin_size_bp: int = 5_000,
    neighborhood_size_cache: dict[tuple[str, int, int], np.ndarray] | None = None,
    groups: list[tuple[str, int, np.ndarray]] | None = None,
    progress_desc: str | None = None,
    tie_mode: str = "hard_cutoff",
) -> list[dict[str, Any]]:
    """Compute tile-level top-contact metrics and median-aggregate by band."""
    arrays = (
        chrom,
        bin_i,
        bin_j,
        tile_row,
        distance_band,
        score_a,
        score_b,
    )
    if any(len(values) != len(chrom) for values in arrays):
        raise ValueError("Top-contact metric inputs must have equal lengths")

    tile_values = np.asarray(tile_row)
    band_values = np.asarray(distance_band).astype(str)
    score_a_values = np.asarray(score_a)
    score_b_values = np.asarray(score_b)
    finite_scores = np.isfinite(score_a_values) & np.isfinite(score_b_values)
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    candidate_groups = (
        build_top_contact_groups(tile_values, band_values) if groups is None else groups
    )
    group_iterator = tqdm(
        candidate_groups,
        desc=progress_desc or "Top-contact groups",
        unit="group",
        leave=False,
        mininterval=5.0,
        disable=progress_desc is None,
    )
    for band, tile, group_indices in group_iterator:
        indices = group_indices[finite_scores[group_indices]]
        if len(indices) < 2:
            continue
        cache_is_safe = len(indices) == len(group_indices)
        local_chrom = np.asarray(chrom)[indices]
        local_i = np.asarray(bin_i)[indices]
        local_j = np.asarray(bin_j)[indices]
        local_a = score_a_values[indices]
        local_b = score_b_values[indices]
        sizes_by_tolerance = {}
        for tolerance_value in tolerances:
            tolerance = int(tolerance_value)
            cache_key = (band, int(tile), tolerance)
            sizes = (
                neighborhood_size_cache.get(cache_key)
                if neighborhood_size_cache is not None and cache_is_safe
                else None
            )
            if sizes is None:
                sizes = candidate_neighborhood_sizes(
                    local_chrom,
                    local_i,
                    local_j,
                    tolerance,
                    bin_size_bp,
                )
                if neighborhood_size_cache is not None and cache_is_safe:
                    neighborhood_size_cache[cache_key] = sizes
            sizes_by_tolerance[tolerance] = sizes
        order_a = np.argsort(-local_a, kind="stable")
        order_b = np.argsort(-local_b, kind="stable")
        for fraction in fractions:
            selected_a, tied_a = _select_top_from_order(
                local_a, float(fraction), order_a, tie_mode
            )
            selected_b, tied_b = _select_top_from_order(
                local_b, float(fraction), order_b, tie_mode
            )
            for tolerance in tolerances:
                tolerance = int(tolerance)
                overlap = float(
                    np.nanmean(
                        [
                            neighborhood_overlap(
                                local_chrom,
                                local_i,
                                local_j,
                                selected_a,
                                selected_b,
                                tolerance,
                                bin_size_bp,
                            ),
                            neighborhood_overlap(
                                local_chrom,
                                local_i,
                                local_j,
                                selected_b,
                                selected_a,
                                tolerance,
                                bin_size_bp,
                            ),
                        ]
                    )
                )
                chance = neighborhood_match_chance(
                    local_chrom,
                    local_i,
                    local_j,
                    selected_a,
                    selected_b,
                    tolerance,
                    bin_size_bp,
                    neighborhood_sizes=sizes_by_tolerance[tolerance],
                )
                grouped[(band, float(fraction), tolerance)].append(
                    {
                        "overlap": overlap,
                        "chance": chance,
                        "enrichment": (
                            overlap / chance
                            if np.isfinite(overlap) and chance > 0
                            else np.nan
                        ),
                        "a_tied": bool(tied_a),
                        "b_tied": bool(tied_b),
                    }
                )

    output = []
    for (band, fraction, tolerance), rows in sorted(grouped.items()):
        output.append(
            {
                "band": band,
                "top_fraction": fraction,
                "match_tolerance_bins": tolerance,
                "overlap": float(np.nanmedian([row["overlap"] for row in rows])),
                "chance_overlap": float(
                    np.nanmedian([row["chance"] for row in rows])
                ),
                "enrichment_over_chance": float(
                    np.nanmedian([row["enrichment"] for row in rows])
                ),
                "candidate_tile_count": len(rows),
                "tie_mode": tie_mode,
                "score_a_cutoff_tied_tiles": sum(row["a_tied"] for row in rows),
                "score_b_cutoff_tied_tiles": sum(row["b_tied"] for row in rows),
            }
        )
    return output


def metric_bundle(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    minimum_candidates: int = 100,
    minimum_supported_candidates: int = 1,
) -> dict[str, Any]:
    finite = np.isfinite(prediction) & np.isfinite(target)
    if finite.sum() < minimum_candidates:
        return {
            "defined": False,
            "candidate_count": int(finite.sum()),
            "reason": "insufficient_candidates",
        }
    pred = prediction[finite]
    truth = target[finite]
    labels = support[finite].astype(bool)
    pearson = pearsonr(pred, truth).statistic if np.std(truth) > 0 else np.nan
    spearman = spearmanr(pred, truth).statistic if np.unique(truth).size > 1 else np.nan
    support_count = int(labels.sum())
    auprc_defined = (
        support_count >= minimum_supported_candidates
        and support_count < len(labels)
    )
    auprc = average_precision_score(labels, pred) if auprc_defined else np.nan
    return {
        "defined": True,
        "candidate_count": int(len(pred)),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "auprc": float(auprc),
        "auprc_defined": bool(auprc_defined),
        "support_count": support_count,
        "support_prevalence": float(labels.mean()),
    }


def chromosome_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | None]:
    by_chrom: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            by_chrom[str(row["chrom"])].append(float(value))
    chromosomes = sorted(by_chrom)
    if len(chromosomes) < 2:
        return {"lower": None, "median": None, "upper": None}
    values = np.array([np.mean(by_chrom[c]) for c in chromosomes])
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(1)
    lower, median, upper = np.quantile(sampled, [0.025, 0.5, 0.975])
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}
