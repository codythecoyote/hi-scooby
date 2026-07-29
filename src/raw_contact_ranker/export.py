from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from numcodecs import Blosc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import nbinom
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    configured_distance_bands,
    distance_range_bp,
    resolution_contract,
    sha256_file,
)


CLAIM = (
    "scHiCAR_expected_long_range_contact_rate_at_10kb_per_million_assay_pairs"
)


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _load_shared_predictions(
    pairs: pd.DataFrame,
    paths: Iterable[Path],
) -> np.ndarray:
    frames = [
        pd.read_parquet(
            path, columns=["pair_id", "shared_residual_score"]
        )
        for path in paths
    ]
    prediction = pd.concat(frames, ignore_index=True).sort_values(
        "pair_id", kind="stable"
    )
    if prediction["pair_id"].duplicated().any():
        raise ValueError("Shared prediction files overlap")
    if not np.array_equal(
        prediction["pair_id"].to_numpy(np.int64),
        pairs["pair_id"].to_numpy(np.int64),
    ):
        raise ValueError("Shared prediction files do not cover every pair")
    values = prediction["shared_residual_score"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Shared residual predictions are non-finite")
    return values


def _context_delta(
    output_row: dict[str, Any],
    pairs: pd.DataFrame,
    cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> np.ndarray:
    path = output_row.get("context_delta_path")
    column = output_row.get("context_delta_column")
    if path is None and column is None:
        return np.zeros(len(pairs), np.float64)
    if not path or not column:
        raise ValueError("Context delta path and column must be specified together")
    key = (str(path), str(column))
    if cache is not None and key in cache:
        return cache[key]
    frame = pd.read_parquet(
        path, columns=["pair_id", str(column)]
    ).sort_values("pair_id", kind="stable")
    if not np.array_equal(
        frame["pair_id"].to_numpy(np.int64),
        pairs["pair_id"].to_numpy(np.int64),
    ):
        raise ValueError(
            f"Context delta does not cover all pairs for {output_row['output_id']}"
        )
    values = frame[str(column)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Context delta contains non-finite values")
    if cache is not None:
        cache[key] = values
    return values


def _owner_band_probability(
    pairs: pd.DataFrame,
    expected: np.ndarray,
    band_ids: list[str],
) -> np.ndarray:
    band_lookup = {band: index for index, band in enumerate(band_ids)}
    band_code = pairs["distance_band"].astype(str).map(band_lookup)
    if band_code.isna().any():
        raise ValueError("Canonical pairs contain an unconfigured distance band")
    group = (
        pairs["tile_row"].to_numpy(np.int64) * len(band_ids)
        + band_code.to_numpy(np.int64)
    )
    totals = np.bincount(group, weights=expected)
    denominator = totals[group]
    if np.any(denominator <= 0):
        raise RuntimeError("Owner tile-band expected-rate total is nonpositive")
    output = expected / denominator
    if not np.allclose(
        np.bincount(group, weights=output)[np.unique(group)],
        np.ones(len(np.unique(group))),
        atol=1e-6,
    ):
        raise RuntimeError("Owner tile-band probabilities do not normalize")
    return output


def _predictive_bounds(
    expected: np.ndarray,
    dispersion: np.ndarray,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < probability < 1:
        raise ValueError("Predictive interval probability must be in (0, 1)")
    if np.any(expected <= 0) or np.any(dispersion <= 0):
        raise ValueError("NB2 prediction parameters must be positive")
    size = 1.0 / dispersion
    success = size / (size + expected)
    tail = (1.0 - probability) / 2.0
    lower = nbinom.ppf(tail, size, success)
    upper = nbinom.ppf(1.0 - tail, size, success)
    return lower.astype(np.float64), upper.astype(np.float64)


def _create_tile_pair_grid(
    config: dict[str, Any],
    pairs: pd.DataFrame,
    tiles: pd.DataFrame,
    store,
) -> tuple[Any, Any]:
    bin_size = int(config["bin_size_bp"])
    target_bins = 1_000_000 // bin_size
    minimum, maximum_exclusive = distance_range_bp(config)
    min_bin = minimum // bin_size
    max_bin = maximum_exclusive // bin_size - 1
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunk_tiles = 64
    pair_grid = store.create_dataset(
        "pair_id",
        shape=(len(tiles), target_bins, target_bins),
        chunks=(chunk_tiles, target_bins, target_bins),
        dtype="i8",
        fill_value=-1,
        compressor=compressor,
    )
    valid = store.create_dataset(
        "valid_mask",
        shape=(len(tiles), target_bins, target_bins),
        chunks=(chunk_tiles, target_bins, target_bins),
        dtype="bool",
        fill_value=False,
        compressor=compressor,
    )
    for chrom, chrom_tiles in tqdm(
        tiles.groupby("chrom", observed=True, sort=False),
        desc="Build visualization pair grid",
        unit="chrom",
    ):
        chrom_pairs = pairs.loc[pairs["chrom"].astype(str).eq(str(chrom))]
        if chrom_pairs.empty:
            continue
        left_bins = chrom_pairs["bin_i"].to_numpy(np.int64) // bin_size
        right_bins = chrom_pairs["bin_j"].to_numpy(np.int64) // bin_size
        stride = int(max(right_bins.max(initial=0), left_bins.max(initial=0))) + 1
        keys = left_bins * stride + right_bins
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_ids = chrom_pairs["pair_id"].to_numpy(np.int64)[order]
        for tile_row, tile in chrom_tiles.iterrows():
            start_bin = int(tile["target_start"]) // bin_size
            matrix = np.full((target_bins, target_bins), -1, np.int64)
            for distance in range(min_bin, max_bin + 1):
                local_i = np.arange(target_bins - distance, dtype=np.int64)
                local_j = local_i + distance
                candidate_keys = (
                    (start_bin + local_i) * stride + start_bin + local_j
                )
                positions = np.searchsorted(sorted_keys, candidate_keys)
                within = positions < len(sorted_keys)
                safe = np.minimum(positions, len(sorted_keys) - 1)
                present = within & (sorted_keys[safe] == candidate_keys)
                if present.any():
                    ids = sorted_ids[safe[present]]
                    i = local_i[present]
                    j = local_j[present]
                    matrix[i, j] = ids
                    matrix[j, i] = ids
            pair_grid[int(tile_row)] = matrix
            valid[int(tile_row)] = matrix >= 0
    return pair_grid, valid


def _scatter_output(
    destination: dict[str, Any],
    pair_grid,
    pairs: pd.DataFrame,
    rates: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    residual: np.ndarray,
    *,
    output_index: int,
    band_ids: list[str],
) -> None:
    band_lookup = {band: index for index, band in enumerate(band_ids)}
    pair_band = (
        pairs["distance_band"].astype(str).map(band_lookup).to_numpy(np.int16)
    )
    tile_count = pair_grid.shape[0]
    for start in tqdm(
        range(0, tile_count, 64),
        desc=f"Scatter output {output_index}",
        unit="tile-block",
    ):
        stop = min(start + 64, tile_count)
        ids = np.asarray(pair_grid[start:stop], np.int64)
        valid = ids >= 0
        safe = np.maximum(ids, 0)
        rate_view = np.full(ids.shape, np.nan, np.float32)
        lower_view = np.full(ids.shape, np.nan, np.float32)
        upper_view = np.full(ids.shape, np.nan, np.float32)
        residual_view = np.full(ids.shape, np.nan, np.float32)
        probability_view = np.full(ids.shape, np.nan, np.float32)
        rate_view[valid] = rates[safe[valid]].astype(np.float32)
        lower_view[valid] = lower[safe[valid]].astype(np.float32)
        upper_view[valid] = upper[safe[valid]].astype(np.float32)
        residual_view[valid] = residual[safe[valid]].astype(np.float32)
        for local_tile in range(stop - start):
            tile_valid = valid[local_tile]
            tile_ids = ids[local_tile]
            upper = np.triu(np.ones(tile_ids.shape, bool), k=1)
            for band_index in range(len(band_ids)):
                selected = tile_valid & (
                    pair_band[np.maximum(tile_ids, 0)] == band_index
                )
                total = float(
                    rate_view[local_tile][selected & upper].sum()
                )
                if selected.any() and total > 0:
                    probability_view[local_tile][selected] = (
                        rate_view[local_tile][selected] / total
                    )
        destination["expected_contacts_per_million"][
            output_index, start:stop
        ] = rate_view
        destination["predictive_lower"][
            output_index, start:stop
        ] = lower_view
        destination["predictive_upper"][
            output_index, start:stop
        ] = upper_view
        destination["residual_score"][
            output_index, start:stop
        ] = residual_view
        destination["tile_band_probability"][
            output_index, start:stop
        ] = probability_view


def export_rate_maps(
    config: dict[str, Any],
    *,
    prediction_paths: Iterable[Path],
    calibration_path: Path,
    rollout_path: Path,
    frozen_release: Path,
    final_test_gate: Path,
    canonical_output: Path,
    map_output: Path,
) -> dict[str, Any]:
    if int(config["bin_size_bp"]) != 10_000:
        raise ValueError("Public long-range rate export requires 10 kb resolution")
    if canonical_output.exists() or map_output.exists():
        raise FileExistsError(
            "Public prediction outputs already exist; export is immutable"
        )
    prediction_paths = [Path(path) for path in prediction_paths]
    final_gate = _read(final_test_gate)
    frozen = _read(frozen_release)
    calibration = _read(calibration_path)
    rollout = _read(rollout_path)
    selection_path = Path(frozen["inputs"]["selection"]["path"])
    if sha256_file(selection_path) != frozen["inputs"]["selection"]["sha256"]:
        raise RuntimeError("Feature selection changed after release freeze")
    selection = _read(selection_path)
    selected_feature_set = str(selection["selected_feature_set"])
    if not final_gate.get("accepted"):
        raise RuntimeError("Final untouched-test gate did not pass")
    if not frozen.get("frozen") or not calibration.get("accepted"):
        raise RuntimeError("Release inputs are not accepted and frozen")
    if (
        final_gate["artifacts"]["frozen_release"]["sha256"]
        != sha256_file(frozen_release)
    ):
        raise RuntimeError("Final gate does not authorize this frozen release")
    if (
        frozen["inputs"]["calibration_gate"]["sha256"]
        != sha256_file(calibration_path)
    ):
        raise RuntimeError("Calibration changed after release freeze")
    if frozen["inputs"]["rollout"]["sha256"] != sha256_file(rollout_path):
        raise RuntimeError("Context rollout changed after release freeze")
    exact_test_report = _read(
        Path(final_gate["artifacts"]["exact_evaluation"]["path"])
    )
    final_topology_report = _read(
        Path(final_gate["artifacts"]["topology_gate"]["path"])
    )
    expected_prediction_hashes = {
        str(calibration["training_predictions_sha256"]),
        str(calibration["validation_predictions_sha256"]),
        str(final_topology_report["prediction_sha256"]),
    }
    actual_prediction_hashes = {
        sha256_file(path) for path in prediction_paths
    }
    if (
        len(prediction_paths) != 3
        or actual_prediction_hashes != expected_prediction_hashes
    ):
        raise RuntimeError(
            "Shared prediction inputs do not match the frozen train, "
            "validation, and one-shot test artifacts"
        )
    if (
        sha256_file(Path(exact_test_report["prediction_path"]))
        != final_topology_report["prediction_sha256"]
    ):
        raise RuntimeError("One-shot test predictions changed after final gate")
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(data_root / "canonical_pairs.parquet").sort_values(
        "pair_id", kind="stable"
    ).reset_index(drop=True)
    tiles = pd.read_parquet(config["paths"]["tiles"]).reset_index(drop=True)
    if not np.array_equal(
        pairs["pair_id"].to_numpy(np.int64),
        np.arange(len(pairs), dtype=np.int64),
    ):
        raise ValueError("Canonical pair IDs are not row aligned")
    shared = _load_shared_predictions(pairs, prediction_paths)
    band_ids = [str(row["id"]) for row in configured_distance_bands(config)]
    outputs = list(rollout["outputs"])
    output_ids = [str(row["output_id"]) for row in outputs]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Context rollout contains duplicate output IDs")
    model_hash = frozen["inputs"]["checkpoint"]["sha256"]
    model_id = f"raw_contact_ranker_10kb_v1:{model_hash[:16]}"
    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_canonical = canonical_output.with_suffix(
        canonical_output.suffix + ".tmp"
    )
    writer: pq.ParquetWriter | None = None
    store = zarr.open_group(str(map_output), mode="w")
    pair_grid, valid = _create_tile_pair_grid(config, pairs, tiles, store)
    target_bins = 1_000_000 // int(config["bin_size_bp"])
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arrays = {
        name: store.create_dataset(
            name,
            shape=(len(outputs), len(tiles), target_bins, target_bins),
            chunks=(1, 64, target_bins, target_bins),
            dtype="f4",
            fill_value=np.nan,
            compressor=compressor,
        )
        for name in (
            "expected_contacts_per_million",
            "predictive_lower",
            "predictive_upper",
            "residual_score",
            "tile_band_probability",
        )
    }
    interval_probability = float(
        config["calibration"]["interval_probability"]
    )
    delta_cache: dict[tuple[str, str], np.ndarray] = {}
    try:
        for output_index, output_row in enumerate(
            tqdm(
                outputs,
                desc="Export calibrated map outputs",
                unit="output",
            )
        ):
            output_id = str(output_row["output_id"])
            calibration_source = str(output_row["calibration_source"])
            if calibration_source not in calibration["parameters"]:
                raise RuntimeError(
                    f"Calibration lacks rollout source {calibration_source}"
                )
            delta = _context_delta(output_row, pairs, delta_cache)
            final_residual = shared + delta
            expected = np.empty(len(pairs), np.float64)
            dispersion = np.empty(len(pairs), np.float64)
            for band in band_ids:
                selected = pairs["distance_band"].astype(str).eq(band).to_numpy()
                parameters = calibration["parameters"][calibration_source][
                    "bands"
                ][band]["model"]
                log_rate = (
                    np.log(
                        np.clip(
                            pairs.loc[selected, "exposure"].to_numpy(float),
                            1e-12,
                            None,
                        )
                    )
                    + pairs.loc[
                        selected, "distance_offset"
                    ].to_numpy(float)
                    + final_residual[selected]
                    + float(parameters["alpha"])
                )
                expected[selected] = np.exp(np.clip(log_rate, -40, 30))
                dispersion[selected] = float(parameters["dispersion"])
            if (
                not np.isfinite(expected).all()
                or np.any(expected < 0)
                or not np.isfinite(final_residual).all()
            ):
                raise RuntimeError(f"Invalid expected rates for {output_id}")
            lower, upper = _predictive_bounds(
                expected, dispersion, interval_probability
            )
            owner_probability = _owner_band_probability(
                pairs, expected, band_ids
            )
            for start in range(0, len(pairs), 1_000_000):
                stop = min(start + 1_000_000, len(pairs))
                frame = pd.DataFrame(
                    {
                        "pair_id": pairs.loc[
                            start : stop - 1, "pair_id"
                        ].to_numpy(np.int64),
                        "output_id": output_id,
                        "output_type": str(output_row["output_type"]),
                        "topology_source": str(
                            output_row["topology_source"]
                        ),
                        "shared_residual": shared[start:stop].astype(
                            np.float32
                        ),
                        "context_delta": delta[start:stop].astype(np.float32),
                        "final_residual": final_residual[start:stop].astype(
                            np.float32
                        ),
                        "expected_contacts_per_million": expected[
                            start:stop
                        ].astype(np.float32),
                        "nb_predictive_lower_95": lower[start:stop].astype(
                            np.float32
                        ),
                        "nb_predictive_upper_95": upper[start:stop].astype(
                            np.float32
                        ),
                        "owner_tile_band_probability": owner_probability[
                            start:stop
                        ].astype(np.float32),
                        "model_id": model_id,
                        "checkpoint_hash": model_hash,
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_canonical,
                        table.schema,
                        compression="zstd",
                        use_dictionary=[
                            "output_id",
                            "output_type",
                            "topology_source",
                            "model_id",
                            "checkpoint_hash",
                        ],
                    )
                writer.write_table(table)
            _scatter_output(
                arrays,
                pair_grid,
                pairs,
                expected,
                lower,
                upper,
                final_residual,
                output_index=output_index,
                band_ids=band_ids,
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("Canonical prediction export produced no rows")
    temporary_canonical.replace(canonical_output)
    store.attrs.update(
        {
            "schema_version": 1,
            "claim": CLAIM,
            "resolution": resolution_contract(config),
            "output_ids": output_ids,
            "output_types": [
                str(row["output_type"]) for row in outputs
            ],
            "units": "expected contacts per one million filtered cis scHiCAR pairs",
            "depth_scaling": (
                "multiply expected_contacts_per_million by "
                "filtered_cis_pair_depth/1000000"
            ),
            "canonical_pair_table_authoritative": True,
            "visualization_tiles_overlap": True,
            "symmetric": True,
            "masked_outside_modeled_distance": True,
            "checkpoint_sha256": model_hash,
            "selected_feature_set": selected_feature_set,
            "alphagenome_claim_authorized": selected_feature_set
            in {"alphagenome", "combined"},
            "frozen_release_sha256": sha256_file(frozen_release),
            "final_test_gate_sha256": sha256_file(final_test_gate),
        }
    )
    report = {
        "schema_version": 1,
        "claim": CLAIM,
        "canonical_predictions": str(canonical_output),
        "contact_maps": str(map_output),
        "pair_count": len(pairs),
        "outputs": outputs,
        "selected_feature_set": selected_feature_set,
        "alphagenome_claim_authorized": selected_feature_set
        in {"alphagenome", "combined"},
        "tiles": len(tiles),
        "map_shape": [len(outputs), len(tiles), target_bins, target_bins],
        "map_chunks": [1, 64, target_bins, target_bins],
        "nonnegative": True,
        "symmetric": True,
        "duplicate_appearances_identical_by_canonical_scatter": True,
        "tile_band_probability_upper_triangle_normalized": True,
        "depth_scalable": True,
        "test_gate": str(final_test_gate),
    }
    atomic_json(
        canonical_output.parent / "map_export_report.json",
        report,
    )
    return report
