from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc
from tqdm.auto import tqdm

from .common import atomic_json, source_record, update_manifest


def area_overlap_matrix(
    input_start: int,
    target_start: int,
    *,
    native_bin_size: int = 2_048,
    target_bin_size: int = 5_000,
    native_bins: int = 512,
    target_bins: int = 200,
) -> np.ndarray:
    native_start = input_start + np.arange(native_bins, dtype=np.int64) * native_bin_size
    target_start_array = target_start + np.arange(target_bins, dtype=np.int64) * target_bin_size
    overlap = np.maximum(
        0,
        np.minimum(target_start_array[:, None] + target_bin_size, native_start[None, :] + native_bin_size)
        - np.maximum(target_start_array[:, None], native_start[None, :]),
    )
    weights = overlap.astype(np.float32) / target_bin_size
    if not np.allclose(weights.sum(1), 1.0, atol=1e-6):
        raise ValueError("Target bins are not fully covered by native bins")
    return weights


def _extract_batch(
    embedding: np.ndarray,
    weights: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = embedding.shape[-1]
    output_pair = np.empty((len(left), channels), np.float32)
    output_i = np.empty_like(output_pair)
    output_j = np.empty_like(output_pair)
    for offset in range(0, len(left), 4096):
        selected = slice(offset, min(offset + 4096, len(left)))
        li, lj = left[selected], right[selected]
        wi, wj = weights[li], weights[lj]
        overlap_width = max(
            int(np.max(np.count_nonzero(wi, axis=1))),
            int(np.max(np.count_nonzero(wj, axis=1))),
        )
        ii = np.argsort(-wi, axis=1)[:, :overlap_width]
        jj = np.argsort(-wj, axis=1)[:, :overlap_width]
        wi4 = np.take_along_axis(wi, ii, axis=1)
        wj4 = np.take_along_axis(wj, jj, axis=1)
        block = embedding[ii[:, :, None], jj[:, None, :], :].astype(np.float32)
        reverse_block = embedding[jj[:, :, None], ii[:, None, :], :].astype(np.float32)
        forward_feature = np.einsum("bxyc,bx,by->bc", block, wi4, wj4, optimize=True)
        reverse_feature = np.einsum("bxyc,bx,by->bc", reverse_block, wj4, wi4, optimize=True)
        output_pair[selected] = 0.5 * (forward_feature + reverse_feature)
        diag_i = embedding[ii[:, :, None], ii[:, None, :], :].astype(np.float32)
        diag_j = embedding[jj[:, :, None], jj[:, None, :], :].astype(np.float32)
        output_i[selected] = np.einsum("bxyc,bx,by->bc", diag_i, wi4, wi4, optimize=True)
        output_j[selected] = np.einsum("bxyc,bx,by->bc", diag_j, wj4, wj4, optimize=True)
    return output_pair, output_i, output_j


def extract_features(
    config: dict[str, Any],
    pairs_path: Path,
) -> dict[str, Any]:
    tiles = pd.read_parquet(config["paths"]["tiles"]).reset_index(drop=True)
    embedding_manifest = pd.read_parquet(config["paths"]["embedding_manifest"])
    manifest_by_file = embedding_manifest.set_index("alphagenome_embedding_file")
    pairs = pq.read_table(
        pairs_path, columns=["pair_id", "tile_row", "bin_i", "bin_j", "split"]
    ).to_pandas()
    pair_count = len(pairs)
    output_root = Path(config["outputs"]["data_root"])
    store_path = output_root / "pair_features.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    channels = int(config["features"]["pair_channels"])
    native_bin_size = int(config["features"]["native_bin_size_bp"])
    native_bins = int(config["features"]["native_bins"])
    target_bin_size = int(config["bin_size_bp"])
    target_bins = 1_000_000 // target_bin_size
    chunk = int(config["features"]["chunk_pairs"])
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arrays = {
        name: root.create_dataset(
            name, shape=(pair_count, channels), chunks=(chunk, channels),
            dtype="f2", compressor=compressor
        )
        for name in ("pair_embedding", "anchor_i_embedding", "anchor_j_embedding")
    }
    root.create_dataset(
        "pair_id", data=np.arange(pair_count, dtype=np.int64),
        chunks=(min(1_000_000, pair_count),), compressor=compressor
    )
    embedding_root = Path(config["paths"]["embeddings"])
    missing = []
    extracted = 0
    rng = np.random.default_rng(int(config["seed"]))
    verification_ids = rng.choice(pair_count, min(1_000, pair_count), replace=False)
    maximum_verification_error = 0.0
    tile_groups = pairs.groupby("tile_row", observed=True, sort=True)
    for tile_row, group in tqdm(
        tile_groups, total=tile_groups.ngroups, desc="Extract pair features", unit="tile"
    ):
        tile = tiles.iloc[int(tile_row)]
        stored = Path(str(tile["embedding_path"]))
        candidates = [
            stored,
            embedding_root / stored.name,
            embedding_root / f"{tile.chrom}_{int(tile.input_start)}.npy",
        ]
        embedding_path = next((path for path in candidates if path.exists()), None)
        if embedding_path is None:
            missing.append(str(tile["tile_id"]))
            continue
        if embedding_path.name not in manifest_by_file.index:
            raise ValueError(f"Embedding is absent from frozen manifest: {embedding_path.name}")
        manifest_row = manifest_by_file.loc[embedding_path.name]
        if (
            str(manifest_row["chrom"]) != str(tile.chrom)
            or int(manifest_row["window_start"]) != int(tile.input_start)
            or int(manifest_row["window_end"]) != int(tile.input_end)
        ):
            raise ValueError(
                f"Embedding coordinate mismatch for {tile.tile_id}: "
                f"manifest={manifest_row[[chr(99)+chr(104)+chr(114)+chr(111)+chr(109), chr(119)+chr(105)+chr(110)+chr(100)+chr(111)+chr(119)+chr(95)+chr(115)+chr(116)+chr(97)+chr(114)+chr(116), chr(119)+chr(105)+chr(110)+chr(100)+chr(111)+chr(119)+chr(95)+chr(101)+chr(110)+chr(100)]].to_dict()} "
                f"tile=({tile.chrom}, {tile.input_start}, {tile.input_end})"
            )
        embedding = np.load(embedding_path, mmap_mode="r")
        if embedding.shape != (native_bins, native_bins, channels):
            raise ValueError(f"Unexpected embedding shape at {embedding_path}: {embedding.shape}")
        weights = area_overlap_matrix(
            int(tile.input_start),
            int(tile.target_start),
            native_bin_size=native_bin_size,
            target_bin_size=target_bin_size,
            native_bins=native_bins,
            target_bins=target_bins,
        )
        left = (
            (group["bin_i"].to_numpy(np.int64) - int(tile.target_start))
            // target_bin_size
        ).astype(np.int64)
        right = (
            (group["bin_j"].to_numpy(np.int64) - int(tile.target_start))
            // target_bin_size
        ).astype(np.int64)
        if np.any(left < 0) or np.any(right >= target_bins):
            raise ValueError(f"Canonical pair lies outside selected tile {tile.tile_id}")
        pair_feature, anchor_i, anchor_j = _extract_batch(embedding, weights, left, right)
        ids = group["pair_id"].to_numpy(np.int64)
        arrays["pair_embedding"].oindex[ids, :] = pair_feature.astype(np.float16)
        arrays["anchor_i_embedding"].oindex[ids, :] = anchor_i.astype(np.float16)
        arrays["anchor_j_embedding"].oindex[ids, :] = anchor_j.astype(np.float16)
        verify_positions = np.flatnonzero(np.isin(ids, verification_ids))
        for name, direct in (
            ("pair_embedding", pair_feature),
            ("anchor_i_embedding", anchor_i),
            ("anchor_j_embedding", anchor_j),
        ):
            if len(verify_positions):
                cached = np.asarray(arrays[name].oindex[ids[verify_positions], :], np.float32)
                maximum_verification_error = max(
                    maximum_verification_error,
                    float(np.max(np.abs(cached - direct[verify_positions]))),
                )
        extracted += len(ids)
    if missing:
        raise FileNotFoundError(f"Missing embeddings for {len(missing)} tiles; examples: {missing[:10]}")
    if extracted != pair_count:
        raise RuntimeError(f"Extracted {extracted} features for {pair_count} pairs")
    if maximum_verification_error > 0.01:
        raise RuntimeError(
            f"Cached features disagree with direct extraction: {maximum_verification_error}"
        )
    root.attrs.update(
        {
            "schema_version": 1,
            "pair_count": pair_count,
            "channels": channels,
            "native_bin_size_bp": native_bin_size,
            "target_bin_size_bp": target_bin_size,
            "target_bins": target_bins,
            "maximum_native_overlaps_per_target_bin": int(
                np.max(
                    np.count_nonzero(
                        area_overlap_matrix(
                            int(tiles.iloc[0].input_start),
                            int(tiles.iloc[0].target_start),
                            native_bin_size=native_bin_size,
                            target_bin_size=target_bin_size,
                            native_bins=native_bins,
                            target_bins=target_bins,
                        ),
                        axis=1,
                    )
                )
            ),
            "extraction": "area-weighted exact pair and diagonal anchor blocks",
            "alphagenome_frozen": True,
            "embedding_manifest_verified": True,
        }
    )
    rng = np.random.default_rng(int(config["seed"]))
    verification_ids = rng.choice(pair_count, min(1_000, pair_count), replace=False)
    finite = all(
        np.isfinite(np.asarray(array.oindex[verification_ids, :], np.float32)).all()
        for array in arrays.values()
    )
    report = {
        "store": str(store_path),
        "pair_count": pair_count,
        "feature_shape": [pair_count, channels],
        "aligned_pair_ids": True,
        "direct_verification_pairs": len(verification_ids),
        "direct_verification_finite": finite,
        "direct_verification_max_absolute_error": maximum_verification_error,
        "embedding_manifest_verified": True,
        "embedding_manifest_source": source_record(Path(config["paths"]["embedding_manifest"])),
        "split_counts": pairs["split"].value_counts().to_dict(),
        "source": source_record(Path(config["paths"]["tiles"])),
    }
    atomic_json(output_root / "feature_extraction_report.json", report)
    update_manifest(output_root, "features", report)
    return report
