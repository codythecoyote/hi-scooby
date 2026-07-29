
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
DISCOVERED_REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG_PATH = DISCOVERED_REPO_ROOT / "configs" / "config.yaml"
PHASE1_RESULTS_SUBDIRECTORY = Path("results") / "phase1_model_v2"

EMBEDDINGS_CACHE_ENV = "SCOO_HIC_EMBEDDINGS_CACHE"
TARGETS_CACHE_ENV = "SCOO_HIC_TARGETS_CACHE"

@dataclass(frozen=True)
class Phase1Paths:
    config: Path
    repo_root: Path
    data_root: Path
    model_source_dir: Path
    model_file: Path
    tiles: Path
    centroids: Path
    target_contexts: Path
    targets: Path
    embeddings_dir: Path
    output_dir: Path
    resume: Path | None

    def as_serializable_dict(self) -> dict[str, str | None]:
        return {
            key: str(value) if value is not None else None
            for key, value in asdict(self).items()
        }


EXPECTED_SPLIT_COUNTS = {
    "train": 2_753,
    "validation": 473,
    "test": 443,
}
EXPECTED_CONTEXTS = 20
EXPECTED_LATENT_DIM = 14
EXPECTED_INPUT_BP = 1_048_576
EXPECTED_TARGET_BP = 1_000_000
EXPECTED_TARGET_BINS = 200
EXPECTED_TARGET_SHAPE = (
    EXPECTED_CONTEXTS,
    3_669,
    EXPECTED_TARGET_BINS,
    EXPECTED_TARGET_BINS,
)
EXPECTED_TARGET_CHUNKS = (1, 64, 200, 200)
EXPECTED_TARGET_DTYPE = np.dtype("float16")
EXPECTED_DIMENSION_ORDER = ("context", "tile", "bin1", "bin2")
EXPECTED_MASKED_DIAGONALS = 4
EXPECTED_BIN_SIZE_BP = 5_000
AUTOSOMES = frozenset(f"chr{index}" for index in range(1, 20))

EXPECTED_PAIR_SHAPE = (512, 512, 128)
EXPECTED_PAIR_DTYPE = np.dtype("float16")


@dataclass(frozen=True)
class Phase1Metadata:
    tiles: pd.DataFrame
    contexts: pd.DataFrame
    context_ids: tuple[str, ...]
    centroids: torch.Tensor
    split_rows: dict[str, np.ndarray]
    target_shape: tuple[int, int, int, int]
    target_chunks: tuple[int, int, int, int]
    target_dtype: str

    def summary(self) -> dict[str, Any]:
        return {
            "tile_count": len(self.tiles),
            "split_counts": {
                split: len(rows)
                for split, rows in self.split_rows.items()
            },
            "context_count": len(self.context_ids),
            "context_ids": list(self.context_ids),
            "centroid_shape": list(self.centroids.shape),
            "target_shape": list(self.target_shape),
            "target_chunks": list(self.target_chunks),
            "target_dtype": self.target_dtype,
        }


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}"
        )


def load_tiles(path: Path) -> pd.DataFrame:
    tiles = pd.read_parquet(path)

    require_columns(
        tiles,
        {
            "tile_id",
            "tile_index",
            "chrom",
            "input_start",
            "input_end",
            "target_start",
            "target_end",
            "split",
            "embedding_path",
        },
        source=str(path),
    )

    # This row position, rather than the gapped tile_index column, indexes
    # axis 1 of the target Zarr store.
    tiles = tiles.copy()
    tiles.insert(
        0,
        "target_tile_row",
        np.arange(len(tiles), dtype=np.int64),
    )

    if tiles["tile_id"].isna().any():
        raise ValueError("Tile IDs must not contain missing values")
    if tiles["tile_id"].duplicated().any():
        duplicates = tiles.loc[
            tiles["tile_id"].duplicated(keep=False),
            "tile_id",
        ].unique()
        raise ValueError(
            f"Tile IDs must be unique; duplicates include {duplicates[:5]}"
        )

    if tiles["embedding_path"].isna().any():
        raise ValueError("Every retained tile requires an embedding_path")

    input_lengths = (
        tiles["input_end"].to_numpy(dtype=np.int64)
        - tiles["input_start"].to_numpy(dtype=np.int64)
    )
    if not np.all(input_lengths == EXPECTED_INPUT_BP):
        raise ValueError(
            f"Every input interval must span {EXPECTED_INPUT_BP:,} bp"
        )

    target_lengths = (
        tiles["target_end"].to_numpy(dtype=np.int64)
        - tiles["target_start"].to_numpy(dtype=np.int64)
    )
    if not np.all(target_lengths == EXPECTED_TARGET_BP):
        raise ValueError(
            f"Every target interval must span {EXPECTED_TARGET_BP:,} bp"
        )

    target_is_contained = (
        (
            tiles["input_start"].to_numpy(dtype=np.int64)
            <= tiles["target_start"].to_numpy(dtype=np.int64)
        )
        & (
            tiles["target_end"].to_numpy(dtype=np.int64)
            <= tiles["input_end"].to_numpy(dtype=np.int64)
        )
    )
    if not np.all(target_is_contained):
        bad_rows = tiles.loc[
            ~target_is_contained,
            ["target_tile_row", "tile_id"],
        ]
        raise ValueError(
            "Every target must be contained in its input interval; "
            f"bad rows include {bad_rows.head().to_dict('records')}"
        )

    if not np.all(
        tiles["target_start"].to_numpy(dtype=np.int64)
        % EXPECTED_BIN_SIZE_BP
        == 0
    ):
        raise ValueError(
            f"Target starts must be aligned to {EXPECTED_BIN_SIZE_BP:,} bp"
        )

    unknown_chromosomes = sorted(set(tiles["chrom"]) - AUTOSOMES)
    if unknown_chromosomes:
        raise ValueError(
            f"Unexpected chromosomes in tile table: {unknown_chromosomes}"
        )

    split_counts = tiles["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected tile split counts: "
            f"expected {EXPECTED_SPLIT_COUNTS}, observed {split_counts}"
        )

    return tiles


def load_contexts_and_centroids(
    context_path: Path,
    centroid_path: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], torch.Tensor]:
    contexts = pd.read_parquet(context_path)
    centroids = pd.read_parquet(centroid_path)

    require_columns(
        contexts,
        {
            "cell_type",
            "n_cells",
            "valid_pairs",
            "context_index",
        },
        source=str(context_path),
    )
    require_columns(
        centroids,
        {"cell_type", "n_cells", "embedding"},
        source=str(centroid_path),
    )

    if contexts["cell_type"].duplicated().any():
        raise ValueError("Target cell types must be unique")
    if contexts["context_index"].duplicated().any():
        raise ValueError("Target context indices must be unique")
    if centroids["cell_type"].duplicated().any():
        raise ValueError("Centroid cell types must be unique")

    contexts = contexts.sort_values(
        "context_index",
        kind="stable",
    ).reset_index(drop=True)

    expected_indices = np.arange(EXPECTED_CONTEXTS, dtype=np.int64)
    observed_indices = contexts["context_index"].to_numpy(dtype=np.int64)

    if len(contexts) != EXPECTED_CONTEXTS:
        raise ValueError(
            f"Expected {EXPECTED_CONTEXTS} target contexts; "
            f"found {len(contexts)}"
        )
    if not np.array_equal(observed_indices, expected_indices):
        raise ValueError(
            "Target context indices must be exactly 0 through "
            f"{EXPECTED_CONTEXTS - 1}; observed {observed_indices.tolist()}"
        )

    selected = contexts.merge(
        centroids,
        on="cell_type",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("_target", "_centroid"),
    )
    selected = selected.sort_values(
        "context_index",
        kind="stable",
    ).reset_index(drop=True)

    if selected["embedding"].isna().any():
        missing = selected.loc[
            selected["embedding"].isna(),
            "cell_type",
        ].tolist()
        raise ValueError(
            f"Target contexts are missing centroids: {missing}"
        )

    mismatched_counts = (
        selected["n_cells_target"].to_numpy(dtype=np.int64)
        != selected["n_cells_centroid"].to_numpy(dtype=np.int64)
    )
    if np.any(mismatched_counts):
        mismatches = selected.loc[
            mismatched_counts,
            ["cell_type", "n_cells_target", "n_cells_centroid"],
        ]
        raise ValueError(
            "Target and centroid cell counts disagree: "
            f"{mismatches.to_dict('records')}"
        )

    centroid_array = np.stack(
        [
            np.asarray(embedding, dtype=np.float32)
            for embedding in selected["embedding"]
        ],
        axis=0,
    )

    if centroid_array.shape != (
        EXPECTED_CONTEXTS,
        EXPECTED_LATENT_DIM,
    ):
        raise ValueError(
            "Expected centroid tensor shape "
            f"({EXPECTED_CONTEXTS}, {EXPECTED_LATENT_DIM}); "
            f"found {centroid_array.shape}"
        )
    if not np.isfinite(centroid_array).all():
        raise ValueError("Centroid embeddings must contain only finite values")

    context_ids = tuple(selected["cell_type"].astype(str))
    centroid_tensor = torch.from_numpy(centroid_array.copy())

    return selected, context_ids, centroid_tensor


def validate_target_store(
    path: Path,
    *,
    tile_count: int,
    context_ids: tuple[str, ...],
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    str,
]:
    root = zarr.open_group(str(path), mode="r")

    if "targets" not in root:
        raise ValueError(f"Target Zarr has no 'targets' array: {path}")

    targets = root["targets"]
    target_shape = tuple(int(value) for value in targets.shape)
    target_chunks = tuple(int(value) for value in targets.chunks)
    target_dtype = np.dtype(targets.dtype)

    expected_shape = (
        EXPECTED_CONTEXTS,
        tile_count,
        EXPECTED_TARGET_BINS,
        EXPECTED_TARGET_BINS,
    )
    if target_shape != expected_shape:
        raise ValueError(
            f"Expected target shape {expected_shape}; found {target_shape}"
        )
    if target_shape != EXPECTED_TARGET_SHAPE:
        raise ValueError(
            "Prepared target shape changed from the reviewed contract: "
            f"expected {EXPECTED_TARGET_SHAPE}, found {target_shape}"
        )
    if target_chunks != EXPECTED_TARGET_CHUNKS:
        raise ValueError(
            f"Expected target chunks {EXPECTED_TARGET_CHUNKS}; "
            f"found {target_chunks}"
        )
    if target_dtype != EXPECTED_TARGET_DTYPE:
        raise ValueError(
            f"Expected target dtype {EXPECTED_TARGET_DTYPE}; "
            f"found {target_dtype}"
        )

    dimension_order = tuple(root.attrs.get("dimension_order", ()))
    if dimension_order != EXPECTED_DIMENSION_ORDER:
        raise ValueError(
            f"Expected dimension order {EXPECTED_DIMENSION_ORDER}; "
            f"found {dimension_order}"
        )

    stored_context_ids = tuple(root.attrs.get("context_ids", ()))
    if stored_context_ids != context_ids:
        raise ValueError(
            "Target Zarr context IDs do not match target_contexts order: "
            f"expected {context_ids}, found {stored_context_ids}"
        )

    bin_size = root.attrs.get("bin_size_bp")
    if bin_size != EXPECTED_BIN_SIZE_BP:
        raise ValueError(
            f"Expected target bin size {EXPECTED_BIN_SIZE_BP}; "
            f"found {bin_size}"
        )

    masked_diagonals = root.attrs.get("masked_diagonals")
    if masked_diagonals != EXPECTED_MASKED_DIAGONALS:
        raise ValueError(
            f"Expected {EXPECTED_MASKED_DIAGONALS} masked diagonals; "
            f"found {masked_diagonals}"
        )

    return target_shape, target_chunks, target_dtype.name


def load_phase1_metadata(paths: Phase1Paths) -> Phase1Metadata:
    tiles = load_tiles(paths.tiles)
    contexts, context_ids, centroids = load_contexts_and_centroids(
        paths.target_contexts,
        paths.centroids,
    )
    target_shape, target_chunks, target_dtype = validate_target_store(
        paths.targets,
        tile_count=len(tiles),
        context_ids=context_ids,
    )

    split_rows = {
        split: np.flatnonzero(
            tiles["split"].to_numpy() == split
        ).astype(np.int64)
        for split in ("train", "validation", "test")
    }

    return Phase1Metadata(
        tiles=tiles,
        contexts=contexts,
        context_ids=context_ids,
        centroids=centroids,
        split_rows=split_rows,
        target_shape=target_shape,
        target_chunks=target_chunks,
        target_dtype=target_dtype,
    )


def print_metadata_summary(metadata: Phase1Metadata) -> None:
    print(
        json.dumps(
            {"metadata": metadata.summary()},
            indent=2,
            sort_keys=True,
        )
    )

UINT64_MAX = 2**64 - 1
TILE_ORDER_NAMESPACE = 0x54494C45  # "TILE"
CONTEXT_NAMESPACE = 0x43545854  # "CTXT"


def require_seed_component(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; found {value!r}")

    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; found {value}")
    if value > UINT64_MAX:
        raise ValueError(
            f"{name} must not exceed {UINT64_MAX}; found {value}"
        )
    return value


def deterministic_rng(
    namespace: int,
    **components: int,
) -> np.random.Generator:
    """Build a stable RNG from named nonnegative integer components."""
    entropy = [namespace]

    # Sorting names prevents call-site keyword order from affecting the seed.
    for name in sorted(components):
        value = require_seed_component(name, components[name])
        entropy.extend(
            [
                value & 0xFFFFFFFF,
                (value >> 32) & 0xFFFFFFFF,
            ]
        )

    return np.random.default_rng(np.random.SeedSequence(entropy))


def make_epoch_tile_order(
    tile_rows: np.ndarray | list[int] | tuple[int, ...],
    *,
    seed: int,
    epoch: int,
    limit: int | None = None,
) -> np.ndarray:
    """Return a deterministic permutation of target-tile row positions."""
    seed = require_seed_component("seed", seed)
    epoch = require_seed_component("epoch", epoch)

    rows = np.asarray(tile_rows)

    if rows.ndim != 1:
        raise ValueError(
            f"tile_rows must be one-dimensional; found shape {rows.shape}"
        )
    if rows.size == 0:
        raise ValueError("tile_rows must not be empty")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError(
            f"tile_rows must contain integers; found dtype {rows.dtype}"
        )

    rows = rows.astype(np.int64, copy=True)

    if np.any(rows < 0):
        raise ValueError("tile_rows must contain only nonnegative values")
    if np.unique(rows).size != rows.size:
        raise ValueError("tile_rows must not contain duplicates")

    if limit is not None:
        limit = require_seed_component("limit", limit)
        if limit == 0:
            raise ValueError("limit must be greater than zero")
        if limit > rows.size:
            raise ValueError(
                f"limit ({limit}) exceeds available tiles ({rows.size})"
            )

    rng = deterministic_rng(
        TILE_ORDER_NAMESPACE,
        seed=seed,
        epoch=epoch,
    )
    ordered_rows = rng.permutation(rows).astype(np.int64, copy=False)

    if limit is not None:
        ordered_rows = ordered_rows[:limit]

    return ordered_rows


def sample_context_indices(
    num_contexts: int,
    count: int,
    seed: int,
    epoch: int,
    tile_row: int,
) -> np.ndarray:
    """Select a deterministic, uniformly sampled context subset.

    Returned indices are sorted into authoritative context-axis order after
    sampling. Sorting does not change the uniform distribution over subsets.
    """
    num_contexts = require_seed_component("num_contexts", num_contexts)
    count = require_seed_component("count", count)
    seed = require_seed_component("seed", seed)
    epoch = require_seed_component("epoch", epoch)
    tile_row = require_seed_component("tile_row", tile_row)

    if num_contexts == 0:
        raise ValueError("num_contexts must be greater than zero")
    if count == 0:
        raise ValueError("count must be greater than zero")
    if count > num_contexts:
        raise ValueError(
            f"count ({count}) cannot exceed num_contexts ({num_contexts})"
        )

    rng = deterministic_rng(
        CONTEXT_NAMESPACE,
        seed=seed,
        epoch=epoch,
        tile_row=tile_row,
    )
    selected = rng.choice(
        num_contexts,
        size=count,
        replace=False,
    )

    return np.sort(selected).astype(np.int64, copy=False)

class Phase1TileDataset(Dataset):
    """Lazily load frozen pair embeddings and matching Phase 1 targets."""

    def __init__(
        self,
        *,
        metadata: Phase1Metadata,
        targets_path: Path,
        embeddings_dir: Path,
        split: str,
        contexts_per_tile: int,
        seed: int,
        tile_rows: np.ndarray | list[int] | None = None,
        epoch: int = 0,
        fixed_context_indices: np.ndarray | list[int] | None=None,
    ) -> None:
        if split not in metadata.split_rows:
            raise ValueError(
                f"Unknown split {split!r}; "
                f"expected one of {tuple(metadata.split_rows)}"
            )

        self.metadata = metadata
        self.targets_path = Path(targets_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_cache_dir: Path | None = None

        embeddings_cache = os.environ.get(EMBEDDINGS_CACHE_ENV)
        if embeddings_cache:
            cache_path = Path(embeddings_cache).expanduser().resolve()
            if not cache_path.is_dir():
                raise FileNotFoundError(
                    f"{EMBEDDINGS_CACHE_ENV} is not a directory: "
                    f"{cache_path}"
                )
            self.embedding_cache_dir = cache_path

        targets_cache = os.environ.get(TARGETS_CACHE_ENV)
        if targets_cache:
            cache_path = Path(targets_cache).expanduser().resolve()
            if not cache_path.is_dir():
                raise FileNotFoundError(
                    f"{TARGETS_CACHE_ENV} is not a directory: "
                    f"{cache_path}"
                )
            self.targets_path = cache_path
        self.split = split
        self.training = split == "train"
        self.seed = require_seed_component("seed", seed)
        self.epoch = require_seed_component("epoch", epoch)

        contexts_per_tile = require_seed_component(
            "contexts_per_tile",
            contexts_per_tile,
        )
        if contexts_per_tile == 0:
            raise ValueError("contexts_per_tile must be greater than zero")
        if contexts_per_tile > len(metadata.context_ids):
            raise ValueError(
                f"contexts_per_tile ({contexts_per_tile}) exceeds "
                f"available contexts ({len(metadata.context_ids)})"
            )
        self.contexts_per_tile = contexts_per_tile
        self.fixed_context_indices: np.ndarray | None = None

        if fixed_context_indices is not None:
            if not self.training:
                raise ValueError(
                    "fixed_context_indices is only valid for training data"
                )

            fixed_indices = np.asarray(fixed_context_indices)
            if fixed_indices.ndim != 1:
                raise ValueError(
                    "fixed_context_indices must be one-dimensional"
                )
            if not np.issubdtype(
                fixed_indices.dtype,
                np.integer,
            ):
                raise TypeError(
                    "fixed_context_indices must contain integers"
                )

            fixed_indices = fixed_indices.astype(
                np.int64,
                copy=True,
            )

            if len(fixed_indices) != contexts_per_tile:
                raise ValueError(
                    "fixed_context_indices length must equal "
                    f"contexts_per_tile ({contexts_per_tile}); "
                    f"found {len(fixed_indices)}"
                )
            if np.unique(fixed_indices).size != len(fixed_indices):
                raise ValueError(
                    "fixed_context_indices must not contain duplicates"
                )
            if (
                np.any(fixed_indices < 0)
                or np.any(
                    fixed_indices >= len(metadata.context_ids)
                )
            ):
                raise IndexError(
                    "fixed_context_indices contains an out-of-range "
                    "context index"
                )

            self.fixed_context_indices = np.sort(fixed_indices)

        if tile_rows is None:
            rows = metadata.split_rows[split].copy()
        else:
            rows = np.asarray(tile_rows)

        if rows.ndim != 1:
            raise ValueError(
                f"tile_rows must be one-dimensional; found {rows.shape}"
            )
        if rows.size == 0:
            raise ValueError("tile_rows must not be empty")
        if not np.issubdtype(rows.dtype, np.integer):
            raise TypeError(
                f"tile_rows must contain integers; found {rows.dtype}"
            )

        rows = rows.astype(np.int64, copy=True)

        if np.any(rows < 0) or np.any(rows >= len(metadata.tiles)):
            raise IndexError(
                "tile_rows contains positions outside the retained tile table"
            )
        if np.unique(rows).size != rows.size:
            raise ValueError("tile_rows must not contain duplicates")

        observed_splits = set(
            metadata.tiles.iloc[rows]["split"].astype(str)
        )
        if observed_splits != {split}:
            raise ValueError(
                f"Dataset split is {split!r}, but selected rows contain "
                f"{sorted(observed_splits)}"
            )

        self.tile_rows = rows

        # Zarr objects are opened lazily in each process. This keeps dataset
        # construction cheap and avoids pickling open stores into workers.
        self._targets_array: Any | None = None

    def __len__(self) -> int:
        return int(self.tile_rows.size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = require_seed_component("epoch", epoch)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_targets_array"] = None
        return state

    def _get_targets_array(self) -> Any:
        if self._targets_array is None:
            root = zarr.open_group(str(self.targets_path), mode="r")
            if "targets" not in root:
                raise ValueError(
                    f"Target Zarr has no 'targets' array: "
                    f"{self.targets_path}"
                )
            self._targets_array = root["targets"]

        return self._targets_array

    def _select_context_indices(
        self,
        target_tile_row: int,
    ) -> np.ndarray:
        if self.fixed_context_indices is not None:
            return self.fixed_context_indices.copy()
        if not self.training:
            return np.arange(
                len(self.metadata.context_ids),
                dtype=np.int64,
            )

        return sample_context_indices(
            num_contexts=len(self.metadata.context_ids),
            count=self.contexts_per_tile,
            seed=self.seed,
            epoch=self.epoch,
            tile_row=target_tile_row,
        )

    def _resolve_embedding_path(self, stored_path: str | Path) -> Path:
        stored = Path(stored_path).expanduser()

        if self.embedding_cache_dir is not None:
            path = self.embedding_cache_dir / stored.name
        elif stored.is_absolute() and stored.is_file():
            path = stored
        elif stored.is_absolute():
            path = self.embeddings_dir / stored.name
        else:
            path = self.embeddings_dir / stored

        return path.resolve()

    def _load_embedding(
        self,
        embedding_path: Path,
        *,
        tile_id: str,
    ) -> torch.Tensor:
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"Missing embedding for tile {tile_id}: {embedding_path}"
            )

        # Direct loading creates one writable, process-owned array.
        # The prior read-only mmap followed by np.array(copy=True) added
        # mapping and copy overhead without reducing resident memory.
        embedding = np.load(
            embedding_path,
            allow_pickle=False,
        )

        if embedding.shape != EXPECTED_PAIR_SHAPE:
            raise ValueError(
                f"Embedding for tile {tile_id} has shape {embedding.shape}; "
                f"expected {EXPECTED_PAIR_SHAPE}"
            )
        if np.dtype(embedding.dtype) != EXPECTED_PAIR_DTYPE:
            raise ValueError(
                f"Embedding for tile {tile_id} has dtype {embedding.dtype}; "
                f"expected {EXPECTED_PAIR_DTYPE}"
            )
        if not embedding.flags.c_contiguous:
            embedding = np.ascontiguousarray(embedding)

        if not np.isfinite(embedding).all():
            raise ValueError(
                f"Embedding for tile {tile_id} contains non-finite values"
            )

        return torch.from_numpy(embedding)

    def _load_targets(
        self,
        context_indices: np.ndarray,
        target_tile_row: int,
        *,
        tile_id: str,
    ) -> torch.Tensor:
        targets = self._get_targets_array()

        # Orthogonal indexing selects C contexts at one target-axis tile row.
        selected = np.asarray(
            targets.oindex[
                context_indices,
                target_tile_row,
                :,
                :,
            ]
        )
        selected = np.array(
            selected,
            dtype=np.float16,
            order="C",
            copy=True,
        )

        expected_shape = (
            len(context_indices),
            EXPECTED_TARGET_BINS,
            EXPECTED_TARGET_BINS,
        )
        if selected.shape != expected_shape:
            raise ValueError(
                f"Targets for tile {tile_id} have shape {selected.shape}; "
                f"expected {expected_shape}"
            )

        # Do not reject NaNs: they are the persisted target mask.
        return torch.from_numpy(selected)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError(
                f"Dataset index must be an integer; found {index!r}"
            )

        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Dataset index {index} is outside [0, {len(self)})"
            )

        target_tile_row = int(self.tile_rows[index])
        tile = self.metadata.tiles.iloc[target_tile_row]

        stored_target_row = int(tile["target_tile_row"])
        if stored_target_row != target_tile_row:
            raise RuntimeError(
                "Persisted target_tile_row no longer matches dataframe "
                f"position: position={target_tile_row}, "
                f"stored={stored_target_row}"
            )

        tile_id = str(tile["tile_id"])
        context_indices = self._select_context_indices(target_tile_row)
        context_index_tensor = torch.from_numpy(
            context_indices.copy()
        ).to(dtype=torch.int64)

        embedding_path = self._resolve_embedding_path(
            tile["embedding_path"]
        )
        pair_embedding = self._load_embedding(
            embedding_path,
            tile_id=tile_id,
        )
        targets = self._load_targets(
            context_indices,
            target_tile_row,
            tile_id=tile_id,
        )

        # Import after main() has prepended src/scoo-hic to sys.path.
        from model_v2 import build_area_overlap_matrix

        resample_weights = build_area_overlap_matrix(
            input_start=int(tile["input_start"]),
            target_start=int(tile["target_start"]),
        )

        context_embedding = self.metadata.centroids.index_select(
            0,
            context_index_tensor,
        ).clone()
        context_ids = tuple(
            self.metadata.context_ids[int(context_index)]
            for context_index in context_indices
        )

        return {
            "pair_embedding": pair_embedding,
            "context_embedding": context_embedding,
            "target": targets,
            "resample_weights": resample_weights,
            "context_indices": context_index_tensor,
            "context_ids": context_ids,
            "target_tile_row": target_tile_row,
            "tile_id": tile_id,
            "chrom": str(tile["chrom"]),
            "input_start": int(tile["input_start"]),
            "target_start": int(tile["target_start"]),
            "embedding_path": str(embedding_path),
        }


def print_tile_example_summary(example: Mapping[str, Any]) -> None:
    pair_embedding = example["pair_embedding"]
    context_embedding = example["context_embedding"]
    targets = example["target"]
    resample_weights = example["resample_weights"]

    summary = {
        "tile_example": {
            "target_tile_row": example["target_tile_row"],
            "tile_id": example["tile_id"],
            "chrom": example["chrom"],
            "context_ids": list(example["context_ids"]),
            "context_indices": example["context_indices"].tolist(),
            "pair_embedding_shape": list(pair_embedding.shape),
            "pair_embedding_dtype": str(pair_embedding.dtype),
            "pair_embedding_finite": bool(
                torch.isfinite(pair_embedding).all().item()
            ),
            "context_embedding_shape": list(context_embedding.shape),
            "context_embedding_dtype": str(context_embedding.dtype),
            "target_shape": list(targets.shape),
            "target_dtype": str(targets.dtype),
            "target_nan_count": int(torch.isnan(targets).sum().item()),
            "resample_weights_shape": list(resample_weights.shape),
            "resample_weights_dtype": str(resample_weights.dtype),
            "resample_max_row_sum_error": float(
                (resample_weights.sum(dim=1) - 1.0)
                .abs()
                .max()
                .item()
            ),
        }
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

def resolve_device(requested: str) -> torch.device:
    """Resolve and validate the requested PyTorch device."""
    if not isinstance(requested, str):
        raise TypeError(
            f"device must be a string; found {type(requested).__name__}"
        )

    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Invalid PyTorch device specification: {requested!r}"
        ) from error

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Phase 1 currently supports cpu and cuda devices; "
            f"found {device.type!r}"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {requested!r} was requested, but CUDA "
                "is not available to PyTorch"
            )

        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device.index < 0 or device.index >= device_count:
                raise RuntimeError(
                    f"CUDA device index {device.index} is outside the "
                    f"available range [0, {device_count})"
                )

    return device


def summarize_gradient(
    name: str,
    parameter: torch.nn.Parameter,
) -> dict[str, Any]:
    """Summarize and validate one model parameter gradient."""
    gradient = parameter.grad

    if gradient is None:
        return {
            "name": name,
            "present": False,
            "finite": False,
            "nonzero": False,
            "norm": None,
        }

    finite = bool(torch.isfinite(gradient).all().item())
    norm = float(gradient.float().norm().item())
    nonzero = finite and norm > 0.0

    return {
        "name": name,
        "present": True,
        "finite": finite,
        "nonzero": nonzero,
        "norm": norm,
    }

def require_finite_training_float(
    name: str,
    value: float,
    *,
    allow_zero: bool,
) -> float:
    """Validate a numeric optimizer or scheduler setting."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric; found {value!r}")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; found {value}")

    if allow_zero:
        if value < 0.0:
            raise ValueError(
                f"{name} must be nonnegative; found {value}"
            )
    elif value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero; found {value}"
        )

    return value


def build_phase1_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW with decay only on matrix-shaped parameters."""
    learning_rate = require_finite_training_float(
        "learning_rate",
        learning_rate,
        allow_zero=False,
    )
    weight_decay = require_finite_training_float(
        "weight_decay",
        weight_decay,
        allow_zero=True,
    )

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")

    # Biases and all other 1D parameters receive no weight decay.
    decay_parameters = [
        parameter
        for _, parameter in trainable_parameters
        if parameter.ndim >= 2
    ]
    no_decay_parameters = [
        parameter
        for _, parameter in trainable_parameters
        if parameter.ndim < 2
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
                "group_name": "matrix_weights",
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "group_name": "biases_and_1d",
            },
        ],
        lr=learning_rate,
    )

    expected_ids = {
        id(parameter)
        for _, parameter in trainable_parameters
    }
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = {
        id(parameter)
        for parameter in optimizer_parameters
    }

    if len(optimizer_parameters) != len(optimizer_ids):
        raise RuntimeError(
            "A trainable parameter appears in multiple optimizer groups"
        )
    if optimizer_ids != expected_ids:
        raise RuntimeError(
            "Optimizer parameters do not exactly match the model's "
            "trainable parameters"
        )

    return optimizer


def build_linear_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up linearly, then decay linearly to zero."""
    warmup_steps = require_seed_component(
        "warmup_steps",
        warmup_steps,
    )
    total_steps = require_seed_component(
        "total_steps",
        total_steps,
    )
    if total_steps == 0:
        raise ValueError("total_steps must be greater than zero")

    def learning_rate_multiplier(scheduler_epoch: int) -> float:
        update_number = scheduler_epoch + 1

        # The scheduler reaches zero after all planned updates.
        if update_number > total_steps:
            return 0.0

        # With no warm-up, the first update uses the full learning rate.
        peak_update = max(warmup_steps, 1)

        if update_number <= peak_update:
            return float(update_number) / float(peak_update)

        # Include the transition to zero after the final optimizer update.
        remaining_updates = total_steps - update_number + 1
        decay_intervals = total_steps - peak_update + 1

        return max(
            0.0,
            float(remaining_updates) / float(decay_intervals),
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )


def safe_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    pair_embedding: torch.Tensor,
    context_embedding: torch.Tensor,
    target: torch.Tensor,
    resample_weights: torch.Tensor,
    clip_norm: float,
    global_step: int,
    tile_id: str,
    context_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Perform one guarded optimizer update for one sequence tile."""
    from model_v2 import masked_upper_triangle_mse

    clip_norm = require_finite_training_float(
        "clip_norm",
        clip_norm,
        allow_zero=False,
    )
    global_step = require_seed_component(
        "global_step",
        global_step,
    )

    diagnostics = (
        f"global_step={global_step}, tile_id={tile_id!r}, "
        f"context_ids={list(context_ids)!r}"
    )

    if pair_embedding.requires_grad:
        raise RuntimeError(
            "Frozen pair embeddings must not require gradients; "
            f"{diagnostics}"
        )

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    model_parameter_ids = {
        id(parameter)
        for _, parameter in trainable_parameters
    }
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_parameter_ids = {
        id(parameter)
        for parameter in optimizer_parameters
    }

    if (
        len(optimizer_parameters) != len(optimizer_parameter_ids)
        or optimizer_parameter_ids != model_parameter_ids
    ):
        raise RuntimeError(
            "Optimizer does not contain each trainable model parameter "
            f"exactly once; {diagnostics}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(
        pair_embedding,
        context_embedding,
        resample_weights,
    )

    try:
        loss, per_map_mse, valid_pixel_counts = (
            masked_upper_triangle_mse(
                prediction,
                target,
                diagonal=EXPECTED_MASKED_DIAGONALS,
            )
        )
    except ValueError as error:
        if str(error) == "No maps contain valid target pixels after masking":
            optimizer.zero_grad(set_to_none=True)
            learning_rates = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]
            return {
                "skipped": True,
                "skip_reason": "no_valid_target_pixels",
                "global_step": global_step,
                "valid_map_count": 0,
                "valid_pixel_count": 0,
                "learning_rates_used": learning_rates,
                "next_learning_rates": learning_rates,
            }

        raise RuntimeError(
            f"Masked loss construction failed; {diagnostics}"
        ) from error
    except TypeError as error:
        raise RuntimeError(
            f"Masked loss construction failed; {diagnostics}"
        ) from error

    if not bool(torch.isfinite(loss).item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Training loss is non-finite: {float(loss.item())}; "
            f"{diagnostics}"
        )

    loss.backward()

    missing_gradients = [
        name
        for name, parameter in trainable_parameters
        if parameter.grad is None
    ]
    if missing_gradients:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            f"Trainable parameters have no gradient: "
            f"{missing_gradients}; {diagnostics}"
        )

    try:
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for _, parameter in trainable_parameters
            ],
            max_norm=clip_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        nonfinite_gradients = [
            name
            for name, parameter in trainable_parameters
            if not bool(
                torch.isfinite(parameter.grad).all().item()
            )
        ]
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "Gradient norm is non-finite; "
            f"nonfinite_gradients={nonfinite_gradients}; "
            f"{diagnostics}"
        ) from error

    gradient_norm = float(gradient_norm_tensor.item())
    if gradient_norm == 0.0:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            f"Global gradient norm is zero; {diagnostics}"
        )

    learning_rates_used = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    optimizer.step()
    scheduler.step()

    next_learning_rates = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    finite_map_losses = per_map_mse[
        torch.isfinite(per_map_mse)
    ]
    if finite_map_losses.numel() == 0:
        raise RuntimeError(
            f"No finite per-map losses were produced; {diagnostics}"
        )

    return {
        "skipped": False,
        "global_step": global_step + 1,
        "loss": float(loss.detach().item()),
        "map_mse_sum": float(
            finite_map_losses.detach().sum().item()
        ),
        "valid_map_count": int(finite_map_losses.numel()),
        "valid_pixel_count": int(
            valid_pixel_counts.detach().sum().item()
        ),
        "gradient_norm_before_clip": gradient_norm,
        "gradient_was_clipped": gradient_norm > clip_norm,
        "clip_norm": clip_norm,
        "learning_rates_used": learning_rates_used,
        "next_learning_rates": next_learning_rates,
    }

def run_training_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metadata: Phase1Metadata,
    paths: Phase1Paths,
    device: torch.device,
    seed: int,
    epoch: int,
    global_step: int,
    contexts_per_tile: int,
    clip_norm: float,
    max_train_tiles: int | None,
    num_workers: int,
    prefetch_factor: int = 4,
    fixed_tile_rows: np.ndarray | list[int] | None = None,
    fixed_context_indices: np.ndarray | list[int] | None = None,
    show_progress: bool = True,
    progress_position: int = 1,
) -> dict[str, Any]:
    """Run one deterministic epoch over training tiles."""
    seed = require_seed_component("seed", seed)
    epoch = require_seed_component("epoch", epoch)
    global_step = require_seed_component(
        "global_step",
        global_step,
    )
    contexts_per_tile = require_seed_component(
        "contexts_per_tile",
        contexts_per_tile,
    )
    num_workers = require_seed_component(
        "num_workers",
        num_workers,
    )
    prefetch_factor = require_seed_component(
        "prefetch_factor",
        prefetch_factor,
    )

    if contexts_per_tile == 0:
        raise ValueError(
            "contexts_per_tile must be greater than zero"
        )
    if contexts_per_tile > len(metadata.context_ids):
        raise ValueError(
            f"contexts_per_tile ({contexts_per_tile}) exceeds "
            f"the {len(metadata.context_ids)} available contexts"
        )
    if num_workers > 0 and prefetch_factor == 0:
        raise ValueError("prefetch_factor must be positive with workers")

    if max_train_tiles is not None:
        max_train_tiles = require_seed_component(
            "max_train_tiles",
            max_train_tiles,
        )
        if max_train_tiles == 0:
            raise ValueError(
                "max_train_tiles must be greater than zero"
            )

    if fixed_tile_rows is None:
        ordered_tile_rows = make_epoch_tile_order(
            metadata.split_rows["train"],
            seed=seed,
            epoch=epoch,
            limit=max_train_tiles,
        )
    else:
        ordered_tile_rows = np.asarray(
            fixed_tile_rows,
            dtype=np.int64,
        ).copy()

        if ordered_tile_rows.ndim != 1:
            raise ValueError(
                "fixed_tile_rows must be one-dimensional"
            )
        if ordered_tile_rows.size == 0:
            raise ValueError("fixed_tile_rows must not be empty")
        if max_train_tiles is not None and (
            len(ordered_tile_rows) != max_train_tiles
        ):
            raise ValueError(
                "fixed_tile_rows length must equal max_train_tiles"
            )

    dataset = Phase1TileDataset(
        metadata=metadata,
        targets_path=paths.targets,
        embeddings_dir=paths.embeddings_dir,
        split="train",
        contexts_per_tile=contexts_per_tile,
        seed=seed,
        tile_rows=ordered_tile_rows,
        epoch=epoch,
        fixed_context_indices=fixed_context_indices,
    )

    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor

    data_loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        **loader_options,
    )

    scheduler_epoch_start = int(scheduler.last_epoch)
    if scheduler_epoch_start != global_step:
        raise RuntimeError(
            "Scheduler position and global step disagree at epoch start: "
            f"scheduler.last_epoch={scheduler_epoch_start}, "
            f"global_step={global_step}"
        )

    map_mse_sum = 0.0
    valid_map_count = 0
    valid_pixel_count = 0
    gradient_norm_sum = 0.0
    maximum_gradient_norm = 0.0
    clipped_step_count = 0

    context_sample_counts = np.zeros(
        len(metadata.context_ids),
        dtype=np.int64,
    )

    first_step_loss: float | None = None
    last_step_loss: float | None = None
    first_learning_rates: list[float] | None = None
    last_learning_rates: list[float] | None = None
    next_learning_rates: list[float] | None = None
    first_context_ids: list[str] | None = None
    last_context_ids: list[str] | None = None
    observed_tile_rows: list[int] = []
    skipped_tile_rows: list[int] = []

    non_blocking = device.type == "cuda"

    progress_bar = tqdm(
        data_loader,
        total=len(dataset),
        desc=f"Train epoch {epoch + 1}",
        unit="tile",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for example in progress_bar:
        target_tile_row = int(example["target_tile_row"])
        observed_tile_rows.append(target_tile_row)

        context_indices = example["context_indices"]
        for context_index in context_indices.tolist():
            context_sample_counts[int(context_index)] += 1

        pair_embedding = (
            example["pair_embedding"]
            .unsqueeze(0)
            .to(
                device=device,
                non_blocking=non_blocking,
            )
        )
        context_embedding = example["context_embedding"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        target = (
            example["target"]
            .unsqueeze(0)
            .to(
                device=device,
                non_blocking=non_blocking,
            )
        )
        resample_weights = (
            example["resample_weights"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
        )

        step_summary = safe_train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            pair_embedding=pair_embedding,
            context_embedding=context_embedding,
            target=target,
            resample_weights=resample_weights,
            clip_norm=clip_norm,
            global_step=global_step,
            tile_id=str(example["tile_id"]),
            context_ids=list(example["context_ids"]),
        )

        if bool(step_summary["skipped"]):
            skipped_tile_rows.append(target_tile_row)
            progress_bar.set_postfix(
                skipped=len(skipped_tile_rows),
                reason=step_summary["skip_reason"],
            )
            del resample_weights
            del target
            del context_embedding
            del pair_embedding
            continue
        global_step = int(step_summary["global_step"])
        step_loss = float(step_summary["loss"])
        step_gradient_norm = float(
            step_summary["gradient_norm_before_clip"]
        )

        if first_step_loss is None:
            first_step_loss = step_loss
            first_learning_rates = list(
                step_summary["learning_rates_used"]
            )
            first_context_ids = list(example["context_ids"])

        last_step_loss = step_loss
        last_learning_rates = list(
            step_summary["learning_rates_used"]
        )
        next_learning_rates = list(
            step_summary["next_learning_rates"]
        )
        last_context_ids = list(example["context_ids"])

        map_mse_sum += float(step_summary["map_mse_sum"])
        valid_map_count += int(
            step_summary["valid_map_count"]
        )
        valid_pixel_count += int(
            step_summary["valid_pixel_count"]
        )
        gradient_norm_sum += step_gradient_norm
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            step_gradient_norm,
        )
        clipped_step_count += int(
            bool(step_summary["gradient_was_clipped"])
        )

        progress_bar.set_postfix(
            mse=f"{map_mse_sum / valid_map_count:.5f}",
            lr=f"{step_summary['learning_rates_used'][0]:.3e}",
            grad=f"{step_gradient_norm:.3f}",
        )

        del resample_weights
        del target
        del context_embedding
        del pair_embedding

    progress_bar.close()

    expected_tile_rows = ordered_tile_rows.tolist()
    if observed_tile_rows != expected_tile_rows:
        raise RuntimeError(
            "Observed training-tile order differs from the deterministic "
            f"epoch order: expected={expected_tile_rows[:10]}, "
            f"observed={observed_tile_rows[:10]}"
        )

    tile_count = len(observed_tile_rows)
    step_count = tile_count - len(skipped_tile_rows)
    if step_count == 0:
        raise RuntimeError("Training epoch performed no optimizer steps")
    if valid_map_count == 0:
        raise RuntimeError(
            "Training epoch produced no valid map losses"
        )

    expected_scheduler_epoch = (
        scheduler_epoch_start + step_count
    )
    if scheduler.last_epoch != expected_scheduler_epoch:
        raise RuntimeError(
            "Scheduler did not advance exactly once per optimizer step: "
            f"expected last_epoch={expected_scheduler_epoch}, "
            f"found {scheduler.last_epoch}"
        )
    if global_step != expected_scheduler_epoch:
        raise RuntimeError(
            "Global step does not match the scheduler position: "
            f"global_step={global_step}, "
            f"scheduler.last_epoch={scheduler.last_epoch}"
        )

    expected_context_samples = (
        tile_count * contexts_per_tile
    )
    if int(context_sample_counts.sum()) != expected_context_samples:
        raise RuntimeError(
            "Context sampling count does not match the epoch geometry: "
            f"expected={expected_context_samples}, "
            f"observed={int(context_sample_counts.sum())}"
        )

    mean_mse = map_mse_sum / valid_map_count
    if not np.isfinite(mean_mse):
        raise FloatingPointError(
            f"Epoch mean MSE is non-finite: {mean_mse}"
        )

    # Full tile-order output is useful for bounded reproducibility tests but
    # would make full-run metric records unnecessarily large.
    compact_tile_order = (
        observed_tile_rows
        if tile_count <= 32
        else None
    )

    return {
        "split": "train",
        "epoch": epoch,
        "global_step_start": scheduler_epoch_start,
        "global_step": global_step,
        "tile_count": tile_count,
        "skipped_tile_count": len(skipped_tile_rows),
        "skipped_tile_rows": skipped_tile_rows,
        "step_count": step_count,
        "tile_order": compact_tile_order,
        "first_tile_rows": observed_tile_rows[:10],
        "last_tile_row": observed_tile_rows[-1],
        "contexts_per_tile": contexts_per_tile,
        "context_sample_counts": {
            context_id: int(context_sample_counts[index])
            for index, context_id in enumerate(
                metadata.context_ids
            )
        },
        "first_context_ids": first_context_ids,
        "last_context_ids": last_context_ids,
        "first_step_loss": first_step_loss,
        "last_step_loss": last_step_loss,
        "mean_mse": mean_mse,
        "valid_map_count": valid_map_count,
        "valid_pixel_count": valid_pixel_count,
        "mean_gradient_norm_before_clip": (
            gradient_norm_sum / step_count
        ),
        "maximum_gradient_norm_before_clip": (
            maximum_gradient_norm
        ),
        "clipped_step_count": clipped_step_count,
        "first_learning_rates": first_learning_rates,
        "last_learning_rates": last_learning_rates,
        "next_learning_rates": next_learning_rates,
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }

def validate_launch_guardrails(
    *,
    args: argparse.Namespace,
    metadata: Phase1Metadata,
    device: torch.device,
) -> dict[str, Any]:
    """Reject unsafe or internally inconsistent training launches."""
    available_train_tiles = len(metadata.split_rows["train"])
    available_validation_tiles = len(
        metadata.split_rows["validation"]
    )

    if (
        args.max_train_tiles is not None
        and args.max_train_tiles > available_train_tiles
    ):
        raise ValueError(
            f"--max-train-tiles ({args.max_train_tiles}) exceeds "
            f"the {available_train_tiles} available training tiles"
        )

    if (
        args.max_validation_tiles is not None
        and args.max_validation_tiles > available_validation_tiles
    ):
        raise ValueError(
            f"--max-validation-tiles "
            f"({args.max_validation_tiles}) exceeds the "
            f"{available_validation_tiles} available validation tiles"
        )

    if args.overfit:
        if args.resume is not None:
            raise ValueError(
                "--overfit must start from a fresh model and cannot resume"
            )
        if args.max_train_tiles != 8:
            raise ValueError(
                "--overfit requires --max-train-tiles 8"
            )
        if args.contexts_per_tile != 3:
            raise ValueError(
                "--overfit requires --contexts-per-tile 3"
            )

        return {
            "mode": "overfit",
            "device": str(device),
            "train_tiles": 8,
            "contexts_per_tile": 3,
            "validation_uses_test_split": False,
        }

    if args.epochs > 40:
        raise ValueError(
            "Standard Phase 1 training is limited to at most 40 epochs"
        )

    bounded_cpu_run = (
        args.max_train_tiles is not None
        and args.max_validation_tiles is not None
    )
    if device.type == "cpu" and not bounded_cpu_run:
        raise RuntimeError(
            "An unbounded/full Phase 1 run requires CUDA. For a CPU "
            "smoke run, provide both --max-train-tiles and "
            "--max-validation-tiles."
        )

    return {
        "mode": "bounded" if bounded_cpu_run else "full",
        "device": str(device),
        "train_tiles": (
            args.max_train_tiles
            if args.max_train_tiles is not None
            else available_train_tiles
        ),
        "validation_tiles": (
            args.max_validation_tiles
            if args.max_validation_tiles is not None
            else available_validation_tiles
        ),
        "contexts_per_tile": args.contexts_per_tile,
        "validation_uses_test_split": False,
    }


def build_overfit_fixture(
    *,
    metadata: Phase1Metadata,
    paths: Phase1Paths,
    seed: int,
) -> tuple[Phase1TileDataset, np.ndarray, np.ndarray]:
    """Build the fixed eight-tile, three-context real-data fixture."""
    fixed_tile_rows = make_epoch_tile_order(
        metadata.split_rows["train"],
        seed=seed,
        epoch=0,
        limit=8,
    )

    fixed_context_indices = sample_context_indices(
        num_contexts=len(metadata.context_ids),
        count=3,
        seed=seed,
        epoch=0,
        tile_row=0,
    )

    dataset = Phase1TileDataset(
        metadata=metadata,
        targets_path=paths.targets,
        embeddings_dir=paths.embeddings_dir,
        split="train",
        contexts_per_tile=3,
        seed=seed,
        tile_rows=fixed_tile_rows,
        epoch=0,
        fixed_context_indices=fixed_context_indices,
    )

    return (
        dataset,
        fixed_tile_rows,
        fixed_context_indices,
    )


def evaluate_overfit_fixture(
    *,
    model: torch.nn.Module,
    dataset: Phase1TileDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fixed overfit maps without changing model state."""
    from model_v2 import masked_upper_triangle_mse

    if dataset.fixed_context_indices is None:
        raise ValueError(
            "Overfit evaluation requires fixed context indices"
        )

    map_mse_sum = 0.0
    valid_map_count = 0
    valid_pixel_count = 0
    first_tile_context_difference: float | None = None
    pair_embeddings_require_grad = False

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for dataset_index in range(len(dataset)):
                example = dataset[dataset_index]

                pair_embedding = (
                    example["pair_embedding"]
                    .unsqueeze(0)
                    .to(device=device)
                )
                context_embedding = (
                    example["context_embedding"]
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
                target = (
                    example["target"]
                    .unsqueeze(0)
                    .to(device=device)
                )
                resample_weights = (
                    example["resample_weights"]
                    .unsqueeze(0)
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                )

                pair_embeddings_require_grad = (
                    pair_embeddings_require_grad
                    or pair_embedding.requires_grad
                )

                prediction = model(
                    pair_embedding,
                    context_embedding,
                    resample_weights,
                )

                loss, per_map_mse, pixel_counts = (
                    masked_upper_triangle_mse(
                        prediction,
                        target,
                        diagonal=EXPECTED_MASKED_DIAGONALS,
                    )
                )

                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "Overfit fixture produced a non-finite loss for "
                        f"tile_id={example['tile_id']!r}"
                    )

                finite_mse = per_map_mse[
                    torch.isfinite(per_map_mse)
                ]
                map_mse_sum += float(finite_mse.sum().item())
                valid_map_count += int(finite_mse.numel())
                valid_pixel_count += int(pixel_counts.sum().item())

                if dataset_index == 0:
                    maximum_difference = 0.0

                    for first_context in range(
                        prediction.shape[1]
                    ):
                        for second_context in range(
                            first_context + 1,
                            prediction.shape[1],
                        ):
                            difference = float(
                                (
                                    prediction[:, first_context]
                                    - prediction[:, second_context]
                                )
                                .abs()
                                .max()
                                .item()
                            )
                            maximum_difference = max(
                                maximum_difference,
                                difference,
                            )

                    first_tile_context_difference = (
                        maximum_difference
                    )
    finally:
        model.train(was_training)

    expected_map_count = (
        len(dataset) * len(dataset.fixed_context_indices)
    )
    if valid_map_count != expected_map_count:
        raise RuntimeError(
            f"Expected {expected_map_count} valid overfit maps; "
            f"found {valid_map_count}"
        )

    return {
        "mean_mse": map_mse_sum / valid_map_count,
        "valid_map_count": valid_map_count,
        "valid_pixel_count": valid_pixel_count,
        "first_tile_context_max_difference": (
            first_tile_context_difference
        ),
        "pair_embeddings_require_grad": (
            pair_embeddings_require_grad
        ),
    }

def compute_overfit_linear_oracle(
    *,
    dataset: Phase1TileDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Compute the best independent linear readout for each context."""
    if dataset.fixed_context_indices is None:
        raise ValueError(
            "Linear oracle requires fixed context indices"
        )

    context_count = len(dataset.fixed_context_indices)
    feature_count = EXPECTED_PAIR_SHAPE[-1] + 1

    xtx = [
        torch.zeros(
            (feature_count, feature_count),
            dtype=torch.float64,
        )
        for _ in range(context_count)
    ]
    xty = [
        torch.zeros(feature_count, dtype=torch.float64)
        for _ in range(context_count)
    ]
    target_square = [0.0] * context_count
    map_counts = [0] * context_count

    upper_triangle = torch.triu(
        torch.ones(
            (
                EXPECTED_TARGET_BINS,
                EXPECTED_TARGET_BINS,
            ),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=EXPECTED_MASKED_DIAGONALS,
    )

    with torch.inference_mode():
        for dataset_index in range(len(dataset)):
            example = dataset[dataset_index]

            pair_embedding = example["pair_embedding"].to(
                device=device,
                dtype=torch.float32,
            )
            pair_embedding = 0.5 * (
                pair_embedding
                + pair_embedding.transpose(0, 1)
            )

            resample_weights = example["resample_weights"].to(
                device=device,
                dtype=torch.float32,
            )

            # Resample each frozen channel so the oracle solves the exact
            # linear problem represented by the Phase 1 decoder.
            left_resampled = torch.einsum(
                "pi,ijd->pjd",
                resample_weights,
                pair_embedding,
            )
            features = torch.einsum(
                "pjd,qj->pqd",
                left_resampled,
                resample_weights,
            )

            # Add the context-specific intercept feature.
            features = torch.cat(
                [
                    features,
                    torch.ones(
                        (*features.shape[:-1], 1),
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=-1,
            )

            targets = example["target"].to(
                device=device,
                dtype=torch.float32,
            )

            for context_position in range(context_count):
                mask = (
                    upper_triangle
                    & torch.isfinite(targets[context_position])
                )
                pixel_count = int(mask.sum().item())

                if pixel_count == 0:
                    raise RuntimeError(
                        "Oracle fixture map has no valid pixels"
                    )

                x = features[mask]
                y = targets[context_position][mask]

                # Each tile/context map receives equal weight regardless
                # of its valid-pixel count.
                map_scale = pixel_count ** -0.5
                x = x * map_scale
                y = y * map_scale

                xtx[context_position] += (
                    x.T @ x
                ).cpu().double()
                xty[context_position] += (
                    x.T @ y
                ).cpu().double()
                target_square[context_position] += float(
                    (y @ y).item()
                )
                map_counts[context_position] += 1

            del targets
            del features
            del left_resampled
            del resample_weights
            del pair_embedding

    context_mse: list[float] = []
    total_squared_error = 0.0

    for context_position in range(context_count):
        matrix = 0.5 * (
            xtx[context_position]
            + xtx[context_position].T
        )
        vector = xty[context_position]

        # Normalize the normal equations before taking the pseudoinverse.
        # The diagnostic showed stable results around this tolerance.
        feature_scale = torch.sqrt(
            torch.clamp(torch.diag(matrix), min=1e-30)
        )
        normalized_matrix = (
            matrix
            / feature_scale[:, None]
            / feature_scale[None, :]
        )
        normalized_vector = vector / feature_scale

        normalized_solution = torch.linalg.pinv(
            normalized_matrix,
            hermitian=True,
            rtol=1e-5,
        ) @ normalized_vector
        weights = normalized_solution / feature_scale

        squared_error = float(
            (
                weights @ matrix @ weights
                - 2.0 * weights @ vector
            ).item()
        ) + target_square[context_position]

        if not np.isfinite(squared_error):
            raise FloatingPointError(
                "Linear oracle produced non-finite squared error"
            )
        if squared_error < -1e-5:
            raise RuntimeError(
                "Linear oracle produced numerically invalid negative "
                f"squared error: {squared_error}"
            )

        squared_error = max(0.0, squared_error)
        total_squared_error += squared_error
        context_mse.append(
            squared_error / map_counts[context_position]
        )

    oracle_mse = total_squared_error / sum(map_counts)

    return {
        "mean_mse": oracle_mse,
        "context_mean_mse": context_mse,
        "map_count": sum(map_counts),
        "rtol": 1e-5,
    }

def run_overfit_qualification(
    *,
    args: argparse.Namespace,
    paths: Phase1Paths,
    metadata: Phase1Metadata,
    device: torch.device,
) -> dict[str, Any]:
    """Fit and qualify the fixed real-data overfit fixture."""
    from model_v2 import Phase1ScooHiC

    seed = require_seed_component("seed", args.seed)

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    fixture, fixed_tile_rows, fixed_context_indices = (
        build_overfit_fixture(
            metadata=metadata,
            paths=paths,
            seed=seed,
        )
    )

    model = Phase1ScooHiC().to(device)

    optimizer = build_phase1_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(fixed_tile_rows)
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    initial_metrics = evaluate_overfit_fixture(
        model=model,
        dataset=fixture,
        device=device,
    )
    initial_mse = float(initial_metrics["mean_mse"])
    oracle_metrics = compute_overfit_linear_oracle(
        dataset=fixture,
        device=device,
    )
    oracle_mse = float(oracle_metrics["mean_mse"])

    if oracle_mse <= 0.0 or oracle_mse >= initial_mse:
        raise RuntimeError(
            "Linear oracle must be positive and better than the "
            f"initialized model: initial={initial_mse}, "
            f"oracle={oracle_mse}"
        )

    if initial_mse <= 0.0 or not np.isfinite(initial_mse):
        raise RuntimeError(
            f"Invalid initialized overfit MSE: {initial_mse}"
        )

    global_step = 0
    epoch_history: list[dict[str, Any]] = []

    epoch_bar = tqdm(
        range(args.epochs),
        total=args.epochs,
        desc="Overfit epochs",
        unit="epoch",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        epoch_summary = run_training_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            paths=paths,
            device=device,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            contexts_per_tile=3,
            clip_norm=args.clip_norm,
            max_train_tiles=8,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            fixed_tile_rows=fixed_tile_rows,
            fixed_context_indices=fixed_context_indices,
        )
        global_step = int(epoch_summary["global_step"])

        epoch_history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "mean_mse": epoch_summary["mean_mse"],
                "first_step_loss": (
                    epoch_summary["first_step_loss"]
                ),
                "last_step_loss": (
                    epoch_summary["last_step_loss"]
                ),
            }
        )

        epoch_bar.set_postfix(
            mse=f"{epoch_summary['mean_mse']:.5f}",
        )

    epoch_bar.close()

    final_metrics = evaluate_overfit_fixture(
        model=model,
        dataset=fixture,
        device=device,
    )
    final_mse = float(final_metrics["mean_mse"])
    mse_ratio = final_mse / initial_mse

    base_gradient = summarize_gradient(
        "w_base",
        model.w_base,
    )
    hypernetwork_gradient = summarize_gradient(
        "hypernetwork.output_weight",
        model.hypernetwork.output_layer.weight,
    )
    spatial_output_gradient = summarize_gradient(
        "spatial_output.weight",
        model.spatial_output.weight,
    )

    optimizer_parameter_count = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    conditioning_difference = float(
        final_metrics["first_tile_context_max_difference"]
    )

    checks = {
        "mse_below_linear_oracle": final_mse < oracle_mse,
        "contexts_are_distinct": conditioning_difference > 1e-6,
        "base_gradient_finite_nonzero": (
            base_gradient["present"]
            and base_gradient["finite"]
            and base_gradient["nonzero"]
        ),
        "hypernetwork_gradient_finite_nonzero": (
            hypernetwork_gradient["present"]
            and hypernetwork_gradient["finite"]
            and hypernetwork_gradient["nonzero"]
        ),
        "spatial_output_gradient_finite_nonzero": (
            spatial_output_gradient["present"]
            and spatial_output_gradient["finite"]
            and spatial_output_gradient["nonzero"]
        ),
        "pair_embeddings_frozen": not bool(
            final_metrics["pair_embeddings_require_grad"]
        ),
        "optimizer_parameter_count_correct": (
            optimizer_parameter_count == 102_271
        ),
    }

    attainable_gap_closed = (
        (initial_mse - final_mse)
        / (initial_mse - oracle_mse)
    )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "tile_rows": fixed_tile_rows.tolist(),
        "context_indices": fixed_context_indices.tolist(),
        "context_ids": [
            metadata.context_ids[int(index)]
            for index in fixed_context_indices
        ],
        "epochs": args.epochs,
        "global_step": global_step,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "final_to_initial_mse_ratio": mse_ratio,
        "first_tile_context_max_difference": (
            conditioning_difference
        ),
        "optimizer_parameter_count": (
            optimizer_parameter_count
        ),
        "base_gradient": base_gradient,
        "hypernetwork_gradient": hypernetwork_gradient,
        "epoch_history": epoch_history,
        "oracle_mse": oracle_mse,
        "oracle_metrics": oracle_metrics,
        "final_to_oracle_mse_ratio": (
            final_mse / oracle_mse
        ),
        "attainable_gap_closed": attainable_gap_closed,
    }

def build_bounded_validation_dataset(
    *,
    metadata: Phase1Metadata,
    paths: Phase1Paths,
    seed: int,
    max_validation_tiles: int | None,
) -> Phase1TileDataset:
    """Build a deterministic validation-only dataset."""
    validation_rows = metadata.split_rows["validation"]
    available_tiles = len(validation_rows)

    if max_validation_tiles is None:
        selected_rows = validation_rows.copy()
    else:
        max_validation_tiles = require_seed_component(
            "max_validation_tiles",
            max_validation_tiles,
        )
        if max_validation_tiles == 0:
            raise ValueError(
                "max_validation_tiles must be greater than zero"
            )
        if max_validation_tiles > available_tiles:
            raise ValueError(
                f"max_validation_tiles ({max_validation_tiles}) exceeds "
                f"the {available_tiles} available validation tiles"
            )

        # Validation is never shuffled. A bounded run uses a stable prefix
        # of the authoritative validation-row order.
        selected_rows = validation_rows[:max_validation_tiles].copy()

    return Phase1TileDataset(
        metadata=metadata,
        targets_path=paths.targets,
        embeddings_dir=paths.embeddings_dir,
        split="validation",
        contexts_per_tile=len(metadata.context_ids),
        seed=seed,
        tile_rows=selected_rows,
        epoch=0,
    )


def run_bounded_validation(
    *,
    model: torch.nn.Module,
    dataset: Phase1TileDataset,
    device: torch.device,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    show_progress: bool = True,
    progress_position: int = 1,
    description: str = "Validation",
) -> dict[str, Any]:
    """Evaluate all contexts for each selected validation tile."""
    from model_v2 import (
        masked_upper_triangle_mse,
        masked_upper_triangle_pearson,
    )

    if dataset.split != "validation" or dataset.training:
        raise ValueError(
            "run_bounded_validation requires a validation-only dataset"
        )
    if len(dataset) == 0:
        raise ValueError("Validation dataset must not be empty")
    num_workers = require_seed_component("num_workers", num_workers)
    prefetch_factor = require_seed_component(
        "prefetch_factor",
        prefetch_factor,
    )
    if num_workers > 0 and prefetch_factor == 0:
        raise ValueError("prefetch_factor must be positive with workers")

    expected_context_ids = tuple(dataset.metadata.context_ids)
    expected_context_count = len(expected_context_ids)

    mse_sum = 0.0
    mse_map_count = 0
    pearson_sum = 0.0
    pearson_map_count = 0
    valid_pixel_count = 0
    evaluated_tile_rows: list[int] = []
    skipped_tile_rows: list[int] = []

    was_training = model.training
    model.eval()
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    validation_loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        **loader_options,
    )
    non_blocking = device.type == "cuda"

    try:
        with torch.inference_mode():
            progress_bar = tqdm(
                validation_loader,
                total=len(dataset),
                desc=description,
                unit="tile",
                position=progress_position,
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            )
            for example in progress_bar:

                if tuple(example["context_ids"]) != expected_context_ids:
                    raise RuntimeError(
                        "Validation context order differs from the "
                        "authoritative target order for "
                        f"tile_id={example['tile_id']!r}"
                    )

                pair_embedding = (
                    example["pair_embedding"]
                    .unsqueeze(0)
                    .to(
                        device=device,
                        non_blocking=non_blocking,
                    )
                )
                context_embedding = example["context_embedding"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                target = (
                    example["target"]
                    .unsqueeze(0)
                    .to(
                        device=device,
                        non_blocking=non_blocking,
                    )
                )
                resample_weights = (
                    example["resample_weights"]
                    .unsqueeze(0)
                    .to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=non_blocking,
                    )
                )

                if context_embedding.shape != (
                    expected_context_count,
                    EXPECTED_LATENT_DIM,
                ):
                    raise RuntimeError(
                        "Unexpected validation context-embedding shape "
                        f"{tuple(context_embedding.shape)} for "
                        f"tile_id={example['tile_id']!r}"
                    )

                expected_target_shape = (
                    1,
                    expected_context_count,
                    EXPECTED_TARGET_BINS,
                    EXPECTED_TARGET_BINS,
                )
                if target.shape != expected_target_shape:
                    raise RuntimeError(
                        f"Unexpected target shape {tuple(target.shape)}; "
                        f"expected {expected_target_shape}; "
                        f"tile_id={example['tile_id']!r}"
                    )

                valid_target_mask = torch.triu(
                    torch.isfinite(target),
                    diagonal=EXPECTED_MASKED_DIAGONALS,
                )
                if not bool(valid_target_mask.any().item()):
                    skipped_tile_rows.append(
                        int(example["target_tile_row"])
                    )
                    progress_bar.set_postfix(
                        skipped=len(skipped_tile_rows),
                    )
                    del resample_weights
                    del target
                    del context_embedding
                    del pair_embedding
                    continue
                prediction = model(
                    pair_embedding,
                    context_embedding,
                    resample_weights,
                )

                if prediction.shape != expected_target_shape:
                    raise RuntimeError(
                        f"Unexpected prediction shape "
                        f"{tuple(prediction.shape)}; "
                        f"expected {expected_target_shape}; "
                        f"tile_id={example['tile_id']!r}"
                    )
                if prediction.dtype != torch.float32:
                    raise RuntimeError(
                        "Validation predictions must be float32; "
                        f"found {prediction.dtype}; "
                        f"tile_id={example['tile_id']!r}"
                    )
                if not bool(torch.isfinite(prediction).all().item()):
                    raise FloatingPointError(
                        "Validation prediction contains non-finite values; "
                        f"tile_id={example['tile_id']!r}"
                    )

                loss, per_map_mse, pixel_counts = (
                    masked_upper_triangle_mse(
                        prediction,
                        target,
                        diagonal=EXPECTED_MASKED_DIAGONALS,
                    )
                )
                per_map_pearson = masked_upper_triangle_pearson(
                    prediction,
                    target,
                    diagonal=EXPECTED_MASKED_DIAGONALS,
                )

                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"Validation MSE is non-finite; "
                        f"tile_id={example['tile_id']!r}"
                    )

                valid_mse = per_map_mse[
                    torch.isfinite(per_map_mse)
                ]
                finite_pearson = per_map_pearson[
                    torch.isfinite(per_map_pearson)
                ]

                if valid_mse.numel() == 0:
                    raise RuntimeError(
                        "Validation tile has no finite per-map MSE values; "
                        f"tile_id={example['tile_id']!r}"
                    )

                maps_with_pixels = pixel_counts > 0
                if int(valid_mse.numel()) != int(
                    maps_with_pixels.sum().item()
                ):
                    raise FloatingPointError(
                        "A validation map with target pixels produced a "
                        f"non-finite MSE; tile_id={example['tile_id']!r}"
                    )

                mse_sum += float(valid_mse.sum().item())
                mse_map_count += int(valid_mse.numel())
                pearson_sum += float(finite_pearson.sum().item())
                pearson_map_count += int(finite_pearson.numel())
                valid_pixel_count += int(pixel_counts.sum().item())
                evaluated_tile_rows.append(
                    int(example["target_tile_row"])
                )

                progress_bar.set_postfix(
                    mse=f"{mse_sum / mse_map_count:.5f}",
                    pearson=(
                        f"{pearson_sum / pearson_map_count:.4f}"
                        if pearson_map_count > 0
                        else "undefined"
                    ),
                )

                # Release each tile before loading the next 512x512x128
                # embedding.
                del prediction
                del resample_weights
                del target
                del context_embedding
                del pair_embedding

            progress_bar.close()
    finally:
        model.train(was_training)

    if mse_map_count == 0:
        raise RuntimeError("Validation produced no finite map MSE values")
    if pearson_map_count == 0:
        raise RuntimeError(
            "Validation produced no defined per-map Pearson values"
        )

    expected_maps = (
        len(evaluated_tile_rows) * expected_context_count
    )
    undefined_pearson_count = expected_maps - pearson_map_count

    return {
        "split": "validation",
        "selected_tile_count": len(dataset),
        "tile_count": len(evaluated_tile_rows),
        "skipped_tile_count": len(skipped_tile_rows),
        "skipped_tile_rows": skipped_tile_rows,
        "tile_rows": evaluated_tile_rows,
        "context_count": expected_context_count,
        "expected_map_count": expected_maps,
        "mse_map_count": mse_map_count,
        "mean_mse": mse_sum / mse_map_count,
        "pearson_map_count": pearson_map_count,
        "undefined_pearson_count": undefined_pearson_count,
        "mean_pearson": pearson_sum / pearson_map_count,
        "valid_pixel_count": valid_pixel_count,
    }
def phase1_tensor_geometry(
    metadata: Phase1Metadata,
) -> dict[str, Any]:
    """Return resume-critical tensor dimensions."""
    return {
        "pair_embedding_shape": list(EXPECTED_PAIR_SHAPE),
        "target_map_shape": [
            EXPECTED_TARGET_BINS,
            EXPECTED_TARGET_BINS,
        ],
        "context_count": len(metadata.context_ids),
        "latent_dim": EXPECTED_LATENT_DIM,
        "masked_diagonals": EXPECTED_MASKED_DIAGONALS,
    }


def build_effective_configuration(
    args: argparse.Namespace,
    paths: Phase1Paths,
) -> dict[str, Any]:
    """Build a JSON-compatible record of the effective run settings."""
    options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    return {
        "options": options,
        "paths": paths.as_serializable_dict(),
    }


RESUME_CRITICAL_OPTION_KEYS = (
    "seed",
    "contexts_per_tile",
    "learning_rate",
    "weight_decay",
    "clip_norm",
    "warmup_steps",
    "max_train_tiles",
    "max_validation_tiles",
)

RESUME_CRITICAL_PATH_KEYS = (
    "model_file",
    "tiles",
    "centroids",
    "target_contexts",
    "targets",
    "embeddings_dir",
    "output_dir",
)


def validate_resume_effective_configuration(
    checkpoint_configuration: Mapping[str, Any],
    current_configuration: Mapping[str, Any],
) -> dict[str, int]:
    """Reject mixed-run resumes while allowing an epoch extension."""
    configuration_pairs: list[
        tuple[str, Mapping[str, Any], Mapping[str, Any]]
    ] = []

    for section_name in ("options", "paths"):
        checkpoint_section = checkpoint_configuration.get(section_name)
        current_section = current_configuration.get(section_name)
        if not isinstance(checkpoint_section, Mapping):
            raise ValueError(
                "Checkpoint effective configuration is missing mapping "
                f"section {section_name!r}"
            )
        if not isinstance(current_section, Mapping):
            raise ValueError(
                "Current effective configuration is missing mapping "
                f"section {section_name!r}"
            )
        configuration_pairs.append(
            (
                section_name,
                checkpoint_section,
                current_section,
            )
        )

    checkpoint_options = configuration_pairs[0][1]
    current_options = configuration_pairs[0][2]
    checkpoint_paths = configuration_pairs[1][1]
    current_paths = configuration_pairs[1][2]

    mismatches: dict[str, dict[str, Any]] = {}

    for key in RESUME_CRITICAL_OPTION_KEYS:
        checkpoint_value = checkpoint_options.get(key)
        current_value = current_options.get(key)
        if checkpoint_value != current_value:
            mismatches[f"options.{key}"] = {
                "checkpoint": checkpoint_value,
                "current": current_value,
            }

    for key in RESUME_CRITICAL_PATH_KEYS:
        checkpoint_value = checkpoint_paths.get(key)
        current_value = current_paths.get(key)
        if checkpoint_value != current_value:
            mismatches[f"paths.{key}"] = {
                "checkpoint": checkpoint_value,
                "current": current_value,
            }

    checkpoint_epochs = require_seed_component(
        "checkpoint configured epochs",
        checkpoint_options.get("epochs"),
    )
    current_epochs = require_seed_component(
        "current configured epochs",
        current_options.get("epochs"),
    )
    if current_epochs < checkpoint_epochs:
        mismatches["options.epochs"] = {
            "checkpoint": checkpoint_epochs,
            "current": current_epochs,
            "reason": "the training horizon may only stay fixed or extend",
        }

    if mismatches:
        raise ValueError(
            "Resume-critical configuration differs from the checkpoint: "
            f"{mismatches}"
        )

    return {
        "checkpoint_epochs": checkpoint_epochs,
        "current_epochs": current_epochs,
    }


def reconcile_resume_learning_rates(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    global_step: int,
) -> list[float]:
    """Position the next optimizer LR on the current schedule horizon."""
    global_step = require_seed_component("global_step", global_step)

    if scheduler.last_epoch != global_step:
        raise RuntimeError(
            "Loaded scheduler position does not match the checkpoint "
            f"global step: scheduler.last_epoch={scheduler.last_epoch}, "
            f"global_step={global_step}"
        )
    if len(scheduler.base_lrs) != len(scheduler.optimizer.param_groups):
        raise RuntimeError(
            "Scheduler base-LR count does not match optimizer groups"
        )
    if len(scheduler.lr_lambdas) != len(scheduler.optimizer.param_groups):
        raise RuntimeError(
            "Scheduler lambda count does not match optimizer groups"
        )

    next_learning_rates = [
        float(base_lr * lr_lambda(global_step))
        for base_lr, lr_lambda in zip(
            scheduler.base_lrs,
            scheduler.lr_lambdas,
            strict=True,
        )
    ]
    for parameter_group, learning_rate in zip(
        scheduler.optimizer.param_groups,
        next_learning_rates,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate

    scheduler._last_lr = next_learning_rates
    return next_learning_rates
def save_phase1_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_metrics: Mapping[str, Any],
    metadata: Phase1Metadata,
    effective_configuration: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Atomically save all state needed to resume Phase 1."""
    epoch = require_seed_component("epoch", epoch)
    global_step = require_seed_component(
        "global_step",
        global_step,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": dict(best_metrics),
        "context_ids": list(metadata.context_ids),
        "tensor_geometry": phase1_tensor_geometry(metadata),
        "effective_configuration": dict(
            effective_configuration
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        ),
    }

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        torch.save(payload, temporary_path)

        # Both paths are in the same directory, so replace is atomic.
        temporary_path.replace(checkpoint_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_phase1_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metadata: Phase1Metadata,
    device: torch.device,
    expected_effective_configuration: (
        Mapping[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    """Validate and restore a Phase 1 checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Checkpoint must contain a mapping: {checkpoint_path}"
        )

    required_keys = {
        "checkpoint_version",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "global_step",
        "best_metrics",
        "context_ids",
        "tensor_geometry",
        "effective_configuration",
        "torch_rng_state",
        "cuda_rng_state",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    if checkpoint["checkpoint_version"] != 1:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{checkpoint['checkpoint_version']!r}"
        )

    expected_context_ids = tuple(metadata.context_ids)
    checkpoint_context_ids = tuple(checkpoint["context_ids"])

    if checkpoint_context_ids != expected_context_ids:
        raise ValueError(
            "Checkpoint context order does not match the current target "
            f"order: checkpoint={checkpoint_context_ids}, "
            f"current={expected_context_ids}"
        )

    expected_geometry = phase1_tensor_geometry(metadata)
    if checkpoint["tensor_geometry"] != expected_geometry:
        raise ValueError(
            "Checkpoint tensor geometry does not match the current data: "
            f"checkpoint={checkpoint['tensor_geometry']}, "
            f"current={expected_geometry}"
        )

    epoch = require_seed_component(
        "checkpoint epoch",
        checkpoint["epoch"],
    )
    global_step = require_seed_component(
        "checkpoint global_step",
        checkpoint["global_step"],
    )

    if not isinstance(checkpoint["best_metrics"], Mapping):
        raise ValueError(
            "Checkpoint best_metrics must be a mapping"
        )
    if not isinstance(
        checkpoint["effective_configuration"],
        Mapping,
    ):
        raise ValueError(
            "Checkpoint effective_configuration must be a mapping"
        )

    if expected_effective_configuration is not None:
        validate_resume_effective_configuration(
            checkpoint["effective_configuration"],
            expected_effective_configuration,
        )

    # Compatibility checks above intentionally occur before any state is
    # applied, especially before loading optimizer state.
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    torch_rng_state = checkpoint["torch_rng_state"]
    if not isinstance(torch_rng_state, torch.Tensor):
        raise ValueError(
            "Checkpoint torch_rng_state must be a tensor"
        )
    torch.set_rng_state(torch_rng_state.cpu())

    cuda_rng_state = checkpoint["cuda_rng_state"]
    if device.type == "cuda" and cuda_rng_state is not None:
        if not isinstance(cuda_rng_state, torch.Tensor):
            raise ValueError(
                "Checkpoint cuda_rng_state must be a tensor or None"
            )
        torch.cuda.set_rng_state(
            cuda_rng_state.cpu(),
            device=device,
        )

    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": dict(checkpoint["best_metrics"]),
        "effective_configuration": dict(
            checkpoint["effective_configuration"]
        ),
    }


def append_jsonl_metric(
    metric_path: Path,
    record: Mapping[str, Any],
) -> None:
    """Append one finite, JSON-compatible metric record."""
    metric_path = Path(metric_path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(
        dict(record),
        sort_keys=True,
        allow_nan=False,
    )

    with metric_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")


def read_jsonl_metrics(
    metric_path: Path,
) -> list[dict[str, Any]]:
    """Read and validate an existing JSONL metric log."""
    metric_path = Path(metric_path)

    if not metric_path.is_file():
        raise FileNotFoundError(
            f"Metric log does not exist: {metric_path}"
        )

    records: list[dict[str, Any]] = []

    with metric_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(
                    f"Blank line in metric log at line {line_number}"
                )

            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(
                    "Metric-log records must be JSON objects; "
                    f"line {line_number} contains "
                    f"{type(record).__name__}"
                )
            records.append(record)

    return records


def run_checkpoint_metric_round_trip(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metadata: Phase1Metadata,
    args: argparse.Namespace,
    paths: Phase1Paths,
    device: torch.device,
    pair_embedding: torch.Tensor,
    context_embedding: torch.Tensor,
    resample_weights: torch.Tensor,
    global_step: int,
    validation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove checkpoint and JSONL state can be written and restored."""
    global_step = require_seed_component(
        "global_step",
        global_step,
    )

    best_metrics = {
        "mean_pearson": float(
            validation_summary["mean_pearson"]
        ),
        "mean_mse": float(validation_summary["mean_mse"]),
        "epoch": 0,
        "global_step": global_step,
    }
    effective_configuration = build_effective_configuration(
        args,
        paths,
    )

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            reference_prediction = model(
                pair_embedding,
                context_embedding,
                resample_weights,
            ).detach().clone()

        saved_learning_rates = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]
        saved_scheduler_epoch = scheduler.last_epoch

        with tempfile.TemporaryDirectory(
            prefix="scoo_hic_phase1_roundtrip_"
        ) as temporary_directory:
            temporary_directory_path = Path(temporary_directory)
            checkpoint_path = (
                temporary_directory_path / "roundtrip.pt"
            )
            metric_path = (
                temporary_directory_path / "metrics.jsonl"
            )

            save_phase1_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                global_step=global_step,
                best_metrics=best_metrics,
                metadata=metadata,
                effective_configuration=effective_configuration,
                device=device,
            )

            metric_record = {
                "event": "validation",
                "epoch": 0,
                "global_step": global_step,
                "metrics": dict(validation_summary),
            }
            append_jsonl_metric(
                metric_path,
                metric_record,
            )

            # Perturb a parameter so the test proves that loading performs
            # a real restoration rather than comparing unchanged state.
            with torch.no_grad():
                model.b_base.add_(1.0)

            with torch.inference_mode():
                perturbed_prediction = model(
                    pair_embedding,
                    context_embedding,
                    resample_weights,
                )

            perturbation_difference = float(
                (
                    perturbed_prediction
                    - reference_prediction
                )
                .abs()
                .max()
                .item()
            )
            if perturbation_difference == 0.0:
                raise RuntimeError(
                    "Checkpoint round-trip perturbation did not change "
                    "the prediction"
                )

            loaded_state = load_phase1_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
                device=device,
                expected_effective_configuration=(
                    effective_configuration
                ),
            )

            with torch.inference_mode():
                restored_prediction = model(
                    pair_embedding,
                    context_embedding,
                    resample_weights,
                )

            restoration_error = float(
                (
                    restored_prediction
                    - reference_prediction
                )
                .abs()
                .max()
                .item()
            )
            if restoration_error > 1e-6:
                raise RuntimeError(
                    "Reloaded checkpoint changed the fixed-batch "
                    f"prediction by {restoration_error}"
                )

            restored_learning_rates = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]
            if restored_learning_rates != saved_learning_rates:
                raise RuntimeError(
                    "Optimizer learning rates were not restored: "
                    f"saved={saved_learning_rates}, "
                    f"restored={restored_learning_rates}"
                )
            if scheduler.last_epoch != saved_scheduler_epoch:
                raise RuntimeError(
                    "Scheduler position was not restored: "
                    f"saved={saved_scheduler_epoch}, "
                    f"restored={scheduler.last_epoch}"
                )

            metric_records = read_jsonl_metrics(metric_path)
            if metric_records != [metric_record]:
                raise RuntimeError(
                    "Metric-log round trip changed the written record"
                )

            result = {
                "checkpoint_version": 1,
                "loaded_epoch": loaded_state["epoch"],
                "loaded_global_step": loaded_state["global_step"],
                "restoration_error": restoration_error,
                "perturbation_difference": (
                    perturbation_difference
                ),
                "optimizer_learning_rates_restored": True,
                "scheduler_last_epoch": scheduler.last_epoch,
                "metric_record_count": len(metric_records),
                "metric_record_round_trip": True,
            }
    finally:
        model.train(was_training)

    return result

def run_real_data_dry_run(
    args: argparse.Namespace,
    paths: Phase1Paths,
    metadata: Phase1Metadata,
) -> torch.device:
    """Run one real-data forward/loss/backward acceptance check."""
    from model_v2 import (
        Phase1ScooHiC,
        masked_upper_triangle_mse,
        masked_upper_triangle_pearson,
    )

    device = resolve_device(args.device)
    seed = require_seed_component("seed", args.seed)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dry_run_rows = make_epoch_tile_order(
        metadata.split_rows["train"],
        seed=seed,
        epoch=0,
        limit=1,
    )
    dry_run_dataset = Phase1TileDataset(
        metadata=metadata,
        targets_path=paths.targets,
        embeddings_dir=paths.embeddings_dir,
        split="train",
        contexts_per_tile=args.contexts_per_tile,
        seed=seed,
        tile_rows=dry_run_rows,
        epoch=0,
    )

    example = dry_run_dataset[0]
    print_tile_example_summary(example)

    # Add the tile batch dimension. Contexts remain the shared [C, 14]
    # form accepted by Phase1ScooHiC.
    pair_embedding = example["pair_embedding"].unsqueeze(0).to(
        device=device,
        non_blocking=False,
    )
    context_embedding = example["context_embedding"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    target = example["target"].unsqueeze(0).to(
        device=device,
        non_blocking=False,
    )
    resample_weights = example["resample_weights"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )

    if pair_embedding.requires_grad:
        raise RuntimeError(
            "Frozen AlphaGenome pair embeddings must not require gradients"
        )

    model = Phase1ScooHiC().to(device)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count != 102_271:
        raise RuntimeError(
            f"Expected 102,271 trainable parameters; found {parameter_count}"
        )

    prediction = model(
        pair_embedding,
        context_embedding,
        resample_weights,
    )

    expected_shape = (
        1,
        len(example["context_ids"]),
        EXPECTED_TARGET_BINS,
        EXPECTED_TARGET_BINS,
    )
    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"Prediction has shape {tuple(prediction.shape)}; "
            f"expected {expected_shape}"
        )
    if prediction.dtype != torch.float32:
        raise RuntimeError(
            f"Prediction must be float32; found {prediction.dtype}"
        )
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("Dry-run prediction contains non-finite values")

    symmetry_error = float(
        (
            prediction
            - prediction.transpose(-1, -2)
        )
        .abs()
        .max()
        .item()
    )
    if symmetry_error > 1e-6:
        raise RuntimeError(
            "Prediction is not symmetric within tolerance; "
            f"maximum error is {symmetry_error}"
        )

    if prediction.shape[1] > 1:
        initial_context_difference = float(
            (
                prediction[:, 1:]
                - prediction[:, :1]
            )
            .abs()
            .max()
            .item()
        )
    else:
        initial_context_difference = 0.0

    if initial_context_difference != 0.0:
        raise RuntimeError(
            "Initial predictions differ across contexts even though the "
            "conditioned residual heads are zero-initialized; maximum "
            f"difference is {initial_context_difference}"
        )

    loss, per_map_mse, valid_pixel_counts = (
        masked_upper_triangle_mse(
            prediction,
            target,
            diagonal=EXPECTED_MASKED_DIAGONALS,
        )
    )
    correlations = masked_upper_triangle_pearson(
        prediction,
        target,
        diagonal=EXPECTED_MASKED_DIAGONALS,
    )

    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(
            f"Dry-run loss is non-finite: {float(loss.item())}"
        )
    if not bool((valid_pixel_counts > 0).all().item()):
        raise RuntimeError(
            "At least one dry-run tile/context map has no valid pixels"
        )

    loss.backward()

    gradient_summaries = {
        summary["name"]: summary
        for summary in (
            summarize_gradient("w_base", model.w_base),
            summarize_gradient("b_base", model.b_base),
            summarize_gradient(
                "hypernetwork.output_weight",
                model.hypernetwork.output_layer.weight,
            ),
            summarize_gradient(
                "hypernetwork.output_bias",
                model.hypernetwork.output_layer.bias,
            ),
            summarize_gradient(
                "spatial_output.weight",
                model.spatial_output.weight,
            ),
            summarize_gradient(
                "spatial_output.bias",
                model.spatial_output.bias,
            ),
        )
    }

    failed_gradients = [
        name
        for name, summary in gradient_summaries.items()
        if not (
            summary["present"]
            and summary["finite"]
            and summary["nonzero"]
        )
    ]
    if failed_gradients:
        raise RuntimeError(
            "Required gradients were absent, non-finite, or zero: "
            f"{failed_gradients}"
        )

    available_train_tiles = len(metadata.split_rows["train"])
    planned_train_tiles = (
        args.max_train_tiles
        if args.max_train_tiles is not None
        else available_train_tiles
    )
    if planned_train_tiles > available_train_tiles:
        raise ValueError(
            f"--max-train-tiles ({planned_train_tiles}) exceeds "
            f"the {available_train_tiles} available training tiles"
        )

    # The dry run first performs the isolated Step 10 update, then exercises
    # the Step 13 epoch loop.
    total_steps = 1 + args.epochs * planned_train_tiles

    optimizer = build_phase1_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    train_step_summary = safe_train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pair_embedding=pair_embedding,
        context_embedding=context_embedding,
        target=target,
        resample_weights=resample_weights,
        clip_norm=args.clip_norm,
        global_step=0,
        tile_id=example["tile_id"],
        context_ids=example["context_ids"],
    )

    epoch_summary = run_training_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        paths=paths,
        device=device,
        seed=seed,
        epoch=0,
        global_step=train_step_summary["global_step"],
        contexts_per_tile=args.contexts_per_tile,
        clip_norm=args.clip_norm,
        max_train_tiles=planned_train_tiles,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Keep a default dry run small. An explicit --max-validation-tiles
    # controls the bound when supplied.
    dry_run_validation_limit = (
        args.max_validation_tiles
        if args.max_validation_tiles is not None
        else 1
    )
    validation_dataset = build_bounded_validation_dataset(
        metadata=metadata,
        paths=paths,
        seed=seed,
        max_validation_tiles=dry_run_validation_limit,
    )
    validation_summary = run_bounded_validation(
        model=model,
        dataset=validation_dataset,
        device=device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    round_trip_summary = run_checkpoint_metric_round_trip(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        args=args,
        paths=paths,
        device=device,
        pair_embedding=pair_embedding,
        context_embedding=context_embedding,
        resample_weights=resample_weights,
        global_step=epoch_summary["global_step"],
        validation_summary=validation_summary,
    )

    correlation_values = [
        (
            float(value)
            if bool(torch.isfinite(value).item())
            else None
        )
        for value in correlations.detach().cpu().flatten()
    ]
    finite_correlations = correlations[
        torch.isfinite(correlations)
    ]
    mean_pearson = (
        float(finite_correlations.mean().item())
        if finite_correlations.numel() > 0
        else None
    )

    summary = {
        "dry_run": {
            "device": str(device),
            "target_tile_row": example["target_tile_row"],
            "tile_id": example["tile_id"],
            "context_ids": list(example["context_ids"]),
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
            "prediction_finite": True,
            "prediction_symmetry_error": symmetry_error,
            "initial_context_max_difference": (
                initial_context_difference
            ),
            "loss": float(loss.detach().item()),
            "per_map_mse": (
                per_map_mse.detach().cpu().flatten().tolist()
            ),
            "valid_pixel_counts": (
                valid_pixel_counts.detach().cpu().flatten().tolist()
            ),
            "per_map_pearson": correlation_values,
            "mean_pearson": mean_pearson,
            "trainable_parameter_count": parameter_count,
            "pair_embedding_requires_grad": (
                pair_embedding.requires_grad
            ),
            "gradients": gradient_summaries,
            "safe_train_step": train_step_summary,
            "bounded_validation": validation_summary,
            "checkpoint_metric_round_trip": round_trip_summary,
            "deterministic_epoch": epoch_summary,
        }
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    return device

def validation_is_better(
    candidate: Mapping[str, Any],
    best_metrics: Mapping[str, Any] | None,
) -> bool:
    """Prefer higher Pearson, using lower MSE as the tie-breaker."""
    candidate_pearson = float(candidate["mean_pearson"])
    candidate_mse = float(candidate["mean_mse"])

    if not np.isfinite(candidate_pearson):
        raise FloatingPointError(
            f"Candidate validation Pearson is non-finite: {candidate_pearson}"
        )
    if not np.isfinite(candidate_mse):
        raise FloatingPointError(
            f"Candidate validation MSE is non-finite: {candidate_mse}"
        )

    if not best_metrics:
        return True

    best_pearson = float(best_metrics["mean_pearson"])
    best_mse = float(best_metrics["mean_mse"])

    return (
        candidate_pearson > best_pearson
        or (
            candidate_pearson == best_pearson
            and candidate_mse < best_mse
        )
    )


def run_phase1_training(
    *,
    args: argparse.Namespace,
    paths: Phase1Paths,
    metadata: Phase1Metadata,
    device: torch.device,
    guardrail_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Run deterministic Phase 1 training with validation and checkpoints."""
    from model_v2 import Phase1ScooHiC

    seed = require_seed_component("seed", args.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    output_dir = paths.output_dir
    last_checkpoint_path = output_dir / "last.pt"
    best_checkpoint_path = output_dir / "best.pt"
    metric_path = output_dir / "metrics.jsonl"

    managed_outputs = (
        last_checkpoint_path,
        best_checkpoint_path,
        metric_path,
    )
    if paths.resume is None:
        existing_outputs = [
            str(path)
            for path in managed_outputs
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "Refusing to overwrite an existing Phase 1 run. Resume "
                "from last.pt or move the existing managed outputs: "
                f"{existing_outputs}"
            )

    train_tile_count = (
        args.max_train_tiles
        if args.max_train_tiles is not None
        else len(metadata.split_rows["train"])
    )
    total_steps = args.epochs * train_tile_count

    model = Phase1ScooHiC().to(device)
    optimizer = build_phase1_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    effective_configuration = build_effective_configuration(
        args,
        paths,
    )

    start_epoch = 0
    global_step = 0
    best_metrics: dict[str, Any] = {}

    if paths.resume is not None:
        if paths.resume.resolve() != last_checkpoint_path.resolve():
            raise ValueError(
                "Phase 1 may only resume from its fixed last checkpoint: "
                f"expected={last_checkpoint_path}, found={paths.resume}"
            )

        loaded_state = load_phase1_checkpoint(
            paths.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            device=device,
            expected_effective_configuration=(
                effective_configuration
            ),
        )
        start_epoch = int(loaded_state["epoch"]) + 1
        global_step = int(loaded_state["global_step"])
        best_metrics = dict(loaded_state["best_metrics"])

        maximum_global_step = start_epoch * train_tile_count
        if global_step > maximum_global_step:
            raise RuntimeError(
                "Resume checkpoint global step exceeds the requested "
                "epoch geometry: "
                f"maximum={maximum_global_step}, found={global_step}"
            )
        reconcile_resume_learning_rates(
            scheduler,
            global_step=global_step,
        )


    if start_epoch >= args.epochs:
        raise ValueError(
            "Resume checkpoint has already completed the requested number "
            f"of epochs: checkpoint_epoch={start_epoch - 1}, "
            f"requested_epochs={args.epochs}"
        )

    validation_dataset = build_bounded_validation_dataset(
        metadata=metadata,
        paths=paths,
        seed=seed,
        max_validation_tiles=args.max_validation_tiles,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        total=args.epochs - start_epoch,
        initial=0,
        desc="Phase 1 epochs",
        unit="epoch",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    try:
        for epoch in epoch_bar:
            training_summary = run_training_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
                paths=paths,
                device=device,
                seed=seed,
                epoch=epoch,
                global_step=global_step,
                contexts_per_tile=args.contexts_per_tile,
                clip_norm=args.clip_norm,
                max_train_tiles=args.max_train_tiles,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                show_progress=True,
                progress_position=1,
            )
            global_step = int(training_summary["global_step"])

            validation_summary = run_bounded_validation(
                model=model,
                dataset=validation_dataset,
                device=device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                show_progress=True,
                progress_position=1,
                description=f"Validate epoch {epoch + 1}",
            )

            candidate_best = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_pearson": float(
                    validation_summary["mean_pearson"]
                ),
                "mean_mse": float(validation_summary["mean_mse"]),
            }
            improved = validation_is_better(
                candidate_best,
                best_metrics,
            )
            if improved:
                best_metrics = candidate_best

            metric_record = {
                "event": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train": training_summary,
                "validation": validation_summary,
                "is_best": improved,
                "best_metrics": best_metrics,
                "guardrails": dict(guardrail_summary),
            }
            if improved:
                save_phase1_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    metadata=metadata,
                    effective_configuration=effective_configuration,
                    device=device,
                )

            # last.pt is the resumable commit marker for the epoch.
            save_phase1_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metrics=best_metrics,
                metadata=metadata,
                effective_configuration=effective_configuration,
                device=device,
            )
            append_jsonl_metric(metric_path, metric_record)

            epoch_bar.set_postfix(
                train_mse=f"{training_summary['mean_mse']:.5f}",
                val_mse=f"{validation_summary['mean_mse']:.5f}",
                val_r=f"{validation_summary['mean_pearson']:.4f}",
                best_epoch=int(best_metrics["epoch"]) + 1,
            )
    finally:
        epoch_bar.close()

    return {
        "output_dir": str(output_dir),
        "last_checkpoint": str(last_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "metric_log": str(metric_path),
        "epochs_completed": args.epochs - start_epoch,
        "start_epoch": start_epoch,
        "final_epoch": args.epochs - 1,
        "global_step": global_step,
        "best_metrics": best_metrics,
    }
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed

def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-AlphaGenome Phase 1 scoo-hic v2 decoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Project YAML configuration.",
    )
    paths.add_argument(
        "--tiles",
        type=Path,
        help="Override the retained fold-safe tile table.",
    )
    paths.add_argument(
        "--centroids",
        type=Path,
        help="Override the RNA latent centroid table.",
    )
    paths.add_argument(
        "--target-contexts",
        type=Path,
        help="Override the ordered target-context table.",
    )
    paths.add_argument(
        "--targets",
        type=Path,
        help="Override the Phase 1 target Zarr store.",
    )
    paths.add_argument(
        "--embeddings-dir",
        type=Path,
        help="Override the cached AlphaGenome embedding directory.",
    )
    paths.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint from which to resume.",
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override the training output directory. The default remains "
            "results/phase1_model_v2."
        ),
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, or cuda:0.",
    )
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--num-workers", type=nonnegative_int, default=0)
    runtime.add_argument(
        "--prefetch-factor",
        type=positive_int,
        default=4,
        help="Samples prefetched per DataLoader worker.",
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load one real training tile and run forward, loss, "
            "metric, and backward checks."
        ),
    )
    runtime.add_argument(
        "--overfit",
        action="store_true",
        help="Run the bounded real-data overfit qualification.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=positive_int, default=40)
    training.add_argument(
        "--contexts-per-tile",
        type=positive_int,
        default=8,
    )
    training.add_argument(
        "--learning-rate",
        type=positive_float,
        default=4e-4,
    )
    training.add_argument(
        "--weight-decay",
        type=nonnegative_float,
        default=1e-6,
    )
    training.add_argument(
        "--clip-norm",
        type=positive_float,
        default=1.0,
    )
    training.add_argument(
        "--warmup-steps",
        type=nonnegative_int,
        default=1000,
    )
    training.add_argument(
        "--max-train-tiles",
        type=positive_int,
        help="Optional deterministic training-tile limit.",
    )
    training.add_argument(
        "--max-validation-tiles",
        type=positive_int,
        help="Optional deterministic validation-tile limit.",
    )

    args = parser.parse_args(argv)

    if args.contexts_per_tile > 20:
        parser.error("--contexts-per-tile cannot exceed 20")

    return args


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, Mapping):
        raise ValueError(
            f"Expected a YAML mapping in configuration: {config_path}"
        )

    return dict(config)


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_override(
    override: Path | None,
    default: Path,
    *,
    relative_to: Path,
) -> Path:
    if override is None:
        return default.resolve()
    return resolve_path(override, relative_to)


def require_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be a mapping")
    return value


def resolve_paths(args: argparse.Namespace) -> Phase1Paths:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    root_paths = require_mapping(config, "root_paths")
    if "repo_root" not in root_paths or "data_root" not in root_paths:
        raise ValueError(
            "Configuration root_paths must define repo_root and data_root"
        )

    configured_repo_root = resolve_path(
        root_paths["repo_root"],
        config_path.parent.parent,
    )
    data_root = resolve_path(
        root_paths["data_root"],
        configured_repo_root,
    )

    mm10 = require_mapping(config, "mm10")
    if "embeddings" not in mm10:
        raise ValueError("Configuration mm10 must define embeddings")

    model_source_dir = configured_repo_root / "src" / "scoo-hic"

    paths = Phase1Paths(
        config=config_path,
        repo_root=configured_repo_root,
        data_root=data_root,
        model_source_dir=model_source_dir,
        model_file=model_source_dir / "model_v2.py",
        tiles=resolve_override(
            args.tiles,
            data_root / "processed" / "multiome" / "tiles.parquet",
            relative_to=configured_repo_root,
        ),
        centroids=resolve_override(
            args.centroids,
            data_root
            / "processed"
            / "multiome"
            / "train_in"
            / "rna_scvi_14_cell_type_centroids.parquet",
            relative_to=configured_repo_root,
        ),
        target_contexts=resolve_override(
            args.target_contexts,
            data_root
            / "processed"
            / "multiome_rna"
            / "target_contexts.parquet",
            relative_to=configured_repo_root,
        ),
        targets=resolve_override(
            args.targets,
            data_root
            / "processed"
            / "multiome_rna"
            / "hic_targets.zarr",
            relative_to=configured_repo_root,
        ),
        embeddings_dir=resolve_override(
            args.embeddings_dir,
            resolve_path(mm10["embeddings"], data_root),
            relative_to=configured_repo_root,
        ),
        output_dir=resolve_override(
            args.output_dir,
            DISCOVERED_REPO_ROOT / PHASE1_RESULTS_SUBDIRECTORY,
            relative_to=configured_repo_root,
        ),
        resume=(
            resolve_path(args.resume, configured_repo_root)
            if args.resume is not None
            else None
        ),
    )

    validate_paths(paths)
    return paths


def validate_paths(paths: Phase1Paths) -> None:
    required_files = {
        "config": paths.config,
        "model": paths.model_file,
        "tiles": paths.tiles,
        "centroids": paths.centroids,
        "target contexts": paths.target_contexts,
    }
    required_directories = {
        "repository root": paths.repo_root,
        "data root": paths.data_root,
        "model source directory": paths.model_source_dir,
        "target Zarr": paths.targets,
        "embedding directory": paths.embeddings_dir,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not path.is_file()
    ]
    missing.extend(
        f"{label}: {path}"
        for label, path in required_directories.items()
        if not path.is_dir()
    )

    if paths.resume is not None and not paths.resume.is_file():
        missing.append(f"resume checkpoint: {paths.resume}")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Required Phase 1 paths are missing:\n  - {details}"
        )


def add_model_source_to_path(model_source_dir: Path) -> None:
    source = str(model_source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)


def print_resolved_configuration(
    args: argparse.Namespace,
    paths: Phase1Paths,
) -> None:
    options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    print(
        json.dumps(
            {
                "paths": paths.as_serializable_dict(),
                "options": options,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    add_model_source_to_path(paths.model_source_dir)

    print_resolved_configuration(args, paths)

    metadata = load_phase1_metadata(paths)
    print_metadata_summary(metadata)

    if args.dry_run:
        device = run_real_data_dry_run(
            args,
            paths,
            metadata,
        )

        # The dry-run tensors and Zarr handles leave scope when the helper
        # returns. Explicit cleanup avoids CUDA shutdown-time aborts.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return

    device = resolve_device(args.device)
    guardrail_summary = validate_launch_guardrails(
        args=args,
        metadata=metadata,
        device=device,
    )

    if args.overfit:
        qualification = run_overfit_qualification(
            args=args,
            paths=paths,
            metadata=metadata,
            device=device,
        )

        paths.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        append_jsonl_metric(
            paths.output_dir / "overfit_metrics.jsonl",
            {
                "event": "overfit_qualification",
                "guardrails": guardrail_summary,
                "qualification": qualification,
            },
        )

        print(
            json.dumps(
                {
                    "guardrails": guardrail_summary,
                    "overfit_qualification": qualification,
                },
                indent=2,
                sort_keys=True,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not qualification["passed"]:
            raise RuntimeError(
                "Overfit qualification failed one or more acceptance "
                f"checks: {qualification['checks']}"
            )

        return

    try:
        training_summary = run_phase1_training(
            args=args,
            paths=paths,
            metadata=metadata,
            device=device,
            guardrail_summary=guardrail_summary,
        )

        print(
            json.dumps(
                {
                    "guardrails": guardrail_summary,
                    "training": training_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
