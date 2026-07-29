"""Atomic, validated output writing for Hi-Scooby inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

from numcodecs import Blosc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import zarr


SMOOTH_BINS = 200
SPARSE_BINS = 100
SPARSE_FLOAT_FIELDS = (
    "expected_contacts_per_million",
    "expected_count",
    "nb2_dispersion",
    "residual_score",
    "tile_band_probability",
)
SPARSE_COUNT_FIELDS = (
    "predictive_lower",
    "predictive_upper",
    "simulated_count",
)
SPARSE_PAIR_FLOAT_FIELDS = (
    "expected_contacts_per_million",
    "expected_count",
    "nb2_dispersion",
    "residual_score",
    "owner_tile_band_probability",
)
SPARSE_PAIR_COLUMNS = (
    "pair_id",
    "tile_row",
    "tile_id",
    "chrom",
    "bin_i",
    "bin_j",
    "owner_row",
    "owner_column",
    "distance_bp",
    "distance_band",
    *SPARSE_PAIR_FLOAT_FIELDS,
    *SPARSE_COUNT_FIELDS,
)


def build_output_tile_table(
    smooth_tiles: pd.DataFrame,
    sparse_tiles: pd.DataFrame,
) -> pd.DataFrame:
    """Combine shared AlphaGenome windows with both target coordinate grids."""
    shared = [
        "tile_id",
        "tile_index",
        "chrom",
        "input_start",
        "input_end",
        "split",
    ]
    smooth_required = set(shared) | {"target_start", "target_end"}
    sparse_required = set(shared) | {"target_start", "target_end"}
    if missing := smooth_required - set(smooth_tiles.columns):
        raise ValueError(
            f"Smooth tile table lacks columns: {sorted(missing)}"
        )
    if missing := sparse_required - set(sparse_tiles.columns):
        raise ValueError(
            f"Sparse tile table lacks columns: {sorted(missing)}"
        )

    smooth = smooth_tiles.reset_index(drop=True)
    sparse = sparse_tiles.reset_index(drop=True)
    if len(smooth) != len(sparse):
        raise ValueError(
            f"Smooth and sparse tile counts differ: "
            f"{len(smooth)} versus {len(sparse)}"
        )
    if not smooth[shared].equals(sparse[shared]):
        raise ValueError(
            "Smooth and sparse tables disagree on AlphaGenome tile identity"
        )
    if not np.array_equal(
        smooth["tile_index"].to_numpy(np.int64),
        np.arange(len(smooth), dtype=np.int64),
    ):
        raise ValueError("Tile indices must be contiguous and row aligned")

    output = smooth[shared].copy()
    output["smooth_target_start"] = smooth["target_start"].to_numpy(np.int64)
    output["smooth_target_end"] = smooth["target_end"].to_numpy(np.int64)
    output["sparse_target_start"] = sparse["target_start"].to_numpy(np.int64)
    output["sparse_target_end"] = sparse["target_end"].to_numpy(np.int64)
    return output


def _symmetric(values: np.ndarray, *, atol: float = 0.0) -> bool:
    transpose = values.T
    if np.issubdtype(values.dtype, np.floating):
        return bool(
            np.allclose(
                values,
                transpose,
                atol=atol,
                rtol=0.0,
                equal_nan=True,
            )
        )
    return bool(np.array_equal(values, transpose))


class ContactMapOutput:
    """Write one inference run privately and publish it only after validation."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        modes: Sequence[str],
        tile_count: int,
        smooth_context_ids: Sequence[str] = (),
        contact_depth: int = 1_000_000,
        sparse_pair_count: int | None = None,
    ) -> None:
        final_path = Path(output_path).expanduser().resolve()
        selected_modes = tuple(dict.fromkeys(str(mode) for mode in modes))
        if not selected_modes or set(selected_modes) - {"smooth", "sparse"}:
            raise ValueError(
                f"modes must contain smooth and/or sparse: {selected_modes}"
            )
        if tile_count <= 0:
            raise ValueError("tile_count must be positive")
        if contact_depth <= 0:
            raise ValueError("contact_depth must be positive")
        context_ids = tuple(str(value) for value in smooth_context_ids)
        if "smooth" in selected_modes:
            if not context_ids or len(set(context_ids)) != len(context_ids):
                raise ValueError(
                    "Smooth output requires unique nonempty context IDs"
                )
        elif context_ids:
            raise ValueError(
                "smooth_context_ids were supplied without smooth mode"
            )
        if "sparse" in selected_modes:
            if sparse_pair_count is None or int(sparse_pair_count) <= 0:
                raise ValueError(
                    "Sparse output requires a positive sparse_pair_count"
                )
            resolved_sparse_pair_count = int(sparse_pair_count)
        else:
            if sparse_pair_count is not None:
                raise ValueError(
                    "sparse_pair_count was supplied without sparse mode"
                )
            resolved_sparse_pair_count = 0
        if final_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {final_path}"
            )

        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial_name = (
            f".{final_path.name}.partial-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        partial_path = final_path.parent / partial_name
        partial_path.mkdir()
        print(
            f"[output] Initializing private partial run: {partial_path}",
            flush=True,
        )

        self.final_path = final_path
        self.partial_path = partial_path
        self.modes = selected_modes
        self.tile_count = int(tile_count)
        self.smooth_context_ids = context_ids
        self.contact_depth = int(contact_depth)
        self.sparse_pair_count = resolved_sparse_pair_count
        self._finalized = False
        self._tiles_written = False
        self._cell_types_written = False
        self._manifest_written = False
        self._smooth_written = np.zeros(self.tile_count, dtype=bool)
        self._sparse_written = np.zeros(self.tile_count, dtype=bool)
        self._sparse_pair_seen = np.zeros(
            self.sparse_pair_count,
            dtype=bool,
        )
        self._sparse_pair_rows = 0
        self._sparse_pair_writer: pq.ParquetWriter | None = None

        compressor = Blosc(
            cname="zstd",
            clevel=5,
            shuffle=Blosc.BITSHUFFLE,
        )
        store_path = partial_path / "contact_maps.zarr"
        self.store = zarr.open_group(str(store_path), mode="w")
        self.store.attrs.update(
            {
                "schema_version": 1,
                "modes": list(self.modes),
                "tile_count": self.tile_count,
                "contact_depth": self.contact_depth,
                "sparse_pair_count": self.sparse_pair_count,
            }
        )

        self.smooth_group = None
        if "smooth" in self.modes:
            group = self.store.create_group("smooth")
            group.attrs.update(
                {
                    "resolution_bp": 5_000,
                    "map_shape": [SMOOTH_BINS, SMOOTH_BINS],
                    "context_ids": list(self.smooth_context_ids),
                    "value": "signed_log_observed_expected",
                    "symmetric": True,
                    "masked_diagonals": 4,
                }
            )
            group.create_dataset(
                "signed_log_observed_expected",
                shape=(
                    len(self.smooth_context_ids),
                    self.tile_count,
                    SMOOTH_BINS,
                    SMOOTH_BINS,
                ),
                chunks=(1, 1, SMOOTH_BINS, SMOOTH_BINS),
                dtype="f4",
                fill_value=np.nan,
                compressor=compressor,
            )
            valid = group.create_dataset(
                "valid_mask",
                shape=(self.tile_count, SMOOTH_BINS, SMOOTH_BINS),
                chunks=(64, SMOOTH_BINS, SMOOTH_BINS),
                dtype="bool",
                fill_value=False,
                compressor=compressor,
            )
            indices = np.arange(SMOOTH_BINS)
            mask = np.abs(indices[:, None] - indices[None, :]) >= 4
            starts = range(0, self.tile_count, 64)
            for start in tqdm(
                starts,
                total=len(starts),
                desc="Initialize smooth valid mask",
                unit="tile-block",
                dynamic_ncols=True,
            ):
                stop = min(start + 64, self.tile_count)
                valid[start:stop] = np.broadcast_to(
                    mask,
                    (stop - start, SMOOTH_BINS, SMOOTH_BINS),
                )
            self.smooth_group = group

        self.sparse_group = None
        if "sparse" in self.modes:
            group = self.store.create_group("sparse")
            group.attrs.update(
                {
                    "resolution_bp": 10_000,
                    "map_shape": [SPARSE_BINS, SPARSE_BINS],
                    "output_ids": ["pooled"],
                    "model_status": "diagnostic",
                    "rna_dependent": False,
                    "minimum_distance_bp_inclusive": 250_000,
                    "maximum_distance_bp_exclusive": 1_000_000,
                    "distribution": "nb2",
                    "symmetric": True,
                }
            )
            for name in SPARSE_FLOAT_FIELDS:
                group.create_dataset(
                    name,
                    shape=(1, self.tile_count, SPARSE_BINS, SPARSE_BINS),
                    chunks=(1, 1, SPARSE_BINS, SPARSE_BINS),
                    dtype="f4",
                    fill_value=np.nan,
                    compressor=compressor,
                )
            for name in SPARSE_COUNT_FIELDS:
                group.create_dataset(
                    name,
                    shape=(1, self.tile_count, SPARSE_BINS, SPARSE_BINS),
                    chunks=(1, 1, SPARSE_BINS, SPARSE_BINS),
                    dtype="i8",
                    fill_value=-1,
                    compressor=compressor,
                )
            group.create_dataset(
                "pair_id",
                shape=(self.tile_count, SPARSE_BINS, SPARSE_BINS),
                chunks=(1, SPARSE_BINS, SPARSE_BINS),
                dtype="i8",
                fill_value=-1,
                compressor=compressor,
            )
            group.create_dataset(
                "valid_mask",
                shape=(self.tile_count, SPARSE_BINS, SPARSE_BINS),
                chunks=(64, SPARSE_BINS, SPARSE_BINS),
                dtype="bool",
                fill_value=False,
                compressor=compressor,
            )
            self.sparse_group = group

        print("[output] Partial run initialized", flush=True)

    def write_smooth_tile(
        self,
        tile_index: int,
        maps: np.ndarray,
    ) -> None:
        """Validate and write all smooth contexts for one tile."""
        if self.smooth_group is None:
            raise RuntimeError("Smooth mode is not enabled")
        tile_index = int(tile_index)
        if not 0 <= tile_index < self.tile_count:
            raise IndexError(f"Invalid smooth tile index: {tile_index}")
        if self._smooth_written[tile_index]:
            raise RuntimeError(
                f"Smooth tile {tile_index} was already written"
            )

        values = np.asarray(maps, dtype=np.float32)
        expected_shape = (
            len(self.smooth_context_ids),
            SMOOTH_BINS,
            SMOOTH_BINS,
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"Smooth tile shape is {values.shape}; "
                f"expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Smooth tile contains non-finite values")
        if not all(_symmetric(value, atol=1e-5) for value in values):
            raise ValueError("Smooth tile contains an asymmetric map")

        self.smooth_group["signed_log_observed_expected"][
            :, tile_index, :, :
        ] = values
        self._smooth_written[tile_index] = True

    def write_sparse_tile(
        self,
        tile_index: int,
        values: Mapping[str, np.ndarray],
    ) -> None:
        """Validate and write one complete pooled sparse visualization tile."""
        if self.sparse_group is None:
            raise RuntimeError("Sparse mode is not enabled")
        tile_index = int(tile_index)
        if not 0 <= tile_index < self.tile_count:
            raise IndexError(f"Invalid sparse tile index: {tile_index}")
        if self._sparse_written[tile_index]:
            raise RuntimeError(
                f"Sparse tile {tile_index} was already written"
            )

        required = {
            *SPARSE_FLOAT_FIELDS,
            *SPARSE_COUNT_FIELDS,
            "pair_id",
            "valid_mask",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(
                f"Sparse tile lacks fields: {sorted(missing)}"
            )

        valid = np.asarray(values["valid_mask"], dtype=bool)
        pair_id = np.asarray(values["pair_id"], dtype=np.int64)
        expected_shape = (SPARSE_BINS, SPARSE_BINS)
        if valid.shape != expected_shape or pair_id.shape != expected_shape:
            raise ValueError("Sparse mask and pair_id must be 100 x 100")
        if not _symmetric(valid) or not _symmetric(pair_id):
            raise ValueError("Sparse mask or pair IDs are asymmetric")
        if np.any(pair_id[valid] < 0) or np.any(pair_id[~valid] != -1):
            raise ValueError("Sparse pair IDs disagree with valid_mask")

        prepared_float: dict[str, np.ndarray] = {}
        for name in SPARSE_FLOAT_FIELDS:
            array = np.asarray(values[name], dtype=np.float32)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Sparse field {name} has shape {array.shape}"
                )
            if not _symmetric(array, atol=1e-6):
                raise ValueError(f"Sparse field {name} is asymmetric")
            if not np.isfinite(array[valid]).all():
                raise ValueError(
                    f"Sparse field {name} is non-finite on valid pixels"
                )
            if not np.isnan(array[~valid]).all():
                raise ValueError(
                    f"Sparse field {name} must be NaN outside valid pixels"
                )
            prepared_float[name] = array

        for positive_name in (
            "expected_contacts_per_million",
            "expected_count",
            "nb2_dispersion",
        ):
            if np.any(prepared_float[positive_name][valid] <= 0.0):
                raise ValueError(
                    f"Sparse field {positive_name} must be positive"
                )
        probability = prepared_float["tile_band_probability"][valid]
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError(
                "Sparse tile-band probabilities must be in [0, 1]"
            )

        prepared_count: dict[str, np.ndarray] = {}
        for name in SPARSE_COUNT_FIELDS:
            array = np.asarray(values[name], dtype=np.int64)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Sparse count field {name} has shape {array.shape}"
                )
            if not _symmetric(array):
                raise ValueError(f"Sparse count field {name} is asymmetric")
            if np.any(array[valid] < 0) or np.any(array[~valid] != -1):
                raise ValueError(
                    f"Sparse count field {name} disagrees with valid_mask"
                )
            prepared_count[name] = array

        for name, array in prepared_float.items():
            self.sparse_group[name][0, tile_index] = array
        for name, array in prepared_count.items():
            self.sparse_group[name][0, tile_index] = array
        self.sparse_group["pair_id"][tile_index] = pair_id
        self.sparse_group["valid_mask"][tile_index] = valid
        self._sparse_written[tile_index] = True

    def write_tiles(self, tiles: pd.DataFrame) -> Path:
        """Write the combined smooth/sparse tile-coordinate table."""
        if self._tiles_written:
            raise RuntimeError("Tile table was already written")
        if len(tiles) != self.tile_count:
            raise ValueError(
                f"Tile table has {len(tiles)} rows; "
                f"expected {self.tile_count}"
            )
        path = self.partial_path / "tiles.parquet"
        tiles.to_parquet(path, index=False, compression="zstd")
        self._tiles_written = True
        return path

    def write_sparse_pairs(self, pairs: pd.DataFrame) -> None:
        """Append one canonical owner-pair batch to sparse_pairs.parquet."""
        if self.sparse_group is None:
            raise RuntimeError("Sparse mode is not enabled")
        if pairs.empty:
            raise ValueError("Sparse pair batch must not be empty")
        missing = set(SPARSE_PAIR_COLUMNS) - set(pairs.columns)
        if missing:
            raise ValueError(
                f"Sparse pair batch lacks columns: {sorted(missing)}"
            )

        frame = pairs.loc[:, SPARSE_PAIR_COLUMNS].copy()
        pair_ids = frame["pair_id"].to_numpy(np.int64)
        if len(np.unique(pair_ids)) != len(pair_ids):
            raise ValueError("Sparse pair batch contains duplicate pair IDs")
        if np.any(pair_ids < 0) or np.any(
            pair_ids >= self.sparse_pair_count
        ):
            raise ValueError(
                "Sparse pair batch contains an out-of-range pair ID"
            )
        if self._sparse_pair_seen[pair_ids].any():
            raise ValueError(
                "Sparse pair batch overlaps a previously written batch"
            )

        for name in SPARSE_PAIR_FLOAT_FIELDS:
            values = frame[name].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Sparse canonical field {name} is non-finite"
                )
        for name in (
            "expected_contacts_per_million",
            "expected_count",
            "nb2_dispersion",
        ):
            if np.any(frame[name].to_numpy(np.float64) <= 0.0):
                raise ValueError(
                    f"Sparse canonical field {name} must be positive"
                )
        for name in SPARSE_COUNT_FIELDS:
            if np.any(frame[name].to_numpy(np.int64) < 0):
                raise ValueError(
                    f"Sparse canonical count field {name} is negative"
                )

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._sparse_pair_writer is None:
            path = self.partial_path / "sparse_pairs.parquet"
            self._sparse_pair_writer = pq.ParquetWriter(
                path,
                table.schema,
                compression="zstd",
                use_dictionary=[
                    "tile_id",
                    "chrom",
                    "distance_band",
                ],
            )
        elif table.schema != self._sparse_pair_writer.schema:
            raise ValueError(
                "Sparse pair batch schema changed between writes"
            )

        self._sparse_pair_writer.write_table(table)
        self._sparse_pair_seen[pair_ids] = True
        self._sparse_pair_rows += len(frame)

    def write_cell_types(self, cell_types: pd.DataFrame) -> Path:
        """Write input cell-type counts and smooth-context metadata."""
        if self._cell_types_written:
            raise RuntimeError("Cell-type table was already written")
        required = {"cell_type", "n_cells"}
        if missing := required - set(cell_types.columns):
            raise ValueError(
                f"Cell-type table lacks columns: {sorted(missing)}"
            )
        if cell_types["cell_type"].astype(str).duplicated().any():
            raise ValueError("Cell-type table contains duplicate labels")
        if np.any(cell_types["n_cells"].to_numpy(np.int64) <= 0):
            raise ValueError("Cell-type counts must be positive")
        path = self.partial_path / "cell_types.parquet"
        cell_types.to_parquet(path, index=False, compression="zstd")
        self._cell_types_written = True
        return path

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        """Write a checksum-free JSON run manifest."""
        if self._manifest_written:
            raise RuntimeError("Run manifest was already written")
        payload = dict(manifest)
        payload["schema_version"] = 1
        payload["modes"] = list(self.modes)
        payload["tile_count"] = self.tile_count
        payload["contact_depth"] = self.contact_depth
        path = self.partial_path / "run_manifest.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._manifest_written = True
        return path

    def _validate_complete(self) -> None:
        if "smooth" in self.modes and not self._smooth_written.all():
            missing = np.flatnonzero(~self._smooth_written)
            raise RuntimeError(
                f"Smooth output is missing {len(missing)} tiles; "
                f"examples: {missing[:10].tolist()}"
            )
        if "sparse" in self.modes and not self._sparse_written.all():
            missing = np.flatnonzero(~self._sparse_written)
            raise RuntimeError(
                f"Sparse output is missing {len(missing)} tiles; "
                f"examples: {missing[:10].tolist()}"
            )
        if "sparse" in self.modes:
            if self._sparse_pair_writer is None:
                raise RuntimeError("Output lacks sparse_pairs.parquet")
            self._sparse_pair_writer.close()
            self._sparse_pair_writer = None
            if (
                self._sparse_pair_rows != self.sparse_pair_count
                or not self._sparse_pair_seen.all()
            ):
                missing = np.flatnonzero(~self._sparse_pair_seen)
                raise RuntimeError(
                    "Canonical sparse-pair coverage is incomplete: "
                    f"rows={self._sparse_pair_rows}, "
                    f"expected={self.sparse_pair_count}, "
                    f"missing examples={missing[:10].tolist()}"
                )
        if not self._tiles_written:
            raise RuntimeError("Output lacks tiles.parquet")
        if not self._cell_types_written:
            raise RuntimeError("Output lacks cell_types.parquet")
        if not self._manifest_written:
            raise RuntimeError("Output lacks run_manifest.json")

        reopened = zarr.open_group(
            str(self.partial_path / "contact_maps.zarr"),
            mode="r",
        )
        if tuple(reopened.attrs["modes"]) != self.modes:
            raise RuntimeError("Reopened output modes disagree")
        if int(reopened.attrs["tile_count"]) != self.tile_count:
            raise RuntimeError("Reopened output tile count disagrees")
        if "smooth" in self.modes:
            expected = (
                len(self.smooth_context_ids),
                self.tile_count,
                SMOOTH_BINS,
                SMOOTH_BINS,
            )
            if reopened["smooth/signed_log_observed_expected"].shape != expected:
                raise RuntimeError("Reopened smooth array shape disagrees")
        if "sparse" in self.modes:
            expected = (1, self.tile_count, SPARSE_BINS, SPARSE_BINS)
            for name in (*SPARSE_FLOAT_FIELDS, *SPARSE_COUNT_FIELDS):
                if reopened[f"sparse/{name}"].shape != expected:
                    raise RuntimeError(
                        f"Reopened sparse array shape disagrees: {name}"
                    )
            pair_file = pq.ParquetFile(
                self.partial_path / "sparse_pairs.parquet"
            )
            if pair_file.metadata.num_rows != self.sparse_pair_count:
                raise RuntimeError(
                    "Reopened sparse_pairs.parquet row count disagrees"
                )
            if tuple(pair_file.schema_arrow.names) != SPARSE_PAIR_COLUMNS:
                raise RuntimeError(
                    "Reopened sparse_pairs.parquet schema disagrees"
                )

    def finalize(self) -> Path:
        """Validate and atomically publish the completed run directory."""
        if self._finalized:
            raise RuntimeError("Output was already finalized")
        self._validate_complete()
        (self.partial_path / "README.txt").write_text(
            "Hi-Scooby inference output. See run_manifest.json and the "
            "Zarr group attributes for provenance, units, and scientific "
            "status.\n",
            encoding="utf-8",
        )
        self.partial_path.replace(self.final_path)
        self._finalized = True
        print(
            f"[output] Published validated run: {self.final_path}",
            flush=True,
        )
        return self.final_path

    def abort(self) -> None:
        """Remove only this writer's unpublished partial directory."""
        if self._sparse_pair_writer is not None:
            self._sparse_pair_writer.close()
            self._sparse_pair_writer = None
        if not self._finalized and self.partial_path.exists():
            shutil.rmtree(self.partial_path)
            print(
                f"[output] Removed incomplete run: {self.partial_path}",
                flush=True,
            )

    def __enter__(self) -> "ContactMapOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.abort()
        return False
