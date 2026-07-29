from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import zarr


ANNOTATION_COLUMNS = (
    "bin_i_tss", "bin_j_tss",
    "bin_i_ccre", "bin_j_ccre",
    "bin_i_ctcf_score", "bin_j_ctcf_score",
    "bin_i_ctcf_orientation", "bin_j_ctcf_orientation",
    "bin_i_gc_fraction", "bin_j_gc_fraction",
    "bin_i_mappability", "bin_j_mappability",
    "bin_i_valid_sequence_fraction", "bin_j_valid_sequence_fraction",
    "bin_i_assay_site_density", "bin_j_assay_site_density",
)
TECHNICAL_COLUMNS = (
    "exposure_clipped",
    "bin_i_mappability_missing",
    "bin_j_mappability_missing",
    "bin_i_valid_sequence_fraction_missing",
    "bin_j_valid_sequence_fraction_missing",
)
FEATURE_ARRAYS = (
    "pair_embedding",
    "anchor_i_embedding",
    "anchor_j_embedding",
)
ANCHOR_LABEL_COLUMNS = (
    "anchor_has_promoter",
    "anchor_has_ccre",
    "anchor_has_ctcf",
    "anchor_ctcf_convergent",
    "anchor_ctcf_divergent",
)
PAIR_ANNOTATION_COLUMNS = ANCHOR_LABEL_COLUMNS


def anchor_stratum_roles(config: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    evaluation = config["evaluation"]
    required = tuple(evaluation["required_anchor_strata"])
    descriptive = tuple(evaluation.get("descriptive_anchor_strata", ()))
    required_set = set(required)
    descriptive_set = set(descriptive)
    known = set(ANCHOR_LABEL_COLUMNS)
    if (
        not required
        or len(required_set) != len(required)
        or len(descriptive_set) != len(descriptive)
        or required_set & descriptive_set
        or required_set | descriptive_set != known
    ):
        raise ValueError(
            "Required and descriptive anchor strata must be unique, disjoint, "
            "and partition ANCHOR_LABEL_COLUMNS"
        )
    return required, descriptive


class PairData:
    """Random-access bridge across the immutable pair Parquet and feature Zarr."""

    def __init__(
        self,
        pairs_path: Path,
        feature_path: Path,
        *,
        preload_features: bool = False,
        progress: bool = True,
        preload_batch_rows: int = 1_048_576,
    ) -> None:
        columns = [
            "pair_id", "chrom", "bin_i", "bin_j", "distance_bp", "distance_bin",
            "distance_band", "tile_row", "split", "anchor_class", "exposure",
            "distance_offset",
            *ANCHOR_LABEL_COLUMNS,
            *ANNOTATION_COLUMNS, *TECHNICAL_COLUMNS,
        ]
        self.pairs = pd.read_parquet(pairs_path, columns=columns).sort_values(
            "pair_id", kind="stable"
        ).reset_index(drop=True)
        expected = np.arange(len(self.pairs), dtype=np.int64)
        if not np.array_equal(self.pairs["pair_id"].to_numpy(np.int64), expected):
            raise ValueError("pair_id must be contiguous and row-aligned")
        self.features = zarr.open_group(str(feature_path), mode="r")
        if self.features["pair_embedding"].shape[0] != len(self.pairs):
            raise ValueError("Feature and pair row counts differ")
        self._feature_cache: dict[str, np.ndarray] | None = None
        self._model_cache: dict[str, np.ndarray] | None = None
        if preload_features:
            self.preload(progress=progress, batch_rows=preload_batch_rows)

    @staticmethod
    def _finite_impute(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, np.float32)
        return np.where(np.isfinite(array), array, 0.0).astype(np.float32)

    def preload(self, *, progress: bool = True, batch_rows: int = 1_048_576) -> None:
        """Load immutable feature/covariate arrays into RAM for random training reads."""
        if self._feature_cache is not None:
            return
        cache: dict[str, np.ndarray] = {}
        for name in FEATURE_ARRAYS:
            source = self.features[name]
            rows_per_chunk = int(source.chunks[0])
            rows_per_batch = max(rows_per_chunk, int(batch_rows))
            rows_per_batch -= rows_per_batch % rows_per_chunk
            output = np.empty(source.shape, dtype=source.dtype)
            bar = tqdm(
                total=len(self.pairs),
                desc=f"Preload {name}",
                unit="pair",
                unit_scale=True,
                disable=not progress,
            )
            for start in range(0, len(self.pairs), rows_per_batch):
                stop = min(start + rows_per_batch, len(self.pairs))
                output[start:stop] = source[start:stop]
                bar.update(stop - start)
            bar.close()
            cache[name] = output
        self._feature_cache = cache

        def numeric_matrix(columns: tuple[str, ...], description: str) -> np.ndarray:
            output = np.empty((len(self.pairs), len(columns)), np.float32)
            for index, column in enumerate(
                tqdm(columns, desc=description, unit="column", disable=not progress)
            ):
                values = self.pairs[column].to_numpy(np.float32)
                output[:, index] = np.where(np.isfinite(values), values, 0.0)
            return output

        self._model_cache = {
            "annotations": numeric_matrix(
                ANNOTATION_COLUMNS, "Preload annotation covariates"
            ),
            "technical": numeric_matrix(
                TECHNICAL_COLUMNS, "Preload technical covariates"
            ),
            "pair_annotations": numeric_matrix(
                PAIR_ANNOTATION_COLUMNS, "Preload pair annotation covariates"
            ),
            "fixed_exposure": np.log(
                np.clip(self.pairs["exposure"].to_numpy(np.float32), 1e-12, None)
            ).astype(np.float32),
            "distance_bp": self.pairs["distance_bp"].to_numpy(np.float32),
            "fixed_distance_offset": self.pairs["distance_offset"].to_numpy(np.float32),
        }

    @property
    def preloaded_bytes(self) -> int:
        arrays = []
        if self._feature_cache is not None:
            arrays.extend(self._feature_cache.values())
        if self._model_cache is not None:
            arrays.extend(self._model_cache.values())
        return int(sum(array.nbytes for array in arrays))

    def tensors(
        self,
        pair_ids: np.ndarray | list[int],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        ids = np.asarray(pair_ids, np.int64)
        if self._model_cache is None:
            rows = self.pairs.iloc[ids]
            annotation = self._finite_impute(
                rows[list(ANNOTATION_COLUMNS)].to_numpy()
            )
            technical = self._finite_impute(
                rows[list(TECHNICAL_COLUMNS)].to_numpy()
            )
            pair_annotations = self._finite_impute(
                rows[list(PAIR_ANNOTATION_COLUMNS)].to_numpy()
            )
            fixed_exposure = np.log(
                np.clip(rows["exposure"].to_numpy(np.float32), 1e-12, None)
            )
            distance_bp = rows["distance_bp"].to_numpy(np.float32)
            fixed_distance_offset = rows["distance_offset"].to_numpy(np.float32)
        else:
            annotation = self._model_cache["annotations"][ids]
            technical = self._model_cache["technical"][ids]
            pair_annotations = self._model_cache["pair_annotations"][ids]
            fixed_exposure = self._model_cache["fixed_exposure"][ids]
            distance_bp = self._model_cache["distance_bp"][ids]
            fixed_distance_offset = self._model_cache["fixed_distance_offset"][ids]

        def feature(name: str) -> np.ndarray:
            source = (
                self._feature_cache[name]
                if self._feature_cache is not None
                else self.features[name]
            )
            values = source[ids] if self._feature_cache is not None else source.oindex[ids, :]
            return np.asarray(values, np.float32)

        return {
            "pair_embedding": torch.as_tensor(feature("pair_embedding"), device=device),
            "anchor_i": torch.as_tensor(feature("anchor_i_embedding"), device=device),
            "anchor_j": torch.as_tensor(feature("anchor_j_embedding"), device=device),
            "annotations": torch.as_tensor(annotation, device=device),
            "technical": torch.as_tensor(technical, device=device),
            "pair_annotations": torch.as_tensor(
                pair_annotations, device=device
            ),
            "fixed_exposure": torch.as_tensor(fixed_exposure, device=device),
            "distance_bp": torch.as_tensor(distance_bp, device=device),
            "fixed_distance_offset": torch.as_tensor(
                fixed_distance_offset, device=device
            ),
        }

    def model_inputs(self, pair_ids, device):
        return self.tensors(pair_ids, device)


def model_from_checkpoint(checkpoint: dict[str, Any], device: str | torch.device):
    from .model import RawContactRanker

    model = RawContactRanker(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device)
