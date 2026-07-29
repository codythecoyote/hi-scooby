"""Inference adapter for the pooled diagnostic 10 kb contact ranker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import torch
import yaml

from hi_scooby.resources import ResourceRegistry, load_resources
from raw_contact_ranker.exact_rate import center_by_group
from raw_contact_ranker.features import _extract_batch, area_overlap_matrix
from raw_contact_ranker.model import RawContactRanker


EXPECTED_PAIR_SHAPE = (512, 512, 128)
TARGET_BIN_SIZE_BP = 10_000
TARGET_BINS = 100
MINIMUM_DISTANCE_BP = 250_000
MAXIMUM_DISTANCE_BP_EXCLUSIVE = 1_000_000
EXPECTED_PARAMETER_COUNT = 69_758
EXPECTED_PAIR_COUNT = 9_429_237

REQUIRED_PAIR_COLUMNS = {
    "pair_id",
    "bin_i",
    "bin_j",
    "distance_bp",
    "distance_bin",
    "tile_row",
    "exposure",
    "distance_offset",
}


@dataclass(frozen=True)
class SparseCalibrationBand:
    """Frozen NB2 calibration for one genomic-distance band."""

    band_id: str
    minimum_bp: int
    maximum_bp: int
    intercept: float
    dispersion: float

    def contains(self, distance_bp: np.ndarray) -> np.ndarray:
        return (
            (distance_bp >= self.minimum_bp)
            & (distance_bp < self.maximum_bp)
        )


@dataclass(frozen=True)
class SparseTilePrediction:
    """Canonical sparse predictions owned by one 10 kb tile."""

    pair_id: np.ndarray
    row: np.ndarray
    column: np.ndarray
    distance_bp: np.ndarray
    distance_band: np.ndarray
    expected_contacts_per_million: np.ndarray
    expected_count: np.ndarray
    nb2_dispersion: np.ndarray
    predictive_lower: np.ndarray
    predictive_upper: np.ndarray
    simulated_count: np.ndarray
    residual_score: np.ndarray
    owner_tile_band_probability: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical prediction fields as a table."""
        return pd.DataFrame(
            {
                "pair_id": self.pair_id,
                "row": self.row,
                "column": self.column,
                "distance_bp": self.distance_bp,
                "distance_band": self.distance_band,
                "expected_contacts_per_million": (
                    self.expected_contacts_per_million
                ),
                "expected_count": self.expected_count,
                "nb2_dispersion": self.nb2_dispersion,
                "predictive_lower": self.predictive_lower,
                "predictive_upper": self.predictive_upper,
                "simulated_count": self.simulated_count,
                "residual_score": self.residual_score,
                "owner_tile_band_probability": (
                    self.owner_tile_band_probability
                ),
            }
        )


def _load_calibration(path: Path) -> tuple[
    str,
    int,
    float,
    tuple[SparseCalibrationBand, ...],
]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported sparse calibration file: {path}")
    if document.get("distribution") != "nb2":
        raise ValueError("Sparse calibration must use the NB2 distribution")
    if document.get("model_status") != "diagnostic":
        raise ValueError("Sparse v1 checkpoint must remain labeled diagnostic")
    if document.get("formal_gate_passed") is not False:
        raise ValueError(
            "Sparse calibration no longer records the failed formal gate"
        )

    reference_depth = int(document["reference_depth"])
    interval_probability = float(document["interval_probability"])
    if reference_depth <= 0:
        raise ValueError("Sparse calibration reference_depth must be positive")
    if not 0.0 < interval_probability < 1.0:
        raise ValueError(
            "Sparse calibration interval_probability must be in (0, 1)"
        )

    raw_bands = document.get("distance_bands")
    if not isinstance(raw_bands, dict):
        raise ValueError("Sparse calibration lacks distance_bands")

    bands = tuple(
        SparseCalibrationBand(
            band_id=str(band_id),
            minimum_bp=int(values["minimum_bp_inclusive"]),
            maximum_bp=int(values["maximum_bp_exclusive"]),
            intercept=float(values["calibration_intercept"]),
            dispersion=float(values["dispersion"]),
        )
        for band_id, values in raw_bands.items()
    )
    if not bands:
        raise ValueError("Sparse calibration contains no distance bands")
    if bands[0].minimum_bp != MINIMUM_DISTANCE_BP:
        raise ValueError("Sparse calibration has an unexpected minimum distance")
    if bands[-1].maximum_bp != MAXIMUM_DISTANCE_BP_EXCLUSIVE:
        raise ValueError("Sparse calibration has an unexpected maximum distance")
    for previous, following in zip(bands, bands[1:], strict=False):
        if previous.maximum_bp != following.minimum_bp:
            raise ValueError("Sparse calibration distance bands are not contiguous")
    if any(
        not np.isfinite(band.intercept) or band.dispersion <= 0.0
        for band in bands
    ):
        raise ValueError("Sparse calibration contains invalid parameters")

    return (
        str(document["model_status"]),
        reference_depth,
        interval_probability,
        bands,
    )


class SparsePredictor:
    """Loaded pooled 10 kb ranker and its diagnostic NB2 calibration."""

    def __init__(
        self,
        *,
        model: RawContactRanker,
        checkpoint_path: Path,
        calibration_path: Path,
        model_status: str,
        reference_depth: int,
        interval_probability: float,
        bands: tuple[SparseCalibrationBand, ...],
        device: torch.device,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.calibration_path = calibration_path
        self.model_status = model_status
        self.reference_depth = reference_depth
        self.interval_probability = interval_probability
        self.bands = bands
        self.device = device

    @classmethod
    def load(
        cls,
        resources: ResourceRegistry | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> "SparsePredictor":
        resources = resources or load_resources()
        checkpoint_path = resources.resolve("sparse_10kb_checkpoint")
        calibration_path = resources.resolve("sparse_calibration")

        if device is None:
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for the sparse head, but PyTorch cannot "
                "access a CUDA device."
            )

        print(
            f"[sparse] Loading diagnostic 10 kb checkpoint on "
            f"{selected_device}",
            flush=True,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if int(checkpoint.get("schema_version", 0)) < 2:
            raise ValueError("Sparse checkpoint lacks its data contract")
        if checkpoint.get("stage") != "shared_rate":
            raise ValueError(
                f"Unexpected sparse checkpoint stage: "
                f"{checkpoint.get('stage')!r}"
            )
        if int(checkpoint.get("fold", -1)) != 0:
            raise ValueError("Sparse v1 inference requires fold 0")

        model_config = dict(checkpoint["model_config"])
        if model_config.get("feature_set") != "alphagenome":
            raise ValueError(
                "Selected sparse checkpoint is not AlphaGenome-only"
            )

        contract = checkpoint["data_contract"]
        if int(contract.get("pair_count", -1)) != EXPECTED_PAIR_COUNT:
            raise ValueError("Sparse checkpoint has an unexpected pair count")
        resolution = contract.get("resolution", {})
        if int(resolution.get("bin_size_bp", -1)) != TARGET_BIN_SIZE_BP:
            raise ValueError("Sparse checkpoint is not a 10 kb model")
        if (
            int(resolution.get("minimum_distance_bp_inclusive", -1))
            != MINIMUM_DISTANCE_BP
            or int(
                resolution.get("maximum_distance_bp_exclusive", -1)
            )
            != MAXIMUM_DISTANCE_BP_EXCLUSIVE
        ):
            raise ValueError(
                "Sparse checkpoint has an unexpected distance range"
            )

        model = RawContactRanker(**model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"Unexpected sparse parameter count: {parameter_count:,}; "
                f"expected {EXPECTED_PARAMETER_COUNT:,}."
            )
        model.to(selected_device).eval()

        (
            model_status,
            reference_depth,
            interval_probability,
            bands,
        ) = _load_calibration(calibration_path)

        print(
            f"[sparse] Ready: {parameter_count:,} parameters; "
            "pooled RNA-independent output",
            flush=True,
        )
        print(
            "[sparse] Scientific status: diagnostic; the formal topology "
            "gate did not pass",
            flush=True,
        )

        return cls(
            model=model,
            checkpoint_path=checkpoint_path,
            calibration_path=calibration_path,
            model_status=model_status,
            reference_depth=reference_depth,
            interval_probability=interval_probability,
            bands=bands,
            device=selected_device,
        )

    def predict_owner_tile(
        self,
        pair_embedding: np.ndarray,
        pairs: pd.DataFrame,
        *,
        tile_row: int,
        input_start: int,
        target_start: int,
        contact_depth: int = 1_000_000,
        seed: int = 0,
    ) -> SparseTilePrediction:
        """Predict canonical pairs whose unique owner is one 10 kb tile."""
        pair_array = np.asarray(pair_embedding)
        if pair_array.shape != EXPECTED_PAIR_SHAPE:
            raise ValueError(
                f"pair_embedding must have shape {EXPECTED_PAIR_SHAPE}; "
                f"found {pair_array.shape}"
            )
        if not np.isfinite(pair_array).all():
            raise ValueError("pair_embedding contains non-finite values")

        missing = REQUIRED_PAIR_COLUMNS - set(pairs.columns)
        if missing:
            raise ValueError(
                f"Canonical pair table lacks columns: {sorted(missing)}"
            )
        if pairs.empty:
            raise ValueError(f"Tile row {tile_row} owns no canonical pairs")
        if not pairs["tile_row"].astype("int64").eq(int(tile_row)).all():
            raise ValueError("Canonical pairs do not share the requested owner")
        if pairs["pair_id"].duplicated().any():
            raise ValueError("Canonical pair IDs are duplicated within a tile")
        if contact_depth <= 0:
            raise ValueError("contact_depth must be positive")

        bin_i = pairs["bin_i"].to_numpy(np.int64)
        bin_j = pairs["bin_j"].to_numpy(np.int64)
        left = (bin_i - int(target_start)) // TARGET_BIN_SIZE_BP
        right = (bin_j - int(target_start)) // TARGET_BIN_SIZE_BP
        if (
            np.any((bin_i - int(target_start)) % TARGET_BIN_SIZE_BP)
            or np.any((bin_j - int(target_start)) % TARGET_BIN_SIZE_BP)
            or np.any(left < 0)
            or np.any(right >= TARGET_BINS)
            or np.any(left >= right)
        ):
            raise ValueError(
                "One or more canonical pairs lie outside the 10 kb tile grid"
            )

        distance_bp = pairs["distance_bp"].to_numpy(np.int64)
        if not np.array_equal(
            distance_bp,
            (right - left) * TARGET_BIN_SIZE_BP,
        ):
            raise ValueError("Canonical pair distances disagree with coordinates")
        if (
            np.any(distance_bp < MINIMUM_DISTANCE_BP)
            or np.any(distance_bp >= MAXIMUM_DISTANCE_BP_EXCLUSIVE)
        ):
            raise ValueError("Canonical pair distance is outside model support")

        weights = area_overlap_matrix(
            int(input_start),
            int(target_start),
            target_bin_size=TARGET_BIN_SIZE_BP,
            target_bins=TARGET_BINS,
        )
        pair_feature, anchor_i, anchor_j = _extract_batch(
            pair_array,
            weights,
            left,
            right,
        )

        count = len(pairs)
        zeros_annotations = np.zeros((count, 16), dtype=np.float32)
        zeros_pair_annotations = np.zeros((count, 5), dtype=np.float32)
        zeros_technical = np.zeros((count, 5), dtype=np.float32)
        exposure = pairs["exposure"].to_numpy(np.float64)
        if not np.isfinite(exposure).all() or np.any(exposure <= 0.0):
            raise ValueError("Canonical exposure values must be finite and positive")

        model_inputs = {
            "pair_embedding": torch.from_numpy(pair_feature).to(self.device),
            "anchor_i": torch.from_numpy(anchor_i).to(self.device),
            "anchor_j": torch.from_numpy(anchor_j).to(self.device),
            "annotations": torch.from_numpy(
                zeros_annotations
            ).to(self.device),
            "pair_annotations": torch.from_numpy(
                zeros_pair_annotations
            ).to(self.device),
            "technical": torch.from_numpy(zeros_technical).to(self.device),
            "fixed_exposure": torch.from_numpy(
                np.log(exposure).astype(np.float32)
            ).to(self.device),
            "fixed_distance_offset": torch.from_numpy(
                pairs["distance_offset"].to_numpy(np.float32)
            ).to(self.device),
            "distance_bp": torch.from_numpy(
                distance_bp.astype(np.float32)
            ).to(self.device),
        }

        with torch.inference_mode():
            output = self.model(**model_inputs)
            learned_score = (
                output["residual_score"] + output["technical_offset"]
            )
            distance_group = torch.from_numpy(
                pairs["distance_bin"].to_numpy(np.int64)
            ).to(self.device)
            residual_score = center_by_group(
                learned_score,
                distance_group,
            ).cpu().numpy().astype(np.float64, copy=False)

        expected_per_million = np.empty(count, dtype=np.float64)
        dispersion = np.empty(count, dtype=np.float64)
        distance_band = np.empty(count, dtype=object)
        covered = np.zeros(count, dtype=bool)
        log_exposure = np.log(exposure)
        distance_offset = pairs["distance_offset"].to_numpy(np.float64)

        for band in self.bands:
            selected = band.contains(distance_bp)
            if np.any(covered & selected):
                raise RuntimeError("Sparse calibration bands overlap")
            covered |= selected
            distance_band[selected] = band.band_id
            log_rate = (
                log_exposure[selected]
                + distance_offset[selected]
                + residual_score[selected]
                + band.intercept
            )
            expected_per_million[selected] = np.exp(
                np.clip(log_rate, -40.0, 30.0)
            )
            dispersion[selected] = band.dispersion

        if not covered.all():
            raise RuntimeError("Sparse calibration does not cover every pair")
        if (
            not np.isfinite(expected_per_million).all()
            or np.any(expected_per_million <= 0.0)
            or not np.isfinite(dispersion).all()
            or np.any(dispersion <= 0.0)
        ):
            raise RuntimeError("Sparse calibration produced invalid parameters")

        expected_count = (
            expected_per_million
            * float(contact_depth)
            / float(self.reference_depth)
        )
        size = 1.0 / dispersion
        success_probability = size / (size + expected_count)
        tail_probability = (1.0 - self.interval_probability) / 2.0
        predictive_lower = nbinom.ppf(
            tail_probability,
            size,
            success_probability,
        ).astype(np.int64)
        predictive_upper = nbinom.ppf(
            1.0 - tail_probability,
            size,
            success_probability,
        ).astype(np.int64)

        random = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(tile_row)])
        )
        simulated_count = random.negative_binomial(
            size,
            success_probability,
        ).astype(np.int64)

        owner_probability = np.empty(count, dtype=np.float64)
        for band in self.bands:
            selected = distance_band == band.band_id
            total = float(expected_per_million[selected].sum())
            if total <= 0.0:
                raise RuntimeError(
                    f"Tile {tile_row} has no positive rate in band "
                    f"{band.band_id}"
                )
            owner_probability[selected] = (
                expected_per_million[selected] / total
            )

        return SparseTilePrediction(
            pair_id=pairs["pair_id"].to_numpy(np.int64),
            row=left.astype(np.int16),
            column=right.astype(np.int16),
            distance_bp=distance_bp.astype(np.int32),
            distance_band=distance_band,
            expected_contacts_per_million=(
                expected_per_million.astype(np.float32)
            ),
            expected_count=expected_count.astype(np.float32),
            nb2_dispersion=dispersion.astype(np.float32),
            predictive_lower=predictive_lower,
            predictive_upper=predictive_upper,
            simulated_count=simulated_count,
            residual_score=residual_score.astype(np.float32),
            owner_tile_band_probability=owner_probability.astype(np.float32),
        )
