from __future__ import annotations

import operator

import torch
import torch.nn as nn


NATIVE_BIN_SIZE_BP = 2_048
TARGET_BIN_SIZE_BP = 5_000
NATIVE_BINS = 512
TARGET_BINS = 200
CELL_EMBEDDING_DIM = 14
PAIR_CHANNELS = 128
MAX_OVERLAPS_PER_TARGET_BIN = 4
HYPERNETWORK_HIDDEN_DIM_1 = 128
HYPERNETWORK_HIDDEN_DIM_2 = 256
HYPERNETWORK_DROPOUT = 0.2
SPATIAL_CHANNELS = 32
SPATIAL_DILATIONS = (1, 2, 4, 8, 16)
SPATIAL_GROUPS = 8
FILM_HIDDEN_DIM = 64


def _require_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")

    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer; found {value!r}") from error


def _require_positive_integer(name: str, value: int) -> int:
    value = _require_integer(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero; found {value}")
    return value


def build_area_overlap_matrix(
    input_start: int,
    target_start: int,
    *,
    native_bin_size_bp: int = NATIVE_BIN_SIZE_BP,
    target_bin_size_bp: int = TARGET_BIN_SIZE_BP,
    native_bins: int = NATIVE_BINS,
    target_bins: int = TARGET_BINS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct area-overlap weights from native to target genomic bins.

    Native bin ``k`` covers::

        [input_start + k * native_bin_size_bp,
         input_start + (k + 1) * native_bin_size_bp)

    Target bin ``i`` covers::

        [target_start + i * target_bin_size_bp,
         target_start + (i + 1) * target_bin_size_bp)

    Each output entry is the fraction of target bin ``i`` covered by
    native bin ``k``. The returned float32 tensor has shape
    ``[target_bins, native_bins]`` and every row sums to one.
    """
    input_start = _require_integer("input_start", input_start)
    target_start = _require_integer("target_start", target_start)
    native_bin_size_bp = _require_positive_integer(
        "native_bin_size_bp",
        native_bin_size_bp,
    )
    target_bin_size_bp = _require_positive_integer(
        "target_bin_size_bp",
        target_bin_size_bp,
    )
    native_bins = _require_positive_integer("native_bins", native_bins)
    target_bins = _require_positive_integer("target_bins", target_bins)

    if input_start < 0:
        raise ValueError(
            f"input_start must be nonnegative; found {input_start}"
        )
    if target_start < 0:
        raise ValueError(
            f"target_start must be nonnegative; found {target_start}"
        )

    input_end = input_start + native_bins * native_bin_size_bp
    target_end = target_start + target_bins * target_bin_size_bp

    if target_start < input_start or target_end > input_end:
        raise ValueError(
            "Target interval must be fully contained in the native input: "
            f"input=[{input_start}, {input_end}), "
            f"target=[{target_start}, {target_end})"
        )

    # Work in coordinates relative to input_start. Keeping the coordinate
    # arithmetic in int64 avoids float32 precision loss at large genomic
    # coordinates.
    target_offset = target_start - input_start

    native_starts = (
        torch.arange(native_bins, dtype=torch.int64, device=device)
        * native_bin_size_bp
    )
    native_ends = native_starts + native_bin_size_bp

    target_starts = (
        target_offset
        + torch.arange(target_bins, dtype=torch.int64, device=device)
        * target_bin_size_bp
    )
    target_ends = target_starts + target_bin_size_bp

    overlap_starts = torch.maximum(
        target_starts[:, None],
        native_starts[None, :],
    )
    overlap_ends = torch.minimum(
        target_ends[:, None],
        native_ends[None, :],
    )
    overlap_bp = torch.clamp(
        overlap_ends - overlap_starts,
        min=0,
    )

    covered_bp = overlap_bp.sum(dim=1)
    fully_covered = covered_bp == target_bin_size_bp

    if not bool(torch.all(fully_covered).item()):
        bad_rows = (
            torch.nonzero(~fully_covered)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        raise ValueError(
            "Target bins are not fully covered by native bins; "
            f"bad target rows include {bad_rows[:10]}"
        )

    weights = overlap_bp.to(dtype=torch.float32) / float(target_bin_size_bp)

    row_sums = weights.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
        rtol=0.0,
    ):
        max_error = float((row_sums - 1.0).abs().max().item())
        raise RuntimeError(
            "Overlap-weight rows do not sum to one; "
            f"maximum error is {max_error}"
        )

    return weights

class CellStateHypernetwork(nn.Module):
    """Generate context-specific residual pair-channel filters."""

    def __init__(
        self,
        *,
        cell_embedding_dim: int = CELL_EMBEDDING_DIM,
        pair_channels: int = PAIR_CHANNELS,
        hidden_dim_1: int = HYPERNETWORK_HIDDEN_DIM_1,
        hidden_dim_2: int = HYPERNETWORK_HIDDEN_DIM_2,
        dropout: float = HYPERNETWORK_DROPOUT,
    ) -> None:
        super().__init__()

        self.cell_embedding_dim = _require_positive_integer(
            "cell_embedding_dim",
            cell_embedding_dim,
        )
        self.pair_channels = _require_positive_integer(
            "pair_channels",
            pair_channels,
        )
        hidden_dim_1 = _require_positive_integer(
            "hidden_dim_1",
            hidden_dim_1,
        )
        hidden_dim_2 = _require_positive_integer(
            "hidden_dim_2",
            hidden_dim_2,
        )

        if not isinstance(dropout, (float, int)) or isinstance(dropout, bool):
            raise TypeError(
                f"dropout must be a number; found {dropout!r}"
            )
        dropout = float(dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1; found {dropout}"
            )

        self.cell_state_to_filter = nn.Sequential(
            nn.Linear(self.cell_embedding_dim, hidden_dim_1),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, self.pair_channels + 1),
        )

        self.reset_parameters()

    @property
    def output_layer(self) -> nn.Linear:
        layer = self.cell_state_to_filter[-1]
        if not isinstance(layer, nn.Linear):
            raise RuntimeError("Hypernetwork output layer is not linear")
        return layer

    def reset_parameters(self) -> None:
        """Apply Xavier initialization and a zero residual output layer."""
        for module in self.cell_state_to_filter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Every context initially produces zero residual weights and bias.
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        cell_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual filter weights and biases.

        Supported input shapes are ``[C, 14]`` and ``[B, C, 14]``.
        Returned shapes are ``[..., 128]`` and ``[..., 1]``.
        """
        if not isinstance(cell_embedding, torch.Tensor):
            raise TypeError(
                "cell_embedding must be a torch.Tensor; "
                f"found {type(cell_embedding).__name__}"
            )
        if cell_embedding.ndim not in (2, 3):
            raise ValueError(
                "cell_embedding must have shape [C, D] or [B, C, D]; "
                f"found {tuple(cell_embedding.shape)}"
            )
        if cell_embedding.shape[-1] != self.cell_embedding_dim:
            raise ValueError(
                f"Expected cell embedding dimension "
                f"{self.cell_embedding_dim}; "
                f"found {cell_embedding.shape[-1]}"
            )
        if not torch.is_floating_point(cell_embedding):
            raise TypeError(
                "cell_embedding must use a floating-point dtype; "
                f"found {cell_embedding.dtype}"
            )
        if not bool(torch.isfinite(cell_embedding).all().item()):
            raise ValueError(
                "cell_embedding contains non-finite values"
            )

        dynamic_parameters = self.cell_state_to_filter(cell_embedding)

        delta_weight = dynamic_parameters[..., : self.pair_channels]
        delta_bias = dynamic_parameters[..., self.pair_channels :]

        expected_weight_shape = (
            *cell_embedding.shape[:-1],
            self.pair_channels,
        )
        expected_bias_shape = (*cell_embedding.shape[:-1], 1)

        if delta_weight.shape != expected_weight_shape:
            raise RuntimeError(
                f"Unexpected residual-weight shape "
                f"{tuple(delta_weight.shape)}; "
                f"expected {expected_weight_shape}"
            )
        if delta_bias.shape != expected_bias_shape:
            raise RuntimeError(
                f"Unexpected residual-bias shape "
                f"{tuple(delta_bias.shape)}; "
                f"expected {expected_bias_shape}"
            )

        return delta_weight, delta_bias


class SpatialFiLMResidualBlock(nn.Module):
    """Apply one context-conditioned, dilated spatial residual update."""

    def __init__(
        self,
        *,
        channels: int,
        dilation: int,
        groups: int,
    ) -> None:
        super().__init__()

        self.channels = _require_positive_integer("channels", channels)
        dilation = _require_positive_integer("dilation", dilation)
        groups = _require_positive_integer("groups", groups)
        if self.channels % groups != 0:
            raise ValueError(
                f"channels ({self.channels}) must be divisible by "
                f"groups ({groups})"
            )

        self.normalization = nn.GroupNorm(groups, self.channels)
        self.activation = nn.GELU()
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=self.channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
        )

    def forward(
        self,
        features: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Return a FiLM-modulated residual update."""
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                "features must have shape [B, K, H, W] with "
                f"K={self.channels}; found {tuple(features.shape)}"
            )

        expected_condition_shape = (
            features.shape[0],
            self.channels,
        )
        if scale.shape != expected_condition_shape:
            raise ValueError(
                f"scale must have shape {expected_condition_shape}; "
                f"found {tuple(scale.shape)}"
            )
        if shift.shape != expected_condition_shape:
            raise ValueError(
                f"shift must have shape {expected_condition_shape}; "
                f"found {tuple(shift.shape)}"
            )

        normalized = self.normalization(features)
        conditioned = (
            normalized * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        update = self.pointwise(
            self.depthwise(self.activation(conditioned))
        )
        return features + update


class Phase1ScooHiC(nn.Module):
    """Decode frozen AlphaGenome pair embeddings into Hi-C contact maps."""

    def __init__(
        self,
        *,
        cell_embedding_dim: int = CELL_EMBEDDING_DIM,
        pair_channels: int = PAIR_CHANNELS,
        spatial_channels: int = SPATIAL_CHANNELS,
        spatial_dilations: tuple[int, ...] = SPATIAL_DILATIONS,
    ) -> None:
        super().__init__()

        self.cell_embedding_dim = _require_positive_integer(
            "cell_embedding_dim",
            cell_embedding_dim,
        )
        self.pair_channels = _require_positive_integer(
            "pair_channels",
            pair_channels,
        )
        self.spatial_channels = _require_positive_integer(
            "spatial_channels",
            spatial_channels,
        )
        if not spatial_dilations:
            raise ValueError("spatial_dilations must not be empty")
        self.spatial_dilations = tuple(
            _require_positive_integer("spatial dilation", dilation)
            for dilation in spatial_dilations
        )
        if self.spatial_channels % SPATIAL_GROUPS != 0:
            raise ValueError(
                f"spatial_channels ({self.spatial_channels}) must be "
                f"divisible by {SPATIAL_GROUPS}"
            )

        self.w_base = nn.Parameter(torch.empty(self.pair_channels))
        self.b_base = nn.Parameter(torch.empty(1))

        self.hypernetwork = CellStateHypernetwork(
            cell_embedding_dim=self.cell_embedding_dim,
            pair_channels=self.pair_channels,
        )

        self.pair_projection = nn.Linear(
            self.pair_channels,
            self.spatial_channels,
            bias=False,
        )
        self.spatial_input = nn.Conv2d(
            self.spatial_channels + 1,
            self.spatial_channels,
            kernel_size=1,
        )
        self.spatial_blocks = nn.ModuleList(
            SpatialFiLMResidualBlock(
                channels=self.spatial_channels,
                dilation=dilation,
                groups=SPATIAL_GROUPS,
            )
            for dilation in self.spatial_dilations
        )
        self.context_to_film = nn.Sequential(
            nn.LayerNorm(self.cell_embedding_dim),
            nn.Linear(self.cell_embedding_dim, FILM_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(
                FILM_HIDDEN_DIM,
                len(self.spatial_dilations)
                * 2
                * self.spatial_channels,
            ),
        )
        self.spatial_output = nn.Conv2d(
            self.spatial_channels,
            1,
            kernel_size=1,
        )

        target_positions = torch.arange(TARGET_BINS, dtype=torch.float32)
        distance_channel = (
            target_positions[:, None] - target_positions[None, :]
        ).abs() / float(TARGET_BINS - 1)
        self.register_buffer(
            "distance_channel",
            distance_channel.view(1, 1, TARGET_BINS, TARGET_BINS),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the linear base and zero-initialize spatial residuals."""
        nn.init.xavier_normal_(self.w_base.unsqueeze(0))
        nn.init.zeros_(self.b_base)
        nn.init.xavier_normal_(self.pair_projection.weight)
        nn.init.xavier_normal_(self.spatial_input.weight)
        nn.init.zeros_(self.spatial_input.bias)

        for block in self.spatial_blocks:
            nn.init.ones_(block.normalization.weight)
            nn.init.zeros_(block.normalization.bias)
            nn.init.kaiming_normal_(
                block.depthwise.weight,
                nonlinearity="linear",
            )
            nn.init.xavier_normal_(block.pointwise.weight)
            nn.init.zeros_(block.pointwise.bias)

        nn.init.ones_(self.context_to_film[0].weight)
        nn.init.zeros_(self.context_to_film[0].bias)
        nn.init.xavier_normal_(self.context_to_film[1].weight)
        nn.init.zeros_(self.context_to_film[1].bias)
        nn.init.zeros_(self.context_to_film[-1].weight)
        nn.init.zeros_(self.context_to_film[-1].bias)
        nn.init.zeros_(self.spatial_output.weight)
        nn.init.zeros_(self.spatial_output.bias)

    def conditioned_parameters(
        self,
        context_embedding: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full context-conditioned filters for each batch item.

        ``context_embedding`` may be shared across the batch with shape
        ``[C, 14]`` or supplied per tile with shape ``[B, C, 14]``.

        Returns:
            weights: ``[B, C, 128]``
            biases: ``[B, C, 1]``
        """
        batch_size = _require_positive_integer("batch_size", batch_size)

        delta_weight, delta_bias = self.hypernetwork(context_embedding)

        if context_embedding.ndim == 2:
            if context_embedding.shape[0] == 0:
                raise ValueError(
                    "context_embedding must contain at least one context"
                )

            delta_weight = delta_weight.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            delta_bias = delta_bias.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        else:
            if context_embedding.shape[0] != batch_size:
                raise ValueError(
                    "Batched context embeddings must match the pair-embedding "
                    f"batch size {batch_size}; found "
                    f"{context_embedding.shape[0]}"
                )
            if context_embedding.shape[1] == 0:
                raise ValueError(
                    "context_embedding must contain at least one context"
                )

        weights = (
            self.w_base.view(1, 1, self.pair_channels)
            + delta_weight
        )
        biases = self.b_base.view(1, 1, 1) + delta_bias

        return weights, biases

    def decode_native(
        self,
        pair_embedding: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
    ) -> torch.Tensor:
        """Decode pair channels into native-resolution contact maps.

        Args:
            pair_embedding: ``[B, 512, 512, 128]``
            weights: ``[B, C, 128]``
            biases: ``[B, C, 1]``

        Returns:
            Float32 native predictions with shape ``[B, C, 512, 512]``.
        """
        if not isinstance(pair_embedding, torch.Tensor):
            raise TypeError(
                "pair_embedding must be a torch.Tensor; "
                f"found {type(pair_embedding).__name__}"
            )
        if pair_embedding.ndim != 4:
            raise ValueError(
                "pair_embedding must have shape [B, N, N, D]; "
                f"found {tuple(pair_embedding.shape)}"
            )
        if not torch.is_floating_point(pair_embedding):
            raise TypeError(
                "pair_embedding must use a floating-point dtype; "
                f"found {pair_embedding.dtype}"
            )

        batch_size, rows, columns, channels = pair_embedding.shape

        if batch_size == 0:
            raise ValueError(
                "pair_embedding must contain at least one batch item"
            )
        if rows != NATIVE_BINS or columns != NATIVE_BINS:
            raise ValueError(
                f"Expected pair grid [{NATIVE_BINS}, {NATIVE_BINS}]; "
                f"found [{rows}, {columns}]"
            )
        if channels != self.pair_channels:
            raise ValueError(
                f"Expected {self.pair_channels} pair channels; "
                f"found {channels}"
            )

        if weights.ndim != 3:
            raise ValueError(
                "weights must have shape [B, C, D]; "
                f"found {tuple(weights.shape)}"
            )
        if weights.shape[0] != batch_size:
            raise ValueError(
                "weights batch dimension does not match pair_embedding: "
                f"{weights.shape[0]} versus {batch_size}"
            )
        if weights.shape[-1] != self.pair_channels:
            raise ValueError(
                f"Expected weights to have {self.pair_channels} channels; "
                f"found {weights.shape[-1]}"
            )

        expected_bias_shape = (*weights.shape[:2], 1)
        if biases.shape != expected_bias_shape:
            raise ValueError(
                f"biases must have shape {expected_bias_shape}; "
                f"found {tuple(biases.shape)}"
            )

        if (
            pair_embedding.device != weights.device
            or pair_embedding.device != biases.device
        ):
            raise ValueError(
                "pair_embedding, weights, and biases must be on the same device"
            )

        # Explicitly disable mixed-precision autocasting for the projection.
        with torch.autocast(
            device_type=pair_embedding.device.type,
            enabled=False,
        ):
            # Projection, coordinate resampling, and the final map
            # symmetrization are linear. Symmetrizing all 128 input
            # channels here is therefore redundant and much more expensive
            # than the existing final output symmetrization.
            native_prediction = torch.einsum(
                "bijd,bcd->bcij",
                pair_embedding.float(),
                weights.float(),
            )
            native_prediction = (
                native_prediction + biases.float().unsqueeze(-1)
            )

        return native_prediction

    def resample_maps(
        self,
        native_prediction: torch.Tensor,
        resample_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Resample native maps onto each tile's target coordinate grid.

        Args:
            native_prediction: ``[B, C, 512, 512]``
            resample_weights: ``[B, 200, 512]``

        Returns:
            Float32 predictions with shape ``[B, C, 200, 200]``.
        """
        if native_prediction.ndim != 4:
            raise ValueError(
                "native_prediction must have shape [B, C, N, N]; "
                f"found {tuple(native_prediction.shape)}"
            )
        if resample_weights.ndim != 3:
            raise ValueError(
                "resample_weights must have shape [B, T, N]; "
                f"found {tuple(resample_weights.shape)}"
            )
        if not torch.is_floating_point(resample_weights):
            raise TypeError(
                "resample_weights must use a floating-point dtype; "
                f"found {resample_weights.dtype}"
            )

        batch_size, _, rows, columns = native_prediction.shape

        if rows != NATIVE_BINS or columns != NATIVE_BINS:
            raise ValueError(
                f"Expected native predictions on a "
                f"[{NATIVE_BINS}, {NATIVE_BINS}] grid; "
                f"found [{rows}, {columns}]"
            )

        expected_resample_shape = (
            batch_size,
            TARGET_BINS,
            NATIVE_BINS,
        )
        if resample_weights.shape != expected_resample_shape:
            raise ValueError(
                f"resample_weights must have shape "
                f"{expected_resample_shape}; "
                f"found {tuple(resample_weights.shape)}"
            )

        if native_prediction.device != resample_weights.device:
            raise ValueError(
                "native_prediction and resample_weights must be on "
                "the same device"
            )

        # Every 5 kb target bin overlaps at most four 2,048 bp
        # native bins. Compacting the dense overlap matrix and gathering
        # only those entries computes R @ P_native @ R.T without the many
        # multiply-by-zero operations in dense matrix products.
        with torch.autocast(
            device_type=native_prediction.device.type,
            enabled=False,
        ):
            native_float = native_prediction.float()
            resample_float = resample_weights.float()
            overlap_values, overlap_indices = torch.topk(
                resample_float,
                k=MAX_OVERLAPS_PER_TARGET_BIN,
                dim=-1,
                sorted=False,
            )

            retained_row_sums = overlap_values.sum(dim=-1)
            dense_row_sums = resample_float.sum(dim=-1)
            if not torch.allclose(
                retained_row_sums,
                dense_row_sums,
                atol=1e-6,
                rtol=0.0,
            ):
                left_resampled = torch.einsum(
                    "bpi,bcij->bcpj",
                    resample_float,
                    native_float,
                )
                return torch.einsum(
                    "bcpj,bqj->bcpq",
                    left_resampled,
                    resample_float,
                )

            target_bins = resample_float.shape[1]
            contexts = native_float.shape[1]
            maximum_overlaps = overlap_values.shape[-1]

            row_source = native_float.unsqueeze(2).expand(
                -1,
                -1,
                target_bins,
                -1,
                -1,
            )
            row_indices = overlap_indices[
                :,
                None,
                :,
                :,
                None,
            ].expand(
                -1,
                contexts,
                -1,
                -1,
                NATIVE_BINS,
            )
            selected_rows = torch.gather(
                row_source,
                dim=3,
                index=row_indices,
            )
            left_resampled = (
                selected_rows
                * overlap_values[:, None, :, :, None]
            ).sum(dim=3)

            column_source = left_resampled.unsqueeze(3).expand(
                -1,
                -1,
                -1,
                target_bins,
                -1,
            )
            column_indices = overlap_indices[
                :,
                None,
                None,
                :,
                :,
            ].expand(
                -1,
                contexts,
                target_bins,
                -1,
                maximum_overlaps,
            )
            selected_columns = torch.gather(
                column_source,
                dim=4,
                index=column_indices,
            )
            prediction = (
                selected_columns
                * overlap_values[:, None, None, :, :]
            ).sum(dim=-1)

        return prediction

    def project_spatial_features(
        self,
        pair_embedding: torch.Tensor,
        resample_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Project pair embeddings and resample 32 feature maps to 5 kb."""
        with torch.autocast(
            device_type=pair_embedding.device.type,
            enabled=False,
        ):
            native_features = torch.einsum(
                "bijd,kd->bkij",
                pair_embedding.float(),
                self.pair_projection.weight.float(),
            )

        return self.resample_maps(native_features, resample_weights)

    def decode_spatial_residual(
        self,
        projected_features: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Decode context-conditioned spatial corrections at 5 kb."""
        batch_size = projected_features.shape[0]

        if context_embedding.ndim == 2:
            batched_context = context_embedding.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        else:
            batched_context = context_embedding

        context_count = batched_context.shape[1]
        if projected_features.shape != (
            batch_size,
            self.spatial_channels,
            TARGET_BINS,
            TARGET_BINS,
        ):
            raise ValueError(
                "projected_features must have shape "
                f"[B, {self.spatial_channels}, {TARGET_BINS}, "
                f"{TARGET_BINS}]; found {tuple(projected_features.shape)}"
            )

        with torch.autocast(
            device_type=projected_features.device.type,
            enabled=False,
        ):
            film = self.context_to_film(
                batched_context.float()
            ).view(
                batch_size,
                context_count,
                len(self.spatial_blocks),
                2,
                self.spatial_channels,
            )

            shared_features = projected_features.float().unsqueeze(1).expand(
                -1,
                context_count,
                -1,
                -1,
                -1,
            )
            distance = self.distance_channel.expand(
                batch_size,
                context_count,
                -1,
                -1,
                -1,
            )
            features = torch.cat(
                (shared_features, distance),
                dim=2,
            ).reshape(
                batch_size * context_count,
                self.spatial_channels + 1,
                TARGET_BINS,
                TARGET_BINS,
            )
            features = self.spatial_input(features)

            for block_index, block in enumerate(self.spatial_blocks):
                scale = film[:, :, block_index, 0].reshape(
                    batch_size * context_count,
                    self.spatial_channels,
                )
                shift = film[:, :, block_index, 1].reshape(
                    batch_size * context_count,
                    self.spatial_channels,
                )
                features = block(features, scale, shift)

            residual = self.spatial_output(features).reshape(
                batch_size,
                context_count,
                TARGET_BINS,
                TARGET_BINS,
            )

        return residual

    def forward(
        self,
        pair_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        resample_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Predict cell-type-specific Hi-C maps.

        Args:
            pair_embedding: ``[B, 512, 512, 128]``
            context_embedding: ``[C, 14]`` or ``[B, C, 14]``
            resample_weights: ``[B, 200, 512]``

        Returns:
            Float32 symmetric predictions with shape ``[B, C, 200, 200]``.
        """
        if not isinstance(context_embedding, torch.Tensor):
            raise TypeError(
                "context_embedding must be a torch.Tensor; "
                f"found {type(context_embedding).__name__}"
            )
        if not isinstance(resample_weights, torch.Tensor):
            raise TypeError(
                "resample_weights must be a torch.Tensor; "
                f"found {type(resample_weights).__name__}"
            )
        if not isinstance(pair_embedding, torch.Tensor):
            raise TypeError(
                "pair_embedding must be a torch.Tensor; "
                f"found {type(pair_embedding).__name__}"
            )
        if pair_embedding.ndim != 4:
            raise ValueError(
                "pair_embedding must have shape [B, N, N, D]; "
                f"found {tuple(pair_embedding.shape)}"
            )

        if (
            pair_embedding.device != context_embedding.device
            or pair_embedding.device != resample_weights.device
        ):
            raise ValueError(
                "pair_embedding, context_embedding, and resample_weights "
                "must be on the same device"
            )

        batch_size = pair_embedding.shape[0]

        weights, biases = self.conditioned_parameters(
            context_embedding,
            batch_size=batch_size,
        )
        native_prediction = self.decode_native(
            pair_embedding,
            weights,
            biases,
        )
        prediction = self.resample_maps(
            native_prediction,
            resample_weights,
        )
        projected_features = self.project_spatial_features(
            pair_embedding,
            resample_weights,
        )
        prediction = prediction + self.decode_spatial_residual(
            projected_features,
            context_embedding,
        )

        # Remove any numerical asymmetry introduced by the matrix products.
        prediction = 0.5 * (
            prediction + prediction.transpose(-1, -2)
        )

        return prediction

def _validate_metric_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    diagonal: int,
) -> int:
    """Validate tensors shared by the masked map metrics."""
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            "prediction must be a torch.Tensor; "
            f"found {type(prediction).__name__}"
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            "target must be a torch.Tensor; "
            f"found {type(target).__name__}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes; "
            f"found {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError(
            "prediction and target must end with two map dimensions; "
            f"found shape {tuple(prediction.shape)}"
        )
    if prediction.shape[-2] != prediction.shape[-1]:
        raise ValueError(
            "Contact maps must be square; "
            f"found trailing shape {tuple(prediction.shape[-2:])}"
        )
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty")
    if not torch.is_floating_point(prediction):
        raise TypeError(
            "prediction must use a floating-point dtype; "
            f"found {prediction.dtype}"
        )
    if not torch.is_floating_point(target):
        raise TypeError(
            "target must use a floating-point dtype; "
            f"found {target.dtype}"
        )
    if prediction.device != target.device:
        raise ValueError(
            "prediction and target must be on the same device"
        )

    diagonal = _require_integer("diagonal", diagonal)
    map_size = prediction.shape[-1]

    if diagonal < 0 or diagonal >= map_size:
        raise ValueError(
            f"diagonal must satisfy 0 <= diagonal < {map_size}; "
            f"found {diagonal}"
        )

    return diagonal


def _upper_triangle_target_mask(
    target: torch.Tensor,
    *,
    diagonal: int,
) -> torch.Tensor:
    """Return the finite-target, upper-triangle mask."""
    map_size = target.shape[-1]
    upper_triangle = torch.triu(
        torch.ones(
            (map_size, map_size),
            dtype=torch.bool,
            device=target.device,
        ),
        diagonal=diagonal,
    )
    return torch.isfinite(target) & upper_triangle


def masked_upper_triangle_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    diagonal: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute equal-map-weighted masked MSE.

    NaN target values, the lower triangle, and diagonals below ``diagonal``
    are excluded. Each map is normalized by its own valid-pixel count
    before the valid map losses are averaged.

    Returns:
        mean_mse:
            Scalar equal-map-weighted loss.
        per_map_mse:
            MSE for each leading map index. Maps without valid pixels are
            represented by NaN.
        valid_pixel_counts:
            Int64 valid-pixel count for each map.
    """
    diagonal = _validate_metric_inputs(
        prediction,
        target,
        diagonal,
    )

    with torch.autocast(
        device_type=prediction.device.type,
        enabled=False,
    ):
        prediction_float = prediction.float()
        target_float = target.float()

        valid_mask = _upper_triangle_target_mask(
            target_float,
            diagonal=diagonal,
        )
        valid_pixel_counts = valid_mask.sum(dim=(-2, -1))
        valid_maps = valid_pixel_counts > 0

        if not bool(valid_maps.any().item()):
            raise ValueError(
                "No maps contain valid target pixels after masking"
            )

        # Replacing invalid target entries before subtraction prevents NaNs
        # from contaminating masked sums.
        target_safe = torch.where(
            valid_mask,
            target_float,
            torch.zeros_like(target_float),
        )
        error = torch.where(
            valid_mask,
            prediction_float - target_safe,
            torch.zeros_like(prediction_float),
        )

        squared_error_sums = error.square().sum(dim=(-2, -1))
        per_map_mse = squared_error_sums / (
            valid_pixel_counts.clamp_min(1).float()
        )
        per_map_mse = torch.where(
            valid_maps,
            per_map_mse,
            torch.full_like(per_map_mse, float("nan")),
        )

        mean_mse = per_map_mse[valid_maps].mean()

    return mean_mse, per_map_mse, valid_pixel_counts


def masked_upper_triangle_pearson(
    prediction: torch.Tensor,
    target: torch.Tensor,
    diagonal: int = 4,
) -> torch.Tensor:
    """Compute Pearson correlation independently for every map.

    Uses the same finite-target upper-triangle mask as the MSE. A map
    returns NaN when it has fewer than two valid pixels, either vector has
    zero variance, or its correlation is otherwise non-finite.

    Returns:
        Per-map correlations with shape ``prediction.shape[:-2]``.
    """
    diagonal = _validate_metric_inputs(
        prediction,
        target,
        diagonal,
    )

    with torch.autocast(
        device_type=prediction.device.type,
        enabled=False,
    ):
        prediction_float = prediction.float()
        target_float = target.float()

        valid_mask = _upper_triangle_target_mask(
            target_float,
            diagonal=diagonal,
        )
        valid_pixel_counts = valid_mask.sum(dim=(-2, -1))
        safe_counts = valid_pixel_counts.clamp_min(1).float()

        masked_prediction = torch.where(
            valid_mask,
            prediction_float,
            torch.zeros_like(prediction_float),
        )
        masked_target = torch.where(
            valid_mask,
            target_float,
            torch.zeros_like(target_float),
        )

        prediction_mean = (
            masked_prediction.sum(dim=(-2, -1)) / safe_counts
        )
        target_mean = (
            masked_target.sum(dim=(-2, -1)) / safe_counts
        )

        centered_prediction = torch.where(
            valid_mask,
            prediction_float - prediction_mean[..., None, None],
            torch.zeros_like(prediction_float),
        )
        centered_target = torch.where(
            valid_mask,
            target_float - target_mean[..., None, None],
            torch.zeros_like(target_float),
        )

        numerator = (
            centered_prediction * centered_target
        ).sum(dim=(-2, -1))
        prediction_sum_squares = centered_prediction.square().sum(
            dim=(-2, -1)
        )
        target_sum_squares = centered_target.square().sum(
            dim=(-2, -1)
        )
        denominator = torch.sqrt(
            prediction_sum_squares * target_sum_squares
        )

        defined = (
            (valid_pixel_counts >= 2)
            & torch.isfinite(numerator)
            & torch.isfinite(denominator)
            & (denominator > 0)
        )

        correlations = torch.full_like(
            numerator,
            float("nan"),
        )
        correlations = torch.where(
            defined,
            (numerator / denominator).clamp(min=-1.0, max=1.0),
            correlations,
        )

    return correlations
