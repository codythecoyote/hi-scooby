from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _finite_tensors(values: dict[str, torch.Tensor]) -> None:
    checks = torch.stack([torch.isfinite(value).all() for value in values.values()])
    if bool(checks.all().item()):
        return
    for name, value in values.items():
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values")


class RawContactRanker(nn.Module):
    """Symmetric sparse pair scorer with an optional centered rank-8 RNA term."""

    def __init__(
        self,
        *,
        pair_dim: int = 128,
        anchor_dim: int = 128,
        annotation_dim: int = 16,
        pair_annotation_dim: int = 5,
        technical_dim: int = 5,
        hidden_dim: int = 128,
        context_dim: int = 14,
        programs: int = 8,
        dropout: float = 0.1,
        feature_set: str = "combined",
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.anchor_dim = anchor_dim
        self.annotation_dim = annotation_dim
        self.pair_annotation_dim = pair_annotation_dim
        self.technical_dim = technical_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.programs = programs
        self.dropout = dropout
        if feature_set not in {"annotations", "alphagenome", "combined"}:
            raise ValueError(f"Unsupported feature_set: {feature_set!r}")
        self.feature_set = feature_set
        symmetric_dim = (
            pair_dim + 2 * anchor_dim + annotation_dim + pair_annotation_dim
        )
        self.sequence = nn.Sequential(
            nn.Linear(symmetric_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.sequence_score = nn.Linear(hidden_dim, 1)
        self.contact_programs = nn.Linear(hidden_dim, programs)
        self.technical = nn.Linear(technical_dim, 1, bias=False)
        self.rna_projection = nn.Linear(context_dim, programs, bias=False)
        nn.init.zeros_(self.rna_projection.weight)

    @staticmethod
    def symmetric_anchor_encoding(
        anchor_i: torch.Tensor,
        anchor_j: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([anchor_i + anchor_j, torch.abs(anchor_i - anchor_j)], dim=-1)

    def forward(
        self,
        pair_embedding: torch.Tensor,
        anchor_i: torch.Tensor,
        anchor_j: torch.Tensor,
        annotations: torch.Tensor,
        pair_annotations: torch.Tensor,
        technical: torch.Tensor,
        fixed_exposure: torch.Tensor,
        fixed_distance_offset: torch.Tensor,
        distance_bp: torch.Tensor,
        context_embedding: torch.Tensor | None = None,
        context_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        finite_inputs = {
            "pair_embedding": pair_embedding,
            "anchor_i": anchor_i,
            "anchor_j": anchor_j,
            "annotations": annotations,
            "pair_annotations": pair_annotations,
            "technical": technical,
            "fixed_exposure": fixed_exposure,
            "fixed_distance_offset": fixed_distance_offset,
            "distance_bp": distance_bp,
        }
        if context_embedding is not None:
            finite_inputs["context_embedding"] = context_embedding
        _finite_tensors(finite_inputs)
        if self.feature_set == "annotations":
            pair_embedding = torch.zeros_like(pair_embedding)
            anchor_i = torch.zeros_like(anchor_i)
            anchor_j = torch.zeros_like(anchor_j)
        elif self.feature_set == "alphagenome":
            annotations = torch.zeros_like(annotations)
            pair_annotations = torch.zeros_like(pair_annotations)
            technical = torch.zeros_like(technical)
        annotation_pairs = annotations.reshape(*annotations.shape[:-1], -1, 2)
        annotations = torch.cat(
            [annotation_pairs.sum(dim=-1), torch.abs(annotation_pairs[..., 0] - annotation_pairs[..., 1])],
            dim=-1,
        )
        technical_pairs = technical[..., 1:].reshape(*technical.shape[:-1], -1, 2)
        technical = torch.cat(
            [
                technical[..., :1],
                technical_pairs.sum(dim=-1),
                torch.abs(technical_pairs[..., 0] - technical_pairs[..., 1]),
            ],
            dim=-1,
        )
        symmetric = torch.cat(
            [
                pair_embedding,
                self.symmetric_anchor_encoding(anchor_i, anchor_j),
                annotations,
                pair_annotations,
            ],
            dim=-1,
        )
        hidden = self.sequence(symmetric)
        residual_score = self.sequence_score(hidden).squeeze(-1)
        technical_offset = self.technical(technical).squeeze(-1)
        base = (
            fixed_exposure
            + fixed_distance_offset
            + technical_offset
            + residual_score
        )
        context_delta: torch.Tensor
        if context_embedding is None:
            if context_index is not None:
                raise ValueError("context_index requires context_embedding")
            context_delta = torch.zeros_like(base)
            log_rate = base
        else:
            coefficients = self.rna_projection(context_embedding)
            coefficients = coefficients - coefficients.mean(dim=0, keepdim=True)
            # A unit-norm program vector removes the bilinear scale escape in
            # which contact_programs grows while the penalized RNA projection
            # shrinks without changing the context delta.
            programs = torch.nn.functional.normalize(
                self.contact_programs(hidden), dim=-1
            )
            if context_index is None:
                context_delta = torch.einsum("bp,cp->bc", programs, coefficients)
                log_rate = base[:, None] + context_delta
            else:
                if context_index.shape != base.shape:
                    raise ValueError("context_index must have one entry per pair")
                if context_index.dtype != torch.long:
                    context_index = context_index.long()
                selected_coefficients = coefficients[context_index]
                context_delta = torch.sum(programs * selected_coefficients, dim=-1)
                log_rate = base + context_delta
        _finite_tensors({"log_rate": log_rate})
        return {
            "log_rate": log_rate,
            "residual_score": residual_score,
            "distance_offset": fixed_distance_offset,
            "exposure_offset": fixed_exposure + technical_offset,
            "technical_offset": technical_offset,
            "context_delta": context_delta,
        }

    @staticmethod
    def probability_within_band(
        log_rate: torch.Tensor,
        group_index: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(log_rate)
        for group in torch.unique(group_index):
            selected = group_index == group
            output[selected] = torch.softmax(log_rate[selected], dim=0)
        return output

    def rna_regularization(self) -> torch.Tensor:
        return (
            self.rna_projection.weight.square().sum()
            + self.contact_programs.weight.square().sum()
            + self.contact_programs.bias.square().sum()
        )

    def freeze_sequence(self) -> None:
        for module in (
            self.sequence,
            self.sequence_score,
            self.technical,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def config_dict(self) -> dict[str, Any]:
        return {
            "pair_dim": self.pair_dim,
            "anchor_dim": self.anchor_dim,
            "annotation_dim": self.annotation_dim,
            "pair_annotation_dim": self.pair_annotation_dim,
            "technical_dim": self.technical_dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "programs": self.programs,
            "dropout": self.dropout,
            "feature_set": self.feature_set,
        }
