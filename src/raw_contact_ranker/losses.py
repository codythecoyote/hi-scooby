from __future__ import annotations

import torch
import torch.nn.functional as F


def importance_corrected_event_loss(
    positive_score: torch.Tensor,
    control_score: torch.Tensor,
    proposal_probability: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sampled-softmax event loss; controls are opportunities, not class labels."""
    if positive_score.ndim != 1 or control_score.ndim != 2:
        raise ValueError("Expected positive [B] and control [B,K] scores")
    if control_score.shape != proposal_probability.shape:
        raise ValueError("Control score and proposal probability shapes differ")
    if torch.any(proposal_probability <= 0):
        raise ValueError("Proposal probabilities must be strictly positive")
    sample_count = torch.as_tensor(
        control_score.shape[1], dtype=control_score.dtype, device=control_score.device
    )
    corrected = control_score - torch.log(proposal_probability) - torch.log(sample_count)
    logits = torch.cat([positive_score[:, None], corrected], dim=1)
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Sampled-softmax correction is non-finite")
    target = torch.zeros(len(positive_score), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target, reduction=reduction)


def confidence_weighted_pairwise_logistic_loss(
    positive_score: torch.Tensor,
    control_score: torch.Tensor,
    support_weight: torch.Tensor,
) -> torch.Tensor:
    if positive_score.ndim != 1 or control_score.ndim != 2:
        raise ValueError("Expected positive [B] and control [B,K] scores")
    if support_weight.shape != positive_score.shape:
        raise ValueError("Support weights must have shape [B]")
    loss = F.softplus(-(positive_score[:, None] - control_score))
    weighted = loss.mean(dim=1) * support_weight
    denominator = support_weight.sum().clamp_min(torch.finfo(weighted.dtype).eps)
    return weighted.sum() / denominator


def raw_contact_loss(
    positive_score: torch.Tensor,
    control_score: torch.Tensor,
    proposal_probability: torch.Tensor,
    support_weight: torch.Tensor,
    *,
    lambda_rank: float,
    rank_positive_score: torch.Tensor | None = None,
    rank_control_score: torch.Tensor | None = None,
    regularization: torch.Tensor | float = 0.0,
    event_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    event_rows = importance_corrected_event_loss(
        positive_score, control_score, proposal_probability, reduction="none"
    )
    if event_count is None:
        event = event_rows.mean()
    else:
        if event_count.shape != positive_score.shape or torch.any(event_count <= 0):
            raise ValueError("Event counts must be positive with shape [B]")
        event = (event_rows * event_count).sum() / event_count.sum()
    rank = confidence_weighted_pairwise_logistic_loss(
        positive_score if rank_positive_score is None else rank_positive_score,
        control_score if rank_control_score is None else rank_control_score,
        support_weight,
    )
    regularization = torch.as_tensor(
        regularization, dtype=event.dtype, device=event.device
    )
    total = event + float(lambda_rank) * rank + regularization
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("Raw-contact loss is non-finite")
    return total, {"event": event, "rank": rank, "regularization": regularization}
