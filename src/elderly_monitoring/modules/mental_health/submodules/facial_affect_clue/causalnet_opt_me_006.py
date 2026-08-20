from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .causalnet import CausalNet
from .causalnet_opt_me_003 import REDUCED_CAUSALNET_CONFIG
from .causalnet_opt_me_004 import PerSampleClassificationLoss
from .causalnet_opt_me_005 import (
    build_candidate_loss as build_me5_candidate_loss,
    build_me5_causalnet,
    positive_margin_loss,
)


@dataclass(frozen=True)
class ME6CandidateContract:
    candidate_id: str
    positive_margin: bool = False
    margin_ramp: bool = False
    non_positive_guard: bool = False
    subject_tail_cvar: bool = False
    complexity_rank: int = 0


CANDIDATE_REGISTRY: dict[str, ME6CandidateContract] = {
    "B0": ME6CandidateContract("B0", complexity_rank=0),
    "P0": ME6CandidateContract("P0", positive_margin=True, complexity_rank=1),
    "P1": ME6CandidateContract("P1", positive_margin=True, margin_ramp=True, complexity_rank=2),
    "P2": ME6CandidateContract(
        "P2", positive_margin=True, margin_ramp=True,
        non_positive_guard=True, complexity_rank=3,
    ),
    "P3": ME6CandidateContract(
        "P3", positive_margin=True, margin_ramp=True,
        subject_tail_cvar=True, complexity_rank=3,
    ),
    "P4": ME6CandidateContract(
        "P4", positive_margin=True, margin_ramp=True,
        non_positive_guard=True, subject_tail_cvar=True, complexity_rank=4,
    ),
}

POSITIVE_LABEL = 1
POSITIVE_MARGIN = 0.20
POSITIVE_MARGIN_MAXIMUM_WEIGHT = 0.05
MARGIN_RAMP_FRACTION = 0.40
GUARD_MARGIN = 0.10
GUARD_WEIGHT = 0.025
TAIL_FRACTION = 0.30
TAIL_MAXIMUM_WEIGHT = 0.05


def build_me6_causalnet() -> CausalNet:
    model = build_me5_causalnet()
    if sum(parameter.numel() for parameter in model.parameters()) != 623963:
        raise RuntimeError("OPT-ME-006 reduced CausalNet parameter drift")
    return model


def build_candidate_loss(*, class_counts: Sequence[int]) -> PerSampleClassificationLoss:
    """Exact OPT-ME-005 B0 LDAM-DRW replay."""
    return build_me5_candidate_loss(class_counts=class_counts)


def loss_ramp(epoch: int, max_epochs: int) -> float:
    if epoch <= 0 or max_epochs <= 0:
        raise ValueError("epoch and max_epochs must be positive")
    return min(1.0, float(epoch) / (float(max_epochs) * MARGIN_RAMP_FRACTION))


def non_positive_boundary_guard_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("guard logits must have shape [batch,3]")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("guard labels must match logits batch")
    mask = labels != POSITIVE_LABEL
    if not torch.any(mask):
        return logits.sum() * 0.0
    selected = logits[mask]
    true_logits = selected.gather(1, labels[mask, None]).squeeze(1)
    return F.softplus(selected[:, POSITIVE_LABEL] - true_logits + GUARD_MARGIN).mean()


def subject_tail_cvar_loss(
    per_sample_losses: torch.Tensor,
    subject_ids: Sequence[str],
) -> torch.Tensor:
    if per_sample_losses.ndim != 1 or len(per_sample_losses) != len(subject_ids):
        raise ValueError("tail loss requires one loss and subject id per sample")
    ordered_subjects = list(dict.fromkeys(str(subject_id) for subject_id in subject_ids))
    if len(ordered_subjects) < 2:
        return per_sample_losses.sum() * 0.0
    subject_means = torch.stack([
        per_sample_losses[
            torch.as_tensor(
                [str(value) == subject for value in subject_ids],
                dtype=torch.bool,
                device=per_sample_losses.device,
            )
        ].mean()
        for subject in ordered_subjects
    ])
    tail_count = max(1, math.ceil(TAIL_FRACTION * len(ordered_subjects)))
    tail_mean = torch.topk(subject_means, k=tail_count, largest=True).values.mean()
    return torch.clamp_min(tail_mean - subject_means.mean(), 0.0)


__all__ = [
    "CANDIDATE_REGISTRY", "GUARD_MARGIN", "GUARD_WEIGHT", "MARGIN_RAMP_FRACTION",
    "ME6CandidateContract", "POSITIVE_LABEL", "POSITIVE_MARGIN",
    "POSITIVE_MARGIN_MAXIMUM_WEIGHT", "REDUCED_CAUSALNET_CONFIG", "TAIL_FRACTION",
    "TAIL_MAXIMUM_WEIGHT", "build_candidate_loss", "build_me6_causalnet", "loss_ramp",
    "non_positive_boundary_guard_loss", "positive_margin_loss", "subject_tail_cvar_loss",
]
