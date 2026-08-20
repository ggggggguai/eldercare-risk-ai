from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .causalnet import CausalNet
from .causalnet_opt_me_003 import REDUCED_CAUSALNET_CONFIG
from .causalnet_opt_me_004 import ME4CandidateContract, PerSampleClassificationLoss, build_me4_causalnet


@dataclass(frozen=True)
class ME5CandidateContract:
    candidate_id: str
    consistency_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    route_dropout_probabilities: tuple[float, float, float] = (0.0, 0.0, 0.0)
    consistency_ramp: bool = False
    positive_margin: bool = False
    complexity_rank: int = 0


CANDIDATE_REGISTRY: dict[str, ME5CandidateContract] = {
    "B0": ME5CandidateContract("B0", complexity_rank=0),
    "C1": ME5CandidateContract("C1", (0.10, 0.10, 0.10), (0.10, 0.10, 0.10), complexity_rank=1),
    "C2": ME5CandidateContract("C2", (0.10, 0.03, 0.10), (0.10, 0.00, 0.10), complexity_rank=2),
    "C3": ME5CandidateContract("C3", (0.10, 0.03, 0.10), (0.10, 0.00, 0.10), True, complexity_rank=3),
    "C4": ME5CandidateContract("C4", positive_margin=True, complexity_rank=2),
    "C5": ME5CandidateContract("C5", (0.10, 0.03, 0.10), (0.10, 0.00, 0.10), True, True, 4),
}

POSITIVE_LABEL = 1
POSITIVE_MARGIN = 0.20
POSITIVE_MARGIN_WEIGHT = 0.05
CONSISTENCY_RAMP_FRACTION = 0.40


def build_me5_causalnet() -> CausalNet:
    model = build_me4_causalnet()
    if sum(parameter.numel() for parameter in model.parameters()) != 623963:
        raise RuntimeError("OPT-ME-005 reduced CausalNet parameter drift")
    return model


def build_candidate_loss(*, class_counts: Sequence[int]) -> PerSampleClassificationLoss:
    return PerSampleClassificationLoss(
        ME4CandidateContract("OPT-ME-005-LDAM", "ldam_drw"), class_counts
    )


def augment_route_consistency_batch(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
    *,
    dropout_probabilities: Sequence[float],
    magnitude_min: float = 0.9,
    magnitude_max: float = 1.1,
    noise_sigma: float = 0.01,
) -> torch.Tensor:
    if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (4, 3, 28, 28):
        raise ValueError("route-consistency inputs must be [batch,4,3,28,28]")
    if labels.ndim != 1 or len(labels) != len(inputs):
        raise ValueError("route-consistency labels must match batch")
    probabilities = torch.as_tensor(dropout_probabilities, dtype=torch.float32)
    if probabilities.shape != (3,) or torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("dropout probabilities must contain three values in [0,1]")
    result = inputs.clone()
    factors = torch.empty((len(result), 1, 1, 1, 1), dtype=result.dtype).uniform_(
        magnitude_min, magnitude_max, generator=generator
    )
    result[:, 0:2] *= factors
    result[:, 0:2] += torch.randn(
        result[:, 0:2].shape, generator=generator, dtype=result.dtype
    ) * float(noise_sigma)
    draws = torch.rand(len(result), generator=generator)
    routes = torch.randint(0, 4, (len(result),), generator=generator)
    cpu_labels = labels.detach().cpu().long()
    for index in range(len(result)):
        if float(draws[index]) < float(probabilities[cpu_labels[index]]):
            result[index, int(routes[index])] = 0.0
    result[:, 0:2, 0:2].clamp_(-1.0, 1.0)
    result[:, 0:2, 2].clamp_(0.0, 1.0)
    result[:, 2:4].clamp_(0.0, 1.0)
    return result


def symmetric_kl_per_sample(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        raise ValueError("symmetric KL logits must have matching [batch,class] shape")
    first_log = F.log_softmax(first_logits, dim=1)
    second_log = F.log_softmax(second_logits, dim=1)
    first = first_log.exp()
    second = second_log.exp()
    return 0.5 * (
        torch.sum(first * (first_log - second_log), dim=1)
        + torch.sum(second * (second_log - first_log), dim=1)
    )


def class_weighted_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Sequence[float],
) -> torch.Tensor:
    per_sample = symmetric_kl_per_sample(first_logits, second_logits)
    class_weights = torch.as_tensor(weights, dtype=per_sample.dtype, device=per_sample.device)
    if class_weights.shape != (3,):
        raise ValueError("consistency weights must contain three values")
    return torch.mean(per_sample * class_weights[labels])


def positive_margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels == POSITIVE_LABEL
    if not torch.any(mask):
        return logits.sum() * 0.0
    selected = logits[mask]
    competing = torch.maximum(selected[:, 0], selected[:, 2])
    return F.softplus(competing - selected[:, POSITIVE_LABEL] + POSITIVE_MARGIN).mean()


def consistency_ramp(epoch: int, max_epochs: int) -> float:
    if epoch <= 0 or max_epochs <= 0:
        raise ValueError("epoch and max_epochs must be positive")
    return min(1.0, float(epoch) / (float(max_epochs) * CONSISTENCY_RAMP_FRACTION))


__all__ = [
    "CANDIDATE_REGISTRY", "CONSISTENCY_RAMP_FRACTION", "ME5CandidateContract",
    "POSITIVE_LABEL", "POSITIVE_MARGIN", "POSITIVE_MARGIN_WEIGHT", "REDUCED_CAUSALNET_CONFIG",
    "augment_route_consistency_batch", "build_candidate_loss", "build_me5_causalnet",
    "class_weighted_consistency_loss", "consistency_ramp", "positive_margin_loss",
    "symmetric_kl_per_sample",
]
