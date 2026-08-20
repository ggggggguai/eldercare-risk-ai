from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .causalnet import CausalNet, CausalNetConfig


@dataclass(frozen=True)
class CandidateContract:
    candidate_id: str
    loss_kind: str
    sampler_kind: str = "random"
    description: str = ""


CANDIDATE_REGISTRY: dict[str, CandidateContract] = {
    "C0": CandidateContract(
        "C0",
        "weighted_cross_entropy",
        description="aligned reduced CausalNet baseline",
    ),
    "C1": CandidateContract(
        "C1",
        "balanced_softmax",
        description="C0 plus Balanced Softmax",
    ),
    "C2": CandidateContract(
        "C2",
        "ldam_drw",
        description="C0 plus LDAM with deferred reweighting",
    ),
    "C3": CandidateContract(
        "C3",
        "class_balanced_focal",
        description="C0 plus class-balanced focal loss",
    ),
    "C4": CandidateContract(
        "C4",
        "weighted_cross_entropy",
        sampler_kind="subject_class_two_level",
        description="C0 plus subject/class two-level sampling",
    ),
}


REDUCED_CAUSALNET_CONFIG = CausalNetConfig(
    image_size=28,
    patch_size=7,
    dim=64,
    heads=2,
    num_hierarchies=3,
    block_repeats=(1, 1, 1),
    num_classes=3,
    gamma=0.5,
    mlp_mult=2,
    cross_heads=4,
    cross_depth=2,
    dropout=0.1,
    channels=3,
    head_hidden_dim=256,
)


def build_me3_causalnet() -> CausalNet:
    model = CausalNet(REDUCED_CAUSALNET_CONFIG)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 623963:
        raise RuntimeError(f"ME3 reduced CausalNet parameter drift: {parameter_count}")
    return model


class CandidateClassificationLoss(nn.Module):
    def __init__(
        self,
        contract: CandidateContract,
        class_counts: Sequence[int],
        *,
        focal_gamma: float = 2.0,
        effective_beta: float = 0.9999,
        ldam_max_margin: float = 0.5,
        ldam_scale: float = 30.0,
        drw_start_epoch: int = 8,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.shape != (3,) or torch.any(counts <= 0):
            raise ValueError("class_counts must contain three positive values")
        self.contract = contract
        self.focal_gamma = float(focal_gamma)
        self.effective_beta = float(effective_beta)
        self.ldam_scale = float(ldam_scale)
        self.drw_start_epoch = int(drw_start_epoch)
        class_weights = counts.sum() / (len(counts) * counts)
        effective_weights = (1.0 - effective_beta) / (
            1.0 - torch.pow(torch.tensor(effective_beta), counts)
        )
        effective_weights = effective_weights / effective_weights.sum() * len(counts)
        margins = torch.pow(counts, -0.25)
        margins = margins / margins.max() * ldam_max_margin
        self.register_buffer("class_counts", counts)
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("effective_weights", effective_weights)
        self.register_buffer("ldam_margins", margins)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, *, epoch: int = 0
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != 3:
            raise ValueError("candidate loss expects [batch,3] logits")
        if labels.ndim != 1 or len(labels) != len(logits):
            raise ValueError("candidate loss labels must match logits batch")
        kind = self.contract.loss_kind
        if kind == "weighted_cross_entropy":
            return F.cross_entropy(logits, labels, weight=self.class_weights.to(logits))
        if kind == "balanced_softmax":
            adjusted = logits + self.class_counts.to(logits).log().unsqueeze(0)
            return F.cross_entropy(adjusted, labels)
        if kind == "class_balanced_focal":
            ce = F.cross_entropy(logits, labels, reduction="none")
            probability = torch.exp(-ce)
            alpha = self.effective_weights.to(logits)[labels]
            return (alpha * torch.pow(1.0 - probability, self.focal_gamma) * ce).mean()
        if kind == "ldam_drw":
            adjusted = logits.clone()
            row_index = torch.arange(len(labels), device=labels.device)
            adjusted[row_index, labels] -= self.ldam_margins.to(logits)[labels]
            weights = (
                self.effective_weights.to(logits)
                if int(epoch) >= self.drw_start_epoch
                else None
            )
            return F.cross_entropy(adjusted * self.ldam_scale, labels, weight=weights)
        raise ValueError(f"unsupported OPT-ME-003 loss: {kind}")


def build_candidate_loss(
    candidate_id: str, *, class_counts: Sequence[int]
) -> CandidateClassificationLoss:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown OPT-ME-003 candidate: {candidate_id}")
    return CandidateClassificationLoss(CANDIDATE_REGISTRY[candidate_id], class_counts)


def two_level_sample_weights(rows: Sequence[Mapping[str, object]]) -> torch.Tensor:
    if not rows:
        raise ValueError("two-level sampler requires non-empty rows")
    subject_counts = Counter(str(row["subject_id"]) for row in rows)
    class_counts = Counter(int(row["label"]) for row in rows)
    if set(class_counts) - {0, 1, 2}:
        raise ValueError("two-level sampler received an invalid label")
    values = [
        1.0
        / math.sqrt(
            subject_counts[str(row["subject_id"])] * class_counts[int(row["label"])]
        )
        for row in rows
    ]
    weights = torch.tensor(values, dtype=torch.double)
    return weights / weights.mean()
