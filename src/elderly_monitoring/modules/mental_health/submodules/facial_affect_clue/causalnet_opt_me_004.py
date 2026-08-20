from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .causalnet import CausalNet
from .causalnet_opt_me_003 import REDUCED_CAUSALNET_CONFIG, build_me3_causalnet


@dataclass(frozen=True)
class ME4CandidateContract:
    candidate_id: str
    loss_kind: str
    use_ema: bool = False
    subject_equalized: bool = False
    route_consistency_ssl: bool = False
    complexity_rank: int = 0


CANDIDATE_REGISTRY: dict[str, ME4CandidateContract] = {
    "M0": ME4CandidateContract("M0", "weighted_cross_entropy", complexity_rank=0),
    "M1": ME4CandidateContract("M1", "ldam_drw", complexity_rank=1),
    "M2": ME4CandidateContract("M2", "ldam_drw", use_ema=True, complexity_rank=2),
    "M3": ME4CandidateContract("M3", "ldam_drw", subject_equalized=True, complexity_rank=2),
    "M4": ME4CandidateContract("M4", "ldam_drw", route_consistency_ssl=True, complexity_rank=2),
    "M5": ME4CandidateContract(
        "M5", "ldam_drw", use_ema=True, subject_equalized=True, complexity_rank=3
    ),
}

EMA_DECAY = 0.999
SUBJECT_SAMPLE_WEIGHT = 0.7
SUBJECT_EQUAL_WEIGHT = 0.3
CONSISTENCY_WEIGHT = 0.1


def build_me4_causalnet() -> CausalNet:
    model = build_me3_causalnet()
    if sum(parameter.numel() for parameter in model.parameters()) != 623963:
        raise RuntimeError("OPT-ME-004 reduced CausalNet parameter drift")
    return model


class PerSampleClassificationLoss(nn.Module):
    def __init__(
        self,
        contract: ME4CandidateContract,
        class_counts: Sequence[int],
        *,
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

    def per_sample(
        self, logits: torch.Tensor, labels: torch.Tensor, *, epoch: int
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != 3:
            raise ValueError("candidate loss expects [batch,3] logits")
        if labels.ndim != 1 or len(labels) != len(logits):
            raise ValueError("candidate loss labels must match logits batch")
        if self.contract.loss_kind == "weighted_cross_entropy":
            return F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights.to(logits),
                reduction="none",
            )
        if self.contract.loss_kind != "ldam_drw":
            raise ValueError(f"unsupported OPT-ME-004 loss: {self.contract.loss_kind}")
        adjusted = logits.clone()
        row_index = torch.arange(len(labels), device=labels.device)
        adjusted[row_index, labels] -= self.ldam_margins.to(logits)[labels]
        weights = (
            self.effective_weights.to(logits)
            if int(epoch) >= self.drw_start_epoch
            else None
        )
        return F.cross_entropy(
            adjusted * self.ldam_scale,
            labels,
            weight=weights,
            reduction="none",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int,
        subject_ids: Sequence[str],
        subject_counts: Mapping[str, int],
    ) -> torch.Tensor:
        values = self.per_sample(logits, labels, epoch=epoch)
        if not self.contract.subject_equalized:
            if self.contract.loss_kind == "weighted_cross_entropy":
                denominator = self.class_weights.to(logits)[labels].sum()
                return values.sum() / denominator
            if self.contract.loss_kind == "ldam_drw" and int(epoch) >= self.drw_start_epoch:
                denominator = self.effective_weights.to(logits)[labels].sum()
                return values.sum() / denominator
            return values.mean()
        total_samples = sum(int(value) for value in subject_counts.values())
        subject_total = len(subject_counts)
        if total_samples <= 0 or subject_total <= 0:
            raise ValueError("subject equalization requires train-fold subject counts")
        weights = torch.as_tensor(
            [
                SUBJECT_SAMPLE_WEIGHT
                + SUBJECT_EQUAL_WEIGHT
                * total_samples
                / (subject_total * int(subject_counts[str(subject_id)]))
                for subject_id in subject_ids
            ],
            dtype=values.dtype,
            device=values.device,
        )
        return (weights * values).mean()


def build_candidate_loss(
    candidate_id: str, *, class_counts: Sequence[int]
) -> PerSampleClassificationLoss:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown OPT-ME-004 candidate: {candidate_id}")
    return PerSampleClassificationLoss(CANDIDATE_REGISTRY[candidate_id], class_counts)


def train_subject_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["subject_id"]) for row in rows)
    if not counts:
        raise ValueError("training rows are empty")
    return dict(counts)


def augment_route_consistency_batch(
    inputs: torch.Tensor,
    generator: torch.Generator,
    *,
    magnitude_min: float = 0.9,
    magnitude_max: float = 1.1,
    noise_sigma: float = 0.01,
    route_dropout_probability: float = 0.10,
) -> torch.Tensor:
    if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (4, 3, 28, 28):
        raise ValueError("route-consistency inputs must be [batch,4,3,28,28]")
    if not 0.0 <= route_dropout_probability <= 1.0:
        raise ValueError("route dropout probability must be in [0,1]")
    result = inputs.clone()
    batch_size = len(result)
    factors = torch.empty((batch_size, 1, 1, 1, 1), dtype=result.dtype).uniform_(
        magnitude_min, magnitude_max, generator=generator
    )
    result[:, 0:2] *= factors
    noise = torch.randn(result[:, 0:2].shape, generator=generator, dtype=result.dtype)
    result[:, 0:2] += noise * float(noise_sigma)
    dropout_draws = torch.rand(batch_size, generator=generator)
    route_ids = torch.randint(0, 4, (batch_size,), generator=generator)
    for index in range(batch_size):
        if float(dropout_draws[index]) < route_dropout_probability:
            result[index, int(route_ids[index])] = 0.0
    result[:, 0:2, 0:2].clamp_(-1.0, 1.0)
    result[:, 0:2, 2].clamp_(0.0, 1.0)
    result[:, 2:4].clamp_(0.0, 1.0)
    return result


def symmetric_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_log = F.log_softmax(first_logits, dim=1)
    second_log = F.log_softmax(second_logits, dim=1)
    first = first_log.exp()
    second = second_log.exp()
    first_to_second = F.kl_div(first_log, second, reduction="batchmean")
    second_to_first = F.kl_div(second_log, first, reduction="batchmean")
    return 0.5 * (first_to_second + second_to_first)


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA model state keys drifted")
        for name, value in current.items():
            target = self.shadow[name]
            if torch.is_floating_point(target):
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach())
        self.num_updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, payload: Mapping[str, object], device: torch.device) -> None:
        if float(payload["decay"]) != self.decay:
            raise ValueError("EMA decay drift")
        shadow = payload["shadow"]
        if not isinstance(shadow, Mapping) or shadow.keys() != self.shadow.keys():
            raise ValueError("EMA state keys drift")
        self.shadow = {name: value.to(device) for name, value in shadow.items()}  # type: ignore[union-attr]
        self.num_updates = int(payload["num_updates"])

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


__all__ = [
    "CANDIDATE_REGISTRY",
    "CONSISTENCY_WEIGHT",
    "EMA_DECAY",
    "ExponentialMovingAverage",
    "ME4CandidateContract",
    "PerSampleClassificationLoss",
    "REDUCED_CAUSALNET_CONFIG",
    "SUBJECT_EQUAL_WEIGHT",
    "SUBJECT_SAMPLE_WEIGHT",
    "augment_route_consistency_batch",
    "build_candidate_loss",
    "build_me4_causalnet",
    "symmetric_kl_divergence",
    "train_subject_counts",
]
