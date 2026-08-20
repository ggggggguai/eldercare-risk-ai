"""Inference-only architecture for the frozen V3.5 subject-level model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


V35_MODALITIES = ("audio", "text", "face")
V35_INPUT_DIMS = {"audio": 768, "text": 768, "face": 512}


class V35ResidualTaskEncoder(nn.Module):
    """Exact runtime counterpart of the frozen V3.5 task encoder."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(V35_INPUT_DIMS[name], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for name in V35_MODALITIES
            }
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        quality: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for index, modality in enumerate(V35_MODALITIES):
            value = self.projections[modality](features[modality])
            available = (~missing_mask[..., index]).to(value.dtype)
            weight = quality[..., index].clamp(0.0, 1.0) * available
            projected.append(value * weight.unsqueeze(-1))
            weights.append(weight)
        denominator = torch.stack(weights, dim=-1).sum(dim=-1).clamp_min(1.0)
        task = torch.stack(projected, dim=0).sum(dim=0) / denominator.unsqueeze(-1)
        return self.output_norm(task + self.residual(task))


class V35SubjectAggregator(nn.Module):
    """Exact weighted-logit/P0 V3.5 graph used by the deployment loader."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.aggregator = "weighted_logit"
        self.protocol = "P0"
        self.encoder = V35ResidualTaskEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.task_position = None
        self.task_logit = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 6, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.subject_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.ordinal_head = nn.Linear(hidden_dim, 3)
        self.moca_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _masked_weights(
        scores: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = scores.masked_fill(~task_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked, dim=1)
        return torch.where(task_mask, weights, torch.zeros_like(weights))

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        task = self.encoder(
            batch["features"],
            batch["quality"],
            batch["missing_mask"],
        )
        task_mask = batch["task_mask"].bool()
        gate_features = torch.cat(
            [task, batch["quality"].float(), batch["missing_mask"].float()],
            dim=-1,
        )
        task_logits = self.task_logit(task).squeeze(-1)
        weights = self._masked_weights(
            self.gate(gate_features).squeeze(-1),
            task_mask,
        )
        subject_logit = (task_logits * weights).sum(dim=1)
        subject_representation = (task * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "subject_logit": subject_logit,
            "task_logits": task_logits,
            "task_weights": weights,
            "subject_representation": subject_representation,
            "ordinal_logits": self.ordinal_head(subject_representation),
            "moca_prediction": self.moca_head(subject_representation).squeeze(-1),
        }


def v35_model_state_is_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


__all__ = [
    "V35_INPUT_DIMS",
    "V35_MODALITIES",
    "V35ResidualTaskEncoder",
    "V35SubjectAggregator",
    "v35_model_state_is_finite",
]
