"""Four-head cognitive-change clue model using frozen cached embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.fusion import (
    FusionOutput,
    QualityAwareFusion,
)


@dataclass(frozen=True)
class CognitiveModelOutput:
    hc_vs_non_hc_logit: Tensor
    mci_vs_hc_logit: Tensor
    ad_mci_hc_logits: Tensor
    moca_standardized: Tensor
    gate_weights: Tensor
    fused: Tensor


class CognitiveChangeClueModel(nn.Module):
    """V3.3 quality-aware fusion and production/research heads."""

    def __init__(
        self,
        *,
        projection_dim: int = 256,
        fusion_hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        input_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        if len(fusion_hidden_dims) != 2:
            raise ValueError("fusion_hidden_dims must contain exactly two dimensions")
        first, second = (int(value) for value in fusion_hidden_dims)
        self.fusion = QualityAwareFusion(
            projection_dim=projection_dim,
            dropout=dropout,
            input_dims=input_dims,
        )
        self.shared = nn.Sequential(
            nn.Linear(int(projection_dim), first, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(first, second, bias=True),
            nn.GELU(),
        )
        self.hc_vs_non_hc_head = nn.Linear(second, 1, bias=True)
        self.mci_vs_hc_head = nn.Linear(second, 1, bias=True)
        self.ad_mci_hc_head = nn.Linear(second, 3, bias=True)
        self.moca_head = nn.Linear(second, 1, bias=True)
        self.shared.apply(_initialize_linear)
        for head in (
            self.hc_vs_non_hc_head,
            self.mci_vs_hc_head,
            self.ad_mci_hc_head,
            self.moca_head,
        ):
            _initialize_linear(head)

    def forward(
        self,
        features: Mapping[str, Tensor],
        quality: Tensor,
        missing_mask: Tensor,
    ) -> CognitiveModelOutput:
        fusion_output: FusionOutput = self.fusion(features, quality, missing_mask)
        representation = self.shared(fusion_output.fused)
        return CognitiveModelOutput(
            hc_vs_non_hc_logit=self.hc_vs_non_hc_head(representation).squeeze(1),
            mci_vs_hc_logit=self.mci_vs_hc_head(representation).squeeze(1),
            ad_mci_hc_logits=self.ad_mci_hc_head(representation),
            moca_standardized=self.moca_head(representation).squeeze(1),
            gate_weights=fusion_output.gate_weights,
            fused=fusion_output.fused,
        )


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def model_state_is_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


__all__ = [
    "CognitiveChangeClueModel",
    "CognitiveModelOutput",
    "model_state_is_finite",
]
