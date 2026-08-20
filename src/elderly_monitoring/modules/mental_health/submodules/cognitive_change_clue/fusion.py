"""Quality-aware multimodal fusion for cognitive-change clues V3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


MODALITY_ORDER = ("audio", "text", "face")
INPUT_DIMS = {"audio": 768, "text": 768, "face": 512}


@dataclass(frozen=True)
class FusionOutput:
    fused: Tensor
    gate_weights: Tensor
    projected: Mapping[str, Tensor]


class QualityAwareFusion(nn.Module):
    """Project modalities and softmax gates only across available branches."""

    def __init__(
        self,
        projection_dim: int = 256,
        dropout: float = 0.2,
        input_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.input_dims = {
            name: int((input_dims or INPUT_DIMS)[name]) for name in MODALITY_ORDER
        }
        if set(self.input_dims) != set(MODALITY_ORDER) or any(
            value <= 0 for value in self.input_dims.values()
        ):
            raise ValueError("input_dims must define positive audio/text/face dimensions")
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.input_dims[name], self.projection_dim, bias=True),
                    nn.LayerNorm(self.projection_dim),
                    nn.GELU(),
                )
                for name in MODALITY_ORDER
            }
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Linear(self.projection_dim + 2, 1, bias=True)
                for name in MODALITY_ORDER
            }
        )
        self.dropout = nn.Dropout(float(dropout))
        self.apply(_initialize_linear)

    def forward(
        self,
        features: Mapping[str, Tensor],
        quality: Tensor,
        missing_mask: Tensor,
    ) -> FusionOutput:
        _validate_inputs(features, quality, missing_mask, input_dims=self.input_dims)
        available = ~missing_mask.bool()
        if not torch.all(available.any(dim=1)):
            raise ValueError("every fusion row must have at least one available modality")

        projected: dict[str, Tensor] = {}
        gate_logits: list[Tensor] = []
        batch_size = quality.shape[0]
        for index, name in enumerate(MODALITY_ORDER):
            branch_available = available[:, index]
            branch_projection = quality.new_zeros((batch_size, self.projection_dim))
            if torch.any(branch_available):
                projection_output = self.projections[name](
                    features[name][branch_available]
                )
                branch_projection[branch_available] = projection_output.to(
                    branch_projection.dtype
                )
            projected[name] = branch_projection

            branch_logit = quality.new_full((batch_size,), float("-inf"))
            if torch.any(branch_available):
                gate_input = torch.cat(
                    (
                        branch_projection[branch_available],
                        quality[branch_available, index : index + 1],
                        available[branch_available, index : index + 1].to(quality.dtype),
                    ),
                    dim=1,
                )
                gate_output = self.gates[name](gate_input).squeeze(1)
                branch_logit[branch_available] = gate_output.to(branch_logit.dtype)
            gate_logits.append(branch_logit)

        gate_weights = torch.softmax(torch.stack(gate_logits, dim=1), dim=1)
        fused = quality.new_zeros((batch_size, self.projection_dim))
        for index, name in enumerate(MODALITY_ORDER):
            fused = fused + gate_weights[:, index : index + 1] * self.dropout(projected[name])
        return FusionOutput(
            fused=fused,
            gate_weights=gate_weights,
            projected=projected,
        )


def _validate_inputs(
    features: Mapping[str, Tensor],
    quality: Tensor,
    missing_mask: Tensor,
    *,
    input_dims: Mapping[str, int] = INPUT_DIMS,
) -> None:
    if quality.ndim != 2 or quality.shape[1] != len(MODALITY_ORDER):
        raise ValueError("quality must have shape [batch, 3]")
    if missing_mask.shape != quality.shape:
        raise ValueError("missing_mask must match quality shape")
    if not torch.isfinite(quality).all():
        raise ValueError("quality contains non-finite values")
    batch_size = quality.shape[0]
    for name in MODALITY_ORDER:
        value = features.get(name)
        expected = (batch_size, int(input_dims[name]))
        if value is None or tuple(value.shape) != expected:
            raise ValueError(f"{name} features must have shape {expected}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} features contain non-finite values")


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "FusionOutput",
    "INPUT_DIMS",
    "MODALITY_ORDER",
    "QualityAwareFusion",
]
