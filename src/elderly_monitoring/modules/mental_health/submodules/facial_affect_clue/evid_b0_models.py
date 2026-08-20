from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .causalnet import CausalNet, ChannelLayerNorm
from .causalnet_opt_me_003 import REDUCED_CAUSALNET_CONFIG


class SmallFourRouteCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != (4, 3, 28, 28):
            raise ValueError("SmallFourRouteCNN expects [B,4,3,28,28]")
        return self.classifier(self.features(inputs.flatten(1, 2)).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, source: int, target: int, stride: int = 1) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(source, target, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(target), nn.ReLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(target, target, 3, padding=1, bias=False), nn.BatchNorm2d(target),
        )
        self.shortcut = (
            nn.Identity() if source == target and stride == 1 else
            nn.Sequential(nn.Conv2d(source, target, 1, stride=stride, bias=False), nn.BatchNorm2d(target))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.second(self.first(inputs)) + self.shortcut(inputs), inplace=True)


class MatchedResidualNetwork(nn.Module):
    def __init__(self, width: int = 30) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(width, width), ResidualBlock(width, width),
            ResidualBlock(width, width * 2, 2), ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, 2), ResidualBlock(width * 4, width * 4),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != (4, 3, 28, 28):
            raise ValueError("MatchedResidualNetwork expects [B,4,3,28,28]")
        return self.head(self.body(self.stem(inputs.flatten(1, 2))))


class MHSSAComparisonAdapter(nn.Module):
    """Project adapter: route/patch self-attention, not an exact paper reproduction."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 7, stride=7)
        self.route_embedding = nn.Parameter(torch.zeros(1, 4, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, 1, 16, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        nn.init.normal_(self.route_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        tokens = self.patch(inputs.reshape(batch * 4, 3, 28, 28)).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, 4, 16, -1) + self.route_embedding + self.position
        return self.head(self.encoder(tokens.flatten(1, 2)).mean(dim=1))


class DSTMComparisonAdapter(nn.Module):
    """Project dual-spatial-temporal adapter over the frozen four-route tensor."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.route_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, dim, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(dim, dim, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.attention = nn.MultiheadAttention(dim * 2, 4, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        encoded = self.route_encoder(inputs.reshape(batch * 4, 3, 28, 28)).flatten(1).reshape(batch, 4, -1)
        temporal, _ = self.temporal(encoded)
        attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
        return self.head(attended.mean(dim=1))


class SimpleConcatCausalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = CausalNet(REDUCED_CAUSALNET_CONFIG)
        self.primary_encoder = base.primary_encoder
        self.secondary_encoder = base.secondary_encoder
        dim = REDUCED_CAUSALNET_CONFIG.dim
        self.head = nn.Sequential(nn.Linear(dim * 4 * 4, 256), nn.GELU(), nn.Linear(256, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [
            self.primary_encoder(inputs[:, 0]), self.primary_encoder(inputs[:, 1]),
            self.secondary_encoder(inputs[:, 2]), self.secondary_encoder(inputs[:, 3]),
        ]
        return self.head(torch.cat([item.flatten(1) for item in features], dim=1))


class NoCrossCausalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = CausalNet(REDUCED_CAUSALNET_CONFIG)
        self.primary_encoder = base.primary_encoder
        self.secondary_encoder = base.secondary_encoder
        self.causal_attention = base.causal_attention
        self.mlp_head = base.mlp_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x0 = self.primary_encoder(inputs[:, 0]) + self.secondary_encoder(inputs[:, 2])
        x1 = self.primary_encoder(inputs[:, 1]) + self.secondary_encoder(inputs[:, 3])
        fused = self.causal_attention(x0.flatten(2).transpose(1, 2), x1.flatten(2).transpose(1, 2))
        return self.mlp_head(fused.flatten(1))


class NoCausalCausalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = CausalNet(REDUCED_CAUSALNET_CONFIG)
        self.primary_encoder = base.primary_encoder
        self.secondary_encoder = base.secondary_encoder
        self.cross_attention = base.cross_attention
        dim = REDUCED_CAUSALNET_CONFIG.dim
        self.head = nn.Sequential(nn.Linear(dim * 2 * 4, 256), nn.GELU(), nn.Linear(256, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x0, x1 = self.primary_encoder(inputs[:, 0]), self.primary_encoder(inputs[:, 1])
        x2, x3 = self.secondary_encoder(inputs[:, 2]), self.secondary_encoder(inputs[:, 3])
        x2, x3 = self.cross_attention(x2, x3)
        return self.head(torch.cat(((x0 + x2).flatten(1), (x1 + x3).flatten(1)), dim=1))


class InputTransformModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        retained_routes: Sequence[int] = (0, 1, 2, 3),
        retain_strain: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.retained_routes = tuple(int(value) for value in retained_routes)
        self.retain_strain = bool(retain_strain)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = inputs.clone()
        for route in range(4):
            if route not in self.retained_routes:
                transformed[:, route] = 0.0
        if not self.retain_strain:
            transformed[:, 0:2, 2] = 0.0
        return self.model(transformed)


class LDAMNoDRWLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int], maximum_margin: float = 0.5, scale: float = 30.0) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        margins = counts.pow(-0.25)
        self.register_buffer("margins", margins / margins.max() * maximum_margin)
        self.scale = float(scale)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adjusted = logits.clone()
        adjusted[torch.arange(len(labels), device=labels.device), labels] -= self.margins.to(logits)[labels]
        return F.cross_entropy(adjusted * self.scale, labels)


@dataclass(frozen=True)
class EvidenceModelSpec:
    method_id: str
    architecture: str
    loss_kind: str
    retained_routes: tuple[int, ...] = (0, 1, 2, 3)
    retain_strain: bool = True


DEEP_SPECS: tuple[EvidenceModelSpec, ...] = (
    EvidenceModelSpec("CMP-3-small-cnn", "small_cnn", "cross_entropy"),
    EvidenceModelSpec("CMP-4-matched-resnet", "matched_resnet", "cross_entropy"),
    EvidenceModelSpec("CMP-5-causalnet-ce", "causalnet", "cross_entropy"),
    EvidenceModelSpec("CMP-7-mhssa-tgcn", "mhssa_adapter", "cross_entropy"),
    EvidenceModelSpec("CMP-8-dstm", "dstm_adapter", "cross_entropy"),
    EvidenceModelSpec("ABL-I1-oa-flow", "causalnet", "ldam_drw", (0,)),
    EvidenceModelSpec("ABL-I2-oa-ao-flow", "causalnet", "ldam_drw", (0, 1)),
    EvidenceModelSpec("ABL-F1-simple-concat", "simple_concat", "ldam_drw"),
    EvidenceModelSpec("ABL-F2-no-cross", "no_cross", "ldam_drw"),
    EvidenceModelSpec("ABL-F3-no-causal", "no_causal", "ldam_drw"),
    EvidenceModelSpec("ABL-L2-ldam", "causalnet", "ldam"),
    EvidenceModelSpec("ABL-P2-no-optical-strain", "causalnet", "ldam_drw", (0, 1, 2, 3), False),
)


def build_evidence_model(spec: EvidenceModelSpec) -> nn.Module:
    builders = {
        "small_cnn": SmallFourRouteCNN,
        "matched_resnet": MatchedResidualNetwork,
        "causalnet": lambda: CausalNet(REDUCED_CAUSALNET_CONFIG),
        "mhssa_adapter": MHSSAComparisonAdapter,
        "dstm_adapter": DSTMComparisonAdapter,
        "simple_concat": SimpleConcatCausalNet,
        "no_cross": NoCrossCausalNet,
        "no_causal": NoCausalCausalNet,
    }
    if spec.architecture not in builders:
        raise ValueError(f"unknown evidence architecture: {spec.architecture}")
    return InputTransformModel(
        builders[spec.architecture](),
        retained_routes=spec.retained_routes,
        retain_strain=spec.retain_strain,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "DEEP_SPECS", "EvidenceModelSpec", "InputTransformModel", "LDAMNoDRWLoss",
    "build_evidence_model", "parameter_count",
]
