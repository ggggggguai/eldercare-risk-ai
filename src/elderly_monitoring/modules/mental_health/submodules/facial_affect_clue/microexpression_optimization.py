"""Lightweight OPT-ME-001 models and training-only domain alignment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from .causalnet import CausalNet, CausalNetConfig


OPTIMIZATION_MODEL_SCHEMA_VERSION = "opt_me_001_model_v1"
MODEL_TYPES = ("causalnet", "cnn_tcn", "cnn_gru", "temporal_transformer")


@dataclass(frozen=True)
class OptimizationModelConfig:
    model_type: str = "causalnet"
    num_classes: int = 3
    embedding_dim: int = 64
    hidden_dim: int = 96
    layers: int = 1
    heads: int = 4
    dropout: float = 0.1
    classifier_hidden_dim: int = 128
    causalnet: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported optimization model: {self.model_type}")
        if self.num_classes != 3:
            raise ValueError("OPT-ME-001 is frozen at three classes")
        if min(self.embedding_dim, self.hidden_dim, self.layers, self.heads) < 1:
            raise ValueError("Model dimensions and depth must be positive")
        if self.embedding_dim % self.heads and self.model_type == "temporal_transformer":
            raise ValueError("Transformer embedding_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OptimizationModelConfig:
        return cls(**dict(values))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteCNN(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, embedding_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, embedding_dim), embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LightweightTemporalModel(nn.Module):
    def __init__(self, config: OptimizationModelConfig) -> None:
        super().__init__()
        if config.model_type == "causalnet":
            raise ValueError("Use CausalNet directly for model_type=causalnet")
        self.config = config
        self.route_encoder = RouteCNN(config.embedding_dim, config.dropout)
        if config.model_type == "cnn_tcn":
            blocks: list[nn.Module] = []
            in_channels = config.embedding_dim
            for _ in range(config.layers):
                blocks.extend(
                    [
                        nn.Conv1d(in_channels, config.hidden_dim, 3, padding=1),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    ]
                )
                in_channels = config.hidden_dim
            self.temporal = nn.Sequential(*blocks)
            self.feature_dim = config.hidden_dim
        elif config.model_type == "cnn_gru":
            self.temporal = nn.GRU(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=config.layers,
                batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            self.feature_dim = config.hidden_dim
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.position = nn.Parameter(torch.zeros(1, 4, config.embedding_dim))
            nn.init.normal_(self.position, std=0.02)
            self.feature_dim = config.embedding_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    @staticmethod
    def _validate(inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (4, 3, 28, 28):
            raise ValueError(f"Temporal model expects [B,4,3,28,28], got {tuple(inputs.shape)}")
        if not torch.isfinite(inputs).all():
            raise ValueError("Temporal model inputs contain NaN or Inf")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate(inputs)
        batch = inputs.shape[0]
        routes = self.route_encoder(inputs.reshape(batch * 4, 3, 28, 28)).reshape(
            batch, 4, self.config.embedding_dim
        )
        if self.config.model_type == "cnn_tcn":
            return self.temporal(routes.transpose(1, 2)).mean(dim=-1)
        if self.config.model_type == "cnn_gru":
            _, hidden = self.temporal(routes)
            return hidden[-1]
        return self.temporal(routes + self.position).mean(dim=1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.classify_features(self.forward_features(inputs))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Temporal model produced NaN or Inf logits")
        return logits


def build_optimization_model(config: OptimizationModelConfig) -> nn.Module:
    if config.model_type == "causalnet":
        values = dict(config.causalnet)
        values.setdefault("num_classes", config.num_classes)
        return CausalNet(CausalNetConfig(**values))
    return LightweightTemporalModel(config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, gradients: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradients, None


class DomainAdversary(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor, *, reversal_scale: float = 1.0) -> torch.Tensor:
        return self.network(_GradientReverse.apply(features, reversal_scale))


def save_optimization_checkpoint(
    path: Any,
    *,
    model: nn.Module,
    model_config: OptimizationModelConfig,
    training_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    domain_adversary: DomainAdversary | None = None,
) -> None:
    payload = {
        "schema_version": OPTIMIZATION_MODEL_SCHEMA_VERSION,
        "model_config": model_config.as_dict(),
        "training_config": dict(training_config),
        "evidence": dict(evidence),
        "model_state_dict": model.state_dict(),
        "domain_adversary_state_dict": (
            domain_adversary.state_dict() if domain_adversary is not None else None
        ),
    }
    torch.save(payload, path)


def load_optimization_checkpoint(path: Any, *, map_location: Any = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != OPTIMIZATION_MODEL_SCHEMA_VERSION:
        raise ValueError("Unexpected optimization checkpoint schema")
    config = OptimizationModelConfig.from_mapping(payload["model_config"])
    model = build_optimization_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
