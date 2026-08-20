from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .casme2_apexfusion_dataset import ROUTE_ORDER


MODEL_SCHEMA = "casme2_apexfusion_model_v1"
CHECKPOINT_SCHEMA = "casme2_apexfusion_checkpoint_v1"


@dataclass(frozen=True)
class ApexFusionConfig:
    hidden_dim: int = 64
    temporal: str = "tcn"
    temporal_layers: int = 2
    dropout: float = 0.25
    projection_dim: int = 32
    num_classes: int = 3
    num_aux_classes: int = 4

    def __post_init__(self) -> None:
        if self.temporal not in {"tcn", "gru", "transformer"}:
            raise ValueError("temporal must be tcn, gru, or transformer")
        if self.hidden_dim <= 0 or self.projection_dim <= 0:
            raise ValueError("hidden dimensions must be positive")


class FrameCNN(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, hidden, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).flatten(1)


class TemporalEncoder(nn.Module):
    def __init__(self, config: ApexFusionConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        if config.temporal == "tcn":
            layers: list[nn.Module] = []
            for index in range(config.temporal_layers):
                dilation = 2**index
                layers.extend((nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(config.dropout)))
            self.encoder: nn.Module = nn.Sequential(*layers)
        elif config.temporal == "gru":
            self.encoder = nn.GRU(hidden, hidden, num_layers=config.temporal_layers, batch_first=True, dropout=config.dropout if config.temporal_layers > 1 else 0.0)
        else:
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=config.dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.temporal_layers, enable_nested_tensor=False)
        self.kind = config.temporal

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.kind == "tcn":
            return self.encoder(sequence.transpose(1, 2)).mean(dim=2)
        output = self.encoder(sequence)
        if self.kind == "gru":
            output = output[0]
        return output.mean(dim=1)


class ApexFusionNet(nn.Module):
    def __init__(self, config: ApexFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ApexFusionConfig()
        hidden = self.config.hidden_dim
        self.appearance_cnn = FrameCNN(3, hidden)
        self.motion_cnn = FrameCNN(4, hidden)
        self.roi_cnn = FrameCNN(1, hidden)
        self.landmark_mlp = nn.Sequential(nn.Linear(68 * 6, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(hidden * 2, hidden))
        self.temporal = nn.ModuleDict({route: TemporalEncoder(self.config) for route in ROUTE_ORDER})
        self.route_gate = nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(self.config.dropout))
        self.main_head = nn.Linear(hidden, self.config.num_classes)
        self.aux_head = nn.Linear(hidden, self.config.num_aux_classes)
        self.supcon_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, self.config.projection_dim))

    def _cnn_sequence(self, value: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        batch, time = value.shape[:2]
        encoded = encoder(value.reshape(batch * time, *value.shape[2:]))
        return encoded.reshape(batch, time, -1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        appearance = self._cnn_sequence(batch["appearance"], self.appearance_cnn)
        motion = self._cnn_sequence(batch["motion"], self.motion_cnn)
        roi = batch["local_roi"]
        batch_size, time, regions = roi.shape[:3]
        roi_encoded = self.roi_cnn(roi.reshape(batch_size * time * regions, *roi.shape[3:])).reshape(batch_size, time, regions, -1).mean(dim=2)
        landmark = self.landmark_mlp(batch["landmark"].reshape(batch_size, time, -1))
        sequences = {"appearance": appearance, "motion": motion, "local_roi": roi_encoded, "landmark": landmark}
        route_features = torch.stack([self.temporal[route](sequences[route]) for route in ROUTE_ORDER], dim=1)
        route_mask = batch["route_mask"].bool()
        if route_mask.ndim == 1:
            route_mask = route_mask.unsqueeze(0).expand(batch_size, -1)
        if route_mask.shape != (batch_size, len(ROUTE_ORDER)) or not torch.all(route_mask.any(dim=1)):
            raise ValueError(f"Invalid route mask shape/content: {route_mask.shape}")
        quality = torch.ones((batch_size, len(ROUTE_ORDER), 1), dtype=route_features.dtype, device=route_features.device)
        gate_logits = self.route_gate(torch.cat((route_features, quality), dim=2)).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~route_mask, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=1)
        fused = self.fusion(torch.sum(route_features * weights.unsqueeze(-1), dim=1))
        projection = F.normalize(self.supcon_head(fused), dim=1)
        return {"main_logits": self.main_head(fused), "aux_logits": self.aux_head(fused), "projection": projection, "fused": fused, "route_weights": weights, "route_features": route_features}


def supervised_contrastive_loss(projection: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if projection.ndim != 2 or labels.ndim != 1 or len(projection) != len(labels):
        raise ValueError("Invalid SupCon inputs")
    similarity = projection @ projection.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not torch.any(valid):
        return projection.sum() * 0.0
    masked = similarity.masked_fill(identity, torch.finfo(similarity.dtype).min)
    log_prob = masked - torch.logsumexp(masked, dim=1, keepdim=True)
    mean_positive = (log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1))[valid]
    return -mean_positive.mean()


def multitask_loss(outputs: Mapping[str, torch.Tensor], labels: torch.Tensor, aux_labels: torch.Tensor, *, aux_weight: float = 0.2, supcon_weight: float = 0.05) -> dict[str, torch.Tensor]:
    main = F.cross_entropy(outputs["main_logits"], labels)
    auxiliary = F.cross_entropy(outputs["aux_logits"], aux_labels)
    contrastive = supervised_contrastive_loss(outputs["projection"], aux_labels)
    total = main + aux_weight * auxiliary + supcon_weight * contrastive
    return {"total": total, "main": main, "auxiliary": auxiliary, "supcon": contrastive}


def save_checkpoint(path: Path, model: ApexFusionNet, *, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": CHECKPOINT_SCHEMA, "model_config": asdict(model.config), "state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> tuple[ApexFusionNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported ApexFusion checkpoint schema")
    model = ApexFusionNet(ApexFusionConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, dict(payload.get("metadata", {}))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
