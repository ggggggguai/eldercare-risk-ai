from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .manifold_graph import ManifoldGraphConfig, PaperGraphBuilder


DSTM_MODEL_SCHEMA_VERSION = "dstm_paper_chapter4_v2"


@dataclass(frozen=True)
class DSTMConfig:
    num_regions: int = 6
    max_patches_per_region: int = 12
    patch_dim: int = 75
    hidden_dim: int = 256
    temporal_cnn_feature_dim: int = 512
    temporal_reducer_dim: int = 64
    transformer_layers: int = 3
    transformer_heads: int = 4
    gcn_layers: int = 2
    temporal_neighbor_k: int = 10
    temporal_neighbor_bias: float = 0.3
    dropout: float = 0.1
    num_classes: int = 3
    alignment_temperature: float = 0.07
    alignment_lambda: float = 0.1
    attention_variant: str = "paper_exact"
    spatial_epsilon: float = 1e-6
    freeze_temporal_cnn: bool = True
    graph: ManifoldGraphConfig = field(default_factory=ManifoldGraphConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DSTMConfig":
        copied = dict(values)
        graph_values = copied.pop("graph", {})
        known = {item.name for item in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in copied.items() if key in known}
        filtered["graph"] = ManifoldGraphConfig(
            **{
                key: value
                for key, value in dict(graph_values).items()
                if key in ManifoldGraphConfig.__dataclass_fields__
            }
        )
        return cls(**filtered)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.num_regions != 6 or self.patch_dim != 75:
            raise ValueError("DSTM SMIC paper-v2 spatial input is fixed to six regions and 75-dimensional patches")
        if self.transformer_heads < 1 or self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.temporal_reducer_dim < 2:
            raise ValueError("temporal_reducer_dim must be at least two")
        if self.temporal_neighbor_k < 1 or self.temporal_neighbor_bias < 0:
            raise ValueError("Temporal neighbor settings are invalid")
        if not 0 < self.alignment_temperature:
            raise ValueError("alignment_temperature must be positive")


class DSTMTemporalMotionCNN(nn.Module):
    """Shared 2D CNN used to encode each optical-flow frame."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(32 * 4 * 4, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1:] != (3, 32, 32):
            raise ValueError(f"DSTM motion CNN expects [N,3,32,32], got {tuple(frames.shape)}")
        output = self.projection(self.encoder(frames))
        if not torch.isfinite(output).all():
            raise RuntimeError("DSTM motion CNN produced non-finite values")
        return output


def temporal_neighbor_bias(
    manifold_features: torch.Tensor,
    temporal_mask: torch.Tensor,
    *,
    neighbor_k: int,
    bias_value: float,
) -> torch.Tensor:
    if manifold_features.ndim != 3:
        raise ValueError("manifold_features must have shape [batch, time, dimension]")
    if temporal_mask.ndim != 2:
        raise ValueError("temporal_mask must have shape [batch, time]")
    if manifold_features.shape[:2] != temporal_mask.shape:
        raise ValueError("manifold feature and temporal mask shapes differ")
    batch, time = temporal_mask.shape
    bias = torch.zeros(
        batch,
        time,
        time,
        device=manifold_features.device,
        dtype=manifold_features.dtype,
    )
    for batch_index in range(batch):
        valid_indices = torch.nonzero(
            temporal_mask[batch_index], as_tuple=False
        ).squeeze(-1)
        valid_count = int(valid_indices.numel())
        if valid_count <= 1:
            continue
        values = manifold_features[batch_index, valid_indices]
        distances = torch.cdist(values, values)
        distances.fill_diagonal_(torch.inf)
        k = min(neighbor_k, valid_count - 1)
        neighbors = torch.topk(distances, k=k, dim=-1, largest=False).indices
        queries = valid_indices[:, None].expand_as(neighbors)
        keys = valid_indices[neighbors]
        bias[batch_index, queries, keys] = float(bias_value)
    valid_query = temporal_mask.unsqueeze(-1)
    valid_key = temporal_mask.unsqueeze(-2)
    bias = bias.masked_fill(~valid_key, torch.finfo(bias.dtype).min)
    bias = bias.masked_fill(~valid_query, 0.0)
    return bias


class DSTMGraphConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(torch.bmm(adjacency, self.linear(values))))


class DSTMTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("Temporal hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        temporal_mask: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, hidden_dim = values.shape
        qkv = self.to_qkv(values).reshape(batch, time, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + bias.unsqueeze(1)
        scores = scores.masked_fill(
            ~temporal_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        attention = attention * temporal_mask[:, None, :, None].to(attention.dtype)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, time, hidden_dim)
        output = self.dropout(self.output(output))
        return output * temporal_mask.unsqueeze(-1).to(output.dtype)


class DSTMTemporalBlock(nn.Module):
    def __init__(self, config: DSTMConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.hidden_dim)
        self.attention = DSTMTemporalAttention(
            config.hidden_dim, config.transformer_heads, config.dropout
        )
        self.norm_ffn = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        temporal_mask: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        numeric_mask = temporal_mask.unsqueeze(-1).to(values.dtype)
        values = values + self.attention(self.norm_attention(values), temporal_mask, bias)
        values = values * numeric_mask
        values = values + self.ffn(self.norm_ffn(values))
        return values * numeric_mask


def symmetric_info_nce(
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if spatial.ndim != 2 or temporal.shape != spatial.shape:
        raise ValueError("InfoNCE inputs must have matching [batch, dimension] shapes")
    if spatial.shape[0] < 2:
        # A one-sample batch has no negative pair.  Returning a finite zero
        # keeps the last partial batch usable without inventing a negative.
        return spatial.sum() * 0.0
    spatial = F.normalize(spatial, dim=-1)
    temporal = F.normalize(temporal, dim=-1)
    logits = torch.matmul(spatial, temporal.transpose(0, 1)) / temperature
    targets = torch.arange(spatial.shape[0], device=spatial.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
    )


class DSTM(nn.Module):
    """Paper chapter-4 DSTM with an explicit fold-only temporal reducer boundary."""

    def __init__(
        self,
        au_adjacency: torch.Tensor,
        config: DSTMConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DSTMConfig()
        config = self.config
        if au_adjacency.shape != (6, 6):
            raise ValueError("DSTM AU adjacency must be [6,6]")
        self.spatial_patch = nn.Sequential(
            nn.Linear(config.patch_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(config.num_regions, config.max_patches_per_region, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        self.graph_builder = PaperGraphBuilder(au_adjacency, config.graph)
        self.spatial_gcn = nn.ModuleList(
            [DSTMGraphConv(config.hidden_dim, config.dropout) for _ in range(config.gcn_layers)]
        )
        self.spatial_attention = nn.Linear(config.hidden_dim, 1)

        self.temporal_motion_cnn = DSTMTemporalMotionCNN(config.temporal_cnn_feature_dim)
        if config.freeze_temporal_cnn:
            for parameter in self.temporal_motion_cnn.parameters():
                parameter.requires_grad_(False)
        self.temporal_projection = nn.Linear(config.temporal_reducer_dim, config.hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            [DSTMTemporalBlock(config) for _ in range(config.transformer_layers)]
        )

        self.spatial_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()
        )
        self.temporal_fusion_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.Sigmoid()
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def encode_motion_frames(self, flow_sequences: torch.Tensor) -> torch.Tensor:
        if flow_sequences.ndim != 5 or flow_sequences.shape[2:] != (3, 32, 32):
            raise ValueError(
                f"DSTM flow sequences must be [batch,time,3,32,32], got {tuple(flow_sequences.shape)}"
            )
        batch, time = flow_sequences.shape[:2]
        flat = flow_sequences.reshape(batch * time, 3, 32, 32)
        encoded = self.temporal_motion_cnn(flat)
        return encoded.reshape(batch, time, -1)

    @staticmethod
    def _transform_with_reducer(
        encoded: torch.Tensor,
        temporal_mask: torch.Tensor,
        reducer: Any,
    ) -> torch.Tensor:
        if not hasattr(reducer, "transform") or getattr(reducer, "fitted", True) is False:
            raise ValueError("DSTM requires a fitted reducer exposing transform only")
        batch, time, feature_dim = encoded.shape
        result = np.zeros((batch * time, int(reducer.output_dim)), dtype=np.float32)
        valid = temporal_mask.reshape(-1).detach().cpu().numpy().astype(bool)
        values = encoded.detach().cpu().numpy().reshape(batch * time, feature_dim)
        if valid.any():
            result[valid] = reducer.transform(values[valid])
        return torch.from_numpy(result.reshape(batch, time, -1)).to(
            device=encoded.device, dtype=encoded.dtype
        )

    def spatial_stream(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (
            self.config.num_regions,
            self.config.max_patches_per_region,
            self.config.patch_dim,
        )
        if patches.ndim != 4 or patches.shape[1:] != expected:
            raise ValueError(f"Invalid DSTM spatial patches shape: {tuple(patches.shape)}")
        if masks.shape != patches.shape[:3] or not masks.any(dim=-1).all():
            raise ValueError("Invalid DSTM spatial mask")
        values = self.spatial_patch(patches)
        values = values + self.spatial_position.unsqueeze(0)
        values = values * masks.unsqueeze(-1).to(values.dtype)
        numeric_mask = masks.unsqueeze(-1).to(values.dtype)
        pooled = (values * numeric_mask).sum(dim=2) / numeric_mask.sum(dim=2).clamp_min(1.0)
        adjacency, details = self.graph_builder(pooled)
        graph_values = pooled
        for layer in self.spatial_gcn:
            graph_values = layer(graph_values, adjacency)
        scores = self.spatial_attention(graph_values).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        spatial = torch.sum(graph_values * weights.unsqueeze(-1), dim=1)
        details = {**details, "spatial_nodes": graph_values, "spatial_attention": weights}
        return spatial, details

    def temporal_stream(
        self,
        temporal_features: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if temporal_features.ndim != 3 or temporal_features.shape[-1] != self.config.temporal_reducer_dim:
            raise ValueError(
                "DSTM temporal features must have shape [batch,time,temporal_reducer_dim]"
            )
        if temporal_mask.shape != temporal_features.shape[:2]:
            raise ValueError("DSTM temporal mask shape mismatch")
        if not temporal_mask.any(dim=-1).all():
            raise ValueError("Every DSTM temporal sequence must contain a valid frame")
        values = self.temporal_projection(temporal_features)
        positions = _sinusoidal_positions(
            values.shape[1], values.shape[-1], values.device, values.dtype
        )
        values = (values + positions.unsqueeze(0)) * temporal_mask.unsqueeze(-1).to(values.dtype)
        bias = temporal_neighbor_bias(
            temporal_features,
            temporal_mask,
            neighbor_k=self.config.temporal_neighbor_k,
            bias_value=self.config.temporal_neighbor_bias,
        )
        for block in self.temporal_blocks:
            values = block(values, temporal_mask, bias)
        numeric_mask = temporal_mask.unsqueeze(-1).to(values.dtype)
        temporal = (values * numeric_mask).sum(dim=1) / numeric_mask.sum(dim=1).clamp_min(1.0)
        return temporal, {"temporal_encoded": values, "temporal_bias": bias}

    def forward(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor,
        keypoints: torch.Tensor | None = None,
        *,
        temporal_features: torch.Tensor | None = None,
        flow_sequences: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        temporal_reducer: Any | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del keypoints  # Coordinates are already encoded by the paper-v2 patch contract.
        if temporal_mask is None:
            if temporal_features is not None:
                temporal_mask = torch.ones(
                    temporal_features.shape[:2], dtype=torch.bool, device=temporal_features.device
                )
            elif flow_sequences is not None:
                temporal_mask = torch.ones(
                    flow_sequences.shape[:2], dtype=torch.bool, device=flow_sequences.device
                )
            else:
                raise ValueError("DSTM requires temporal_features or flow_sequences")
        if temporal_features is None:
            if flow_sequences is None or temporal_reducer is None:
                raise ValueError("Flow input requires a fitted temporal_reducer")
            encoded = self.encode_motion_frames(flow_sequences)
            temporal_features = self._transform_with_reducer(
                encoded, temporal_mask, temporal_reducer
            )
        spatial, spatial_aux = self.spatial_stream(patches, masks)
        temporal, temporal_aux = self.temporal_stream(temporal_features, temporal_mask)
        projected_spatial = self.spatial_projection(spatial)
        projected_temporal = self.temporal_fusion_projection(temporal)
        alignment_loss = symmetric_info_nce(
            projected_spatial,
            projected_temporal,
            temperature=self.config.alignment_temperature,
        )
        gate = self.gate(torch.cat((projected_spatial, projected_temporal), dim=-1))
        fused = gate * projected_spatial + (1.0 - gate) * projected_temporal
        logits = self.classifier(fused)
        if not torch.isfinite(logits).all() or not torch.isfinite(alignment_loss):
            raise RuntimeError("DSTM produced non-finite outputs")
        if return_aux:
            return logits, {
                **spatial_aux,
                **temporal_aux,
                "spatial_feature": projected_spatial,
                "temporal_feature": projected_temporal,
                "alignment_loss": alignment_loss,
                "gate": gate,
                "fused_feature": fused,
            }
        return logits


def _sinusoidal_positions(
    length: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / hidden_dim)
    )
    result = torch.zeros(length, hidden_dim, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[: result[:, 1::2].shape[1]])
    return result


def state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_dstm_checkpoint(
    path: Path,
    *,
    model: DSTM,
    training_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    from hashlib import sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_schema_version": DSTM_MODEL_SCHEMA_VERSION,
        "model_config": model.config.as_dict(),
        "training_config": dict(training_config),
        "au_adjacency": model.graph_builder.au_adjacency.detach().cpu(),
        "model_state_dict": state_dict_on_cpu(model),
        **dict(metadata),
    }
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    torch.save(checkpoint, temporary)
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dstm_checkpoint(
    path: str | Path,
    *,
    au_adjacency: torch.Tensor | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[DSTM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("model_schema_version") != DSTM_MODEL_SCHEMA_VERSION:
        raise ValueError("Checkpoint is not a MODEL-ME-006 DSTM checkpoint")
    config = DSTMConfig.from_mapping(checkpoint["model_config"])
    if au_adjacency is None:
        au_adjacency = torch.as_tensor(checkpoint["au_adjacency"], dtype=torch.float32)
    model = DSTM(au_adjacency, config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint
