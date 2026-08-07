from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


MODEL_SCHEMA_VERSION = "mhssa_tgcn_v1"


@dataclass(frozen=True)
class MHSSATGCNConfig:
    num_regions: int = 6
    max_patches_per_region: int = 12
    patch_dim: int = 147
    hidden_dim: int = 256
    region_embedding_dim: int = 32
    intra_depth: int = 2
    intra_heads: int = 4
    inter_depth: int = 1
    inter_heads: int = 4
    gcn_layers: int = 2
    num_classes: int = 3
    dropout: float = 0.15
    spatial_temperature: float = 8.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MHSSATGCNConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_region_prior() -> torch.Tensor:
    """Symmetric six-region geometry/AU prior with self loops."""
    adjacency = torch.eye(6, dtype=torch.float32)
    edges = (
        (0, 1, 0.25),
        (0, 2, 1.00),
        (0, 4, 0.75),
        (1, 3, 1.00),
        (1, 4, 0.75),
        (2, 3, 0.40),
        (2, 4, 0.85),
        (3, 4, 0.85),
        (4, 5, 1.00),
        (2, 5, 0.25),
        (3, 5, 0.25),
    )
    for left, right, weight in edges:
        adjacency[left, right] = weight
        adjacency[right, left] = weight
    return adjacency


class MaskedSpatialAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        spatial_temperature: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.spatial_temperature = spatial_temperature
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, hidden = values.shape
        qkv = self.to_qkv(values).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        distances = torch.cdist(coordinates, coordinates, p=2)
        spatial_bias = -distances / max(self.spatial_temperature, 1e-6)
        scores = scores + spatial_bias.unsqueeze(1)

        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)
        query_mask = valid_mask[:, None, :, None].to(attention.dtype)
        attention = attention * query_mask

        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, tokens, hidden)
        output = self.output_dropout(self.to_out(output))
        return output * valid_mask.unsqueeze(-1).to(output.dtype)


class MaskedTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        spatial_temperature: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MaskedSpatialAttention(
            hidden_dim, heads, dropout, spatial_temperature
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        numeric_mask = valid_mask.unsqueeze(-1).to(values.dtype)
        values = (
            values
            + self.attention(self.attention_norm(values), valid_mask, coordinates)
        ) * numeric_mask
        values = (values + self.feed_forward(self.feed_forward_norm(values))) * numeric_mask
        return values


class ResidualGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        update = torch.bmm(adjacency, self.linear(values))
        update = self.dropout(F.gelu(update))
        return self.norm(values + update)


class MHSSATGCN(nn.Module):
    def __init__(
        self,
        config: MHSSATGCNConfig | None = None,
        region_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MHSSATGCNConfig()
        config = self.config
        if config.num_regions != 6:
            raise ValueError("The v1 region prior requires exactly six regions")
        self.patch_embedding = nn.Linear(config.patch_dim, config.hidden_dim)
        self.region_embedding = nn.Embedding(
            config.num_regions, config.region_embedding_dim
        )
        self.merge_embedding = nn.Linear(
            config.hidden_dim + config.region_embedding_dim, config.hidden_dim
        )
        self.intra_blocks = nn.ModuleList(
            [
                MaskedTransformerBlock(
                    config.hidden_dim,
                    config.intra_heads,
                    config.dropout,
                    config.spatial_temperature,
                )
                for _ in range(config.intra_depth)
            ]
        )
        if config.inter_depth > 0:
            inter_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.inter_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.inter_transformer: nn.Module | None = nn.TransformerEncoder(
                inter_layer,
                num_layers=config.inter_depth,
                enable_nested_tensor=False,
            )
        else:
            self.inter_transformer = None
        self.graph_layers = nn.ModuleList(
            [ResidualGraphLayer(config.hidden_dim, config.dropout) for _ in range(config.gcn_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
        prior = region_prior if region_prior is not None else default_region_prior()
        if prior.shape != (config.num_regions, config.num_regions):
            raise ValueError(f"Invalid region prior shape: {tuple(prior.shape)}")
        self.register_buffer("region_prior", prior.float())
        self.adjacency_mix_logit = nn.Parameter(torch.tensor(0.0))

    def _validate_inputs(
        self,
        patches: torch.Tensor,
        region_ids: torch.Tensor,
        masks: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> None:
        config = self.config
        expected_tokens = config.num_regions * config.max_patches_per_region
        if patches.ndim != 3 or patches.shape[1:] != (expected_tokens, config.patch_dim):
            raise ValueError(f"Invalid patches shape: {tuple(patches.shape)}")
        if region_ids.shape != patches.shape[:2]:
            raise ValueError(f"Invalid region_ids shape: {tuple(region_ids.shape)}")
        if masks.shape != patches.shape[:2]:
            raise ValueError(f"Invalid masks shape: {tuple(masks.shape)}")
        if keypoints.shape != (*patches.shape[:2], 2):
            raise ValueError(f"Invalid keypoints shape: {tuple(keypoints.shape)}")
        if not torch.isfinite(patches).all() or not torch.isfinite(keypoints).all():
            raise ValueError("Inputs contain non-finite values")

    def build_adjacency(self, region_features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(region_features, p=2, dim=-1, eps=1e-6)
        feature_adjacency = torch.bmm(normalized, normalized.transpose(1, 2))
        feature_adjacency = (feature_adjacency + 1.0) * 0.5
        prior = self.region_prior.unsqueeze(0).expand(region_features.shape[0], -1, -1)
        mix = torch.sigmoid(self.adjacency_mix_logit)
        adjacency = mix * feature_adjacency + (1.0 - mix) * prior
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        identity = torch.eye(
            self.config.num_regions,
            device=adjacency.device,
            dtype=adjacency.dtype,
        ).unsqueeze(0)
        adjacency = adjacency + identity
        degrees = adjacency.sum(dim=-1).clamp_min(1e-6)
        inverse_sqrt = degrees.rsqrt()
        return inverse_sqrt.unsqueeze(-1) * adjacency * inverse_sqrt.unsqueeze(-2)

    def forward(
        self,
        patches: torch.Tensor,
        region_ids: torch.Tensor,
        masks: torch.Tensor,
        keypoints: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(patches, region_ids, masks, keypoints)
        config = self.config
        batch_size = patches.shape[0]
        valid_mask = masks > 0.5
        patch_values = self.patch_embedding(patches)
        region_values = self.region_embedding(region_ids)
        values = self.merge_embedding(torch.cat((patch_values, region_values), dim=-1))
        values = values * valid_mask.unsqueeze(-1).to(values.dtype)

        values = values.reshape(
            batch_size * config.num_regions,
            config.max_patches_per_region,
            config.hidden_dim,
        )
        region_mask = valid_mask.reshape(
            batch_size * config.num_regions, config.max_patches_per_region
        )
        coordinates = keypoints.reshape(
            batch_size * config.num_regions, config.max_patches_per_region, 2
        )
        for block in self.intra_blocks:
            values = block(values, region_mask, coordinates)
        numeric_mask = region_mask.unsqueeze(-1).to(values.dtype)
        pooled = (values * numeric_mask).sum(dim=1) / numeric_mask.sum(dim=1).clamp_min(1.0)
        region_features = pooled.reshape(
            batch_size, config.num_regions, config.hidden_dim
        )
        if self.inter_transformer is not None:
            region_features = self.inter_transformer(region_features)
        adjacency = self.build_adjacency(region_features)
        graph_features = region_features
        for layer in self.graph_layers:
            graph_features = layer(graph_features, adjacency)
        logits = self.classifier(graph_features.mean(dim=1))
        if return_aux:
            return logits, {
                "adjacency": adjacency,
                "region_features": region_features,
                "graph_features": graph_features,
            }
        return logits


def load_model_checkpoint(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MHSSATGCN, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = MHSSATGCNConfig.from_mapping(checkpoint["model_config"])
    model = MHSSATGCN(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint
