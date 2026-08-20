from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .manifold_graph import ManifoldGraphConfig, PaperGraphBuilder
from .paper_dataset import PAPER_MAX_PATCHES_PER_REGION


PAPER_MODEL_SCHEMA_VERSION = "mhssa_tgcn_paper_exact_v2"


@dataclass(frozen=True)
class PaperMHSSATGCNConfig:
    num_regions: int = 6
    max_patches_per_region: int = PAPER_MAX_PATCHES_PER_REGION
    patch_dim: int = 75
    hidden_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    gcn_layers: int = 3
    num_classes: int = 3
    dropout: float = 0.1
    attention_variant: str = "paper_exact"
    spatial_epsilon: float = 1e-6
    graph: ManifoldGraphConfig = field(default_factory=ManifoldGraphConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PaperMHSSATGCNConfig":
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


def reciprocal_distance_weights(
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float,
    variant: str,
) -> torch.Tensor:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [batch, tokens, 2]")
    if valid_mask.shape != coordinates.shape[:2]:
        raise ValueError("valid_mask shape does not match coordinates")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    distances = torch.cdist(coordinates, coordinates, p=2)
    weights = 1.0 / (distances + epsilon)
    pair_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    weights = weights * pair_mask.to(weights.dtype)
    if variant == "paper_exact":
        pass
    elif variant == "stable_variant":
        diagonal = torch.eye(
            weights.shape[-1], dtype=torch.bool, device=weights.device
        ).unsqueeze(0)
        weights = torch.where(diagonal & pair_mask, torch.ones_like(weights), weights)
        row_scale = weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        valid_count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        weights = weights / row_scale * valid_count.to(weights.dtype)
    else:
        raise ValueError(f"Unsupported attention variant: {variant}")
    if not torch.isfinite(weights).all():
        raise RuntimeError("Reciprocal-distance weights contain non-finite values")
    return weights


class PaperMHSSAAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        *,
        epsilon: float,
        variant: str,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.epsilon = epsilon
        self.variant = variant
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, hidden_dim = values.shape
        qkv = self.to_qkv(values).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        spatial = reciprocal_distance_weights(
            coordinates,
            valid_mask,
            epsilon=self.epsilon,
            variant=self.variant,
        )
        scores = scores * spatial.unsqueeze(1)
        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid_mask[:, None, :, None].to(attention.dtype)
        attention = self.attention_dropout(attention)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, tokens, hidden_dim)
        output = self.output_dropout(self.to_output(output))
        output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        if return_attention:
            return output, attention
        return output


class PaperTransformerBlock(nn.Module):
    def __init__(self, config: PaperMHSSATGCNConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = PaperMHSSAAttention(
            config.hidden_dim,
            config.transformer_heads,
            config.dropout,
            epsilon=config.spatial_epsilon,
            variant=config.attention_variant,
        )
        self.feed_forward_norm = nn.LayerNorm(config.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        numeric_mask = valid_mask.unsqueeze(-1).to(values.dtype)
        attention_output = self.attention(
            self.attention_norm(values), valid_mask, coordinates
        )
        values = (values + attention_output) * numeric_mask
        values = (
            values + self.feed_forward(self.feed_forward_norm(values))
        ) * numeric_mask
        return values


class PaperGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(torch.bmm(adjacency, self.linear(values))))


class PaperMHSSATGCN(nn.Module):
    def __init__(
        self,
        au_adjacency: torch.Tensor,
        config: PaperMHSSATGCNConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PaperMHSSATGCNConfig()
        config = self.config
        if config.num_regions != 6:
            raise ValueError("The paper-v2 model requires exactly six regions")
        if config.patch_dim != 75:
            raise ValueError("The SMIC paper-v2 model requires 5 x 5 x 3 patches")
        if config.transformer_heads != 4:
            raise ValueError("The frozen SMIC paper configuration uses four heads")
        self.patch_mlp = nn.Sequential(
            nn.Linear(config.patch_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(
                config.num_regions,
                config.max_patches_per_region,
                config.hidden_dim,
            )
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.transformer_blocks = nn.ModuleList(
            [PaperTransformerBlock(config) for _ in range(config.transformer_layers)]
        )
        self.region_output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.graph_builder = PaperGraphBuilder(au_adjacency, config.graph)
        self.gcn_layers = nn.ModuleList(
            [
                PaperGCNLayer(config.hidden_dim, config.dropout)
                for _ in range(config.gcn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def _validate_inputs(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> None:
        config = self.config
        expected = (
            config.num_regions,
            config.max_patches_per_region,
            config.patch_dim,
        )
        if patches.ndim != 4 or patches.shape[1:] != expected:
            raise ValueError(f"Invalid paper-v2 patches shape: {tuple(patches.shape)}")
        if masks.shape != patches.shape[:3]:
            raise ValueError(f"Invalid paper-v2 mask shape: {tuple(masks.shape)}")
        if keypoints.shape != (*patches.shape[:3], 2):
            raise ValueError(
                f"Invalid paper-v2 keypoint shape: {tuple(keypoints.shape)}"
            )
        if not masks.any(dim=-1).all():
            raise ValueError("Every facial region must contain at least one real patch")
        if not torch.isfinite(patches).all() or not torch.isfinite(keypoints).all():
            raise ValueError("Paper-v2 model inputs contain non-finite values")

    def forward(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor,
        keypoints: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(patches, masks, keypoints)
        config = self.config
        batch_size = patches.shape[0]
        values = self.patch_mlp(patches)
        values = values + self.position_embedding.unsqueeze(0)
        values = values * masks.unsqueeze(-1).to(values.dtype)
        values = values.reshape(
            batch_size * config.num_regions,
            config.max_patches_per_region,
            config.hidden_dim,
        )
        flat_masks = masks.reshape(
            batch_size * config.num_regions, config.max_patches_per_region
        )
        flat_keypoints = keypoints.reshape(
            batch_size * config.num_regions,
            config.max_patches_per_region,
            2,
        )
        for block in self.transformer_blocks:
            values = block(values, flat_masks, flat_keypoints)
        numeric_mask = flat_masks.unsqueeze(-1).to(values.dtype)
        pooled = (values * numeric_mask).sum(dim=1) / numeric_mask.sum(
            dim=1
        ).clamp_min(1.0)
        region_features = self.region_output(pooled).reshape(
            batch_size, config.num_regions, config.hidden_dim
        )
        adjacency, graph_details = self.graph_builder(region_features)
        graph_features = region_features
        for layer in self.gcn_layers:
            graph_features = layer(graph_features, adjacency)
        logits = self.classifier(graph_features.mean(dim=1))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Paper-v2 model produced non-finite logits")
        if return_aux:
            return logits, {
                **graph_details,
                "region_features": region_features,
                "graph_features": graph_features,
            }
        return logits


def load_paper_model_checkpoint(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PaperMHSSATGCN, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if checkpoint.get("model_schema_version") != PAPER_MODEL_SCHEMA_VERSION:
        raise ValueError("Checkpoint is not a paper-v2 MHSSA-TGCN checkpoint")
    config = PaperMHSSATGCNConfig.from_mapping(checkpoint["model_config"])
    au_adjacency = torch.as_tensor(checkpoint["au_adjacency"], dtype=torch.float32)
    model = PaperMHSSATGCN(au_adjacency, config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint
