from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .causalnet_preprocess import (
    CAUSALNET_UPSTREAM_COMMIT,
    CAUSALNET_UPSTREAM_LICENSE,
    CAUSALNET_UPSTREAM_REPOSITORY,
)


MODEL_SCHEMA_VERSION = "causalnet_project_adaptation_v1"
ARCHITECTURE_CORRECTION = "preserve_real_2x2_before_cross_attention"


@dataclass(frozen=True)
class CausalNetConfig:
    image_size: int = 28
    patch_size: int = 7
    dim: int = 256
    heads: int = 3
    num_hierarchies: int = 3
    block_repeats: tuple[int, ...] = (3, 3, 9)
    num_classes: int = 3
    gamma: float = 0.5
    mlp_mult: int = 4
    cross_heads: int = 8
    cross_depth: int = 9
    dropout: float = 0.0
    channels: int = 3
    head_hidden_dim: int = 2048

    def __post_init__(self) -> None:
        if self.image_size != 28 or self.patch_size != 7:
            raise ValueError("CausalNet adaptation is frozen at image_size=28 and patch_size=7")
        if self.dim < 8 or self.heads < 1:
            raise ValueError("dim and heads must be positive testable values")
        if self.num_hierarchies != 3 or len(self.block_repeats) != self.num_hierarchies:
            raise ValueError("This architecture requires exactly three hierarchy stages")
        if any(depth < 1 for depth in self.block_repeats):
            raise ValueError("block_repeats must contain positive depths")
        if self.num_classes != 3:
            raise ValueError("SMIC classification is frozen at three classes")
        if self.dim // self.heads < 1 or self.dim // self.cross_heads < 1:
            raise ValueError("attention head counts must leave a positive per-head width")
        if self.cross_depth < 1 or self.mlp_mult < 1:
            raise ValueError("cross_depth and mlp_mult must be positive")
        if self.head_hidden_dim < self.num_classes:
            raise ValueError("head_hidden_dim must be at least num_classes")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


class FeedForward2d(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttention2d(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.inner_dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(self.inner_dim, dim, 1), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        query, key, value = self.to_qkv(x).chunk(3, dim=1)
        tokens = height * width
        query = query.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        key = key.reshape(batch, self.heads, self.head_dim, tokens)
        value = value.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        attention = torch.softmax(torch.matmul(query, key) * self.scale, dim=-1)
        output = torch.matmul(attention, value).transpose(-2, -1).reshape(batch, -1, height, width)
        return self.to_out(output)


class TransformerStage(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(4))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ChannelLayerNorm(dim),
                        SpatialAttention2d(dim, heads, dropout),
                        ChannelLayerNorm(dim),
                        FeedForward2d(dim, mlp_mult, dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[-2] * x.shape[-1]
        position = self.position[:tokens].reshape(1, 1, x.shape[-2], x.shape[-1])
        x = x + position
        for norm1, attention, norm2, feed_forward in self.layers:
            x = x + attention(norm1(x))
            x = x + feed_forward(norm2(x))
        return x


class RouteEncoder(nn.Module):
    """Hierarchical encoder with the project spatial correction.

    The last hierarchy intentionally keeps the 2x2 map instead of collapsing
    it to 1x1 and reshaping channels. Cross attention therefore receives real
    spatial tokens rather than fabricated tokens.
    """

    def __init__(self, config: CausalNetConfig) -> None:
        super().__init__()
        self.dim = config.dim
        patch_dim = config.channels * config.patch_size**2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(patch_dim, config.dim, 1),
            ChannelLayerNorm(config.dim),
        )
        self.stages = nn.ModuleList(
            [
                TransformerStage(config.dim, config.heads, depth, config.mlp_mult, config.dropout)
                for depth in config.block_repeats
            ]
        )
        self.aggregate = nn.Sequential(
            nn.Conv2d(config.dim, config.dim, 3, padding=1),
            ChannelLayerNorm(config.dim),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    @staticmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if (height, width) != (28, 28):
            raise ValueError(f"Route image shape must be [B,3,28,28], got {tuple(x.shape)}")
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches.permute(0, 1, 4, 5, 2, 3).reshape(batch, channels * patch_size**2, height // patch_size, width // patch_size)

    @staticmethod
    def block_view(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch, channels, height, width = x.shape
        if height % block_size or width % block_size:
            raise ValueError("feature map is not divisible by block size")
        view = x.reshape(batch, channels, block_size, height // block_size, block_size, width // block_size)
        view = view.permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, height // block_size, width // block_size)
        return view, (batch, block_size, block_size, channels)

    @staticmethod
    def restore(x: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
        batch, block_height, block_width, channels = shape
        height, width = x.shape[-2:]
        x = x.reshape(batch, block_height, block_width, channels, height, width)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, block_height * height, block_width * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(self.patchify(x, 7))
        for stage_index, stage in enumerate(self.stages):
            block_size = 4 if stage_index == 0 else 2 if stage_index == 1 else 1
            view, shape = self.block_view(x, block_size)
            view = stage(view)
            x = self.restore(view, shape)
            if stage_index == 0:
                x = self.aggregate(x)
            # Stages 1 and 2 intentionally preserve the 2x2 map.
        if x.shape[1:] != (self.dim, 2, 2):
            raise RuntimeError(f"Architecture correction failed: got {tuple(x.shape)}")
        return x


def _neighborhood_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1)), device=device, dtype=dtype)
    return torch.cdist(positions, positions) <= 1.0


def _causal_spatial_weights(scores: torch.Tensor, gamma: float) -> torch.Tensor:
    tokens = scores.shape[-1]
    mask = _neighborhood_mask(scores.device, scores.dtype).view(1, 1, tokens, tokens)
    column = torch.arange(1, tokens + 1, device=scores.device, dtype=scores.dtype)
    penalty = (~mask).to(scores.dtype) * (-gamma * column.view(1, 1, 1, tokens))
    weights = torch.softmax(scores * mask.to(scores.dtype) + penalty, dim=-1)
    return weights * mask.to(weights.dtype)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, gamma: float, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.inner_dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.gamma = gamma
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_mult, dim),
            nn.Dropout(dropout),
        )

    def attend(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = values.shape
        query = self.to_q(values).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        key, value = self.to_kv(context).chunk(2, dim=-1)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = _causal_spatial_weights(scores, self.gamma)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.to_out(output)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_norm = self.norm(left)
        right_norm = self.norm(right)
        left = left + self.attend(left_norm, right_norm)
        right = right + self.attend(right_norm, left_norm)
        return left + self.ff(left), right + self.ff(right)


class CrossTransformer(nn.Module):
    def __init__(self, config: CausalNetConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(config.dim, config.cross_heads, config.gamma, config.mlp_mult, config.dropout) for _ in range(config.cross_depth)]
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = left.shape
        if (channels, height, width) != (self.dim, 2, 2):
            raise ValueError(f"CrossTransformer requires real [B,{self.dim},2,2], got {tuple(left.shape)}")
        left_seq = left.flatten(2).transpose(1, 2)
        right_seq = right.flatten(2).transpose(1, 2)
        for layer in self.layers:
            left_seq, right_seq = layer(left_seq, right_seq)
        return left_seq.transpose(1, 2).reshape(batch, channels, 2, 2), right_seq.transpose(1, 2).reshape(batch, channels, 2, 2)


class CausalAttentionBlock(nn.Module):
    def __init__(self, config: CausalNetConfig) -> None:
        super().__init__()
        dim = config.dim
        heads = config.cross_heads
        self.norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads
        self.inner_dim = heads * self.head_dim
        self.spatial_qkv = nn.Linear(dim, self.inner_dim * 3)
        self.spatial_out = nn.Linear(self.inner_dim, dim)
        self.temporal_q = nn.Linear(dim, self.inner_dim)
        self.temporal_kv = nn.Linear(dim, self.inner_dim * 2)
        self.temporal_out = nn.Linear(self.inner_dim, dim)
        self.crm_q = nn.Linear(dim, self.inner_dim)
        self.crm_k = nn.Linear(dim, self.inner_dim)
        self.crm_v = nn.Linear(dim, self.inner_dim)
        self.crm_out = nn.Linear(self.inner_dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * config.mlp_mult), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim * config.mlp_mult, dim), nn.Dropout(config.dropout))
        self.heads = heads
        self.scale = self.head_dim**-0.5
        self.gamma = config.gamma

    def spatial(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        query, key, value = self.spatial_qkv(x).chunk(3, dim=-1)
        query = query.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = _causal_spatial_weights(scores, self.gamma)
        return self.spatial_out(torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, dim))

    def temporal(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = source.shape
        query = self.temporal_q(target).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        source_key, source_value = self.temporal_kv(source).chunk(2, dim=-1)
        target_key, target_value = self.temporal_kv(target).chunk(2, dim=-1)
        keys = torch.stack((source_key, target_key), dim=2).reshape(batch, tokens, 2, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        values = torch.stack((source_value, target_value), dim=2).reshape(batch, tokens, 2, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        scores = (query.unsqueeze(3) * keys).sum(dim=-1) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return self.temporal_out((weights.unsqueeze(-1) * values).sum(dim=3).transpose(1, 2).reshape(batch, tokens, dim))

    def causal_relation(self, forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = forward.shape
        query = self.crm_q(forward).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        key = self.crm_k(backward).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        value = self.crm_v(forward + backward).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.crm_out(output + forward + backward)

    def stca(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = first + self.spatial(self.norm(first))
        second = second + self.spatial(self.norm(second))
        second = second + self.temporal(first, second)
        return first, second

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_forward, second_forward = self.stca(first, second)
        first_forward = first_forward + self.ff(first_forward)
        second_forward = second_forward + self.ff(second_forward)
        first_backward, second_backward = self.stca(second, first)
        first_backward = first_backward + self.ff(first_backward)
        second_backward = second_backward + self.ff(second_backward)
        long = self.causal_relation(second_forward, second_backward)
        return torch.cat((long, first_forward, first_backward), dim=1).reshape(first.shape[0], 3, 4, first.shape[-1])


class CausalNet(nn.Module):
    def __init__(self, config: CausalNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalNetConfig()
        self.feature_dim = self.config.dim * 3 * 4
        self.primary_encoder = RouteEncoder(self.config)
        self.secondary_encoder = RouteEncoder(self.config)
        self.cross_attention = CrossTransformer(self.config)
        self.causal_attention = CausalAttentionBlock(self.config)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.head_hidden_dim, self.config.num_classes),
        )

    @staticmethod
    def _validate_input(inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (4, 3, 28, 28):
            raise ValueError(f"CausalNet expects [B,4,3,28,28], got {tuple(inputs.shape)}")
        if not torch.isfinite(inputs).all():
            raise ValueError("CausalNet inputs contain NaN or Inf")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        x0 = self.primary_encoder(inputs[:, 0])
        x1 = self.primary_encoder(inputs[:, 1])
        x2 = self.secondary_encoder(inputs[:, 2])
        x3 = self.secondary_encoder(inputs[:, 3])
        x2, x3 = self.cross_attention(x2, x3)
        x0 = x0 + x2
        x1 = x1 + x3
        causal = self.causal_attention(x0.flatten(2).transpose(1, 2), x1.flatten(2).transpose(1, 2))
        return causal.flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        logits = self.classify_features(features)
        if not torch.isfinite(logits).all():
            raise RuntimeError("CausalNet produced NaN or Inf logits")
        return logits

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp_head(features)


def code_hash(path: Path | None = None) -> str:
    path = path or Path(__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(
    model: CausalNet,
    *,
    training_config: Mapping[str, Any],
    source_manifest_hash: str,
    preprocessing_config_hash: str,
    source_code_hash: str,
) -> dict[str, Any]:
    return {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model_config": model.config.as_dict(),
        "training_config": dict(training_config),
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "architecture_correction": ARCHITECTURE_CORRECTION,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
        "source_manifest_hash": source_manifest_hash,
        "preprocessing_config_hash": preprocessing_config_hash,
        "code_hash": source_code_hash,
    }


def save_causalnet_checkpoint(
    path: Path,
    model: CausalNet,
    *,
    training_config: Mapping[str, Any],
    source_manifest_hash: str,
    preprocessing_config_hash: str,
    source_code_hash: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(
        model,
        training_config=training_config,
        source_manifest_hash=source_manifest_hash,
        preprocessing_config_hash=preprocessing_config_hash,
        source_code_hash=source_code_hash,
    )
    torch.save({**metadata, "model_state_dict": model.state_dict()}, path)
    return metadata


def load_causalnet_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[CausalNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "model_schema_version", "model_config", "model_state_dict", "training_config",
        "upstream_repository", "upstream_commit", "upstream_license", "architecture_correction",
        "upstream_bit_exact", "paper_reproduction_claim", "source_manifest_hash",
        "preprocessing_config_hash", "code_hash",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint metadata is incomplete: {sorted(missing)}")
    expected_metadata = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "architecture_correction": ARCHITECTURE_CORRECTION,
    }
    for key, expected in expected_metadata.items():
        if payload[key] != expected:
            raise ValueError(f"Checkpoint {key} does not match the frozen model contract")
    if payload["architecture_correction"] != ARCHITECTURE_CORRECTION:
        raise ValueError("Checkpoint does not declare the project spatial correction")
    if payload["upstream_bit_exact"] or payload["paper_reproduction_claim"]:
        raise ValueError("This adaptation cannot claim upstream bit exactness or paper reproduction")
    model = CausalNet(CausalNetConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
