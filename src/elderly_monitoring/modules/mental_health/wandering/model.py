"""Pure forward contract for the step-9 TopoWander-MPT architecture.

This module is deliberately isolated from project datasets, artifacts, runtime
entrypoints, and the shared mental-health pipeline.  It validates synthetic or
caller-provided tensors, implements the frozen forward mathematics, and offers
only a deterministic numeric state round trip guarded by an external config
hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml


TOPOWANDER_PARAMETER_COUNT = 204466
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "wandering-topowander-mpt-forward-config-v1",
    "purpose": "architecture_forward_contract_only",
    "input": {
        "sequence_length": 80,
        "model_feature_channels": 14,
        "shape_point_channels": 2,
        "dtype": "float32",
        "mask_dtype": "float32",
        "mask_channel_index": 12,
        "time_channel_indices": [10, 11],
        "quality_channel_index": 13,
        "temporal_features_enabled": False,
        "minimum_sample_valid_points": 8,
        "require_same_device": True,
        "require_finite": True,
        "masked_non_mask_values_invariant": True,
    },
    "tcn": {
        "input_projection": {"in_channels": 14, "out_channels": 64, "kernel_size": 1, "bias": True},
        "convolution": "noncausal_same_padding",
        "layer_norm": {"eps": 1.0e-5, "elementwise_affine": True},
        "activation": "gelu",
        "blocks": [
            {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "dilation": 1},
            {"in_channels": 64, "out_channels": 96, "kernel_size": 3, "dilation": 2},
            {"in_channels": 96, "out_channels": 96, "kernel_size": 3, "dilation": 4},
        ],
        "depthwise_bias": True,
        "pointwise_bias": True,
        "channel_change_residual_bias": False,
        "dropout": 0.10,
        "post_input_projection_mask": True,
        "post_block_mask": True,
    },
    "patch": {
        "length": 8,
        "stride": 4,
        "count": 19,
        "minimum_valid_points": 6,
        "minimum_valid_adjacent_displacements": 2,
        "step_epsilon": 1.0e-8,
        "reversal": {
            "direction_span": 2,
            "angle_deg": 120.0,
            "merge_gap": 2,
            "require_all_span_positions_valid": True,
        },
        "revisit_radius": 0.10,
        "semantic_fields": [
            "path_length",
            "net_displacement",
            "path_efficiency",
            "mean_abs_turn",
            "max_abs_turn",
            "cumulative_abs_curvature",
            "reversal_count",
            "nearest_prior_patch_revisit_distance",
            "bbox_coverage_area",
            "valid_ratio",
            "mean_quality",
        ],
        "semantic_projection": {
            "input_channels": 11,
            "hidden_channels": 64,
            "output_channels": 96,
            "layer_norm": {"eps": 1.0e-5, "elementwise_affine": True},
            "linear_bias": True,
            "activation": "gelu",
            "dropout": 0.0,
        },
        "forbidden_semantics": ["dwell_ratio", "speed", "duration"],
    },
    "relation": {
        "symmetric_fields": [
            "absolute_patch_distance",
            "is_adjacent",
            "center_distance",
            "is_revisit",
            "is_reverse_heading",
            "bbox_iou",
            "minimum_pair_quality",
        ],
        "center_distance_scale": 0.10,
        "center_distance_max": 4.0,
        "revisit_minimum_patch_gap": 2,
        "revisit_radius": 0.10,
        "reverse_heading_cosine_max": -0.5,
        "mlp": {
            "input_channels": 7,
            "hidden_channels": 32,
            "output_heads": 4,
            "bias": True,
            "activation": "gelu",
            "dropout": 0.0,
        },
        "signed_offset": {"minimum": -18, "maximum": 18, "bucket_count": 37, "output_heads": 4},
        "cls_explicit_bias": 0.0,
        "shared_across_transformer_layers": True,
        "absolute_position_embedding": "none",
    },
    "transformer": {
        "layers": 2,
        "d_model": 96,
        "heads": 4,
        "head_dim": 24,
        "norm_position": "pre",
        "input_token_dropout": 0.0,
        "layer_norm": {"eps": 1.0e-5, "elementwise_affine": True},
        "q_bias": True,
        "k_bias": True,
        "v_bias": True,
        "output_projection_bias": True,
        "attention_weight_dropout": 0.10,
        "attention_output_dropout": 0.10,
        "ffn": {
            "hidden_channels": 192,
            "activation": "gelu",
            "hidden_dropout": 0.10,
            "output_dropout": 0.10,
            "linear_bias": True,
        },
        "final_layer_norm": {"eps": 1.0e-5, "elementwise_affine": True},
        "invalid_patch_as_key_value": False,
        "zero_invalid_patch_after_each_layer": True,
    },
    "heads": {
        "binary": {
            "input_channels": 96,
            "hidden_channels": 64,
            "output_channels": 1,
            "activation": "gelu",
            "dropout": 0.10,
            "bias": True,
        },
        "subtype": {
            "input_channels": 96,
            "hidden_channels": 64,
            "output_channels": 3,
            "activation": "gelu",
            "dropout": 0.10,
            "bias": True,
        },
        "projection": {"input_channels": 96, "output_channels": 64, "bias": True, "normalization": "none"},
        "forward_outputs": ["binary_logit", "subtype_logits", "projection_embedding"],
        "probability_helper": "hierarchical_four_class_only",
    },
    "initialization": {
        "seed": 20260731,
        "preserve_caller_rng_state": True,
        "linear_and_conv_weight": {"name": "xavier_uniform", "gain": 1.0},
        "linear_and_conv_bias": "zeros",
        "layer_norm_weight": "ones",
        "layer_norm_bias": "zeros",
        "cls_and_signed_offset": {"name": "normal", "mean": 0.0, "std": 0.02},
    },
    "state": {
        "format": "deterministic_npz_v1",
        "allow_pickle": False,
        "state_key_order": "sorted",
        "member_name": "state_key_plus_dot_npy",
        "zip_compression": "stored",
        "zip_timestamp": [1980, 1, 1, 0, 0, 0],
        "zip_create_system": 3,
        "zip_unix_mode": "0600",
        "array_dtype": "little_endian_float32",
        "config_trust": "external_expected_sha256",
        "parameter_count": 204466,
    },
    "verification": {
        "device": "cpu",
        "amp": False,
        "deterministic_algorithms": True,
        "intra_op_threads": 1,
        "inter_op_threads": 1,
    },
    "forbidden": {
        "data_access": True,
        "training_or_optimizer": True,
        "loss_or_decoder": True,
        "performance_metrics": True,
        "model_artifact_publication": True,
        "pickle_joblib_torch_load": True,
        "top_level_runtime_export": True,
    },
}


class TopoWanderContractError(ValueError):
    """The frozen step-9 forward contract was violated."""


class TopoWanderStateError(TopoWanderContractError):
    """A config or numeric state payload is not trusted."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_unique_safe_yaml(payload: str) -> Any:
    loader = _UniqueKeySafeLoader(payload)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _exact_config_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or len(actual) != len(expected):
            return False
        for expected_key, expected_value in expected.items():
            matching_keys = [
                actual_key
                for actual_key in actual
                if type(actual_key) is type(expected_key) and actual_key == expected_key
            ]
            if len(matching_keys) != 1:
                return False
            if not _exact_config_matches(actual[matching_keys[0]], expected_value):
                return False
        return True
    if isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            return False
        return all(
            _exact_config_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _capture_plain_config(value: Any) -> Any:
    """Capture each value once from exact built-in config containers."""

    if type(value) is dict:
        snapshot: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TopoWanderContractError("model construction requires the exact TopoWander v1 config")
            snapshot[key] = _capture_plain_config(item)
        return snapshot
    if type(value) is list:
        return [_capture_plain_config(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TopoWanderContractError("model construction requires the exact TopoWander v1 config")


def _load_topowander_config_bytes(payload: bytes) -> dict[str, Any]:
    """Parse and validate one exact raw-byte snapshot of the step-9 config."""

    if type(payload) is not bytes:
        raise TopoWanderContractError("TopoWander config payload must be raw bytes")
    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = _load_unique_safe_yaml(decoded)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TopoWanderContractError("cannot decode or parse TopoWander config bytes") from exc
    if not _exact_config_matches(value, _EXPECTED_CONFIG):
        raise TopoWanderContractError("TopoWander v1 config fields or frozen values have drifted")
    return value


def _freeze_config(value: Any) -> Any:
    """Copy mappings/lists into a recursively immutable private snapshot."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_config(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_config(item) for item in value)
    return value


@dataclass(frozen=True)
class PatchGeometry:
    semantics: torch.Tensor
    patch_mask: torch.Tensor
    centers: torch.Tensor
    headings: torch.Tensor
    heading_available: torch.Tensor
    bboxes: torch.Tensor
    mean_quality: torch.Tensor


@dataclass(frozen=True)
class TopoWanderEncoding:
    cls_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    patch_mask: torch.Tensor
    relation_bias: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...]


def load_topowander_config(path: str | Path) -> dict[str, Any]:
    """Load the complete exact step-9 config; any field/value drift fails."""

    config_path = Path(path)
    try:
        payload = config_path.read_bytes()
    except OSError as exc:
        raise TopoWanderContractError(f"cannot read TopoWander config: {config_path}") from exc
    return _load_topowander_config_bytes(payload)


class MaskedTemporalResidualBlock(nn.Module):
    """Pre-LN depthwise temporal block with explicit residual masking."""

    def __init__(self, spec: Mapping[str, Any], *, dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        in_channels = int(spec["in_channels"])
        out_channels = int(spec["out_channels"])
        kernel_size = int(spec["kernel_size"])
        dilation = int(spec["dilation"])
        self.normalization = nn.LayerNorm(in_channels, eps=layer_norm_eps, elementwise_affine=True)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values * mask.unsqueeze(-1)
        residual = self.residual_projection(masked.transpose(1, 2)).transpose(1, 2)
        main = F.gelu(self.normalization(masked)).transpose(1, 2)
        main = self.depthwise(main)
        main = self.pointwise(main).transpose(1, 2)
        main = self.dropout(F.gelu(main))
        return (main + residual) * mask.unsqueeze(-1)


class ExplicitRelationAttention(nn.Module):
    """Four-head attention that requires an explicit additive relation bias."""

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        dropout: float,
        q_bias: bool = True,
        k_bias: bool = True,
        v_bias: bool = True,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or heads <= 0 or d_model % heads != 0:
            raise TopoWanderContractError("attention dimensions are invalid")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.query = nn.Linear(d_model, d_model, bias=q_bias)
        self.key = nn.Linear(d_model, d_model, bias=k_bias)
        self.value = nn.Linear(d_model, d_model, bias=v_bias)
        self.output = nn.Linear(d_model, d_model, bias=output_bias)
        self.weight_dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        key_mask: torch.Tensor,
        additive_bias: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[-1] != self.d_model:
            raise TopoWanderContractError("attention values must be [B,T,d_model]")
        batch_size, token_count, _ = values.shape
        if key_mask.shape != (batch_size, token_count) or key_mask.dtype != torch.bool:
            raise TopoWanderContractError("attention key mask must be bool [B,T]")
        expected_bias_shape = (batch_size, self.heads, token_count, token_count)
        if additive_bias.shape != expected_bias_shape or additive_bias.dtype != values.dtype:
            raise TopoWanderContractError("attention bias must be float32 [B,heads,T,T]")
        if values.device != key_mask.device or values.device != additive_bias.device:
            raise TopoWanderContractError("attention inputs must share one device")
        if torch.any(~key_mask.any(dim=1)):
            raise TopoWanderContractError("attention requires at least one key per sample")

        def project(layer: nn.Linear) -> torch.Tensor:
            projected = layer(values).reshape(batch_size, token_count, self.heads, self.head_dim)
            return projected.transpose(1, 2)

        query = project(self.query)
        key = project(self.key)
        value = project(self.value)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        logits = logits + additive_bias
        logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        dropped_weights = self.weight_dropout(weights)
        attended = torch.matmul(dropped_weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, self.d_model)
        output = self.output(attended)
        if return_attention:
            return output, dropped_weights
        return output


class RelationTransformerLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        eps = float(config["layer_norm"]["eps"])
        self.attention_norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=True)
        self.attention = ExplicitRelationAttention(
            d_model=d_model,
            heads=int(config["heads"]),
            dropout=float(config["attention_weight_dropout"]),
            q_bias=bool(config["q_bias"]),
            k_bias=bool(config["k_bias"]),
            v_bias=bool(config["v_bias"]),
            output_bias=bool(config["output_projection_bias"]),
        )
        self.attention_output_dropout = nn.Dropout(float(config["attention_output_dropout"]))
        self.ffn_norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=True)
        hidden_channels = int(config["ffn"]["hidden_channels"])
        self.ffn_in = nn.Linear(d_model, hidden_channels, bias=True)
        self.ffn_hidden_dropout = nn.Dropout(float(config["ffn"]["hidden_dropout"]))
        self.ffn_out = nn.Linear(hidden_channels, d_model, bias=True)
        self.ffn_output_dropout = nn.Dropout(float(config["ffn"]["output_dropout"]))

    def forward(
        self,
        values: torch.Tensor,
        token_mask: torch.Tensor,
        relation_bias: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_result = self.attention(
            self.attention_norm(values),
            token_mask,
            relation_bias,
            return_attention=return_attention,
        )
        if return_attention:
            attention_output, attention_weights = attention_result
        else:
            attention_output = attention_result
            attention_weights = None
        values = values + self.attention_output_dropout(attention_output)
        hidden = self.ffn_in(self.ffn_norm(values))
        hidden = self.ffn_hidden_dropout(F.gelu(hidden))
        values = values + self.ffn_output_dropout(self.ffn_out(hidden))
        values = values * token_mask.unsqueeze(-1)
        return values, attention_weights


class TopoWanderMPT(nn.Module):
    """The exact step-9 mask-aware TCN, semantic-patch, relation-MPT model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config_snapshot = _capture_plain_config(config)
        if not _exact_config_matches(config_snapshot, _EXPECTED_CONFIG):
            raise TopoWanderContractError("model construction requires the exact TopoWander v1 config")
        self._config = _freeze_config(config_snapshot)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(self.config["initialization"]["seed"]))
            self._build_modules()
            self._initialize_parameters()
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != TOPOWANDER_PARAMETER_COUNT:
            raise TopoWanderContractError(
                f"TopoWander parameter count drifted: expected {TOPOWANDER_PARAMETER_COUNT}, got {parameter_count}"
            )

    @property
    def config(self) -> Mapping[str, Any]:
        """The recursively immutable config snapshot owned by this model."""

        return self._config

    def _build_modules(self) -> None:
        tcn = self.config["tcn"]
        projection = tcn["input_projection"]
        self.input_projection = nn.Conv1d(
            int(projection["in_channels"]),
            int(projection["out_channels"]),
            kernel_size=int(projection["kernel_size"]),
            bias=bool(projection["bias"]),
        )
        self.tcn_blocks = nn.ModuleList(
            [
                MaskedTemporalResidualBlock(
                    spec,
                    dropout=float(tcn["dropout"]),
                    layer_norm_eps=float(tcn["layer_norm"]["eps"]),
                )
                for spec in tcn["blocks"]
            ]
        )

        semantic = self.config["patch"]["semantic_projection"]
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(
                int(semantic["input_channels"]),
                eps=float(semantic["layer_norm"]["eps"]),
                elementwise_affine=True,
            ),
            nn.Linear(
                int(semantic["input_channels"]),
                int(semantic["hidden_channels"]),
                bias=bool(semantic["linear_bias"]),
            ),
            nn.GELU(),
            nn.Linear(
                int(semantic["hidden_channels"]),
                int(semantic["output_channels"]),
                bias=bool(semantic["linear_bias"]),
            ),
        )

        relation = self.config["relation"]
        relation_mlp = relation["mlp"]
        self.relation_mlp = nn.Sequential(
            nn.Linear(
                int(relation_mlp["input_channels"]),
                int(relation_mlp["hidden_channels"]),
                bias=bool(relation_mlp["bias"]),
            ),
            nn.GELU(),
            nn.Linear(
                int(relation_mlp["hidden_channels"]),
                int(relation_mlp["output_heads"]),
                bias=bool(relation_mlp["bias"]),
            ),
        )
        signed = relation["signed_offset"]
        self.signed_offset_embedding = nn.Embedding(int(signed["bucket_count"]), int(signed["output_heads"]))

        transformer = self.config["transformer"]
        d_model = int(transformer["d_model"])
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.transformer_layers = nn.ModuleList(
            [RelationTransformerLayer(transformer) for _ in range(int(transformer["layers"]))]
        )
        self.final_norm = nn.LayerNorm(
            d_model,
            eps=float(transformer["final_layer_norm"]["eps"]),
            elementwise_affine=True,
        )

        heads = self.config["heads"]
        self.binary_head = self._classification_head(heads["binary"])
        self.subtype_head = self._classification_head(heads["subtype"])
        self.projection_head = nn.Linear(
            int(heads["projection"]["input_channels"]),
            int(heads["projection"]["output_channels"]),
            bias=bool(heads["projection"]["bias"]),
        )

    @staticmethod
    def _classification_head(spec: Mapping[str, Any]) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(spec["input_channels"]), int(spec["hidden_channels"]), bias=bool(spec["bias"])),
            nn.GELU(),
            nn.Dropout(float(spec["dropout"])),
            nn.Linear(int(spec["hidden_channels"]), int(spec["output_channels"]), bias=bool(spec["bias"])),
        )

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.signed_offset_embedding.weight, mean=0.0, std=0.02)

    def _tcn_stem(self, model_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        values = model_features * point_mask.unsqueeze(-1)
        values = self.input_projection(values.transpose(1, 2)).transpose(1, 2)
        values = values * point_mask.unsqueeze(-1)
        for block in self.tcn_blocks:
            values = block(values, point_mask)
        return values

    def _patch_tokens(
        self,
        tcn_values: torch.Tensor,
        geometry: PatchGeometry,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        patch_config = self.config["patch"]
        patch_length = int(patch_config["length"])
        patch_stride = int(patch_config["stride"])
        pooled = []
        for patch_index in range(int(patch_config["count"])):
            start = patch_index * patch_stride
            end = start + patch_length
            local_mask = point_mask[:, start:end]
            denominator = local_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            local = (tcn_values[:, start:end] * local_mask.unsqueeze(-1)).sum(dim=1) / denominator
            pooled.append(local)
        tcn_patches = torch.stack(pooled, dim=1) * geometry.patch_mask.unsqueeze(-1)
        semantic = self.semantic_projection(geometry.semantics) * geometry.patch_mask.unsqueeze(-1)
        return (tcn_patches + semantic) * geometry.patch_mask.unsqueeze(-1)

    def _relation_bias(self, geometry: PatchGeometry) -> torch.Tensor:
        relation, pair_mask = build_symmetric_relation_features(geometry, self.config["relation"])
        symmetric_bias = self.relation_mlp(relation)
        indices = signed_offset_indices(geometry.patch_mask.shape[1], self.config["relation"]["signed_offset"])
        indices = indices.to(device=geometry.patch_mask.device)
        signed_bias = self.signed_offset_embedding(indices).unsqueeze(0)
        patch_bias = (symmetric_bias + signed_bias) * pair_mask.unsqueeze(-1)
        patch_bias = patch_bias.permute(0, 3, 1, 2)
        batch_size, heads, patch_count, _ = patch_bias.shape
        full_bias = patch_bias.new_zeros((batch_size, heads, patch_count + 1, patch_count + 1))
        full_bias[:, :, 1:, 1:] = patch_bias
        return full_bias

    def encode(
        self,
        model_features: torch.Tensor,
        shape_normalized_points: torch.Tensor,
        point_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> TopoWanderEncoding:
        validate_topowander_inputs(model_features, shape_normalized_points, point_mask, self.config["input"])
        masked_shape = shape_normalized_points * point_mask.unsqueeze(-1)
        quality = model_features[:, :, int(self.config["input"]["quality_channel_index"])] * point_mask
        geometry = build_patch_geometry(masked_shape, point_mask, quality, self.config["patch"])
        if torch.any(geometry.patch_mask.sum(dim=1) <= 0.0):
            raise TopoWanderContractError("each sample must contain at least one valid semantic patch")

        tcn_values = self._tcn_stem(model_features, point_mask)
        patch_tokens = self._patch_tokens(tcn_values, geometry, point_mask)
        batch_size = patch_tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        values = torch.cat((cls, patch_tokens), dim=1)
        patch_mask_bool = geometry.patch_mask.to(dtype=torch.bool)
        token_mask = torch.cat(
            (torch.ones((batch_size, 1), dtype=torch.bool, device=point_mask.device), patch_mask_bool),
            dim=1,
        )
        relation_bias = self._relation_bias(geometry)
        recorded_attention = []
        for layer in self.transformer_layers:
            values, attention = layer(
                values,
                token_mask,
                relation_bias,
                return_attention=return_attention,
            )
            if attention is not None:
                recorded_attention.append(attention)
        values = self.final_norm(values) * token_mask.unsqueeze(-1)
        return TopoWanderEncoding(
            cls_embedding=values[:, 0],
            patch_embeddings=values[:, 1:] * geometry.patch_mask.unsqueeze(-1),
            patch_mask=geometry.patch_mask,
            relation_bias=relation_bias,
            attention_weights=tuple(recorded_attention),
        )

    def forward(
        self,
        model_features: torch.Tensor,
        shape_normalized_points: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoding = self.encode(model_features, shape_normalized_points, point_mask)
        outputs = {
            "binary_logit": self.binary_head(encoding.cls_embedding),
            "subtype_logits": self.subtype_head(encoding.cls_embedding),
            "projection_embedding": self.projection_head(encoding.cls_embedding),
        }
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in outputs.values()):
            raise TopoWanderContractError("TopoWander forward outputs must be finite float32 tensors")
        return outputs


def create_topowander_model(config: dict[str, Any]) -> TopoWanderMPT:
    """Construct the exact CPU forward module without changing caller RNG."""

    return TopoWanderMPT(config)


def create_topowander_training_model(config: dict[str, Any], *, seed: int) -> TopoWanderMPT:
    """Construct the exact forward module with a reproducible training initialization.

    The forward contract keeps its historical canonical initialization seed so
    that ordinary construction remains byte-stable.  M0-S needs the same
    initialization *method* under three explicit seeds; this factory therefore
    reapplies that unchanged method without mutating the trusted forward config
    or the caller RNG.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TopoWanderContractError("training initialization seed must be a non-negative integer")
    model = TopoWanderMPT(config)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        model._initialize_parameters()
    return model


def validate_topowander_inputs(
    model_features: torch.Tensor,
    shape_normalized_points: torch.Tensor,
    point_mask: torch.Tensor,
    config: Mapping[str, Any],
) -> None:
    if not all(isinstance(value, torch.Tensor) for value in (model_features, shape_normalized_points, point_mask)):
        raise TopoWanderContractError("all three model inputs must be torch tensors")
    sequence_length = int(config["sequence_length"])
    feature_channels = int(config["model_feature_channels"])
    shape_channels = int(config["shape_point_channels"])
    if model_features.ndim != 3 or model_features.shape[0] < 1 or model_features.shape[1:] != (
        sequence_length,
        feature_channels,
    ):
        raise TopoWanderContractError("model_features must be float32 [B,80,14]")
    batch_size = model_features.shape[0]
    if shape_normalized_points.shape != (batch_size, sequence_length, shape_channels):
        raise TopoWanderContractError("shape_normalized_points must be float32 [B,80,2]")
    if point_mask.shape != (batch_size, sequence_length):
        raise TopoWanderContractError("point_mask must be float32 [B,80]")
    devices = {model_features.device, shape_normalized_points.device, point_mask.device}
    if len(devices) != 1:
        raise TopoWanderContractError("all three model inputs must be on the same device")
    if any(value.dtype != torch.float32 for value in (model_features, shape_normalized_points, point_mask)):
        raise TopoWanderContractError("TopoWander v1 accepts explicit float32 inputs only")
    if not all(bool(torch.isfinite(value).all()) for value in (model_features, shape_normalized_points, point_mask)):
        raise TopoWanderContractError("all three model inputs must be finite")
    if not torch.all((point_mask == 0.0) | (point_mask == 1.0)):
        raise TopoWanderContractError("point_mask must contain exact 0/1 values")
    mask_channel = int(config["mask_channel_index"])
    if not torch.equal(model_features[:, :, mask_channel], point_mask):
        raise TopoWanderContractError("model feature mask channel must exactly equal point_mask")
    quality = model_features[:, :, int(config["quality_channel_index"])]
    if torch.any((quality < 0.0) | (quality > 1.0)):
        raise TopoWanderContractError("model feature quality channel must remain in [0,1]")
    valid = point_mask.to(dtype=torch.bool)
    time_channels = model_features[:, :, list(config["time_channel_indices"])][valid]
    if not torch.equal(time_channels, torch.zeros_like(time_channels)):
        raise TopoWanderContractError("temporal channels 10/11 must be zero at valid positions")
    if torch.any(point_mask.sum(dim=1) < float(config["minimum_sample_valid_points"])):
        raise TopoWanderContractError("each sample requires at least 8 valid points")


def build_patch_geometry(
    shape_normalized_points: torch.Tensor,
    point_mask: torch.Tensor,
    quality: torch.Tensor,
    config: Mapping[str, Any],
) -> PatchGeometry:
    """Build 19 exact patch masks, geometric summaries, and 11 semantics."""

    if shape_normalized_points.ndim != 3 or shape_normalized_points.shape[1:] != (80, 2):
        raise TopoWanderContractError("patch shape points must be [B,80,2]")
    if point_mask.shape != shape_normalized_points.shape[:2] or quality.shape != point_mask.shape:
        raise TopoWanderContractError("patch mask and quality must be [B,80]")
    if shape_normalized_points.device != point_mask.device or shape_normalized_points.device != quality.device:
        raise TopoWanderContractError("patch inputs must share one device")
    if any(value.dtype != torch.float32 for value in (shape_normalized_points, point_mask, quality)):
        raise TopoWanderContractError("patch inputs must be float32")
    if not torch.all((point_mask == 0.0) | (point_mask == 1.0)):
        raise TopoWanderContractError("patch point mask must be binary")

    points = shape_normalized_points * point_mask.unsqueeze(-1)
    quality = quality * point_mask
    batch_size = points.shape[0]
    length = int(config["length"])
    stride = int(config["stride"])
    count = int(config["count"])
    epsilon = float(config["step_epsilon"])
    radius = float(config["revisit_radius"])

    patch_masks: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    headings: list[torch.Tensor] = []
    heading_available: list[torch.Tensor] = []
    bboxes: list[torch.Tensor] = []
    mean_qualities: list[torch.Tensor] = []
    base_semantics: list[list[torch.Tensor]] = []

    for batch_index in range(batch_size):
        batch_masks = []
        batch_centers = []
        batch_headings = []
        batch_heading_available = []
        batch_bboxes = []
        batch_qualities = []
        batch_base_semantics = []
        for patch_index in range(count):
            start = patch_index * stride
            end = start + length
            local_points = points[batch_index, start:end]
            local_mask = point_mask[batch_index, start:end].to(dtype=torch.bool)
            local_quality = quality[batch_index, start:end]
            adjacent_valid = local_mask[:-1] & local_mask[1:]
            valid_count = int(local_mask.sum().item())
            adjacent_count = int(adjacent_valid.sum().item())
            patch_valid = (
                valid_count >= int(config["minimum_valid_points"])
                and adjacent_count >= int(config["minimum_valid_adjacent_displacements"])
            )
            batch_masks.append(points.new_tensor(1.0 if patch_valid else 0.0))
            if not patch_valid:
                zero = points.new_zeros(())
                batch_centers.append(points.new_zeros((2,)))
                batch_headings.append(points.new_zeros((2,)))
                batch_heading_available.append(torch.tensor(False, device=points.device))
                batch_bboxes.append(points.new_zeros((4,)))
                batch_qualities.append(zero)
                batch_base_semantics.append([zero for _ in range(10)])
                continue

            valid_points = local_points[local_mask]
            center = valid_points.mean(dim=0)
            minimum = valid_points.amin(dim=0)
            maximum = valid_points.amax(dim=0)
            bbox = torch.cat((minimum, maximum))
            bbox_area = (maximum[0] - minimum[0]) * (maximum[1] - minimum[1])
            displacements = local_points[1:] - local_points[:-1]
            valid_displacements = displacements * adjacent_valid.unsqueeze(-1)
            lengths = torch.linalg.vector_norm(valid_displacements, dim=-1)
            path_length = lengths.sum()
            net_displacement = torch.linalg.vector_norm(valid_displacements.sum(dim=0))
            path_efficiency = net_displacement / torch.clamp(path_length, min=epsilon)

            turns = []
            for local_index in range(2, length):
                if not bool(local_mask[local_index - 2 : local_index + 1].all()):
                    continue
                incoming = local_points[local_index - 1] - local_points[local_index - 2]
                outgoing = local_points[local_index] - local_points[local_index - 1]
                if float(torch.linalg.vector_norm(incoming).detach()) <= epsilon:
                    continue
                if float(torch.linalg.vector_norm(outgoing).detach()) <= epsilon:
                    continue
                cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                dot = torch.dot(incoming, outgoing)
                turns.append(torch.abs(torch.atan2(cross, dot)))
            if turns:
                turn_values = torch.stack(turns)
                mean_abs_turn = turn_values.mean()
                max_abs_turn = turn_values.max()
                cumulative_abs_curvature = turn_values.sum()
            else:
                mean_abs_turn = points.new_zeros(())
                max_abs_turn = points.new_zeros(())
                cumulative_abs_curvature = points.new_zeros(())

            reversal_candidates = []
            span = int(config["reversal"]["direction_span"])
            threshold = math.radians(float(config["reversal"]["angle_deg"]))
            for local_index in range(span, length - span):
                if not bool(local_mask[local_index - span : local_index + span + 1].all()):
                    continue
                incoming = local_points[local_index] - local_points[local_index - span]
                outgoing = local_points[local_index + span] - local_points[local_index]
                incoming_norm = torch.linalg.vector_norm(incoming)
                outgoing_norm = torch.linalg.vector_norm(outgoing)
                if float(incoming_norm.detach()) <= epsilon or float(outgoing_norm.detach()) <= epsilon:
                    continue
                cosine = torch.dot(incoming, outgoing) / (incoming_norm * outgoing_norm)
                angle = torch.acos(torch.clamp(cosine, min=-1.0, max=1.0))
                if float(angle.detach()) >= threshold:
                    reversal_candidates.append(local_index)
            reversal_count = 0
            previous = None
            for candidate in reversal_candidates:
                if previous is None or candidate - previous > int(config["reversal"]["merge_gap"]):
                    reversal_count += 1
                previous = candidate

            valid_unit_displacements = []
            for local_index in range(length - 1):
                if not bool(adjacent_valid[local_index]):
                    continue
                displacement = displacements[local_index]
                displacement_norm = torch.linalg.vector_norm(displacement)
                if float(displacement_norm.detach()) > epsilon:
                    valid_unit_displacements.append(displacement / displacement_norm)
            if valid_unit_displacements:
                mean_direction = torch.stack(valid_unit_displacements).mean(dim=0)
                mean_direction_norm = torch.linalg.vector_norm(mean_direction)
                available = float(mean_direction_norm.detach()) > epsilon
                heading = mean_direction / mean_direction_norm if available else points.new_zeros((2,))
            else:
                heading = points.new_zeros((2,))
                available = False
            mean_quality = local_quality[local_mask].mean()
            valid_ratio = points.new_tensor(float(valid_count) / float(length))

            batch_centers.append(center)
            batch_headings.append(heading)
            batch_heading_available.append(torch.tensor(available, device=points.device))
            batch_bboxes.append(bbox)
            batch_qualities.append(mean_quality)
            batch_base_semantics.append(
                [
                    path_length,
                    net_displacement,
                    path_efficiency,
                    mean_abs_turn,
                    max_abs_turn,
                    cumulative_abs_curvature,
                    points.new_tensor(float(reversal_count)),
                    bbox_area,
                    valid_ratio,
                    mean_quality,
                ]
            )
        patch_masks.append(torch.stack(batch_masks))
        centers.append(torch.stack(batch_centers))
        headings.append(torch.stack(batch_headings))
        heading_available.append(torch.stack(batch_heading_available))
        bboxes.append(torch.stack(batch_bboxes))
        mean_qualities.append(torch.stack(batch_qualities))
        base_semantics.append(batch_base_semantics)

    patch_mask_tensor = torch.stack(patch_masks)
    center_tensor = torch.stack(centers)
    heading_tensor = torch.stack(headings)
    heading_available_tensor = torch.stack(heading_available)
    bbox_tensor = torch.stack(bboxes)
    mean_quality_tensor = torch.stack(mean_qualities)

    semantics = []
    for batch_index in range(batch_size):
        batch_semantics = []
        for patch_index in range(count):
            if not bool(patch_mask_tensor[batch_index, patch_index]):
                batch_semantics.append(points.new_zeros((11,)))
                continue
            prior_indices = [
                prior
                for prior in range(0, patch_index - 1)
                if bool(patch_mask_tensor[batch_index, prior])
            ]
            if prior_indices:
                prior_centers = center_tensor[batch_index, prior_indices]
                distances = torch.linalg.vector_norm(
                    prior_centers - center_tensor[batch_index, patch_index].unsqueeze(0), dim=1
                )
                nearest = torch.clamp(distances.min(), max=radius)
            else:
                nearest = points.new_tensor(radius)
            base = base_semantics[batch_index][patch_index]
            batch_semantics.append(torch.stack((*base[:7], nearest, *base[7:])))
        semantics.append(torch.stack(batch_semantics))

    return PatchGeometry(
        semantics=torch.stack(semantics) * patch_mask_tensor.unsqueeze(-1),
        patch_mask=patch_mask_tensor,
        centers=center_tensor * patch_mask_tensor.unsqueeze(-1),
        headings=heading_tensor * patch_mask_tensor.unsqueeze(-1),
        heading_available=heading_available_tensor & patch_mask_tensor.to(dtype=torch.bool),
        bboxes=bbox_tensor * patch_mask_tensor.unsqueeze(-1),
        mean_quality=mean_quality_tensor * patch_mask_tensor,
    )


def build_symmetric_relation_features(
    geometry: PatchGeometry,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact symmetric seven-field relations and valid-pair mask."""

    batch_size, patch_count = geometry.patch_mask.shape
    relation = geometry.centers.new_zeros((batch_size, patch_count, patch_count, 7))
    pair_mask = geometry.patch_mask.to(dtype=torch.bool).unsqueeze(2) & geometry.patch_mask.to(
        dtype=torch.bool
    ).unsqueeze(1)
    epsilon = 1.0e-8
    for batch_index in range(batch_size):
        for left in range(patch_count):
            if not bool(geometry.patch_mask[batch_index, left]):
                continue
            for right in range(left, patch_count):
                if not bool(geometry.patch_mask[batch_index, right]):
                    continue
                raw_distance = torch.linalg.vector_norm(
                    geometry.centers[batch_index, left] - geometry.centers[batch_index, right]
                )
                center_distance = torch.clamp(
                    raw_distance / float(config["center_distance_scale"]),
                    max=float(config["center_distance_max"]),
                )
                index_gap = abs(left - right)
                is_revisit = float(
                    index_gap >= int(config["revisit_minimum_patch_gap"])
                    and float(raw_distance.detach()) < float(config["revisit_radius"])
                )
                is_reverse = 0.0
                if bool(geometry.heading_available[batch_index, left]) and bool(
                    geometry.heading_available[batch_index, right]
                ):
                    cosine = torch.dot(
                        geometry.headings[batch_index, left], geometry.headings[batch_index, right]
                    )
                    is_reverse = float(float(cosine.detach()) <= float(config["reverse_heading_cosine_max"]))

                left_bbox = geometry.bboxes[batch_index, left]
                right_bbox = geometry.bboxes[batch_index, right]
                intersection_min = torch.maximum(left_bbox[:2], right_bbox[:2])
                intersection_max = torch.minimum(left_bbox[2:], right_bbox[2:])
                intersection_size = torch.clamp(intersection_max - intersection_min, min=0.0)
                intersection = intersection_size[0] * intersection_size[1]
                left_size = torch.clamp(left_bbox[2:] - left_bbox[:2], min=0.0)
                right_size = torch.clamp(right_bbox[2:] - right_bbox[:2], min=0.0)
                union = left_size[0] * left_size[1] + right_size[0] * right_size[1] - intersection
                bbox_iou = intersection / union if float(union.detach()) > epsilon else relation.new_zeros(())
                values = torch.stack(
                    (
                        relation.new_tensor(float(index_gap) / float(patch_count - 1)),
                        relation.new_tensor(float(index_gap == 1)),
                        center_distance,
                        relation.new_tensor(is_revisit),
                        relation.new_tensor(is_reverse),
                        bbox_iou,
                        torch.minimum(
                            geometry.mean_quality[batch_index, left], geometry.mean_quality[batch_index, right]
                        ),
                    )
                )
                relation[batch_index, left, right] = values
                relation[batch_index, right, left] = values
    return relation, pair_mask


def signed_offset_indices(patch_count: int, config: Mapping[str, Any]) -> torch.Tensor:
    """Map directed j-i offsets to the exact -18..18 embedding indices."""

    minimum = int(config["minimum"])
    maximum = int(config["maximum"])
    if (
        minimum != -(patch_count - 1)
        or maximum != patch_count - 1
        or int(config["bucket_count"]) != maximum - minimum + 1
    ):
        raise TopoWanderContractError("signed-offset patch count or bucket count drifted")
    positions = torch.arange(patch_count, dtype=torch.long)
    offsets = positions.unsqueeze(0) - positions.unsqueeze(1)
    if int(offsets.min()) < minimum or int(offsets.max()) > maximum:
        raise TopoWanderContractError("signed patch offset is outside the frozen range")
    return offsets - minimum


def hierarchical_four_class_probabilities(
    binary_logit: torch.Tensor,
    subtype_logits: torch.Tensor,
) -> torch.Tensor:
    """Pure hierarchical probability algebra; no calibration or decision rule."""

    if not isinstance(binary_logit, torch.Tensor) or not isinstance(subtype_logits, torch.Tensor):
        raise TopoWanderContractError("hierarchical logits must be torch tensors")
    if binary_logit.ndim != 2 or binary_logit.shape[1] != 1:
        raise TopoWanderContractError("binary_logit must be [B,1]")
    if subtype_logits.shape != (binary_logit.shape[0], 3):
        raise TopoWanderContractError("subtype_logits must be [B,3]")
    if binary_logit.device != subtype_logits.device or binary_logit.dtype != subtype_logits.dtype:
        raise TopoWanderContractError("hierarchical logits must share dtype and device")
    if not torch.isfinite(binary_logit).all() or not torch.isfinite(subtype_logits).all():
        raise TopoWanderContractError("hierarchical logits must be finite")
    wandering = torch.sigmoid(binary_logit)
    subtype = torch.softmax(subtype_logits, dim=1)
    return torch.cat((1.0 - wandering, wandering * subtype), dim=1)


def deterministic_npz_bytes(model: TopoWanderMPT) -> bytes:
    """Serialize sorted little-endian float32 arrays with fixed ZIP metadata."""

    if not isinstance(model, TopoWanderMPT):
        raise TopoWanderStateError("state serialization requires TopoWanderMPT")
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for key, tensor in sorted(model.state_dict().items()):
            array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.dtype("<f4"))
            if array.dtype.str != "<f4" or not np.isfinite(array).all():
                raise TopoWanderStateError(f"state is not finite little-endian float32: {key}")
            array_bytes = io.BytesIO()
            np.lib.format.write_array(array_bytes, array, allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_bytes.getvalue())
    return output.getvalue()


def safe_load_topowander_model(
    state_path: str | Path,
    *,
    config_path: str | Path,
    expected_config_sha256: str,
) -> TopoWanderMPT:
    """Verify external config trust and complete numeric state before assignment."""

    if not isinstance(expected_config_sha256, str) or _SHA256_RE.fullmatch(expected_config_sha256) is None:
        raise TopoWanderStateError("expected config SHA-256 must be 64 lowercase hex characters")
    resolved_config = Path(config_path)
    try:
        config_bytes = resolved_config.read_bytes()
    except OSError as exc:
        raise TopoWanderStateError(f"cannot read trusted config: {resolved_config}") from exc
    actual_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if actual_config_sha256 != expected_config_sha256:
        raise TopoWanderStateError("external expected config SHA-256 does not match raw config bytes")
    try:
        config = _load_topowander_config_bytes(config_bytes)
    except TopoWanderContractError as exc:
        raise TopoWanderStateError("trusted config does not match the exact forward contract") from exc
    model = create_topowander_model(config)

    resolved_state = Path(state_path)
    try:
        payload = resolved_state.read_bytes()
    except OSError as exc:
        raise TopoWanderStateError(f"cannot read numeric state: {resolved_state}") from exc
    if not payload:
        raise TopoWanderStateError("numeric state must be non-empty")
    expected_state = model.state_dict()
    expected_keys = sorted(expected_state)
    expected_members = [f"{key}.npy" for key in expected_keys]
    loaded: dict[str, torch.Tensor] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = archive.namelist()
            if members != expected_members or len(members) != len(set(members)):
                raise TopoWanderStateError("numeric state keys or sorted member order drifted")
            for info in archive.infolist():
                mode = (info.external_attr >> 16) & 0o777
                if (
                    info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.create_system != 3
                    or mode != 0o600
                ):
                    raise TopoWanderStateError("numeric state ZIP metadata drifted")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != expected_keys:
                raise TopoWanderStateError("numeric state array order drifted")
            for key in expected_keys:
                array = archive[key]
                expected = expected_state[key]
                if array.dtype.str != "<f4":
                    raise TopoWanderStateError(f"numeric state dtype drifted: {key}")
                if array.shape != tuple(expected.shape):
                    raise TopoWanderStateError(f"numeric state shape drifted: {key}")
                if not np.isfinite(array).all():
                    raise TopoWanderStateError(f"numeric state contains non-finite values: {key}")
                loaded[key] = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    except TopoWanderStateError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed ZIP/NPY parser errors are untrusted state
        raise TopoWanderStateError("cannot safely parse deterministic numeric state") from exc
    try:
        model.load_state_dict(loaded, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise TopoWanderStateError("verified numeric state cannot be assigned atomically") from exc
    return model


__all__ = [
    "TOPOWANDER_PARAMETER_COUNT",
    "ExplicitRelationAttention",
    "PatchGeometry",
    "TopoWanderContractError",
    "TopoWanderEncoding",
    "TopoWanderMPT",
    "TopoWanderStateError",
    "build_patch_geometry",
    "build_symmetric_relation_features",
    "create_topowander_model",
    "create_topowander_training_model",
    "deterministic_npz_bytes",
    "hierarchical_four_class_probabilities",
    "load_topowander_config",
    "safe_load_topowander_model",
    "signed_offset_indices",
    "validate_topowander_inputs",
]
