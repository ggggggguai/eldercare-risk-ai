"""Deterministic two-task pure-TCN comparison baseline for wandering step 6.

This module is deliberately isolated from camera, augmentation, calibration,
rejection, episodes, daily evidence, and the shared mental-health runtime.  It
reuses the step-5 fail-closed preprocessing loader and never has a SmartCare
official-validation path.  Development can expose only train/validation;
frozen evaluation can expose only the fixed 240-row WP shape benchmark.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import random
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    BUNDLE_MODE_FROZEN_WP_TEST,
    _canonical_json_bytes,
    _commit_new_output_directory,
    _sha256_file,
    load_preprocessing_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.rf_baseline import (
    load_trusted_development_manifest as load_trusted_rf_development_manifest,
)


TASK_FOUR_CLASS = "four_class"
TASK_BINARY = "binary"
FOUR_CLASS_NAMES = ("direct", "pacing", "lapping", "random")
BINARY_CLASS_NAMES = ("direct_or_non_wandering", "wandering_like")
TCN_SEEDS = (20260731, 20260801, 20260802, 20260803, 20260804)
PRIMARY_SEED = 20260731
TCN_FOUR_CLASS_PARAMETER_COUNT = 40836
TCN_BINARY_PARAMETER_COUNT = 34433
TCN_CONFIG_SCHEMA_VERSION = "wandering-tcn-config-v1"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = "wandering-tcn-development-manifest-v1"
PUBLIC_BENCHMARK_MANIFEST_SCHEMA_VERSION = "wandering-tcn-public-shape-benchmark-manifest-v1"
PREDICTION_ROW_SCHEMA_VERSION = "wandering-tcn-prediction-row-v1"
NPZ_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_VALID_SHA256 = frozenset("0123456789abcdef")

_EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": TCN_CONFIG_SCHEMA_VERSION,
    "purpose": "comparison_only",
    "trust_roots": {
        "rf_config": {
            "path": "configs/modules/wandering_rf_v1.yaml",
            "sha256": "d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35",
        },
        "rf_development_manifest": {
            "path": "reports/mental_health/wandering_step5/development/v1/manifest.json",
            "sha256": "fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2",
        },
        "rf_public_manifest": {
            "path": "reports/mental_health/wandering_step5/public_shape_benchmark/v1/manifest.json",
            "sha256": "8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c",
        },
    },
    "preprocessing": {
        "manifest_schema": "wandering-preprocessing-manifest-v1",
        "manifest_sha256": "242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593",
        "sample_schema": "wandering-preprocessed-sample-v1",
        "split_sha256": "4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf",
        "feature_field": "model_features",
        "input_shape": [80, 14],
        "input_dtype": "float32",
        "mask_field": "point_mask",
        "mask_channel_index": 12,
        "time_channel_indices": [10, 11],
        "quality_channel_index": 13,
        "temporal_features_enabled": False,
    },
    "tasks": {
        "four_class": {
            "class_names": list(FOUR_CLASS_NAMES),
            "training_sources": ["wandering_patterns"],
            "architecture": "hierarchical_gate_and_subtype",
            "decision_rule": "argmax",
            "sample_weighting": "uniform",
            "primary_validation_metric": "wp_validation_macro_f1",
        },
        "binary": {
            "class_names": list(BINARY_CLASS_NAMES),
            "training_sources": ["wandering_patterns", "smartcare"],
            "architecture": "sigmoid",
            "decision_threshold": 0.5,
            "sample_weighting": "equal_source_x_class_total",
            "primary_validation_metric": "source_macro_macro_f1",
        },
    },
    "model": {
        "input_channels": 14,
        "sequence_length": 80,
        "input_projection_channels": 64,
        "input_projection_bias": True,
        "normalization": "layer_norm",
        "activation": "gelu",
        "convolution": "noncausal_same_padding",
        "blocks": [
            {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "dilation": 1, "dropout": 0.10},
            {"in_channels": 64, "out_channels": 96, "kernel_size": 3, "dilation": 2, "dropout": 0.10},
            {"in_channels": 96, "out_channels": 96, "kernel_size": 3, "dilation": 4, "dropout": 0.10},
        ],
        "depthwise_bias": True,
        "pointwise_bias": True,
        "channel_change_residual_bias": False,
        "pooling": "masked_mean",
        "head_hidden_channels": 64,
        "head_dropout": 0.10,
        "head_bias": True,
        "four_class_parameter_count": TCN_FOUR_CLASS_PARAMETER_COUNT,
        "binary_parameter_count": TCN_BINARY_PARAMETER_COUNT,
        "forbidden_components": ["patch", "attention", "relation_bias", "self_supervision", "prototype", "energy", "augmentation"],
    },
    "training": {
        "device": "cpu",
        "dtype": "float32",
        "seeds": list(TCN_SEEDS),
        "primary_seed": PRIMARY_SEED,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "batch_size": 48,
        "num_workers": 0,
        "train_shuffle": True,
        "validation_shuffle": False,
        "max_epochs": 150,
        "minimum_epochs": 30,
        "early_stopping_patience": 20,
        "minimum_improvement": 1e-4,
        "early_stopping_tie_policy": "keep_earlier",
        "gradient_clip_global_norm": 1.0,
        "scheduler": "none",
        "amp": False,
        "augmentation": "none",
        "deterministic_algorithms": True,
        "intra_op_threads": 1,
        "inter_op_threads": 1,
    },
    "checkpoint": {
        "format": "deterministic_npz_v1",
        "allow_pickle": False,
        "state_key_order": "sorted",
        "zip_timestamp": list(NPZ_TIMESTAMP),
        "array_dtype": "float32",
        "semantic_fingerprint_schema": "wandering-tcn-semantic-fingerprint-v1",
    },
    "runtime_benchmark": {"batch_size": 1, "warmup_iterations": 20, "measurement_iterations": 200},
    "expected_counts": {
        "total_records": 1790,
        "ready": 1775,
        "unavailable": 15,
        "four_class": {"train": 1120, "validation": 240, "test": 240},
        "binary": {"train": 1257, "validation": 278, "test": 240},
    },
    "output_schemas": {
        "data_index": "wandering-tcn-data-index-v1",
        "training_history": "wandering-tcn-training-history-v1",
        "model_metadata": "wandering-tcn-model-metadata-v1",
        "prediction_row": PREDICTION_ROW_SCHEMA_VERSION,
        "development_manifest": DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        "public_shape_benchmark_manifest": PUBLIC_BENCHMARK_MANIFEST_SCHEMA_VERSION,
    },
}


class TCNBaselineError(ValueError):
    """The fixed step-6 training or evaluation protocol was violated."""


class ModelTrustError(TCNBaselineError):
    """A manifest, source, NPZ array, or semantic fingerprint is untrusted."""


@dataclass(frozen=True)
class TensorCohort:
    features: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TCNDevelopmentBuildResult:
    output_dir: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    failure_cases: Mapping[str, Any]


@dataclass(frozen=True)
class TCNFrozenBenchmarkBuildResult:
    output_dir: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    test_metrics: Mapping[str, Any]
    failure_cases: Mapping[str, Any]


class TemporalResidualBlock(nn.Module):
    """Pre-LN depthwise temporal residual block with explicit post-block mask."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(in_channels)
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
        residual = self.residual_projection(values.transpose(1, 2)).transpose(1, 2)
        main = F.gelu(self.normalization(values)).transpose(1, 2)
        main = self.depthwise(main)
        main = self.pointwise(main).transpose(1, 2)
        main = self.dropout(F.gelu(main))
        return (main + residual) * mask.unsqueeze(-1)


class PureTCNBaseline(nn.Module):
    """The exact pure-TCN architecture shared by definition, not by parameters."""

    def __init__(self, task: str) -> None:
        super().__init__()
        if task not in {TASK_FOUR_CLASS, TASK_BINARY}:
            raise TCNBaselineError(f"unknown TCN task: {task!r}")
        self.task = task
        self.input_projection = nn.Conv1d(14, 64, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(64, 64, kernel_size=3, dilation=1, dropout=0.10),
                TemporalResidualBlock(64, 96, kernel_size=3, dilation=2, dropout=0.10),
                TemporalResidualBlock(96, 96, kernel_size=3, dilation=4, dropout=0.10),
            ]
        )
        if task == TASK_FOUR_CLASS:
            self.head = nn.ModuleDict(
                {
                    "gate": _task_head(1),
                    "subtype": _task_head(3),
                }
            )
        else:
            self.head = nn.ModuleDict({"binary": _task_head(1)})

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_model_inputs(features, mask)
        values = features * mask.unsqueeze(-1)
        values = self.input_projection(values.transpose(1, 2)).transpose(1, 2)
        values = values * mask.unsqueeze(-1)
        for block in self.blocks:
            values = block(values, mask)
        pooled = (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        if self.task == TASK_FOUR_CLASS:
            return {
                "gate_logit": self.head["gate"](pooled),
                "subtype_logits": self.head["subtype"](pooled),
            }
        return {"binary_logit": self.head["binary"](pooled)}


def _task_head(output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(96, 64, bias=True),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(64, output_channels, bias=True),
    )


def load_tcn_config(path: str | Path) -> dict[str, Any]:
    """Load exactly the pre-registered v1 configuration; extensions fail closed."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TCNBaselineError(f"cannot read TCN config: {config_path}") from exc
    if value != _EXPECTED_CONFIG:
        raise TCNBaselineError("TCN v1 config fields or frozen values have drifted")
    return value


def create_tcn_model(task: str, *, seed: int) -> PureTCNBaseline:
    """Create one independently initialized fixed model for a registered seed."""

    if isinstance(seed, bool) or seed not in TCN_SEEDS:
        raise TCNBaselineError(f"seed must be one of the five fixed seeds: {TCN_SEEDS}")
    if task not in {TASK_FOUR_CLASS, TASK_BINARY}:
        raise TCNBaselineError(f"unknown TCN task: {task!r}")
    torch.manual_seed(seed)
    model = PureTCNBaseline(task)
    expected = TCN_FOUR_CLASS_PARAMETER_COUNT if task == TASK_FOUR_CLASS else TCN_BINARY_PARAMETER_COUNT
    if sum(parameter.numel() for parameter in model.parameters()) != expected:
        raise TCNBaselineError(f"fixed {task} parameter count drifted")
    return model


def validate_model_inputs(features: torch.Tensor, mask: torch.Tensor) -> None:
    """Validate the exact float32 [B,80,14] input and explicit binary mask."""

    if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TCNBaselineError("features and mask must be torch tensors")
    if features.ndim != 3 or features.shape[1:] != (80, 14) or features.shape[0] < 1:
        raise TCNBaselineError("model features must be float32 [B,80,14]")
    if mask.shape != features.shape[:2]:
        raise TCNBaselineError("point mask must be float32 [B,80]")
    if features.dtype != torch.float32 or mask.dtype != torch.float32:
        raise TCNBaselineError("TCN v1 accepts float32 features and mask only")
    if not torch.isfinite(features).all() or not torch.isfinite(mask).all():
        raise TCNBaselineError("model features and mask must be finite")
    if not torch.all((mask == 0.0) | (mask == 1.0)):
        raise TCNBaselineError("point mask must be binary")
    if torch.any(mask.sum(dim=1) <= 0.0):
        raise TCNBaselineError("all-zero point masks are unavailable, not classifiable")
    if not torch.equal(features[:, :, 12], mask):
        raise TCNBaselineError("feature channel 12 must exactly equal the external mask")
    valid_temporal = features[:, :, 10:12][mask.to(dtype=torch.bool)]
    if not torch.equal(valid_temporal, torch.zeros_like(valid_temporal)):
        raise TCNBaselineError("TCN v1 temporal channels 10/11 must remain zero at valid positions")


def four_class_probabilities(gate_logits: torch.Tensor, subtype_logits: torch.Tensor) -> torch.Tensor:
    if gate_logits.ndim != 2 or gate_logits.shape[1] != 1:
        raise TCNBaselineError("gate logits must be [B,1]")
    if subtype_logits.shape != (gate_logits.shape[0], 3):
        raise TCNBaselineError("subtype logits must be [B,3]")
    if not torch.isfinite(gate_logits).all() or not torch.isfinite(subtype_logits).all():
        raise TCNBaselineError("classification logits must be finite")
    gate = torch.sigmoid(gate_logits)
    subtype = torch.softmax(subtype_logits, dim=1)
    return torch.cat((1.0 - gate, gate * subtype), dim=1)


def four_class_hierarchical_loss(
    gate_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels.ndim != 1 or labels.shape[0] != gate_logits.shape[0] or not torch.all((labels >= 0) & (labels <= 3)):
        raise TCNBaselineError("four-class labels must be int [B] in 0..3")
    gate_target = (labels != 0).to(dtype=gate_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(gate_logits.squeeze(1), gate_target, reduction="none")
    wandering = labels != 0
    if torch.any(wandering):
        subtype_loss = F.cross_entropy(subtype_logits[wandering], labels[wandering] - 1, reduction="none")
        losses = losses.clone()
        losses[wandering] += subtype_loss
    return _weighted_mean(losses, sample_weights)


def binary_weighted_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    values = logits.squeeze(1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
    if values.ndim != 1 or labels.shape != values.shape or sample_weights.shape != values.shape:
        raise TCNBaselineError("binary logits, labels, and weights must be aligned [B]")
    losses = F.binary_cross_entropy_with_logits(values, labels.to(values.dtype), reduction="none")
    return _weighted_mean(losses, sample_weights)


def _weighted_mean(losses: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(dtype=losses.dtype, device=losses.device)
    if weights.shape != losses.shape or not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
        raise TCNBaselineError("sample weights must be finite positive [B]")
    return torch.sum(losses * weights) / torch.sum(weights)


def build_binary_sample_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Reuse the exact step-5 equal source x class total weighting."""

    if not rows or any(row.get("split") != "train" for row in rows):
        raise TCNBaselineError("binary weights accept train rows only")
    groups = Counter((row.get("source_dataset"), row.get("binary_label")) for row in rows)
    expected = {
        ("wandering_patterns", 0),
        ("wandering_patterns", 1),
        ("smartcare", 0),
        ("smartcare", 1),
    }
    if set(groups) != expected or any(count <= 0 for count in groups.values()):
        raise TCNBaselineError("binary train must contain all four source x class groups")
    total = len(rows)
    weights = np.asarray(
        [total / (4.0 * groups[(row["source_dataset"], row["binary_label"])]) for row in rows],
        dtype=np.float64,
    )
    totals: dict[tuple[str, int], float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        totals[(str(row["source_dataset"]), int(row["binary_label"]))] += float(weight)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0) or max(totals.values()) - min(totals.values()) > 1e-9:
        raise TCNBaselineError("binary source x class weights drifted")
    return weights


def build_tensor_cohort(records: Sequence[Mapping[str, Any]], *, task: str, split: str) -> TensorCohort:
    if task not in {TASK_FOUR_CLASS, TASK_BINARY} or split not in {"train", "validation", "test"}:
        raise TCNBaselineError("unknown tensor cohort task or split")
    rows = tuple(record for record in records if record.get("split") == split)
    expected = {
        (TASK_FOUR_CLASS, "train"): 1120,
        (TASK_FOUR_CLASS, "validation"): 240,
        (TASK_FOUR_CLASS, "test"): 240,
        (TASK_BINARY, "train"): 1257,
        (TASK_BINARY, "validation"): 278,
        (TASK_BINARY, "test"): 240,
    }[(task, split)]
    if len(rows) != expected:
        raise TCNBaselineError(f"fixed {task} {split} count drifted")
    sample_ids = [row.get("sample_id") for row in rows]
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise TCNBaselineError("tensor cohort IDs must be unique and sorted")
    if any(row.get("preprocess_status") != "ready" for row in rows):
        raise TCNBaselineError("unavailable rows cannot enter TCN tensors")
    if task == TASK_FOUR_CLASS:
        if any(row.get("source_dataset") != "wandering_patterns" or row.get("pattern_supervision_eligible") is not True for row in rows):
            raise TCNBaselineError("SmartCare or ineligible rows entered four-class TCN")
        mapping = {name: index for index, name in enumerate(FOUR_CLASS_NAMES)}
        try:
            labels_np = np.asarray([mapping[str(row["pattern_label"])] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise TCNBaselineError("unknown four-class label") from exc
    else:
        if any(row.get("binary_supervision_eligible") is not True for row in rows):
            raise TCNBaselineError("ineligible row entered binary TCN")
        labels_np = np.asarray([row["binary_label"] for row in rows], dtype=np.int64)
        if not np.all(np.isin(labels_np, [0, 1])):
            raise TCNBaselineError("binary labels must be 0/1")
    try:
        features_np = np.asarray([row["model_features"] for row in rows], dtype=np.float32)
        masks_np = np.asarray([row["point_mask"] for row in rows], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise TCNBaselineError("TCN tensors cannot be built from model_features/mask") from exc
    features = torch.from_numpy(np.ascontiguousarray(features_np))
    masks = torch.from_numpy(np.ascontiguousarray(masks_np))
    validate_model_inputs(features, masks)
    if task == TASK_BINARY and split == "train":
        weights_np = build_binary_sample_weights(rows)
    else:
        weights_np = np.ones(len(rows), dtype=np.float64)
    return TensorCohort(
        features=features,
        masks=masks,
        labels=torch.from_numpy(labels_np),
        weights=torch.from_numpy(weights_np.astype(np.float32)),
        records=rows,
    )


def select_early_stopping_epoch(
    metrics: Sequence[float],
    *,
    minimum_epochs: int = 30,
    patience: int = 20,
    minimum_improvement: float = 1e-4,
) -> dict[str, Any]:
    """Apply the exact strict-delta, earlier-tie validation selection rule."""

    if not metrics or minimum_epochs != 30 or patience != 20 or minimum_improvement != 1e-4:
        raise TCNBaselineError("early stopping v1 parameters have drifted")
    best = -math.inf
    best_epoch = 0
    stopped_epoch = len(metrics)
    reason = "maximum_epochs"
    for epoch, value in enumerate(metrics, start=1):
        metric = float(value)
        if not math.isfinite(metric):
            raise TCNBaselineError("validation metric must be finite")
        if metric > best + minimum_improvement:
            best = metric
            best_epoch = epoch
        if epoch >= minimum_epochs and epoch - best_epoch >= patience:
            stopped_epoch = epoch
            reason = "early_stopping_patience"
            break
    return {
        "best_epoch": best_epoch,
        "best_metric": best,
        "stopped_epoch": stopped_epoch,
        "stop_reason": reason,
        "minimum_epochs": minimum_epochs,
        "patience": patience,
        "minimum_improvement": minimum_improvement,
        "strictly_greater_improvement": True,
        "tie_policy": "keep_earlier",
        "validation_used_for_checkpoint_selection": True,
        "test_used_for_checkpoint_selection": False,
    }


def deterministic_npz_bytes(model: nn.Module) -> bytes:
    """Serialize sorted float32 state arrays with fixed ZIP metadata and no pickle."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for key, tensor in sorted(model.state_dict().items()):
            array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
            if array.dtype.kind == "O" or not np.isfinite(array).all():
                raise ModelTrustError(f"checkpoint state is not finite float32: {key}")
            array_bytes = io.BytesIO()
            np.lib.format.write_array(array_bytes, array, allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=NPZ_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_bytes.getvalue())
    return output.getvalue()


def load_deterministic_npz(model: nn.Module, payload: bytes) -> None:
    """Strictly load a pure-array NPZ after key/dtype/shape/finite verification."""

    if not isinstance(payload, bytes) or not payload:
        raise ModelTrustError("checkpoint payload must be non-empty bytes")
    expected_state = model.state_dict()
    expected_keys = sorted(expected_state)
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            names = archive.namelist()
            expected_names = [f"{key}.npy" for key in expected_keys]
            if names != expected_names or len(names) != len(set(names)):
                raise ModelTrustError("NPZ state keys/order have drifted")
            for info in archive.infolist():
                if info.date_time != NPZ_TIMESTAMP or info.compress_type != zipfile.ZIP_STORED:
                    raise ModelTrustError("NPZ metadata is not deterministic v1")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != expected_keys:
                raise ModelTrustError("NPZ array keys have drifted")
            loaded: dict[str, torch.Tensor] = {}
            for key in expected_keys:
                array = archive[key]
                expected = expected_state[key]
                if array.dtype != np.dtype("float32") or array.dtype.kind == "O":
                    raise ModelTrustError(f"NPZ dtype drifted: {key}")
                if array.shape != tuple(expected.shape) or not np.isfinite(array).all():
                    raise ModelTrustError(f"NPZ shape/finite contract drifted: {key}")
                loaded[key] = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    except ModelTrustError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap malformed archive/parser errors
        raise ModelTrustError("cannot safely parse deterministic NPZ") from exc
    try:
        model.load_state_dict(loaded, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ModelTrustError("NPZ state cannot be copied into the fixed model") from exc


def model_semantic_fingerprint(model: PureTCNBaseline, *, task: str) -> dict[str, Any]:
    if task != model.task or task not in {TASK_FOUR_CLASS, TASK_BINARY}:
        raise TCNBaselineError("model task does not match fingerprint task")
    state = model.state_dict()
    digest = hashlib.sha256()
    architecture = _architecture_document(task)
    digest.update(_canonical_without_lf({"architecture": architecture}))
    arrays = []
    for key, tensor in sorted(state.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        if not np.isfinite(array).all():
            raise TCNBaselineError(f"model state is non-finite: {key}")
        descriptor = {"key": key, "dtype": "float32", "shape": list(array.shape)}
        digest.update(_canonical_without_lf(descriptor))
        digest.update(array.tobytes(order="C"))
        arrays.append(descriptor)
    return {
        "schema_version": "wandering-tcn-semantic-fingerprint-v1",
        "task": task,
        "sha256": digest.hexdigest(),
        "architecture": architecture,
        "state_arrays": arrays,
    }


def current_environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "scikit_learn": sklearn.__version__,
    }


def build_tcn_development_artifacts(
    *,
    tcn_config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> TCNDevelopmentBuildResult:
    """Train 10 independent CPU models and atomically freeze development."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    config = load_tcn_config(tcn_config_path)
    root = Path(project_root).resolve(strict=True)
    _verify_tcn_config_location(root, tcn_config_path)
    _configure_deterministic_runtime()
    rf_config_path, rf_development_dir = _verify_development_trust_roots(root, config)
    rf_manifest = load_trusted_rf_development_manifest(
        rf_development_dir,
        expected_manifest_sha256=config["trust_roots"]["rf_development_manifest"]["sha256"],
    )
    bundle = load_preprocessing_bundle(
        rf_config_path=rf_config_path,
        project_root=root,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    _verify_bundle_identity(bundle, config)
    source_hashes = _source_hashes(root)

    cohorts: dict[tuple[str, str], TensorCohort] = {}
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        task_records = bundle.task_records(task)
        cohorts[(task, "train")] = build_tensor_cohort(task_records, task=task, split="train")
        cohorts[(task, "validation")] = build_tensor_cohort(task_records, task=task, split="validation")

    files: dict[str, bytes] = {}
    validation_metrics: dict[str, Any] = {
        "schema_version": "wandering-tcn-validation-metrics-v1",
        "metric_weighting": "unweighted_real_samples",
        "binary_primary_metric": "source_macro_macro_f1",
        "four_class_primary_metric": "wp_validation_macro_f1",
        "tasks": {},
    }
    failure_cases: dict[str, Any] = {
        "schema_version": "wandering-tcn-failure-cases-v1",
        "selection": "primary_seed_high_confidence_errors_top_10_per_cohort",
        "cohorts": {},
    }
    fingerprint_document: dict[str, Any] = {
        "schema_version": "wandering-tcn-model-fingerprints-v1",
        "models": {},
    }
    model_entries: dict[str, Any] = {}

    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        task_metrics: dict[str, Any] = {"seeds": {}}
        train = cohorts[(task, "train")]
        validation = cohorts[(task, "validation")]
        for seed in TCN_SEEDS:
            trained = _train_one_model(task=task, seed=seed, train=train, validation=validation, config=config)
            model = trained["model"]
            probabilities = trained["validation_probabilities"]
            predicted = _predicted_labels(probabilities, task)
            y_true = validation.labels.numpy()
            seed_metrics = _task_metrics(validation.records, task=task, y_true=y_true, predicted=predicted, probabilities=probabilities)
            task_metrics["seeds"][str(seed)] = seed_metrics
            prediction_rows = _prediction_rows(
                validation.records,
                task=task,
                seed=seed,
                y_true=y_true,
                predicted=predicted,
                probabilities=probabilities,
                split="validation",
            )
            files[f"validation_predictions/{task}/seed_{seed}.jsonl"] = _canonical_jsonl_bytes(prediction_rows)
            files[f"training_histories/{task}/seed_{seed}.json"] = _canonical_json_bytes(trained["history"])
            checkpoint = deterministic_npz_bytes(model)
            checkpoint_name = f"models/{task}/seed_{seed}.npz"
            files[checkpoint_name] = checkpoint
            fingerprint = model_semantic_fingerprint(model, task=task)
            fingerprint_document["models"][f"{task}/{seed}"] = fingerprint
            metadata = {
                "schema_version": "wandering-tcn-model-metadata-v1",
                "task": task,
                "seed": seed,
                "class_names": list(FOUR_CLASS_NAMES if task == TASK_FOUR_CLASS else BINARY_CLASS_NAMES),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "architecture": _architecture_document(task),
                "checkpoint_format": "deterministic_npz_v1",
                "checkpoint_artifact": checkpoint_name,
                "best_epoch": trained["history"]["selection"]["best_epoch"],
                "stopped_epoch": trained["history"]["selection"]["stopped_epoch"],
                "semantic_fingerprint": fingerprint,
                "probabilities_calibrated": False,
            }
            metadata_name = f"model_metadata/{task}/seed_{seed}.json"
            files[metadata_name] = _canonical_json_bytes(metadata)
            model_entries[f"{task}/{seed}"] = {
                "checkpoint": checkpoint_name,
                "metadata": metadata_name,
                "classes": list(range(4 if task == TASK_FOUR_CLASS else 2)),
                "semantic_fingerprint": fingerprint,
            }
            if seed == PRIMARY_SEED:
                for cohort_name, indices in _cohort_indices(validation.records, task, phase="validation").items():
                    failure_cases["cohorts"][f"{task}/{cohort_name}"] = _high_confidence_errors(
                        [validation.records[index] for index in indices],
                        y_true=y_true[indices],
                        predicted=predicted[indices],
                        probabilities=probabilities[indices],
                    )
        task_metrics["summary"] = _summarize_seed_metrics(task_metrics["seeds"], task=task, phase="validation")
        validation_metrics["tasks"][task] = task_metrics

    files["validation_metrics.json"] = _canonical_json_bytes(validation_metrics)
    files["failure_cases.json"] = _canonical_json_bytes(failure_cases)
    files["model_fingerprints.json"] = _canonical_json_bytes(fingerprint_document)
    # RF predictions/models never enter TCN optimization or checkpoint selection.
    # Parse the trusted RF validation metrics only after all TCN runs are complete.
    rf_metrics = _load_manifest_bound_json(rf_development_dir, rf_manifest, "validation_metrics.json")
    rf_comparison = _compare_with_rf(validation_metrics, rf_metrics, phase="validation")
    files["rf_validation_comparison.json"] = _canonical_json_bytes(rf_comparison)
    data_index = _development_data_index(bundle=bundle, cohorts=cohorts, config=config, source_hashes=source_hashes)
    files["data_index.json"] = _canonical_json_bytes(data_index)

    manifest = {
        "schema_version": DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        "purpose": "comparison_only",
        "tcn_config_sha256": _sha256_file(Path(tcn_config_path)),
        "rf_config_sha256": config["trust_roots"]["rf_config"]["sha256"],
        "rf_development_manifest_sha256": config["trust_roots"]["rf_development_manifest"]["sha256"],
        "preprocessing_manifest_sha256": bundle.input_hashes["preprocessing_manifest"],
        "split_sha256": bundle.split_sha256,
        "environment": current_environment_versions(),
        "source_hashes": source_hashes,
        "temporal_features_enabled": False,
        "input_shape": [80, 14],
        "fit_partition": "train",
        "validation_used_for_checkpoint_selection": True,
        "test_artifacts_present": False,
        "official_source_opened": False,
        "models": model_entries,
        "artifacts": _artifact_descriptors(files),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    return TCNDevelopmentBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest=manifest,
        validation_metrics=validation_metrics,
        failure_cases=failure_cases,
    )


def build_tcn_frozen_wp_test_artifacts(
    *,
    tcn_config_path: str | Path,
    project_root: str | Path,
    development_dir: str | Path,
    expected_development_manifest_sha256: str,
    expected_rf_public_manifest_sha256: str,
    output_dir: str | Path,
) -> TCNFrozenBenchmarkBuildResult:
    """Evaluate trusted train-only TCNs once on the fixed 240-row WP test."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"public benchmark output already exists: {output}")
    config = load_tcn_config(tcn_config_path)
    root = Path(project_root).resolve(strict=True)
    _verify_tcn_config_location(root, tcn_config_path)
    _configure_deterministic_runtime()
    rf_config_path = _resolve_bound_path(root, config["trust_roots"]["rf_config"], "rf_config")
    if expected_rf_public_manifest_sha256 != config["trust_roots"]["rf_public_manifest"]["sha256"]:
        raise ModelTrustError("RF public manifest external trust root does not match TCN config")
    development_manifest = load_trusted_tcn_development_manifest(
        development_dir,
        expected_manifest_sha256=expected_development_manifest_sha256,
        tcn_config_path=tcn_config_path,
        project_root=root,
    )
    models = {
        (task, seed): safe_load_tcn_model(
            development_dir,
            task=task,
            seed=seed,
            expected_manifest_sha256=expected_development_manifest_sha256,
            tcn_config_path=tcn_config_path,
            project_root=root,
        )
        for task in (TASK_FOUR_CLASS, TASK_BINARY)
        for seed in TCN_SEEDS
    }
    rf_public_path = _resolve_bound_path(root, config["trust_roots"]["rf_public_manifest"], "rf_public_manifest")
    rf_public_manifest = _load_trusted_manifest_directory(
        rf_public_path.parent,
        expected_manifest_sha256=expected_rf_public_manifest_sha256,
        expected_schema="wandering-rf-public-shape-benchmark-manifest-v1",
    )
    if rf_public_manifest.get("development_manifest_sha256") != config["trust_roots"]["rf_development_manifest"]["sha256"]:
        raise ModelTrustError("RF public manifest does not bind the fixed RF development manifest")
    rf_test_metrics = _load_manifest_bound_json(rf_public_path.parent, rf_public_manifest, "test_metrics.json")

    bundle = load_preprocessing_bundle(
        rf_config_path=rf_config_path,
        project_root=root,
        mode=BUNDLE_MODE_FROZEN_WP_TEST,
    )
    _verify_bundle_identity(bundle, config)
    cohorts = {
        task: build_tensor_cohort(bundle.task_records(task), task=task, split="test")
        for task in (TASK_FOUR_CLASS, TASK_BINARY)
    }
    files: dict[str, bytes] = {}
    metrics: dict[str, Any] = {
        "schema_version": "wandering-tcn-public-shape-benchmark-metrics-v1",
        "scope": "public_shape_benchmark",
        "limitations": [
            "4054_cross_partition_shape_neighbor_pairs_below_0.05",
            "not_person_level_generalization",
            "not_camera_validation",
            "not_real_elder_validation",
            "not_clinical_validation",
        ],
        "tasks": {},
    }
    failure_cases: dict[str, Any] = {
        "schema_version": "wandering-tcn-failure-cases-v1",
        "selection": "primary_seed_high_confidence_errors_top_10_per_cohort",
        "cohorts": {},
    }
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        cohort = cohorts[task]
        y_true = cohort.labels.numpy()
        task_metrics: dict[str, Any] = {"seeds": {}}
        for seed in TCN_SEEDS:
            probabilities = _predict_probabilities(models[(task, seed)], cohort)
            predicted = _predicted_labels(probabilities, task)
            seed_metrics = {
                "primary_metric": "wp_test_macro_f1",
                "primary_metric_value": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
                "cohorts": {"wp_test": _cohort_metrics(y_true, predicted, probabilities, task)},
            }
            task_metrics["seeds"][str(seed)] = seed_metrics
            predictions = _prediction_rows(
                cohort.records,
                task=task,
                seed=seed,
                y_true=y_true,
                predicted=predicted,
                probabilities=probabilities,
                split="test",
            )
            files[f"test_predictions/{task}/seed_{seed}.jsonl"] = _canonical_jsonl_bytes(predictions)
            if seed == PRIMARY_SEED:
                failure_cases["cohorts"][f"{task}/wp_test"] = _high_confidence_errors(
                    cohort.records,
                    y_true=y_true,
                    predicted=predicted,
                    probabilities=probabilities,
                )
        task_metrics["summary"] = _summarize_seed_metrics(task_metrics["seeds"], task=task, phase="test")
        metrics["tasks"][task] = task_metrics
    files["test_metrics.json"] = _canonical_json_bytes(metrics)
    files["failure_cases.json"] = _canonical_json_bytes(failure_cases)
    rf_comparison = _compare_with_rf(metrics, rf_test_metrics, phase="test")
    files["rf_test_comparison.json"] = _canonical_json_bytes(rf_comparison)
    manifest = {
        "schema_version": PUBLIC_BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "scope": "public_shape_benchmark",
        "comparison_only": True,
        "development_manifest_sha256": expected_development_manifest_sha256,
        "development_manifest_schema": development_manifest["schema_version"],
        "rf_public_manifest_sha256": expected_rf_public_manifest_sha256,
        "tcn_config_sha256": _sha256_file(Path(tcn_config_path)),
        "temporal_features_enabled": False,
        "split_sha256": bundle.split_sha256,
        "test_record_count": 240,
        "official_source_opened": False,
        "models_retrained": False,
        "artifacts": _artifact_descriptors(files),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    return TCNFrozenBenchmarkBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest=manifest,
        test_metrics=metrics,
        failure_cases=failure_cases,
    )


def load_trusted_tcn_development_manifest(
    development_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    tcn_config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    manifest = _load_trusted_manifest_directory(
        Path(development_dir),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_schema=DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
    )
    if manifest.get("environment") != current_environment_versions():
        raise ModelTrustError("TCN development dependency versions do not match")
    if manifest.get("tcn_config_sha256") != _sha256_file(Path(tcn_config_path)):
        raise ModelTrustError("current TCN config does not match trusted development")
    root = Path(project_root).resolve(strict=True)
    _verify_tcn_config_location(root, tcn_config_path)
    if manifest.get("source_hashes") != _source_hashes(root):
        raise ModelTrustError("current step-6 source files do not match trusted development")
    if manifest.get("official_source_opened") is not False or manifest.get("test_artifacts_present") is not False:
        raise ModelTrustError("trusted development phase boundary has drifted")
    return manifest


def safe_load_tcn_model(
    development_dir: str | Path,
    *,
    task: str,
    seed: int,
    expected_manifest_sha256: str,
    tcn_config_path: str | Path,
    project_root: str | Path,
) -> PureTCNBaseline:
    if task not in {TASK_FOUR_CLASS, TASK_BINARY} or seed not in TCN_SEEDS:
        raise ModelTrustError("unknown task or non-registered seed")
    root = Path(development_dir).resolve(strict=True)
    manifest = load_trusted_tcn_development_manifest(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        tcn_config_path=tcn_config_path,
        project_root=project_root,
    )
    entry = manifest.get("models", {}).get(f"{task}/{seed}")
    if not isinstance(entry, dict):
        raise ModelTrustError("TCN manifest model entry is missing")
    checkpoint_name = entry.get("checkpoint")
    metadata_name = entry.get("metadata")
    if checkpoint_name not in manifest["artifacts"] or metadata_name not in manifest["artifacts"]:
        raise ModelTrustError("TCN model files are not manifest-bound")
    metadata = _load_manifest_bound_json(root, manifest, metadata_name)
    expected_classes = list(range(4 if task == TASK_FOUR_CLASS else 2))
    if entry.get("classes") != expected_classes or metadata.get("task") != task or metadata.get("seed") != seed:
        raise ModelTrustError("TCN model task/seed/classes metadata drifted")
    checkpoint_path = _trusted_relative_path(root, checkpoint_name)
    payload = checkpoint_path.read_bytes()
    model = create_tcn_model(task, seed=seed)
    load_deterministic_npz(model, payload)
    actual_fingerprint = model_semantic_fingerprint(model, task=task)
    if entry.get("semantic_fingerprint") != actual_fingerprint or metadata.get("semantic_fingerprint") != actual_fingerprint:
        raise ModelTrustError("loaded TCN semantic fingerprint drifted")
    model.eval()
    return model


def benchmark_tcn_runtime(
    *,
    tcn_config_path: str | Path,
    project_root: str | Path,
    development_dir: str | Path,
    expected_development_manifest_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"runtime benchmark output already exists: {output}")
    config = load_tcn_config(tcn_config_path)
    root = Path(project_root).resolve(strict=True)
    _verify_tcn_config_location(root, tcn_config_path)
    _configure_deterministic_runtime()
    rf_config_path = _resolve_bound_path(root, config["trust_roots"]["rf_config"], "rf_config")
    manifest = load_trusted_tcn_development_manifest(
        development_dir,
        expected_manifest_sha256=expected_development_manifest_sha256,
        tcn_config_path=tcn_config_path,
        project_root=root,
    )
    bundle = load_preprocessing_bundle(
        rf_config_path=rf_config_path,
        project_root=root,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    models = {
        task: safe_load_tcn_model(
            development_dir,
            task=task,
            seed=PRIMARY_SEED,
            expected_manifest_sha256=expected_development_manifest_sha256,
            tcn_config_path=tcn_config_path,
            project_root=root,
        )
        for task in (TASK_FOUR_CLASS, TASK_BINARY)
    }
    warmup = int(config["runtime_benchmark"]["warmup_iterations"])
    iterations = int(config["runtime_benchmark"]["measurement_iterations"])
    result: dict[str, Any] = {
        "schema_version": "wandering-tcn-runtime-benchmark-v1",
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()},
        "warmup_iterations": warmup,
        "measurement_iterations": iterations,
        "clock": "perf_counter_ns",
        "deterministic_hash_excluded": True,
        "all_ten_checkpoint_bytes": sum(
            manifest["artifacts"][f"models/{task}/seed_{seed}.npz"]["byte_count"]
            for task in (TASK_FOUR_CLASS, TASK_BINARY)
            for seed in TCN_SEEDS
        ),
        "tasks": {},
    }
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        record = next(row for row in bundle.task_records(task) if row["split"] == "validation")
        cached_features, cached_mask = _single_record_tensors(record)
        model = models[task]
        with torch.no_grad():
            for _ in range(warmup):
                model(cached_features, cached_mask)
            tensor_ns: list[int] = []
            forward_ns: list[int] = []
            total_ns: list[int] = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                _single_record_tensors(record)
                tensor_ns.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                model(cached_features, cached_mask)
                forward_ns.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                features, mask = _single_record_tensors(record)
                model(features, mask)
                total_ns.append(time.perf_counter_ns() - start)
        result["tasks"][task] = {
            "sample_id": record["sample_id"],
            "tensor_construction_ms": _latency_summary(tensor_ns),
            "model_forward_ms": _latency_summary(forward_ns),
            "combined_ms": _latency_summary(total_ns),
            "primary_model_bytes": manifest["artifacts"][f"models/{task}/seed_{PRIMARY_SEED}.npz"]["byte_count"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    _write_new_json_file(output, result)
    return result


def _train_one_model(
    *,
    task: str,
    seed: int,
    train: TensorCohort,
    validation: TensorCohort,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = create_tcn_model(task, seed=seed).to(device="cpu", dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(train.features, train.masks, train.labels, train.weights)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs: list[dict[str, Any]] = []
    stopped_epoch = int(config["training"]["max_epochs"])
    stop_reason = "maximum_epochs"
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        loss_numerator = 0.0
        loss_denominator = 0.0
        for features, mask, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, mask)
            if task == TASK_FOUR_CLASS:
                loss = four_class_hierarchical_loss(outputs["gate_logit"], outputs["subtype_logits"], labels, weights)
            else:
                loss = binary_weighted_loss(outputs["binary_logit"], labels, weights)
            if not torch.isfinite(loss):
                raise TCNBaselineError("training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(gradient_norm):
                raise TCNBaselineError("gradient norm became non-finite")
            optimizer.step()
            batch_denominator = float(weights.sum())
            loss_numerator += float(loss.detach()) * batch_denominator
            loss_denominator += batch_denominator
        validation_probabilities = _predict_probabilities(model, validation)
        validation_predicted = _predicted_labels(validation_probabilities, task)
        validation_metric_document = _task_metrics(
            validation.records,
            task=task,
            y_true=validation.labels.numpy(),
            predicted=validation_predicted,
            probabilities=validation_probabilities,
        )
        metric = float(validation_metric_document["primary_metric_value"])
        validation_loss = _validation_loss(model, validation, task)
        improved = metric > best_metric + 1e-4
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        epochs.append(
            {
                "epoch": epoch,
                "train_loss": loss_numerator / loss_denominator,
                "validation_loss": validation_loss,
                "validation_primary_metric": metric,
                "strict_improvement": improved,
            }
        )
        if epoch >= 30 and epoch - best_epoch >= 20:
            stopped_epoch = epoch
            stop_reason = "early_stopping_patience"
            break
    if best_state is None or best_epoch < 1:
        raise TCNBaselineError("training never produced a finite best checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    final_probabilities = _predict_probabilities(model, validation)
    selection = select_early_stopping_epoch(
        [row["validation_primary_metric"] for row in epochs],
        minimum_epochs=30,
        patience=20,
        minimum_improvement=1e-4,
    )
    if selection["best_epoch"] != best_epoch or selection["stopped_epoch"] != stopped_epoch or selection["stop_reason"] != stop_reason:
        raise TCNBaselineError("online early stopping and frozen selection audit disagree")
    history = {
        "schema_version": "wandering-tcn-training-history-v1",
        "task": task,
        "seed": seed,
        "fit_partition": "train",
        "validation_partition": "validation",
        "test_seen": False,
        "official_source_opened": False,
        "optimizer": dict(config["training"]["optimizer"]),
        "batch_size": 48,
        "num_workers": 0,
        "gradient_clip_global_norm": 1.0,
        "scheduler": "none",
        "amp": False,
        "augmentation": "none",
        "epochs": epochs,
        "selection": selection,
    }
    return {"model": model, "history": history, "validation_probabilities": final_probabilities}


def _validation_loss(model: PureTCNBaseline, cohort: TensorCohort, task: str) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(cohort.features, cohort.masks)
        unit_weights = torch.ones(len(cohort.records), dtype=torch.float32)
        if task == TASK_FOUR_CLASS:
            loss = four_class_hierarchical_loss(outputs["gate_logit"], outputs["subtype_logits"], cohort.labels, unit_weights)
        else:
            loss = binary_weighted_loss(outputs["binary_logit"], cohort.labels, unit_weights)
    return float(loss)


def _predict_probabilities(model: PureTCNBaseline, cohort: TensorCohort) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(cohort.features, cohort.masks),
        batch_size=48,
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for features, masks in loader:
            result = model(features, masks)
            if model.task == TASK_FOUR_CLASS:
                probabilities = four_class_probabilities(result["gate_logit"], result["subtype_logits"])
            else:
                positive = torch.sigmoid(result["binary_logit"])
                probabilities = torch.cat((1.0 - positive, positive), dim=1)
            outputs.append(probabilities.detach().cpu().numpy().astype(np.float64))
    matrix = np.concatenate(outputs, axis=0)
    expected_width = 4 if model.task == TASK_FOUR_CLASS else 2
    if matrix.shape != (len(cohort.records), expected_width) or not np.isfinite(matrix).all():
        raise TCNBaselineError("prediction probability matrix drifted")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise TCNBaselineError("prediction probabilities do not sum to one")
    return matrix


def _configure_deterministic_runtime() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise TCNBaselineError("PyTorch inter-op threads must be set to 1 before training") from exc
    torch.use_deterministic_algorithms(True)


def _verify_development_trust_roots(root: Path, config: Mapping[str, Any]) -> tuple[Path, Path]:
    rf_config_path = _resolve_bound_path(root, config["trust_roots"]["rf_config"], "rf_config")
    rf_development_manifest = _resolve_bound_path(
        root,
        config["trust_roots"]["rf_development_manifest"],
        "rf_development_manifest",
    )
    # Deliberately do not resolve/read rf_public_manifest in development.
    return rf_config_path, rf_development_manifest.parent


def _verify_tcn_config_location(root: Path, config_path: str | Path) -> None:
    expected = (root / "configs/modules/wandering_tcn_v1.yaml").resolve(strict=False)
    actual = Path(config_path).resolve(strict=False)
    if actual != expected or not actual.is_file():
        raise ModelTrustError("TCN config must be the fixed repository path")


def _verify_bundle_identity(bundle: Any, config: Mapping[str, Any]) -> None:
    if bundle.input_hashes.get("preprocessing_manifest") != config["preprocessing"]["manifest_sha256"]:
        raise TCNBaselineError("step-4 preprocessing manifest identity drifted")
    if bundle.split_sha256 != config["preprocessing"]["split_sha256"]:
        raise TCNBaselineError("step-4 split identity drifted")
    if bundle.total_record_count != 1790 or bundle.ready_count != 1775 or bundle.unavailable_count != 15:
        raise TCNBaselineError("step-4 denominator counts drifted")


def _resolve_bound_path(root: Path, descriptor: Mapping[str, Any], role: str) -> Path:
    path = (root / Path(str(descriptor.get("path", "")))).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelTrustError(f"TCN trust-root path escapes project root: {role}") from exc
    if not path.is_file() or _sha256_file(path) != descriptor.get("sha256"):
        raise ModelTrustError(f"TCN trust-root file/hash mismatch: {role}")
    return path


def _source_hashes(root: Path) -> dict[str, str]:
    relative_paths = (
        "src/elderly_monitoring/modules/mental_health/wandering/tcn_baseline.py",
        "scripts/wandering/train_tcn_baseline.py",
        "scripts/wandering/evaluate_tcn_baseline.py",
        "scripts/wandering/benchmark_tcn_baseline.py",
    )
    output: dict[str, str] = {}
    for relative in relative_paths:
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TCNBaselineError("step-6 source path escapes project root") from exc
        if not path.is_file():
            raise TCNBaselineError(f"step-6 source file is missing: {relative}")
        output[relative] = _sha256_file(path)
    return output


def _load_trusted_manifest_directory(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_schema: str,
) -> dict[str, Any]:
    if not _valid_sha256(expected_manifest_sha256):
        raise ModelTrustError("expected manifest SHA-256 is invalid")
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ModelTrustError("manifest does not match the external trust root")
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelTrustError("cannot parse trusted manifest") from exc
    if not isinstance(manifest, dict) or raw != _canonical_json_bytes(manifest) or manifest.get("schema_version") != expected_schema:
        raise ModelTrustError("trusted manifest schema or canonical encoding drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ModelTrustError("trusted manifest has no artifacts")
    for relative, descriptor in artifacts.items():
        path = _trusted_relative_path(root, relative)
        if not path.is_file() or descriptor != {"byte_count": path.stat().st_size, "sha256": _sha256_file(path)}:
            raise ModelTrustError(f"manifest-bound artifact mismatch: {relative}")
    return manifest


def _load_manifest_bound_json(root: Path, manifest: Mapping[str, Any], relative: str) -> dict[str, Any]:
    if relative not in manifest.get("artifacts", {}):
        raise ModelTrustError(f"JSON artifact is not manifest-bound: {relative}")
    path = _trusted_relative_path(Path(root), relative)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelTrustError(f"cannot parse manifest-bound JSON: {relative}") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ModelTrustError(f"manifest-bound JSON is not canonical: {relative}")
    return value


def _trusted_relative_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ModelTrustError("manifest artifact path is invalid")
    path = (root / Path(relative)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelTrustError("manifest artifact escapes its bundle") from exc
    return path


def _development_data_index(
    *,
    bundle: Any,
    cohorts: Mapping[tuple[str, str], TensorCohort],
    config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "wandering-tcn-data-index-v1",
        "purpose": "comparison_only",
        "temporal_features_enabled": False,
        "input_shape": [80, 14],
        "input_dtype": "float32",
        "mask_channel_index": 12,
        "time_channel_indices": [10, 11],
        "bundle_denominator": {
            "total_records": bundle.total_record_count,
            "ready": bundle.ready_count,
            "unavailable": bundle.unavailable_count,
            "unavailable_sample_ids": [row["sample_id"] for row in bundle.unavailable_records()],
            "unavailable_entered_tensor_or_loss": False,
        },
        "task_counts": {
            task: {split: len(cohorts[(task, split)].records) for split in ("train", "validation")}
            for task in (TASK_FOUR_CLASS, TASK_BINARY)
        },
        "task_source_counts": {
            task: {
                split: dict(sorted(Counter(row["source_dataset"] for row in cohorts[(task, split)].records).items()))
                for split in ("train", "validation")
            }
            for task in (TASK_FOUR_CLASS, TASK_BINARY)
        },
        "input_hashes": dict(sorted(bundle.input_hashes.items())),
        "split_sha256": bundle.split_sha256,
        "environment": current_environment_versions(),
        "cpu": platform.processor() or platform.machine(),
        "source_hashes": dict(source_hashes),
        "seeds": list(TCN_SEEDS),
        "primary_seed": PRIMARY_SEED,
        "independent_task_models": True,
        "development_constraints": {
            "fit_partitions": ["train"],
            "evaluation_partitions": ["validation"],
            "test_tensor_generated": False,
            "test_predictions_generated": False,
            "test_metrics_generated": False,
            "rf_public_manifest_opened": False,
            "official_source_opened": False,
            "augmentation_used": False,
            "probability_calibration_used": False,
            "rejection_used": False,
        },
        "training_config": dict(config["training"]),
    }


def _architecture_document(task: str) -> dict[str, Any]:
    return {
        "model_type": "pure_tcn_baseline",
        "task": task,
        "input_shape": [80, 14],
        "input_projection": {"in_channels": 14, "out_channels": 64, "kernel_size": 1, "bias": True},
        "blocks": _EXPECTED_CONFIG["model"]["blocks"],
        "block_order": ["layer_norm", "gelu", "depthwise_conv1d", "pointwise_conv1d", "gelu", "dropout", "residual_add", "mask"],
        "channel_change_residual_bias": False,
        "pooling": "masked_mean",
        "head": "gate_96_64_1_plus_subtype_96_64_3" if task == TASK_FOUR_CLASS else "binary_96_64_1",
        "parameter_count": TCN_FOUR_CLASS_PARAMETER_COUNT if task == TASK_FOUR_CLASS else TCN_BINARY_PARAMETER_COUNT,
        "forbidden_components_present": False,
    }


def _predicted_labels(probabilities: np.ndarray, task: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    width = 4 if task == TASK_FOUR_CLASS else 2
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise TCNBaselineError("probability matrix shape drifted")
    if task == TASK_FOUR_CLASS:
        return np.argmax(values, axis=1).astype(np.int64)
    if task == TASK_BINARY:
        return (values[:, 1] >= 0.5).astype(np.int64)
    raise TCNBaselineError("unknown prediction task")


def _task_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    cohorts = {
        name: _cohort_metrics(y_true[indices], predicted[indices], probabilities[indices], task)
        for name, indices in _cohort_indices(records, task, phase="validation").items()
    }
    if task == TASK_FOUR_CLASS:
        primary_name = "wp_validation_macro_f1"
        primary_value = cohorts["wp_validation"]["macro_f1"]
    else:
        primary_name = "source_macro_macro_f1"
        primary_value = float(np.mean([cohorts["wp_validation"]["macro_f1"], cohorts["smartcare_validation"]["macro_f1"]]))
    return {"primary_metric": primary_name, "primary_metric_value": primary_value, "cohorts": cohorts}


def _cohort_indices(records: Sequence[Mapping[str, Any]], task: str, *, phase: str) -> dict[str, np.ndarray]:
    suffix = "validation" if phase == "validation" else "test"
    if task == TASK_FOUR_CLASS:
        return {f"wp_{suffix}": np.arange(len(records), dtype=np.int64)}
    sources = np.asarray([row["source_dataset"] for row in records], dtype=object)
    output = {f"wp_{suffix}": np.flatnonzero(sources == "wandering_patterns")}
    if phase == "validation":
        output["smartcare_validation"] = np.flatnonzero(sources == "smartcare")
    return output


def _cohort_metrics(y_true: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray, task: str) -> dict[str, Any]:
    labels = list(range(4 if task == TASK_FOUR_CLASS else 2))
    class_names = FOUR_CLASS_NAMES if task == TASK_FOUR_CLASS else BINARY_CLASS_NAMES
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predicted, labels=labels, zero_division=0)
    result: dict[str, Any] = {
        "sample_count": len(y_true),
        "class_counts": {str(label): int(np.count_nonzero(y_true == label)) for label in labels},
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).astype(int).tolist(),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "per_class": {
            str(label): {
                "class_name": class_names[label],
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in labels
        },
    }
    if task == TASK_BINARY:
        positive = np.clip(probabilities[:, 1], 1e-15, 1.0 - 1e-15)
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        result.update(
            {
                "auroc": float(roc_auc_score(y_true, probabilities[:, 1])),
                "auprc": float(average_precision_score(y_true, probabilities[:, 1])),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "brier": float(brier_score_loss(y_true, probabilities[:, 1])),
                "nll": float(log_loss(y_true, np.column_stack((1.0 - positive, positive)), labels=[0, 1])),
                "positive_probability_clip": [1e-15, 1.0 - 1e-15],
                "decision_threshold": 0.5,
                "probabilities_calibrated": False,
            }
        )
    else:
        result["decision_rule"] = "argmax"
    return result


def _summarize_seed_metrics(seeds: Mapping[str, Mapping[str, Any]], *, task: str, phase: str) -> dict[str, Any]:
    primary = [value["primary_metric_value"] for value in seeds.values()]
    cohort_names = sorted(next(iter(seeds.values()))["cohorts"])
    cohorts: dict[str, Any] = {}
    for cohort in cohort_names:
        cohorts[cohort] = {
            name: _mean_std([seed["cohorts"][cohort][name] for seed in seeds.values()])
            for name in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        if task == TASK_BINARY:
            for name in ("auroc", "auprc", "sensitivity", "specificity", "brier", "nll"):
                cohorts[cohort][name] = _mean_std([seed["cohorts"][cohort][name] for seed in seeds.values()])
    return {
        "seed_count": len(seeds),
        "primary_metric": "wp_test_macro_f1" if phase == "test" else ("wp_validation_macro_f1" if task == TASK_FOUR_CLASS else "source_macro_macro_f1"),
        "primary_metric_mean_std": _mean_std(primary),
        "population_std_ddof": 0,
        "cohorts": cohorts,
        "primary_artifact_seed": PRIMARY_SEED,
        "seed_selection_used": False,
    }


def _prediction_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    split: str,
) -> list[dict[str, Any]]:
    class_names = FOUR_CLASS_NAMES if task == TASK_FOUR_CLASS else BINARY_CLASS_NAMES
    return [
        {
            "schema_version": PREDICTION_ROW_SCHEMA_VERSION,
            "task": task,
            "seed": seed,
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
            "split": split,
            "true_label": int(truth),
            "true_class_name": class_names[int(truth)],
            "predicted_label": int(prediction),
            "predicted_class_name": class_names[int(prediction)],
            "probabilities": [float(value) for value in probs],
            "correct": bool(truth == prediction),
            "probabilities_calibrated": False,
        }
        for row, truth, prediction, probs in zip(records, y_true, predicted, probabilities, strict=True)
    ]


def _high_confidence_errors(
    records: Sequence[Mapping[str, Any]],
    *,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    errors = [
        {
            "sample_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
            "true_label": int(truth),
            "predicted_label": int(prediction),
            "predicted_confidence": float(probs[int(prediction)]),
            "probabilities": probs.tolist(),
        }
        for row, truth, prediction, probs in zip(records, y_true, predicted, probabilities, strict=True)
        if int(truth) != int(prediction)
    ]
    errors.sort(key=lambda row: (-row["predicted_confidence"], row["sample_id"]))
    return errors[:10]


def _compare_with_rf(tcn_metrics: Mapping[str, Any], rf_metrics: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": f"wandering-tcn-rf-{phase}-comparison-v1",
        "comparison": "same_seed_tcn_minus_rf",
        "statistical_significance_claimed": False,
        "tasks": {},
    }
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        seeds = {}
        for seed in TCN_SEEDS:
            key = str(seed)
            tcn_value = float(tcn_metrics["tasks"][task]["seeds"][key]["primary_metric_value"])
            rf_value = float(rf_metrics["tasks"][task]["seeds"][key]["primary_metric_value"])
            seeds[key] = {"tcn": tcn_value, "rf": rf_value, "tcn_minus_rf": tcn_value - rf_value}
        output["tasks"][task] = {
            "primary_metric": tcn_metrics["tasks"][task]["seeds"][str(PRIMARY_SEED)]["primary_metric"],
            "seeds": seeds,
            "delta_mean_std": _mean_std([value["tcn_minus_rf"] for value in seeds.values()]),
        }
    return output


def _single_record_tensors(record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(np.asarray(record["model_features"], dtype=np.float32).reshape(1, 80, 14).copy())
    mask = torch.from_numpy(np.asarray(record["point_mask"], dtype=np.float32).reshape(1, 80).copy())
    validate_model_inputs(features, mask)
    return features, mask


def _artifact_descriptors(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(files.items())
    }


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _canonical_without_lf(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _VALID_SHA256


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _latency_summary(values_ns: Sequence[int]) -> dict[str, float]:
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "median": float(np.quantile(values_ms, 0.50, method="linear")),
        "p95": float(np.quantile(values_ms, 0.95, method="linear")),
    }


def _write_new_json_file(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(dict(value))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "BINARY_CLASS_NAMES",
    "FOUR_CLASS_NAMES",
    "PRIMARY_SEED",
    "TASK_BINARY",
    "TASK_FOUR_CLASS",
    "TCN_BINARY_PARAMETER_COUNT",
    "TCN_FOUR_CLASS_PARAMETER_COUNT",
    "TCN_SEEDS",
    "ModelTrustError",
    "PureTCNBaseline",
    "TCNBaselineError",
    "TCNDevelopmentBuildResult",
    "TCNFrozenBenchmarkBuildResult",
    "benchmark_tcn_runtime",
    "binary_weighted_loss",
    "build_binary_sample_weights",
    "build_tcn_development_artifacts",
    "build_tcn_frozen_wp_test_artifacts",
    "build_tensor_cohort",
    "create_tcn_model",
    "deterministic_npz_bytes",
    "four_class_hierarchical_loss",
    "four_class_probabilities",
    "load_deterministic_npz",
    "load_tcn_config",
    "load_trusted_tcn_development_manifest",
    "model_semantic_fingerprint",
    "safe_load_tcn_model",
    "select_early_stopping_epoch",
    "validate_model_inputs",
]
