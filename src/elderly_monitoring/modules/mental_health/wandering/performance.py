"""Supervised TopoWander-MPT development training and joint validation.

This module deliberately exposes only the fixed public development view of the
step-4 preprocessing bundle.  It never requests the WP frozen test partition,
SmartCare official data, camera data, or the shared mental-health runtime.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from .model import (
    TOPOWANDER_PARAMETER_COUNT,
    TopoWanderMPT,
    create_topowander_model,
    create_topowander_training_model,
    deterministic_npz_bytes,
    hierarchical_four_class_probabilities,
    load_topowander_config,
    safe_load_topowander_model,
)
from .preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)


PERFORMANCE_SCHEMA_VERSION = "wandering-performance-config-v2"
CHECKPOINT_SCHEMA_VERSION = "wandering-performance-checkpoint-v2"
METRICS_SCHEMA_VERSION = "wandering-performance-joint-metrics-v1"
PREDICTION_SCHEMA_VERSION = "wandering-performance-prediction-v2"
DEVELOPMENT_BUNDLE_SCHEMA_VERSION = "wandering-performance-development-bundle-v2"
M0S_SEEDS = (20260731, 20260801, 20260802)
M0S_PRIMARY_SEED = 20260731
PATTERN_LABELS = ("direct", "pacing", "lapping", "random")
SUBTYPE_LABELS = ("pacing", "lapping", "random")
BINARY_LABELS = ("direct_or_non_wandering", "wandering_like")
ALLOWED_SOURCES = ("wandering_patterns", "smartcare")
_SHA256_CHARS = frozenset("0123456789abcdef")
_EPOCH_DIRECTORY = re.compile(r"^epoch-(\d{4})$")


def _data_access_audit() -> dict[str, Any]:
    """Return the explicit M0-S data boundary without conflating parse and use."""

    return {
        "shared_preprocessing_container": {
            "contains_wp_test_rows": True,
            "wp_test_integrity_parsed": True,
            "purpose": "whole-container integrity validation only",
        },
        "development_accessor": {
            "exposed_splits": ["train", "validation"],
            "wp_test_exposed": False,
        },
        "wp_frozen_test": {
            "materialized": False,
            "tensorized": False,
            "used_for_inference": False,
            "used_for_scoring": False,
            "used_for_training": False,
            "used_for_selection": False,
        },
        "smartcare_official": {"opened": False},
        "sealed_camera": {"opened": False},
    }


class PerformanceConfigError(ValueError):
    """The performance configuration is invalid or mixes model concerns."""


class PerformanceDataError(ValueError):
    """A supervised sample, batch, logit, label, or metric input is invalid."""


class CheckpointError(ValueError):
    """A numeric checkpoint is incomplete, inconsistent, or unsafe to load."""


@dataclass(frozen=True)
class TrainingWeights:
    binary_group_weights: Mapping[str, float]
    subtype_class_weights: tuple[float, float, float]
    binary_group_counts: Mapping[str, int]
    subtype_class_counts: Mapping[str, int]


@dataclass(frozen=True)
class JointEvaluation:
    metrics: Mapping[str, Any]
    predictions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LoadedCheckpoint:
    epoch: int
    model: TopoWanderMPT
    optimizer: torch.optim.Optimizer
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingRunResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_metrics: Mapping[str, Any]
    history: tuple[dict[str, Any], ...]
    stopped_reason: str


@dataclass(frozen=True)
class OverfitSmokeResult:
    run_dir: Path
    checkpoint: Path
    metrics: Mapping[str, Any]
    passed: bool


@dataclass(frozen=True)
class FreshEvaluationResult:
    output_dir: Path
    metrics: Mapping[str, Any]
    predictions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RecoveryResult:
    latest_checkpoint: Path
    best_checkpoint: Path
    history: tuple[dict[str, Any], ...]
    repaired: tuple[str, ...]


@dataclass(frozen=True)
class StabilityAggregationResult:
    output_dir: Path
    report: Mapping[str, Any]


class WanderingPerformanceDataset(Dataset[dict[str, Any]]):
    """Validated train/validation tensors for the single joint model."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise PerformanceDataError("performance dataset cannot be empty")
        validated = tuple(_validate_performance_record(record) for record in records)
        self.records = validated
        self.model_features = torch.from_numpy(
            np.stack([record["model_features_array"] for record in validated]).astype(np.float32, copy=False)
        )
        self.shape_normalized_points = torch.from_numpy(
            np.stack([record["shape_normalized_points_array"] for record in validated]).astype(np.float32, copy=False)
        )
        self.point_mask = torch.from_numpy(
            np.stack([record["point_mask_array"] for record in validated]).astype(np.float32, copy=False)
        )
        self.binary_labels = torch.tensor([record["binary_label"] for record in validated], dtype=torch.long)
        self.subtype_labels = torch.tensor([record["subtype_label"] for record in validated], dtype=torch.long)
        self.pattern_labels = torch.tensor([record["pattern_label_index"] for record in validated], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "model_features": self.model_features[index],
            "shape_normalized_points": self.shape_normalized_points[index],
            "point_mask": self.point_mask[index],
            "binary_label": self.binary_labels[index],
            "subtype_label": self.subtype_labels[index],
            "pattern_label": self.pattern_labels[index],
            "sample_id": record["sample_id"],
            "source_dataset": record["source_dataset"],
            "split": record["split"],
            "pattern_label_name": record["pattern_label_name"],
        }


def collate_performance_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack the exact three model inputs and keep evaluation identities."""

    if not samples:
        raise PerformanceDataError("cannot collate an empty performance batch")
    tensor_fields = (
        "model_features",
        "shape_normalized_points",
        "point_mask",
        "binary_label",
        "subtype_label",
        "pattern_label",
    )
    try:
        result = {field: torch.stack([sample[field] for sample in samples]) for field in tensor_fields}
    except (KeyError, TypeError, RuntimeError) as exc:
        raise PerformanceDataError("performance batch tensor fields cannot be stacked") from exc
    result.update(
        {
            "sample_id": tuple(str(sample["sample_id"]) for sample in samples),
            "source_dataset": tuple(str(sample["source_dataset"]) for sample in samples),
            "split": tuple(str(sample["split"]) for sample in samples),
            "pattern_label_name": tuple(str(sample["pattern_label_name"]) for sample in samples),
        }
    )
    if result["model_features"].shape[1:] != (80, 14):
        raise PerformanceDataError("collated model_features shape must be [B,80,14]")
    if result["shape_normalized_points"].shape[1:] != (80, 2):
        raise PerformanceDataError("collated shape_normalized_points shape must be [B,80,2]")
    if result["point_mask"].shape[1:] != (80,):
        raise PerformanceDataError("collated point_mask shape must be [B,80]")
    return result


def load_performance_config(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    """Load the training-only config while keeping the forward config separate."""

    resolved = Path(path)
    try:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PerformanceConfigError(f"cannot read performance config: {resolved}") from exc
    expected_fields = {
        "schema_version",
        "purpose",
        "model",
        "data",
        "runtime",
        "training",
        "loss",
        "optimizer",
        "early_stopping",
        "checkpoint",
        "smoke",
        "reporting",
        "baselines",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise PerformanceConfigError("performance config fields have drifted")
    if value["schema_version"] != PERFORMANCE_SCHEMA_VERSION or value["purpose"] != "supervised_development":
        raise PerformanceConfigError("performance config schema or purpose is invalid")
    if any(name in value for name in ("input", "tcn", "patch", "relation", "transformer", "heads", "state")):
        raise PerformanceConfigError("forward architecture fields cannot enter performance config")

    model = value["model"]
    if not isinstance(model, dict) or set(model) != {"forward_config_path", "initialization"}:
        raise PerformanceConfigError("model performance fields have drifted")
    if model["initialization"] != "scratch":
        raise PerformanceConfigError("M0 performance config must use scratch initialization")
    forward_relative = _relative_path(model["forward_config_path"], "model.forward_config_path")
    root = Path(project_root).resolve(strict=True)
    forward_path = (root / forward_relative).resolve(strict=True)
    if root not in forward_path.parents or not forward_path.is_file():
        raise PerformanceConfigError("forward config must be a file inside project_root")
    load_topowander_config(forward_path)

    data = value["data"]
    if not isinstance(data, dict) or set(data) != {
        "rf_config_path",
        "development_bundle_path",
        "bundle_mode",
    }:
        raise PerformanceConfigError("data performance fields have drifted")
    if data["bundle_mode"] != "development":
        raise PerformanceConfigError("performance config may expose only development train/validation")
    rf_path = (root / _relative_path(data["rf_config_path"], "data.rf_config_path")).resolve(strict=True)
    if root not in rf_path.parents or not rf_path.is_file():
        raise PerformanceConfigError("RF config must be a file inside project_root")
    _relative_path(data["development_bundle_path"], "data.development_bundle_path")

    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "device",
        "intra_op_threads",
        "inter_op_threads",
        "num_workers",
        "pin_memory",
    }:
        raise PerformanceConfigError("runtime fields have drifted")
    if runtime["device"] != "cpu":
        raise PerformanceConfigError(
            "M0-S runtime.device must be cpu; CUDA resume is not claimed by this configuration"
        )
    _positive_int(runtime["intra_op_threads"], "runtime.intra_op_threads")
    _positive_int(runtime["inter_op_threads"], "runtime.inter_op_threads")
    if not isinstance(runtime["num_workers"], int) or isinstance(runtime["num_workers"], bool) or runtime["num_workers"] < 0:
        raise PerformanceConfigError("runtime.num_workers must be a non-negative integer")
    if not isinstance(runtime["pin_memory"], bool):
        raise PerformanceConfigError("runtime.pin_memory must be boolean")

    training = value["training"]
    if not isinstance(training, dict) or set(training) != {
        "seeds",
        "primary_seed",
        "max_epochs",
        "batch_size",
        "gradient_clip_norm",
        "shuffle",
    }:
        raise PerformanceConfigError("training fields have drifted")
    if training["seeds"] != list(M0S_SEEDS) or training["primary_seed"] != M0S_PRIMARY_SEED:
        raise PerformanceConfigError("M0-S seeds and primary seed are fixed")
    _positive_int(training["max_epochs"], "training.max_epochs")
    if training["max_epochs"] > 50:
        raise PerformanceConfigError("M0 max_epochs cannot exceed 50")
    _positive_int(training["batch_size"], "training.batch_size")
    _positive_float(training["gradient_clip_norm"], "training.gradient_clip_norm")
    if training["shuffle"] is not True:
        raise PerformanceConfigError("training.shuffle must be true")

    loss = value["loss"]
    if not isinstance(loss, dict) or set(loss) != {
        "subtype_loss_weight",
        "binary_weighting",
        "subtype_weighting",
    }:
        raise PerformanceConfigError("loss fields have drifted")
    _positive_float(loss["subtype_loss_weight"], "loss.subtype_loss_weight")
    if loss["binary_weighting"] != "train_equal_source_class":
        raise PerformanceConfigError("binary weighting must be computed from train source/class groups")
    if loss["subtype_weighting"] != "train_inverse_frequency":
        raise PerformanceConfigError("subtype weighting must be computed from train only")

    optimizer = value["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "name",
        "learning_rate",
        "weight_decay",
        "betas",
        "eps",
    }:
        raise PerformanceConfigError("optimizer fields have drifted")
    if optimizer["name"] != "adamw":
        raise PerformanceConfigError("only AdamW is supported")
    _positive_float(optimizer["learning_rate"], "optimizer.learning_rate")
    _nonnegative_float(optimizer["weight_decay"], "optimizer.weight_decay")
    if (
        not isinstance(optimizer["betas"], list)
        or len(optimizer["betas"]) != 2
        or not all(isinstance(item, float) and 0.0 < item < 1.0 for item in optimizer["betas"])
    ):
        raise PerformanceConfigError("optimizer.betas must contain two floats in (0,1)")
    _positive_float(optimizer["eps"], "optimizer.eps")

    early = value["early_stopping"]
    if not isinstance(early, dict) or set(early) != {"min_epochs", "patience"}:
        raise PerformanceConfigError("early_stopping fields have drifted")
    _positive_int(early["min_epochs"], "early_stopping.min_epochs")
    _positive_int(early["patience"], "early_stopping.patience")
    if early["min_epochs"] > training["max_epochs"]:
        raise PerformanceConfigError("early_stopping.min_epochs exceeds max_epochs")

    checkpoint = value["checkpoint"]
    if checkpoint != {"save_every_epochs": 1, "keep_epoch_checkpoints": True}:
        raise PerformanceConfigError("checkpoint policy must preserve every epoch in a fresh run")

    smoke = value["smoke"]
    if not isinstance(smoke, dict) or set(smoke) != {
        "seed",
        "samples_per_pattern",
        "max_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "maximum_final_loss_ratio",
        "mask_invariance_atol",
    }:
        raise PerformanceConfigError("smoke fields have drifted")
    _positive_int(smoke["seed"], "smoke.seed")
    _positive_int(smoke["samples_per_pattern"], "smoke.samples_per_pattern")
    _positive_int(smoke["max_epochs"], "smoke.max_epochs")
    _positive_int(smoke["batch_size"], "smoke.batch_size")
    _positive_float(smoke["learning_rate"], "smoke.learning_rate")
    _nonnegative_float(smoke["weight_decay"], "smoke.weight_decay")
    if not isinstance(smoke["maximum_final_loss_ratio"], float) or not 0.0 < smoke["maximum_final_loss_ratio"] < 1.0:
        raise PerformanceConfigError("smoke.maximum_final_loss_ratio must be in (0,1)")
    _positive_float(smoke["mask_invariance_atol"], "smoke.mask_invariance_atol")

    reporting = value["reporting"]
    if not isinstance(reporting, dict) or set(reporting) != {
        "target_macro_f1",
        "class_recall_gate",
        "representative_error_limit",
        "inference_batch_size",
        "inference_warmup_iterations",
        "inference_measurement_iterations",
    }:
        raise PerformanceConfigError("reporting fields have drifted")
    for name in ("target_macro_f1", "class_recall_gate"):
        if not isinstance(reporting[name], float) or not 0.0 < reporting[name] <= 1.0:
            raise PerformanceConfigError(f"reporting.{name} must be in (0,1]")
    _positive_int(reporting["representative_error_limit"], "reporting.representative_error_limit")
    _positive_int(reporting["inference_batch_size"], "reporting.inference_batch_size")
    _positive_int(reporting["inference_warmup_iterations"], "reporting.inference_warmup_iterations")
    _positive_int(reporting["inference_measurement_iterations"], "reporting.inference_measurement_iterations")

    baselines = value["baselines"]
    if not isinstance(baselines, dict) or set(baselines) != {"rf", "tcn"}:
        raise PerformanceConfigError("baseline prediction sources have drifted")
    for baseline_name, task_paths in baselines.items():
        if not isinstance(task_paths, dict) or set(task_paths) != {"four_class", "binary"}:
            raise PerformanceConfigError(f"baselines.{baseline_name} task paths have drifted")
        for task, template in task_paths.items():
            if not isinstance(template, str) or template.count("{seed}") != 1:
                raise PerformanceConfigError(
                    f"baselines.{baseline_name}.{task} must contain exactly one {{seed}} token"
                )
            for seed in M0S_SEEDS:
                baseline_path = (root / _relative_path(
                    template.format(seed=seed), f"baselines.{baseline_name}.{task}"
                )).resolve(strict=True)
                if root not in baseline_path.parents or not baseline_path.is_file():
                    raise PerformanceConfigError("baseline predictions must be files inside project_root")
    return value


def _validate_m0s_seed(config: Mapping[str, Any], seed: int) -> None:
    configured = tuple(config["training"]["seeds"])
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in configured:
        raise PerformanceConfigError(
            f"seed must be one of the fixed M0-S seeds: {list(configured)}"
        )


def compute_training_weights(records: Sequence[Mapping[str, Any]]) -> TrainingWeights:
    """Compute source/class and subtype weights from train records only."""

    validated = tuple(_validate_performance_record(record) for record in records)
    if any(record["split"] != "train" for record in validated):
        raise PerformanceDataError("training weights can be computed from train records only")
    group_counts = Counter(f"{record['source_dataset']}:{record['binary_label']}" for record in validated)
    if not group_counts:
        raise PerformanceDataError("binary training groups cannot be empty")
    total = len(validated)
    group_total = total / len(group_counts)
    binary_weights = {group: float(group_total / count) for group, count in sorted(group_counts.items())}

    subtype_counts = Counter(
        SUBTYPE_LABELS[record["subtype_label"]]
        for record in validated
        if record["subtype_label"] >= 0
    )
    if set(subtype_counts) != set(SUBTYPE_LABELS):
        raise PerformanceDataError("train records must contain pacing, lapping, and random subtype supervision")
    subtype_total = sum(subtype_counts.values())
    subtype_weights = tuple(float(subtype_total / (len(SUBTYPE_LABELS) * subtype_counts[name])) for name in SUBTYPE_LABELS)
    return TrainingWeights(
        binary_group_weights=binary_weights,
        subtype_class_weights=subtype_weights,
        binary_group_counts=dict(sorted(group_counts.items())),
        subtype_class_counts={name: int(subtype_counts[name]) for name in SUBTYPE_LABELS},
    )


def joint_supervised_loss(
    binary_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    binary_labels: torch.Tensor,
    subtype_labels: torch.Tensor,
    *,
    binary_sample_weights: torch.Tensor,
    subtype_class_weights: torch.Tensor,
    subtype_loss_weight: float,
) -> torch.Tensor:
    """Binary loss for all samples plus subtype loss only for legal subtypes."""

    total, _binary, _subtype = _joint_loss_parts(
        binary_logits,
        subtype_logits,
        binary_labels,
        subtype_labels,
        binary_sample_weights=binary_sample_weights,
        subtype_class_weights=subtype_class_weights,
        subtype_loss_weight=subtype_loss_weight,
    )
    return total


def evaluate_joint_predictions(
    records: Sequence[Mapping[str, Any]],
    binary_logits: np.ndarray | torch.Tensor | Sequence[float],
    subtype_logits: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
) -> JointEvaluation:
    """Evaluate both tasks from one model and one ordered prediction table."""

    validated = tuple(_validate_performance_record(record) for record in records)
    if not validated:
        raise PerformanceDataError("joint evaluator requires at least one record")
    binary = _to_numpy(binary_logits)
    subtype = _to_numpy(subtype_logits)
    if binary.ndim == 2 and binary.shape[1] == 1:
        binary = binary[:, 0]
    if binary.shape != (len(validated),):
        raise PerformanceDataError("binary logits shape must be [N] or [N,1]")
    if subtype.shape != (len(validated), 3):
        raise PerformanceDataError("subtype logits shape must be [N,3]")
    if not np.isfinite(binary).all() or not np.isfinite(subtype).all():
        raise PerformanceDataError("joint evaluator logits must be finite")

    binary_probability = 1.0 / (1.0 + np.exp(-np.clip(binary, -60.0, 60.0)))
    shifted = subtype - np.max(subtype, axis=1, keepdims=True)
    subtype_probability = np.exp(shifted)
    subtype_probability /= np.sum(subtype_probability, axis=1, keepdims=True)
    four_probability = np.concatenate(
        ((1.0 - binary_probability)[:, None], binary_probability[:, None] * subtype_probability),
        axis=1,
    )
    if not np.isfinite(four_probability).all() or not np.allclose(four_probability.sum(axis=1), 1.0, atol=1.0e-7):
        raise PerformanceDataError("derived four-class probabilities must be finite and normalized")
    binary_predicted = (binary_probability >= 0.5).astype(np.int64)
    pattern_predicted = np.argmax(four_probability, axis=1).astype(np.int64)

    wp_indices = np.asarray(
        [index for index, record in enumerate(validated) if record["source_dataset"] == "wandering_patterns"],
        dtype=np.int64,
    )
    smartcare_indices = np.asarray(
        [index for index, record in enumerate(validated) if record["source_dataset"] == "smartcare"],
        dtype=np.int64,
    )
    if wp_indices.size == 0 or smartcare_indices.size == 0:
        raise PerformanceDataError("joint evaluator requires both WP and SmartCare records")
    pattern_true = np.asarray([record["pattern_label_index"] for record in validated], dtype=np.int64)
    binary_true = np.asarray([record["binary_label"] for record in validated], dtype=np.int64)
    if np.any(pattern_true[wp_indices] < 0):
        raise PerformanceDataError("WP evaluator record has invalid pattern label")

    wp_four = _classification_metrics(
        pattern_true[wp_indices],
        pattern_predicted[wp_indices],
        labels=PATTERN_LABELS,
    )
    wp_binary = _classification_metrics(
        binary_true[wp_indices],
        binary_predicted[wp_indices],
        labels=BINARY_LABELS,
    )
    smartcare_binary = _classification_metrics(
        binary_true[smartcare_indices],
        binary_predicted[smartcare_indices],
        labels=BINARY_LABELS,
    )
    pooled_binary = _classification_metrics(binary_true, binary_predicted, labels=BINARY_LABELS)
    source_equal = float((wp_binary["macro_f1"] + smartcare_binary["macro_f1"]) / 2.0)
    selection_score = float(min(wp_four["macro_f1"], source_equal))
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "wp_four_class": wp_four,
        "binary": {
            "wp": wp_binary,
            "smartcare": smartcare_binary,
            "pooled": pooled_binary,
            "source_equal_macro_f1": source_equal,
        },
        "wandering_positive_recall": {
            "wp": float(wp_binary["per_class"]["wandering_like"]["recall"]),
            "smartcare": float(smartcare_binary["per_class"]["wandering_like"]["recall"]),
            "source_equal": float(
                (
                    wp_binary["per_class"]["wandering_like"]["recall"]
                    + smartcare_binary["per_class"]["wandering_like"]["recall"]
                )
                / 2.0
            ),
            "pooled": float(pooled_binary["per_class"]["wandering_like"]["recall"]),
        },
        "selection_score": selection_score,
        "sample_counts": {
            "total": len(validated),
            "wp": int(wp_indices.size),
            "smartcare": int(smartcare_indices.size),
        },
    }
    predictions = []
    for index, record in enumerate(validated):
        true_pattern = record["pattern_label_name"] if record["pattern_label_index"] >= 0 else None
        predictions.append(
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "sample_id": record["sample_id"],
                "source_dataset": record["source_dataset"],
                "split": record["split"],
                "true_binary_label": int(binary_true[index]),
                "predicted_binary_label": int(binary_predicted[index]),
                "binary_probability": float(binary_probability[index]),
                "true_pattern_label": true_pattern,
                "predicted_pattern_label": PATTERN_LABELS[int(pattern_predicted[index])],
                "subtype_probabilities": {
                    name: float(subtype_probability[index, subtype_index])
                    for subtype_index, name in enumerate(SUBTYPE_LABELS)
                },
                "four_class_probabilities": {
                    name: float(four_probability[index, pattern_index])
                    for pattern_index, name in enumerate(PATTERN_LABELS)
                },
            }
        )
    return JointEvaluation(metrics=metrics, predictions=tuple(predictions))


def selection_rank(
    selection_score: float,
    wp_four_class_macro_f1: float,
    source_equal_binary_macro_f1: float,
    validation_loss: float,
    epoch: int,
) -> tuple[float, float, float, float, int]:
    """Fixed best-checkpoint ordering; larger tuples are better."""

    values = (selection_score, wp_four_class_macro_f1, source_equal_binary_macro_f1, validation_loss)
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise PerformanceDataError("selection values must be finite")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise PerformanceDataError("selection epoch must be a positive integer")
    return (
        float(selection_score),
        float(wp_four_class_macro_f1),
        float(source_equal_binary_macro_f1),
        -float(validation_loss),
        -epoch,
    )


def create_training_optimizer(
    model: TopoWanderMPT,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
) -> torch.optim.AdamW:
    """Create the one optimizer for trunk plus binary/subtype heads.

    The historical projection branch is not part of the supervised M0 loss and
    is explicitly frozen; it remains in the strict full-state checkpoint.
    """

    if not isinstance(model, TopoWanderMPT):
        raise PerformanceDataError("training optimizer requires TopoWanderMPT")
    _positive_float(learning_rate, "learning_rate", error_type=PerformanceDataError)
    _nonnegative_float(weight_decay, "weight_decay", error_type=PerformanceDataError)
    for parameter in model.projection_head.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise PerformanceDataError("joint model has no optimized parameters")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)


def save_training_checkpoint(
    checkpoints_root: str | Path,
    *,
    epoch: int,
    seed: int,
    model: TopoWanderMPT,
    optimizer: torch.optim.Optimizer,
    forward_config_path: str | Path,
    performance_config_sha256: str,
    metrics: Mapping[str, Any],
) -> Path:
    """Save a new, hash-bound numeric epoch checkpoint without overwriting."""

    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise CheckpointError("checkpoint epoch must be a positive integer")
    if seed not in M0S_SEEDS:
        raise CheckpointError("checkpoint seed is not one of the fixed M0-S seeds")
    if not _valid_sha256(performance_config_sha256):
        raise CheckpointError("performance config SHA-256 must be lowercase hex")
    root = Path(checkpoints_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"epoch-{epoch:04d}"
    if final.exists():
        raise CheckpointError(f"checkpoint already exists: {final}")
    forward = Path(forward_config_path)
    try:
        forward_bytes = forward.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"cannot read forward config: {forward}") from exc
    forward_sha256 = hashlib.sha256(forward_bytes).hexdigest()
    load_topowander_config(forward)
    model_bytes = deterministic_npz_bytes(model)
    optimizer_bytes = _optimizer_npz_bytes(optimizer, model)
    group = optimizer.param_groups[0]
    optimizer_metadata = {
        "name": "adamw",
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "optimized_parameter_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
    }
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "seed": seed,
        "forward_config_sha256": forward_sha256,
        "performance_config_sha256": performance_config_sha256,
        "model_state": {"path": "model_state.npz", "sha256": hashlib.sha256(model_bytes).hexdigest()},
        "optimizer_state": {
            "path": "optimizer_state.npz",
            "sha256": hashlib.sha256(optimizer_bytes).hexdigest(),
        },
        "optimizer": optimizer_metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimized_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "metrics": _jsonable(metrics),
        "rng_resume": {
            "torch_cpu_rng_saved_in_optimizer_state": True,
            "torch_cpu_rng_restored": True,
            "cuda_rng_saved": False,
            "cuda_resume_supported": False,
            "reason": "the fixed M0-S runtime configuration is CPU-only",
        },
        "data_access": _data_access_audit(),
    }
    if metadata["parameter_count"] != TOPOWANDER_PARAMETER_COUNT:
        raise CheckpointError("checkpoint model parameter count drifted")
    temporary = Path(tempfile.mkdtemp(prefix=f".epoch-{epoch:04d}-", dir=root))
    try:
        (temporary / "model_state.npz").write_bytes(model_bytes)
        (temporary / "optimizer_state.npz").write_bytes(optimizer_bytes)
        (temporary / "metadata.json").write_bytes(_canonical_json_bytes(metadata))
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    forward_config_path: str | Path,
    expected_performance_config_sha256: str,
    expected_seed: int,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
) -> LoadedCheckpoint:
    """Verify and restore full model, AdamW numeric state, epoch, and RNG."""

    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {checkpoint}")
    metadata = _load_json(checkpoint / "metadata.json", CheckpointError)
    expected_fields = {
        "schema_version",
        "epoch",
        "seed",
        "forward_config_sha256",
        "performance_config_sha256",
        "model_state",
        "optimizer_state",
        "optimizer",
        "parameter_count",
        "optimized_parameter_count",
        "metrics",
        "rng_resume",
        "data_access",
    }
    if set(metadata) != expected_fields or metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("checkpoint metadata schema has drifted")
    if metadata["performance_config_sha256"] != expected_performance_config_sha256:
        raise CheckpointError("checkpoint performance config binding mismatch")
    if metadata["seed"] != expected_seed or expected_seed not in M0S_SEEDS:
        raise CheckpointError("checkpoint seed binding mismatch")
    if metadata["parameter_count"] != TOPOWANDER_PARAMETER_COUNT:
        raise CheckpointError("checkpoint parameter count mismatch")
    forward = Path(forward_config_path)
    forward_sha256 = _sha256_file(forward)
    if metadata["forward_config_sha256"] != forward_sha256:
        raise CheckpointError("checkpoint forward config binding mismatch")
    if metadata["data_access"] != _data_access_audit():
        raise CheckpointError("checkpoint data-access boundary has drifted")
    if metadata["rng_resume"] != {
        "torch_cpu_rng_saved_in_optimizer_state": True,
        "torch_cpu_rng_restored": True,
        "cuda_rng_saved": False,
        "cuda_resume_supported": False,
        "reason": "the fixed M0-S runtime configuration is CPU-only",
    }:
        raise CheckpointError("checkpoint RNG-resume contract has drifted")
    model_path = _bound_checkpoint_file(checkpoint, metadata["model_state"], "model state")
    optimizer_path = _bound_checkpoint_file(checkpoint, metadata["optimizer_state"], "optimizer state")
    model = safe_load_topowander_model(
        model_path,
        config_path=forward,
        expected_config_sha256=forward_sha256,
    )
    optimizer = create_training_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    expected_optimizer = metadata["optimizer"]
    if (
        expected_optimizer.get("name") != "adamw"
        or float(expected_optimizer.get("learning_rate", -1.0)) != float(learning_rate)
        or float(expected_optimizer.get("weight_decay", -1.0)) != float(weight_decay)
        or tuple(expected_optimizer.get("betas", ())) != tuple(float(value) for value in betas)
        or float(expected_optimizer.get("eps", -1.0)) != float(eps)
        or expected_optimizer.get("optimized_parameter_names")
        != [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    ):
        raise CheckpointError("checkpoint optimizer configuration mismatch")
    _load_optimizer_npz(optimizer_path, optimizer, model)
    epoch = metadata["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise CheckpointError("checkpoint epoch is invalid")
    return LoadedCheckpoint(epoch=epoch, model=model, optimizer=optimizer, metadata=metadata)


def load_development_records(
    config: Mapping[str, Any], *, project_root: str | Path
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], Mapping[str, Any]]:
    """Load the materialized train/validation-only performance bundle."""

    root = Path(project_root).resolve(strict=True)
    bundle_root = (root / str(config["data"]["development_bundle_path"])).resolve(strict=True)
    if root not in bundle_root.parents or not bundle_root.is_dir():
        raise PerformanceDataError("development performance bundle must be inside project_root")
    manifest = _load_json(bundle_root / "manifest.json", PerformanceDataError)
    expected_fields = {
        "schema_version",
        "records",
        "record_count",
        "counts_by_split",
        "counts_by_split_source",
        "split_sha256",
        "preprocessing_config_sha256",
        "source_input_hashes",
        "source_integrity_report",
        "data_access",
    }
    if set(manifest) != expected_fields or manifest["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION:
        raise PerformanceDataError("development performance bundle manifest has drifted")
    if manifest["data_access"] != _data_access_audit():
        raise PerformanceDataError("development performance bundle data boundary has drifted")
    records_path = _bound_development_file(bundle_root, manifest["records"])
    rows = tuple(_load_jsonl(records_path))
    if len(rows) != 1535 or manifest["record_count"] != 1535:
        raise PerformanceDataError("development performance bundle record count drifted")
    for row in rows:
        _validate_performance_record(row)
    train = tuple(row for row in rows if row["split"] == "train")
    validation = tuple(row for row in rows if row["split"] == "validation")
    _validate_development_counts(train, validation)
    actual_by_split = dict(sorted(Counter(row["split"] for row in rows).items()))
    actual_by_split_source = {
        f"{split}:{source}": count
        for (split, source), count in sorted(Counter((row["split"], row["source_dataset"]) for row in rows).items())
    }
    if manifest["counts_by_split"] != actual_by_split or manifest["counts_by_split_source"] != actual_by_split_source:
        raise PerformanceDataError("development performance bundle manifest counts drifted")
    return train, validation, {
        "split_sha256": manifest["split_sha256"],
        "preprocessing_config_sha256": manifest["preprocessing_config_sha256"],
        "input_hashes": dict(manifest["source_input_hashes"]),
        "integrity_report": dict(manifest["source_integrity_report"]),
        "development_records_sha256": manifest["records"]["sha256"],
        "shared_container_wp_test_integrity_parsed": True,
        "wp_test_exposed_to_development_accessor": False,
        "wp_test_materialized": False,
        "wp_test_used_for_inference": False,
        "wp_test_used_for_scoring": False,
        "wp_test_used_for_training": False,
        "wp_test_used_for_selection": False,
        "smartcare_official_opened": False,
        "sealed_camera_opened": False,
    }


def materialize_development_bundle(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Create a fresh train/validation-only bundle from the guarded accessor."""

    root = Path(project_root).resolve(strict=True)
    config = load_performance_config(config_path, project_root=root)
    configured = (root / str(config["data"]["development_bundle_path"])).resolve()
    output = configured if output_dir is None else Path(output_dir).resolve()
    if output != configured or root not in output.parents:
        raise PerformanceDataError("development bundle output must equal the configured project path")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing development bundle: {output}")
    bundle = load_preprocessing_bundle(
        rf_config_path=root / str(config["data"]["rf_config_path"]),
        project_root=root,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    train = tuple(bundle.records_for_split("train"))
    validation = tuple(bundle.records_for_split("validation"))
    _validate_development_counts(train, validation)
    rows = train + validation
    records_bytes = _canonical_jsonl_bytes(rows)
    manifest = {
        "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        "records": {"path": "records.jsonl", "sha256": hashlib.sha256(records_bytes).hexdigest()},
        "record_count": len(rows),
        "counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "counts_by_split_source": {
            f"{split}:{source}": count
            for (split, source), count in sorted(
                Counter((row["split"], row["source_dataset"]) for row in rows).items()
            )
        },
        "split_sha256": bundle.split_sha256,
        "preprocessing_config_sha256": bundle.preprocessing_config_sha256,
        "source_input_hashes": dict(bundle.input_hashes),
        "source_integrity_report": dict(bundle.integrity_report),
        "data_access": _data_access_audit(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "records.jsonl").write_bytes(records_bytes)
        (temporary / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _validate_development_counts(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]
) -> None:
    if len(train) != 1257 or len(validation) != 278:
        raise PerformanceDataError("fixed development train/validation counts drifted")
    if Counter(record["source_dataset"] for record in train) != Counter(
        {"wandering_patterns": 1120, "smartcare": 137}
    ):
        raise PerformanceDataError("fixed train source counts drifted")
    if Counter(record["source_dataset"] for record in validation) != Counter(
        {"wandering_patterns": 240, "smartcare": 38}
    ):
        raise PerformanceDataError("fixed validation source counts drifted")


def run_overfit_smoke(
    config_path: str | Path,
    *,
    project_root: str | Path,
    run_dir: str | Path,
) -> OverfitSmokeResult:
    """Overfit a four-pattern train subset and verify gradients, mask, and reload."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_performance_config(config_file, project_root=root)
    output = _create_new_run_dir(run_dir)
    train_records, _validation_records, data_identity = load_development_records(config, project_root=root)
    smoke = config["smoke"]
    selected: list[Mapping[str, Any]] = []
    for pattern in PATTERN_LABELS:
        candidates = sorted(
            (
                record
                for record in train_records
                if record["source_dataset"] == "wandering_patterns" and record["pattern_label"] == pattern
            ),
            key=lambda record: str(record["sample_id"]),
        )
        if len(candidates) < int(smoke["samples_per_pattern"]):
            raise PerformanceDataError(f"overfit smoke lacks train samples for {pattern}")
        selected.extend(candidates[: int(smoke["samples_per_pattern"])])
    dataset = WanderingPerformanceDataset(selected)
    weights = compute_training_weights(selected)
    _configure_runtime(config["runtime"])
    device = _resolve_device(config["runtime"]["device"])
    torch.manual_seed(int(smoke["seed"]))
    forward_path = root / str(config["model"]["forward_config_path"])
    model = create_topowander_model(load_topowander_config(forward_path))
    optimizer = create_training_optimizer(
        model,
        learning_rate=float(smoke["learning_rate"]),
        weight_decay=float(smoke["weight_decay"]),
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )
    model.to(device)
    batch = collate_performance_batch([dataset[index] for index in range(len(dataset))])
    batch = _batch_to_device(batch, device)
    binary_weights = _batch_binary_weights(batch, weights, device)
    subtype_weights = torch.tensor(weights.subtype_class_weights, dtype=torch.float32, device=device)
    subtype_loss_weight = float(config["loss"]["subtype_loss_weight"])
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(("binary_head.", "subtype_head."))
    }
    model.eval()
    with torch.no_grad():
        initial_outputs = model(
            batch["model_features"], batch["shape_normalized_points"], batch["point_mask"]
        )
        initial_loss = joint_supervised_loss(
            initial_outputs["binary_logit"],
            initial_outputs["subtype_logits"],
            batch["binary_label"],
            batch["subtype_label"],
            binary_sample_weights=binary_weights,
            subtype_class_weights=subtype_weights,
            subtype_loss_weight=subtype_loss_weight,
        )
    gradient_max = {"trunk": 0.0, "binary_head": 0.0, "subtype_head": 0.0}
    losses = [float(initial_loss)]
    for _epoch in range(1, int(smoke["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["model_features"], batch["shape_normalized_points"], batch["point_mask"])
        loss = joint_supervised_loss(
            outputs["binary_logit"],
            outputs["subtype_logits"],
            batch["binary_label"],
            batch["subtype_label"],
            binary_sample_weights=binary_weights,
            subtype_class_weights=subtype_weights,
            subtype_loss_weight=subtype_loss_weight,
        )
        if not torch.isfinite(loss):
            raise PerformanceDataError("overfit smoke loss became non-finite")
        loss.backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise PerformanceDataError(f"overfit smoke gradient became non-finite: {name}")
            group = "binary_head" if name.startswith("binary_head.") else (
                "subtype_head" if name.startswith("subtype_head.") else "trunk"
            )
            gradient_max[group] = max(gradient_max[group], float(parameter.grad.detach().abs().max()))
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            evaluated = model(
                batch["model_features"], batch["shape_normalized_points"], batch["point_mask"]
            )
            evaluated_loss = joint_supervised_loss(
                evaluated["binary_logit"],
                evaluated["subtype_logits"],
                batch["binary_label"],
                batch["subtype_label"],
                binary_sample_weights=binary_weights,
                subtype_class_weights=subtype_weights,
                subtype_loss_weight=subtype_loss_weight,
            )
        losses.append(float(evaluated_loss))
        if losses[-1] <= losses[0] * float(smoke["maximum_final_loss_ratio"]):
            break
    final_loss = losses[-1]
    trunk_update_max = max(
        float((parameter.detach() - before[name]).abs().max())
        for name, parameter in model.named_parameters()
        if name in before
    )
    mask_difference = _mask_invariance_difference(model, batch, float(smoke["mask_invariance_atol"]))
    model.to("cpu")
    _optimizer_to_device(optimizer, torch.device("cpu"))
    performance_sha256 = _sha256_file(config_file)
    checkpoint = save_training_checkpoint(
        output / "checkpoints",
        epoch=len(losses) - 1,
        seed=int(smoke["seed"]),
        model=model,
        optimizer=optimizer,
        forward_config_path=forward_path,
        performance_config_sha256=performance_sha256,
        metrics={"initial_loss": losses[0], "final_loss": final_loss},
    )
    restored = load_training_checkpoint(
        checkpoint,
        forward_config_path=forward_path,
        expected_performance_config_sha256=performance_sha256,
        expected_seed=int(smoke["seed"]),
        learning_rate=float(smoke["learning_rate"]),
        weight_decay=float(smoke["weight_decay"]),
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )
    model.eval()
    restored.model.eval()
    cpu_batch = _batch_to_device(batch, torch.device("cpu"))
    with torch.no_grad():
        original_outputs = model(
            cpu_batch["model_features"], cpu_batch["shape_normalized_points"], cpu_batch["point_mask"]
        )
        restored_outputs = restored.model(
            cpu_batch["model_features"], cpu_batch["shape_normalized_points"], cpu_batch["point_mask"]
        )
    reload_difference = max(
        float((original_outputs[name] - restored_outputs[name]).abs().max())
        for name in ("binary_logit", "subtype_logits")
    )
    passed = bool(
        final_loss <= losses[0] * float(smoke["maximum_final_loss_ratio"])
        and all(value > 0.0 and math.isfinite(value) for value in gradient_max.values())
        and trunk_update_max > 0.0
        and mask_difference <= float(smoke["mask_invariance_atol"])
        and reload_difference <= 1.0e-7
        and all(math.isfinite(value) for value in losses)
    )
    smoke_metrics = {
        "schema_version": "wandering-performance-overfit-smoke-v1",
        "passed": passed,
        "sample_count": len(dataset),
        "patterns": list(PATTERN_LABELS),
        "epochs_run": len(losses) - 1,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "final_loss_ratio": final_loss / losses[0],
        "gradient_max": gradient_max,
        "trunk_update_max": trunk_update_max,
        "mask_invariance_max_abs_difference": mask_difference,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint": str(checkpoint),
        "data_identity": data_identity,
        "data_access": _data_access_audit(),
    }
    _write_new_bytes(output / "smoke_metrics.json", _canonical_json_bytes(smoke_metrics))
    _write_new_bytes(
        output / "README.md",
        (
            "# TopoWander-MPT M0 overfit smoke\n\n"
            f"- Result: {'PASS' if passed else 'FAIL'}\n"
            f"- Samples: {len(dataset)} (direct/pacing/lapping/random)\n"
            f"- Loss: {losses[0]:.6f} -> {final_loss:.6f}\n"
            f"- Epochs: {len(losses) - 1}\n"
            f"- Checkpoint: `{checkpoint}`\n"
            "- WP frozen test / SmartCare official / sealed camera consumed: no\n"
        ).encode("utf-8"),
    )
    return OverfitSmokeResult(run_dir=output, checkpoint=checkpoint, metrics=smoke_metrics, passed=passed)


def train_performance_model(
    config_path: str | Path,
    *,
    project_root: str | Path,
    run_dir: str | Path,
    seed: int,
    resume: bool = False,
) -> TrainingRunResult:
    """Run or resume one explicit seed of M0-S development training."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_performance_config(config_file, project_root=root)
    _validate_m0s_seed(config, seed)
    output = Path(run_dir)
    if resume:
        if not output.is_dir():
            raise CheckpointError("resume run directory does not exist")
    else:
        output = _create_new_run_dir(output)
    lock = output / ".training.lock"
    _acquire_run_lock(lock)
    try:
        return _train_performance_model_locked(config_file, config, root, output, seed=seed, resume=resume)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def recover_training_run(
    config_path: str | Path,
    *,
    project_root: str | Path,
    run_dir: str | Path,
    seed: int,
) -> RecoveryResult:
    """Reconcile history and pointers from contiguous, fully verified checkpoints.

    A checkpoint directory becomes the epoch commit only after its atomic rename.
    Therefore a crash may leave history/latest/best behind, but those mutable
    indexes never override a complete immutable checkpoint.
    """

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_performance_config(config_file, project_root=root)
    _validate_m0s_seed(config, seed)
    run = Path(run_dir).resolve(strict=True)
    checkpoints_root = (run / "checkpoints").resolve(strict=True)
    optimizer_config = config["optimizer"]
    performance_sha256 = _sha256_file(config_file)
    forward_path = root / str(config["model"]["forward_config_path"])

    indexed: list[tuple[int, Path]] = []
    for candidate in checkpoints_root.iterdir():
        if not candidate.is_dir():
            continue
        match = _EPOCH_DIRECTORY.fullmatch(candidate.name)
        if match is not None:
            indexed.append((int(match.group(1)), candidate))
    indexed.sort()
    if not indexed:
        raise CheckpointError("resume requires at least one complete epoch checkpoint")
    epochs = [epoch for epoch, _ in indexed]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise CheckpointError("complete epoch checkpoints must form a contiguous prefix starting at one")

    canonical_history: list[dict[str, Any]] = []
    ranks: list[tuple[tuple[float, float, float, float, int], Path]] = []
    for expected_epoch, checkpoint in indexed:
        loaded = load_training_checkpoint(
            checkpoint,
            forward_config_path=forward_path,
            expected_performance_config_sha256=performance_sha256,
            expected_seed=seed,
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )
        if loaded.epoch != expected_epoch:
            raise CheckpointError("checkpoint directory and metadata epochs disagree")
        metrics = dict(loaded.metadata["metrics"])
        if metrics.get("epoch") != expected_epoch:
            raise CheckpointError("checkpoint metrics epoch disagrees with committed epoch")
        joint = metrics.get("joint_metrics")
        if not isinstance(joint, dict) or not isinstance(joint.get("binary"), dict):
            raise CheckpointError("checkpoint lacks joint metrics required for recovery")
        rank = selection_rank(
            float(joint["selection_score"]),
            float(joint["wp_four_class"]["macro_f1"]),
            float(joint["binary"]["source_equal_macro_f1"]),
            float(metrics["validation_loss"]),
            expected_epoch,
        )
        canonical_history.append(metrics)
        ranks.append((rank, checkpoint))

    latest_checkpoint = indexed[-1][1]
    best_rank, best_checkpoint = max(ranks, key=lambda item: item[0])
    repaired: list[str] = []
    history_path = run / "training_history.jsonl"
    try:
        existing_history = _load_jsonl(history_path)
    except (OSError, UnicodeError, json.JSONDecodeError, PerformanceDataError, CheckpointError):
        existing_history = []
    if existing_history != canonical_history:
        _write_atomic_bytes(history_path, _canonical_jsonl_bytes(canonical_history))
        repaired.append("training_history")

    latest_value = {"epoch": epochs[-1], "checkpoint": latest_checkpoint.name}
    try:
        existing_latest = _load_json(checkpoints_root / "latest.json", CheckpointError)
    except CheckpointError:
        existing_latest = {}
    if existing_latest != latest_value:
        _write_atomic_json(checkpoints_root / "latest.json", latest_value)
        repaired.append("latest_pointer")

    best_value = {
        "epoch": int(best_checkpoint.name.removeprefix("epoch-")),
        "checkpoint": best_checkpoint.name,
        "selection_rank": list(best_rank),
    }
    try:
        existing_best = _load_json(checkpoints_root / "best.json", CheckpointError)
    except CheckpointError:
        existing_best = {}
    if existing_best != best_value:
        _write_atomic_json(checkpoints_root / "best.json", best_value)
        repaired.append("best_pointer")

    if repaired:
        _append_jsonl(
            run / "recovery_log.jsonl",
            {
                "event": "resume_indexes_reconciled",
                "seed": seed,
                "authoritative_epoch": epochs[-1],
                "best_epoch": best_value["epoch"],
                "repaired": repaired,
            },
        )
    return RecoveryResult(
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
        history=tuple(canonical_history),
        repaired=tuple(repaired),
    )


def evaluate_fresh_checkpoint(
    config_path: str | Path,
    *,
    project_root: str | Path,
    run_dir: str | Path,
    seed: int,
) -> FreshEvaluationResult:
    """Fresh-process best-checkpoint reload, validation, inference, and report."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_performance_config(config_file, project_root=root)
    _validate_m0s_seed(config, seed)
    run = Path(run_dir).resolve(strict=True)
    fresh_dir = run / "fresh_reload"
    if fresh_dir.exists():
        raise FileExistsError(f"fresh reload output already exists: {fresh_dir}")
    best_pointer = _load_json(run / "checkpoints" / "best.json", CheckpointError)
    checkpoint = (run / "checkpoints" / str(best_pointer["checkpoint"])).resolve(strict=True)
    if (run / "checkpoints").resolve() not in checkpoint.parents:
        raise CheckpointError("best checkpoint pointer escapes checkpoint root")
    optimizer_config = config["optimizer"]
    loaded = load_training_checkpoint(
        checkpoint,
        forward_config_path=root / str(config["model"]["forward_config_path"]),
        expected_performance_config_sha256=_sha256_file(config_file),
        expected_seed=seed,
        learning_rate=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    train_records, validation_records, data_identity = load_development_records(config, project_root=root)
    weights = compute_training_weights(train_records)
    _configure_runtime(config["runtime"])
    device = _resolve_device(config["runtime"]["device"])
    loaded.model.to(device)
    _optimizer_to_device(loaded.optimizer, device)
    validation_dataset = WanderingPerformanceDataset(validation_records)
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=seed,
        runtime=config["runtime"],
    )
    validation_loss, joint = _evaluate_model(
        loaded.model,
        validation_loader,
        validation_records,
        weights,
        device=device,
        subtype_loss_weight=float(config["loss"]["subtype_loss_weight"]),
    )
    saved_metrics = loaded.metadata["metrics"]
    saved_joint = saved_metrics.get("joint_metrics")
    if not isinstance(saved_joint, dict):
        raise CheckpointError("best checkpoint lacks saved joint metrics")
    comparisons = {
        "wp_four_class_macro_f1": abs(
            float(joint.metrics["wp_four_class"]["macro_f1"])
            - float(saved_joint["wp_four_class"]["macro_f1"])
        ),
        "source_equal_binary_macro_f1": abs(
            float(joint.metrics["binary"]["source_equal_macro_f1"])
            - float(saved_joint["binary"]["source_equal_macro_f1"])
        ),
        "selection_score": abs(
            float(joint.metrics["selection_score"]) - float(saved_joint["selection_score"])
        ),
        "validation_loss": abs(float(validation_loss) - float(saved_metrics["validation_loss"])),
    }
    fresh_match = all(value <= 1.0e-10 for value in comparisons.values())
    if not fresh_match:
        raise CheckpointError(f"fresh reload metrics do not match saved best metrics: {comparisons}")
    target = float(config["reporting"]["target_macro_f1"])
    recall_gate = float(config["reporting"]["class_recall_gate"])
    class_recalls = _all_reported_class_recalls(joint.metrics)
    attained = bool(
        joint.metrics["wp_four_class"]["macro_f1"] >= target
        and joint.metrics["binary"]["source_equal_macro_f1"] >= target
        and all(value >= recall_gate for value in class_recalls.values())
    )
    history = _load_jsonl(run / "training_history.jsonl")
    summary = _load_json(run / "training_summary.json", CheckpointError)
    errors = _representative_errors(
        joint.predictions,
        limit=int(config["reporting"]["representative_error_limit"]),
    )
    largest_shortfall = _largest_shortfall(joint.metrics)
    single = _single_sample_inference(loaded.model, validation_dataset[0], device)
    latency = _benchmark_fixed_inference_batch(
        loaded.model,
        validation_dataset,
        device=device,
        batch_size=int(config["reporting"]["inference_batch_size"]),
        warmup_iterations=int(config["reporting"]["inference_warmup_iterations"]),
        measurement_iterations=int(config["reporting"]["inference_measurement_iterations"]),
    )
    seeded_predictions = tuple({**row, "seed": seed} for row in joint.predictions)
    model_state_path = _bound_checkpoint_file(checkpoint, loaded.metadata["model_state"], "model state")
    if summary.get("seed") != seed:
        raise CheckpointError("training summary seed does not match requested fresh evaluation seed")
    resolved = _load_json(run / "resolved_config.json", CheckpointError)
    if resolved.get("seed") != seed:
        raise CheckpointError("resolved config seed does not match requested fresh evaluation seed")
    report_metrics = {
        "schema_version": "wandering-performance-fresh-report-v2",
        "candidate": summary.get("candidate", "M0-S"),
        "initialization": config["model"]["initialization"],
        "seed": seed,
        "max_epochs": int(config["training"]["max_epochs"]),
        "best_epoch": loaded.epoch,
        "parameter_count": sum(parameter.numel() for parameter in loaded.model.parameters()),
        "optimized_parameter_count": sum(
            parameter.numel() for parameter in loaded.model.parameters() if parameter.requires_grad
        ),
        "model_state_bytes": model_state_path.stat().st_size,
        "fixed_batch_inference_latency": latency,
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "source_counts": {
            "train": dict(sorted(Counter(record["source_dataset"] for record in train_records).items())),
            "validation": dict(
                sorted(Counter(record["source_dataset"] for record in validation_records).items())
            ),
        },
        "data_identity": data_identity,
        "training_total_seconds": float(summary["training_total_seconds"]),
        "average_epoch_seconds": float(summary["average_epoch_seconds"]),
        "validation_loss": float(validation_loss),
        "joint_metrics": joint.metrics,
        "best_checkpoint": str(checkpoint.relative_to(run)),
        "last_checkpoint": str(Path("checkpoints") / str(summary["last_checkpoint"])),
        "source_identity": resolved["source_identity"],
        "fresh_reload": {
            "passed": True,
            "metric_absolute_differences": comparisons,
            "single_sample_inference": single,
        },
        "representative_errors": errors,
        "largest_shortfall": largest_shortfall,
        "target_macro_f1": target,
        "class_recall_gate": recall_gate,
        "class_recalls": class_recalls,
        "target_attained": attained,
        "data_access": _data_access_audit(),
        "evidence_boundary": "public trajectory development validation only; not camera, real-older-adult, product, or clinical evidence",
    }
    fresh_dir.mkdir(parents=False, exist_ok=False)
    _write_new_bytes(fresh_dir / "metrics.json", _canonical_json_bytes(report_metrics))
    _write_new_bytes(fresh_dir / "predictions.jsonl", _canonical_jsonl_bytes(seeded_predictions))
    confusion = {
        "schema_version": "wandering-performance-confusion-matrices-v1",
        "wp_four_class": joint.metrics["wp_four_class"]["confusion_matrix"],
        "wp_binary": joint.metrics["binary"]["wp"]["confusion_matrix"],
        "smartcare_binary": joint.metrics["binary"]["smartcare"]["confusion_matrix"],
        "pooled_binary": joint.metrics["binary"]["pooled"]["confusion_matrix"],
    }
    _write_new_bytes(fresh_dir / "confusion_matrix.json", _canonical_json_bytes(confusion))
    _write_new_bytes(fresh_dir / "single_sample_inference.json", _canonical_json_bytes(single))
    _write_new_bytes(fresh_dir / "training_history.json", _canonical_json_bytes({"epochs": history}))
    _write_new_bytes(fresh_dir / "README.md", _fresh_readme(report_metrics).encode("utf-8"))
    return FreshEvaluationResult(output_dir=fresh_dir, metrics=report_metrics, predictions=seeded_predictions)


def aggregate_m0s_stability(
    config_path: str | Path,
    *,
    project_root: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
) -> StabilityAggregationResult:
    """Aggregate the fixed three M0-S seeds and same-seed saved baselines."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_performance_config(config_file, project_root=root)
    runs = Path(runs_root).resolve(strict=True)
    output = _create_new_run_dir(output_dir)
    seed_reports: list[dict[str, Any]] = []
    m0_errors: dict[str, dict[int, set[str]]] = {"four_class": {}, "binary": {}}
    baseline_comparisons: dict[str, dict[str, Any]] = {}
    expected_source_identity: Mapping[str, Any] | None = None
    expected_sample_ids: set[str] | None = None

    for seed in M0S_SEEDS:
        run = (runs / f"seed-{seed}").resolve(strict=True)
        if runs not in run.parents:
            raise PerformanceDataError("seed run path escapes runs root")
        metrics = _load_json(run / "fresh_reload" / "metrics.json", PerformanceDataError)
        predictions = tuple(_load_jsonl(run / "fresh_reload" / "predictions.jsonl"))
        if metrics.get("seed") != seed or any(row.get("seed") != seed for row in predictions):
            raise PerformanceDataError("fresh report or predictions have a seed binding mismatch")
        source_identity = metrics.get("source_identity")
        if expected_source_identity is None:
            expected_source_identity = source_identity
        elif source_identity != expected_source_identity:
            raise PerformanceDataError("all three runs must use the exact same source identity")
        sample_ids = {str(row["sample_id"]) for row in predictions}
        if len(sample_ids) != 278:
            raise PerformanceDataError("each fresh prediction file must contain 278 unique validation samples")
        if expected_sample_ids is None:
            expected_sample_ids = sample_ids
        elif sample_ids != expected_sample_ids:
            raise PerformanceDataError("validation sample identities differ across seeds")

        four_errors = {
            str(row["sample_id"])
            for row in predictions
            if row["source_dataset"] == "wandering_patterns"
            and row["true_pattern_label"] != row["predicted_pattern_label"]
        }
        binary_errors = {
            str(row["sample_id"])
            for row in predictions
            if int(row["true_binary_label"]) != int(row["predicted_binary_label"])
        }
        m0_errors["four_class"][seed] = four_errors
        m0_errors["binary"][seed] = binary_errors
        joint = metrics["joint_metrics"]
        seed_reports.append(
            {
                "seed": seed,
                "wp_four_class_macro_f1": float(joint["wp_four_class"]["macro_f1"]),
                "source_equal_binary_macro_f1": float(joint["binary"]["source_equal_macro_f1"]),
                "class_recalls": dict(metrics["class_recalls"]),
                "best_epoch": int(metrics["best_epoch"]),
                "training_total_seconds": float(metrics["training_total_seconds"]),
                "average_epoch_seconds": float(metrics["average_epoch_seconds"]),
                "model_state_bytes": int(metrics["model_state_bytes"]),
                "fixed_batch_inference_latency": dict(metrics["fixed_batch_inference_latency"]),
                "fresh_reload_passed": metrics["fresh_reload"]["passed"] is True,
                "target_attained": metrics["target_attained"] is True,
                "error_counts": {"four_class": len(four_errors), "binary": len(binary_errors)},
            }
        )

        per_seed_comparison: dict[str, Any] = {}
        for baseline_name in ("rf", "tcn"):
            task_comparisons: dict[str, Any] = {}
            for task in ("four_class", "binary"):
                relative = config["baselines"][baseline_name][task].format(seed=seed)
                path = (root / relative).resolve(strict=True)
                rows = _load_jsonl(path)
                universe = {
                    str(row["sample_id"])
                    for row in rows
                }
                required_universe = {
                    str(row["sample_id"])
                    for row in predictions
                    if task == "binary" or row["source_dataset"] == "wandering_patterns"
                }
                if universe != required_universe or any(
                    row.get("seed") != seed or row.get("task") != task for row in rows
                ):
                    raise PerformanceDataError("baseline prediction identity or seed/task binding drifted")
                baseline_errors = {
                    str(row["sample_id"])
                    for row in rows
                    if int(row["true_label"]) != int(row["predicted_label"])
                }
                task_comparisons[task] = paired_error_comparison(
                    universe,
                    m0_errors[task][seed],
                    baseline_errors,
                    baseline_name=baseline_name,
                    baseline_predictions_sha256=_sha256_file(path),
                    baseline_predictions_path=str(path.relative_to(root)).replace("\\", "/"),
                )
            per_seed_comparison[baseline_name] = task_comparisons
        baseline_comparisons[str(seed)] = per_seed_comparison

    metric_values = {
        "wp_four_class_macro_f1": [row["wp_four_class_macro_f1"] for row in seed_reports],
        "source_equal_binary_macro_f1": [row["source_equal_binary_macro_f1"] for row in seed_reports],
    }
    recall_names = sorted(seed_reports[0]["class_recalls"])
    recalls = {
        name: [float(row["class_recalls"][name]) for row in seed_reports]
        for name in recall_names
    }
    target = float(config["reporting"]["target_macro_f1"])
    recall_gate = float(config["reporting"]["class_recall_gate"])
    all_seed_gate = all(
        row["wp_four_class_macro_f1"] >= target
        and row["source_equal_binary_macro_f1"] >= target
        and all(value >= recall_gate for value in row["class_recalls"].values())
        and row["fresh_reload_passed"]
        for row in seed_reports
    )
    error_stability = {}
    for task, by_seed in m0_errors.items():
        sets = [by_seed[seed] for seed in M0S_SEEDS]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        error_stability[task] = {
            "per_seed": {str(seed): sorted(by_seed[seed]) for seed in M0S_SEEDS},
            "intersection": sorted(intersection),
            "union": sorted(union),
            "intersection_count": len(intersection),
            "union_count": len(union),
        }
    report = {
        "schema_version": "wandering-performance-m0s-stability-v1",
        "candidate": "M0-S",
        "fixed_seeds": list(M0S_SEEDS),
        "primary_seed_policy": M0S_PRIMARY_SEED,
        "primary_seed_frozen": M0S_PRIMARY_SEED if all_seed_gate else None,
        "source_identity": expected_source_identity,
        "seed_reports": seed_reports,
        "aggregate": {
            "metrics": {name: _distribution_summary(values) for name, values in metric_values.items()},
            "class_recalls": {name: _distribution_summary(values) for name, values in recalls.items()},
            "best_epoch": _distribution_summary([float(row["best_epoch"]) for row in seed_reports]),
            "training_total_seconds": _distribution_summary(
                [row["training_total_seconds"] for row in seed_reports]
            ),
            "model_state_bytes": _distribution_summary(
                [float(row["model_state_bytes"]) for row in seed_reports]
            ),
            "fixed_batch_mean_ms": _distribution_summary(
                [row["fixed_batch_inference_latency"]["mean_batch_ms"] for row in seed_reports]
            ),
        },
        "gates": {
            "macro_f1_threshold": target,
            "class_recall_threshold": recall_gate,
            "all_three_seeds_required": True,
            "no_extra_seed_or_ensemble": True,
            "passed": all_seed_gate,
        },
        "m0_error_intersection_union": error_stability,
        "same_seed_saved_baseline_paired_counts": baseline_comparisons,
        "data_access": _data_access_audit(),
        "evidence_boundary": "public trajectory development validation only; frozen test was not exposed or used",
    }
    _write_new_bytes(output / "stability_report.json", _canonical_json_bytes(report))
    _write_new_bytes(output / "README.md", _stability_readme(report).encode("utf-8"))
    return StabilityAggregationResult(output_dir=output, report=report)


def paired_error_comparison(
    universe: set[str],
    m0_errors: set[str],
    baseline_errors: set[str],
    *,
    baseline_name: str,
    baseline_predictions_sha256: str,
    baseline_predictions_path: str,
) -> dict[str, Any]:
    if not m0_errors <= universe or not baseline_errors <= universe:
        raise PerformanceDataError("paired error sets must stay inside the shared universe")
    both_wrong = m0_errors & baseline_errors
    either_wrong = m0_errors | baseline_errors
    return {
        "baseline": baseline_name,
        "universe_count": len(universe),
        "m0_error_count": len(m0_errors),
        "baseline_error_count": len(baseline_errors),
        "both_wrong_count": len(both_wrong),
        "either_wrong_count": len(either_wrong),
        "both_correct_count": len(universe - either_wrong),
        "m0_wrong_baseline_correct_count": len(m0_errors - baseline_errors),
        "m0_correct_baseline_wrong_count": len(baseline_errors - m0_errors),
        "both_wrong_sample_ids": sorted(both_wrong),
        "either_wrong_sample_ids": sorted(either_wrong),
        "baseline_predictions_path": baseline_predictions_path,
        "baseline_predictions_sha256": baseline_predictions_sha256,
        "baseline_retrained": False,
    }


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise PerformanceDataError("aggregate distributions require finite values")
    return {
        "mean": statistics.fmean(numeric),
        "population_std": statistics.pstdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _stability_readme(report: Mapping[str, Any]) -> str:
    rows = []
    for item in report["seed_reports"]:
        rows.append(
            f"| {item['seed']} | {item['wp_four_class_macro_f1']:.6f} | "
            f"{item['source_equal_binary_macro_f1']:.6f} | {item['best_epoch']} | "
            f"{item['training_total_seconds']:.1f} | "
            f"{item['fixed_batch_inference_latency']['mean_batch_ms']:.3f} |"
        )
    return (
        "# M0-S three-seed stability\n\n"
        f"- Gate: {'PASS' if report['gates']['passed'] else 'FAIL'}\n"
        f"- Fixed seeds: {report['fixed_seeds']}\n"
        f"- Frozen primary seed: {report['primary_seed_frozen']}\n"
        "- RF/TCN comparisons reuse saved same-seed validation predictions; no baseline retraining.\n"
        "- Shared preprocessing integrity parsing is distinguished from frozen-test exposure/use.\n\n"
        "| Seed | WP 4-class macro-F1 | Source-equal binary macro-F1 | Best epoch | Train sec | Batch latency ms |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nFrozen WP test, SmartCare official, and sealed camera data were not exposed or used. "
        "This is development-validation evidence, not camera, real-older-adult, product, or clinical evidence.\n"
    )


def _train_performance_model_locked(
    config_file: Path,
    config: Mapping[str, Any],
    root: Path,
    output: Path,
    *,
    seed: int,
    resume: bool,
) -> TrainingRunResult:
    performance_sha256 = _sha256_file(config_file)
    forward_path = root / str(config["model"]["forward_config_path"])
    train_records, validation_records, data_identity = load_development_records(config, project_root=root)
    train_dataset = WanderingPerformanceDataset(train_records)
    validation_dataset = WanderingPerformanceDataset(validation_records)
    weights = compute_training_weights(train_records)
    _configure_runtime(config["runtime"])
    device = _resolve_device(config["runtime"]["device"])
    optimizer_config = config["optimizer"]
    checkpoints_root = output / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.jsonl"
    log_path = output / "training_log.jsonl"
    if resume:
        recovery = recover_training_run(
            config_file,
            project_root=root,
            run_dir=output,
            seed=seed,
        )
        latest_checkpoint = recovery.latest_checkpoint
        loaded = load_training_checkpoint(
            latest_checkpoint,
            forward_config_path=forward_path,
            expected_performance_config_sha256=performance_sha256,
            expected_seed=seed,
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )
        model = loaded.model
        optimizer = loaded.optimizer
        start_epoch = loaded.epoch + 1
        history = list(recovery.history)
        completed_summary_path = output / "training_summary.json"
        if completed_summary_path.exists():
            completed_summary = _load_json(completed_summary_path, CheckpointError)
            if completed_summary.get("stopped_reason") not in {
                "validation_early_stopping",
                "max_epochs",
                "already_complete",
            }:
                raise CheckpointError("completed training summary has an invalid stopped_reason")
            best_pointer = _load_json(checkpoints_root / "best.json", CheckpointError)
            best_checkpoint = checkpoints_root / str(best_pointer["checkpoint"])
            best_metadata = _load_json(best_checkpoint / "metadata.json", CheckpointError)
            return TrainingRunResult(
                run_dir=output,
                best_checkpoint=best_checkpoint,
                last_checkpoint=latest_checkpoint,
                best_metrics=best_metadata["metrics"],
                history=tuple(history),
                stopped_reason="already_complete",
            )
    else:
        torch.manual_seed(seed)
        model = create_topowander_training_model(load_topowander_config(forward_path), seed=seed)
        optimizer = create_training_optimizer(
            model,
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )
        start_epoch = 1
        history = []
        _write_new_bytes(output / "performance_config.yaml", config_file.read_bytes())
        source_identity = _source_identity(root, config_file, forward_path)
        resolved = {
            "schema_version": "wandering-performance-resolved-config-v2",
            "seed": seed,
            "performance_config_sha256": performance_sha256,
            "forward_config_sha256": _sha256_file(forward_path),
            "config": config,
            "training_weights": {
                "binary_group_weights": dict(weights.binary_group_weights),
                "binary_group_counts": dict(weights.binary_group_counts),
                "subtype_class_weights": list(weights.subtype_class_weights),
                "subtype_class_counts": dict(weights.subtype_class_counts),
            },
            "data_identity": data_identity,
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "optimized_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "source_identity": source_identity,
            "rng_resume": {
                "torch_cpu_rng_saved_and_restored": True,
                "cuda_resume_supported": False,
                "reason": "the fixed M0-S runtime configuration is CPU-only",
            },
            "data_access": _data_access_audit(),
        }
        _write_new_bytes(output / "resolved_config.json", _canonical_json_bytes(resolved))
    model.to(device)
    _optimizer_to_device(optimizer, device)
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=seed,
        runtime=config["runtime"],
    )
    best_pointer_path = checkpoints_root / "best.json"
    best_rank: tuple[float, float, float, float, int] | None = None
    best_checkpoint: Path | None = None
    best_metrics: Mapping[str, Any] | None = None
    if best_pointer_path.exists():
        pointer = _load_json(best_pointer_path, CheckpointError)
        best_checkpoint = checkpoints_root / str(pointer["checkpoint"])
        best_metadata = _load_json(best_checkpoint / "metadata.json", CheckpointError)
        saved = best_metadata["metrics"]
        best_metrics = saved
        joint = saved["joint_metrics"]
        best_rank = selection_rank(
            float(joint["selection_score"]),
            float(joint["wp_four_class"]["macro_f1"]),
            float(joint["binary"]["source_equal_macro_f1"]),
            float(saved["validation_loss"]),
            int(best_metadata["epoch"]),
        )
    no_improvement_epochs = 0 if not history else int(history[-1].get("no_improvement_epochs", 0))
    training_started = time.perf_counter()
    stopped_reason = "max_epochs"
    last_checkpoint: Path | None = None
    for epoch in range(start_epoch, int(config["training"]["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loader = _make_loader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
            runtime=config["runtime"],
        )
        train_loss, train_binary_loss, train_subtype_loss = _train_epoch(
            model,
            optimizer,
            train_loader,
            weights,
            device=device,
            subtype_loss_weight=float(config["loss"]["subtype_loss_weight"]),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        validation_loss, joint = _evaluate_model(
            model,
            validation_loader,
            validation_records,
            weights,
            device=device,
            subtype_loss_weight=float(config["loss"]["subtype_loss_weight"]),
        )
        epoch_seconds = time.perf_counter() - epoch_started
        rank = selection_rank(
            float(joint.metrics["selection_score"]),
            float(joint.metrics["wp_four_class"]["macro_f1"]),
            float(joint.metrics["binary"]["source_equal_macro_f1"]),
            validation_loss,
            epoch,
        )
        is_best = best_rank is None or rank > best_rank
        no_improvement_epochs = 0 if is_best else no_improvement_epochs + 1
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_binary_loss": train_binary_loss,
            "train_subtype_loss": train_subtype_loss,
            "validation_loss": validation_loss,
            "joint_metrics": joint.metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": epoch_seconds,
            "is_best": is_best,
            "no_improvement_epochs": no_improvement_epochs,
        }
        model.to("cpu")
        _optimizer_to_device(optimizer, torch.device("cpu"))
        last_checkpoint = save_training_checkpoint(
            checkpoints_root,
            epoch=epoch,
            seed=seed,
            model=model,
            optimizer=optimizer,
            forward_config_path=forward_path,
            performance_config_sha256=performance_sha256,
            metrics=epoch_metrics,
        )
        _write_atomic_json(
            checkpoints_root / "latest.json",
            {"epoch": epoch, "checkpoint": last_checkpoint.name},
        )
        if is_best:
            best_rank = rank
            best_checkpoint = last_checkpoint
            best_metrics = epoch_metrics
            _write_atomic_json(
                best_pointer_path,
                {"epoch": epoch, "checkpoint": best_checkpoint.name, "selection_rank": list(rank)},
            )
        model.to(device)
        _optimizer_to_device(optimizer, device)
        history.append(epoch_metrics)
        _append_jsonl(history_path, epoch_metrics)
        elapsed = sum(float(row["epoch_seconds"]) for row in history)
        average = elapsed / len(history)
        eta_seconds = average * (int(config["training"]["max_epochs"]) - epoch)
        event = {
            "event": "epoch_complete",
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "wp_four_class_macro_f1": joint.metrics["wp_four_class"]["macro_f1"],
            "source_equal_binary_macro_f1": joint.metrics["binary"]["source_equal_macro_f1"],
            "selection_score": joint.metrics["selection_score"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": epoch_seconds,
            "average_epoch_seconds": average,
            "eta_seconds_to_max_epochs": eta_seconds,
            "is_best": is_best,
            "best_epoch": int(_load_json(best_pointer_path, CheckpointError)["epoch"]),
        }
        _append_jsonl(log_path, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if (
            epoch >= int(config["early_stopping"]["min_epochs"])
            and no_improvement_epochs >= int(config["early_stopping"]["patience"])
        ):
            stopped_reason = "validation_early_stopping"
            break
    if last_checkpoint is None or best_checkpoint is None or best_metrics is None:
        if resume and start_epoch > int(config["training"]["max_epochs"]):
            latest_pointer = _load_json(checkpoints_root / "latest.json", CheckpointError)
            best_pointer = _load_json(checkpoints_root / "best.json", CheckpointError)
            last_checkpoint = checkpoints_root / str(latest_pointer["checkpoint"])
            best_checkpoint = checkpoints_root / str(best_pointer["checkpoint"])
            best_metrics = _load_json(best_checkpoint / "metadata.json", CheckpointError)["metrics"]
            stopped_reason = "already_complete"
        else:
            raise CheckpointError("training did not produce latest and best checkpoints")
    current_seconds = time.perf_counter() - training_started
    prior_seconds = sum(float(row["epoch_seconds"]) for row in history[:-max(0, len(history) - (start_epoch - 1))]) if resume else 0.0
    total_seconds = sum(float(row["epoch_seconds"]) for row in history)
    summary = {
        "schema_version": "wandering-performance-training-summary-v2",
        "candidate": "M0-S",
        "seed": seed,
        "stopped_reason": stopped_reason,
        "epochs_completed": len(history),
        "best_epoch": int(_load_json(best_pointer_path, CheckpointError)["epoch"]),
        "best_checkpoint": best_checkpoint.name,
        "last_checkpoint": last_checkpoint.name,
        "training_total_seconds": total_seconds,
        "average_epoch_seconds": total_seconds / len(history),
        "current_process_seconds": current_seconds,
        "prior_process_seconds": prior_seconds,
        "best_metrics": best_metrics,
        "data_identity": data_identity,
        "resume_supported": True,
        "rng_resume": {
            "torch_cpu_rng_saved_and_restored": True,
            "cuda_resume_supported": False,
            "reason": "the fixed M0-S runtime configuration is CPU-only",
        },
        "data_access": _data_access_audit(),
    }
    _write_atomic_json(output / "training_summary.json", summary)
    return TrainingRunResult(
        run_dir=output,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        best_metrics=best_metrics,
        history=tuple(history),
        stopped_reason=stopped_reason,
    )


def _train_epoch(
    model: TopoWanderMPT,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[dict[str, Any]],
    weights: TrainingWeights,
    *,
    device: torch.device,
    subtype_loss_weight: float,
    gradient_clip_norm: float,
) -> tuple[float, float, float]:
    model.train()
    total_sum = binary_sum = subtype_sum = 0.0
    sample_count = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["model_features"], batch["shape_normalized_points"], batch["point_mask"])
        total, binary, subtype = _joint_loss_parts(
            outputs["binary_logit"],
            outputs["subtype_logits"],
            batch["binary_label"],
            batch["subtype_label"],
            binary_sample_weights=_batch_binary_weights(batch, weights, device),
            subtype_class_weights=torch.tensor(
                weights.subtype_class_weights, dtype=torch.float32, device=device
            ),
            subtype_loss_weight=subtype_loss_weight,
        )
        if not torch.isfinite(total):
            raise PerformanceDataError("training loss became non-finite")
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise PerformanceDataError(f"training gradient became non-finite: {name}")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], gradient_clip_norm
        )
        optimizer.step()
        count = int(batch["binary_label"].shape[0])
        total_sum += float(total.detach()) * count
        binary_sum += float(binary.detach()) * count
        subtype_sum += float(subtype.detach()) * count
        sample_count += count
    if sample_count == 0:
        raise PerformanceDataError("training loader produced no samples")
    return total_sum / sample_count, binary_sum / sample_count, subtype_sum / sample_count


def _evaluate_model(
    model: TopoWanderMPT,
    loader: DataLoader[dict[str, Any]],
    ordered_records: Sequence[Mapping[str, Any]],
    weights: TrainingWeights,
    *,
    device: torch.device,
    subtype_loss_weight: float,
) -> tuple[float, JointEvaluation]:
    model.eval()
    total_sum = 0.0
    count_sum = 0
    binary_logits = []
    subtype_logits = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            outputs = model(
                batch["model_features"], batch["shape_normalized_points"], batch["point_mask"]
            )
            loss = joint_supervised_loss(
                outputs["binary_logit"],
                outputs["subtype_logits"],
                batch["binary_label"],
                batch["subtype_label"],
                binary_sample_weights=_batch_binary_weights(batch, weights, device),
                subtype_class_weights=torch.tensor(
                    weights.subtype_class_weights, dtype=torch.float32, device=device
                ),
                subtype_loss_weight=subtype_loss_weight,
            )
            batch_count = int(batch["binary_label"].shape[0])
            total_sum += float(loss) * batch_count
            count_sum += batch_count
            binary_logits.append(outputs["binary_logit"].detach().cpu())
            subtype_logits.append(outputs["subtype_logits"].detach().cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    expected_ids = [str(record["sample_id"]) for record in ordered_records]
    if sample_ids != expected_ids:
        raise PerformanceDataError("validation loader order drifted from ordered records")
    if count_sum != len(ordered_records):
        raise PerformanceDataError("validation sample count drifted")
    joint = evaluate_joint_predictions(
        ordered_records,
        torch.cat(binary_logits, dim=0),
        torch.cat(subtype_logits, dim=0),
    )
    return total_sum / count_sum, joint


def _joint_loss_parts(
    binary_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    binary_labels: torch.Tensor,
    subtype_labels: torch.Tensor,
    *,
    binary_sample_weights: torch.Tensor,
    subtype_class_weights: torch.Tensor,
    subtype_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if binary_logits.ndim == 2 and binary_logits.shape[1] == 1:
        binary_logits = binary_logits[:, 0]
    batch_size = binary_logits.shape[0] if binary_logits.ndim == 1 else -1
    if batch_size < 1 or subtype_logits.shape != (batch_size, 3):
        raise PerformanceDataError("joint loss logit shapes must be [B,1] and [B,3]")
    if binary_labels.shape != (batch_size,) or subtype_labels.shape != (batch_size,):
        raise PerformanceDataError("joint loss label shapes must be [B]")
    if binary_sample_weights.shape != (batch_size,) or subtype_class_weights.shape != (3,):
        raise PerformanceDataError("joint loss weight shapes are invalid")
    if binary_labels.dtype != torch.long or subtype_labels.dtype != torch.long:
        raise PerformanceDataError("joint loss labels must be int64")
    if not torch.all((binary_labels == 0) | (binary_labels == 1)):
        raise PerformanceDataError("binary labels must be 0 or 1")
    if not torch.all((subtype_labels >= -1) & (subtype_labels <= 2)):
        raise PerformanceDataError("subtype labels must be -1 or 0..2")
    subtype_mask = subtype_labels >= 0
    if torch.any(subtype_mask & (binary_labels != 1)):
        raise PerformanceDataError("subtype label requires wandering binary label")
    tensors = (binary_logits, subtype_logits, binary_sample_weights, subtype_class_weights)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise PerformanceDataError("joint loss inputs and weights must be finite")
    if torch.any(binary_sample_weights <= 0.0) or torch.any(subtype_class_weights <= 0.0):
        raise PerformanceDataError("joint loss weights must be positive")
    _positive_float(subtype_loss_weight, "subtype_loss_weight", error_type=PerformanceDataError)
    binary_losses = F.binary_cross_entropy_with_logits(
        binary_logits, binary_labels.to(dtype=torch.float32), reduction="none"
    )
    binary_loss = (binary_losses * binary_sample_weights).sum() / binary_sample_weights.sum()
    if torch.any(subtype_mask):
        subtype_loss = F.cross_entropy(
            subtype_logits[subtype_mask],
            subtype_labels[subtype_mask],
            weight=subtype_class_weights,
        )
    else:
        subtype_loss = subtype_logits.sum() * 0.0
    total = binary_loss + float(subtype_loss_weight) * subtype_loss
    if not torch.isfinite(total):
        raise PerformanceDataError("joint supervised loss must be finite")
    return total, binary_loss, subtype_loss


def _validate_performance_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise PerformanceDataError("performance record must be a mapping")
    required = {
        "sample_id",
        "source_dataset",
        "split",
        "preprocess_status",
        "binary_label",
        "binary_supervision_eligible",
        "pattern_label",
        "pattern_supervision_eligible",
        "model_features",
        "shape_normalized_points",
        "point_mask",
    }
    if not required.issubset(record):
        raise PerformanceDataError("performance record fields are incomplete")
    sample_id = record["sample_id"]
    source = record["source_dataset"]
    split = record["split"]
    if not isinstance(sample_id, str) or not sample_id:
        raise PerformanceDataError("sample_id must be non-empty")
    if source not in ALLOWED_SOURCES:
        raise PerformanceDataError("source_dataset is invalid")
    if split not in {"train", "validation"}:
        raise PerformanceDataError("performance records may expose train/validation only")
    if record["preprocess_status"] != "ready" or record["binary_supervision_eligible"] is not True:
        raise PerformanceDataError("performance record must have ready binary supervision")
    binary_label = record["binary_label"]
    if not isinstance(binary_label, int) or isinstance(binary_label, bool) or binary_label not in {0, 1}:
        raise PerformanceDataError("binary label must be integer 0 or 1")
    pattern_name = record["pattern_label"]
    if source == "wandering_patterns":
        if record["pattern_supervision_eligible"] is not True or pattern_name not in PATTERN_LABELS:
            raise PerformanceDataError("WP pattern label is invalid")
        pattern_index = PATTERN_LABELS.index(pattern_name)
        expected_binary = 0 if pattern_name == "direct" else 1
        if binary_label != expected_binary:
            raise PerformanceDataError("WP binary and pattern label mapping drifted")
        subtype_label = -1 if pattern_name == "direct" else SUBTYPE_LABELS.index(pattern_name)
    else:
        if record["pattern_supervision_eligible"] is not False or pattern_name != "unknown":
            raise PerformanceDataError("SmartCare must not receive a subtype or pattern label")
        pattern_index = -1
        subtype_label = -1
    model_features = _finite_array(record["model_features"], (80, 14), "model_features")
    shape_points = _finite_array(
        record["shape_normalized_points"], (80, 2), "shape_normalized_points"
    )
    point_mask = _finite_array(record["point_mask"], (80,), "point_mask")
    if not np.all((point_mask == 0.0) | (point_mask == 1.0)) or point_mask.sum() < 8:
        raise PerformanceDataError("point_mask must be binary with at least eight valid points")
    if not np.array_equal(model_features[:, 12], point_mask):
        raise PerformanceDataError("model feature mask channel does not match point_mask")
    if np.any(model_features[:, 13] < 0.0) or np.any(model_features[:, 13] > 1.0):
        raise PerformanceDataError("model feature quality channel must remain in [0,1]")
    return {
        "sample_id": sample_id,
        "source_dataset": source,
        "split": split,
        "binary_label": binary_label,
        "subtype_label": subtype_label,
        "pattern_label_index": pattern_index,
        "pattern_label_name": pattern_name,
        "model_features_array": model_features,
        "shape_normalized_points_array": shape_points,
        "point_mask_array": point_mask,
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Sequence[str]) -> dict[str, Any]:
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise PerformanceDataError("classification metric arrays must be non-empty aligned vectors")
    label_count = len(labels)
    if np.any(y_true < 0) or np.any(y_true >= label_count) or np.any(y_pred < 0) or np.any(y_pred >= label_count):
        raise PerformanceDataError("classification metric label index is out of range")
    matrix = np.zeros((label_count, label_count), dtype=np.int64)
    for true, predicted in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        matrix[true, predicted] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1_values = []
    recall_values = []
    for index, name in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        f1_values.append(f1)
        recall_values.append(recall)
    return {
        "sample_count": int(y_true.size),
        "labels": list(labels),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _optimizer_npz_bytes(optimizer: torch.optim.Optimizer, model: TopoWanderMPT) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    state_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter)
        if not state:
            raise CheckpointError(f"optimizer state is missing for trained parameter: {name}")
        state_names.append(name)
        for key in sorted(state):
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise CheckpointError(f"optimizer state must be numeric tensor: {name}:{key}")
            array = np.asarray(value.detach().cpu().numpy()).copy()
            if array.ndim > 0:
                array = np.ascontiguousarray(array)
            if array.dtype.kind not in {"f", "i", "u"} or not np.isfinite(array).all():
                raise CheckpointError(f"optimizer state must be finite numeric: {name}:{key}")
            arrays[f"state::{name}::{key}"] = array
    arrays["__torch_rng_state__"] = torch.get_rng_state().cpu().numpy().astype(np.uint8, copy=False)
    arrays["__optimized_name_count__"] = np.asarray([len(state_names)], dtype=np.int64)
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _load_optimizer_npz(path: Path, optimizer: torch.optim.Optimizer, model: TopoWanderMPT) -> None:
    parameter_by_name = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if "__torch_rng_state__" not in keys or "__optimized_name_count__" not in keys:
                raise CheckpointError("optimizer checkpoint lacks RNG or name count")
            if int(np.asarray(archive["__optimized_name_count__"]).reshape(-1)[0]) != len(parameter_by_name):
                raise CheckpointError("optimizer checkpoint parameter count mismatch")
            restored_names: set[str] = set()
            for key in sorted(keys - {"__torch_rng_state__", "__optimized_name_count__"}):
                parts = key.split("::", 2)
                if len(parts) != 3 or parts[0] != "state" or parts[1] not in parameter_by_name:
                    raise CheckpointError(f"optimizer checkpoint key is invalid: {key}")
                name, state_key = parts[1], parts[2]
                array = np.asarray(archive[key])
                if array.dtype.kind not in {"f", "i", "u"} or not np.isfinite(array).all():
                    raise CheckpointError(f"optimizer checkpoint array is invalid: {key}")
                copied = np.asarray(array).copy()
                if copied.ndim > 0:
                    copied = np.ascontiguousarray(copied)
                tensor = torch.from_numpy(copied)
                optimizer.state[parameter_by_name[name]][state_key] = tensor
                restored_names.add(name)
            if restored_names != set(parameter_by_name):
                raise CheckpointError("optimizer checkpoint states are incomplete")
            rng = np.asarray(archive["__torch_rng_state__"])
            if rng.dtype != np.uint8 or rng.ndim != 1:
                raise CheckpointError("optimizer checkpoint torch RNG state is invalid")
            torch.set_rng_state(torch.from_numpy(np.ascontiguousarray(rng.copy())))
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"cannot safely load optimizer checkpoint: {path}") from exc


def _bound_checkpoint_file(checkpoint: Path, descriptor: Any, role: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise CheckpointError(f"{role} descriptor is invalid")
    relative = _relative_path(descriptor["path"], f"{role}.path", error_type=CheckpointError)
    resolved = (checkpoint / relative).resolve(strict=True)
    if checkpoint.resolve() not in resolved.parents or not resolved.is_file():
        raise CheckpointError(f"{role} escapes checkpoint directory")
    if not _valid_sha256(descriptor["sha256"]) or _sha256_file(resolved) != descriptor["sha256"]:
        raise CheckpointError(f"{role} SHA-256 mismatch")
    return resolved


def _bound_development_file(bundle_root: Path, descriptor: Any) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise PerformanceDataError("development records descriptor is invalid")
    relative = _relative_path(
        descriptor["path"], "development records path", error_type=PerformanceDataError
    )
    resolved = (bundle_root / relative).resolve(strict=True)
    if bundle_root.resolve() not in resolved.parents or not resolved.is_file():
        raise PerformanceDataError("development records path escapes bundle")
    if not _valid_sha256(descriptor["sha256"]) or _sha256_file(resolved) != descriptor["sha256"]:
        raise PerformanceDataError("development records SHA-256 mismatch")
    return resolved


def _configure_runtime(runtime: Mapping[str, Any]) -> None:
    torch.set_num_threads(int(runtime["intra_op_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["inter_op_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(runtime["inter_op_threads"]):
            raise PerformanceConfigError("torch inter-op thread count was already fixed to another value")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise PerformanceConfigError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(
    dataset: WanderingPerformanceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    runtime: Mapping[str, Any],
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        collate_fn=collate_performance_batch,
        generator=generator,
        drop_last=False,
    )


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for field in (
        "model_features",
        "shape_normalized_points",
        "point_mask",
        "binary_label",
        "subtype_label",
        "pattern_label",
    ):
        result[field] = batch[field].to(device)
    return result


def _batch_binary_weights(
    batch: Mapping[str, Any], weights: TrainingWeights, device: torch.device
) -> torch.Tensor:
    values = []
    labels = batch["binary_label"].detach().cpu().tolist()
    for source, label in zip(batch["source_dataset"], labels):
        key = f"{source}:{int(label)}"
        if key not in weights.binary_group_weights:
            raise PerformanceDataError(f"binary source/class group was absent from train weights: {key}")
        values.append(weights.binary_group_weights[key])
    return torch.tensor(values, dtype=torch.float32, device=device)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _mask_invariance_difference(
    model: TopoWanderMPT, batch: Mapping[str, Any], atol: float
) -> float:
    model.eval()
    features = batch["model_features"][:1].detach().clone()
    points = batch["shape_normalized_points"][:1].detach().clone()
    mask = batch["point_mask"][:1].detach().clone()
    mask[:, -8:] = 0.0
    features[:, -8:] = 0.0
    features[:, -8:, 12] = 0.0
    points[:, -8:] = 0.0
    changed_features = features.clone()
    changed_points = points.clone()
    changed_features[:, -8:, :] = 123.0
    changed_features[:, -8:, 10:12] = -456.0
    changed_features[:, -8:, 12] = 0.0
    changed_features[:, -8:, 13] = 0.5
    changed_points[:, -8:] = -321.0
    with torch.no_grad():
        first = model(features, points, mask)
        second = model(changed_features, changed_points, mask)
    difference = max(
        float((first[name] - second[name]).abs().max())
        for name in ("binary_logit", "subtype_logits", "projection_embedding")
    )
    if not math.isfinite(difference) or difference > max(atol, 1.0):
        raise PerformanceDataError("mask invariance check produced invalid difference")
    return difference


def _single_sample_inference(
    model: TopoWanderMPT, sample: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        outputs = model(
            sample["model_features"].unsqueeze(0).to(device),
            sample["shape_normalized_points"].unsqueeze(0).to(device),
            sample["point_mask"].unsqueeze(0).to(device),
        )
        probabilities = hierarchical_four_class_probabilities(
            outputs["binary_logit"], outputs["subtype_logits"]
        )[0]
    return {
        "sample_id": str(sample["sample_id"]),
        "finite": bool(torch.isfinite(probabilities).all()),
        "four_class_probabilities": {
            name: float(probabilities[index].detach().cpu())
            for index, name in enumerate(PATTERN_LABELS)
        },
        "predicted_pattern_label": PATTERN_LABELS[int(torch.argmax(probabilities).detach().cpu())],
    }


def _representative_errors(
    predictions: Sequence[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    errors = []
    for row in predictions:
        pattern_error = row["true_pattern_label"] is not None and (
            row["predicted_pattern_label"] != row["true_pattern_label"]
        )
        binary_error = row["predicted_binary_label"] != row["true_binary_label"]
        if pattern_error or binary_error:
            errors.append(dict(row))
        if len(errors) >= limit:
            break
    return errors


def _largest_shortfall(metrics: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for name, values in metrics["wp_four_class"]["per_class"].items():
        candidates.append((float(values["recall"]), f"wp_four_class_recall:{name}"))
    for source in ("wp", "smartcare"):
        for name, values in metrics["binary"][source]["per_class"].items():
            candidates.append((float(values["recall"]), f"{source}_binary_recall:{name}"))
    value, name = min(candidates, key=lambda item: (item[0], item[1]))
    return {"metric": name, "value": value}


def _fresh_readme(report: Mapping[str, Any]) -> str:
    joint = report["joint_metrics"]
    per_class_lines = "\n".join(
        f"| {name} | {values['precision']:.6f} | {values['recall']:.6f} | {values['f1']:.6f} | {values['support']} |"
        for name, values in joint["wp_four_class"]["per_class"].items()
    )
    return (
        "# TopoWander-MPT supervised development validation\n\n"
        f"- Candidate: {report['candidate']}\n"
        f"- Initialization: {report['initialization']}\n"
        f"- Seed: {report['seed']}\n"
        f"- Train / validation: {report['train_sample_count']} / {report['validation_sample_count']}\n"
        f"- Best epoch: {report['best_epoch']} of at most {report['max_epochs']}\n"
        f"- Parameters: {report['parameter_count']} total, {report['optimized_parameter_count']} optimized\n"
        f"- Training seconds: {report['training_total_seconds']:.3f}; average epoch: {report['average_epoch_seconds']:.3f}\n"
        f"- WP four-class macro-F1: {joint['wp_four_class']['macro_f1']:.6f}\n"
        f"- WP binary macro-F1: {joint['binary']['wp']['macro_f1']:.6f}\n"
        f"- SmartCare binary macro-F1: {joint['binary']['smartcare']['macro_f1']:.6f}\n"
        f"- Source-equal binary macro-F1: {joint['binary']['source_equal_macro_f1']:.6f}\n"
        f"- Selection score: {joint['selection_score']:.6f}\n"
        f"- Target >= {report['target_macro_f1']:.2f}: {'yes' if report['target_attained'] else 'no'}\n"
        f"- Fresh reload: {'PASS' if report['fresh_reload']['passed'] else 'FAIL'}\n"
        f"- Best checkpoint: `{report['best_checkpoint']}`\n"
        f"- Last checkpoint: `{report['last_checkpoint']}`\n"
        f"- Largest shortfall: {report['largest_shortfall']['metric']} = {report['largest_shortfall']['value']:.6f}\n\n"
        "| WP class | Precision | Recall | F1 | Support |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"{per_class_lines}\n\n"
        "WP frozen test, SmartCare official, and sealed camera data were not consumed. "
        "These are public trajectory development-validation results, not camera, real-older-adult, product, or clinical evidence.\n"
    )


def _all_reported_class_recalls(metrics: Mapping[str, Any]) -> dict[str, float]:
    recalls: dict[str, float] = {}
    groups = {
        "wp_four_class": metrics["wp_four_class"],
        "binary_wp": metrics["binary"]["wp"],
        "binary_smartcare": metrics["binary"]["smartcare"],
        "binary_pooled": metrics["binary"]["pooled"],
    }
    for group_name, group in groups.items():
        for class_name, values in group["per_class"].items():
            recalls[f"{group_name}:{class_name}"] = float(values["recall"])
    return dict(sorted(recalls.items()))


def _benchmark_fixed_inference_batch(
    model: TopoWanderMPT,
    dataset: WanderingPerformanceDataset,
    *,
    device: torch.device,
    batch_size: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    if len(dataset) < batch_size:
        raise PerformanceDataError("fixed inference batch exceeds validation dataset")
    batch = collate_performance_batch([dataset[index] for index in range(batch_size)])
    device_batch = _batch_to_device(batch, device)
    model.eval()

    def invoke() -> None:
        with torch.no_grad():
            model(
                device_batch["model_features"],
                device_batch["shape_normalized_points"],
                device_batch["point_mask"],
            )

    for _ in range(warmup_iterations):
        invoke()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms: list[float] = []
    for _ in range(measurement_iterations):
        started = time.perf_counter()
        invoke()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    sample_ids_sha256 = hashlib.sha256(
        "\n".join(batch["sample_id"]).encode("utf-8")
    ).hexdigest()
    return {
        "device": str(device),
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "batch_sample_ids_sha256": sample_ids_sha256,
        "mean_batch_ms": statistics.fmean(elapsed_ms),
        "population_std_batch_ms": statistics.pstdev(elapsed_ms),
        "min_batch_ms": min(elapsed_ms),
        "max_batch_ms": max(elapsed_ms),
        "mean_per_sample_ms": statistics.fmean(elapsed_ms) / batch_size,
    }


def _source_identity(root: Path, config_file: Path, forward_path: Path) -> dict[str, Any]:
    candidates = {
        "performance_module": Path(__file__).resolve(strict=True),
        "model_module": Path(__file__).with_name("model.py").resolve(strict=True),
        "preprocessing_bundle_module": Path(__file__).with_name("preprocessing_bundle.py").resolve(strict=True),
        "cli": (root / "scripts/wandering/train_wandering_performance.py").resolve(strict=True),
        "performance_config": config_file.resolve(strict=True),
        "forward_config": forward_path.resolve(strict=True),
    }
    files: dict[str, str] = {}
    for role, path in candidates.items():
        if path != root and root not in path.parents:
            raise PerformanceConfigError(f"source identity path escapes project root: {role}")
        files[str(path.relative_to(root)).replace("\\", "/")] = _sha256_file(path)
    files = dict(sorted(files.items()))
    return {
        "method": "sha256 of exact M0-S source/config files",
        "files": files,
        "bundle_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
    }


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PerformanceDataError(f"{name} must be numeric") from exc
    if array.shape != shape:
        raise PerformanceDataError(f"{name} shape must be {shape}")
    if not np.isfinite(array).all():
        raise PerformanceDataError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PerformanceDataError("joint evaluator logits must be numeric") from exc


def _relative_path(value: Any, field: str, *, error_type: type[ValueError] = PerformanceConfigError) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise error_type(f"{field} must be a non-empty POSIX-style relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise error_type(f"{field} must stay inside project_root")
    return path


def _positive_int(value: Any, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PerformanceConfigError(f"{field} must be a positive integer")


def _positive_float(
    value: Any, field: str, *, error_type: type[ValueError] = PerformanceConfigError
) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        raise error_type(f"{field} must be a positive finite float")


def _nonnegative_float(
    value: Any, field: str, *, error_type: type[ValueError] = PerformanceConfigError
) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
        raise error_type(f"{field} must be a non-negative finite float")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS


def _sha256_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise CheckpointError(f"cannot hash file: {path}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _load_json(path: Path, error_type: type[ValueError]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise error_type(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise CheckpointError(f"JSONL rows must be objects: {path}")
    return rows


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_atomic_bytes(path, _canonical_json_bytes(value))


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _create_new_run_dir(path: str | Path) -> Path:
    output = Path(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    return output


def _acquire_run_lock(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CheckpointError(f"training lock already exists; another run may be active: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "FreshEvaluationResult",
    "JointEvaluation",
    "LoadedCheckpoint",
    "M0S_PRIMARY_SEED",
    "M0S_SEEDS",
    "OverfitSmokeResult",
    "PATTERN_LABELS",
    "PERFORMANCE_SCHEMA_VERSION",
    "PerformanceConfigError",
    "PerformanceDataError",
    "RecoveryResult",
    "StabilityAggregationResult",
    "TrainingRunResult",
    "TrainingWeights",
    "WanderingPerformanceDataset",
    "collate_performance_batch",
    "compute_training_weights",
    "create_training_optimizer",
    "aggregate_m0s_stability",
    "evaluate_fresh_checkpoint",
    "evaluate_joint_predictions",
    "joint_supervised_loss",
    "load_development_records",
    "load_performance_config",
    "load_training_checkpoint",
    "materialize_development_bundle",
    "paired_error_comparison",
    "recover_training_run",
    "run_overfit_smoke",
    "save_training_checkpoint",
    "selection_rank",
    "train_performance_model",
]
