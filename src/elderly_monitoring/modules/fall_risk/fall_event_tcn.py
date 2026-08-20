from __future__ import annotations

import hashlib
import json
import os
import platform
import random
try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource.
    resource = None
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.fall_risk.fall_event_training import (
    FALL_EVENT_CHANNELS,
    FALL_EVENT_JOINTS,
)


TASK = "fall_event_candidate_clip_tcn_v1"
TARGET_TASK = "fall_event_v1"
FALL_SUBTYPE_LABELS = ("forward", "lateral", "backward", "seated")
_ACTION_TO_SUBTYPE = {"D01": 0, "D02": 1, "D03": 2, "D05": 3}
_MIRROR_JOINT_INDICES = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13)


@dataclass(frozen=True)
class FallEventTCNConfig:
    epochs: int = 40
    batch_size: int = 64
    hidden_channels: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    subtype_loss_weight: float = 0.35
    threshold: float = 0.5
    seed: int = 42
    device: str = "auto"
    augment_mirror: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number of at least 3")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.subtype_loss_weight < 0:
            raise ValueError("subtype_loss_weight must be non-negative")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be within (0, 1)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")


class _TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_count(channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class FallEventCandidateTCN(nn.Module):
    def __init__(
        self,
        *,
        joint_count: int = len(FALL_EVENT_JOINTS),
        input_channels: int = len(FALL_EVENT_CHANNELS),
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if joint_count < 1 or input_channels < 1:
            raise ValueError("joint_count and input_channels must be positive")
        self.joint_count = joint_count
        self.input_channels = input_channels
        self.input_projection = nn.Sequential(
            nn.Conv1d(joint_count * input_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(_group_norm_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            *[
                _TemporalResidualBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.presence_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_channels, 2)
        )
        self.subtype_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_channels, len(FALL_SUBTYPE_LABELS))
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError("fall-event TCN input must have shape [B, T, V, C]")
        if inputs.shape[2:] != (self.joint_count, self.input_channels):
            raise ValueError(
                f"expected V,C={self.joint_count},{self.input_channels}; "
                f"got {inputs.shape[2]},{inputs.shape[3]}"
            )
        frame_mask = (inputs[..., -1].amax(dim=2) > 0).to(inputs.dtype)
        batch_size, frame_count, _, _ = inputs.shape
        encoded = inputs.reshape(batch_size, frame_count, -1).transpose(1, 2)
        encoded = self.temporal_blocks(self.input_projection(encoded))
        weights = frame_mask.unsqueeze(1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return {
            "presence_logits": self.presence_head(pooled),
            "subtype_logits": self.subtype_head(pooled),
        }


class _FallEventWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    def __init__(
        self,
        features: np.ndarray,
        presence_labels: np.ndarray,
        subtype_labels: np.ndarray,
        sample_weights: np.ndarray,
        indices: np.ndarray,
        *,
        augment_mirror: bool,
    ) -> None:
        self.features = features
        self.presence_labels = presence_labels
        self.subtype_labels = subtype_labels
        self.sample_weights = sample_weights
        self.indices = indices.astype(np.int64, copy=False)
        self.augment_mirror = augment_mirror

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        source_index = int(self.indices[item])
        features = torch.from_numpy(self.features[source_index]).clone()
        if self.augment_mirror and bool(torch.rand(()) < 0.5):
            features = features[:, _MIRROR_JOINT_INDICES, :].clone()
            features[..., 0] *= -1.0
            features[..., 2] *= -1.0
        return (
            features,
            torch.tensor(int(self.presence_labels[source_index]), dtype=torch.long),
            torch.tensor(int(self.subtype_labels[source_index]), dtype=torch.long),
            torch.tensor(float(self.sample_weights[source_index]), dtype=torch.float32),
            source_index,
        )


def build_fall_subtype_labels(action_ids: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_ids).astype(str)
    labels = np.full(len(actions), -1, dtype=np.int64)
    for action_id, subtype in _ACTION_TO_SUBTYPE.items():
        labels[actions == action_id] = subtype
    return labels


def compute_fall_multitask_loss(
    presence_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    presence_labels: torch.Tensor,
    subtype_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    subtype_loss_weight: float,
) -> dict[str, Any]:
    if presence_logits.ndim != 2 or presence_logits.shape[1] != 2:
        raise ValueError("fall-event presence logits must have shape [B,2]")
    if subtype_logits.ndim != 2 or subtype_logits.shape[1] != len(FALL_SUBTYPE_LABELS):
        raise ValueError("fall-event subtype logits have an invalid shape")
    if any(
        len(values) != len(presence_logits)
        for values in (subtype_logits, presence_labels, subtype_labels, sample_weights)
    ):
        raise ValueError("fall-event loss inputs have inconsistent batch lengths")
    if torch.any(sample_weights <= 0) or not torch.isfinite(sample_weights).all():
        raise ValueError("fall-event sample weights must be finite and positive")
    presence_losses = F.cross_entropy(
        presence_logits, presence_labels, reduction="none"
    )
    presence_loss = (presence_losses * sample_weights).sum() / sample_weights.sum()
    subtype_mask = (presence_labels == 1) & (subtype_labels >= 0)
    subtype_count = int(subtype_mask.sum().detach().cpu())
    if subtype_count:
        subtype_weights = sample_weights[subtype_mask]
        subtype_losses = F.cross_entropy(
            subtype_logits[subtype_mask], subtype_labels[subtype_mask], reduction="none"
        )
        subtype_loss = (subtype_losses * subtype_weights).sum() / subtype_weights.sum()
    else:
        subtype_loss = presence_loss.new_zeros(())
    return {
        "loss": presence_loss + (subtype_loss_weight * subtype_loss),
        "presence_loss": presence_loss,
        "subtype_loss": subtype_loss,
        "subtype_count": subtype_count,
    }


def train_fall_event_candidate_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: FallEventTCNConfig | None = None,
    allow_provisional: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if not allow_provisional:
        raise ValueError(
            "fall-event candidate TCN is provisional; pass allow_provisional explicitly"
        )
    training = config or FallEventTCNConfig()
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    metadata, arrays = load_fall_event_candidate_dataset(
        source_path, source_metadata_path
    )
    destination = Path(output_dir)
    paths = {
        "best": destination / "best_model.pt",
        "last": destination / "last_checkpoint.pt",
        "config": destination / "config.json",
        "environment": destination / "environment.json",
        "history": destination / "history.jsonl",
        "metrics": destination / "metrics.json",
        "predictions": destination / "validation_predictions.jsonl",
        "failures": destination / "failure_cases.jsonl",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"fall-event TCN output already exists: {existing[0]}")
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    _seed_everything(training.seed)
    device = _select_device(training.device)
    features = arrays["features"].astype(np.float32, copy=False)
    presence_labels = arrays["labels"].astype(np.int64, copy=False)
    subtype_labels = build_fall_subtype_labels(arrays["action_ids"])
    partitions = arrays["partitions"].astype(str)
    indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation")
    }
    for partition, partition_indices in indices.items():
        if len(partition_indices) == 0:
            raise ValueError(f"fall-event {partition} partition is empty")
        if set(presence_labels[partition_indices].tolist()) != {0, 1}:
            raise ValueError(f"fall-event {partition} partition lacks presence classes")
        positive_subtypes = subtype_labels[
            partition_indices[presence_labels[partition_indices] == 1]
        ]
        if set(positive_subtypes.tolist()) != set(range(len(FALL_SUBTYPE_LABELS))):
            raise ValueError(f"fall-event {partition} partition lacks fall subtypes")

    effective_weights = arrays["sample_weights"].astype(np.float32, copy=True)
    effective_weights[indices["train"]] = _balanced_training_weights(
        presence_labels,
        effective_weights,
        arrays["source_group_ids"].astype(str),
        indices["train"],
    )
    datasets = {
        partition: _FallEventWindowDataset(
            features,
            presence_labels,
            subtype_labels,
            effective_weights,
            partition_indices,
            augment_mirror=training.augment_mirror and partition == "train",
        )
        for partition, partition_indices in indices.items()
    }
    generator = torch.Generator().manual_seed(training.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=training.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    model_config = {
        "joint_count": len(FALL_EVENT_JOINTS),
        "input_channels": len(FALL_EVENT_CHANNELS),
        "hidden_channels": training.hidden_channels,
        "kernel_size": training.kernel_size,
        "dilations": list(training.dilations),
        "dropout": training.dropout,
    }
    model = FallEventCandidateTCN(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    resumed = resume_checkpoint is not None
    resumed_from_sha256: str | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"fall-event resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(payload, metadata, model_config, training)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        best_state = payload["best_state_dict"]
        best_validation_score = float(payload["best_validation_score"])
        best_epoch = int(payload["best_epoch"])
        stale_epochs = int(payload["stale_epochs"])
        history = [dict(row) for row in payload["history"]]
        start_epoch = int(payload["epoch"]) + 1
        resumed_from_sha256 = _sha256_file(resume_path)

    for epoch in range(start_epoch, training.epochs + 1):
        train_losses = _run_epoch(
            model,
            loaders["train"],
            device,
            subtype_loss_weight=training.subtype_loss_weight,
            optimizer=optimizer,
        )
        validation = _evaluate(
            model,
            loaders["validation"],
            arrays,
            device,
            subtype_loss_weight=training.subtype_loss_weight,
            threshold=training.threshold,
        )
        selection_score = _selection_score(validation)
        history.append(
            {
                "epoch": epoch,
                "train": train_losses,
                "validation_loss": validation["loss"],
                "validation_presence": validation["presence"],
                "validation_subtype": validation["subtype"],
                "selection_score": selection_score,
                "test_access": False,
            }
        )
        if selection_score > best_validation_score + 1e-8:
            best_validation_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        _write_torch_atomic(
            paths["last"],
            _training_checkpoint(
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training=training,
                metadata=metadata,
                epoch=epoch,
                history=history,
                best_state=best_state,
                best_epoch=best_epoch,
                best_validation_score=best_validation_score,
                stale_epochs=stale_epochs,
            ),
        )
        if stale_epochs >= training.patience:
            break

    if best_state is None or not history:
        raise RuntimeError("fall-event TCN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation = _evaluate(
        model,
        loaders["validation"],
        arrays,
        device,
        subtype_loss_weight=training.subtype_loss_weight,
        threshold=training.threshold,
    )
    code = _code_fingerprint()
    best_checkpoint = {
        "schema_version": "fall-event-candidate-tcn-model-v1",
        "task": TASK,
        "target_task": TARGET_TASK,
        "status": "provisional",
        "model_version": "fall-event-candidate-tcn-v0.1-provisional",
        "state_dict": best_state,
        "model_config": model_config,
        "joint_order": list(FALL_EVENT_JOINTS),
        "channel_order": list(FALL_EVENT_CHANNELS),
        "label_mapping": {
            "presence": {"explicit_non_fall_action": 0, "fall_action": 1},
            "subtype": {
                name: index for index, name in enumerate(FALL_SUBTYPE_LABELS)
            },
        },
        "threshold": training.threshold,
        "input_contract": _input_contract(metadata),
        "best_epoch": best_epoch,
        "selection_metric": "validation_presence_balanced_accuracy",
        "best_validation_score": best_validation_score,
        "dataset_sha256": metadata["dataset_sha256"],
        "training_config": asdict(training),
        "code_sha256": code["sha256"],
        "test_evaluated": False,
    }
    _write_torch_atomic(paths["best"], best_checkpoint)
    _write_jsonl_atomic(paths["history"], history)
    _write_jsonl_atomic(paths["predictions"], validation["predictions"])
    _write_jsonl_atomic(paths["failures"], validation["failures"])

    config_payload = {
        "schema_version": "fall-event-candidate-tcn-run-config-v1",
        "task": TASK,
        "target_task": TARGET_TASK,
        "status": "provisional",
        "run_id": destination.name,
        "training_config": asdict(training),
        "dataset_path": source_path.as_posix(),
        "metadata_path": source_metadata_path.as_posix(),
        "data_provenance": _data_provenance(metadata, source_metadata_path),
        "input_contract": _input_contract(metadata),
        "code": code,
        "test_access": False,
        "resume_checkpoint": (
            Path(resume_checkpoint).as_posix() if resume_checkpoint else None
        ),
        "resume_checkpoint_sha256": resumed_from_sha256,
    }
    _write_json_atomic(paths["config"], config_payload)
    elapsed = time.perf_counter() - started
    environment = {
        "schema_version": "fall-event-candidate-tcn-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "peak_rss_mb": (
            round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0 * 1024.0),
                3,
            )
            if resource is not None
            else None
        ),
    }
    _write_json_atomic(paths["environment"], environment)
    metrics = {
        "schema_version": "fall-event-candidate-tcn-training-report-v1",
        "task": TASK,
        "target_task": TARGET_TASK,
        "status": "provisional",
        "run_id": destination.name,
        "model_version": best_checkpoint["model_version"],
        "seed": training.seed,
        "resumed": resumed,
        "epochs_completed": int(history[-1]["epoch"]),
        "best_epoch": best_epoch,
        "best_validation_score": best_validation_score,
        "elapsed_sec": round(elapsed, 6),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "train_segment_count": int(len(indices["train"])),
        "validation_segment_count": int(len(indices["validation"])),
        "validation": {
            "loss": validation["loss"],
            "presence": validation["presence"],
            "subtype": validation["subtype"],
            "dataset_presence": validation["dataset_presence"],
            "action_presence": validation["action_presence"],
        },
        "failure_case_count": len(validation["failures"]),
        "test_evaluated": False,
        "test": None,
        "dataset_sha256": metadata["dataset_sha256"],
        "checkpoint_sha256": _sha256_file(paths["best"]),
        "run_config_sha256": _sha256_file(paths["config"]),
        "code_sha256": code["sha256"],
        "input_contract": _input_contract(metadata),
        "limitations": [
            "this model classifies pre-segmented labelled action clips and does not localize falls",
            "the non-fall class is limited to explicitly labelled proxy actions",
            "no test pose, test feature or test metric was read",
            "the score cannot be presented as confirmed fall probability",
        ],
    }
    _write_json_atomic(paths["metrics"], metrics)
    return {
        "output_dir": destination.as_posix(),
        "run_id": destination.name,
        "checkpoint_path": paths["best"].as_posix(),
        "last_checkpoint_path": paths["last"].as_posix(),
        "metrics_path": paths["metrics"].as_posix(),
        "history_path": paths["history"].as_posix(),
        "resumed": resumed,
        "epochs_completed": metrics["epochs_completed"],
        "best_epoch": best_epoch,
        "elapsed_sec": round(elapsed, 6),
        "validation": metrics["validation"],
        "test_evaluated": False,
    }


def evaluate_fall_event_candidate_tcn(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    metadata, arrays = load_fall_event_candidate_dataset(
        source_path, source_metadata_path
    )
    checkpoint_source = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    _validate_model_checkpoint(checkpoint)
    if checkpoint.get("dataset_sha256") != metadata["dataset_sha256"]:
        raise ValueError("fall-event candidate TCN dataset SHA-256 mismatch")
    model = FallEventCandidateTCN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    selected_device = _select_device(device)
    model.to(selected_device)
    indices = np.flatnonzero(arrays["partitions"].astype(str) == "validation")
    dataset = _FallEventWindowDataset(
        arrays["features"].astype(np.float32, copy=False),
        arrays["labels"].astype(np.int64, copy=False),
        build_fall_subtype_labels(arrays["action_ids"]),
        arrays["sample_weights"].astype(np.float32, copy=False),
        indices,
        augment_mirror=False,
    )
    evaluation = _evaluate(
        model,
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        arrays,
        selected_device,
        subtype_loss_weight=float(
            checkpoint["training_config"]["subtype_loss_weight"]
        ),
        threshold=float(checkpoint["threshold"]),
    )
    report = {
        "schema_version": "fall-event-candidate-tcn-evaluation-v1",
        "task": TASK,
        "status": "provisional",
        "partition": "validation",
        "test_evaluated": False,
        "dataset_sha256": metadata["dataset_sha256"],
        "checkpoint_sha256": _sha256_file(checkpoint_source),
        "presence": evaluation["presence"],
        "subtype": evaluation["subtype"],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"fall-event TCN evaluation output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, report)
    return report


class FallEventTCNPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        source = Path(checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"fall-event TCN checkpoint not found: {source}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        _validate_model_checkpoint(checkpoint)
        self.checkpoint_path = source
        self.checkpoint_sha256 = _sha256_file(source)
        self.device = _select_device(device)
        self.threshold = float(checkpoint["threshold"])
        self.model_version = str(checkpoint["model_version"])
        self.model = FallEventCandidateTCN(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_tensor(self, tensor: np.ndarray) -> dict[str, Any]:
        values = np.asarray(tensor, dtype=np.float32)
        expected = (
            len(FALL_EVENT_JOINTS),
            len(FALL_EVENT_CHANNELS),
        )
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError(
                "fall-event predictor expects [T,14,7], got " + str(values.shape)
            )
        if not np.isfinite(values).all():
            raise ValueError("fall-event predictor input contains non-finite values")
        with torch.no_grad():
            outputs = self.model(
                torch.from_numpy(values).unsqueeze(0).to(self.device)
            )
            presence = torch.softmax(outputs["presence_logits"], dim=1)[0]
        score = float(presence[1].detach().cpu())
        return {
            "fall_event_score": score,
            "fall_event_detected": score >= self.threshold,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "status": "provisional_shadow",
        }


def load_fall_event_candidate_dataset(
    dataset_path: str | Path, metadata_path: str | Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_path = Path(dataset_path)
    source_metadata_path = Path(metadata_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"fall-event dataset not found: {source_path}")
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"fall-event metadata not found: {source_metadata_path}")
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "fall-action-presence-proxy-dataset-v1":
        raise ValueError("unsupported fall-event prepared dataset schema")
    if metadata.get("task") != "fall_action_presence_proxy_v1":
        raise ValueError("fall-event candidate TCN dataset task mismatch")
    if metadata.get("status") != "provisional" or metadata.get("test_pose_read") is not False:
        raise ValueError("fall-event candidate TCN requires a test-locked provisional dataset")
    if metadata.get("joint_order") != list(FALL_EVENT_JOINTS):
        raise ValueError("fall-event candidate TCN joint order mismatch")
    if metadata.get("channel_order") != list(FALL_EVENT_CHANNELS):
        raise ValueError("fall-event candidate TCN channel order mismatch")
    if metadata.get("dataset_sha256") != _sha256_file(source_path):
        raise ValueError("fall-event candidate dataset SHA-256 mismatch")
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "features",
        "labels",
        "partitions",
        "sample_ids",
        "segment_ids",
        "video_ids",
        "split_group_ids",
        "source_group_ids",
        "datasets",
        "action_ids",
        "sample_weights",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"fall-event candidate dataset is missing arrays: {missing}")
    sample_count = len(arrays["labels"])
    if any(len(values) != sample_count for values in arrays.values()):
        raise ValueError("fall-event candidate arrays have inconsistent lengths")
    expected_shape = (
        sample_count,
        int(metadata["preparation_config"]["target_fps"] * metadata["preparation_config"]["window_sec"]),
        len(FALL_EVENT_JOINTS),
        len(FALL_EVENT_CHANNELS),
    )
    if arrays["features"].shape != expected_shape:
        raise ValueError("fall-event candidate feature tensor shape mismatch")
    if set(arrays["partitions"].astype(str).tolist()) != {"train", "validation"}:
        raise ValueError("fall-event candidate dataset must remain train/validation only")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("fall-event candidate feature tensor contains non-finite values")
    _validate_group_isolation(arrays)
    return metadata, arrays


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    subtype_loss_weight: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    model.train()
    totals = defaultdict(float)
    batches = 0
    for features, presence, subtype, weights, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features.to(device))
        losses = compute_fall_multitask_loss(
            outputs["presence_logits"],
            outputs["subtype_logits"],
            presence.to(device),
            subtype.to(device),
            weights.to(device),
            subtype_loss_weight=subtype_loss_weight,
        )
        losses["loss"].backward()
        optimizer.step()
        for name in ("loss", "presence_loss", "subtype_loss"):
            totals[name] += float(losses[name].detach().cpu())
        batches += 1
    if not batches:
        raise ValueError("fall-event training loader is empty")
    averages: dict[str, Any] = {
        name: value / batches for name, value in totals.items()
    }
    if subtype_loss_weight == 0:
        averages["subtype_loss"] = None
        averages["subtype_status"] = "not_trained"
    else:
        averages["subtype_status"] = "trained"
    return averages


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    subtype_loss_weight: float,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for features, presence, subtype, weights, source_indices in loader:
            outputs = model(features.to(device))
            losses = compute_fall_multitask_loss(
                outputs["presence_logits"],
                outputs["subtype_logits"],
                presence.to(device),
                subtype.to(device),
                weights.to(device),
                subtype_loss_weight=subtype_loss_weight,
            )
            total_loss += float(losses["loss"].detach().cpu())
            batches += 1
            presence_probabilities = torch.softmax(
                outputs["presence_logits"], dim=1
            )[:, 1].detach().cpu().numpy()
            subtype_probabilities = None
            if subtype_loss_weight > 0:
                subtype_probabilities = torch.softmax(
                    outputs["subtype_logits"], dim=1
                ).detach().cpu().numpy()
            for batch_index, source_index_value in enumerate(source_indices.tolist()):
                source_index = int(source_index_value)
                score = float(presence_probabilities[batch_index])
                row = {
                    "sample_id": str(arrays["sample_ids"][source_index]),
                    "segment_id": str(arrays["segment_ids"][source_index]),
                    "video_id": str(arrays["video_ids"][source_index]),
                    "split_group_id": str(arrays["split_group_ids"][source_index]),
                    "source_group_id": str(arrays["source_group_ids"][source_index]),
                    "dataset": str(arrays["datasets"][source_index]),
                    "action_id": str(arrays["action_ids"][source_index]),
                    "presence_label": int(arrays["labels"][source_index]),
                    "presence_probability": score,
                    "presence_prediction": int(score >= threshold),
                }
                if subtype_probabilities is not None:
                    subtype_scores = subtype_probabilities[batch_index].astype(float)
                    subtype_prediction = int(np.argmax(subtype_scores))
                    row.update(
                        {
                            "subtype_label": int(subtype[batch_index]),
                            "subtype_probabilities": subtype_scores.tolist(),
                            "subtype_prediction": subtype_prediction,
                            "fall_subtype": FALL_SUBTYPE_LABELS[subtype_prediction],
                        }
                    )
                rows.append(row)
    if not batches:
        raise ValueError("fall-event validation loader is empty")
    rows.sort(key=lambda row: str(row["sample_id"]))
    presence_metrics = _binary_metrics(rows, threshold)
    if subtype_loss_weight > 0:
        subtype_metrics = _subtype_metrics(rows)
    else:
        subtype_metrics = {
            "status": "not_trained",
            "reason": "subtype_loss_weight_is_zero",
        }
    failures = [
        row
        for row in rows
        if row["presence_prediction"] != row["presence_label"]
        or (
            subtype_loss_weight > 0
            and row["presence_label"] == 1
            and row["subtype_prediction"] != row["subtype_label"]
        )
    ]
    dataset_presence = {
        name: _binary_metrics(group, threshold)
        for name, group in _group_rows(rows, "dataset").items()
        if {int(row["presence_label"]) for row in group} == {0, 1}
    }
    action_presence = {
        name: _binary_metrics(group, threshold)
        for name, group in _group_rows(rows, "action_id").items()
    }
    return {
        "loss": total_loss / batches,
        "presence": presence_metrics,
        "subtype": subtype_metrics,
        "dataset_presence": dataset_presence,
        "action_presence": action_presence,
        "predictions": rows,
        "failures": failures,
    }


def _binary_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    labels = np.asarray([row["presence_label"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["presence_probability"] for row in rows], dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    both_classes = set(labels.tolist()) == {0, 1}
    return {
        "count": len(rows),
        "threshold": threshold,
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions)) if both_classes else None
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else None,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).astype(int).tolist(),
    }


def _subtype_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if int(row["presence_label"]) == 1]
    labels = np.asarray([row["subtype_label"] for row in positive], dtype=np.int64)
    predictions = np.asarray(
        [row["subtype_prediction"] for row in positive], dtype=np.int64
    )
    expected = list(range(len(FALL_SUBTYPE_LABELS)))
    if set(labels.tolist()) != set(expected):
        raise ValueError("fall-event subtype evaluation requires all four subtypes")
    return {
        "count": len(positive),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=expected, average="macro")
        ),
        "per_class_recall": {
            FALL_SUBTYPE_LABELS[index]: float(
                recall_score(labels == index, predictions == index, zero_division=0)
            )
            for index in expected
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=expected
        ).astype(int).tolist(),
    }


def _balanced_training_weights(
    labels: np.ndarray,
    base_weights: np.ndarray,
    source_groups: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    selected_labels = labels[train_indices]
    selected_sources = source_groups[train_indices]
    weights = base_weights[train_indices].astype(np.float64, copy=True)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("fall-event training weights must be finite and positive")
    for label in (0, 1):
        label_mask = selected_labels == label
        sources = sorted(set(selected_sources[label_mask].tolist()))
        if not sources:
            raise ValueError("fall-event training class has no source groups")
        for source in sources:
            mask = label_mask & (selected_sources == source)
            source_mass = float(weights[mask].sum())
            weights[mask] *= (0.5 / len(sources)) / source_mass
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def _training_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: Mapping[str, Any],
    training: FallEventTCNConfig,
    metadata: Mapping[str, Any],
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    best_state: Mapping[str, torch.Tensor] | None,
    best_epoch: int,
    best_validation_score: float,
    stale_epochs: int,
) -> dict[str, Any]:
    if best_state is None:
        raise RuntimeError("fall-event TCN has no best state for recovery")
    return {
        "schema_version": "fall-event-candidate-tcn-training-checkpoint-v1",
        "task": TASK,
        "status": "provisional",
        "dataset_sha256": metadata["dataset_sha256"],
        "model_config": dict(model_config),
        "resume_signature": _resume_signature(training),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "best_state_dict": dict(best_state),
        "epoch": epoch,
        "history": [dict(row) for row in history],
        "best_epoch": best_epoch,
        "best_validation_score": best_validation_score,
        "stale_epochs": stale_epochs,
        "test_evaluated": False,
    }


def _validate_resume_checkpoint(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    model_config: Mapping[str, Any],
    training: FallEventTCNConfig,
) -> None:
    if payload.get("schema_version") != "fall-event-candidate-tcn-training-checkpoint-v1":
        raise ValueError("unsupported fall-event TCN resume checkpoint")
    if payload.get("task") != TASK:
        raise ValueError("fall-event TCN resume task mismatch")
    if payload.get("dataset_sha256") != metadata["dataset_sha256"]:
        raise ValueError("fall-event TCN resume dataset SHA-256 mismatch")
    if payload.get("model_config") != dict(model_config):
        raise ValueError("fall-event TCN resume model contract mismatch")
    if payload.get("resume_signature") != _resume_signature(training):
        raise ValueError("fall-event TCN resume training contract mismatch")
    if payload.get("test_evaluated") is not False:
        raise ValueError("fall-event TCN resume checkpoint must remain test locked")


def _validate_model_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "fall-event-candidate-tcn-model-v1":
        raise ValueError("unsupported fall-event candidate TCN checkpoint")
    if payload.get("task") != TASK or payload.get("target_task") != TARGET_TASK:
        raise ValueError("fall-event candidate TCN checkpoint task mismatch")
    if payload.get("status") != "provisional" or payload.get("test_evaluated") is not False:
        raise ValueError("fall-event candidate TCN checkpoint governance mismatch")
    if payload.get("joint_order") != list(FALL_EVENT_JOINTS):
        raise ValueError("fall-event candidate TCN checkpoint joint order mismatch")
    if payload.get("channel_order") != list(FALL_EVENT_CHANNELS):
        raise ValueError("fall-event candidate TCN checkpoint channel order mismatch")


def _selection_score(validation: Mapping[str, Any]) -> float:
    return float(validation["presence"]["balanced_accuracy"])


def _resume_signature(config: FallEventTCNConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("epochs")
    payload.pop("patience")
    payload.pop("device")
    return payload


def _validate_group_isolation(arrays: Mapping[str, np.ndarray]) -> None:
    partitions = arrays["partitions"].astype(str)
    for field in ("segment_ids", "video_ids", "split_group_ids", "source_group_ids"):
        partition_by_value: dict[str, set[str]] = defaultdict(set)
        for value, partition in zip(arrays[field], partitions, strict=True):
            partition_by_value[str(value)].add(str(partition))
        leaked = sorted(
            value for value, values in partition_by_value.items() if len(values) > 1
        )
        if leaked:
            raise ValueError(f"fall-event {field} crosses partitions: {leaked[0]}")


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return dict(grouped)


def _group_norm_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _input_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_order": list(FALL_EVENT_JOINTS),
        "channel_order": list(FALL_EVENT_CHANNELS),
        "preparation_config": metadata["preparation_config"],
        "normalization_contract": metadata["normalization_contract"],
    }


def _data_provenance(
    metadata: Mapping[str, Any], metadata_path: Path
) -> dict[str, Any]:
    return {
        "dataset_sha256": metadata["dataset_sha256"],
        "metadata_sha256": _sha256_file(metadata_path),
        "input_sha256": dict(metadata["input_sha256"]),
        "audit_sha256": metadata["audit_sha256"],
        "source_split_id": metadata["source_split_id"],
        "derived_split_sha256": metadata["derived_split_sha256"],
        "derived_assignments_sha256": metadata["derived_assignments_sha256"],
        "samples_sha256": metadata["samples_sha256"],
    }


def _code_fingerprint() -> dict[str, Any]:
    paths = [Path(__file__), Path(__file__).with_name("fall_event_training.py")]
    digest = hashlib.sha256()
    entries: dict[str, str] = {}
    for path in sorted(paths):
        file_digest = _sha256_file(path)
        entries[path.as_posix()] = file_digest
        digest.update(path.name.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {"sha256": digest.hexdigest(), "files": entries, "git_commit": git_commit}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    torch.save(dict(payload), partial)
    os.replace(partial, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
