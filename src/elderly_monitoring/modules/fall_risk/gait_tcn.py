from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.fall_risk.kinecal_gait import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
)
from elderly_monitoring.modules.fall_risk.gait_tensor import build_gait_tensor


_MIRROR_JOINT_INDICES = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13)


@dataclass(frozen=True)
class GaitTCNTrainingConfig:
    epochs: int = 80
    batch_size: int = 16
    hidden_channels: int = 48
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42
    device: str = "auto"
    augment_mirror: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number of at least 3")
        if not self.dilations or any(dilation < 1 for dilation in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")


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
        group_count = _group_norm_count(channels)
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
            nn.GroupNorm(group_count, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class LightweightGaitTCN(nn.Module):
    def __init__(
        self,
        *,
        joint_count: int = len(CANONICAL_GAIT_JOINTS),
        input_channels: int = len(GAIT_TCN_CHANNELS),
        hidden_channels: int = 48,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.20,
        class_count: int = 2,
    ) -> None:
        super().__init__()
        if joint_count < 1 or input_channels < 1:
            raise ValueError("joint_count and input_channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number of at least 3")
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
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("gait TCN input must have shape [B, T, V, C]")
        if inputs.shape[2] != self.joint_count or inputs.shape[3] != self.input_channels:
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
        return self.classifier(pooled)


class _GaitWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        augment_mirror: bool,
    ) -> None:
        self.features = features
        self.labels = labels
        self.indices = indices.astype(np.int64, copy=False)
        self.augment_mirror = augment_mirror

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = int(self.indices[item])
        features = torch.from_numpy(self.features[source_index]).clone()
        if self.augment_mirror and bool(torch.rand(()) < 0.5):
            features = features[:, _MIRROR_JOINT_INDICES, :].clone()
            features[..., 0] *= -1.0
            features[..., 2] *= -1.0
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        return features, label, source_index


def aggregate_participant_predictions(
    *,
    participant_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    if not (len(participant_ids) == len(labels) == len(probabilities)):
        raise ValueError("participant IDs, labels and probabilities must have equal length")
    grouped: dict[str, dict[str, Any]] = {}
    for participant_id, label, probability in zip(
        participant_ids, labels, probabilities, strict=True
    ):
        participant = str(participant_id)
        normalized_label = int(label)
        if participant not in grouped:
            grouped[participant] = {"label": normalized_label, "probabilities": []}
        elif grouped[participant]["label"] != normalized_label:
            raise ValueError(f"participant {participant} has inconsistent labels")
        grouped[participant]["probabilities"].append(float(probability))

    rows: list[dict[str, Any]] = []
    for participant_id in sorted(grouped, key=_participant_sort_key):
        item = grouped[participant_id]
        probability = float(np.mean(item["probabilities"]))
        rows.append(
            {
                "participant_id": participant_id,
                "label": int(item["label"]),
                "probability": probability,
                "predicted_label": int(probability >= 0.5),
                "window_count": len(item["probabilities"]),
            }
        )
    return rows


def load_prepared_gait_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "features",
            "labels",
            "participant_ids",
            "groups",
            "partitions",
            "sample_ids",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"prepared gait dataset is missing arrays: {missing}")
        arrays = {name: archive[name] for name in archive.files}

    features = arrays["features"]
    labels = arrays["labels"]
    if features.ndim != 4 or features.shape[2:] != (
        len(CANONICAL_GAIT_JOINTS),
        len(GAIT_TCN_CHANNELS),
    ):
        raise ValueError(f"unexpected prepared feature shape: {features.shape}")
    if len(features) != len(labels) or any(
        len(arrays[name]) != len(features)
        for name in ("participant_ids", "groups", "partitions", "sample_ids")
    ):
        raise ValueError("prepared gait arrays have inconsistent lengths")
    if not np.isfinite(features).all():
        raise ValueError("prepared gait features contain non-finite values")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("prepared gait labels must contain both binary classes")
    partitions = {str(value) for value in np.unique(arrays["partitions"])}
    if partitions != {"train", "validation", "test"}:
        raise ValueError(f"unexpected prepared partitions: {sorted(partitions)}")

    partition_by_participant: dict[str, set[str]] = defaultdict(set)
    for participant, partition in zip(
        arrays["participant_ids"], arrays["partitions"], strict=True
    ):
        partition_by_participant[str(participant)].add(str(partition))
    leaked = sorted(
        participant
        for participant, participant_partitions in partition_by_participant.items()
        if len(participant_partitions) != 1
    )
    if leaked:
        raise ValueError(f"participants cross data partitions: {leaked}")
    return arrays


def predict_gait_risk_scores(
    checkpoint_path: str | Path,
    features: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> np.ndarray:
    predictor = GaitTCNPredictor(
        checkpoint_path,
        device=device,
        batch_size=batch_size,
    )
    return predictor.predict(features)


class GaitTCNPredictor:
    """Load one TCN checkpoint once and score pose windows repeatedly."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 64,
        window_frames: int = 64,
        expected_task: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("schema_version") != "gait-tcn-checkpoint-v1":
            raise ValueError("unsupported gait TCN checkpoint schema")
        self.task = str(checkpoint.get("task", ""))
        if expected_task is not None and self.task != expected_task:
            raise ValueError(
                f"checkpoint task {self.task!r} does not match {expected_task!r}"
            )
        label_mapping = checkpoint.get("label_mapping")
        if expected_task == "gait_instability_vs_normal_activity" and label_mapping != {
            "normal_activity": 0,
            "gait_instability": 1,
        }:
            raise ValueError("checkpoint label mapping is not a gait stability contract")
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, Mapping):
            raise ValueError("gait TCN checkpoint is missing model_config")
        self.model = LightweightGaitTCN(**dict(model_config))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = _select_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.window_frames = window_frames
        self.model_version = str(
            checkpoint.get("model_version")
            or f"gait-tcn-checkpoint-v1:{_sha256_file(self.checkpoint_path)[:12]}"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        inputs = np.asarray(features, dtype=np.float32)
        if inputs.ndim == 3:
            inputs = inputs[None, ...]
        if inputs.ndim != 4:
            raise ValueError("gait TCN inference input must have shape [N, T, V, C]")
        if inputs.shape[2:] != (
            len(CANONICAL_GAIT_JOINTS),
            len(GAIT_TCN_CHANNELS),
        ):
            raise ValueError(f"unexpected gait TCN input shape: {inputs.shape}")
        if not np.isfinite(inputs).all():
            raise ValueError("gait TCN inference input contains non-finite values")

        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                batch = torch.from_numpy(inputs[start : start + self.batch_size]).to(
                    self.device
                )
                scores = torch.softmax(self.model(batch), dim=1)[:, 1]
                probabilities.append(scores.detach().cpu().numpy())
        if not probabilities:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(probabilities).astype(np.float32, copy=False)

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> float:
        selected = list(records)[-self.window_frames :]
        tensor = build_gait_tensor(selected, window_frames=self.window_frames)
        if not np.any(tensor[..., -1] > 0):
            raise ValueError("gait window has no usable keypoint observations")
        return float(self.predict(tensor)[0])


def train_gait_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: GaitTCNTrainingConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    training = config or GaitTCNTrainingConfig()
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path) if metadata_path is not None else source_path.with_name("metadata.json")
    )
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"prepared gait metadata not found: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("dataset_sha256") != _sha256_file(source_path):
        raise ValueError("prepared gait dataset SHA-256 does not match metadata")
    if source_metadata.get("joint_order") != list(CANONICAL_GAIT_JOINTS):
        raise ValueError("prepared gait joint order does not match the TCN contract")
    if source_metadata.get("channel_order") != list(GAIT_TCN_CHANNELS):
        raise ValueError("prepared gait channel order does not match the TCN contract")

    arrays = load_prepared_gait_dataset(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_model.pt"
    metrics_path = destination / "metrics.json"
    history_path = destination / "history.jsonl"
    predictions_path = destination / "test_participant_predictions.jsonl"
    window_predictions_path = destination / "test_window_predictions.jsonl"
    for path in (
        checkpoint_path,
        metrics_path,
        history_path,
        predictions_path,
        window_predictions_path,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(f"training output already exists: {path}")

    _seed_everything(training.seed)
    device = _select_device(training.device)
    features = arrays["features"].astype(np.float32, copy=False)
    labels = arrays["labels"].astype(np.int64, copy=False)
    partitions = arrays["partitions"].astype(str)
    aggregate_groups = source_metadata.get("task") == (
        "retrospective_fall_history_proxy_binary_classification"
    )
    indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation", "test")
    }
    for partition, partition_indices in indices.items():
        if len(partition_indices) == 0:
            raise ValueError(f"prepared gait {partition} partition is empty")
        if set(labels[partition_indices].tolist()) != {0, 1}:
            raise ValueError(f"prepared gait {partition} partition lacks a binary class")

    datasets = {
        partition: _GaitWindowDataset(
            features,
            labels,
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
        "test": DataLoader(
            datasets["test"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    model = LightweightGaitTCN(
        hidden_channels=training.hidden_channels,
        kernel_size=training.kernel_size,
        dilations=training.dilations,
        dropout=training.dropout,
    ).to(device)
    train_counts = np.bincount(labels[indices["train"]], minlength=2).astype(np.float64)
    class_weights = train_counts.sum() / (2.0 * np.maximum(train_counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, training.epochs + 1):
        train_loss = _train_epoch(model, loaders["train"], criterion, optimizer, device)
        validation = _evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            participant_ids=arrays["participant_ids"],
            sample_ids=arrays["sample_ids"],
            aggregate_groups=aggregate_groups,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_window_metrics": validation["window_metrics"],
            "validation_participant_metrics": validation["participant_metrics"],
        }
        history.append(history_row)
        if validation["loss"] < best_validation_loss - 1e-8:
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            break

    if best_state is None:
        raise RuntimeError("gait TCN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation = _evaluate(
        model,
        loaders["validation"],
        criterion,
        device,
        participant_ids=arrays["participant_ids"],
        sample_ids=arrays["sample_ids"],
        aggregate_groups=aggregate_groups,
    )
    test = _evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        participant_ids=arrays["participant_ids"],
        sample_ids=arrays["sample_ids"],
        aggregate_groups=aggregate_groups,
    )

    model_config = {
        "joint_count": len(CANONICAL_GAIT_JOINTS),
        "input_channels": len(GAIT_TCN_CHANNELS),
        "hidden_channels": training.hidden_channels,
        "kernel_size": training.kernel_size,
        "dilations": list(training.dilations),
        "dropout": training.dropout,
        "class_count": 2,
    }
    checkpoint = {
        "schema_version": "gait-tcn-checkpoint-v1",
        "state_dict": best_state,
        "model_config": model_config,
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "label_mapping": source_metadata["label_mapping"],
        "task": source_metadata["task"],
        "threshold": 0.5,
        "best_epoch": best_epoch,
        "dataset_sha256": source_metadata["dataset_sha256"],
        "training_config": asdict(training),
    }
    _write_torch_atomic(checkpoint_path, checkpoint)
    _write_jsonl_atomic(history_path, history)
    _write_jsonl_atomic(predictions_path, test["participant_predictions"])
    _write_jsonl_atomic(window_predictions_path, test["window_predictions"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "schema_version": "gait-tcn-training-report-v1",
        "task": source_metadata["task"],
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "device": str(device),
        "parameter_count": parameter_count,
        "class_weights": class_weights.tolist(),
        "validation": {
            "loss": validation["loss"],
            "window_metrics": validation["window_metrics"],
            "participant_metrics": validation["participant_metrics"],
        },
        "test": {
            "loss": test["loss"],
            "window_metrics": test["window_metrics"],
            "participant_metrics": test["participant_metrics"],
        },
        "dataset_path": source_path.as_posix(),
        "dataset_sha256": source_metadata["dataset_sha256"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "limitations": list(source_metadata.get("limitations", [])),
    }
    _write_json_atomic(metrics_path, metrics)
    return {
        "output_dir": destination.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameter_count": parameter_count,
        "device": str(device),
        "validation_participant_metrics": validation["participant_metrics"],
        "test_participant_metrics": test["participant_metrics"],
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for features, labels, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        sample_count += len(labels)
    return total_loss / max(sample_count, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    participant_ids: np.ndarray,
    sample_ids: np.ndarray,
    aggregate_groups: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels_out: list[int] = []
    probabilities: list[float] = []
    participants: list[str] = []
    samples: list[str] = []
    with torch.inference_mode():
        for features, labels, source_indices in loader:
            features = features.to(device)
            labels_device = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels_device)
            positive_probability = torch.softmax(logits, dim=1)[:, 1]
            total_loss += float(loss.detach().cpu()) * len(labels)
            sample_count += len(labels)
            labels_out.extend(int(value) for value in labels.tolist())
            probabilities.extend(float(value) for value in positive_probability.cpu().tolist())
            participants.extend(
                str(participant_ids[int(index)]) for index in source_indices.tolist()
            )
            samples.extend(str(sample_ids[int(index)]) for index in source_indices.tolist())
    participant_rows = (
        aggregate_participant_predictions(
            participant_ids=participants,
            labels=labels_out,
            probabilities=probabilities,
        )
        if aggregate_groups
        else []
    )
    window_rows = [
        {
            "sample_id": sample_id,
            "label": label,
            "gait_risk_score": probability,
            "predicted_label": int(probability >= 0.5),
        }
        for sample_id, label, probability in zip(
            samples, labels_out, probabilities, strict=True
        )
    ]
    return {
        "loss": total_loss / max(sample_count, 1),
        "window_metrics": _binary_metrics(labels_out, probabilities),
        "participant_metrics": (
            _binary_metrics(
                [row["label"] for row in participant_rows],
                [row["probability"] for row in participant_rows],
            )
            if participant_rows
            else None
        ),
        "participant_predictions": participant_rows,
        "window_predictions": window_rows,
    }


def _binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "sample_count": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": _roc_auc(y_true, y_score),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": 0.5,
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if not positive_count or not negative_count:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)


def _select_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    return torch.device(normalized)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_norm_count(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


def _participant_sort_key(participant_id: str) -> tuple[str, int]:
    prefix = participant_id.rstrip("0123456789")
    suffix = participant_id[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


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
