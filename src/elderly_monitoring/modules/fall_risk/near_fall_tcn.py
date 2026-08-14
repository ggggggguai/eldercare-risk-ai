from __future__ import annotations

import hashlib
import json
import os
import random
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

from elderly_monitoring.modules.fall_risk.near_fall_training import (
    NEAR_FALL_CHANNELS,
    NEAR_FALL_JOINTS,
    apply_normalization,
)


TASK = "near_fall_recovery_confirmation_v1"


@dataclass(frozen=True)
class NearFallTCNConfig:
    epochs: int = 40
    batch_size: int = 32
    hidden_channels: int = 24
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    threshold: float = 0.5
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be within (0, 1)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")


class _CausalDepthwiseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = F.pad(inputs, (self.left_padding, 0))
        encoded = self.depthwise(encoded)
        encoded = self.pointwise(encoded)
        encoded = self.dropout(F.gelu(self.norm(encoded)))
        return inputs + encoded


class NearFallTCN(nn.Module):
    """Binary causal TCN for recovery-aligned near-fall confirmation."""

    def __init__(
        self,
        *,
        joint_count: int = len(NEAR_FALL_JOINTS),
        input_channels: int = len(NEAR_FALL_CHANNELS),
        hidden_channels: int = 24,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.joint_count = int(joint_count)
        self.input_channels = int(input_channels)
        self.input_projection = nn.Sequential(
            nn.Conv1d(self.joint_count * self.input_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            *[
                _CausalDepthwiseBlock(
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
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("near-fall TCN input must have shape [B,T,V,C]")
        if inputs.shape[2:] != (self.joint_count, self.input_channels):
            raise ValueError(
                f"expected V,C={self.joint_count},{self.input_channels}; "
                f"got {inputs.shape[2]},{inputs.shape[3]}"
            )
        frame_mask = (inputs[..., 7].amax(dim=2) > 0).to(inputs.dtype)
        batch_size, frame_count, _, _ = inputs.shape
        encoded = inputs.reshape(batch_size, frame_count, -1).transpose(1, 2)
        encoded = self.temporal_blocks(self.input_projection(encoded))
        weights = frame_mask.unsqueeze(1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return self.classifier(pooled)


class _WindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.labels = labels
        self.weights = weights
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            torch.tensor(float(self.weights[index]), dtype=torch.float32),
        )


def train_near_fall_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: NearFallTCNConfig | None = None,
    allow_synthetic: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    training = config or NearFallTCNConfig()
    data_path = Path(dataset_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"near-fall dataset not found: {data_path}")
    metadata_file = (
        Path(metadata_path) if metadata_path is not None else data_path.with_name("metadata.json")
    )
    if not metadata_file.is_file():
        raise FileNotFoundError(f"near-fall metadata not found: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    _validate_metadata(metadata, data_path, allow_synthetic=allow_synthetic)
    arrays = _load_dataset(data_path)
    _validate_dataset(arrays)

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"near-fall training output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_path = destination / "best_model.pt"
    last_checkpoint_path = destination / "last_model.pt"
    metrics_path = destination / "metrics.json"
    config_path = destination / "config.json"
    validation_predictions_path = destination / "validation_predictions.jsonl"

    _set_deterministic(training.seed)
    device = _resolve_device(training.device)
    model = NearFallTCN(
        hidden_channels=training.hidden_channels,
        kernel_size=training.kernel_size,
        dilations=training.dilations,
        dropout=training.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    start_epoch = 0
    best_validation_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    resumed = False
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_contract(checkpoint, arrays, metadata)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        history = list(checkpoint.get("history", []))
        resumed = True

    train_indices = np.flatnonzero(arrays["partitions"] == "train")
    validation_indices = np.flatnonzero(arrays["partitions"] == "validation")
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        _WindowDataset(
            arrays["features"], arrays["labels"], arrays["sample_weights"], train_indices
        ),
        batch_size=training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _WindowDataset(
            arrays["features"],
            arrays["labels"],
            arrays["sample_weights"],
            validation_indices,
        ),
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, training.epochs):
        train_result = _run_epoch(
            model, train_loader, device=device, optimizer=optimizer
        )
        validation_result = _run_epoch(
            model, validation_loader, device=device, optimizer=None
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "validation_loss": validation_result["loss"],
            "validation_accuracy": validation_result["accuracy"],
        }
        history.append(epoch_record)
        improved = validation_result["loss"] < best_validation_loss - 1e-12
        if improved:
            best_validation_loss = validation_result["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _write_torch_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    history=history,
                    arrays=arrays,
                    metadata=metadata,
                    training=training,
                    dataset_path=data_path,
                    metadata_path=metadata_file,
                ),
            )
        else:
            epochs_without_improvement += 1
        _write_torch_atomic(
            last_checkpoint_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                history=history,
                arrays=arrays,
                metadata=metadata,
                training=training,
                dataset_path=data_path,
                metadata_path=metadata_file,
            ),
        )
        if epochs_without_improvement >= training.patience:
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("near-fall training did not produce a checkpoint")
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    train_metrics = _evaluate_partition(
        model,
        arrays,
        train_indices,
        partition="train",
        device=device,
        batch_size=training.batch_size,
        threshold=training.threshold,
    )
    validation_metrics = _evaluate_partition(
        model,
        arrays,
        validation_indices,
        partition="validation",
        device=device,
        batch_size=training.batch_size,
        threshold=training.threshold,
    )
    validation_predictions = validation_metrics.pop("predictions")
    train_metrics.pop("predictions")
    _write_jsonl_atomic(validation_predictions_path, validation_predictions)
    run_status = (
        "synthetic_smoke_only"
        if metadata.get("synthetic")
        else str(metadata.get("status", "development_provisional"))
    )
    metrics = {
        "schema_version": "near-fall-tcn-training-metrics-v1",
        "task": TASK,
        "status": run_status,
        "synthetic": bool(metadata.get("synthetic")),
        "disclaimer": (
            "synthetic infrastructure smoke; not a supervised real-data model result"
            if metadata.get("synthetic")
            else "development train/validation only; test remains locked"
        ),
        "seed": training.seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train": train_metrics,
        "validation": validation_metrics,
        "test": None,
        "test_evaluated": False,
        "history": history,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "data_provenance": {
            "dataset_sha256": _sha256_file(data_path),
            "metadata_sha256": _sha256_file(metadata_file),
            "source_split_id": metadata.get("source_split_id"),
        },
    }
    _write_json_atomic(metrics_path, metrics)
    _write_json_atomic(
        config_path,
        {
            "schema_version": "near-fall-tcn-run-config-v1",
            "task": TASK,
            "training": asdict(training),
            "synthetic": bool(metadata.get("synthetic")),
            "test_policy": "not_read_not_evaluated",
            "dataset_sha256": _sha256_file(data_path),
            "metadata_sha256": _sha256_file(metadata_file),
        },
    )
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "last_checkpoint_path": last_checkpoint_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "config_path": config_path.as_posix(),
        "validation_predictions_path": validation_predictions_path.as_posix(),
        "epochs_completed": len(history),
        "resumed": resumed,
        "synthetic": bool(metadata.get("synthetic")),
        "test_evaluated": False,
    }


class NearFallTCNPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = _resolve_device(device)
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        _validate_checkpoint_static(checkpoint)
        model_config = dict(checkpoint["model_config"])
        model_config["dilations"] = tuple(model_config["dilations"])
        self.model = NearFallTCN(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.input_contract = dict(checkpoint["input_contract"])
        self.window_frames = int(self.input_contract["window_frames"])
        normalization = self.input_contract["normalization"]
        self.normalization = {
            "mean": np.asarray(normalization["mean"], dtype=np.float32),
            "std": np.asarray(normalization["std"], dtype=np.float32),
        }

    def predict_tensor(
        self, tensor: np.ndarray, *, already_normalized: bool = False
    ) -> dict[str, Any]:
        values = np.asarray(tensor, dtype=np.float32)
        expected_shape = (self.window_frames, len(NEAR_FALL_JOINTS), len(NEAR_FALL_CHANNELS))
        if values.shape != expected_shape:
            raise ValueError(
                f"near-fall predictor expected shape {expected_shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            return {"status": "unavailable", "reason": "non_finite_input", "near_fall_event_score": None}
        valid_ratio = float(np.mean(values[..., 7] > 0))
        if valid_ratio < float(self.input_contract["min_valid_joint_ratio"]):
            return {
                "status": "unavailable",
                "reason": "insufficient_input_quality",
                "near_fall_event_score": None,
            }
        if not already_normalized:
            quality_values = values[..., 6][values[..., 7] > 0]
            mean_quality = float(np.mean(quality_values)) if len(quality_values) else 0.0
            if mean_quality < float(self.input_contract["min_mean_joint_quality"]):
                return {
                    "status": "unavailable",
                    "reason": "insufficient_input_quality",
                    "near_fall_event_score": None,
                }
            values = apply_normalization(values[None, ...], self.normalization)[0]
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(values[None, ...]).to(self.device)
            )
            score = float(torch.softmax(logits, dim=1)[0, 1].cpu())
        if not np.isfinite(score):
            return {"status": "inference_error", "reason": "non_finite_output", "near_fall_event_score": None}
        return {
            "status": "valid",
            "near_fall_event_score": score,
            "task": TASK,
            "confirmation_timing": "recovery_frame",
        }


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    required = {
        "features",
        "labels",
        "partitions",
        "sample_ids",
        "event_ids",
        "subject_ids",
        "source_group_ids",
        "sample_group_ids",
        "split_group_ids",
        "sample_weights",
        "loss_eligible",
        "normalization_mean",
        "normalization_std",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"near-fall dataset is missing arrays: {missing}")
        return {name: archive[name] for name in required}


def _validate_metadata(
    metadata: Mapping[str, Any], data_path: Path, *, allow_synthetic: bool
) -> None:
    if metadata.get("schema_version") != "near-fall-event-dataset-v1":
        raise ValueError("unsupported near-fall dataset metadata schema")
    if metadata.get("task") != TASK:
        raise ValueError("near-fall dataset task mismatch")
    synthetic = metadata.get("synthetic") is True
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic near-fall training requires allow_synthetic")
    if not synthetic and metadata.get("training_ready") is not True:
        raise ValueError("near-fall real dataset is not training ready")
    if metadata.get("test_pose_read") is not False or metadata.get("test_evaluated") is not False:
        raise ValueError("near-fall dataset violates the locked-test contract")
    if metadata.get("joint_order") != list(NEAR_FALL_JOINTS):
        raise ValueError("near-fall dataset joint order mismatch")
    if metadata.get("channel_order") != list(NEAR_FALL_CHANNELS):
        raise ValueError("near-fall dataset channel order mismatch")
    if metadata.get("dataset_sha256") != _sha256_file(data_path):
        raise ValueError("near-fall dataset SHA-256 mismatch")
    normalization = metadata.get("normalization", {})
    if normalization.get("fit_partition") != "train":
        raise ValueError("near-fall normalization was not fit on train only")


def _validate_dataset(arrays: Mapping[str, np.ndarray]) -> None:
    features = np.asarray(arrays["features"], dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    partitions = np.asarray(arrays["partitions"]).astype(str)
    if features.ndim != 4 or features.shape[2:] != (10, 8):
        raise ValueError("near-fall dataset features must have shape [N,T,10,8]")
    count = len(features)
    if any(len(np.asarray(value)) != count for name, value in arrays.items() if name not in {"normalization_mean", "normalization_std"}):
        raise ValueError("near-fall dataset array length mismatch")
    if set(partitions.tolist()) != {"train", "validation"}:
        raise ValueError("near-fall dataset must contain train/validation only")
    if not np.all(np.asarray(arrays["loss_eligible"], dtype=bool)):
        raise ValueError("ignore or low-quality near-fall rows cannot enter loss")
    if not np.isfinite(features).all():
        raise ValueError("near-fall dataset contains non-finite features")
    weights = np.asarray(arrays["sample_weights"], dtype=np.float32)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("near-fall sample weights must be finite and positive")
    for partition in ("train", "validation"):
        if set(labels[partitions == partition].tolist()) != {0, 1}:
            raise ValueError(f"near-fall {partition} partition lacks a binary class")
    for field in ("event_ids", "subject_ids", "source_group_ids", "sample_group_ids", "split_group_ids"):
        seen: dict[str, set[str]] = {}
        for value, partition in zip(np.asarray(arrays[field]).astype(str), partitions, strict=True):
            normalized = value.strip()
            if not normalized or normalized.lower() in {"unknown", "none", "null"}:
                continue
            seen.setdefault(normalized, set()).add(partition)
        leaked = [value for value, values in seen.items() if len(values) > 1]
        if leaked:
            raise ValueError(f"near-fall {field} crosses partitions: {leaked[:5]}")
    event_ids = np.asarray(arrays["event_ids"]).astype(str)
    for event_id in np.unique(event_ids):
        if not np.isclose(float(weights[event_ids == event_id].sum()), 1.0, atol=1e-6):
            raise ValueError(f"near-fall event weights do not sum to 1: {event_id}")


def _run_epoch(
    model: NearFallTCN,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    loss_numerator = 0.0
    weight_total = 0.0
    correct_numerator = 0.0
    for features, labels, weights in loader:
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        losses = F.cross_entropy(logits, labels, reduction="none")
        loss = (losses * weights).sum() / weights.sum()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch_weight = float(weights.sum().detach().cpu())
        loss_numerator += float((losses * weights).sum().detach().cpu())
        correct_numerator += float(
            (((logits.argmax(dim=1) == labels).to(weights.dtype)) * weights)
            .sum()
            .detach()
            .cpu()
        )
        weight_total += batch_weight
    if weight_total <= 0:
        raise ValueError("near-fall loader has no positive sample weight")
    return {
        "loss": loss_numerator / weight_total,
        "accuracy": correct_numerator / weight_total,
    }


def _evaluate_partition(
    model: NearFallTCN,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    partition: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    labels = np.asarray(arrays["labels"], dtype=np.int64)[indices]
    weights = np.asarray(arrays["sample_weights"], dtype=np.float64)[indices]
    scores = np.zeros(len(indices), dtype=np.float64)
    losses = np.zeros(len(indices), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            features = torch.from_numpy(
                np.asarray(arrays["features"])[batch_indices]
            ).to(device)
            batch_labels = torch.from_numpy(
                np.asarray(arrays["labels"], dtype=np.int64)[batch_indices]
            ).to(device)
            logits = model(features)
            end = start + len(batch_indices)
            scores[start:end] = (
                torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            )
            losses[start:end] = (
                F.cross_entropy(logits, batch_labels, reduction="none")
                .detach()
                .cpu()
                .numpy()
            )
    predictions = (scores >= threshold).astype(np.int64)
    rows = [
        {
            "sample_id": str(arrays["sample_ids"][source_index]),
            "event_id": str(arrays["event_ids"][source_index]),
            "subject_id": str(arrays["subject_ids"][source_index]),
            "source_group_id": str(arrays["source_group_ids"][source_index]),
            "split_group_id": str(arrays["split_group_ids"][source_index]),
            "partition": partition,
            "label": int(labels[offset]),
            "probability": float(scores[offset]),
            "prediction": int(predictions[offset]),
            "sample_weight": float(weights[offset]),
        }
        for offset, source_index in enumerate(indices.tolist())
    ]
    event_rows = _aggregate_event_predictions(rows, threshold=threshold)
    weight_total = float(weights.sum())
    return {
        "loss": float(np.sum(losses * weights) / weight_total),
        "accuracy": float(np.sum((predictions == labels) * weights) / weight_total),
        "window": _binary_metrics(
            labels,
            scores,
            threshold=threshold,
            sample_weight=weights,
        ),
        "event": _binary_metrics(
            np.asarray([row["label"] for row in event_rows], dtype=np.int64),
            np.asarray(
                [row["probability"] for row in event_rows], dtype=np.float64
            ),
            threshold=threshold,
        ),
        "event_count": len(event_rows),
        "predictions": rows,
    }


def _aggregate_event_predictions(
    rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["event_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for event_id, group in sorted(grouped.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"near-fall event has inconsistent labels: {event_id}")
        probability = max(float(row["probability"]) for row in group)
        output.append(
            {
                "event_id": event_id,
                "label": next(iter(labels)),
                "probability": probability,
                "prediction": int(probability >= threshold),
            }
        )
    return output


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    both_classes = set(labels.tolist()) == {0, 1}
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
        sample_weight=sample_weight,
    )
    return {
        "count": int(len(labels)),
        "threshold": threshold,
        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                    sample_weight=sample_weight,
                )
            )
            if both_classes
            else None
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
                sample_weight=sample_weight,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
                sample_weight=sample_weight,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
                sample_weight=sample_weight,
            )
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores, sample_weight=sample_weight))
            if both_classes
            else None
        ),
        "pr_auc": (
            float(
                average_precision_score(
                    labels,
                    scores,
                    sample_weight=sample_weight,
                )
            )
            if both_classes
            else None
        ),
        "confusion_matrix": matrix.astype(float).tolist(),
    }


def _checkpoint_payload(
    *,
    model: NearFallTCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    history: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    training: NearFallTCNConfig,
    dataset_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "near-fall-tcn-checkpoint-v1",
        "task": TASK,
        "status": (
            "synthetic_smoke_only"
            if metadata.get("synthetic")
            else str(metadata.get("status", "development_provisional"))
        ),
        "target_semantics": "recovery-confirmed binary near-fall; not onset early warning",
        "label_mapping": {"explicit_negative": 0, "near_fall": 1},
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": list(history),
        "model_config": {
            "joint_count": len(NEAR_FALL_JOINTS),
            "input_channels": len(NEAR_FALL_CHANNELS),
            "hidden_channels": training.hidden_channels,
            "kernel_size": training.kernel_size,
            "dilations": list(training.dilations),
            "dropout": training.dropout,
        },
        "input_contract": {
            "window_frames": int(arrays["features"].shape[1]),
            "joint_order": list(NEAR_FALL_JOINTS),
            "channel_order": list(NEAR_FALL_CHANNELS),
            "causal": True,
            "positive_window_end": "recovery_frame",
            "min_valid_joint_ratio": 0.50,
            "min_mean_joint_quality": 0.30,
            "normalization": {
                "fit_partition": "train",
                "mean": np.asarray(arrays["normalization_mean"]).astype(float).tolist(),
                "std": np.asarray(arrays["normalization_std"]).astype(float).tolist(),
            },
        },
        "training_config": asdict(training),
        "synthetic": bool(metadata.get("synthetic")),
        "test_evaluated": False,
        "data_provenance": {
            "dataset_sha256": _sha256_file(dataset_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "source_split_id": metadata.get("source_split_id"),
        },
    }


def _validate_checkpoint_static(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != "near-fall-tcn-checkpoint-v1":
        raise ValueError("unsupported near-fall TCN checkpoint schema")
    if checkpoint.get("task") != TASK:
        raise ValueError("near-fall checkpoint task mismatch")
    if checkpoint.get("label_mapping") != {"explicit_negative": 0, "near_fall": 1}:
        raise ValueError("near-fall checkpoint label mapping mismatch")
    contract = checkpoint.get("input_contract", {})
    if contract.get("joint_order") != list(NEAR_FALL_JOINTS):
        raise ValueError("near-fall checkpoint joint order mismatch")
    if contract.get("channel_order") != list(NEAR_FALL_CHANNELS):
        raise ValueError("near-fall checkpoint channel order mismatch")
    if contract.get("causal") is not True or contract.get("positive_window_end") != "recovery_frame":
        raise ValueError("near-fall checkpoint is not recovery-aligned causal inference")
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError("near-fall checkpoint violates locked-test policy")


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    _validate_checkpoint_static(checkpoint)
    contract = checkpoint["input_contract"]
    if int(contract["window_frames"]) != int(arrays["features"].shape[1]):
        raise ValueError("resume checkpoint window contract mismatch")
    if bool(checkpoint.get("synthetic")) != bool(metadata.get("synthetic")):
        raise ValueError("resume checkpoint synthetic status mismatch")


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _write_jsonl_atomic(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(partial, path)
