from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


TASK = "sit_stand_continuous_causal_tcn_v1"


@dataclass(frozen=True)
class ContinuousSitStandTCNConfig:
    epochs: int = 2
    batch_size: int = 64
    hidden_channels: int = 16
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    frame_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.5
    boundary_positive_weight: float = 12.0
    boundary_tolerance_frames: int = 1
    presence_loss_weight: float = 0.5
    direction_loss_weight: float = 0.5
    patience: int = 8
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
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
        if any(
            value < 0
            for value in (
                self.frame_loss_weight,
                self.boundary_loss_weight,
                self.presence_loss_weight,
                self.direction_loss_weight,
            )
        ):
            raise ValueError("loss weights must be non-negative")
        if self.boundary_positive_weight < 1:
            raise ValueError("boundary_positive_weight must be at least 1")
        if self.boundary_tolerance_frames < 0:
            raise ValueError("boundary_tolerance_frames must be non-negative")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps or auto")


@dataclass(frozen=True)
class SitStandStreamDecoderConfig:
    presence_threshold: float = 0.5
    state_threshold: float = 0.5
    boundary_threshold: float = 0.5
    confirmation_frames: int = 2
    event_merge_gap_sec: float = 0.5
    max_sequence_gap_sec: float = 0.5
    minimum_duration_sec: float = 0.125

    def __post_init__(self) -> None:
        for field in (
            "presence_threshold",
            "state_threshold",
            "boundary_threshold",
        ):
            value = float(getattr(self, field))
            if not 0 < value < 1:
                raise ValueError(f"{field} must be within (0, 1)")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if self.event_merge_gap_sec < 0:
            raise ValueError("event_merge_gap_sec must be non-negative")
        if self.max_sequence_gap_sec <= 0:
            raise ValueError("max_sequence_gap_sec must be positive")
        if self.minimum_duration_sec <= 0:
            raise ValueError("minimum_duration_sec must be positive")


def decode_sit_stand_stream(
    rows: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    stream_id: str,
    config: SitStandStreamDecoderConfig | None = None,
    model_version: str = "sit-stand-continuous-tcn-v1-provisional",
) -> list[dict[str, Any]]:
    """Decode causal per-cutoff probabilities into non-overlapping events."""
    contract = config or SitStandStreamDecoderConfig()
    ordered = sorted((dict(row) for row in rows), key=lambda row: float(row["timestamp_sec"]))
    timestamps = [float(row["timestamp_sec"]) for row in ordered]
    if any(not np.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("stream timestamps must be finite and non-negative")
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("stream timestamps must be unique")

    events: list[dict[str, Any]] = []
    candidate_direction: str | None = None
    candidate_rows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    previous_timestamp: float | None = None

    for row, timestamp in zip(ordered, timestamps, strict=True):
        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp > contract.max_sequence_gap_sec
        ):
            if active is not None:
                _append_decoded_event(
                    events,
                    active,
                    end_time=float(active["last_evidence_time"]),
                    video_id=video_id,
                    stream_id=stream_id,
                    model_version=model_version,
                    config=contract,
                )
            active = None
            candidate_direction = None
            candidate_rows = []
        direction, evidence_score = _stream_direction(row, contract)
        onset_score, offset_score = _boundary_scores(row)
        if active is not None:
            if direction == active["transition_type"]:
                active["last_evidence_time"] = timestamp
                active["score"] = max(float(active["score"]), evidence_score)
                active["background_count"] = 0
            elif direction is None:
                active["background_count"] = int(active["background_count"]) + 1
            if offset_score >= contract.boundary_threshold or (
                direction is not None and direction != active["transition_type"]
            ) or int(active["background_count"]) >= contract.confirmation_frames:
                end_time = (
                    timestamp
                    if offset_score >= contract.boundary_threshold
                    else float(active["last_evidence_time"])
                )
                _append_decoded_event(
                    events,
                    active,
                    end_time=float(end_time if end_time is not None else timestamp),
                    video_id=video_id,
                    stream_id=stream_id,
                    model_version=model_version,
                    config=contract,
                )
                active = None

        if active is None:
            if direction is None:
                candidate_direction = None
                candidate_rows = []
            else:
                if direction != candidate_direction:
                    candidate_direction = direction
                    candidate_rows = []
                candidate_rows.append(
                    {
                        "timestamp": timestamp,
                        "score": evidence_score,
                        "onset_score": onset_score,
                    }
                )
                if len(candidate_rows) >= contract.confirmation_frames:
                    confirmed = candidate_rows[-contract.confirmation_frames :]
                    onset_candidates = [
                        item
                        for item in candidate_rows
                        if float(item["onset_score"]) >= contract.boundary_threshold
                    ]
                    onset_time = float(
                        onset_candidates[0]["timestamp"]
                        if onset_candidates
                        else confirmed[0]["timestamp"]
                    )
                    active = {
                        "transition_type": direction,
                        "onset_time": onset_time,
                        "last_evidence_time": timestamp,
                        "score": max(float(item["score"]) for item in candidate_rows),
                        "background_count": 0,
                    }
                    candidate_direction = None
                    candidate_rows = []
                    if offset_score >= contract.boundary_threshold:
                        _append_decoded_event(
                            events,
                            active,
                            end_time=timestamp,
                            video_id=video_id,
                            stream_id=stream_id,
                            model_version=model_version,
                            config=contract,
                        )
                        active = None
        previous_timestamp = timestamp

    if active is not None:
        _append_decoded_event(
            events,
            active,
            end_time=float(active["last_evidence_time"]),
            video_id=video_id,
            stream_id=stream_id,
            model_version=model_version,
            config=contract,
        )
    return _merge_decoded_events(events, contract.event_merge_gap_sec)


def _stream_direction(
    row: Mapping[str, Any], config: SitStandStreamDecoderConfig
) -> tuple[str | None, float]:
    presence = _probability(row.get("presence_score"), "presence_score")
    frame = _probability_vector(row.get("frame_probabilities"), 3, "frame_probabilities")
    direction = _probability_vector(
        row.get("direction_probabilities"), 2, "direction_probabilities"
    )
    direction_index = int(np.argmax(direction))
    state_index = direction_index + 1
    score = min(presence, frame[state_index], direction[direction_index])
    if presence < config.presence_threshold or frame[state_index] < config.state_threshold:
        return None, score
    return ("sit_to_stand" if direction_index == 0 else "stand_to_sit"), score


def _boundary_scores(row: Mapping[str, Any]) -> tuple[float, float]:
    values = _probability_vector(
        row.get("boundary_probabilities"), 2, "boundary_probabilities"
    )
    return values[0], values[1]


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _probability_vector(value: Any, size: int, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{field} must contain {size} probabilities")
    result = [_probability(item, field) for item in value]
    return result


def _append_decoded_event(
    events: list[dict[str, Any]],
    active: Mapping[str, Any],
    *,
    end_time: float,
    video_id: str,
    stream_id: str,
    model_version: str,
    config: SitStandStreamDecoderConfig,
) -> None:
    onset = float(active["onset_time"])
    offset = max(end_time, onset + config.minimum_duration_sec)
    identity = f"{video_id}|{stream_id}|{active['transition_type']}|{onset:.6f}|{offset:.6f}"
    events.append(
        {
            "prediction_id": "sitstandpred_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            "video_id": video_id,
            "stream_id": stream_id,
            "transition_type": str(active["transition_type"]),
            "onset_time": round(onset, 6),
            "offset_time": round(offset, 6),
            "score": round(float(active["score"]), 8),
            "quality_state": "valid",
            "model_version": model_version,
        }
    )


def _merge_decoded_events(
    events: Sequence[Mapping[str, Any]], gap_sec: float
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for value in sorted(events, key=lambda row: (float(row["onset_time"]), str(row["prediction_id"]))):
        row = dict(value)
        if (
            merged
            and merged[-1]["transition_type"] == row["transition_type"]
            and float(row["onset_time"]) - float(merged[-1]["offset_time"]) <= gap_sec
        ):
            merged[-1]["offset_time"] = max(
                float(merged[-1]["offset_time"]), float(row["offset_time"])
            )
            merged[-1]["score"] = max(float(merged[-1]["score"]), float(row["score"]))
        else:
            merged.append(row)
    return merged


class _CausalTemporalBlock(nn.Module):
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
        self.norm = _PerFrameLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = F.pad(inputs, (self.left_padding, 0))
        values = self.depthwise(values)
        values = self.pointwise(values)
        values = self.dropout(F.gelu(self.norm(values)))
        return inputs + values


class ContinuousSitStandTCN(nn.Module):
    def __init__(
        self,
        *,
        joint_count: int = 14,
        input_channels: int = 9,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.joint_count = joint_count
        self.input_channels = input_channels
        self.input_projection = nn.Sequential(
            nn.Conv1d(joint_count * input_channels, hidden_channels, 1, bias=False),
            _PerFrameLayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                _CausalTemporalBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.frame_head = nn.Conv1d(hidden_channels, 3, 1)
        self.boundary_head = nn.Conv1d(hidden_channels, 2, 1)
        self.presence_head = nn.Linear(hidden_channels, 2)
        self.direction_head = nn.Linear(hidden_channels, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError("continuous sit-stand input must be [B,T,V,C]")
        if tuple(inputs.shape[2:]) != (self.joint_count, self.input_channels):
            raise ValueError("continuous sit-stand input joint/channel shape mismatch")
        frame_mask = (inputs[..., -1].amax(dim=2) > 0).to(inputs.dtype)
        batch, frames, _, _ = inputs.shape
        encoded = inputs.reshape(batch, frames, -1).transpose(1, 2)
        encoded = self.temporal(self.input_projection(encoded))
        weights = frame_mask.unsqueeze(1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return {
            "frame_logits": self.frame_head(encoded).transpose(1, 2),
            "boundary_logits": self.boundary_head(encoded).transpose(1, 2),
            "presence_logits": self.presence_head(pooled),
            "direction_logits": self.direction_head(pooled),
        }


class _ContinuousDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        return (
            torch.from_numpy(self.arrays["features"][index]),
            torch.tensor(int(self.arrays["targets"][index]), dtype=torch.long),
            torch.from_numpy(self.arrays["frame_targets"][index]),
            torch.from_numpy(self.arrays["boundary_targets"][index]),
            torch.from_numpy(self.arrays["supervision_masks"][index]),
            torch.tensor(float(self.arrays["sample_weights"][index])),
            torch.tensor(index, dtype=torch.long),
        )


def train_continuous_sit_stand_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: ContinuousSitStandTCNConfig | None = None,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    if not allow_provisional:
        raise ValueError("continuous sit-stand training is provisional")
    training = config or ContinuousSitStandTCNConfig()
    source = Path(dataset_path)
    metadata_file = Path(metadata_path) if metadata_path else source.with_name("metadata.json")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("continuous_model_training_allowed") is not True or not metadata.get(
        "materialized_development_gate", {}
    ).get("passed"):
        raise ValueError("materialized development gate has not passed")
    if any(metadata.get(key) is True for key in ("test_pose_read", "test_features_generated", "test_evaluated")):
        raise ValueError("continuous sit-stand dataset accessed test data")
    arrays = _load_dataset(source)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"continuous sit-stand output already exists: {destination}")
    destination.mkdir(parents=True)
    _seed(training.seed)
    device = _device(training.device)
    partitions = arrays["partitions"].astype(str)
    indices = {
        name: np.flatnonzero(partitions == name) for name in ("train", "validation")
    }
    for name, values in indices.items():
        if len(values) == 0:
            raise ValueError(f"continuous sit-stand {name} partition is empty")
        classes = set(arrays["targets"][values].tolist())
        if classes != {0, 1, 2}:
            raise ValueError(f"continuous sit-stand {name} lacks target classes")
    loaders = {
        name: DataLoader(
            _ContinuousDataset(arrays, values),
            batch_size=training.batch_size,
            shuffle=name == "train",
            generator=torch.Generator().manual_seed(training.seed),
            num_workers=0,
        )
        for name, values in indices.items()
    }
    model_config = {
        "joint_count": int(arrays["features"].shape[2]),
        "input_channels": int(arrays["features"].shape[3]),
        "hidden_channels": training.hidden_channels,
        "kernel_size": training.kernel_size,
        "dilations": list(training.dilations),
        "dropout": training.dropout,
    }
    model = ContinuousSitStandTCN(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, training.epochs + 1):
        train_loss = _run_epoch(model, loaders["train"], device, training, optimizer)
        validation = _evaluate(model, loaders["validation"], device, training)
        score = (
            validation["presence"]["f1"]
            + validation["direction"]["macro_f1"]
            + validation["frame"]["macro_f1"]
            + validation["boundary_tolerant"]["f1"]
        ) / 4.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
                "selection_score": score,
                "test_access": False,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            break
    if best_state is None:
        raise RuntimeError("continuous sit-stand training produced no checkpoint")
    model.load_state_dict(best_state)
    final_validation = _evaluate(model, loaders["validation"], device, training)
    checkpoint = {
        "schema_version": "sit-stand-continuous-tcn-checkpoint-v1",
        "task": TASK,
        "status": "development_provisional",
        "state_dict": best_state,
        "model_config": model_config,
        "training_config": asdict(training),
        "dataset_sha256": _sha256(source),
        "metadata_sha256": _sha256(metadata_file),
        "best_epoch": best_epoch,
        "validation": final_validation,
        "test_access": {
            "test_pose_read": False,
            "test_features_generated": False,
            "test_evaluated": False,
        },
    }
    _atomic_torch_save(destination / "best_model.pt", checkpoint)
    (destination / "history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "sit-stand-continuous-tcn-training-v1",
        "task": TASK,
        "status": "development_provisional",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_metric": "mean_presence_direction_frame_boundary_tolerant_f1",
        "validation": final_validation,
        "dataset_sha256": checkpoint["dataset_sha256"],
        "checkpoint_sha256": _sha256(destination / "best_model.pt"),
        "test_access": checkpoint["test_access"],
    }
    (destination / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "config.json").write_text(
        json.dumps(asdict(training), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _loss(
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    frame_targets: torch.Tensor,
    boundary_targets: torch.Tensor,
    supervision_masks: torch.Tensor,
    sample_weights: torch.Tensor,
    config: ContinuousSitStandTCNConfig,
) -> dict[str, torch.Tensor]:
    frame_losses = F.cross_entropy(
        outputs["frame_logits"].transpose(1, 2), frame_targets, ignore_index=-100, reduction="none"
    )
    weights = supervision_masks * sample_weights[:, None]
    frame_loss = (frame_losses * weights).sum() / weights.sum().clamp_min(1.0)
    boundary_supervision = _dilate_boundary_targets(
        boundary_targets, config.boundary_tolerance_frames
    )
    positive_weight = torch.full(
        (2,),
        config.boundary_positive_weight,
        dtype=outputs["boundary_logits"].dtype,
        device=outputs["boundary_logits"].device,
    )
    boundary_losses = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"],
        boundary_supervision,
        pos_weight=positive_weight,
        reduction="none",
    ).mean(dim=2)
    boundary_loss = (boundary_losses * weights).sum() / weights.sum().clamp_min(1.0)
    presence = (targets > 0).long()
    presence_loss = F.cross_entropy(outputs["presence_logits"], presence)
    direction_mask = targets > 0
    direction_loss = F.cross_entropy(
        outputs["direction_logits"][direction_mask], targets[direction_mask] - 1
    )
    total = (
        config.frame_loss_weight * frame_loss
        + config.boundary_loss_weight * boundary_loss
        + config.presence_loss_weight * presence_loss
        + config.direction_loss_weight * direction_loss
    )
    return {"loss": total, "frame": frame_loss, "boundary": boundary_loss, "presence": presence_loss, "direction": direction_loss}


def _dilate_boundary_targets(targets: torch.Tensor, tolerance_frames: int) -> torch.Tensor:
    if tolerance_frames == 0:
        return targets
    kernel_size = 2 * tolerance_frames + 1
    values = targets.transpose(1, 2)
    return F.max_pool1d(
        values,
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance_frames,
    ).transpose(1, 2)


def _run_epoch(
    model: ContinuousSitStandTCN,
    loader: DataLoader[Any],
    device: torch.device,
    config: ContinuousSitStandTCNConfig,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("loss", "frame", "boundary", "presence", "direction")}
    count = 0
    for batch in loader:
        features, targets, frame_targets, boundaries, masks, weights, _ = [value.to(device) for value in batch]
        optimizer.zero_grad(set_to_none=True)
        losses = _loss(model(features), targets, frame_targets, boundaries, masks, weights, config)
        losses["loss"].backward()
        optimizer.step()
        size = len(features)
        count += size
        for name in totals:
            totals[name] += float(losses[name].detach().cpu()) * size
    return {name: value / count for name, value in totals.items()}


def _evaluate(
    model: ContinuousSitStandTCN,
    loader: DataLoader[Any],
    device: torch.device,
    config: ContinuousSitStandTCNConfig,
) -> dict[str, Any]:
    model.eval()
    targets_all: list[int] = []
    presence_predictions: list[int] = []
    direction_predictions: list[int] = []
    frame_truth: list[int] = []
    frame_predictions: list[int] = []
    boundary_truth: list[int] = []
    boundary_predictions: list[int] = []
    boundary_scores: list[float] = []
    boundary_tolerant_truth: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features, targets, frame_targets, boundaries, masks, weights, _ = [value.to(device) for value in batch]
            outputs = model(features)
            losses.append(float(_loss(outputs, targets, frame_targets, boundaries, masks, weights, config)["loss"].cpu()))
            targets_all.extend(targets.cpu().tolist())
            presence_predictions.extend(outputs["presence_logits"].argmax(dim=1).cpu().tolist())
            direction_predictions.extend(outputs["direction_logits"].argmax(dim=1).cpu().tolist())
            supervised = masks > 0
            frame_truth.extend(frame_targets[supervised].cpu().tolist())
            frame_predictions.extend(outputs["frame_logits"].argmax(dim=2)[supervised].cpu().tolist())
            expanded = supervised.unsqueeze(2).expand_as(boundaries)
            boundary_truth.extend(boundaries[expanded].long().cpu().tolist())
            tolerant_boundaries = _dilate_boundary_targets(
                boundaries, config.boundary_tolerance_frames
            )
            boundary_tolerant_truth.extend(
                tolerant_boundaries[expanded].long().cpu().tolist()
            )
            probabilities = torch.sigmoid(outputs["boundary_logits"])
            boundary_scores.extend(probabilities[expanded].cpu().tolist())
            boundary_predictions.extend(
                (probabilities[expanded] >= 0.5).long().cpu().tolist()
            )
    targets_array = np.asarray(targets_all)
    presence_truth = (targets_array > 0).astype(np.int64)
    event_mask = targets_array > 0
    direction_truth = targets_array[event_mask] - 1
    direction_prediction = np.asarray(direction_predictions)[event_mask]
    return {
        "loss": float(np.mean(losses)),
        "presence": _binary_metrics(presence_truth, np.asarray(presence_predictions)),
        "direction": {
            "macro_f1": float(f1_score(direction_truth, direction_prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(direction_truth, direction_prediction)),
        },
        "frame": {
            "macro_f1": float(f1_score(frame_truth, frame_predictions, average="macro", zero_division=0)),
            "supervised_frame_count": len(frame_truth),
        },
        "boundary": {
            **_binary_metrics(
                np.asarray(boundary_truth), np.asarray(boundary_predictions)
            ),
            "average_precision": float(
                average_precision_score(boundary_truth, boundary_scores)
            ),
            "positive_weight": config.boundary_positive_weight,
        },
        "boundary_tolerant": {
            **_binary_metrics(
                np.asarray(boundary_tolerant_truth),
                np.asarray(boundary_predictions),
            ),
            "tolerance_frames": config.boundary_tolerance_frames,
        },
    }


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    required = {
        "features",
        "targets",
        "frame_targets",
        "boundary_targets",
        "supervision_masks",
        "sample_weights",
        "sample_ids",
        "partitions",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"continuous sit-stand dataset missing arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    if arrays["features"].ndim != 4 or arrays["features"].shape[-2:] != (14, 9):
        raise ValueError("continuous sit-stand features must have shape [N,T,14,9]")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("continuous sit-stand features contain non-finite values")
    return arrays


class _PerFrameLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.transpose(1, 2)).transpose(1, 2)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return torch.device(requested)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
