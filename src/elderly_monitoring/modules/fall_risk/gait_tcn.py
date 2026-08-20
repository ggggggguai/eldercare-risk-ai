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
from elderly_monitoring.modules.fall_risk.gait_training import (
    GAIT_WALKING_ACTION_IDS,
    resample_pose_records,
    validate_frozen_partition_selection,
)


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
    evaluate_test: bool | None = None
    partition_scheme: str = "frozen"
    pretrained_checkpoint: str | None = None
    freeze_encoder_epochs: int = 0
    use_quality_as_feature: bool = False
    hierarchical_walking_gate: bool = True
    walking_gate_loss_weight: float = 0.25
    temporal_shift_frames: int = 1
    keypoint_dropout_probability: float = 0.05
    coordinate_jitter_std: float = 0.005

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
        if self.partition_scheme not in {"frozen", "fold_a", "fold_b"}:
            raise ValueError("partition_scheme must be frozen, fold_a or fold_b")
        if not 0 <= self.freeze_encoder_epochs < self.epochs:
            raise ValueError("freeze_encoder_epochs must be within [0, epochs)")
        if self.freeze_encoder_epochs and self.pretrained_checkpoint is None:
            raise ValueError("freeze_encoder_epochs requires pretrained_checkpoint")
        if self.walking_gate_loss_weight < 0:
            raise ValueError("walking_gate_loss_weight must be non-negative")
        if self.temporal_shift_frames < 0:
            raise ValueError("temporal_shift_frames must be non-negative")
        if not 0 <= self.keypoint_dropout_probability < 1:
            raise ValueError("keypoint_dropout_probability must be within [0, 1)")
        if self.coordinate_jitter_std < 0:
            raise ValueError("coordinate_jitter_std must be non-negative")


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
        use_quality_as_feature: bool = True,
        hierarchical_walking_gate: bool = False,
    ) -> None:
        super().__init__()
        if joint_count < 1 or input_channels < 1:
            raise ValueError("joint_count and input_channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number of at least 3")
        self.joint_count = joint_count
        self.input_channels = input_channels
        self.use_quality_as_feature = use_quality_as_feature
        self.hierarchical_walking_gate = hierarchical_walking_gate
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
        self.walking_classifier = (
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 2),
            )
            if hierarchical_walking_gate
            else None
        )

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("gait TCN input must have shape [B, T, V, C]")
        if inputs.shape[2] != self.joint_count or inputs.shape[3] != self.input_channels:
            raise ValueError(
                f"expected V,C={self.joint_count},{self.input_channels}; "
                f"got {inputs.shape[2]},{inputs.shape[3]}"
            )
        frame_mask = (inputs[..., -1].amax(dim=2) > 0).to(inputs.dtype)
        encoded_inputs = inputs
        if not self.use_quality_as_feature:
            encoded_inputs = torch.cat(
                (
                    inputs[..., :-1],
                    (inputs[..., -1:] > 0).to(inputs.dtype),
                ),
                dim=-1,
            )
        batch_size, frame_count, _, _ = inputs.shape
        encoded = encoded_inputs.reshape(batch_size, frame_count, -1).transpose(1, 2)
        encoded = self.temporal_blocks(self.input_projection(encoded))
        weights = frame_mask.unsqueeze(1)
        return (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    def forward_heads(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = self._encode(inputs)
        gait_logits = self.classifier(pooled)
        walking_logits = (
            self.walking_classifier(pooled)
            if self.walking_classifier is not None
            else None
        )
        return gait_logits, walking_logits

    def predict_probabilities(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        gait_logits, walking_logits = self.forward_heads(inputs)
        conditional = torch.softmax(gait_logits, dim=1)[:, 1]
        walking = (
            torch.softmax(walking_logits, dim=1)[:, 1]
            if walking_logits is not None
            else torch.ones_like(conditional)
        )
        return {
            "conditional_gait_probability": conditional,
            "walking_probability": walking,
            "gait_risk_probability": conditional * walking,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gait_logits, _ = self.forward_heads(inputs)
        return gait_logits


def _augment_gait_window(
    features: torch.Tensor,
    *,
    temporal_shift_frames: int,
    keypoint_dropout_probability: float,
    coordinate_jitter_std: float,
) -> torch.Tensor:
    output = features.clone()
    frame_count = int(output.shape[0])
    base_joint_count = 12
    maximum_shift = min(temporal_shift_frames, max(0, frame_count - 1))
    if maximum_shift:
        shift = int(torch.randint(-maximum_shift, maximum_shift + 1, ()).item())
        shifted = torch.zeros_like(output)
        if shift > 0:
            shifted[shift:] = output[:-shift]
        elif shift < 0:
            shifted[:shift] = output[-shift:]
        else:
            shifted = output
        output = shifted

    base_valid = output[..., :base_joint_count, -1] > 0
    if keypoint_dropout_probability:
        dropped = (
            torch.rand_like(output[..., :base_joint_count, -1])
            < keypoint_dropout_probability
        )
        base_valid = base_valid & ~dropped
    output[..., :base_joint_count, :] *= base_valid.unsqueeze(-1)

    if coordinate_jitter_std:
        noise = (
            torch.randn_like(output[..., :base_joint_count, :2])
            * coordinate_jitter_std
        )
        output[..., :base_joint_count, :2] += noise * base_valid.unsqueeze(-1)

    pelvis_valid = base_valid[..., 6] & base_valid[..., 7]
    base_valid = base_valid & pelvis_valid.unsqueeze(-1)
    output[..., :base_joint_count, :] *= base_valid.unsqueeze(-1)
    pelvis = (
        output[..., 6, :2] + output[..., 7, :2]
    ) / 2.0
    output[..., :base_joint_count, :2] = (
        output[..., :base_joint_count, :2] - pelvis.unsqueeze(-2)
    ) * base_valid.unsqueeze(-1)

    output[..., 12:, :] = 0.0
    pelvis_quality = torch.minimum(output[..., 6, -1], output[..., 7, -1])
    output[..., 12, -1] = torch.where(
        pelvis_valid,
        pelvis_quality,
        torch.zeros_like(pelvis_quality),
    )
    shoulder_valid = base_valid[..., 0] & base_valid[..., 1]
    output[..., 13, :2] = (
        output[..., 0, :2] + output[..., 1, :2]
    ) / 2.0
    shoulder_quality = torch.minimum(output[..., 0, -1], output[..., 1, -1])
    output[..., 13, -1] = torch.where(
        shoulder_valid,
        shoulder_quality,
        torch.zeros_like(shoulder_quality),
    )

    valid = output[..., -1] > 0
    output[..., :4] *= valid.unsqueeze(-1)
    output[..., 2:4] = 0.0
    if frame_count > 1:
        valid_pairs = valid[1:] & valid[:-1]
        differences = output[1:, :, :2] - output[:-1, :, :2]
        deltas = torch.zeros_like(differences)
        deltas[valid_pairs] = differences[valid_pairs]
        output[1:, :, 2:4] = deltas
    return output


class _GaitWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        walking_targets: np.ndarray,
        indices: np.ndarray,
        *,
        augment_mirror: bool,
        temporal_shift_frames: int = 0,
        keypoint_dropout_probability: float = 0.0,
        coordinate_jitter_std: float = 0.0,
    ) -> None:
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
        self.walking_targets = walking_targets
        self.indices = indices.astype(np.int64, copy=False)
        self.augment_mirror = augment_mirror
        self.temporal_shift_frames = temporal_shift_frames
        self.keypoint_dropout_probability = keypoint_dropout_probability
        self.coordinate_jitter_std = coordinate_jitter_std

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
        if (
            self.temporal_shift_frames
            or self.keypoint_dropout_probability
            or self.coordinate_jitter_std
        ):
            features = _augment_gait_window(
                features,
                temporal_shift_frames=self.temporal_shift_frames,
                keypoint_dropout_probability=self.keypoint_dropout_probability,
                coordinate_jitter_std=self.coordinate_jitter_std,
            )
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        sample_weight = torch.tensor(
            float(self.sample_weights[source_index]), dtype=torch.float32
        )
        walking_target = torch.tensor(
            int(self.walking_targets[source_index]), dtype=torch.long
        )
        return features, label, walking_target, sample_weight, source_index


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


def aggregate_group_predictions(
    *,
    group_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    group_field: str = "group_id",
) -> list[dict[str, Any]]:
    if not (len(group_ids) == len(labels) == len(probabilities)):
        raise ValueError("group IDs, labels and probabilities must have equal length")
    grouped: dict[str, dict[str, Any]] = {}
    for group_id, label, probability in zip(
        group_ids, labels, probabilities, strict=True
    ):
        key = str(group_id)
        normalized_label = int(label)
        if key not in grouped:
            grouped[key] = {"label": normalized_label, "probabilities": []}
        elif grouped[key]["label"] != normalized_label:
            raise ValueError(f"group {key} has inconsistent labels")
        grouped[key]["probabilities"].append(float(probability))
    rows: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        item = grouped[group_id]
        probability = float(np.mean(item["probabilities"]))
        rows.append(
            {
                group_field: group_id,
                "label": int(item["label"]),
                "probability": probability,
                "predicted_label": int(probability >= threshold),
                "window_count": len(item["probabilities"]),
            }
        )
    return rows


def select_balanced_accuracy_threshold(
    labels: Sequence[int], scores: Sequence[float]
) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if len(y_true) == 0 or set(y_true.tolist()) != {0, 1}:
        raise ValueError("threshold selection requires both validation classes")
    unique_scores = np.unique(np.clip(y_score, 0.0, 1.0))
    candidates = [0.0, 1.0]
    candidates.extend(
        float((left + right) / 2.0)
        for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=True)
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predictions = (y_score >= threshold).astype(np.int64)
        recall = float(np.mean(predictions[y_true == 1] == 1))
        specificity = float(np.mean(predictions[y_true == 0] == 0))
        balanced_accuracy = (recall + specificity) / 2.0
        candidate = (balanced_accuracy, -abs(threshold - 0.5), -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return float(-best[2])


def compute_balanced_class_weights(
    labels: np.ndarray,
    effective_weights: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Balance classes after segment, tier, and source weighting."""

    if len(labels) != len(effective_weights):
        raise ValueError("labels and effective weights must have equal length")
    selected_labels = labels[train_indices].astype(np.int64, copy=False)
    selected_weights = effective_weights[train_indices].astype(np.float64, copy=False)
    if set(selected_labels.tolist()) != {0, 1}:
        raise ValueError("class balancing requires both training classes")
    if not np.isfinite(selected_weights).all() or np.any(selected_weights <= 0):
        raise ValueError("effective training weights must be finite and positive")
    class_masses = np.asarray(
        [selected_weights[selected_labels == label].sum() for label in (0, 1)],
        dtype=np.float64,
    )
    target_mass = float(class_masses.sum() / 2.0)
    return (target_mass / class_masses).astype(np.float32)


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
    for optional in (
        "sample_weights",
        "action_segment_ids",
        "datasets",
        "source_group_ids",
        "segment_durations_sec",
        "action_ids",
        "walking_targets",
    ):
        if optional in arrays and len(arrays[optional]) != len(features):
            raise ValueError(f"prepared gait {optional} has inconsistent length")
    partitions = {str(value) for value in np.unique(arrays["partitions"])}
    if not {"train", "validation", "test"}.issubset(partitions) or not partitions.issubset(
        {"train", "validation", "test", "excluded"}
    ):
        raise ValueError(f"unexpected prepared partitions: {sorted(partitions)}")

    partition_by_participant: dict[str, set[str]] = defaultdict(set)
    for participant, partition in zip(
        arrays["participant_ids"], arrays["partitions"], strict=True
    ):
        if str(partition) != "excluded":
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
        window_frames: int | None = None,
        expected_task: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if window_frames is not None and window_frames < 2:
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
        input_contract = checkpoint.get("input_contract")
        contract = dict(input_contract) if isinstance(input_contract, Mapping) else {}
        checkpoint_window_frames = int(contract.get("window_frames", window_frames or 64))
        if window_frames is not None and window_frames != checkpoint_window_frames:
            raise ValueError(
                "configured gait window_frames does not match checkpoint input contract"
            )
        self.window_frames = checkpoint_window_frames
        self.target_fps = (
            float(contract["target_fps"]) if contract.get("target_fps") is not None else None
        )
        self.max_gap_sec = float(contract.get("max_gap_sec", 0.5))
        self.min_observed_frames = int(contract.get("min_observed_frames", 1))
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("checkpoint min_observed_frames is outside the gait window")
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
                scores = self.model.predict_probabilities(batch)[
                    "gait_risk_probability"
                ]
                probabilities.append(scores.detach().cpu().numpy())
        if not probabilities:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(probabilities).astype(np.float32, copy=False)

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> float:
        selected: Sequence[Mapping[str, Any] | None]
        source_records = list(records)
        timestamps = [
            float(record["timestamp_sec"])
            for record in source_records
            if record.get("timestamp_sec") is not None
        ]
        if self.target_fps is not None and timestamps:
            start_time = max(timestamps) - ((self.window_frames - 1) / self.target_fps)
            selected = resample_pose_records(
                source_records,
                start_time_sec=start_time,
                window_frames=self.window_frames,
                target_fps=self.target_fps,
                max_gap_sec=self.max_gap_sec,
            )
        else:
            selected = source_records[-self.window_frames :]
        observed_frame_count = sum(record is not None for record in selected)
        if observed_frame_count < self.min_observed_frames:
            raise ValueError(
                "gait window has fewer observed pose frames than the checkpoint contract"
            )
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
    evaluate_test = (
        training.evaluate_test
        if training.evaluate_test is not None
        else source_metadata.get("split_protocol") != "frozen_training_labels_v3"
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_model.pt"
    metrics_path = destination / "metrics.json"
    history_path = destination / "history.jsonl"
    predictions_path = destination / "test_group_predictions.jsonl"
    window_predictions_path = destination / "test_window_predictions.jsonl"
    output_paths = [
        checkpoint_path,
        metrics_path,
        history_path,
    ]
    if evaluate_test:
        output_paths.extend([predictions_path, window_predictions_path])
    for path in output_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"training output already exists: {path}")

    _seed_everything(training.seed)
    device = _select_device(training.device)
    features = arrays["features"].astype(np.float32, copy=False)
    labels = arrays["labels"].astype(np.int64, copy=False)
    partition_schemes = source_metadata.get("partition_schemes", {"frozen": "partitions"})
    if training.partition_scheme not in partition_schemes:
        raise ValueError(
            f"prepared dataset does not provide partition scheme {training.partition_scheme}"
        )
    partition_array_name = str(partition_schemes[training.partition_scheme])
    if partition_array_name not in arrays:
        raise ValueError(f"prepared dataset is missing {partition_array_name}")
    partitions = arrays[partition_array_name].astype(str)
    if source_metadata.get("split_protocol") == "frozen_training_labels_v3":
        validate_frozen_partition_selection(arrays["partitions"], partitions)
    aggregate_participants = source_metadata.get("task") == (
        "retrospective_fall_history_proxy_binary_classification"
    )
    sample_weights = arrays.get(
        "sample_weights", np.ones(len(labels), dtype=np.float32)
    ).astype(np.float32, copy=False)
    use_walking_gate = training.hierarchical_walking_gate and source_metadata.get(
        "task"
    ) == "gait_instability_vs_normal_activity"
    walking_gate_contract = (
        source_metadata.get("walking_gate_contract")
        if use_walking_gate
        else None
    )
    if use_walking_gate and not isinstance(walking_gate_contract, Mapping):
        walking_gate_contract = {
            "walking_action_ids": sorted(GAIT_WALKING_ACTION_IDS),
            "non_walking_policy": "not_applicable_for_conditional_gait_head",
            "final_probability": "p_walking_times_p_abnormal_given_walking",
        }
    walking_targets = np.ones(len(labels), dtype=np.int64)
    if use_walking_gate:
        if "walking_targets" in arrays:
            walking_targets = arrays["walking_targets"].astype(np.int64, copy=False)
        elif "action_ids" in arrays:
            walking_targets = np.asarray(
                [
                    int(str(action_id) in GAIT_WALKING_ACTION_IDS)
                    for action_id in arrays["action_ids"]
                ],
                dtype=np.int64,
            )
        else:
            raise ValueError(
                "hierarchical walking gate requires prepared gait action_ids"
            )
        if not set(np.unique(walking_targets)).issubset({0, 1}):
            raise ValueError("prepared gait walking_targets must be binary")
    aggregate_ids = arrays.get("action_segment_ids", arrays["participant_ids"]).astype(str)
    aggregate_field = (
        "action_segment_id" if "action_segment_ids" in arrays else "participant_id"
    )
    indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation", "test")
    }
    required_partitions = ["train", "validation"] + (["test"] if evaluate_test else [])
    for partition in required_partitions:
        partition_indices = indices[partition]
        if len(partition_indices) == 0:
            raise ValueError(f"prepared gait {partition} partition is empty")
        if set(labels[partition_indices].tolist()) != {0, 1}:
            raise ValueError(f"prepared gait {partition} partition lacks a binary class")
    source_balance_factors: dict[str, float] = {}
    if "datasets" in arrays:
        sample_weights, source_balance_factors = _apply_source_balance(
            sample_weights,
            arrays["datasets"].astype(str),
            indices["train"],
        )

    datasets = {
        partition: _GaitWindowDataset(
            features,
            labels,
            sample_weights,
            walking_targets,
            partition_indices,
            augment_mirror=training.augment_mirror and partition == "train",
            temporal_shift_frames=(
                training.temporal_shift_frames if partition == "train" else 0
            ),
            keypoint_dropout_probability=(
                training.keypoint_dropout_probability if partition == "train" else 0.0
            ),
            coordinate_jitter_std=(
                training.coordinate_jitter_std if partition == "train" else 0.0
            ),
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
        use_quality_as_feature=training.use_quality_as_feature,
        hierarchical_walking_gate=use_walking_gate,
    ).to(device)
    pretraining_transfer: dict[str, Any] | None = None
    if training.pretrained_checkpoint is not None:
        from elderly_monitoring.modules.fall_risk.gait_action_pretraining import (
            transfer_action_pretrained_encoder,
        )

        pretrained_path = Path(training.pretrained_checkpoint)
        pretrained_checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        pretraining_transfer = transfer_action_pretrained_encoder(
            model, pretrained_checkpoint
        )
        pretraining_transfer.update(
            {
                "checkpoint_path": pretrained_path.as_posix(),
                "checkpoint_sha256": _sha256_file(pretrained_path),
            }
        )
    if training.freeze_encoder_epochs:
        _set_encoder_trainable(model, False)
    gait_train_indices = indices["train"]
    if use_walking_gate:
        gait_train_indices = gait_train_indices[
            walking_targets[gait_train_indices] == 1
        ]
    class_weights = compute_balanced_class_weights(
        labels,
        sample_weights,
        gait_train_indices,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction="none",
    )
    walking_criterion: nn.Module | None = None
    walking_class_weights: np.ndarray | None = None
    if use_walking_gate:
        walking_class_weights = compute_balanced_class_weights(
            walking_targets,
            sample_weights,
            indices["train"],
        )
        walking_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                walking_class_weights, dtype=torch.float32, device=device
            ),
            reduction="none",
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
        if epoch == training.freeze_encoder_epochs + 1:
            _set_encoder_trainable(model, True)
        train_loss = _train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            walking_criterion=walking_criterion,
            walking_gate_loss_weight=training.walking_gate_loss_weight,
        )
        validation = _evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            aggregate_ids=aggregate_ids,
            sample_ids=arrays["sample_ids"],
            aggregate_groups=aggregate_participants or "action_segment_ids" in arrays,
            aggregate_field=aggregate_field,
            threshold=0.5,
            walking_criterion=walking_criterion,
            walking_gate_loss_weight=training.walking_gate_loss_weight,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_window_metrics": validation["window_metrics"],
            "validation_group_metrics": validation["group_metrics"],
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
        aggregate_ids=aggregate_ids,
        sample_ids=arrays["sample_ids"],
        aggregate_groups=aggregate_participants or "action_segment_ids" in arrays,
        aggregate_field=aggregate_field,
        threshold=0.5,
        walking_criterion=walking_criterion,
        walking_gate_loss_weight=training.walking_gate_loss_weight,
    )
    threshold_rows = validation["group_predictions"] or validation["window_predictions"]
    selected_threshold = select_balanced_accuracy_threshold(
        [int(row["label"]) for row in threshold_rows],
        [float(row.get("probability", row.get("gait_risk_score"))) for row in threshold_rows],
    )
    validation = _evaluate(
        model,
        loaders["validation"],
        criterion,
        device,
        aggregate_ids=aggregate_ids,
        sample_ids=arrays["sample_ids"],
        aggregate_groups=aggregate_participants or "action_segment_ids" in arrays,
        aggregate_field=aggregate_field,
        threshold=selected_threshold,
        walking_criterion=walking_criterion,
        walking_gate_loss_weight=training.walking_gate_loss_weight,
    )
    test = (
        _evaluate(
            model,
            loaders["test"],
            criterion,
            device,
            aggregate_ids=aggregate_ids,
            sample_ids=arrays["sample_ids"],
            aggregate_groups=aggregate_participants or "action_segment_ids" in arrays,
            aggregate_field=aggregate_field,
            threshold=selected_threshold,
            walking_criterion=walking_criterion,
            walking_gate_loss_weight=training.walking_gate_loss_weight,
        )
        if evaluate_test
        else None
    )
    _attach_group_context(validation, arrays)
    if test is not None:
        _attach_group_context(test, arrays)

    model_config = {
        "joint_count": len(CANONICAL_GAIT_JOINTS),
        "input_channels": len(GAIT_TCN_CHANNELS),
        "hidden_channels": training.hidden_channels,
        "kernel_size": training.kernel_size,
        "dilations": list(training.dilations),
        "dropout": training.dropout,
        "class_count": 2,
        "use_quality_as_feature": training.use_quality_as_feature,
        "hierarchical_walking_gate": use_walking_gate,
    }
    checkpoint = {
        "schema_version": "gait-tcn-checkpoint-v1",
        "state_dict": best_state,
        "model_config": model_config,
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "label_mapping": source_metadata["label_mapping"],
        "target_contract": source_metadata.get("target_contract"),
        "walking_gate_contract": walking_gate_contract,
        "task": source_metadata["task"],
        "threshold": selected_threshold,
        "input_contract": source_metadata.get("preparation_config"),
        "best_epoch": best_epoch,
        "dataset_sha256": source_metadata["dataset_sha256"],
        "training_config": asdict(training),
        "pretraining_transfer": pretraining_transfer,
    }
    _write_torch_atomic(checkpoint_path, checkpoint)
    _write_jsonl_atomic(history_path, history)
    if test is not None:
        _write_jsonl_atomic(predictions_path, test["group_predictions"])
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
        "training_config": asdict(training),
        "class_weights": class_weights.tolist(),
        "walking_class_weights": (
            walking_class_weights.tolist()
            if walking_class_weights is not None
            else None
        ),
        "quality_usage": (
            "classification_feature"
            if training.use_quality_as_feature
            else "mask_and_pooling_only"
        ),
        "hierarchical_walking_gate": use_walking_gate,
        "walking_gate_contract": walking_gate_contract,
        "target_contract": source_metadata.get("target_contract"),
        "source_balance_factors": source_balance_factors,
        "pretraining_transfer": pretraining_transfer,
        "selected_threshold": selected_threshold,
        "test_evaluated": evaluate_test,
        "partition_scheme": training.partition_scheme,
        "validation": {
            "loss": validation["loss"],
            "window_metrics": validation["window_metrics"],
            "group_metrics": validation["group_metrics"],
            "walking_window_metrics": validation["walking_window_metrics"],
            "dataset_metrics": _metrics_by_group_field(
                validation["group_predictions"], "dataset", selected_threshold
            ),
            "normal_false_positives_per_hour": _normal_false_positives_per_hour(
                validation["group_predictions"]
            ),
        },
        "test": (
            {
                "loss": test["loss"],
                "window_metrics": test["window_metrics"],
                "group_metrics": test["group_metrics"],
                "walking_window_metrics": test["walking_window_metrics"],
                "dataset_metrics": _metrics_by_group_field(
                    test["group_predictions"], "dataset", selected_threshold
                ),
                "normal_false_positives_per_hour": _normal_false_positives_per_hour(
                    test["group_predictions"]
                ),
            }
            if test is not None
            else None
        ),
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
        "selected_threshold": selected_threshold,
        "validation_group_metrics": validation["group_metrics"],
        "test_group_metrics": test["group_metrics"] if test is not None else None,
    }


def _train_epoch(
    model: LightweightGaitTCN,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    walking_criterion: nn.Module | None,
    walking_gate_loss_weight: float,
) -> float:
    model.train()
    gait_loss_numerator = 0.0
    gait_loss_denominator = 0.0
    walking_loss_numerator = 0.0
    walking_loss_denominator = 0.0
    for features, labels, walking_targets, sample_weights, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        walking_targets = walking_targets.to(device)
        sample_weights = sample_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        gait_logits, walking_logits = model.forward_heads(features)
        gait_weights = sample_weights
        if walking_criterion is not None:
            gait_weights = gait_weights * (walking_targets == 1)
        gait_losses = criterion(gait_logits, labels)
        gait_numerator, gait_denominator = _weighted_loss_terms(
            gait_losses,
            gait_weights,
            labels,
            criterion.weight,
        )
        loss = gait_numerator / gait_denominator.clamp_min(1e-8)
        walking_numerator: torch.Tensor | None = None
        walking_denominator: torch.Tensor | None = None
        if walking_criterion is not None:
            assert walking_logits is not None
            walking_losses = walking_criterion(walking_logits, walking_targets)
            walking_numerator, walking_denominator = _weighted_loss_terms(
                walking_losses,
                sample_weights,
                walking_targets,
                walking_criterion.weight,
            )
            walking_loss = walking_numerator / walking_denominator.clamp_min(1e-8)
            loss = loss + (walking_gate_loss_weight * walking_loss)
        loss.backward()
        optimizer.step()
        gait_loss_numerator += float(gait_numerator.detach().cpu())
        gait_loss_denominator += float(gait_denominator.detach().cpu())
        if walking_numerator is not None and walking_denominator is not None:
            walking_loss_numerator += float(walking_numerator.detach().cpu())
            walking_loss_denominator += float(walking_denominator.detach().cpu())
    gait_loss = gait_loss_numerator / max(gait_loss_denominator, 1e-8)
    walking_loss = walking_loss_numerator / max(walking_loss_denominator, 1e-8)
    return gait_loss + (walking_gate_loss_weight * walking_loss)


def _weighted_loss_terms(
    losses: torch.Tensor,
    sample_weights: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_weights = sample_weights
    if class_weights is not None:
        effective_weights = effective_weights * class_weights[labels]
    return (losses * sample_weights).sum(), effective_weights.sum()


def _set_encoder_trainable(model: LightweightGaitTCN, trainable: bool) -> None:
    for module in (model.input_projection, model.temporal_blocks):
        for parameter in module.parameters():
            parameter.requires_grad = trainable


def _evaluate(
    model: LightweightGaitTCN,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    aggregate_ids: np.ndarray,
    sample_ids: np.ndarray,
    aggregate_groups: bool,
    aggregate_field: str,
    threshold: float,
    walking_criterion: nn.Module | None,
    walking_gate_loss_weight: float,
) -> dict[str, Any]:
    model.eval()
    gait_loss_numerator = 0.0
    gait_loss_denominator = 0.0
    walking_loss_numerator = 0.0
    walking_loss_denominator = 0.0
    labels_out: list[int] = []
    probabilities: list[float] = []
    conditional_probabilities: list[float] = []
    walking_probabilities: list[float] = []
    walking_labels_out: list[int] = []
    groups: list[str] = []
    samples: list[str] = []
    with torch.inference_mode():
        for features, labels, walking_targets, sample_weights, source_indices in loader:
            features = features.to(device)
            labels_device = labels.to(device)
            walking_targets_device = walking_targets.to(device)
            sample_weights_device = sample_weights.to(device)
            gait_logits, walking_logits = model.forward_heads(features)
            gait_weights = sample_weights_device
            if walking_criterion is not None:
                gait_weights = gait_weights * (walking_targets_device == 1)
            gait_losses = criterion(gait_logits, labels_device)
            gait_numerator, gait_denominator = _weighted_loss_terms(
                gait_losses,
                gait_weights,
                labels_device,
                criterion.weight,
            )
            if walking_criterion is not None:
                assert walking_logits is not None
                walking_losses = walking_criterion(
                    walking_logits, walking_targets_device
                )
                walking_numerator, walking_denominator = _weighted_loss_terms(
                    walking_losses,
                    sample_weights_device,
                    walking_targets_device,
                    walking_criterion.weight,
                )
                walking_loss_numerator += float(walking_numerator.cpu())
                walking_loss_denominator += float(walking_denominator.cpu())
            conditional_probability = torch.softmax(gait_logits, dim=1)[:, 1]
            walking_probability = (
                torch.softmax(walking_logits, dim=1)[:, 1]
                if walking_logits is not None
                else torch.ones_like(conditional_probability)
            )
            gait_risk_probability = conditional_probability * walking_probability
            gait_loss_numerator += float(gait_numerator.cpu())
            gait_loss_denominator += float(gait_denominator.cpu())
            labels_out.extend(int(value) for value in labels.tolist())
            walking_labels_out.extend(int(value) for value in walking_targets.tolist())
            probabilities.extend(
                float(value)
                for value in gait_risk_probability.cpu().tolist()
            )
            conditional_probabilities.extend(
                float(value)
                for value in conditional_probability.cpu().tolist()
            )
            walking_probabilities.extend(
                float(value)
                for value in walking_probability.cpu().tolist()
            )
            groups.extend(
                str(aggregate_ids[int(index)]) for index in source_indices.tolist()
            )
            samples.extend(str(sample_ids[int(index)]) for index in source_indices.tolist())
    group_rows = (
        aggregate_group_predictions(
            group_ids=groups,
            labels=labels_out,
            probabilities=probabilities,
            threshold=threshold,
            group_field=aggregate_field,
        )
        if aggregate_groups
        else []
    )
    window_rows = [
        {
            "sample_id": sample_id,
            "label": label,
            "gait_risk_score": probability,
            "conditional_gait_probability": conditional_probability,
            "walking_probability": walking_probability,
            "predicted_label": int(probability >= threshold),
        }
        for sample_id, label, probability, conditional_probability, walking_probability in zip(
            samples,
            labels_out,
            probabilities,
            conditional_probabilities,
            walking_probabilities,
            strict=True,
        )
    ]
    return {
        "loss": (
            gait_loss_numerator / max(gait_loss_denominator, 1e-8)
            + walking_gate_loss_weight
            * walking_loss_numerator
            / max(walking_loss_denominator, 1e-8)
        ),
        "window_metrics": _binary_metrics(labels_out, probabilities, threshold=threshold),
        "group_metrics": (
            _binary_metrics(
                [row["label"] for row in group_rows],
                [row["probability"] for row in group_rows],
                threshold=threshold,
            )
            if group_rows
            else None
        ),
        "group_predictions": group_rows,
        "window_predictions": window_rows,
        "walking_window_metrics": (
            _binary_metrics(walking_labels_out, walking_probabilities)
            if set(walking_labels_out) == {0, 1}
            else None
        ),
    }


def _binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    y_score = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    y_pred = (y_score >= threshold).astype(np.int64)
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
        "brier_score": float(np.mean((y_score - y_true) ** 2)),
        "log_loss": float(
            -np.mean(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))
        ),
        "ece": _expected_calibration_error(y_true, y_score),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": threshold,
    }


def _attach_group_context(
    evaluation: dict[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    if "action_segment_ids" not in arrays:
        return
    contexts: dict[str, dict[str, Any]] = {}
    datasets = arrays.get("datasets", np.full(len(arrays["labels"]), "unknown"))
    source_groups = arrays.get(
        "source_group_ids", np.full(len(arrays["labels"]), "unknown")
    )
    durations = arrays.get(
        "segment_durations_sec", np.zeros(len(arrays["labels"]), dtype=np.float32)
    )
    for segment_id, dataset, source_group, duration in zip(
        arrays["action_segment_ids"], datasets, source_groups, durations, strict=True
    ):
        key = str(segment_id)
        context = {
            "dataset": str(dataset),
            "source_group_id": str(source_group),
            "duration_sec": float(duration),
        }
        if key in contexts and contexts[key] != context:
            raise ValueError(f"inconsistent context for gait action segment {key}")
        contexts[key] = context
    for row in evaluation["group_predictions"]:
        row.update(contexts[str(row["action_segment_id"])])


def _metrics_by_group_field(
    rows: Sequence[Mapping[str, Any]], field: str, threshold: float
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    values = sorted({str(row.get(field, "unknown")) for row in rows})
    for value in values:
        selected = [row for row in rows if str(row.get(field, "unknown")) == value]
        labels = [int(row["label"]) for row in selected]
        if set(labels) != {0, 1}:
            output[value] = {"sample_count": len(selected), "status": "single_class"}
            continue
        output[value] = _binary_metrics(
            labels,
            [float(row["probability"]) for row in selected],
            threshold=threshold,
        )
    return output


def _normal_false_positives_per_hour(rows: Sequence[Mapping[str, Any]]) -> float | None:
    normal = [row for row in rows if int(row["label"]) == 0]
    duration_sec = sum(float(row.get("duration_sec", 0.0)) for row in normal)
    if duration_sec <= 0:
        return None
    false_positives = sum(int(row["predicted_label"]) == 1 for row in normal)
    return float(false_positives * 3600.0 / duration_sec)


def _expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, *, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & (
            scores <= upper if index == bins - 1 else scores < upper
        )
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(scores[mask]))
        )
    return error


def _apply_source_balance(
    base_weights: np.ndarray,
    datasets: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    output = base_weights.astype(np.float32, copy=True)
    masses = {
        dataset: float(base_weights[train_indices[datasets[train_indices] == dataset]].sum())
        for dataset in sorted(set(datasets[train_indices].tolist()))
    }
    positive_masses = {key: value for key, value in masses.items() if value > 0}
    if not positive_masses:
        return output, {}
    target_mass = sum(positive_masses.values()) / len(positive_masses)
    factors = {key: target_mass / value for key, value in positive_masses.items()}
    for index in train_indices:
        output[index] *= factors[str(datasets[index])]
    return output, factors


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
