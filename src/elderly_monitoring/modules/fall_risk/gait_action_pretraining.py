from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.fall_risk.gait_tensor import build_gait_tensor
from elderly_monitoring.modules.fall_risk.gait_training import (
    _resolve_pose_path,
    _select_labeled_track,
    resample_pose_records,
)
from elderly_monitoring.modules.fall_risk.gait_contract import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
)


ACTION_PRETRAINING_TARGETS = (
    "normal_locomotion",
    "normal_transition_or_activity",
    "functional_impairment",
    "balance_loss",
    "fall_or_post_fall",
)

_TARGET_BY_ACTION_ID = {
    "A01": "normal_locomotion",
    "A02": "normal_locomotion",
    "A03": "normal_transition_or_activity",
    "A04": "normal_transition_or_activity",
    "A05": "normal_transition_or_activity",
    "A06": "normal_transition_or_activity",
    "A07": "normal_transition_or_activity",
    "A08": "normal_transition_or_activity",
    "A09": "normal_transition_or_activity",
    "A10": "normal_locomotion",
    "A11": "normal_transition_or_activity",
    "A12": "normal_transition_or_activity",
    "B01": "functional_impairment",
    "B02": "functional_impairment",
    "B03": "functional_impairment",
    "B04": "functional_impairment",
    "B05": "functional_impairment",
    "B06": "functional_impairment",
    "C01": "functional_impairment",
    "C02": "balance_loss",
    "C03": "balance_loss",
    "C04": "balance_loss",
    "C05": "balance_loss",
    "D01": "fall_or_post_fall",
    "D02": "fall_or_post_fall",
    "D03": "fall_or_post_fall",
    "D04": "fall_or_post_fall",
    "D05": "fall_or_post_fall",
}

_MIRROR_JOINT_INDICES = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13)


@dataclass(frozen=True)
class ActionPretrainingPreparationConfig:
    window_frames: int = 16
    stride_frames: int = 8
    min_observed_frames: int = 10
    target_fps: float = 4.0
    max_gap_sec: float = 0.5
    max_windows_per_segment: int = 2
    auxiliary_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.stride_frames < 1:
            raise ValueError("stride_frames must be positive")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        if self.target_fps <= 0 or self.max_gap_sec <= 0:
            raise ValueError("target_fps and max_gap_sec must be positive")
        if self.max_windows_per_segment < 1:
            raise ValueError("max_windows_per_segment must be positive")
        if not 0 < self.auxiliary_weight <= 1:
            raise ValueError("auxiliary_weight must be within (0, 1]")


@dataclass(frozen=True)
class ActionTCNPretrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    hidden_channels: int = 48
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
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


def build_action_pretraining_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    assignment_rows: Iterable[Mapping[str, Any]],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignments = _index_unique(
        (row for row in assignment_rows if row.get("label_kind") == "action"),
        "label_id",
        "action split assignment",
    )
    manifests = _index_unique(
        (row for row in manifest_rows if row.get("video_id")),
        "video_id",
        "video manifest",
    )
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("schema_version") != "fall-risk-action-label-v3":
            raise ValueError("action pretraining requires v3 action labels")
        action_id = str(row.get("action_id", ""))
        target = _TARGET_BY_ACTION_ID.get(action_id)
        if target is None:
            continue
        if row.get("training_tier") == "ignore" or row.get(
            "action_type_training_tier"
        ) == "ignore":
            continue
        label_id = _required_string(row, "label_id")
        assignment = assignments.get(label_id)
        if assignment is None:
            raise ValueError(f"missing frozen split assignment for action label {label_id}")
        _validate_assignment(row, assignment)
        if assignment.get("partition") == "test":
            continue
        video_id = _required_string(row, "video_id")
        manifest = manifests.get(video_id)
        if manifest is None:
            raise ValueError(f"missing manifest video for action label {label_id}")
        if manifest.get("eligibility") is not True:
            raise ValueError(f"ineligible manifest video for action label {label_id}")
        if str(manifest.get("asset_id", "")) != str(row.get("asset_id", "")):
            raise ValueError(f"manifest asset mismatch for action label {label_id}")
        end_frame_exclusive = row.get("end_frame_exclusive")
        start_frame = row.get("start_frame")
        if (
            not isinstance(start_frame, int)
            or not isinstance(end_frame_exclusive, int)
            or end_frame_exclusive <= start_frame
        ):
            raise ValueError(f"invalid frame bounds for action label {label_id}")
        partition = str(assignment["partition"])
        effective_tier = (
            "auxiliary"
            if "auxiliary"
            in {
                str(row.get("training_tier")),
                str(row.get("action_type_training_tier")),
            }
            else "primary"
        )
        if partition == "validation" and effective_tier != "primary":
            partition = "excluded"
        row.update(
            {
                "end_frame": end_frame_exclusive - 1,
                "dataset": str(manifest.get("dataset", "unknown")),
                "scene": str(manifest.get("scene_region", "unknown")),
                "split_group_id": str(assignment["split_group_id"]),
                "partition": partition,
                "effective_training_tier": effective_tier,
                "pretraining_target": target,
                "pretraining_label": ACTION_PRETRAINING_TARGETS.index(target),
            }
        )
        output.append(row)
    output.sort(
        key=lambda row: (
            str(row["video_id"]),
            int(row["start_frame"]),
            str(row["label_id"]),
        )
    )
    return output


def prepare_action_pretraining_dataset(
    labels_path: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    assignments_path: str | Path,
    split_report_path: str | Path,
    config: ActionPretrainingPreparationConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    preparation = config or ActionPretrainingPreparationConfig()
    label_source = Path(labels_path)
    manifest_source = Path(manifest_path)
    assignments_source = Path(assignments_path)
    split_report_source = Path(split_report_path)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    split_report = _validate_split_report(
        label_source,
        manifest_source,
        assignments_source,
        split_report_source,
    )
    rows = build_action_pretraining_rows(
        _read_jsonl(label_source),
        assignment_rows=_read_jsonl(assignments_source),
        manifest_rows=_read_jsonl(manifest_source),
    )
    if not rows:
        raise ValueError("no eligible non-test action labels were selected")

    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_video[str(row["video_id"])].append(row)
    samples: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    pose_inputs: dict[str, dict[str, str]] = {}
    rejected: Counter[str] = Counter()
    for video_id in sorted(rows_by_video):
        pose_path = _resolve_pose_path(pose_root, video_id)
        if pose_path is None:
            raise FileNotFoundError(f"pose-quality JSONL not found for {video_id}")
        pose_inputs[video_id] = {
            "path": pose_path.as_posix(),
            "sha256": _sha256_file(pose_path),
        }
        pose_records = _read_jsonl(pose_path)
        for row in rows_by_video[video_id]:
            selected_track = _select_labeled_track(row, pose_records)
            if not selected_track:
                rejected["no_matching_pose_track"] += 1
                continue
            start_time = float(row["start_time"])
            end_time = float(row["end_time_exclusive"])
            segment_frames = max(
                1, int(np.ceil(max(0.0, end_time - start_time) * preparation.target_fps))
            )
            slots = resample_pose_records(
                list(selected_track.values()),
                start_time_sec=start_time,
                window_frames=segment_frames,
                target_fps=preparation.target_fps,
                max_gap_sec=preparation.max_gap_sec,
            )
            starts = _limited_window_starts(
                len(slots),
                window_frames=preparation.window_frames,
                stride_frames=preparation.stride_frames,
                maximum=preparation.max_windows_per_segment,
            )
            accepted: list[tuple[int, np.ndarray, int]] = []
            for window_start in starts:
                window = slots[window_start : window_start + preparation.window_frames]
                observed = sum(record is not None for record in window)
                if observed < preparation.min_observed_frames:
                    rejected["insufficient_observed_frames"] += 1
                    continue
                tensor = build_gait_tensor(window, window_frames=preparation.window_frames)
                if not np.any(tensor[..., -1] > 0):
                    rejected["empty_quality_mask"] += 1
                    continue
                accepted.append((window_start, tensor, observed))
            if not accepted:
                rejected["segments_without_usable_window"] += 1
                continue
            tier_weight = (
                preparation.auxiliary_weight
                if row["effective_training_tier"] == "auxiliary"
                else 1.0
            )
            segment_weight = tier_weight / len(accepted)
            for window_index, (window_start, tensor, observed) in enumerate(accepted):
                tensors.append(tensor)
                samples.append(
                    {
                        "sample_index": len(samples),
                        "sample_id": f"{row['label_id']}:window-{window_index:03d}",
                        "label_id": str(row["label_id"]),
                        "video_id": video_id,
                        "split_group_id": str(row["split_group_id"]),
                        "source_group_id": str(row["source_group_id"]),
                        "sample_group_id": str(row["sample_group_id"]),
                        "dataset": str(row["dataset"]),
                        "action_id": str(row["action_id"]),
                        "action_family": str(row.get("action_family", "unknown")),
                        "action_type": str(row.get("action_type", "unknown")),
                        "pretraining_target": str(row["pretraining_target"]),
                        "pretraining_label": int(row["pretraining_label"]),
                        "training_tier": str(row["effective_training_tier"]),
                        "partition": str(row["partition"]),
                        "sample_weight": segment_weight,
                        "window_start_time_sec": start_time
                        + window_start / preparation.target_fps,
                        "observed_frame_count": observed,
                    }
                )
        del pose_records
    if not samples:
        raise ValueError("no usable action pretraining windows were generated")

    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    samples_path = destination / "samples.jsonl"
    for path in (dataset_path, metadata_path, samples_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"action pretraining output already exists: {path}")
    arrays = {
        "features": np.stack(tensors).astype(np.float32),
        "labels": np.asarray([sample["pretraining_label"] for sample in samples], dtype=np.int64),
        "sample_ids": np.asarray([sample["sample_id"] for sample in samples]),
        "action_segment_ids": np.asarray([sample["label_id"] for sample in samples]),
        "video_ids": np.asarray([sample["video_id"] for sample in samples]),
        "split_group_ids": np.asarray([sample["split_group_id"] for sample in samples]),
        "source_group_ids": np.asarray([sample["source_group_id"] for sample in samples]),
        "sample_group_ids": np.asarray([sample["sample_group_id"] for sample in samples]),
        "datasets": np.asarray([sample["dataset"] for sample in samples]),
        "action_ids": np.asarray([sample["action_id"] for sample in samples]),
        "target_names": np.asarray([sample["pretraining_target"] for sample in samples]),
        "training_tiers": np.asarray([sample["training_tier"] for sample in samples]),
        "partitions": np.asarray([sample["partition"] for sample in samples]),
        "sample_weights": np.asarray([sample["sample_weight"] for sample in samples], dtype=np.float32),
    }
    _validate_no_partition_leakage(arrays)
    _write_npz_atomic(dataset_path, **arrays)
    _write_jsonl_atomic(samples_path, samples)
    partition_counts = Counter(str(value) for value in arrays["partitions"])
    action_counts = Counter(str(value) for value in arrays["action_ids"])
    target_counts = Counter(str(value) for value in arrays["target_names"])
    metadata = {
        "schema_version": "fall-risk-action-window-dataset-v1",
        "task": "fall_risk_action_semantic_pretraining",
        "training_ready": False,
        "training_ready_reason": "source v3 split reports training_ready.action_type=false",
        "test_labels_used": 0,
        "target_mapping": {
            name: index for index, name in enumerate(ACTION_PRETRAINING_TARGETS)
        },
        "action_target_mapping": dict(sorted(_TARGET_BY_ACTION_ID.items())),
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "preparation_config": asdict(preparation),
        "eligible_label_count": len(rows),
        "used_action_segment_count": len({sample["label_id"] for sample in samples}),
        "sample_count": len(samples),
        "partition_counts": dict(sorted(partition_counts.items())),
        "action_window_counts": dict(sorted(action_counts.items())),
        "target_window_counts": dict(sorted(target_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "split_protocol": "frozen_training_labels_v3_non_test_only",
        "source_split_id": split_report.get("split_id"),
        "labels_path": label_source.as_posix(),
        "labels_sha256": _sha256_file(label_source),
        "manifest_path": manifest_source.as_posix(),
        "manifest_sha256": _sha256_file(manifest_source),
        "assignments_path": assignments_source.as_posix(),
        "assignments_sha256": _sha256_file(assignments_source),
        "split_report_path": split_report_source.as_posix(),
        "split_report_sha256": _sha256_file(split_report_source),
        "pose_inputs": pose_inputs,
        "samples_sha256": _sha256_file(samples_path),
        "limitations": [
            "locked test assignments are excluded from representation learning",
            "rare actions are grouped into semantic targets instead of treated as stable standalone classes",
            "the source action_type readiness gate is false, so results are development evidence only",
        ],
    }
    metadata["dataset_sha256"] = _sha256_file(dataset_path)
    _write_json_atomic(metadata_path, metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "eligible_label_count": len(rows),
        "used_action_segment_count": metadata["used_action_segment_count"],
        "sample_count": len(samples),
        "partition_counts": metadata["partition_counts"],
        "rejected_counts": metadata["rejected_counts"],
    }


class _ActionWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        *,
        augment_mirror: bool,
    ) -> None:
        self.features = features
        self.labels = labels
        self.weights = weights
        self.indices = indices.astype(np.int64, copy=False)
        self.augment_mirror = augment_mirror

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        source_index = int(self.indices[item])
        features = torch.from_numpy(self.features[source_index]).clone()
        if self.augment_mirror and bool(torch.rand(()) < 0.5):
            features = features[:, _MIRROR_JOINT_INDICES, :].clone()
            features[..., 0] *= -1.0
            features[..., 2] *= -1.0
        return (
            features,
            torch.tensor(int(self.labels[source_index]), dtype=torch.long),
            torch.tensor(float(self.weights[source_index]), dtype=torch.float32),
            source_index,
        )


def train_action_pretraining_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: ActionTCNPretrainingConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    from elderly_monitoring.modules.fall_risk.gait_tcn import (
        LightweightGaitTCN,
        _select_device,
    )

    training = config or ActionTCNPretrainingConfig()
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path) if metadata_path else source_path.with_name("metadata.json")
    )
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("schema_version") != "fall-risk-action-window-dataset-v1":
        raise ValueError("unsupported action pretraining dataset schema")
    if source_metadata.get("dataset_sha256") != _sha256_file(source_path):
        raise ValueError("action pretraining dataset SHA-256 mismatch")
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_pretraining_arrays(arrays)
    features = arrays["features"].astype(np.float32, copy=False)
    labels = arrays["labels"].astype(np.int64, copy=False)
    partitions = arrays["partitions"].astype(str)
    weights = arrays["sample_weights"].astype(np.float32, copy=True)
    indices = {
        name: np.flatnonzero(partitions == name) for name in ("train", "validation")
    }
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError("action pretraining requires non-empty train and validation data")
    train_classes = set(labels[indices["train"]].tolist())
    validation_classes = set(labels[indices["validation"]].tolist())
    if not validation_classes.issubset(train_classes):
        missing = sorted(validation_classes - train_classes)
        raise ValueError(f"validation contains classes absent from training: {missing}")
    weights, source_balance = _apply_source_balance(
        weights, arrays["datasets"].astype(str), indices["train"]
    )
    class_weights = _multiclass_weights(
        labels, weights, indices["train"], len(ACTION_PRETRAINING_TARGETS)
    )
    _seed_everything(training.seed)
    device = _select_device(training.device)
    datasets = {
        name: _ActionWindowDataset(
            features,
            labels,
            weights,
            index,
            augment_mirror=training.augment_mirror and name == "train",
        )
        for name, index in indices.items()
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
        "joint_count": len(CANONICAL_GAIT_JOINTS),
        "input_channels": len(GAIT_TCN_CHANNELS),
        "hidden_channels": training.hidden_channels,
        "kernel_size": training.kernel_size,
        "dilations": list(training.dilations),
        "dropout": training.dropout,
        "class_count": len(ACTION_PRETRAINING_TARGETS),
        "use_quality_as_feature": False,
    }
    model = LightweightGaitTCN(**model_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, training.epochs + 1):
        train_loss = _train_epoch(model, loaders["train"], criterion, optimizer, device)
        validation = _evaluate_multiclass(
            model,
            loaders["validation"],
            criterion,
            device,
            labels=labels,
            segment_ids=arrays["action_segment_ids"].astype(str),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_metrics": validation["metrics"],
            }
        )
        if validation["loss"] < best_validation_loss - 1e-8:
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= training.patience:
            break
    if best_state is None:
        raise RuntimeError("action pretraining did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation = _evaluate_multiclass(
        model,
        loaders["validation"],
        criterion,
        device,
        labels=labels,
        segment_ids=arrays["action_segment_ids"].astype(str),
    )
    encoder_state = {
        name: value
        for name, value in best_state.items()
        if name.startswith(("input_projection.", "temporal_blocks."))
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "encoder.pt"
    metrics_path = destination / "metrics.json"
    history_path = destination / "history.jsonl"
    for path in (checkpoint_path, metrics_path, history_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"action pretraining output already exists: {path}")
    checkpoint = {
        "schema_version": "fall-risk-action-tcn-pretraining-v1",
        "task": source_metadata["task"],
        "encoder_state_dict": encoder_state,
        "model_config": model_config,
        "target_mapping": source_metadata["target_mapping"],
        "action_target_mapping": source_metadata["action_target_mapping"],
        "input_contract": source_metadata["preparation_config"],
        "dataset_sha256": source_metadata["dataset_sha256"],
        "best_epoch": best_epoch,
        "training_config": asdict(training),
    }
    _write_torch_atomic(checkpoint_path, checkpoint)
    _write_jsonl_atomic(history_path, history)
    metrics = {
        "schema_version": "fall-risk-action-tcn-pretraining-report-v1",
        "task": source_metadata["task"],
        "training_ready": False,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "device": str(device),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "class_weights": class_weights.tolist(),
        "source_balance_factors": source_balance,
        "validation": validation,
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
        "device": str(device),
        "validation_metrics": validation["metrics"],
    }


def transfer_action_pretrained_encoder(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    if checkpoint.get("schema_version") != "fall-risk-action-tcn-pretraining-v1":
        raise ValueError("unsupported action pretraining checkpoint schema")
    source = checkpoint.get("encoder_state_dict")
    if not isinstance(source, Mapping):
        raise ValueError("action pretraining checkpoint is missing encoder_state_dict")
    expected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(("input_projection.", "temporal_blocks."))
    }
    if set(source) != set(expected):
        raise ValueError("incompatible pretrained encoder parameter names")
    copied: dict[str, torch.Tensor] = {}
    for name, target_value in expected.items():
        source_value = source[name]
        if not isinstance(source_value, torch.Tensor) or source_value.shape != target_value.shape:
            raise ValueError(f"incompatible pretrained encoder tensor: {name}")
        copied[name] = source_value.detach().to(dtype=target_value.dtype)
    result = model.load_state_dict(copied, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [
        name
        for name in result.missing_keys
        if not name.startswith(("classifier.", "walking_classifier."))
    ]
    if unexpected or missing:
        raise ValueError("incompatible pretrained encoder state")
    return {
        "transferred_parameter_count": sum(value.numel() for value in copied.values()),
        "source_task": checkpoint.get("task"),
        "source_dataset_sha256": checkpoint.get("dataset_sha256"),
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
    total_weight = 0.0
    for features, labels, sample_weights, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(features), labels)
        loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()
        total_loss += float((losses.detach() * sample_weights).sum().cpu())
        total_weight += float(sample_weights.sum().cpu())
    return total_loss / max(total_weight, 1e-8)


def _evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    labels: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for features, batch_labels, sample_weights, source_indices in loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)
            sample_weights = sample_weights.to(device)
            logits = model(features)
            losses = criterion(logits, batch_labels)
            total_loss += float((losses * sample_weights).sum().cpu())
            total_weight += float(sample_weights.sum().cpu())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            indices.append(source_indices.numpy())
    selected_indices = np.concatenate(indices)
    selected_probabilities = np.concatenate(probabilities)
    grouped: dict[str, dict[str, Any]] = {}
    for source_index, probability in zip(
        selected_indices, selected_probabilities, strict=True
    ):
        segment_id = str(segment_ids[source_index])
        grouped.setdefault(
            segment_id,
            {"label": int(labels[source_index]), "probabilities": []},
        )["probabilities"].append(probability)
    grouped_labels = np.asarray(
        [grouped[key]["label"] for key in sorted(grouped)], dtype=np.int64
    )
    grouped_probabilities = np.stack(
        [np.mean(grouped[key]["probabilities"], axis=0) for key in sorted(grouped)]
    )
    predictions = np.argmax(grouped_probabilities, axis=1)
    metrics = _multiclass_metrics(grouped_labels, predictions)
    return {
        "loss": total_loss / max(total_weight, 1e-8),
        "segment_count": len(grouped_labels),
        "metrics": metrics,
    }


def _multiclass_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    classes = sorted(set(labels.tolist()))
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in classes:
        true_positive = int(np.sum((labels == class_id) & (predictions == class_id)))
        false_negative = int(np.sum((labels == class_id) & (predictions != class_id)))
        false_positive = int(np.sum((labels != class_id) & (predictions == class_id)))
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[ACTION_PRETRAINING_TARGETS[class_id]] = {
            "support": true_positive + false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }


def _multiclass_weights(
    labels: np.ndarray,
    effective_weights: np.ndarray,
    train_indices: np.ndarray,
    class_count: int,
) -> np.ndarray:
    masses = np.zeros(class_count, dtype=np.float64)
    for class_id in range(class_count):
        mask = labels[train_indices] == class_id
        masses[class_id] = effective_weights[train_indices][mask].sum()
    present = masses > 0
    target_mass = float(masses[present].sum() / max(1, int(present.sum())))
    output = np.zeros(class_count, dtype=np.float32)
    output[present] = (target_mass / masses[present]).astype(np.float32)
    return output


def _apply_source_balance(
    weights: np.ndarray, datasets: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    output = weights.copy()
    masses = {
        dataset: float(output[train_indices][datasets[train_indices] == dataset].sum())
        for dataset in sorted(set(datasets[train_indices].tolist()))
    }
    target = sum(masses.values()) / max(1, len(masses))
    factors = {
        dataset: target / mass for dataset, mass in masses.items() if mass > 0
    }
    for index in train_indices:
        output[index] *= factors[str(datasets[index])]
    return output, factors


def _limited_window_starts(
    frame_count: int, *, window_frames: int, stride_frames: int, maximum: int
) -> list[int]:
    if frame_count <= window_frames:
        return [0]
    final_start = frame_count - window_frames
    starts = list(range(0, final_start + 1, stride_frames))
    if starts[-1] != final_start:
        starts.append(final_start)
    if len(starts) <= maximum:
        return starts
    indices = np.linspace(0, len(starts) - 1, num=maximum, dtype=np.int64)
    return [starts[index] for index in sorted(set(indices.tolist()))]


def _validate_pretraining_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "features",
        "labels",
        "partitions",
        "sample_weights",
        "datasets",
        "action_segment_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"action pretraining dataset is missing arrays: {missing}")
    features = arrays["features"]
    if features.ndim != 4 or features.shape[2:] != (
        len(CANONICAL_GAIT_JOINTS),
        len(GAIT_TCN_CHANNELS),
    ):
        raise ValueError(f"unexpected action pretraining feature shape: {features.shape}")
    if any(len(value) != len(features) for value in arrays.values()):
        raise ValueError("action pretraining arrays have inconsistent lengths")
    if not np.isfinite(features).all():
        raise ValueError("action pretraining features contain non-finite values")


def _validate_no_partition_leakage(arrays: Mapping[str, np.ndarray]) -> None:
    for field in ("split_group_ids", "source_group_ids", "sample_group_ids"):
        partitions_by_group: dict[str, set[str]] = defaultdict(set)
        for group, partition in zip(arrays[field], arrays["partitions"], strict=True):
            if str(partition) != "excluded":
                partitions_by_group[str(group)].add(str(partition))
        leaked = [group for group, values in partitions_by_group.items() if len(values) > 1]
        if leaked:
            raise ValueError(f"action pretraining {field} cross partitions: {leaked[:10]}")
    if "test" in {str(value) for value in arrays["partitions"]}:
        raise ValueError("locked test data entered action pretraining dataset")


def _validate_assignment(label: Mapping[str, Any], assignment: Mapping[str, Any]) -> None:
    label_id = str(label["label_id"])
    for field in (
        "asset_id",
        "video_id",
        "source_group_id",
        "sample_group_id",
        "training_tier",
    ):
        if str(label.get(field, "")) != str(assignment.get(field, "")):
            raise ValueError(f"frozen split {field} mismatch for action label {label_id}")
    if assignment.get("partition") not in {"train", "validation", "test"}:
        raise ValueError(f"invalid frozen partition for action label {label_id}")
    if not str(assignment.get("split_group_id", "")):
        raise ValueError(f"missing split_group_id for action label {label_id}")


def _validate_split_report(
    labels_path: Path,
    manifest_path: Path,
    assignments_path: Path,
    split_report_path: Path,
) -> dict[str, Any]:
    report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "fall-risk-training-split-v3":
        raise ValueError("unsupported frozen action split report schema")
    if report.get("leakage_issues"):
        raise ValueError("frozen action split report contains leakage issues")
    hashes = report.get("input_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("frozen action split report is missing input hashes")
    if hashes.get("action_labels") != _sha256_file(labels_path):
        raise ValueError("frozen action split action label SHA-256 mismatch")
    if hashes.get("manifest") != _sha256_file(manifest_path):
        raise ValueError("frozen action split manifest SHA-256 mismatch")
    if report.get("assignments_sha256") != _sha256_file(assignments_path):
        raise ValueError("frozen action split assignments SHA-256 mismatch")
    return dict(report)


def _index_unique(
    rows: Iterable[Mapping[str, Any]], field: str, description: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key:
            continue
        if key in output:
            raise ValueError(f"duplicate {description} {field}: {key}")
        output[key] = dict(row)
    return output


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"action label is missing {field}")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("wb") as file:
        np.savez_compressed(file, **arrays)
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


def _write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    torch.save(dict(payload), partial)
    os.replace(partial, path)
