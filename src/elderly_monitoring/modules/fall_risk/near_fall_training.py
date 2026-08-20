from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NEAR_FALL_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NEAR_FALL_CHANNELS = (
    "x",
    "y",
    "dx",
    "dy",
    "ddx",
    "ddy",
    "quality",
    "valid_mask",
)
NEAR_FALL_HARD_NEGATIVES = (
    "normal_turn",
    "normal_step_adjustment",
    "routine_support_contact",
    "fast_but_controlled_sit",
    "controlled_squat",
    "controlled_bend",
    "exercise_or_stretch",
    "progressed_to_fall",
)
_ALLOWED_NEGATIVES = {*NEAR_FALL_HARD_NEGATIVES, "background"}
_DEVELOPMENT_PARTITIONS = {"train", "validation"}
_ALL_PARTITIONS = {"train", "validation", "test"}
_SEVERE_QUALITY_FLAGS = {
    "heavy_occlusion",
    "off_screen",
    "multi_person_uncertain",
    "camera_cut",
}
_REVIEWED_STATUSES = {"double_reviewed", "adjudicated"}
_NEGATIVE_REVIEWED_STATUSES = {"single_reviewed", *_REVIEWED_STATUSES}
_PRIMARY_TARGET_STATUSES = {"confirmed", "single_person_assumed"}


@dataclass(frozen=True)
class NearFallDatasetConfig:
    target_fps: float = 8.0
    window_sec: float = 3.0
    stride_sec: float = 0.5
    max_gap_sec: float = 0.20
    min_observed_frames: int = 12
    min_usable_frame_ratio: float = 0.60
    min_joint_coverage: float = 0.70
    min_mean_joint_quality: float = 0.30
    max_windows_per_event: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_fps <= 0 or self.window_sec <= 0 or self.stride_sec <= 0:
            raise ValueError("near-fall timing values must be positive")
        if self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.window_frames < 3:
            raise ValueError("near-fall window must contain at least 3 frames")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        for name, value in (
            ("min_usable_frame_ratio", self.min_usable_frame_ratio),
            ("min_joint_coverage", self.min_joint_coverage),
            ("min_mean_joint_quality", self.min_mean_joint_quality),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be within (0, 1]")
        if self.max_windows_per_event < 1:
            raise ValueError("max_windows_per_event must be positive")

    @property
    def window_frames(self) -> int:
        return int(round(self.target_fps * self.window_sec))


def build_near_fall_tensor(
    records: Sequence[Mapping[str, Any] | None],
    *,
    window_frames: int,
    target_fps: float,
) -> np.ndarray:
    """Build causal body-normalized pose features with shape [T,10,8]."""

    if window_frames < 3:
        raise ValueError("window_frames must be at least 3")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if len(records) > window_frames:
        raise ValueError("records exceed window_frames")

    joint_count = len(NEAR_FALL_JOINTS)
    xy = np.zeros((window_frames, joint_count, 2), dtype=np.float32)
    quality = np.zeros((window_frames, joint_count), dtype=np.float32)
    valid = np.zeros((window_frames, joint_count), dtype=np.float32)
    for frame_index, record in enumerate(records):
        if record is None:
            continue
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        for joint_index, name in enumerate(NEAR_FALL_JOINTS):
            point = points.get(name)
            if (
                point is None
                or point.get("valid") is not True
                or point.get("is_jump_outlier") is True
            ):
                continue
            x = _optional_float(point.get("x_smooth", point.get("x")))
            y = _optional_float(point.get("y_smooth", point.get("y")))
            point_quality = _optional_float(
                point.get("quality_weight", point.get("score"))
            )
            if x is None or y is None or point_quality is None or point_quality <= 0:
                continue
            xy[frame_index, joint_index] = (x, y)
            quality[frame_index, joint_index] = min(1.0, max(0.0, point_quality))
            valid[frame_index, joint_index] = 1.0

    hip_valid = (valid[:, 4] > 0) & (valid[:, 5] > 0)
    shoulder_valid = (valid[:, 0] > 0) & (valid[:, 1] > 0)
    hip_centers = (xy[:, 4] + xy[:, 5]) / 2.0
    shoulder_centers = (xy[:, 0] + xy[:, 1]) / 2.0
    torso_lengths = np.linalg.norm(shoulder_centers - hip_centers, axis=1)
    scale_valid = (
        hip_valid
        & shoulder_valid
        & np.isfinite(torso_lengths)
        & (torso_lengths > 1e-6)
    )
    tensor = np.zeros(
        (window_frames, joint_count, len(NEAR_FALL_CHANNELS)), dtype=np.float32
    )
    if not scale_valid.any():
        return tensor
    body_scale = float(np.median(torso_lengths[scale_valid]))

    centered = np.zeros_like(xy)
    centered[hip_valid] = (
        xy[hip_valid] - hip_centers[hip_valid, None, :]
    ) / body_scale
    valid[~hip_valid] = 0.0
    quality[~hip_valid] = 0.0
    centered[valid <= 0] = 0.0

    velocity = np.zeros_like(centered)
    velocity_valid = np.zeros_like(valid, dtype=bool)
    velocity_valid[1:] = (valid[1:] > 0) & (valid[:-1] > 0)
    velocity[1:] = (centered[1:] - centered[:-1]) * float(target_fps)
    velocity[~velocity_valid] = 0.0

    acceleration = np.zeros_like(centered)
    acceleration_valid = np.zeros_like(valid, dtype=bool)
    acceleration_valid[2:] = velocity_valid[2:] & velocity_valid[1:-1]
    acceleration[2:] = (velocity[2:] - velocity[1:-1]) * float(target_fps)
    acceleration[~acceleration_valid] = 0.0

    tensor[..., 0:2] = centered
    tensor[..., 2:4] = velocity
    tensor[..., 4:6] = acceleration
    tensor[..., 6] = quality
    tensor[..., 7] = valid
    tensor[valid <= 0, :7] = 0.0
    if not np.isfinite(tensor).all():
        raise ValueError("near-fall tensor contains non-finite values")
    return tensor


def select_near_fall_training_labels(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select only explicit, reviewed v3 near-fall binary supervision."""

    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("schema_version") != "fall-risk-event-label-v3":
            continue
        if row.get("task_type") != "near_fall_event":
            continue
        if row.get("training_tier") != "primary":
            continue
        if row.get("label_role") not in {"positive", "negative"}:
            continue
        if row.get("target_status") not in _PRIMARY_TARGET_STATUSES:
            continue
        quality_flags = set(row.get("quality_flags") or [])
        if quality_flags & _SEVERE_QUALITY_FLAGS:
            continue
        label_id = str(row.get("label_id", "unknown"))
        start = row.get("start_frame")
        end = row.get("end_frame_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            raise ValueError(f"near-fall label has invalid half-open bounds: {label_id}")

        if row["label_role"] == "positive":
            if not str(row.get("physical_event_id") or "").strip():
                raise ValueError(
                    f"near-fall positive lacks physical_event_id: {label_id}"
                )
            if row.get("review_status") not in _REVIEWED_STATUSES:
                raise ValueError(
                    f"near-fall positive is not double reviewed: {label_id}"
                )
            onset = row.get("onset_frame")
            recovery = row.get("recovery_frame")
            if (
                row.get("event_type") != "near_fall"
                or row.get("event_outcome") != "recovered_without_fall"
                or not isinstance(onset, int)
                or not isinstance(recovery, int)
                or not start <= onset <= recovery < end
                or row.get("impact_frame") is not None
            ):
                raise ValueError(
                    f"invalid recovery-confirmed near-fall positive label: {label_id}"
                )
            row["label"] = 1
            row["target_name"] = "near_fall_recovered_without_fall"
        else:
            if row.get("physical_event_id") is not None:
                raise ValueError(
                    f"near-fall negative has physical_event_id: {label_id}"
                )
            if row.get("review_status") not in _NEGATIVE_REVIEWED_STATUSES:
                raise ValueError(f"near-fall negative is not reviewed: {label_id}")
            if not any(
                isinstance(reference, Mapping)
                and reference.get("source_type") == "manual_v3"
                for reference in row.get("source_refs") or []
            ):
                raise ValueError(
                    f"near-fall negative lacks manual_v3 confirmation: {label_id}"
                )
            hard_negative = row.get("hard_negative_type")
            if hard_negative not in _ALLOWED_NEGATIVES:
                raise ValueError(
                    f"near-fall negative lacks an explicit allowed type: {label_id}"
                )
            if row.get("event_type") is not None or row.get("recovery_frame") is not None:
                raise ValueError(f"near-fall negative has positive event fields: {label_id}")
            row["label"] = 0
            row["target_name"] = "explicit_near_fall_negative"
        selected.append(row)
    return selected


def require_near_fall_training_ready(
    validation_report: Mapping[str, Any], split_report: Mapping[str, Any]
) -> None:
    if validation_report.get("training_ready", {}).get("near_fall_event") is True:
        return
    counts = validation_report.get("counts", {})
    positive = int(counts.get("primary_near_fall_positive", 0) or 0)
    negative = int(counts.get("primary_near_fall_negative", 0) or 0)
    missing = validation_report.get("hard_negative_coverage", {}).get(
        "near_fall_event", {}
    ).get("missing", list(NEAR_FALL_HARD_NEGATIVES))
    split_counts = split_report.get("partition_supervision_counts", {}).get(
        "near_fall_event", {}
    )
    partition_summary = ", ".join(
        f"{partition}:positive={int(split_counts.get(partition, {}).get('primary_positive', 0) or 0)} "
        f"negative={int(split_counts.get(partition, {}).get('primary_negative', 0) or 0)}"
        for partition in ("train", "validation", "test")
    )
    raise ValueError(
        "near-fall training data gate blocked: "
        "training_ready.near_fall_event=false; "
        f"primary positive={positive} negative={negative}; "
        f"partitions [{partition_summary}]; "
        f"missing hard negatives={list(missing)}"
    )


def fit_normalization_statistics(
    features: np.ndarray, partitions: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    partition_values = np.asarray(partitions).astype(str)
    if values.ndim != 4 or values.shape[-2:] != (
        len(NEAR_FALL_JOINTS),
        len(NEAR_FALL_CHANNELS),
    ):
        raise ValueError("near-fall normalization expects [N,T,10,8]")
    if len(values) != len(partition_values):
        raise ValueError("normalization partition length mismatch")
    train = values[partition_values == "train"]
    if len(train) == 0:
        raise ValueError("normalization requires train samples")
    valid = train[..., 7] > 0
    means = np.zeros(len(NEAR_FALL_CHANNELS), dtype=np.float32)
    stds = np.ones(len(NEAR_FALL_CHANNELS), dtype=np.float32)
    for channel in range(7):
        channel_values = train[..., channel][valid]
        if len(channel_values) == 0:
            raise ValueError(f"train has no valid values for channel {channel}")
        means[channel] = float(np.mean(channel_values, dtype=np.float64))
        std = float(np.std(channel_values, dtype=np.float64))
        stds[channel] = std if std > 1e-6 else 1.0
    return {"mean": means, "std": stds}


def apply_normalization(
    features: np.ndarray, statistics: Mapping[str, np.ndarray]
) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    means = np.asarray(statistics["mean"], dtype=np.float32)
    stds = np.asarray(statistics["std"], dtype=np.float32)
    if means.shape != (8,) or stds.shape != (8,) or np.any(stds <= 0):
        raise ValueError("invalid near-fall normalization statistics")
    valid = output[..., 7] > 0
    for channel in range(7):
        normalized = (output[..., channel] - means[channel]) / stds[channel]
        output[..., channel] = np.where(valid, normalized, 0.0)
    output[..., 7] = valid.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("normalized near-fall features contain non-finite values")
    return output


def prepare_near_fall_event_dataset(
    *,
    labels: str | Path,
    manifest: str | Path,
    assignments: str | Path,
    split: str | Path,
    validation: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    config: NearFallDatasetConfig | None = None,
) -> dict[str, Any]:
    preparation = config or NearFallDatasetConfig()
    paths = {
        "labels": Path(labels),
        "manifest": Path(manifest),
        "assignments": Path(assignments),
        "split": Path(split),
        "validation": Path(validation),
    }
    pose_root = Path(pose_dir)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"near-fall {name} input not found: {path}")
    if not pose_root.is_dir():
        raise FileNotFoundError(f"near-fall pose directory not found: {pose_root}")

    split_report = json.loads(paths["split"].read_text(encoding="utf-8"))
    validation_report = json.loads(paths["validation"].read_text(encoding="utf-8"))
    require_near_fall_training_ready(validation_report, split_report)
    _validate_input_hashes(paths, split_report, validation_report)

    label_rows = _read_jsonl(paths["labels"])
    manifest_index = _manifest_index(_read_jsonl(paths["manifest"]))
    assignment_index = _assignment_index(_read_jsonl(paths["assignments"]))
    selected = select_near_fall_training_labels(label_rows)
    _validate_assignment_links(selected, assignment_index)
    leakage = _protection_unit_leakage(selected, assignment_index)
    if leakage:
        first = leakage[0]
        raise ValueError(
            "near-fall partition leakage for "
            f"{first['field']}={first['value']}: {first['partitions']}"
        )
    if split_report.get("leakage_issues"):
        raise ValueError("source near-fall split report contains leakage issues")
    _validate_selected_supervision(selected, assignment_index, split_report)

    destination = Path(output_dir)
    output_paths = {
        "dataset": destination / "dataset.npz",
        "metadata": destination / "metadata.json",
        "samples": destination / "samples.jsonl",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"near-fall dataset output already exists: {existing[0]}")

    tensors: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    pose_paths: dict[str, Path] = {}
    locked_test_labels: list[dict[str, Any]] = []
    for label in sorted(
        selected,
        key=lambda row: (
            str(assignment_index[str(row["label_id"])]["partition"]),
            str(row["video_id"]),
            int(row["start_frame"]),
            str(row["label_id"]),
        ),
    ):
        assignment = assignment_index[str(label["label_id"])]
        partition = str(assignment["partition"])
        if partition == "test":
            locked_test_labels.append(label)
            continue
        manifest_row = manifest_index.get(str(label["video_id"]))
        if manifest_row is None:
            raise ValueError(
                f"near-fall label references missing manifest video: {label['video_id']}"
            )
        _validate_manifest_link(label, manifest_row)
        video_id = str(label["video_id"])
        pose_path = _resolve_pose_path(pose_root, video_id)
        if pose_path is None:
            raise FileNotFoundError(
                f"development near-fall pose JSONL not found for {video_id}"
            )
        pose_paths[video_id] = pose_path
        track = _select_labeled_track(label, _read_jsonl(pose_path))
        if not track:
            rejected["no_matching_pose_track"] += 1
            continue
        anchors = _window_anchor_frames(label, track, preparation)
        if not anchors:
            rejected["insufficient_causal_context"] += 1
            continue
        for window_index, anchor_frame in enumerate(anchors):
            anchor_record = track[anchor_frame]
            anchor_time = float(anchor_record["timestamp_sec"])
            slots = _resample_causal_window(
                list(track.values()),
                anchor_time_sec=anchor_time,
                anchor_frame=anchor_frame,
                config=preparation,
            )
            quality = _window_quality(slots, preparation.window_frames)
            if quality["observed_frame_count"] < preparation.min_observed_frames:
                rejected["insufficient_observed_frames"] += 1
                continue
            if (
                quality["usable_frame_ratio"] < preparation.min_usable_frame_ratio
                or quality["joint_coverage"] < preparation.min_joint_coverage
                or quality["mean_joint_quality"] < preparation.min_mean_joint_quality
            ):
                rejected["insufficient_quality"] += 1
                continue
            tensor = build_near_fall_tensor(
                slots,
                window_frames=preparation.window_frames,
                target_fps=preparation.target_fps,
            )
            if not np.any(tensor[..., 7] > 0):
                rejected["empty_valid_mask"] += 1
                continue
            source_frames = [
                int(record["frame_id"]) for record in slots if record is not None
            ]
            event_id = str(label.get("physical_event_id") or label["label_id"])
            sample = {
                "sample_id": f"{label['label_id']}:window-{window_index:03d}",
                "event_id": event_id,
                "label_id": str(label["label_id"]),
                "video_id": video_id,
                "asset_id": str(label["asset_id"]),
                "label": int(label["label"]),
                "target_name": str(label["target_name"]),
                "hard_negative_type": label.get("hard_negative_type"),
                "partition": partition,
                "subject_id": str(label["subject_id"]),
                "source_group_id": str(label["source_group_id"]),
                "sample_group_id": str(label["sample_group_id"]),
                "split_group_id": str(assignment["split_group_id"]),
                "physical_event_id": label.get("physical_event_id"),
                "dataset": str(manifest_row.get("dataset", "unknown")),
                "anchor_reason": (
                    "recovery_frame" if int(label["label"]) == 1 else "explicit_negative_window_end"
                ),
                "window_end_frame": anchor_frame,
                "window_end_time_sec": anchor_time,
                "max_source_frame": max(source_frames),
                "quality": quality,
                "loss_eligible": True,
            }
            tensors.append(tensor)
            samples.append(sample)

    if not samples:
        raise ValueError("no usable near-fall development windows were generated")
    _validate_development_samples(samples)
    event_counts = Counter(str(sample["event_id"]) for sample in samples)
    weights = np.asarray(
        [1.0 / event_counts[str(sample["event_id"])] for sample in samples],
        dtype=np.float32,
    )
    _validate_event_weights(samples, weights)
    raw_features = np.stack(tensors).astype(np.float32)
    partitions = np.asarray([sample["partition"] for sample in samples])
    normalization = fit_normalization_statistics(raw_features, partitions)
    features = apply_normalization(raw_features, normalization)
    arrays = {
        "features": features,
        "labels": np.asarray([sample["label"] for sample in samples], dtype=np.int64),
        "partitions": partitions,
        "sample_ids": np.asarray([sample["sample_id"] for sample in samples]),
        "event_ids": np.asarray([sample["event_id"] for sample in samples]),
        "subject_ids": np.asarray([sample["subject_id"] for sample in samples]),
        "source_group_ids": np.asarray([sample["source_group_id"] for sample in samples]),
        "sample_group_ids": np.asarray([sample["sample_group_id"] for sample in samples]),
        "split_group_ids": np.asarray([sample["split_group_id"] for sample in samples]),
        "sample_weights": weights,
        "loss_eligible": np.ones(len(samples), dtype=np.bool_),
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
    }
    metadata = {
        "schema_version": "near-fall-event-dataset-v1",
        "task": "near_fall_recovery_confirmation_v1",
        "status": "training_ready_source",
        "synthetic": False,
        "training_ready": True,
        "target_semantics": (
            "binary near-fall confirmation using only a causal window ending at recovery_frame"
        ),
        "negative_semantics": (
            "explicit primary negatives; progressed_to_fall is a binary negative"
        ),
        "label_mapping": {"explicit_negative": 0, "near_fall": 1},
        "joint_order": list(NEAR_FALL_JOINTS),
        "channel_order": list(NEAR_FALL_CHANNELS),
        "feature_shape": list(features.shape),
        "preparation_config": asdict(preparation),
        "normalization": {
            "fit_partition": "train",
            "mean": normalization["mean"].tolist(),
            "std": normalization["std"].tolist(),
            "valid_mask_normalized": False,
        },
        "sample_count": len(samples),
        "event_count": len(event_counts),
        "partition_counts": dict(sorted(Counter(partitions.tolist()).items())),
        "locked_test_label_count": len(locked_test_labels),
        "test_pose_read": False,
        "test_evaluated": False,
        "rejected_window_counts": dict(sorted(rejected.items())),
        "event_weight_sums": {
            event_id: float(weights[arrays["event_ids"] == event_id].sum())
            for event_id in sorted(event_counts)
        },
        "source_split_id": split_report.get("split_id"),
        "input_sha256": {
            name: _sha256_file(path) for name, path in paths.items()
        },
        "pose_inputs": {
            video_id: {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for video_id, path in sorted(pose_paths.items())
        },
        "limitations": [
            "this task confirms recovery after recovery_frame; it is not onset-time early warning",
            "test poses and test metrics are not read or materialized",
            "the rule candidate generator is not used to select supervision",
        ],
    }

    destination.mkdir(parents=True, exist_ok=False)
    _write_npz_atomic(output_paths["dataset"], **arrays)
    _write_jsonl_atomic(output_paths["samples"], samples)
    metadata["samples_sha256"] = _sha256_file(output_paths["samples"])
    metadata["dataset_sha256"] = _sha256_file(output_paths["dataset"])
    _write_json_atomic(output_paths["metadata"], metadata)
    return {
        "dataset_path": output_paths["dataset"].as_posix(),
        "metadata_path": output_paths["metadata"].as_posix(),
        "samples_path": output_paths["samples"].as_posix(),
        "sample_count": len(samples),
        "synthetic": False,
        "test_evaluated": False,
    }


def create_synthetic_near_fall_dataset(
    output_dir: str | Path,
    *,
    seed: int = 42,
    sample_count_per_class: int = 12,
    window_frames: int = 16,
    target_fps: float = 8.0,
) -> dict[str, Any]:
    """Create a deterministic, explicitly synthetic overfit fixture."""

    if sample_count_per_class < 4:
        raise ValueError("synthetic dataset requires at least four samples per class")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"synthetic near-fall output already exists: {destination}")
    rng = np.random.default_rng(seed)
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    partitions: list[str] = []
    event_ids: list[str] = []
    for label in (0, 1):
        for sample_index in range(sample_count_per_class):
            tensor = _synthetic_tensor(
                rng,
                label=label,
                window_frames=window_frames,
                target_fps=target_fps,
            )
            partition = (
                "validation"
                if sample_index >= sample_count_per_class - 2
                else "train"
            )
            tensors.append(tensor)
            labels.append(label)
            partitions.append(partition)
            event_ids.append(f"synthetic-{label}-{sample_index:03d}")
    raw = np.stack(tensors).astype(np.float32)
    partition_array = np.asarray(partitions)
    normalization = fit_normalization_statistics(raw, partition_array)
    features = apply_normalization(raw, normalization)
    identifiers = np.asarray(event_ids)
    arrays = {
        "features": features,
        "labels": np.asarray(labels, dtype=np.int64),
        "partitions": partition_array,
        "sample_ids": identifiers,
        "event_ids": identifiers,
        "subject_ids": np.asarray([f"subject-{value}" for value in event_ids]),
        "source_group_ids": np.asarray([f"source-{value}" for value in event_ids]),
        "sample_group_ids": np.asarray([f"sample-{value}" for value in event_ids]),
        "split_group_ids": np.asarray([f"split-{value}" for value in event_ids]),
        "sample_weights": np.ones(len(features), dtype=np.float32),
        "loss_eligible": np.ones(len(features), dtype=np.bool_),
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
    }
    destination.mkdir(parents=True, exist_ok=False)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    _write_npz_atomic(dataset_path, **arrays)
    metadata = {
        "schema_version": "near-fall-event-dataset-v1",
        "task": "near_fall_recovery_confirmation_v1",
        "status": "synthetic_smoke_only",
        "synthetic": True,
        "training_ready": False,
        "target_semantics": "synthetic recovery-confirmation overfit fixture",
        "joint_order": list(NEAR_FALL_JOINTS),
        "channel_order": list(NEAR_FALL_CHANNELS),
        "feature_shape": list(features.shape),
        "window_frames": window_frames,
        "target_fps": target_fps,
        "normalization": {
            "fit_partition": "train",
            "mean": normalization["mean"].tolist(),
            "std": normalization["std"].tolist(),
            "valid_mask_normalized": False,
        },
        "dataset_sha256": _sha256_file(dataset_path),
        "seed": seed,
        "test_pose_read": False,
        "test_evaluated": False,
        "limitations": [
            "synthetic data only; no supervised real-data result",
            "not an onset-time early-warning model",
        ],
    }
    _write_json_atomic(metadata_path, metadata)
    return {
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "synthetic": True,
        "test_evaluated": False,
    }


def _synthetic_tensor(
    rng: np.random.Generator,
    *,
    label: int,
    window_frames: int,
    target_fps: float,
) -> np.ndarray:
    tensor = np.zeros((window_frames, 10, 8), dtype=np.float32)
    base_x = np.asarray([-0.25, 0.25, -0.45, 0.45, -0.16, 0.16, -0.16, 0.16, -0.18, 0.18])
    base_y = np.asarray([-1.0, -1.0, -0.3, -0.3, 0.0, 0.0, 0.8, 0.8, 1.6, 1.6])
    phase = np.linspace(0.0, 1.0, window_frames, dtype=np.float32)
    bump = np.sin(np.pi * phase) if label else np.zeros_like(phase)
    noise = rng.normal(0.0, 0.004, size=(window_frames, 10, 2)).astype(np.float32)
    tensor[..., 0] = base_x[None, :] + noise[..., 0]
    tensor[..., 1] = base_y[None, :] + noise[..., 1]
    if label:
        tensor[:, 0:4, 0] += 0.75 * bump[:, None]
        tensor[:, 0:4, 1] += 0.25 * bump[:, None]
        tensor[:, 6:10, 0] -= 0.12 * bump[:, None]
    tensor[..., 6] = 0.95
    tensor[..., 7] = 1.0
    tensor[1:, :, 2:4] = (tensor[1:, :, 0:2] - tensor[:-1, :, 0:2]) * target_fps
    tensor[2:, :, 4:6] = (tensor[2:, :, 2:4] - tensor[1:-1, :, 2:4]) * target_fps
    return tensor


def _validate_input_hashes(
    paths: Mapping[str, Path],
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> None:
    if split_report.get("schema_version") != "fall-risk-training-split-v3":
        raise ValueError("unsupported near-fall source split schema")
    expected = {
        "event_labels": _sha256_file(paths["labels"]),
        "manifest": _sha256_file(paths["manifest"]),
    }
    split_hashes = split_report.get("input_sha256", {})
    for key, digest in expected.items():
        if split_hashes.get(key) != digest:
            raise ValueError(f"near-fall source split {key} SHA-256 mismatch")
    assignments_hash = _sha256_file(paths["assignments"])
    if split_report.get("assignments_sha256") != assignments_hash:
        raise ValueError("near-fall source assignments SHA-256 mismatch")
    validation_expected = {
        "event_labels": expected["event_labels"],
        "manifest": expected["manifest"],
        "split_assignments": assignments_hash,
        "split_report": _sha256_file(paths["split"]),
    }
    validation_hashes = validation_report.get("input_sha256", {})
    for key, digest in validation_expected.items():
        if validation_hashes.get(key) != digest:
            raise ValueError(f"near-fall validation {key} SHA-256 mismatch")
    if validation_report.get("valid") is not True:
        raise ValueError("near-fall v3 validation report is not structurally valid")


def _assignment_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source in rows:
        label_id = str(source.get("label_id", ""))
        if not label_id:
            continue
        if label_id in index:
            raise ValueError(f"duplicate near-fall split assignment: {label_id}")
        partition = str(source.get("partition", ""))
        if partition not in _ALL_PARTITIONS:
            raise ValueError(f"invalid near-fall partition for {label_id}: {partition}")
        index[label_id] = dict(source)
    return index


def _manifest_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source in rows:
        raw_video_id = source.get("video_id")
        if raw_video_id is None:
            continue
        video_id = str(raw_video_id).strip()
        if not video_id:
            continue
        if video_id in index:
            raise ValueError(f"duplicate near-fall manifest video_id: {video_id}")
        index[video_id] = dict(source)
    return index


def _validate_assignment_links(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    for label in labels:
        label_id = str(label["label_id"])
        assignment = assignments.get(label_id)
        if assignment is None:
            raise ValueError(f"missing near-fall split assignment for {label_id}")
        if assignment.get("label_kind") != "event" or assignment.get("task_type") != "near_fall_event":
            raise ValueError(f"near-fall assignment task mismatch for {label_id}")
        for field in (
            "asset_id",
            "video_id",
            "content_sha256",
            "subject_id",
            "source_group_id",
            "sample_group_id",
            "physical_event_id",
            "training_tier",
        ):
            if str(assignment.get(field, "")) != str(label.get(field, "")):
                raise ValueError(f"near-fall assignment {field} mismatch for {label_id}")


def _protection_unit_leakage(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for label in labels:
        assignment = assignments[str(label["label_id"])]
        partition = str(assignment["partition"])
        fields = {
            "subject_id": label.get("subject_id"),
            "physical_event_id": label.get("physical_event_id"),
            "content_sha256": label.get("content_sha256"),
            "source_group_id": label.get("source_group_id"),
            "sample_group_id": label.get("sample_group_id"),
            "video_id": label.get("video_id"),
            "split_group_id": assignment.get("split_group_id"),
        }
        for field, raw_value in fields.items():
            value = str(raw_value or "").strip()
            if value and value.lower() not in {"unknown", "none", "null"}:
                values[(field, value)].add(partition)
    return [
        {"field": field, "value": value, "partitions": sorted(partitions)}
        for (field, value), partitions in sorted(values.items())
        if len(partitions) > 1
    ]


def _validate_selected_supervision(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    split_report: Mapping[str, Any],
) -> None:
    coverage = {
        str(row.get("hard_negative_type"))
        for row in labels
        if int(row["label"]) == 0
    }
    missing = sorted(set(NEAR_FALL_HARD_NEGATIVES) - coverage)
    if missing:
        raise ValueError(f"near-fall selected labels lack hard negatives: {missing}")
    reported_counts = split_report.get("partition_supervision_counts", {}).get(
        "near_fall_event", {}
    )
    for partition in sorted(_ALL_PARTITIONS):
        partition_labels = [
            row
            for row in labels
            if assignments[str(row["label_id"])]["partition"] == partition
        ]
        targets = {int(row["label"]) for row in partition_labels}
        if targets != {0, 1}:
            raise ValueError(f"near-fall {partition} partition lacks a binary class")
        computed = {
            "primary_positive": sum(int(row["label"]) == 1 for row in partition_labels),
            "primary_negative": sum(int(row["label"]) == 0 for row in partition_labels),
        }
        reported = reported_counts.get(partition, {})
        if any(int(reported.get(key, -1)) != value for key, value in computed.items()):
            raise ValueError(
                f"near-fall source split supervision counts mismatch for {partition}"
            )


def _validate_manifest_link(
    label: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    video_id = str(label["video_id"])
    if manifest.get("eligibility") is not True:
        raise ValueError(f"near-fall label references ineligible video: {video_id}")
    if str(manifest.get("asset_id", "")) != str(label.get("asset_id", "")):
        raise ValueError(f"near-fall label/manifest asset mismatch: {video_id}")
    if str(manifest.get("sha256", "")) != str(label.get("content_sha256", "")):
        raise ValueError(f"near-fall label/manifest content SHA-256 mismatch: {video_id}")


def _resolve_pose_path(root: Path, video_id: str) -> Path | None:
    return next(
        (
            path
            for path in (
                root / f"{video_id}.jsonl",
                root / f"{video_id}_poses_cleaned.jsonl",
            )
            if path.is_file()
        ),
        None,
    )


def _select_labeled_track(
    label: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    start = int(label["start_frame"])
    end = int(label["end_frame_exclusive"])
    by_track: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    requested_track = str(label.get("track_id") or "")
    for source in records:
        frame_id = int(source.get("frame_id", -1))
        if not start <= frame_id < end:
            continue
        track_id = str(source.get("track_id", source.get("person_id", "unknown")))
        if requested_track and track_id != requested_track:
            continue
        record = dict(source)
        previous = by_track[track_id].get(frame_id)
        if previous is None or _record_confidence(record) > _record_confidence(previous):
            by_track[track_id][frame_id] = record
    if not by_track:
        return {}
    return max(
        by_track.values(),
        key=lambda track: (
            len(track),
            float(np.mean([_record_confidence(row) for row in track.values()])),
        ),
    )


def _record_confidence(record: Mapping[str, Any]) -> float:
    return _optional_float(record.get("pose_confidence")) or 0.0


def _window_anchor_frames(
    label: Mapping[str, Any],
    track: Mapping[int, Mapping[str, Any]],
    config: NearFallDatasetConfig,
) -> list[int]:
    if int(label["label"]) == 1:
        recovery = int(label["recovery_frame"])
        return [recovery] if recovery in track else []
    ordered = sorted(
        (
            float(record["timestamp_sec"]),
            frame_id,
        )
        for frame_id, record in track.items()
        if _optional_float(record.get("timestamp_sec")) is not None
    )
    if not ordered:
        return []
    span = (config.window_frames - 1) / config.target_fps
    first_time = ordered[0][0] + span
    eligible = [(time_value, frame_id) for time_value, frame_id in ordered if time_value >= first_time - 1e-9]
    if not eligible:
        return []
    targets = list(
        np.arange(eligible[0][0], eligible[-1][0] + 1e-9, config.stride_sec)
    )
    if not targets or abs(targets[-1] - eligible[-1][0]) > 1e-6:
        targets.append(eligible[-1][0])
    anchors: list[int] = []
    for target in targets:
        _, frame_id = min(eligible, key=lambda item: (abs(item[0] - target), item[1]))
        if frame_id not in anchors:
            anchors.append(frame_id)
    if len(anchors) > config.max_windows_per_event:
        indices = np.linspace(0, len(anchors) - 1, config.max_windows_per_event, dtype=np.int64)
        anchors = [anchors[index] for index in sorted(set(indices.tolist()))]
    return anchors


def _resample_causal_window(
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_time_sec: float,
    anchor_frame: int,
    config: NearFallDatasetConfig,
) -> list[dict[str, Any] | None]:
    timed = sorted(
        (
            float(row["timestamp_sec"]),
            int(row["frame_id"]),
            dict(row),
        )
        for row in records
        if _optional_float(row.get("timestamp_sec")) is not None
        and int(row.get("frame_id", -1)) <= anchor_frame
        and float(row["timestamp_sec"]) <= anchor_time_sec + 1e-9
    )
    output: list[dict[str, Any] | None] = []
    for target_index in range(config.window_frames):
        target_time = anchor_time_sec - (
            (config.window_frames - 1 - target_index) / config.target_fps
        )
        candidates = [
            (abs(source_time - target_time), frame_id, row)
            for source_time, frame_id, row in timed
            if abs(source_time - target_time) <= config.max_gap_sec
        ]
        if not candidates:
            output.append(None)
            continue
        _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        output.append(dict(selected))
    return output


def _window_quality(
    records: Sequence[Mapping[str, Any] | None], window_frames: int
) -> dict[str, Any]:
    observed = [record for record in records if record is not None]
    if not observed:
        return {
            "observed_frame_count": 0,
            "usable_frame_ratio": 0.0,
            "joint_coverage": 0.0,
            "mean_joint_quality": 0.0,
        }
    usable = sum(
        record.get("window_quality", {}).get("usable_for_near_fall") is True
        for record in observed
    )
    valid_points = 0
    qualities: list[float] = []
    for record in observed:
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        for name in NEAR_FALL_JOINTS:
            point = points.get(name)
            if point is None or point.get("valid") is not True or point.get("is_jump_outlier") is True:
                continue
            quality = _optional_float(point.get("quality_weight", point.get("score")))
            if quality is None or quality <= 0:
                continue
            valid_points += 1
            qualities.append(quality)
    denominator = window_frames * len(NEAR_FALL_JOINTS)
    return {
        "observed_frame_count": len(observed),
        "usable_frame_ratio": usable / window_frames,
        "joint_coverage": valid_points / denominator,
        "mean_joint_quality": float(np.mean(qualities)) if qualities else 0.0,
    }


def _validate_development_samples(samples: Sequence[Mapping[str, Any]]) -> None:
    for partition in sorted(_DEVELOPMENT_PARTITIONS):
        targets = {
            int(sample["label"])
            for sample in samples
            if sample["partition"] == partition
        }
        if targets != {0, 1}:
            raise ValueError(f"near-fall {partition} windows lack a binary class")
    for field in (
        "subject_id",
        "source_group_id",
        "sample_group_id",
        "split_group_id",
        "physical_event_id",
    ):
        partitions: dict[str, set[str]] = defaultdict(set)
        for sample in samples:
            value = str(sample.get(field) or "").strip()
            if not value or value.lower() in {"unknown", "none", "null"}:
                continue
            partitions[value].add(str(sample["partition"]))
        leaked = [value for value, parts in partitions.items() if len(parts) > 1]
        if leaked:
            raise ValueError(f"near-fall {field} values cross partitions: {leaked[:5]}")


def _validate_event_weights(
    samples: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> None:
    event_ids = np.asarray([str(sample["event_id"]) for sample in samples])
    for event_id in np.unique(event_ids):
        if not np.isclose(float(weights[event_ids == event_id].sum()), 1.0, atol=1e-6):
            raise ValueError(f"near-fall event weights do not sum to 1: {event_id}")


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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


def _write_jsonl_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
