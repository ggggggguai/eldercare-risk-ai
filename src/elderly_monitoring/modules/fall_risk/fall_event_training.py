from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.sit_stand_training import (
    SIT_STAND_CHANNELS,
    SIT_STAND_JOINTS,
    build_sit_stand_tensor,
)


FALL_EVENT_JOINTS = SIT_STAND_JOINTS
FALL_EVENT_CHANNELS = SIT_STAND_CHANNELS
POSITIVE_ACTION_IDS = ("D01", "D02", "D03", "D05")
PROXY_NEGATIVE_ACTION_IDS = ("A03", "A05", "A06", "A09")
DEVELOPMENT_PARTITIONS = {"train", "validation"}
ALL_PARTITIONS = {*DEVELOPMENT_PARTITIONS, "test"}


@dataclass(frozen=True)
class FallEventDatasetConfig:
    target_fps: float = 8.0
    window_sec: float = 4.0
    max_gap_sec: float = 0.25
    min_observed_frames: int = 12
    min_usable_frame_ratio: float = 0.60
    min_core_joint_coverage: float = 0.70
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_fps <= 0 or self.window_sec <= 0:
            raise ValueError("target_fps and window_sec must be positive")
        if self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.window_frames < 2:
            raise ValueError("fall-event window must contain at least two frames")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        if not 0 < self.min_usable_frame_ratio <= 1:
            raise ValueError("min_usable_frame_ratio must be within (0, 1]")
        if not 0 < self.min_core_joint_coverage <= 1:
            raise ValueError("min_core_joint_coverage must be within (0, 1]")

    @property
    def window_frames(self) -> int:
        return int(round(self.target_fps * self.window_sec))


def build_fall_event_tensor(
    records: Sequence[Mapping[str, Any] | None],
    *,
    window_frames: int,
    target_fps: float,
) -> np.ndarray:
    """Build the fixed fall-event candidate [T,14,7] pose contract."""

    return build_sit_stand_tensor(
        records,
        window_frames=window_frames,
        target_fps=target_fps,
    )


def audit_fall_event_training_inputs(
    *,
    labels: str | Path,
    manifest: str | Path,
    assignments: str | Path,
    split: str | Path,
    validation: str | Path,
    pose_dir: str | Path,
) -> dict[str, Any]:
    labels_path = Path(labels)
    manifest_path = Path(manifest)
    assignments_path = Path(assignments)
    split_path = Path(split)
    validation_path = Path(validation)
    pose_root = Path(pose_dir)
    for path in (
        labels_path,
        manifest_path,
        assignments_path,
        split_path,
        validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required fall-event training input not found: {path}")
    if not pose_root.is_dir():
        raise FileNotFoundError(f"fall-event pose directory not found: {pose_root}")

    label_rows = _read_jsonl(labels_path)
    manifest_rows = _read_jsonl(manifest_path)
    assignment_rows = _read_jsonl(assignments_path)
    _validate_label_rows(label_rows)
    assignment_index = _assignment_index(assignment_rows)
    selected = _selected_labels(label_rows)
    if not selected:
        raise ValueError("no explicitly labelled fall-event proxy actions were found")
    _validate_assignment_links(selected, assignment_index)
    _validate_group_isolation(selected, assignment_index)

    split_report = json.loads(split_path.read_text(encoding="utf-8"))
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    _validate_input_hashes(
        labels_path=labels_path,
        manifest_path=manifest_path,
        assignments_path=assignments_path,
        split_path=split_path,
        split_report=split_report,
        validation_report=validation_report,
    )
    if split_report.get("leakage_issues"):
        raise ValueError("source training split contains leakage issues")
    if validation_report.get("valid") is not True:
        raise ValueError("fall-risk training-label validation report is not valid")

    manifest_index = _manifest_index(manifest_rows)
    counts: dict[str, dict[str, Any]] = {}
    missing_pose_ids: list[str] = []
    for partition in sorted(ALL_PARTITIONS):
        partition_rows = [
            row
            for row in selected
            if assignment_index[str(row["label_id"])]["partition"] == partition
        ]
        action_counts = Counter(str(row["action_id"]) for row in partition_rows)
        counts[partition] = {
            "label_count": len(partition_rows),
            "positive_count": sum(
                row["action_id"] in POSITIVE_ACTION_IDS for row in partition_rows
            ),
            "negative_count": sum(
                row["action_id"] in PROXY_NEGATIVE_ACTION_IDS for row in partition_rows
            ),
            "action_counts": dict(sorted(action_counts.items())),
        }
        if partition not in DEVELOPMENT_PARTITIONS:
            continue
        for row in partition_rows:
            video_id = str(row["video_id"])
            manifest_row = manifest_index.get(video_id)
            if manifest_row is None:
                raise ValueError(f"fall-event label references missing manifest video: {video_id}")
            if manifest_row.get("eligibility") is not True:
                raise ValueError(f"fall-event label references ineligible video: {video_id}")
            if _resolve_pose_path(pose_root, video_id) is None:
                missing_pose_ids.append(video_id)

    proxy_ready = not missing_pose_ids and all(
        counts[partition]["positive_count"] > 0
        and counts[partition]["negative_count"] > 0
        and all(
            counts[partition]["action_counts"].get(action_id, 0) > 0
            for action_id in POSITIVE_ACTION_IDS
        )
        for partition in sorted(DEVELOPMENT_PARTITIONS)
    )
    formal_event_ready = bool(
        validation_report.get("training_ready", {}).get("fall_event", False)
    )
    return {
        "schema_version": "fall-event-proxy-training-audit-v1",
        "task": "fall_action_presence_proxy_v1",
        "target_task": "fall_event_v1",
        "status": "provisional",
        "source_split_id": split_report.get("split_id"),
        "input_sha256": {
            "labels": _sha256_file(labels_path),
            "manifest": _sha256_file(manifest_path),
            "assignments": _sha256_file(assignments_path),
            "split": _sha256_file(split_path),
            "validation": _sha256_file(validation_path),
        },
        "proxy_classes": {
            "positive_action_ids": list(POSITIVE_ACTION_IDS),
            "negative_action_ids": list(PROXY_NEGATIVE_ACTION_IDS),
            "negative_semantics": "explicitly labelled confusable non-fall actions",
        },
        "partition_counts": counts,
        "training_gates": {
            "clip_proxy_ready": proxy_ready,
            "formal_event_ready": formal_event_ready,
            "continuous_event_localization_ready": False,
            "continuous_background_ready": False,
        },
        "data_access": {
            "development_pose_missing": len(set(missing_pose_ids)),
            "development_pose_missing_video_ids": sorted(set(missing_pose_ids)),
            "test_pose_read": False,
        },
        "limitations": [
            "the negative class contains only explicitly labelled action clips",
            "unlabelled video intervals are never inferred to be background",
            "test pose data is not opened during audit or provisional preparation",
            "the proxy supports candidate-clip classification, not event localization",
        ],
    }


def prepare_fall_event_proxy_dataset(
    *,
    labels: str | Path,
    manifest: str | Path,
    assignments: str | Path,
    split: str | Path,
    validation: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    config: FallEventDatasetConfig | None = None,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    if not allow_provisional:
        raise ValueError(
            "fall-event proxy preparation is provisional; pass allow_provisional explicitly"
        )
    preparation = config or FallEventDatasetConfig()
    labels_path = Path(labels)
    manifest_path = Path(manifest)
    assignments_path = Path(assignments)
    split_path = Path(split)
    validation_path = Path(validation)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    paths = {
        "audit": destination / "audit.json",
        "dataset": destination / "dataset.npz",
        "metadata": destination / "metadata.json",
        "samples": destination / "samples.jsonl",
        "split": destination / "split.json",
        "assignments": destination / "assignments.jsonl",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"fall-event dataset output already exists: {existing[0]}")

    audit = audit_fall_event_training_inputs(
        labels=labels_path,
        manifest=manifest_path,
        assignments=assignments_path,
        split=split_path,
        validation=validation_path,
        pose_dir=pose_root,
    )
    missing = audit["data_access"]["development_pose_missing_video_ids"]
    if missing:
        raise FileNotFoundError(
            "development pose-quality JSONL not found for "
            + ", ".join(str(value) for value in missing[:10])
        )
    if not audit["training_gates"]["clip_proxy_ready"]:
        raise ValueError("fall-event clip proxy data gate is not ready")

    label_rows = _selected_labels(_read_jsonl(labels_path))
    assignment_index = _assignment_index(_read_jsonl(assignments_path))
    manifest_index = _manifest_index(_read_jsonl(manifest_path))
    label_rows.sort(
        key=lambda row: (
            str(row.get("video_id", "")),
            int(row.get("start_frame", 0)),
            str(row.get("label_id", "")),
        )
    )

    development: list[dict[str, Any]] = []
    locked_test: list[dict[str, Any]] = []
    for label in label_rows:
        row = dict(label)
        assignment = assignment_index[str(label["label_id"])]
        row["partition"] = str(assignment["partition"])
        row["split_group_id"] = str(assignment["split_group_id"])
        if row["partition"] in DEVELOPMENT_PARTITIONS:
            development.append(row)
        elif row["partition"] == "test":
            locked_test.append(row)

    pose_cache: dict[str, list[dict[str, Any]]] = {}
    pose_paths: dict[str, Path] = {}
    tensors: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for label in development:
        video_id = str(label["video_id"])
        manifest_row = manifest_index[video_id]
        if str(manifest_row.get("asset_id", "")) != str(label.get("asset_id", "")):
            raise ValueError(f"fall-event label/manifest asset mismatch: {video_id}")
        if video_id not in pose_cache:
            pose_path = _resolve_pose_path(pose_root, video_id)
            if pose_path is None:
                raise FileNotFoundError(
                    f"development pose-quality JSONL not found for {video_id}"
                )
            pose_paths[video_id] = pose_path
            pose_cache[video_id] = _read_jsonl(pose_path)
        track_records = _select_labeled_track(label, pose_cache[video_id])
        if not track_records:
            rejected["no_matching_pose_track"] += 1
            continue
        window_start = _centered_window_start(label, preparation.window_sec)
        slots = _resample_pose_records(
            track_records,
            start_time_sec=window_start,
            window_frames=preparation.window_frames,
            target_fps=preparation.target_fps,
            max_gap_sec=preparation.max_gap_sec,
        )
        observed = [record for record in slots if record is not None]
        if len(observed) < preparation.min_observed_frames:
            rejected["insufficient_observed_frames"] += 1
            continue
        quality = _window_quality(slots)
        if (
            quality["usable_frame_ratio"] < preparation.min_usable_frame_ratio
            or quality["core_joint_coverage"]
            < preparation.min_core_joint_coverage
        ):
            rejected["insufficient_quality"] += 1
            continue
        tensor = build_fall_event_tensor(
            slots,
            window_frames=preparation.window_frames,
            target_fps=preparation.target_fps,
        )
        if not np.any(tensor[..., -1] > 0):
            rejected["empty_valid_mask"] += 1
            continue
        action_id = str(label["action_id"])
        target = int(action_id in POSITIVE_ACTION_IDS)
        sample = {
            "sample_id": str(label["label_id"]),
            "segment_id": str(label["label_id"]),
            "label_id": str(label["label_id"]),
            "video_id": video_id,
            "asset_id": str(label["asset_id"]),
            "action_id": action_id,
            "action_type": str(label["action_type"]),
            "label": target,
            "target_name": "fall_action" if target else "explicit_non_fall_action",
            "partition": str(label["partition"]),
            "split_group_id": str(label["split_group_id"]),
            "source_group_id": str(label["source_group_id"]),
            "sample_group_id": str(label["sample_group_id"]),
            "subject_id": str(label.get("subject_id", "unknown")),
            "dataset": str(manifest_row.get("dataset", "unknown")),
            "scene_region": str(manifest_row.get("scene_region", "unknown")),
            "label_start_time_sec": float(label["start_time"]),
            "label_end_time_exclusive": float(label["end_time_exclusive"]),
            "window_start_time_sec": round(window_start, 6),
            "window_end_time_exclusive": round(
                window_start + preparation.window_sec, 6
            ),
            "observed_frame_count": len(observed),
            "quality": quality,
        }
        tensors.append(tensor)
        samples.append(sample)

    if not samples:
        raise ValueError("no usable fall-event proxy segments were generated")
    _validate_development_samples(samples)

    destination.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(paths["audit"], audit)
    _write_jsonl_atomic(paths["samples"], samples)
    derived_assignments = [
        {
            "schema_version": "fall-event-proxy-split-assignment-v1",
            "label_id": str(row["label_id"]),
            "video_id": str(row["video_id"]),
            "split_group_id": str(
                assignment_index[str(row["label_id"])]["split_group_id"]
            ),
            "source_group_id": str(row["source_group_id"]),
            "sample_group_id": str(row["sample_group_id"]),
            "subject_id": str(row.get("subject_id", "unknown")),
            "action_id": str(row["action_id"]),
            "action_type_training_tier": str(row["action_type_training_tier"]),
            "partition": str(assignment_index[str(row["label_id"])]["partition"]),
            "features_materialized": (
                assignment_index[str(row["label_id"])]["partition"]
                in DEVELOPMENT_PARTITIONS
            ),
        }
        for row in label_rows
    ]
    _write_jsonl_atomic(paths["assignments"], derived_assignments)

    arrays = {
        "features": np.stack(tensors).astype(np.float32),
        "labels": np.asarray([sample["label"] for sample in samples], dtype=np.int64),
        "partitions": np.asarray([sample["partition"] for sample in samples]),
        "sample_ids": np.asarray([sample["sample_id"] for sample in samples]),
        "segment_ids": np.asarray([sample["segment_id"] for sample in samples]),
        "video_ids": np.asarray([sample["video_id"] for sample in samples]),
        "split_group_ids": np.asarray([sample["split_group_id"] for sample in samples]),
        "source_group_ids": np.asarray([sample["source_group_id"] for sample in samples]),
        "sample_group_ids": np.asarray([sample["sample_group_id"] for sample in samples]),
        "datasets": np.asarray([sample["dataset"] for sample in samples]),
        "action_ids": np.asarray([sample["action_id"] for sample in samples]),
        "sample_weights": np.ones(len(samples), dtype=np.float32),
    }
    _write_npz_atomic(paths["dataset"], **arrays)

    partition_counts = Counter(str(value) for value in arrays["partitions"])
    partition_class_counts = {
        partition: {
            "negative": int(
                np.sum((arrays["partitions"] == partition) & (arrays["labels"] == 0))
            ),
            "positive": int(
                np.sum((arrays["partitions"] == partition) & (arrays["labels"] == 1))
            ),
        }
        for partition in sorted(DEVELOPMENT_PARTITIONS)
    }
    source_statistics = {
        partition: {
            "segment_count": sum(
                sample["partition"] == partition for sample in samples
            ),
            "split_group_count": len(
                {
                    str(sample["split_group_id"])
                    for sample in samples
                    if sample["partition"] == partition
                }
            ),
            "source_group_count": len(
                {
                    str(sample["source_group_id"])
                    for sample in samples
                    if sample["partition"] == partition
                }
            ),
            "dataset_counts": dict(
                sorted(
                    Counter(
                        str(sample["dataset"])
                        for sample in samples
                        if sample["partition"] == partition
                    ).items()
                )
            ),
            "action_counts": dict(
                sorted(
                    Counter(
                        str(sample["action_id"])
                        for sample in samples
                        if sample["partition"] == partition
                    ).items()
                )
            ),
        }
        for partition in sorted(DEVELOPMENT_PARTITIONS)
    }
    split_payload = {
        "schema_version": "fall-event-proxy-split-v1",
        "task": "fall_action_presence_proxy_v1",
        "status": "provisional",
        "source_split_id": audit["source_split_id"],
        "source_split_sha256": _sha256_file(split_path),
        "assignments_sha256": _sha256_file(paths["assignments"]),
        "protection_fields": [
            "subject_id",
            "source_group_id",
            "sample_group_id",
            "video_id",
            "split_group_id",
        ],
        "leakage_issues": [],
        "test_policy": "labels_counted_but_pose_and_features_not_read",
        "development_segment_counts": dict(sorted(partition_counts.items())),
        "development_class_counts": partition_class_counts,
        "source_statistics": source_statistics,
        "locked_test_label_count": len(locked_test),
    }
    _write_json_atomic(paths["split"], split_payload)
    metadata = {
        "schema_version": "fall-action-presence-proxy-dataset-v1",
        "task": "fall_action_presence_proxy_v1",
        "target_task": "fall_event_v1",
        "status": "provisional",
        "target_semantics": (
            "pre-segmented fall actions versus explicitly labelled confusable non-fall actions"
        ),
        "label_mapping": {"explicit_non_fall_action": 0, "fall_action": 1},
        "positive_action_ids": list(POSITIVE_ACTION_IDS),
        "negative_action_ids": list(PROXY_NEGATIVE_ACTION_IDS),
        "joint_order": list(FALL_EVENT_JOINTS),
        "channel_order": list(FALL_EVENT_CHANNELS),
        "feature_shape": list(arrays["features"].shape),
        "preparation_config": asdict(preparation),
        "normalization_contract": (
            "pelvis_centered_torso_scaled_xy_with_image_y_preserved_and_per_second_velocity"
        ),
        "sample_count": len(samples),
        "segment_count": len(samples),
        "development_segment_counts": dict(sorted(partition_counts.items())),
        "development_class_counts": partition_class_counts,
        "source_statistics": source_statistics,
        "locked_test_label_count": len(locked_test),
        "test_pose_read": False,
        "rejected_segment_counts": dict(sorted(rejected.items())),
        "input_sha256": dict(audit["input_sha256"]),
        "audit_sha256": _sha256_file(paths["audit"]),
        "source_split_id": audit["source_split_id"],
        "derived_split_sha256": _sha256_file(paths["split"]),
        "derived_assignments_sha256": _sha256_file(paths["assignments"]),
        "samples_sha256": _sha256_file(paths["samples"]),
        "pose_inputs": {
            video_id: {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for video_id, path in sorted(pose_paths.items())
        },
        "limitations": [
            "this model target is a provisional candidate-clip proxy, not continuous event localization",
            "no unlabelled interval is treated as a negative example",
            "no test pose, test feature or test metric is available in this dataset",
            "the score is not a medically or manually confirmed fall conclusion",
        ],
    }
    metadata["dataset_sha256"] = _sha256_file(paths["dataset"])
    _write_json_atomic(paths["metadata"], metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": paths["dataset"].as_posix(),
        "metadata_path": paths["metadata"].as_posix(),
        "audit_path": paths["audit"].as_posix(),
        "split_path": paths["split"].as_posix(),
        "sample_count": len(samples),
        "locked_test_label_count": len(locked_test),
    }


def _selected_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    allowed = {*POSITIVE_ACTION_IDS, *PROXY_NEGATIVE_ACTION_IDS}
    return [
        dict(row)
        for row in rows
        if str(row.get("action_id")) in allowed
        and row.get("action_type_training_tier") == "primary"
    ]


def _validate_label_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    allowed_tiers = {"primary", "auxiliary", "ignore"}
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "fall-risk-action-label-v3":
            raise ValueError("fall-event proxy requires action-label schema v3")
        label_id = str(row.get("label_id", ""))
        if not label_id or label_id in seen:
            raise ValueError(f"duplicate or missing fall-event action label_id: {label_id}")
        seen.add(label_id)
        if row.get("action_type_training_tier") not in allowed_tiers:
            raise ValueError(
                f"invalid action_type_training_tier for fall-event proxy: {label_id}"
            )


def _assignment_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        label_id = str(row.get("label_id", ""))
        if not label_id or label_id in output:
            raise ValueError(f"duplicate or missing split assignment label_id: {label_id}")
        partition = str(row.get("partition", ""))
        if partition not in ALL_PARTITIONS:
            raise ValueError(f"invalid split assignment partition: {partition}")
        output[label_id] = dict(row)
    return output


def _validate_assignment_links(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    for label in labels:
        label_id = str(label["label_id"])
        assignment = assignments.get(label_id)
        if assignment is None:
            raise ValueError(f"fall-event label has no split assignment: {label_id}")
        if assignment.get("label_kind") != "action":
            raise ValueError(f"fall-event proxy assignment is not an action: {label_id}")
        for field in ("video_id", "asset_id", "subject_id", "source_group_id"):
            if str(assignment.get(field, "")) != str(label.get(field, "")):
                raise ValueError(f"fall-event label/assignment {field} mismatch: {label_id}")


def _validate_group_isolation(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    for field in (
        "subject_id",
        "source_group_id",
        "sample_group_id",
        "video_id",
        "split_group_id",
    ):
        partitions: dict[str, set[str]] = defaultdict(set)
        for label in labels:
            assignment = assignments[str(label["label_id"])]
            value = str(assignment.get(field, label.get(field, ""))).strip()
            if value and value.lower() != "unknown":
                partitions[value].add(str(assignment["partition"]))
        leaked = sorted(value for value, values in partitions.items() if len(values) > 1)
        if leaked:
            raise ValueError(f"fall-event partition leakage for {field}: {leaked[0]}")


def _validate_input_hashes(
    *,
    labels_path: Path,
    manifest_path: Path,
    assignments_path: Path,
    split_path: Path,
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> None:
    actual = {
        "action_labels": _sha256_file(labels_path),
        "manifest": _sha256_file(manifest_path),
        "split_assignments": _sha256_file(assignments_path),
        "split_report": _sha256_file(split_path),
    }
    split_inputs = split_report.get("input_sha256", {})
    for field in ("action_labels", "manifest"):
        expected = split_inputs.get(field)
        if expected is not None and expected != actual[field]:
            raise ValueError(f"fall-event source split {field} SHA-256 mismatch")
    expected_assignments = split_report.get("assignments_sha256")
    if expected_assignments is not None and expected_assignments != actual["split_assignments"]:
        raise ValueError("fall-event source split assignments SHA-256 mismatch")
    validation_inputs = validation_report.get("input_sha256", {})
    for field, value in actual.items():
        expected = validation_inputs.get(field)
        if expected is not None and expected != value:
            raise ValueError(f"fall-event validation {field} SHA-256 mismatch")


def _manifest_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_video_id = row.get("video_id")
        if raw_video_id in (None, ""):
            continue
        video_id = str(raw_video_id)
        if video_id in output:
            raise ValueError(f"duplicate manifest video_id: {video_id}")
        output[video_id] = dict(row)
    return output


def _resolve_pose_path(root: Path, video_id: str) -> Path | None:
    candidates = (
        root / f"{video_id}.jsonl",
        root / f"{video_id}_poses_cleaned.jsonl",
    )
    return next((path for path in candidates if path.is_file()), None)


def _select_labeled_track(
    label: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    start_frame = int(label["start_frame"])
    end_frame = int(label["end_frame_exclusive"]) - 1
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        frame_id = int(source.get("frame_id", -1))
        if start_frame <= frame_id <= end_frame:
            track_id = str(source.get("track_id", source.get("person_id", "unknown")))
            candidates[track_id].append(dict(source))
    if not candidates:
        return []
    selected_id = max(
        candidates,
        key=lambda track_id: (
            len(candidates[track_id]),
            float(np.mean([_record_confidence(row) for row in candidates[track_id]])),
        ),
    )
    output = [
        dict(row)
        for row in records
        if str(row.get("track_id", row.get("person_id", "unknown"))) == selected_id
    ]
    output.sort(key=lambda row: (float(row.get("timestamp_sec", 0.0)), int(row.get("frame_id", 0))))
    return output


def _record_confidence(record: Mapping[str, Any]) -> float:
    value = _optional_float(record.get("pose_confidence", record.get("keypoint_quality")))
    return value if value is not None else 0.0


def _centered_window_start(label: Mapping[str, Any], window_sec: float) -> float:
    start = float(label["start_time"])
    end = float(label["end_time_exclusive"])
    if end <= start:
        raise ValueError(f"invalid fall-event label time range: {label['label_id']}")
    return max(0.0, ((start + end) / 2.0) - (window_sec / 2.0))


def _resample_pose_records(
    records: Sequence[Mapping[str, Any]],
    *,
    start_time_sec: float,
    window_frames: int,
    target_fps: float,
    max_gap_sec: float,
) -> list[dict[str, Any] | None]:
    timed = sorted(
        (
            (float(row["timestamp_sec"]), dict(row))
            for row in records
            if _optional_float(row.get("timestamp_sec")) is not None
        ),
        key=lambda item: item[0],
    )
    selected: dict[int, tuple[float, int]] = {}
    for source_index, (source_time, _) in enumerate(timed):
        target_index = int(np.floor(((source_time - start_time_sec) * target_fps) + 0.5))
        if not 0 <= target_index < window_frames:
            continue
        target_time = start_time_sec + (target_index / target_fps)
        distance = abs(source_time - target_time)
        if distance > max_gap_sec:
            continue
        candidate = (distance, source_index)
        if target_index not in selected or candidate < selected[target_index]:
            selected[target_index] = candidate
    output: list[dict[str, Any] | None] = []
    for target_index in range(window_frames):
        candidate = selected.get(target_index)
        if candidate is None:
            output.append(None)
            continue
        row = dict(timed[candidate[1]][1])
        row["timestamp_sec"] = round(start_time_sec + (target_index / target_fps), 6)
        output.append(row)
    return output


def _window_quality(records: Sequence[Mapping[str, Any] | None]) -> dict[str, float]:
    observed = [record for record in records if record is not None]
    if not observed:
        return {
            "usable_frame_ratio": 0.0,
            "core_joint_coverage": 0.0,
            "mean_joint_quality": 0.0,
        }
    usable = sum(
        record.get("window_quality", {}).get("usable_for_near_fall") is True
        for record in observed
    )
    core_names = {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    valid_core = 0
    qualities: list[float] = []
    for record in observed:
        for point in record.get("keypoints", []):
            if not isinstance(point, Mapping) or point.get("name") not in core_names:
                continue
            if point.get("valid") is True and point.get("is_jump_outlier") is not True:
                valid_core += 1
                quality = _optional_float(point.get("quality_weight", point.get("score")))
                if quality is not None:
                    qualities.append(quality)
    denominator = max(1, len(observed) * len(core_names))
    return {
        "usable_frame_ratio": usable / len(observed),
        "core_joint_coverage": valid_core / denominator,
        "mean_joint_quality": float(np.mean(qualities)) if qualities else 0.0,
    }


def _validate_development_samples(samples: Sequence[Mapping[str, Any]]) -> None:
    for partition in sorted(DEVELOPMENT_PARTITIONS):
        targets = {
            int(sample["label"])
            for sample in samples
            if sample["partition"] == partition
        }
        if targets != {0, 1}:
            raise ValueError(f"fall-event {partition} partition lacks a binary class")
        positive_actions = {
            str(sample["action_id"])
            for sample in samples
            if sample["partition"] == partition and int(sample["label"]) == 1
        }
        if positive_actions != set(POSITIVE_ACTION_IDS):
            raise ValueError(f"fall-event {partition} partition lacks fall subtypes")
    for field in (
        "subject_id",
        "source_group_id",
        "sample_group_id",
        "video_id",
        "split_group_id",
        "segment_id",
    ):
        partitions: dict[str, set[str]] = defaultdict(set)
        for sample in samples:
            value = str(sample.get(field, "")).strip()
            if value and value.lower() != "unknown":
                partitions[value].add(str(sample["partition"]))
        leaked = sorted(value for value, values in partitions.items() if len(values) > 1)
        if leaked:
            raise ValueError(f"fall-event {field} crosses partitions: {leaked[0]}")


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
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


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
