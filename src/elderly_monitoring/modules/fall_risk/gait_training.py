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

from elderly_monitoring.modules.fall_risk.gait import (
    GAIT_STABILITY_FEATURE_NAMES,
    GaitAnalysisConfig,
    extract_gait_windows,
)
from elderly_monitoring.modules.fall_risk.gait_tensor import build_gait_tensor
from elderly_monitoring.modules.fall_risk.gait_contract import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
)


GAIT_TABULAR_FEATURE_NAMES = GAIT_STABILITY_FEATURE_NAMES + (
    "usable_frame_ratio",
    "gait_keypoint_coverage",
    "mean_core_keypoint_quality",
    "interpolated_point_ratio",
    "jump_outlier_per_frame",
)
GAIT_TARGET_MAPPING = {"normal_activity": 0, "gait_instability": 1}
GAIT_TARGET_PROFILES = (
    "gait_instability_b01_b04",
    "observable_instability_b02_b04",
)
_NORMAL_ACTIVITY_ACTION_IDS = tuple(f"A{index:02d}" for index in range(1, 13))
_UNSTABLE_GAIT_ACTION_IDS = ("B01", "B02", "B03", "B04")
_NORMAL_ACTIVITY_ACTIONS = {
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
    "A10",
    "A11",
    "A12",
    "normal_walk",
    "normal_turn",
    "controlled_sit_down",
    "normal_sit_to_stand",
    "controlled_squat",
    "controlled_bend",
    "controlled_lie_down",
    "routine_support_contact",
    "kneel_or_floor_activity",
    "normal_step_adjustment",
    "assisted_sit_or_lowering",
    "normal_hop",
}
_UNSTABLE_GAIT_ACTIONS = {
    "B01",
    "B02",
    "B03",
    "B04",
    "slow_walk",
    "dragging_walk",
    "shuffling_walk",
    "swaying_walk",
}
GAIT_WALKING_ACTION_IDS = frozenset({"A01", "B01", "B02", "B03", "B04"})
_PARTITIONS = ("train", "validation", "test")
_ALLOWED_PARTITIONS = (*_PARTITIONS, "excluded")
_V3_ACTION_SCHEMA = "fall-risk-action-label-v3"


@dataclass(frozen=True)
class GaitWindowPreparationConfig:
    window_frames: int = 16
    stride_frames: int = 8
    min_observed_frames: int = 10
    require_usable_gait: bool = True
    target_fps: float = 4.0
    max_gap_sec: float = 0.5
    max_windows_per_segment: int = 4
    auxiliary_weight: float = 0.35
    context_expansion: bool = False
    primary_min_labeled_observations: int = 10
    weak_min_labeled_observations: int = 5
    representation_min_labeled_observations: int = 2
    weak_context_weight: float = 0.35
    target_profile: str = "gait_instability_b01_b04"
    excluded_action_ids: tuple[str, ...] = ()
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    split_search_iterations: int = 4096

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.stride_frames < 1:
            raise ValueError("stride_frames must be positive")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")
        if self.split_search_iterations < 1:
            raise ValueError("split_search_iterations must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.max_windows_per_segment < 1:
            raise ValueError("max_windows_per_segment must be positive")
        if not 0 < self.auxiliary_weight <= 1:
            raise ValueError("auxiliary_weight must be within (0, 1]")
        if self.context_expansion and not (
            1
            <= self.representation_min_labeled_observations
            <= self.weak_min_labeled_observations
            <= self.primary_min_labeled_observations
            <= self.window_frames
        ):
            raise ValueError("gait evidence thresholds must be ordered within the window")
        if not 0 < self.weak_context_weight <= 1:
            raise ValueError("weak_context_weight must be within (0, 1]")
        if self.target_profile not in GAIT_TARGET_PROFILES:
            raise ValueError(
                "target_profile must be gait_instability_b01_b04 or "
                "observable_instability_b02_b04"
            )
        if any(not action_id.strip() for action_id in self.excluded_action_ids):
            raise ValueError("excluded_action_ids must not contain empty values")
        if len(set(self.excluded_action_ids)) != len(self.excluded_action_ids):
            raise ValueError("excluded_action_ids must be unique")


def select_gait_training_labels(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        action_id = str(row.get("action_id", ""))
        is_v3 = row.get("schema_version") == _V3_ACTION_SCHEMA
        action_name = str(
            row.get("action_type", "") if is_v3 else row.get("action_name", "")
        )
        event_type = str(row.get("event_type", ""))
        if action_id in _UNSTABLE_GAIT_ACTIONS or action_name in _UNSTABLE_GAIT_ACTIONS:
            if not is_v3 and event_type != "gait_instability":
                continue
            label = 1
            target_name = "gait_instability"
        elif action_id in _NORMAL_ACTIVITY_ACTIONS or action_name in _NORMAL_ACTIVITY_ACTIONS:
            if not is_v3 and event_type != "normal_activity":
                continue
            label = 0
            target_name = "normal_activity"
        else:
            continue
        if is_v3:
            if row.get("training_tier") == "ignore":
                continue
            end_exclusive = row.get("end_frame_exclusive")
            if not isinstance(end_exclusive, int) or end_exclusive <= int(
                row.get("start_frame", -1)
            ):
                raise ValueError(
                    f"v3 gait label {row.get('label_id')} has invalid half-open frame bounds"
                )
            row["action_name"] = action_name
            row["event_type"] = target_name
            row["end_frame"] = end_exclusive - 1
            if row.get("end_time_exclusive") is not None:
                row["end_time"] = row["end_time_exclusive"]
        elif row.get("quality") not in (None, "", "clear", "partial_occlusion"):
            continue
        row["label"] = label
        row["target_name"] = target_name
        selected.append(row)
    return selected


def _target_contract(
    config: GaitWindowPreparationConfig,
    excluded_action_ids: set[str],
) -> dict[str, Any]:
    return {
        "profile": config.target_profile,
        "negative_action_ids": [
            action_id
            for action_id in _NORMAL_ACTIVITY_ACTION_IDS
            if action_id not in excluded_action_ids
        ],
        "positive_action_ids": [
            action_id
            for action_id in _UNSTABLE_GAIT_ACTION_IDS
            if action_id not in excluded_action_ids
        ],
        "functional_proxy_action_ids": (
            ["B01"]
            if config.target_profile == "observable_instability_b02_b04"
            else []
        ),
    }


def enrich_gait_training_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select gait labels and join their authoritative video manifest metadata."""

    manifests: dict[str, dict[str, Any]] = {}
    for source in manifest_rows:
        manifest = dict(source)
        raw_video_id = manifest.get("video_id")
        if raw_video_id is None or raw_video_id == "":
            continue
        video_id = str(raw_video_id)
        if video_id in manifests:
            raise ValueError(f"duplicate gait manifest video_id: {video_id}")
        manifests[video_id] = manifest

    enriched: list[dict[str, Any]] = []
    for label in select_gait_training_labels(rows):
        video_id = str(label.get("video_id", ""))
        manifest = manifests.get(video_id)
        if manifest is None:
            raise ValueError(f"gait label references missing manifest video: {video_id}")
        if manifest.get("eligibility") is not True:
            raise ValueError(f"gait label references ineligible manifest video: {video_id}")
        label_asset = str(label.get("asset_id", ""))
        manifest_asset = str(manifest.get("asset_id", ""))
        if label_asset and manifest_asset and label_asset != manifest_asset:
            raise ValueError(f"gait label/manifest asset mismatch for {video_id}")
        path = str(manifest.get("path", ""))
        if not path:
            raise ValueError(f"gait manifest is missing video path: {video_id}")
        row = dict(label)
        row.update(
            {
                "file_path": path,
                "scene": str(manifest.get("scene_region", "unknown")),
                "dataset": str(manifest.get("dataset", "unknown")),
                "fps": manifest.get("fps"),
                "frame_count": manifest.get("frame_count"),
                "manifest_content_sha256": manifest.get("sha256"),
            }
        )
        enriched.append(row)
    return enriched


def prepare_gait_window_dataset(
    labels_path: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    assignments_path: str | Path | None = None,
    split_report_path: str | Path | None = None,
    config: GaitWindowPreparationConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    preparation = config or GaitWindowPreparationConfig()
    label_source = Path(labels_path)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    source_rows = _read_jsonl(label_source)
    is_v3 = bool(source_rows) and all(
        row.get("schema_version") == _V3_ACTION_SCHEMA for row in source_rows
    )
    manifest_source = Path(manifest_path) if manifest_path is not None else None
    assignments_source = Path(assignments_path) if assignments_path is not None else None
    split_report_source = Path(split_report_path) if split_report_path is not None else None
    frozen_assignments: dict[str, dict[str, Any]] = {}
    if is_v3:
        if (
            manifest_source is None
            or assignments_source is None
            or split_report_source is None
        ):
            raise ValueError(
                "v3 gait preparation requires manifest_path, assignments_path and split_report_path"
            )
        split_report = _validate_split_report(
            label_source,
            manifest_source,
            assignments_source,
            split_report_source,
        )
        labels = enrich_gait_training_labels(
            source_rows,
            manifest_rows=_read_jsonl(manifest_source),
        )
        frozen_assignments = _load_frozen_assignments(
            labels,
            _read_jsonl(assignments_source),
        )
    else:
        split_report = None
        if any(
            path is not None
            for path in (manifest_source, assignments_source, split_report_source)
        ):
            raise ValueError("manifest/assignments/split report require v3 labels")
        labels = select_gait_training_labels(source_rows)
    excluded_action_ids = set(preparation.excluded_action_ids)
    if preparation.target_profile == "observable_instability_b02_b04":
        excluded_action_ids.add("B01")
    labels = [
        label
        for label in labels
        if str(label.get("action_id", "")) not in excluded_action_ids
    ]
    if not labels:
        raise ValueError("no gait_instability or normal_walk labels were selected")

    test_label_audit: dict[str, Any] = {
        "label_count": 0,
        "action_id_counts": {},
        "source_group_count": 0,
        "sample_group_count": 0,
        "split_group_count": 0,
        "pose_read": False,
    }
    if is_v3:
        sealed_test_labels = [
            label
            for label in labels
            if frozen_assignments[str(label["label_id"])]["partition"] == "test"
        ]
        test_label_audit = _sealed_test_label_audit(
            sealed_test_labels,
            frozen_assignments,
        )
        labels = [
            label
            for label in labels
            if frozen_assignments[str(label["label_id"])]["partition"]
            in {"train", "validation"}
            and not (
                label.get("training_tier") == "auxiliary"
                and frozen_assignments[str(label["label_id"])]["partition"]
                != "train"
            )
        ]
        if not labels:
            raise ValueError("no train/validation gait labels remain after test isolation")

    split_group = "split_group_id" if is_v3 else _split_group_field(labels)
    pose_cache: dict[str, list[dict[str, Any]]] = {}
    pose_paths: dict[str, Path] = {}
    samples: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    label_span_masks: list[np.ndarray] = []
    tabular_rows: list[list[float]] = []
    rule_scores: list[float] = []
    rejected = Counter()

    labels.sort(
        key=lambda row: (
            str(row.get("video_id", "")),
            int(row.get("start_frame", 0)),
            str(row.get("label_id", "")),
        )
    )
    for label in labels:
        video_id = str(label.get("video_id", ""))
        if not video_id:
            raise ValueError("gait training label is missing video_id")
        if video_id not in pose_cache:
            path = _resolve_pose_path(pose_root, video_id)
            if path is None:
                raise FileNotFoundError(
                    f"pose-quality JSONL not found for {video_id} in {pose_root}"
                )
            pose_paths[video_id] = path
            pose_cache[video_id] = _read_jsonl(path)

        source_pose_records = pose_cache[video_id]
        selected_records = _select_labeled_track(label, source_pose_records)
        if not selected_records:
            rejected["no_matching_pose_track"] += 1
            continue
        start_frame = int(label["start_frame"])
        end_frame = int(label["end_frame"])
        if is_v3:
            assignment = frozen_assignments[str(label["label_id"])]
            label["split_group_id"] = str(assignment["split_group_id"])
            start_time = float(label["start_time"])
            end_time = float(label["end_time_exclusive"])
            segment_frame_count = max(
                1,
                int(np.ceil(max(0.0, end_time - start_time) * preparation.target_fps)),
            )
            if preparation.context_expansion and segment_frame_count < preparation.window_frames:
                context_window = build_context_gait_window(
                    label,
                    source_pose_records,
                    window_frames=preparation.window_frames,
                    target_fps=preparation.target_fps,
                    max_gap_sec=preparation.max_gap_sec,
                )
                frame_records = list(context_window["records"])
                context_valid_mask = np.asarray(
                    context_window["valid_mask"], dtype=np.uint8
                )
                context_label_span_mask = np.asarray(
                    context_window["label_span_mask"], dtype=np.uint8
                )
                frame_records_start_time = float(context_window["start_time_sec"])
            else:
                frame_records = resample_pose_records(
                    list(selected_records.values()),
                    start_time_sec=start_time,
                    window_frames=segment_frame_count,
                    target_fps=preparation.target_fps,
                    max_gap_sec=preparation.max_gap_sec,
                )
                context_valid_mask = None
                context_label_span_mask = None
                frame_records_start_time = start_time
        else:
            assignment = None
            context_valid_mask = None
            context_label_span_mask = None
            frame_records_start_time = None
            frame_records = [
                selected_records.get(frame_id)
                for frame_id in range(start_frame, end_frame + 1)
            ]
        window_starts = _limit_window_starts(
            _window_starts(len(frame_records), preparation),
            preparation.max_windows_per_segment,
        )
        for window_index, window_start in enumerate(window_starts):
            slots = frame_records[
                window_start : window_start + preparation.window_frames
            ]
            observed_records = [record for record in slots if record is not None]
            if len(observed_records) < preparation.min_observed_frames:
                rejected["insufficient_observed_frames"] += 1
                continue
            valid_mask = (
                context_valid_mask[
                    window_start : window_start + preparation.window_frames
                ]
                if context_valid_mask is not None
                else np.asarray(
                    [record is not None for record in slots], dtype=np.uint8
                )
            )
            if context_label_span_mask is not None:
                label_span_mask = context_label_span_mask[
                    window_start : window_start + preparation.window_frames
                ]
            elif is_v3:
                slot_start_time = start_time + (
                    window_start / preparation.target_fps
                )
                label_span_mask = np.asarray(
                    [
                        start_time
                        <= slot_start_time + (index / preparation.target_fps)
                        < end_time
                        for index in range(len(slots))
                    ],
                    dtype=np.uint8,
                )
                label_span_mask *= valid_mask
            else:
                label_span_mask = valid_mask.copy()
            labeled_records = [
                record
                for record, is_labeled in zip(slots, label_span_mask, strict=True)
                if record is not None and bool(is_labeled)
            ]
            labeled_observed_count = len(labeled_records)
            evidence_tier = (
                _gait_evidence_tier(labeled_observed_count, preparation)
                if preparation.context_expansion
                else "primary"
            )
            rule_records = (
                labeled_records
                if len(labeled_records) >= preparation.weak_min_labeled_observations
                else observed_records
            )
            rule = _rule_baseline(rule_records)
            if rule is None:
                rejected["rule_window_unavailable"] += 1
                continue
            quality = rule["quality_coverage"]
            if preparation.require_usable_gait and quality["insufficient_gait_quality"]:
                rejected["insufficient_gait_quality"] += 1
                continue
            tensor = build_gait_tensor(
                slots,
                window_frames=preparation.window_frames,
            )
            if not np.any(tensor[..., -1] > 0):
                rejected["empty_quality_mask"] += 1
                continue

            sample_id = f"{label['label_id']}:window-{window_index:03d}"
            split_group_id = str(label[split_group])
            observed_frame_ids = [
                int(record["frame_id"])
                for record in observed_records
                if record.get("frame_id") is not None
            ]
            if is_v3:
                assert frame_records_start_time is not None
                window_start_time = frame_records_start_time + (
                    window_start / preparation.target_fps
                )
                proposed_window_end = window_start_time + (
                    preparation.window_frames / preparation.target_fps
                )
                window_end_time_exclusive = (
                    proposed_window_end
                    if context_label_span_mask is not None
                    else min(end_time, proposed_window_end)
                )
                window_start_frame = min(observed_frame_ids)
                window_end_frame = max(observed_frame_ids)
                window_end_time = window_end_time_exclusive - (
                    1.0 / preparation.target_fps
                )
                segment_duration_sec = float(label["end_time_exclusive"]) - float(
                    label["start_time"]
                )
            else:
                window_start_time = (
                    float(slots[0]["timestamp_sec"])
                    if slots and slots[0] is not None
                    else None
                )
                window_end_time_exclusive = None
                window_start_frame = start_frame + window_start
                window_end_frame = min(
                    end_frame,
                    start_frame + window_start + preparation.window_frames - 1,
                )
                window_end_time = (
                    float(slots[-1]["timestamp_sec"])
                    if slots and slots[-1] is not None
                    else None
                )
                segment_duration_sec = max(
                    0.0,
                    float(label.get("end_time", 0.0))
                    - float(label.get("start_time", 0.0)),
                )
            tensors.append(tensor)
            valid_masks.append(np.asarray(valid_mask, dtype=np.uint8))
            label_span_masks.append(np.asarray(label_span_mask, dtype=np.uint8))
            rule_scores.append(float(rule["gait_risk_score"]))
            tabular_rows.append(_tabular_features(rule))
            samples.append(
                {
                    "sample_index": len(samples),
                    "sample_id": sample_id,
                    "label_id": str(label["label_id"]),
                    "video_id": video_id,
                    "split_group_id": split_group_id,
                    "source_group_id": str(label.get("source_group_id", split_group_id)),
                    "sample_group_id": str(label.get("sample_group_id", label["label_id"])),
                    "dataset": str(label.get("dataset", "unknown")),
                    "training_tier": str(label.get("training_tier", "primary")),
                    "label": int(label["label"]),
                    "target_name": str(label["target_name"]),
                    "action_id": str(label.get("action_id", "")),
                    "action_name": str(label.get("action_name", "")),
                    "scene": str(label.get("scene", "unknown")),
                    "start_frame": window_start_frame,
                    "end_frame": window_end_frame,
                    "observed_frame_count": len(observed_records),
                    "labeled_observed_frame_count": labeled_observed_count,
                    "evidence_tier": evidence_tier,
                    "primary_evaluation": evidence_tier == "primary",
                    "sensitivity_evaluation": evidence_tier == "weak_context",
                    "start_time_sec": window_start_time,
                    "end_time_sec": window_end_time,
                    "end_time_exclusive": window_end_time_exclusive,
                    "segment_duration_sec": segment_duration_sec,
                    "rule_gait_risk_score": float(rule["gait_risk_score"]),
                    "partition": str(assignment["partition"]) if assignment else None,
                    "frozen_partition": (
                        str(assignment["partition"]) if assignment else None
                    ),
                }
            )

    if not samples:
        raise ValueError("no usable gait windows were generated")
    labels_array = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    group_ids = np.asarray([sample["split_group_id"] for sample in samples])
    if is_v3:
        assignments: dict[str, str] = {}
        for sample in samples:
            group = str(sample["split_group_id"])
            frozen_partition = str(sample["frozen_partition"])
            if group in assignments and assignments[group] != frozen_partition:
                raise ValueError(f"frozen gait split group {group} has conflicting partitions")
            assignments[group] = frozen_partition
        partitions = np.asarray([str(sample["partition"]) for sample in samples])
        _validate_formal_partitions(samples)
    else:
        assignments = _search_group_split(labels_array, group_ids, preparation)
        partitions = np.asarray([assignments[str(group)] for group in group_ids])
    for sample, partition in zip(samples, partitions, strict=True):
        sample["partition"] = str(partition)

    fold_a_partitions = np.asarray(
        [_cross_source_partition(sample, holdout="pre_vfallp") for sample in samples]
    )
    fold_b_partitions = np.asarray(
        [_cross_source_partition(sample, holdout="le2i_imvia") for sample in samples]
    )
    partition_schemes = {"frozen": "partitions"}
    if not is_v3:
        partition_schemes.update(
            {"fold_a": "fold_a_partitions", "fold_b": "fold_b_partitions"}
        )

    segment_window_counts = Counter(str(sample["label_id"]) for sample in samples)
    duration_match_factors = _duration_match_negative_factors(samples, preparation)
    sample_weights = np.asarray(
        [
            _gait_supervision_weight(sample, preparation)
            * (
                duration_match_factors.get(str(sample["evidence_tier"]), 1.0)
                if sample["partition"] == "train" and sample["label"] == 0
                else 1.0
            )
            / segment_window_counts[str(sample["label_id"])]
            for sample in samples
        ],
        dtype=np.float32,
    )

    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    samples_path = destination / "samples.jsonl"
    split_path = destination / "split.json"
    assignments_output_path = destination / "assignments.jsonl"
    for path in (
        dataset_path,
        metadata_path,
        samples_path,
        split_path,
        assignments_output_path,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(f"gait dataset output already exists: {path}")

    features_array = np.stack(tensors).astype(np.float32)
    valid_masks_array = np.stack(valid_masks).astype(np.uint8)
    label_span_masks_array = np.stack(label_span_masks).astype(np.uint8)
    tabular_array = np.asarray(tabular_rows, dtype=np.float32)
    rule_array = np.asarray(rule_scores, dtype=np.float32)
    sample_ids = np.asarray([sample["sample_id"] for sample in samples])
    video_ids = np.asarray([sample["video_id"] for sample in samples])
    scenes = np.asarray([sample["scene"] for sample in samples])
    source_group_ids = np.asarray([sample["source_group_id"] for sample in samples])
    sample_group_ids = np.asarray([sample["sample_group_id"] for sample in samples])
    datasets = np.asarray([sample["dataset"] for sample in samples])
    training_tiers = np.asarray([sample["training_tier"] for sample in samples])
    action_segment_ids = np.asarray([sample["label_id"] for sample in samples])
    action_ids = np.asarray([sample["action_id"] for sample in samples])
    walking_targets = np.asarray(
        [int(action_id in GAIT_WALKING_ACTION_IDS) for action_id in action_ids],
        dtype=np.int64,
    )
    for sample, walking_target in zip(samples, walking_targets, strict=True):
        sample["walking_target"] = int(walking_target)
    segment_durations_sec = np.asarray(
        [sample["segment_duration_sec"] for sample in samples], dtype=np.float32
    )
    _write_npz_atomic(
        dataset_path,
        features=features_array,
        valid_masks=valid_masks_array,
        label_span_masks=label_span_masks_array,
        labels=labels_array,
        sample_ids=sample_ids,
        split_group_ids=group_ids,
        participant_ids=group_ids,
        video_ids=video_ids,
        groups=scenes,
        source_group_ids=source_group_ids,
        sample_group_ids=sample_group_ids,
        datasets=datasets,
        training_tiers=training_tiers,
        action_segment_ids=action_segment_ids,
        action_ids=action_ids,
        walking_targets=walking_targets,
        evidence_tiers=np.asarray([sample["evidence_tier"] for sample in samples]),
        labeled_observed_frame_counts=np.asarray(
            [sample["labeled_observed_frame_count"] for sample in samples],
            dtype=np.int64,
        ),
        primary_evaluation_mask=np.asarray(
            [sample["primary_evaluation"] for sample in samples], dtype=np.uint8
        ),
        sensitivity_evaluation_mask=np.asarray(
            [sample["sensitivity_evaluation"] for sample in samples], dtype=np.uint8
        ),
        segment_durations_sec=segment_durations_sec,
        sample_weights=sample_weights,
        partitions=partitions,
        fold_a_partitions=fold_a_partitions,
        fold_b_partitions=fold_b_partitions,
        tabular_features=tabular_array,
        rule_scores=rule_array,
    )
    _write_jsonl_atomic(samples_path, samples)
    _write_jsonl_atomic(
        assignments_output_path,
        [
            {
                "sample_id": sample["sample_id"],
                "label_id": sample["label_id"],
                "split_group_id": sample["split_group_id"],
                "source_group_id": sample["source_group_id"],
                "sample_group_id": sample["sample_group_id"],
                "training_tier": sample["training_tier"],
                "partition": sample["partition"],
            }
            for sample in samples
        ],
    )

    class_counts = Counter(int(value) for value in labels_array)
    partition_counts = Counter(str(value) for value in partitions)
    group_partition_counts = Counter(assignments.values())
    metadata = {
        "schema_version": (
            "gait-context-window-dataset-v2"
            if preparation.context_expansion
            else "gait-window-dataset-v1"
        ),
        "task": "gait_instability_vs_normal_activity",
        "negative_label_policy": "A01_A12_normal_activity_hard_negatives",
        "excluded_action_ids": sorted(excluded_action_ids),
        "target_contract": _target_contract(preparation, excluded_action_ids),
        "walking_gate_contract": {
            "walking_action_ids": sorted(GAIT_WALKING_ACTION_IDS),
            "non_walking_policy": "not_applicable_for_conditional_gait_head",
            "final_probability": "p_walking_times_p_abnormal_given_walking",
        },
        "label_mapping": dict(GAIT_TARGET_MAPPING),
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "tabular_feature_names": list(GAIT_TABULAR_FEATURE_NAMES),
        "feature_shape": list(features_array.shape),
        "preparation_config": asdict(preparation),
        "context_protocol": (
            {
                "enabled": True,
                "source": "same_pose_track_real_frames_only",
                "label_span_semantics": "half_open_annotation_interval",
                "context_semantics": "unlabeled_not_negative",
                "supervision_pooling": "label_span_mask",
                "validation_policy": "primary_only",
                "negative_duration_match_factors": duration_match_factors,
                "evidence_tiers": {
                    "primary": f">={preparation.primary_min_labeled_observations}",
                    "weak_context": (
                        f"{preparation.weak_min_labeled_observations}-"
                        f"{preparation.primary_min_labeled_observations - 1}"
                    ),
                    "representation_only": (
                        f"{preparation.representation_min_labeled_observations}-"
                        f"{preparation.weak_min_labeled_observations - 1}"
                    ),
                    "audit_only": (
                        f"<{preparation.representation_min_labeled_observations}"
                    ),
                },
            }
            if preparation.context_expansion
            else {"enabled": False}
        ),
        "split_group": split_group,
        "protocol_status": (
            "frozen"
            if is_v3
            and str(
                split_report.get("status") or split_report.get("split_status") or ""
            )
            == "frozen"
            else "development_provisional"
        ),
        "split_is_provisional": (
            not is_v3
            or str(
                split_report.get("status") or split_report.get("split_status") or ""
            )
            != "frozen"
        ),
        "split_limitations": (
            [
                "source labels do not provide usable subject_id values",
                "video-grouped separation cannot rule out the same actor across videos",
            ]
            if not is_v3
            else []
        ),
        "split_protocol": (
            "frozen_training_labels_v3"
            if is_v3
            and str(
                split_report.get("status") or split_report.get("split_status") or ""
            )
            == "frozen"
            else "development_provisional_training_labels_v3"
            if is_v3
            else "development_random_group"
        ),
        "sample_count": len(samples),
        "split_group_count": len(assignments),
        "class_counts": {str(key): class_counts.get(key, 0) for key in (0, 1)},
        "window_partition_counts": {
            partition: partition_counts.get(partition, 0)
            for partition in _ALLOWED_PARTITIONS
        },
        "group_partition_counts": {
            partition: group_partition_counts.get(partition, 0)
            for partition in _ALLOWED_PARTITIONS
        },
        "partition_schemes": partition_schemes,
        "group_assignments": dict(sorted(assignments.items())),
        "rejected_window_counts": dict(sorted(rejected.items())),
        "labels_path": label_source.as_posix(),
        "labels_sha256": _sha256_file(label_source),
        "manifest_path": manifest_source.as_posix() if manifest_source else None,
        "manifest_sha256": _sha256_file(manifest_source) if manifest_source else None,
        "assignments_path": assignments_source.as_posix() if assignments_source else None,
        "assignments_sha256": _sha256_file(assignments_source) if assignments_source else None,
        "source_split_report_path": (
            split_report_source.as_posix() if split_report_source else None
        ),
        "source_split_report_sha256": (
            _sha256_file(split_report_source) if split_report_source else None
        ),
        "source_split_id": split_report.get("split_id") if split_report else None,
        "quality_usage": "mask_and_pooling_only",
        "test_pose_read": False,
        "test_tensor_generated": False,
        "test_evaluated": False,
        "test_label_audit": test_label_audit,
        "input_sha256": {
            "labels": _sha256_file(label_source),
            "manifest": _sha256_file(manifest_source) if manifest_source else None,
            "assignments": _sha256_file(assignments_source) if assignments_source else None,
            "split_report": _sha256_file(split_report_source) if split_report_source else None,
        },
        "pose_inputs": {
            video_id: {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for video_id, path in sorted(pose_paths.items())
        },
        "samples_sha256": _sha256_file(samples_path),
        "assignments_output_sha256": _sha256_file(assignments_output_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "limitations": [
            "labels describe observable gait actions and are not clinical diagnoses",
            "probabilities require calibration before deployment",
            "the dataset is isolated from AlgorithmEvent and the realtime main path",
        ],
    }
    _write_json_atomic(
        split_path,
        {
            "schema_version": "gait-window-split-v1",
            "protocol": metadata["split_protocol"],
            "group_assignments": metadata["group_assignments"],
            "window_partition_counts": metadata["window_partition_counts"],
            "labels_sha256": metadata["labels_sha256"],
            "assignments_sha256": metadata["assignments_sha256"],
        },
    )
    metadata["split_sha256"] = _sha256_file(split_path)
    _write_json_atomic(metadata_path, metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "sample_count": len(samples),
        "split_group": split_group,
        "split_group_count": len(assignments),
        "window_partition_counts": metadata["window_partition_counts"],
        "rejected_window_counts": metadata["rejected_window_counts"],
    }


def _split_group_field(labels: Sequence[Mapping[str, Any]]) -> str:
    subject_ids = [str(row.get("subject_id", "")).strip() for row in labels]
    if subject_ids and all(value and value.lower() != "unknown" for value in subject_ids):
        return "subject_id"
    return "video_id"


def resample_pose_records(
    records: Sequence[Mapping[str, Any]],
    *,
    start_time_sec: float,
    window_frames: int,
    target_fps: float,
    max_gap_sec: float,
) -> list[dict[str, Any] | None]:
    """Select nearest pose observations on a fixed source-time grid."""

    if window_frames < 1 or target_fps <= 0 or max_gap_sec <= 0:
        raise ValueError("invalid gait resampling contract")
    timed = sorted(
        (
            (float(record["timestamp_sec"]), dict(record))
            for record in records
            if _optional_float(record.get("timestamp_sec")) is not None
        ),
        key=lambda item: item[0],
    )
    if not timed:
        return [None] * window_frames
    selected_by_target: dict[int, tuple[float, int]] = {}
    final_target_index = window_frames - 1
    for source_index, (source_time, _) in enumerate(timed):
        relative_index = (source_time - start_time_sec) * target_fps
        target_index = int(np.floor(relative_index + 0.5))
        target_index = min(final_target_index, max(0, target_index))
        target_time = start_time_sec + (target_index / target_fps)
        distance = abs(source_time - target_time)
        if distance > max_gap_sec:
            continue
        candidate = (distance, source_index)
        if target_index not in selected_by_target or candidate < selected_by_target[
            target_index
        ]:
            selected_by_target[target_index] = candidate

    output: list[dict[str, Any] | None] = []
    for target_index in range(window_frames):
        candidate = selected_by_target.get(target_index)
        if candidate is None:
            output.append(None)
            continue
        selected = dict(timed[candidate[1]][1])
        selected["timestamp_sec"] = round(
            start_time_sec + (target_index / target_fps), 6
        )
        output.append(selected)
    return output


def build_context_gait_window(
    label: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    window_frames: int,
    target_fps: float,
    max_gap_sec: float,
) -> dict[str, Any]:
    """Build one fixed window from real observations on the labeled pose track."""

    labeled_track = _select_labeled_track(label, records)
    if not labeled_track:
        raise ValueError("cannot expand gait context without a labeled pose track")
    anchor = next(iter(labeled_track.values()))
    track_key = str(anchor.get("track_id", anchor.get("person_id", "unknown")))
    track_records: dict[int, Mapping[str, Any]] = {}
    for record in records:
        candidate_key = str(
            record.get("track_id", record.get("person_id", "unknown"))
        )
        if candidate_key != track_key:
            continue
        frame_id = int(record.get("frame_id", -1))
        previous = track_records.get(frame_id)
        if previous is None or _record_confidence(record) > _record_confidence(previous):
            track_records[frame_id] = record
    timed = [
        float(record["timestamp_sec"])
        for record in track_records.values()
        if _optional_float(record.get("timestamp_sec")) is not None
    ]
    if not timed:
        raise ValueError("cannot expand gait context without source timestamps")
    label_start = float(label["start_time"])
    label_end = float(label["end_time_exclusive"])
    grid_span_sec = (window_frames - 1) / target_fps
    desired_start = ((label_start + label_end) / 2.0) - (grid_span_sec / 2.0)
    earliest = min(timed)
    latest_start = max(earliest, max(timed) - grid_span_sec)
    window_start = min(max(desired_start, earliest), latest_start)
    window_records = resample_pose_records(
        list(track_records.values()),
        start_time_sec=window_start,
        window_frames=window_frames,
        target_fps=target_fps,
        max_gap_sec=max_gap_sec,
    )
    target_times = [window_start + (index / target_fps) for index in range(window_frames)]
    valid_mask = [record is not None for record in window_records]
    label_span_mask = [
        bool(is_valid and label_start <= target_time < label_end)
        for target_time, is_valid in zip(target_times, valid_mask, strict=True)
    ]
    return {
        "records": window_records,
        "valid_mask": valid_mask,
        "label_span_mask": label_span_mask,
        "start_time_sec": window_start,
        "end_time_sec": window_start + grid_span_sec,
        "track_id": track_key,
    }


def _gait_evidence_tier(
    labeled_observations: int, config: GaitWindowPreparationConfig
) -> str:
    if labeled_observations >= config.primary_min_labeled_observations:
        return "primary"
    if labeled_observations >= config.weak_min_labeled_observations:
        return "weak_context"
    if labeled_observations >= config.representation_min_labeled_observations:
        return "representation_only"
    return "audit_only"


def _gait_supervision_weight(
    sample: Mapping[str, Any], config: GaitWindowPreparationConfig
) -> float:
    evidence_tier = str(sample.get("evidence_tier", "primary"))
    if evidence_tier in {"representation_only", "audit_only"}:
        return 0.0
    weight = config.weak_context_weight if evidence_tier == "weak_context" else 1.0
    if sample.get("training_tier") == "auxiliary":
        weight *= config.auxiliary_weight
    return weight


def _duration_match_negative_factors(
    samples: Sequence[Mapping[str, Any]], config: GaitWindowPreparationConfig
) -> dict[str, float]:
    """Match negative evidence-tier mass to positives without dropping segments."""

    if not config.context_expansion:
        return {}
    segments: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if sample.get("partition") == "train":
            segments.setdefault(str(sample["label_id"]), sample)
    positive_mass: Counter[str] = Counter()
    negative_mass: Counter[str] = Counter()
    for sample in segments.values():
        weight = _gait_supervision_weight(sample, config)
        if weight <= 0:
            continue
        target = positive_mass if int(sample["label"]) == 1 else negative_mass
        target[str(sample["evidence_tier"])] += weight
    positive_total = sum(positive_mass.values())
    negative_total = sum(negative_mass.values())
    if positive_total <= 0 or negative_total <= 0:
        return {}
    factors: dict[str, float] = {}
    for tier in sorted(set(positive_mass) | set(negative_mass)):
        if positive_mass[tier] <= 0 or negative_mass[tier] <= 0:
            continue
        factors[tier] = (
            (positive_mass[tier] / positive_total)
            / (negative_mass[tier] / negative_total)
        )
    return factors


def _load_frozen_assignments(
    labels: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for source in assignment_rows:
        if source.get("label_kind") != "action":
            continue
        label_id = str(source.get("label_id", ""))
        if not label_id:
            continue
        if label_id in assignments:
            raise ValueError(f"duplicate frozen split assignment: {label_id}")
        assignments[label_id] = dict(source)
    for label in labels:
        label_id = str(label["label_id"])
        assignment = assignments.get(label_id)
        if assignment is None:
            raise ValueError(f"missing frozen split assignment for gait label {label_id}")
        for field in (
            "asset_id",
            "video_id",
            "source_group_id",
            "sample_group_id",
            "training_tier",
        ):
            if str(assignment.get(field, "")) != str(label.get(field, "")):
                raise ValueError(f"frozen split {field} mismatch for gait label {label_id}")
        if assignment.get("partition") not in _PARTITIONS:
            raise ValueError(f"invalid frozen partition for gait label {label_id}")
        if not str(assignment.get("split_group_id", "")):
            raise ValueError(f"missing split_group_id for gait label {label_id}")
    return assignments


def _sealed_test_label_audit(
    labels: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    action_counts = Counter(str(label.get("action_id", "")) for label in labels)
    source_groups = {str(label.get("source_group_id", "")) for label in labels}
    sample_groups = {str(label.get("sample_group_id", "")) for label in labels}
    split_groups = {
        str(assignments[str(label["label_id"])].get("split_group_id", ""))
        for label in labels
    }
    return {
        "label_count": len(labels),
        "action_id_counts": dict(sorted(action_counts.items())),
        "source_group_count": len(source_groups - {""}),
        "sample_group_count": len(sample_groups - {""}),
        "split_group_count": len(split_groups - {""}),
        "pose_read": False,
    }


def _validate_split_report(
    labels_path: Path,
    manifest_path: Path,
    assignments_path: Path,
    split_report_path: Path,
) -> dict[str, Any]:
    report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "fall-risk-training-split-v3":
        raise ValueError("unsupported frozen gait split report schema")
    if report.get("leakage_issues"):
        raise ValueError("frozen gait split report contains leakage issues")
    input_hashes = report.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("frozen gait split report is missing input hashes")
    if input_hashes.get("action_labels") != _sha256_file(labels_path):
        raise ValueError("frozen gait split action label SHA-256 mismatch")
    if input_hashes.get("manifest") != _sha256_file(manifest_path):
        raise ValueError("frozen gait split manifest SHA-256 mismatch")
    if report.get("assignments_sha256") != _sha256_file(assignments_path):
        raise ValueError("frozen gait split assignments SHA-256 mismatch")
    return dict(report)


def _validate_formal_partitions(samples: Sequence[Mapping[str, Any]]) -> None:
    partition_by_group: dict[str, set[str]] = defaultdict(set)
    protected_partition_values: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in ("source_group_id", "sample_group_id")
    }
    for sample in samples:
        partition = str(sample["partition"])
        if partition != "excluded":
            partition_by_group[str(sample["split_group_id"])].add(partition)
            for field, values in protected_partition_values.items():
                value = str(sample.get(field, ""))
                if value:
                    values[value].add(partition)
        if partition in {"validation", "test"} and sample["training_tier"] != "primary":
            raise ValueError("auxiliary gait samples must not enter validation or test")
    leaked = sorted(group for group, values in partition_by_group.items() if len(values) != 1)
    if leaked:
        raise ValueError(f"frozen gait split groups cross partitions: {leaked[:10]}")
    for field, values in protected_partition_values.items():
        leaked_values = sorted(value for value, partitions in values.items() if len(partitions) != 1)
        if leaked_values:
            raise ValueError(f"frozen gait {field} crosses partitions: {leaked_values[:10]}")


def _cross_source_partition(
    sample: Mapping[str, Any], *, holdout: str
) -> str:
    if str(sample.get("frozen_partition", "")) == "test":
        return "excluded"
    dataset = str(sample.get("dataset", "unknown"))
    tier = str(sample.get("training_tier", "primary"))
    training_sources = {"le2i_imvia", "pre_vfallp", "caucafall", "ur_fall"}
    if dataset == holdout:
        return "validation" if tier == "primary" else "excluded"
    if dataset in training_sources:
        return "train"
    return "excluded"


def validate_frozen_partition_selection(
    frozen_partitions: Sequence[Any], selected_partitions: Sequence[Any]
) -> None:
    if len(frozen_partitions) != len(selected_partitions):
        raise ValueError("frozen and selected gait partitions have inconsistent lengths")
    violations = Counter()
    for frozen, selected in zip(
        frozen_partitions, selected_partitions, strict=True
    ):
        frozen_name = str(frozen)
        selected_name = str(selected)
        if selected_name != "excluded" and selected_name != frozen_name:
            violations[f"{frozen_name}_to_{selected_name}"] += 1
    if violations:
        raise ValueError(
            "selected gait partition scheme reuses frozen assignments: "
            f"{dict(sorted(violations.items()))}"
        )


def _validate_cross_source_fold(
    samples: Sequence[Mapping[str, Any]], partitions: np.ndarray, name: str
) -> None:
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    for partition in ("train", "validation"):
        if set(labels[partitions == partition].tolist()) != {0, 1}:
            raise ValueError(f"gait {name} {partition} partition lacks a binary class")
    for field in ("split_group_id", "source_group_id", "sample_group_id"):
        partition_by_group: dict[str, set[str]] = defaultdict(set)
        for sample, partition in zip(samples, partitions, strict=True):
            if partition != "excluded":
                partition_by_group[str(sample[field])].add(str(partition))
        leaked = [
            group for group, values in partition_by_group.items() if len(values) != 1
        ]
        if leaked:
            raise ValueError(
                f"gait {name} {field} values cross partitions: {leaked[:10]}"
            )


def _limit_window_starts(starts: Sequence[int], maximum: int) -> list[int]:
    if len(starts) <= maximum:
        return list(starts)
    indices = np.linspace(0, len(starts) - 1, num=maximum, dtype=np.int64)
    return [int(starts[index]) for index in sorted(set(indices.tolist()))]


def _resolve_pose_path(root: Path, video_id: str) -> Path | None:
    candidates = (
        root / f"{video_id}.jsonl",
        root / f"{video_id}_poses_cleaned.jsonl",
    )
    return next((path for path in candidates if path.is_file()), None)


def _select_labeled_track(
    label: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    start_frame = int(label["start_frame"])
    end_frame = int(label["end_frame"])
    by_track: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        frame_id = int(record.get("frame_id", -1))
        if not start_frame <= frame_id <= end_frame:
            continue
        track_id = str(record.get("track_id", record.get("person_id", "unknown")))
        previous = by_track[track_id].get(frame_id)
        if previous is None or _record_confidence(record) > _record_confidence(previous):
            by_track[track_id][frame_id] = record
    if not by_track:
        return {}

    best_track = max(
        sorted(by_track),
        key=lambda track_id: _track_match_score(
            label, by_track[track_id], start_frame, end_frame
        ),
    )
    return by_track[best_track]


def _track_match_score(
    label: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
) -> tuple[float, float, float, float]:
    frame_span = max(1, end_frame - start_frame + 1)
    coverage = len(records) / frame_span
    overlaps: list[float] = []
    for frame_id, record in records.items():
        actual = record.get("bbox_pixels", record.get("bbox"))
        expected = _interpolated_bbox(label, frame_id, start_frame, end_frame)
        if _is_bbox(actual) and expected is not None:
            overlaps.append(_bbox_iou(actual, expected))
    mean_iou = float(np.mean(overlaps)) if overlaps else 0.0
    mean_confidence = float(
        np.mean([_record_confidence(record) for record in records.values()])
    )
    # CVAT 的动作框比“在片段中出现更久”更能识别目标人物。覆盖率只作为
    # 辅助项，避免背景人物凭完整轨迹压过与标注框实际重合的短轨迹。
    return mean_iou + (0.25 * coverage), mean_iou, coverage, mean_confidence


def _interpolated_bbox(
    label: Mapping[str, Any],
    frame_id: int,
    start_frame: int,
    end_frame: int,
) -> list[float] | None:
    start = label.get("bbox_start")
    end = label.get("bbox_end")
    if not _is_bbox(start) or not _is_bbox(end):
        return None
    ratio = (
        (frame_id - start_frame) / (end_frame - start_frame)
        if end_frame > start_frame
        else 0.0
    )
    return [
        float(first) + ((float(second) - float(first)) * ratio)
        for first, second in zip(start, end, strict=True)
    ]


def _bbox_iou(first: Sequence[Any], second: Sequence[Any]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 4
    )


def _record_confidence(record: Mapping[str, Any]) -> float:
    value = _optional_float(record.get("pose_confidence", record.get("keypoint_quality")))
    return value if value is not None else 0.0


def _window_starts(
    frame_count: int,
    config: GaitWindowPreparationConfig,
) -> list[int]:
    if frame_count <= config.window_frames:
        return [0]
    final_start = frame_count - config.window_frames
    starts = list(range(0, final_start + 1, config.stride_frames))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _rule_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    config = GaitAnalysisConfig(
        window_frames=len(records), min_window_frames=min(5, len(records))
    )
    windows = extract_gait_windows(records, config=config)
    return windows[0] if len(windows) == 1 else None


def _tabular_features(rule: Mapping[str, Any]) -> list[float]:
    gait = rule["gait_stability_features"]
    quality = rule["quality_coverage"]
    frame_count = max(1, int(quality.get("frame_count", 0)))
    values = [float(gait.get(name) or 0.0) for name in GAIT_STABILITY_FEATURE_NAMES]
    values.extend(
        [
            float(quality.get("usable_frame_ratio", 0.0)),
            float(quality.get("gait_keypoint_coverage", 0.0)),
            float(quality.get("mean_core_keypoint_quality", 0.0)),
            float(quality.get("interpolated_point_ratio", 0.0)),
            float(quality.get("jump_outlier_count", 0.0)) / frame_count,
        ]
    )
    return values


def _search_group_split(
    labels: np.ndarray,
    group_ids: np.ndarray,
    config: GaitWindowPreparationConfig,
) -> dict[str, str]:
    normalized_group_ids = group_ids.astype(str)
    groups = sorted(set(normalized_group_ids.tolist()))
    if len(groups) < 6:
        raise ValueError("at least 6 split groups are required for three binary partitions")
    test_fraction = 1.0 - config.train_fraction - config.validation_fraction
    fractions = (config.train_fraction, config.validation_fraction, test_fraction)
    group_counts = {
        group: np.bincount(labels[normalized_group_ids == group], minlength=2)
        for group in groups
    }
    train_count = max(1, int(round(len(groups) * config.train_fraction)))
    validation_count = max(1, int(round(len(groups) * config.validation_fraction)))
    if train_count + validation_count >= len(groups):
        validation_count = max(1, len(groups) - train_count - 1)
    sizes = (
        train_count,
        validation_count,
        len(groups) - train_count - validation_count,
    )

    best: tuple[float, dict[str, str]] | None = None
    for attempt in range(config.split_search_iterations):
        shuffled = groups.copy()
        random.Random(config.seed + (attempt * 1_000_003)).shuffle(shuffled)
        partitions = {
            "train": shuffled[: sizes[0]],
            "validation": shuffled[sizes[0] : sizes[0] + sizes[1]],
            "test": shuffled[sizes[0] + sizes[1] :],
        }
        partition_class_counts = {
            partition: sum(
                (group_counts[group] for group in partition_groups),
                start=np.zeros(2, dtype=np.int64),
            )
            for partition, partition_groups in partitions.items()
        }
        if any(np.any(counts == 0) for counts in partition_class_counts.values()):
            continue
        total_class_counts = np.bincount(labels, minlength=2)
        score = 0.0
        for partition, fraction in zip(_PARTITIONS, fractions, strict=True):
            actual = partition_class_counts[partition] / total_class_counts
            score += float(np.abs(actual - fraction).sum())
        assignments = {
            group: partition
            for partition, partition_groups in partitions.items()
            for group in partition_groups
        }
        if best is None or score < best[0]:
            best = (score, assignments)
            if score <= 1e-12:
                break
    if best is None:
        raise ValueError(
            "unable to create train/validation/test group split with both classes"
        )
    return best[1]


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


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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
