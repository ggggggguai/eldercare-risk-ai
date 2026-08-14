from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.sit_stand_training import SIT_STAND_JOINTS
from elderly_monitoring.modules.fall_risk.sit_stand_development_gate import (
    evaluate_development_localization_gate,
)


SIT_STAND_CONTINUOUS_JOINTS = SIT_STAND_JOINTS
SIT_STAND_CONTINUOUS_CHANNELS = (
    "x_body",
    "y_body",
    "motion_x_per_sec",
    "motion_y_per_sec",
    "quality",
    "image_y",
    "valid_mask",
    "delta_t_sec",
    "frame_mask",
)
_EPSILON = 1e-8


@dataclass(frozen=True)
class SitStandContinuousConfig:
    target_fps: float = 8.0
    context_sec: float = 8.0
    max_gap_sec: float = 0.5
    min_observed_frames: int = 16
    min_partial_observed_frames: int | None = None
    min_valid_joint_ratio: float = 0.5
    target_track_min_coverage: float = 0.6
    target_track_min_dominance: float = 1.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not math.isfinite(self.context_sec) or self.context_sec <= 0:
            raise ValueError("context_sec must be positive")
        if not math.isfinite(self.max_gap_sec) or self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.window_frames < 2:
            raise ValueError("continuous window must contain at least two frames")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        partial_minimum = (
            min(6, self.min_observed_frames)
            if self.min_partial_observed_frames is None
            else self.min_partial_observed_frames
        )
        object.__setattr__(self, "min_partial_observed_frames", partial_minimum)
        if not 1 <= partial_minimum <= self.min_observed_frames:
            raise ValueError(
                "min_partial_observed_frames must be within min_observed_frames"
            )
        if not 0 < self.min_valid_joint_ratio <= 1:
            raise ValueError("min_valid_joint_ratio must be within (0, 1]")
        if not 0 < self.target_track_min_coverage <= 1:
            raise ValueError("target_track_min_coverage must be within (0, 1]")
        if self.target_track_min_dominance <= 1:
            raise ValueError("target_track_min_dominance must be greater than 1")

    @property
    def window_frames(self) -> int:
        return int(round(self.target_fps * self.context_sec))


@dataclass(frozen=True)
class SitStandCausalWindow:
    tensor: np.ndarray
    slot_timestamps_sec: tuple[float, ...]
    source_timestamps_sec: tuple[float | None, ...]
    status: str
    metadata: dict[str, Any]


def build_sit_stand_causal_window(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff_time_sec: float,
    config: SitStandContinuousConfig | None = None,
    required_observed_frames: int | None = None,
) -> SitStandCausalWindow:
    contract = config or SitStandContinuousConfig()
    required_observed = (
        contract.min_observed_frames
        if required_observed_frames is None
        else int(required_observed_frames)
    )
    if not 1 <= required_observed <= contract.min_observed_frames:
        raise ValueError("required_observed_frames is outside the configured range")
    cutoff = _finite_nonnegative(cutoff_time_sec, "cutoff_time_sec")
    future_count = sum(_timestamp(row) > cutoff + _EPSILON for row in records)
    slots, slot_times, source_times, track_switch_count = _resample_causal(
        records, cutoff_time_sec=cutoff, config=contract
    )
    tensor = _build_tensor(
        slots,
        slot_times=slot_times,
        source_times=source_times,
        max_gap_sec=contract.max_gap_sec,
    )
    frame_mask_index = SIT_STAND_CONTINUOUS_CHANNELS.index("frame_mask")
    valid_index = SIT_STAND_CONTINUOUS_CHANNELS.index("valid_mask")
    observed_mask = tensor[:, 0, frame_mask_index] > 0
    observed = int(np.sum(observed_mask))
    valid_ratio = (
        float(np.mean(tensor[observed_mask, :, valid_index] > 0))
        if observed
        else 0.0
    )
    status = "valid"
    reason = None
    if observed < required_observed:
        status = "unavailable"
        reason = "insufficient_observed_frames"
    elif valid_ratio < contract.min_valid_joint_ratio:
        status = "unavailable"
        reason = "insufficient_valid_joint_ratio"
    metadata = {
        "schema_version": "sit-stand-causal-window-v1",
        "status": status,
        "unavailable_reason": reason,
        "causal": True,
        "cutoff_time_sec": cutoff,
        "target_fps": contract.target_fps,
        "context_sec": contract.context_sec,
        "window_frames": contract.window_frames,
        "max_gap_sec": contract.max_gap_sec,
        "joint_order": list(SIT_STAND_CONTINUOUS_JOINTS),
        "channel_order": list(SIT_STAND_CONTINUOUS_CHANNELS),
        "future_observation_count": future_count,
        "observed_frame_count": observed,
        "required_observed_frame_count": required_observed,
        "partial_context": required_observed < contract.min_observed_frames,
        "stitched_track_switch_count": track_switch_count,
        "valid_joint_ratio": round(valid_ratio, 6),
        "valid_joint_ratio_denominator": "observed_frames_only",
        "test_pose_read": False,
    }
    return SitStandCausalWindow(
        tensor=tensor,
        slot_timestamps_sec=tuple(slot_times),
        source_timestamps_sec=tuple(source_times),
        status=status,
        metadata=metadata,
    )


def prepare_sit_stand_continuous_dataset(
    labels: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    *,
    pose_reader: Callable[[str], list[dict[str, Any]]],
    output_dir: Path,
    config: SitStandContinuousConfig | None = None,
    manifest: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contract = config or SitStandContinuousConfig()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"continuous dataset output already exists: {destination}")
    label_rows = sorted(
        [dict(row) for row in labels], key=lambda row: str(row.get("label_id", ""))
    )
    assignment_index = {
        str(row["label_id"]): dict(row)
        for row in assignments
        if isinstance(row.get("label_id"), str)
    }
    manifest_index = {
        str(row["video_id"]): dict(row)
        for row in (manifest or [])
        if isinstance(row.get("video_id"), str)
    }
    tensors: list[np.ndarray] = []
    frame_targets: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    supervision_masks: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    pose_cache: dict[str, list[dict[str, Any]]] = {}
    locked_test = 0
    rejected: Counter[str] = Counter()
    target_selection_counts: Counter[str] = Counter()
    for label in label_rows:
        label_id = str(label.get("label_id", ""))
        assignment = assignment_index.get(label_id)
        if assignment is None:
            raise ValueError(f"missing continuous split assignment for {label_id}")
        partition = str(assignment.get("partition", ""))
        if partition == "test":
            locked_test += 1
            continue
        if partition not in {"train", "validation"}:
            raise ValueError(f"invalid development partition for {label_id}")
        interval_type = str(label.get("interval_type", ""))
        if interval_type == "ignore":
            continue
        if interval_type not in {"event", "explicit_background"}:
            raise ValueError(f"invalid interval_type for {label_id}")
        video_id = str(label.get("video_id", ""))
        if video_id not in pose_cache:
            if len(pose_cache) >= 8:
                pose_cache.pop(next(iter(pose_cache)))
            pose_cache[video_id] = pose_reader(video_id)
        onset = _finite_nonnegative(label.get("onset_time"), f"{label_id}.onset_time")
        cutoff = _finite_nonnegative(label.get("offset_time"), f"{label_id}.offset_time")
        required_observed = _required_observed_frames(
            onset_time_sec=onset,
            offset_time_sec=cutoff,
            config=contract,
        )
        try:
            selected_pose, target_selection = _select_pose_target(
                pose_cache[video_id],
                onset_time_sec=onset,
                cutoff_time_sec=cutoff,
                config=contract,
            )
            window = build_sit_stand_causal_window(
                selected_pose,
                cutoff_time_sec=cutoff,
                config=contract,
                required_observed_frames=required_observed,
            )
        except ValueError as exc:
            rejected[_rejection_reason(exc)] += 1
            continue
        if window.status != "valid":
            rejected[str(window.metadata["unavailable_reason"])] += 1
            continue
        target = (
            1
            if label.get("transition_type") == "sit_to_stand"
            else 2
            if label.get("transition_type") == "stand_to_sit"
            else 0
        )
        tensors.append(window.tensor)
        frame_target, boundary_target, supervision_mask = _build_supervision(
            window,
            onset_time_sec=onset,
            offset_time_sec=cutoff,
            target=target,
        )
        frame_targets.append(frame_target)
        boundary_targets.append(boundary_target)
        supervision_masks.append(supervision_mask)
        target_selection_counts[str(target_selection["policy"])] += 1
        samples.append(
            {
                "sample_id": f"{label_id}:cutoff-{cutoff:.6f}",
                "label_id": label_id,
                "event_id": str(label.get("event_id", label_id)),
                "video_id": video_id,
                "partition": partition,
                "split_group_id": str(assignment.get("split_group_id", "unknown")),
                "interval_type": interval_type,
                "target": target,
                "transition_type": label.get("transition_type"),
                "onset_time_sec": onset,
                "cutoff_time_sec": cutoff,
                "sample_weight": 1.0,
                "observed_frame_count": int(window.metadata["observed_frame_count"]),
                "required_observed_frame_count": int(
                    window.metadata["required_observed_frame_count"]
                ),
                "partial_context": bool(window.metadata["partial_context"]),
                "context_coverage_ratio": round(
                    int(window.metadata["observed_frame_count"])
                    / contract.window_frames,
                    6,
                ),
                "target_selection": target_selection,
            }
        )
    if not samples:
        raise ValueError("no valid train/validation continuous windows were generated")
    event_counts = Counter(str(sample["event_id"]) for sample in samples)
    for sample in samples:
        sample["sample_weight"] = 1.0 / event_counts[str(sample["event_id"])]
    arrays = {
        "features": np.stack(tensors).astype(np.float32),
        "targets": np.asarray([sample["target"] for sample in samples], dtype=np.int64),
        "frame_targets": np.stack(frame_targets).astype(np.int64),
        "boundary_targets": np.stack(boundary_targets).astype(np.float32),
        "supervision_masks": np.stack(supervision_masks).astype(np.float32),
        "sample_weights": np.asarray(
            [sample["sample_weight"] for sample in samples], dtype=np.float32
        ),
        "sample_ids": np.asarray([sample["sample_id"] for sample in samples]),
        "partitions": np.asarray([sample["partition"] for sample in samples]),
    }
    destination.mkdir(parents=True)
    _write_deterministic_npz(destination / "dataset.npz", arrays)
    _write_jsonl(destination / "samples.jsonl", samples)
    development_assignments = sorted(
        (
            row
            for row in assignment_index.values()
            if row.get("partition") in {"train", "validation"}
        ),
        key=lambda row: str(row["label_id"]),
    )
    _write_jsonl(destination / "assignments.jsonl", development_assignments)
    semantic_hash = _arrays_hash(arrays)
    materialized_label_ids = {str(sample["label_id"]) for sample in samples}
    materialized_gate = evaluate_development_localization_gate(
        [
            row
            for row in label_rows
            if str(row.get("label_id")) in materialized_label_ids
        ],
        assignment_index=assignment_index,
        manifest_index=manifest_index,
    )
    metadata = {
        "schema_version": "sit-stand-continuous-dataset-v1",
        "status": "development_provisional",
        "task": "sit_stand_event_localization_v1",
        "sample_count": len(samples),
        "locked_test_label_count": locked_test,
        "test_pose_read": False,
        "test_features_generated": False,
        "test_evaluated": False,
        "joint_order": list(SIT_STAND_CONTINUOUS_JOINTS),
        "channel_order": list(SIT_STAND_CONTINUOUS_CHANNELS),
        "config": {
            "target_fps": contract.target_fps,
            "context_sec": contract.context_sec,
            "window_frames": contract.window_frames,
            "max_gap_sec": contract.max_gap_sec,
        },
        "rejected": dict(sorted(rejected.items())),
        "target_selection_counts": dict(sorted(target_selection_counts.items())),
        "partial_context_sample_count": sum(
            bool(sample["partial_context"]) for sample in samples
        ),
        "semantic_arrays_sha256": semantic_hash,
        "materialized_development_gate": materialized_gate,
        "continuous_model_training_allowed": bool(materialized_gate["passed"]),
    }
    _write_json(destination / "metadata.json", metadata)
    _write_json(destination / "preparation_report.json", metadata)
    _write_json(
        destination / "split.json",
        {
            "schema_version": "sit-stand-continuous-derived-split-v1",
            "status": "development_provisional",
            "test_policy": "count_only_pose_not_opened",
            "partition_counts": dict(
                sorted(Counter(sample["partition"] for sample in samples).items())
            ),
        },
    )
    return metadata


def _resample_causal(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff_time_sec: float,
    config: SitStandContinuousConfig,
) -> tuple[
    list[Mapping[str, Any] | None],
    list[float],
    list[float | None],
    int,
]:
    slot_times = [
        round(
            cutoff_time_sec - (config.window_frames - 1 - index) / config.target_fps,
            9,
        )
        for index in range(config.window_frames)
    ]
    earliest_relevant = slot_times[0] - config.max_gap_sec
    indexed: list[tuple[float, int, Mapping[str, Any]]] = []
    targets: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        timestamp = _timestamp(record)
        if timestamp > cutoff_time_sec + _EPSILON or timestamp < earliest_relevant:
            continue
        indexed.append((timestamp, index, record))
        targets.add((str(record.get("person_id", "")), str(record.get("track_id", ""))))
    indexed.sort(key=lambda item: (item[0], item[1]))
    timestamps = [item[0] for item in indexed]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("sit-stand causal input has duplicate target timestamps")
    track_switch_count = _validated_track_switch_count(
        indexed, max_gap_sec=config.max_gap_sec
    )
    slots: list[Mapping[str, Any] | None] = []
    source_times: list[float | None] = []
    cursor = -1
    latest: tuple[float, int, Mapping[str, Any]] | None = None
    for slot_time in slot_times:
        while cursor + 1 < len(indexed) and indexed[cursor + 1][0] <= slot_time + _EPSILON:
            cursor += 1
            latest = indexed[cursor]
        if latest is None or slot_time - latest[0] > config.max_gap_sec + _EPSILON:
            slots.append(None)
            source_times.append(None)
        else:
            slots.append(latest[2])
            source_times.append(latest[0])
    return slots, slot_times, source_times, track_switch_count


def _required_observed_frames(
    *,
    onset_time_sec: float,
    offset_time_sec: float,
    config: SitStandContinuousConfig,
) -> int:
    duration = max(0.0, offset_time_sec - onset_time_sec)
    interval_frames = max(1, int(math.floor(duration * config.target_fps + _EPSILON)))
    return min(
        config.min_observed_frames,
        max(config.min_partial_observed_frames, interval_frames),
    )


def _build_supervision(
    window: SitStandCausalWindow,
    *,
    onset_time_sec: float,
    offset_time_sec: float,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(window.slot_timestamps_sec, dtype=np.float64)
    frame_mask_index = SIT_STAND_CONTINUOUS_CHANNELS.index("frame_mask")
    observed = window.tensor[:, 0, frame_mask_index] > 0
    supervised = (
        observed
        & (timestamps >= onset_time_sec - _EPSILON)
        & (timestamps <= offset_time_sec + _EPSILON)
    )
    frame_target = np.full(len(timestamps), -100, dtype=np.int64)
    frame_target[supervised] = target
    boundary_target = np.zeros((len(timestamps), 2), dtype=np.float32)
    indices = np.flatnonzero(supervised)
    if target in {1, 2} and len(indices):
        boundary_target[int(indices[0]), 0] = 1.0
        boundary_target[int(indices[-1]), 1] = 1.0
    return frame_target, boundary_target, supervised.astype(np.float32)


def _select_pose_target(
    records: Sequence[Mapping[str, Any]],
    *,
    onset_time_sec: float,
    cutoff_time_sec: float,
    config: SitStandContinuousConfig,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    earliest = cutoff_time_sec - config.context_sec - config.max_gap_sec
    relevant = [
        row
        for row in records
        if earliest - _EPSILON <= _timestamp(row) <= cutoff_time_sec + _EPSILON
    ]
    timestamp_counts = Counter(_timestamp(row) for row in relevant)
    if not any(count > 1 for count in timestamp_counts.values()):
        return list(records), {
            "policy": "strict_single_observation",
            "candidate_track_count": len({_track_identity(row) for row in relevant}),
        }

    interval_records = [
        row
        for row in relevant
        if onset_time_sec - _EPSILON <= _timestamp(row) <= cutoff_time_sec + _EPSILON
    ]
    by_track: dict[tuple[str, str], set[float]] = defaultdict(set)
    for row in interval_records:
        by_track[_track_identity(row)].add(_timestamp(row))
    ranked = sorted(
        ((len(timestamps), identity) for identity, timestamps in by_track.items()),
        key=lambda item: (-item[0], item[1]),
    )
    unique_timestamps = len({_timestamp(row) for row in interval_records})
    if not ranked or unique_timestamps == 0:
        raise ValueError("sit-stand pose target is ambiguous")
    top_count, top_identity = ranked[0]
    runner_count = ranked[1][0] if len(ranked) > 1 else 0
    coverage = top_count / unique_timestamps
    dominance = top_count / runner_count if runner_count else float("inf")
    if (
        top_count < config.min_partial_observed_frames
        or coverage < config.target_track_min_coverage
        or dominance < config.target_track_min_dominance
    ):
        raise ValueError("sit-stand pose target is ambiguous")
    selected = [row for row in records if _track_identity(row) == top_identity]
    return selected, {
        "policy": "dominant_track",
        "candidate_track_count": len(ranked),
        "selected_person_id": top_identity[0],
        "selected_track_id": top_identity[1],
        "interval_timestamp_coverage": round(coverage, 6),
        "dominance_ratio": (
            round(dominance, 6) if math.isfinite(dominance) else None
        ),
    }


def _validated_track_switch_count(
    indexed: Sequence[tuple[float, int, Mapping[str, Any]]],
    *,
    max_gap_sec: float,
) -> int:
    switches = 0
    previous: tuple[float, int, Mapping[str, Any]] | None = None
    for current in indexed:
        if previous is not None and _track_identity(previous[2]) != _track_identity(current[2]):
            gap = current[0] - previous[0]
            left = _bbox_center(previous[2].get("bbox"))
            right = _bbox_center(current[2].get("bbox"))
            if (
                gap > max_gap_sec + _EPSILON
                or left is None
                or right is None
                or math.dist(left, right) > 0.25
            ):
                raise ValueError("sit-stand causal input contains multiple target tracks")
            switches += 1
        previous = current
    return switches


def _track_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("person_id", "")), str(record.get("track_id", ""))


def _bbox_center(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    coordinates = [_optional_float(item) for item in value]
    if any(item is None for item in coordinates):
        return None
    left, top, right, bottom = (float(item) for item in coordinates)
    return (left + right) / 2.0, (top + bottom) / 2.0


def _build_tensor(
    slots: Sequence[Mapping[str, Any] | None],
    *,
    slot_times: Sequence[float],
    source_times: Sequence[float | None],
    max_gap_sec: float,
) -> np.ndarray:
    frame_count = len(slots)
    joint_count = len(SIT_STAND_CONTINUOUS_JOINTS)
    xy = np.zeros((frame_count, joint_count, 2), dtype=np.float32)
    quality = np.zeros((frame_count, joint_count), dtype=np.float32)
    valid = np.zeros((frame_count, joint_count), dtype=bool)
    for frame_index, record in enumerate(slots):
        if record is None:
            continue
        if record.get("coordinate_system", "image_normalized_0_1") != "image_normalized_0_1":
            raise ValueError("sit-stand continuous input requires normalized coordinates")
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        for joint_index, name in enumerate(SIT_STAND_CONTINUOUS_JOINTS[:12]):
            point = points.get(name)
            if point is None or point.get("valid") is not True:
                continue
            if point.get("is_jump_outlier") is True:
                continue
            x = _optional_float(point.get("x_smooth", point.get("x")))
            y = _optional_float(point.get("y_smooth", point.get("y")))
            score = _optional_float(point.get("quality_weight", point.get("score")))
            if x is None or y is None or score is None or score <= 0:
                continue
            xy[frame_index, joint_index] = (x, y)
            quality[frame_index, joint_index] = min(1.0, max(0.0, score))
            valid[frame_index, joint_index] = True
        _derive_center(xy, quality, valid, frame_index, 6, 7, 12)
        _derive_center(xy, quality, valid, frame_index, 0, 1, 13)
    torso = np.linalg.norm(xy[:, 13] - xy[:, 12], axis=1)
    valid_scale = valid[:, 12] & valid[:, 13] & (torso > 1e-6)
    scale = float(np.median(torso[valid_scale])) if valid_scale.any() else 1.0
    centered = np.zeros_like(xy)
    for frame_index in range(frame_count):
        if not valid[frame_index, 12]:
            valid[frame_index] = False
            quality[frame_index] = 0.0
            continue
        centered[frame_index] = (
            xy[frame_index] - xy[frame_index, 12:13]
        ) / scale
        centered[frame_index, ~valid[frame_index]] = 0.0
    motion = np.zeros_like(centered)
    delta_t = np.zeros(frame_count, dtype=np.float32)
    for index in range(1, frame_count):
        left_time = source_times[index - 1]
        right_time = source_times[index]
        if left_time is None or right_time is None or right_time <= left_time:
            continue
        dt = right_time - left_time
        if dt > max_gap_sec + _EPSILON:
            continue
        pairs = valid[index] & valid[index - 1]
        motion[index, pairs] = (centered[index, pairs] - centered[index - 1, pairs]) / dt
        delta_t[index] = dt
    tensor = np.zeros(
        (frame_count, joint_count, len(SIT_STAND_CONTINUOUS_CHANNELS)),
        dtype=np.float32,
    )
    tensor[..., 0:2] = centered
    tensor[..., 2:4] = motion
    tensor[..., 4] = quality
    tensor[..., 5] = np.where(valid, xy[..., 1], 0.0)
    tensor[..., 6] = valid.astype(np.float32)
    tensor[..., 7] = delta_t[:, None]
    tensor[..., 8] = np.asarray([slot is not None for slot in slots], dtype=np.float32)[:, None]
    if not np.isfinite(tensor).all():
        raise ValueError("sit-stand continuous tensor contains non-finite values")
    return tensor


def _derive_center(
    xy: np.ndarray,
    quality: np.ndarray,
    valid: np.ndarray,
    frame: int,
    left: int,
    right: int,
    output: int,
) -> None:
    if valid[frame, left] and valid[frame, right]:
        xy[frame, output] = (xy[frame, left] + xy[frame, right]) / 2.0
        quality[frame, output] = min(quality[frame, left], quality[frame, right])
        valid[frame, output] = True


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, array in sorted(arrays.items()):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _arrays_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        value = np.asarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _timestamp(record: Mapping[str, Any]) -> float:
    return _finite_nonnegative(record.get("timestamp_sec"), "timestamp_sec")


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rejection_reason(error: ValueError) -> str:
    message = str(error)
    known = {
        "sit-stand causal input contains multiple target tracks": "multiple_target_tracks",
        "sit-stand causal input has duplicate target timestamps": "duplicate_timestamps",
        "insufficient causal history before stream origin": "insufficient_causal_history",
        "sit-stand pose target is ambiguous": "ambiguous_target_tracks",
    }
    return known.get(message, "invalid_pose_contract")
