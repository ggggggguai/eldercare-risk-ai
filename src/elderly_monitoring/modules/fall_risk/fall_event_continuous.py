from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.pose import COCO_KEYPOINT_NAMES


FALL_EVENT_CONTINUOUS_JOINTS = COCO_KEYPOINT_NAMES
FALL_EVENT_CONTINUOUS_CHANNELS = (
    "x_body",
    "y_body",
    "motion_x_per_sec",
    "motion_y_per_sec",
    "acceleration_x_per_sec2",
    "acceleration_y_per_sec2",
    "bone_x",
    "bone_y",
    "image_y",
    "bbox_height",
    "quality",
    "valid_mask",
    "interpolated_mask",
    "jump_outlier_mask",
    "frame_mask",
    "delta_t_sec",
    "gap_mask",
    "track_continuity_mask",
    "relative_time_sec",
    "effective_fps",
)

_JOINT_INDEX = {name: index for index, name in enumerate(FALL_EVENT_CONTINUOUS_JOINTS)}
_BONE_PARENTS = (
    0,  # nose is the root
    0,
    0,
    1,
    2,
    0,
    0,
    5,
    6,
    7,
    8,
    5,
    6,
    11,
    12,
    13,
    14,
)
_COORDINATE_SYSTEM = "image_normalized_0_1"
_EPSILON = 1e-6


@dataclass(frozen=True)
class FallEventCausalWindow:
    tensor: np.ndarray
    slot_timestamps_sec: tuple[float, ...]
    source_timestamps_sec: tuple[float | None, ...]
    status: str
    metadata: dict[str, Any]


def build_fall_event_causal_window(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff_time_sec: float,
    window_sec: float,
    target_fps: float,
    max_gap_sec: float,
) -> FallEventCausalWindow:
    """Build one causal 17-joint development window ending at cutoff_time_sec."""

    future_observation_count = sum(
        _required_timestamp(record) > cutoff_time_sec + _EPSILON
        for record in records
    )
    slots, slot_times, source_times = resample_causal_pose_records(
        records,
        cutoff_time_sec=cutoff_time_sec,
        window_sec=window_sec,
        target_fps=target_fps,
        max_gap_sec=max_gap_sec,
    )
    tensor = build_fall_event_continuous_tensor(
        slots,
        slot_timestamps_sec=slot_times,
        cutoff_time_sec=cutoff_time_sec,
        max_gap_sec=max_gap_sec,
    )
    valid = tensor[..., FALL_EVENT_CONTINUOUS_CHANNELS.index("valid_mask")]
    quality = tensor[..., FALL_EVENT_CONTINUOUS_CHANNELS.index("quality")]
    interpolated = tensor[
        ..., FALL_EVENT_CONTINUOUS_CHANNELS.index("interpolated_mask")
    ]
    jump = tensor[..., FALL_EVENT_CONTINUOUS_CHANNELS.index("jump_outlier_mask")]
    frame_mask = tensor[:, 0, FALL_EVENT_CONTINUOUS_CHANNELS.index("frame_mask")]
    continuity = tensor[
        :, 0, FALL_EVENT_CONTINUOUS_CHANNELS.index("track_continuity_mask")
    ]
    valid_count = int(np.sum(valid > 0))
    observed_count = int(np.sum(frame_mask > 0))
    unique_source_times = sorted(
        {float(value) for value in source_times if value is not None}
    )
    source_gaps = [
        right - left
        for left, right in zip(
            unique_source_times, unique_source_times[1:], strict=False
        )
    ]
    metadata = {
        "schema_version": "fall-event-causal-window-v1",
        "status": "synthetic_or_development_infrastructure_only",
        "causal": True,
        "cutoff_time_sec": float(cutoff_time_sec),
        "window_sec": float(window_sec),
        "target_fps": float(target_fps),
        "max_gap_sec": float(max_gap_sec),
        "joint_order": list(FALL_EVENT_CONTINUOUS_JOINTS),
        "channel_order": list(FALL_EVENT_CONTINUOUS_CHANNELS),
        "feature_shape": list(tensor.shape),
        "future_observation_count": int(future_observation_count),
        "observed_frame_count": observed_count,
        "observed_frame_ratio": round(observed_count / len(slots), 6),
        "valid_joint_ratio": round(valid_count / valid.size, 6),
        "mean_joint_quality": round(
            float(np.sum(quality) / valid_count) if valid_count else 0.0, 6
        ),
        "interpolated_joint_ratio": round(
            float(np.sum(interpolated) / valid_count) if valid_count else 0.0,
            6,
        ),
        "jump_outlier_count": int(np.sum(jump > 0)),
        "track_continuity_ratio": round(
            float(np.sum(continuity) / observed_count) if observed_count else 0.0,
            6,
        ),
        "max_source_gap_sec": round(max(source_gaps, default=0.0), 6),
        "limitations": [
            "this window is input-contract infrastructure, not a trained model result",
            "no test truth, threshold, calibration or event metric is consumed",
            "the tensor does not replace FallStateDetector or emit AlgorithmEvent",
        ],
    }
    return FallEventCausalWindow(
        tensor=tensor,
        slot_timestamps_sec=tuple(slot_times),
        source_timestamps_sec=tuple(source_times),
        status="synthetic_or_development_infrastructure_only",
        metadata=metadata,
    )


def resample_causal_pose_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff_time_sec: float,
    window_sec: float,
    target_fps: float,
    max_gap_sec: float,
) -> tuple[list[Mapping[str, Any] | None], list[float], list[float | None]]:
    """Sample the latest observation at or before each target slot."""

    cutoff = _required_finite_positive_or_zero(cutoff_time_sec, "cutoff_time_sec")
    duration = _required_finite_positive(window_sec, "window_sec")
    fps = _required_finite_positive(target_fps, "target_fps")
    max_gap = _required_finite_positive(max_gap_sec, "max_gap_sec")
    window_frames = int(round(duration * fps))
    if window_frames < 2:
        raise ValueError("causal fall-event window must contain at least 2 frames")

    indexed_past: list[tuple[float, int, Mapping[str, Any]]] = []
    target_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        timestamp = _required_timestamp(record)
        if timestamp > cutoff + _EPSILON:
            continue
        target_keys.add(_target_key(record))
        indexed_past.append((timestamp, index, record))
    if len(target_keys) > 1:
        raise ValueError("fall-event causal input contains multiple target tracks")
    indexed_past.sort(key=lambda item: (item[0], item[1]))
    timestamps = [item[0] for item in indexed_past]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("fall-event causal input has duplicate target timestamps")

    slot_times = [
        round(cutoff - (window_frames - 1 - index) / fps, 9)
        for index in range(window_frames)
    ]
    if slot_times[0] < -_EPSILON:
        raise ValueError("insufficient causal history before stream origin")
    slots: list[Mapping[str, Any] | None] = []
    source_times: list[float | None] = []
    source_index = -1
    latest: tuple[float, int, Mapping[str, Any]] | None = None
    for slot_time in slot_times:
        while (
            source_index + 1 < len(indexed_past)
            and indexed_past[source_index + 1][0] <= slot_time + _EPSILON
        ):
            source_index += 1
            latest = indexed_past[source_index]
        if latest is None or slot_time - latest[0] > max_gap + _EPSILON:
            slots.append(None)
            source_times.append(None)
            continue
        slots.append(latest[2])
        source_times.append(latest[0])
    return slots, slot_times, source_times


def build_fall_event_continuous_tensor(
    records: Sequence[Mapping[str, Any] | None],
    *,
    slot_timestamps_sec: Sequence[float],
    cutoff_time_sec: float,
    max_gap_sec: float,
) -> np.ndarray:
    """Convert causal pose slots into the fixed [T,17,20] input contract."""

    if not records or len(records) != len(slot_timestamps_sec):
        raise ValueError("records and slot_timestamps_sec must have the same non-zero length")
    cutoff = _required_finite_positive_or_zero(cutoff_time_sec, "cutoff_time_sec")
    max_gap = _required_finite_positive(max_gap_sec, "max_gap_sec")
    slot_times = [
        _required_finite_positive_or_zero(value, "slot timestamp")
        for value in slot_timestamps_sec
    ]
    if any(right <= left for left, right in zip(slot_times, slot_times[1:], strict=False)):
        raise ValueError("slot timestamps must be strictly increasing")
    if slot_times[-1] > cutoff + _EPSILON:
        raise ValueError("slot timestamp exceeds causal cutoff")

    frame_count = len(records)
    joint_count = len(FALL_EVENT_CONTINUOUS_JOINTS)
    channel_count = len(FALL_EVENT_CONTINUOUS_CHANNELS)
    smooth_xy = np.zeros((frame_count, joint_count, 2), dtype=np.float32)
    raw_xy = np.zeros_like(smooth_xy)
    quality = np.zeros((frame_count, joint_count), dtype=np.float32)
    valid = np.zeros((frame_count, joint_count), dtype=bool)
    interpolated = np.zeros((frame_count, joint_count), dtype=np.float32)
    jump = np.zeros((frame_count, joint_count), dtype=np.float32)
    frame_mask = np.zeros(frame_count, dtype=np.float32)
    bbox_height = np.zeros(frame_count, dtype=np.float32)
    source_times: list[float | None] = [None] * frame_count
    target_keys: list[tuple[str, str] | None] = [None] * frame_count

    for frame_index, record in enumerate(records):
        if record is None:
            continue
        coordinate_system = record.get("coordinate_system", _COORDINATE_SYSTEM)
        if coordinate_system != _COORDINATE_SYSTEM:
            raise ValueError(
                "fall-event continuous tensor requires image_normalized_0_1 coordinates"
            )
        source_time = _required_timestamp(record)
        if source_time > slot_times[frame_index] + _EPSILON:
            raise ValueError("future pose observation assigned to causal slot")
        if source_time > cutoff + _EPSILON:
            raise ValueError("future pose observation exceeds causal cutoff")
        source_times[frame_index] = source_time
        target_keys[frame_index] = _target_key(record)
        frame_mask[frame_index] = 1.0
        bbox = record.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            y1 = _optional_finite_float(bbox[1])
            y2 = _optional_finite_float(bbox[3])
            if y1 is not None and y2 is not None and y2 >= y1:
                bbox_height[frame_index] = float(y2 - y1)
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        for joint_name, joint_index in _JOINT_INDEX.items():
            point = points.get(joint_name)
            if point is None or point.get("valid") is not True:
                continue
            raw_x = _optional_finite_float(point.get("x"))
            raw_y = _optional_finite_float(point.get("y"))
            smooth_x = _optional_finite_float(point.get("x_smooth", raw_x))
            smooth_y = _optional_finite_float(point.get("y_smooth", raw_y))
            point_quality = _optional_finite_float(
                point.get("quality_weight", point.get("score"))
            )
            if None in (raw_x, raw_y, smooth_x, smooth_y, point_quality):
                continue
            assert raw_x is not None
            assert raw_y is not None
            assert smooth_x is not None
            assert smooth_y is not None
            assert point_quality is not None
            raw_xy[frame_index, joint_index] = (raw_x, raw_y)
            smooth_xy[frame_index, joint_index] = (smooth_x, smooth_y)
            quality[frame_index, joint_index] = min(
                1.0, max(0.0, point_quality)
            )
            valid[frame_index, joint_index] = True
            interpolated[frame_index, joint_index] = float(
                point.get("source") == "interpolated"
            )
            jump[frame_index, joint_index] = float(
                point.get("is_jump_outlier") is True
            )

    observed_keys = {key for key in target_keys if key is not None}
    if len(observed_keys) > 1:
        raise ValueError("fall-event continuous tensor contains multiple target tracks")

    left_hip = _JOINT_INDEX["left_hip"]
    right_hip = _JOINT_INDEX["right_hip"]
    left_shoulder = _JOINT_INDEX["left_shoulder"]
    right_shoulder = _JOINT_INDEX["right_shoulder"]
    centered_smooth = np.zeros_like(smooth_xy)
    centered_raw = np.zeros_like(raw_xy)
    usable_valid = valid.copy()
    observed_scales: list[float] = []
    for frame_index in range(frame_count):
        root_valid = valid[frame_index, left_hip] and valid[frame_index, right_hip]
        shoulder_valid = (
            valid[frame_index, left_shoulder]
            and valid[frame_index, right_shoulder]
        )
        if root_valid and shoulder_valid:
            hip_center = (
                smooth_xy[frame_index, left_hip]
                + smooth_xy[frame_index, right_hip]
            ) / 2.0
            shoulder_center = (
                smooth_xy[frame_index, left_shoulder]
                + smooth_xy[frame_index, right_shoulder]
            ) / 2.0
            scale = float(np.linalg.norm(shoulder_center - hip_center))
            if math.isfinite(scale) and scale > _EPSILON:
                observed_scales.append(scale)
        if not root_valid or not observed_scales:
            usable_valid[frame_index] = False
            continue
        causal_scale = float(np.median(observed_scales))
        smooth_hip_center = (
            smooth_xy[frame_index, left_hip]
            + smooth_xy[frame_index, right_hip]
        ) / 2.0
        raw_hip_center = (
            raw_xy[frame_index, left_hip] + raw_xy[frame_index, right_hip]
        ) / 2.0
        centered_smooth[frame_index] = (
            smooth_xy[frame_index] - smooth_hip_center
        ) / causal_scale
        centered_raw[frame_index] = (
            raw_xy[frame_index] - raw_hip_center
        ) / causal_scale
        centered_smooth[frame_index, ~usable_valid[frame_index]] = 0.0
        centered_raw[frame_index, ~usable_valid[frame_index]] = 0.0

    gap_mask = np.zeros(frame_count, dtype=np.float32)
    track_continuity = np.zeros(frame_count, dtype=np.float32)
    source_delta_times = np.zeros(frame_count, dtype=np.float32)
    effective_fps = np.zeros(frame_count, dtype=np.float32)
    previous_source_time: float | None = None
    previous_key: tuple[str, str] | None = None
    for frame_index, source_time in enumerate(source_times):
        if source_time is None:
            gap_mask[frame_index] = 1.0
            continue
        key = target_keys[frame_index]
        if previous_key is None or key == previous_key:
            track_continuity[frame_index] = 1.0
        if (
            previous_source_time is not None
            and source_time > previous_source_time
        ):
            source_delta = source_time - previous_source_time
            source_delta_times[frame_index] = source_delta
            effective_fps[frame_index] = 1.0 / source_delta
            if source_delta > max_gap + _EPSILON:
                gap_mask[frame_index] = 1.0
        previous_source_time = source_time
        previous_key = key

    motion = np.zeros_like(centered_raw)
    motion_valid = np.zeros_like(usable_valid)
    for frame_index in range(1, frame_count):
        delta_t = float(source_delta_times[frame_index])
        if delta_t <= 0:
            continue
        pair_valid = (
            usable_valid[frame_index]
            & usable_valid[frame_index - 1]
            & (jump[frame_index] == 0)
            & (jump[frame_index - 1] == 0)
            & (gap_mask[frame_index] == 0)
            & (track_continuity[frame_index] > 0)
        )
        motion[frame_index, pair_valid] = (
            centered_raw[frame_index, pair_valid]
            - centered_raw[frame_index - 1, pair_valid]
        ) / delta_t
        motion_valid[frame_index] = pair_valid

    acceleration = np.zeros_like(motion)
    for frame_index in range(2, frame_count):
        delta_t = float(source_delta_times[frame_index])
        if delta_t <= 0:
            continue
        acceleration_valid = motion_valid[frame_index] & motion_valid[frame_index - 1]
        acceleration[frame_index, acceleration_valid] = (
            motion[frame_index, acceleration_valid]
            - motion[frame_index - 1, acceleration_valid]
        ) / delta_t

    bones = np.zeros_like(centered_smooth)
    for joint_index, parent_index in enumerate(_BONE_PARENTS):
        if joint_index == parent_index:
            continue
        bone_valid = usable_valid[:, joint_index] & usable_valid[:, parent_index]
        bones[bone_valid, joint_index] = (
            centered_smooth[bone_valid, joint_index]
            - centered_smooth[bone_valid, parent_index]
        )

    tensor = np.zeros(
        (frame_count, joint_count, channel_count), dtype=np.float32
    )
    tensor[..., 0:2] = centered_smooth
    tensor[..., 2:4] = motion
    tensor[..., 4:6] = acceleration
    tensor[..., 6:8] = bones
    tensor[..., 8] = smooth_xy[..., 1]
    tensor[..., 9] = bbox_height[:, None]
    tensor[..., 10] = quality
    tensor[..., 11] = usable_valid.astype(np.float32)
    tensor[..., 12] = interpolated
    tensor[..., 13] = jump
    tensor[..., 14] = frame_mask[:, None]
    tensor[..., 15] = source_delta_times[:, None]
    tensor[..., 16] = gap_mask[:, None]
    tensor[..., 17] = track_continuity[:, None]
    relative_times = np.asarray(slot_times, dtype=np.float32) - float(cutoff)
    tensor[..., 18] = relative_times[:, None]
    tensor[..., 19] = effective_fps[:, None]

    invalid = ~usable_valid
    tensor[..., 0:11][invalid] = 0.0
    tensor[..., 12:14][invalid] = 0.0
    if not np.isfinite(tensor).all():
        raise ValueError("fall-event continuous tensor contains non-finite values")
    return tensor


def _target_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("person_id", "unknown")), str(
        record.get("track_id", "unknown")
    )


def _required_timestamp(record: Mapping[str, Any]) -> float:
    timestamp = _optional_finite_float(record.get("timestamp_sec"))
    if timestamp is None or timestamp < 0:
        raise ValueError("pose record requires a non-negative finite timestamp_sec")
    return timestamp


def _required_finite_positive(value: Any, name: str) -> float:
    result = _optional_finite_float(value)
    if result is None or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _required_finite_positive_or_zero(value: Any, name: str) -> float:
    result = _optional_finite_float(value)
    if result is None or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
