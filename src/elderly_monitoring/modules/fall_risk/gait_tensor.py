from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.gait_contract import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
)


_BASE_JOINT_COUNT = 12


def build_gait_tensor(
    records: Sequence[Mapping[str, Any] | None],
    *,
    window_frames: int,
) -> np.ndarray:
    """Convert cleaned pose records into one normalized [T, V, C] window."""

    if window_frames < 2:
        raise ValueError("window_frames must be at least 2")
    if len(records) > window_frames:
        raise ValueError("records exceed the requested gait window length")

    joint_count = len(CANONICAL_GAIT_JOINTS)
    xy = np.zeros((window_frames, joint_count, 2), dtype=np.float32)
    quality = np.zeros((window_frames, joint_count), dtype=np.float32)
    for frame_index, record in enumerate(records):
        if record is None:
            continue
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        for joint_index, name in enumerate(CANONICAL_GAIT_JOINTS[:_BASE_JOINT_COUNT]):
            point = points.get(name)
            if point is None or point.get("valid") is not True:
                continue
            if point.get("is_jump_outlier") is True:
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

        _derive_center_joint(xy, quality, frame_index, 6, 7, 12)
        _derive_center_joint(xy, quality, frame_index, 0, 1, 13)

    torso_lengths = np.linalg.norm(xy[:, 13] - xy[:, 12], axis=1)
    valid_scale = (
        (quality[:, 12] > 0)
        & (quality[:, 13] > 0)
        & np.isfinite(torso_lengths)
        & (torso_lengths > 1e-6)
    )
    if not valid_scale.any():
        return np.zeros(
            (window_frames, joint_count, len(GAIT_TCN_CHANNELS)),
            dtype=np.float32,
        )
    body_scale = float(np.median(torso_lengths[valid_scale]))
    valid_center = quality[:, 12] > 0
    centered = np.zeros_like(xy)
    centered[valid_center] = (
        xy[valid_center] - xy[valid_center, 12:13]
    ) / body_scale
    centered[quality <= 0] = 0.0
    quality[~valid_center] = 0.0

    delta = np.zeros_like(centered)
    if window_frames > 1:
        delta[1:] = centered[1:] - centered[:-1]
        valid_pairs = (quality[1:] > 0) & (quality[:-1] > 0)
        delta[1:][~valid_pairs] = 0.0
    tensor = np.concatenate([centered, delta, quality[:, :, None]], axis=2)
    if not np.isfinite(tensor).all():
        raise ValueError("gait tensor contains non-finite values")
    return tensor.astype(np.float32, copy=False)


def _derive_center_joint(
    xy: np.ndarray,
    quality: np.ndarray,
    frame_index: int,
    left_index: int,
    right_index: int,
    output_index: int,
) -> None:
    if quality[frame_index, left_index] <= 0 or quality[frame_index, right_index] <= 0:
        return
    xy[frame_index, output_index] = (
        xy[frame_index, left_index] + xy[frame_index, right_index]
    ) / 2.0
    quality[frame_index, output_index] = min(
        quality[frame_index, left_index], quality[frame_index, right_index]
    )


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
