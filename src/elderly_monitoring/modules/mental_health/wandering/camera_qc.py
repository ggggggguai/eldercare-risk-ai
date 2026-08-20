"""Camera-specific segmentation, quality control, and bbox-height compensation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    BBOX_TRACKLET_SCHEMA_VERSION,
    CameraAdapterInput,
    CameraObservation,
    canonical_json_bytes,
    group_observations,
    weighted_bucket_observations,
)


CAMERA_WINDOW_SCHEMA_VERSION = "wandering-camera-window-v1"
MODEL_TRAINING_SPLIT_SHA256 = "4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf"
_SEVERE_BOUNDARY_REASONS = frozenset(
    {"long_internal_gap", "suspected_id_switch", "height_position_discontinuity"}
)


class CameraQCError(ValueError):
    """Camera QC input or fixed-parameter invariants are invalid."""


@dataclass(frozen=True)
class HeightCompensationResult:
    smoothed_heights: np.ndarray
    gains: np.ndarray
    corrected_points: np.ndarray
    point_quality: np.ndarray


@dataclass(frozen=True)
class CameraWindowInput:
    """Internal step-7 input after camera gap handling and compensation."""

    window_record: Mapping[str, Any]
    corrected_points: np.ndarray
    point_quality: np.ndarray


@dataclass(frozen=True)
class CameraQCResult:
    tracklet_records: tuple[Mapping[str, Any], ...]
    window_records: tuple[Mapping[str, Any], ...]
    ready_inputs: tuple[CameraWindowInput, ...]


def rolling_median_valid(values: Sequence[float] | np.ndarray, *, window: int = 5) -> np.ndarray:
    """Centered rolling median with shortened edges and finite positive values."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise CameraQCError("rolling median values must be a non-empty vector")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise CameraQCError("rolling median window must be a positive odd integer")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise CameraQCError("bbox heights must be finite and positive")
    radius = window // 2
    output = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index - radius)
        stop = min(len(array), index + radius + 1)
        output[index] = float(np.median(array[start:stop]))
    return output


def compensate_bbox_height(
    points: Sequence[Sequence[float]] | np.ndarray,
    heights: Sequence[float] | np.ndarray,
    point_quality: Sequence[float] | np.ndarray,
    *,
    rolling_window: int = 5,
    gain_clip: Sequence[float] = (0.5, 2.0),
) -> HeightCompensationResult:
    """Apply the frozen rolling-height local displacement compensation."""

    point_array = np.asarray(points, dtype=np.float64)
    height_array = np.asarray(heights, dtype=np.float64)
    quality = np.asarray(point_quality, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2 or len(point_array) < 2:
        raise CameraQCError("points must be finite shape [N,2], N >= 2")
    if not np.isfinite(point_array).all():
        raise CameraQCError("points must be finite")
    if height_array.shape != (len(point_array),) or quality.shape != (len(point_array),):
        raise CameraQCError("height and quality vectors must align with points")
    if not np.isfinite(quality).all() or np.any((quality < 0.0) | (quality > 1.0)):
        raise CameraQCError("point quality must be finite within [0,1]")
    if (
        not isinstance(gain_clip, Sequence)
        or len(gain_clip) != 2
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in gain_clip)
    ):
        raise CameraQCError("gain_clip must be two finite positive numbers")
    gain_low, gain_high = float(gain_clip[0]), float(gain_clip[1])
    if not math.isfinite(gain_low + gain_high) or not (0.0 < gain_low <= gain_high):
        raise CameraQCError("gain_clip must be ordered and positive")
    smoothed = rolling_median_valid(height_array, window=rolling_window)
    reference = float(np.median(smoothed))
    gains = np.clip(reference / smoothed, gain_low, gain_high)
    corrected = np.empty_like(point_array)
    corrected[0] = point_array[0]
    for index in range(1, len(point_array)):
        corrected[index] = corrected[index - 1] + gains[index] * (
            point_array[index] - point_array[index - 1]
        )
    if not np.isfinite(corrected).all():
        raise CameraQCError("height compensation produced non-finite points")
    return HeightCompensationResult(
        smoothed_heights=smoothed,
        gains=gains,
        corrected_points=corrected,
        point_quality=quality.copy(),
    )


def run_camera_qc(
    adapter_input: CameraAdapterInput,
    config: Mapping[str, Any],
) -> CameraQCResult:
    """Split camera tracks, build global-grid windows, and gate model inputs."""

    if not isinstance(adapter_input, CameraAdapterInput):
        raise CameraQCError("run_camera_qc expects CameraAdapterInput")
    sampling = config.get("sampling")
    qc = config.get("camera_qc")
    compensation = config.get("height_compensation")
    model_input = config.get("model_input")
    trust_roots = config.get("trust_roots")
    if not all(isinstance(item, Mapping) for item in (sampling, qc, compensation, model_input, trust_roots)):
        raise CameraQCError("camera config sections are missing")

    groups = group_observations(adapter_input.observations)
    tracklet_records: list[Mapping[str, Any]] = []
    window_records: list[Mapping[str, Any]] = []
    ready_inputs: list[CameraWindowInput] = []
    for _scope, observations in groups.items():
        segments = _split_observations(observations, sampling=sampling, qc=qc)
        for segment_index, (segment, boundary_flags) in enumerate(segments):
            tracklet = _tracklet_record(
                segment,
                segment_index=segment_index,
                boundary_flags=boundary_flags,
                adapter_input=adapter_input,
                minimum_track_confidence=float(sampling["minimum_track_confidence"]),
            )
            tracklet_records.append(tracklet)
            records, inputs = _window_segment(
                segment,
                tracklet=tracklet,
                boundary_flags=boundary_flags,
                media=adapter_input.media_sidecar,
                config=config,
            )
            window_records.extend(records)
            ready_inputs.extend(inputs)
    tracklet_records.sort(key=lambda row: (str(row["tracklet_id"]), int(row["segment_index"])))
    window_records.sort(key=lambda row: str(row["window_id"]))
    ready_inputs.sort(key=lambda item: str(item.window_record["window_id"]))
    return CameraQCResult(
        tracklet_records=tuple(tracklet_records),
        window_records=tuple(window_records),
        ready_inputs=tuple(ready_inputs),
    )


def _split_observations(
    observations: Sequence[CameraObservation],
    *,
    sampling: Mapping[str, Any],
    qc: Mapping[str, Any],
) -> list[tuple[tuple[CameraObservation, ...], frozenset[str]]]:
    threshold = float(sampling["minimum_track_confidence"])
    bucket_seconds = float(sampling["bucket_seconds"])
    max_gap = int(qc["maximum_internal_gap_buckets"])
    accepted_indices = [index for index, item in enumerate(observations) if item.track_confidence >= threshold]
    if not accepted_indices:
        return [(tuple(observations), frozenset({"insufficient_observed_ratio"}))]
    accepted_heights = np.asarray(
        [observations[index].bbox_height for index in accepted_indices],
        dtype=np.float64,
    )
    smoothed_heights = rolling_median_valid(accepted_heights, window=5)
    boundaries: list[tuple[int, str]] = []
    for accepted_position, (left_index, right_index) in enumerate(
        zip(accepted_indices, accepted_indices[1:], strict=False)
    ):
        left = observations[left_index]
        right = observations[right_index]
        left_bucket = int(math.floor(left.timestamp_sec / bucket_seconds))
        right_bucket = int(math.floor(right.timestamp_sec / bucket_seconds))
        missing_buckets = right_bucket - left_bucket - 1
        if missing_buckets > max_gap:
            boundaries.append((right_index, "long_internal_gap"))
            continue
        dt = right.timestamp_sec - left.timestamp_sec
        if dt <= 0.0:
            raise CameraQCError("validated camera timestamps must remain increasing")
        displacement = float(
            np.linalg.norm(np.asarray(right.bbox_bottom_point) - np.asarray(left.bbox_bottom_point))
        )
        left_height = float(smoothed_heights[accepted_position])
        right_height = float(smoothed_heights[accepted_position + 1])
        median_height = float(np.median([left_height, right_height]))
        body_step = displacement / median_height
        step_rate = body_step / dt
        height_ratio = max(left_height, right_height) / min(left_height, right_height)
        if (
            height_ratio > float(qc["height_ratio_discontinuity"])
            and body_step > float(qc["concurrent_step_body_heights"])
        ):
            boundaries.append((right_index, "height_position_discontinuity"))
        elif step_rate > float(qc["hard_jump_body_heights_per_second"]):
            boundaries.append((right_index, "suspected_id_switch"))

    if not boundaries:
        return [(tuple(observations), frozenset())]
    output: list[tuple[tuple[CameraObservation, ...], frozenset[str]]] = []
    start = 0
    pending_flags: set[str] = set()
    for index, reason in boundaries:
        if index <= start:
            pending_flags.add(reason)
            continue
        output.append((tuple(observations[start:index]), frozenset(pending_flags | {reason})))
        start = index
        pending_flags = {reason}
    if start < len(observations):
        output.append((tuple(observations[start:]), frozenset(pending_flags)))
    return [(segment, flags) for segment, flags in output if segment]


def _tracklet_record(
    observations: Sequence[CameraObservation],
    *,
    segment_index: int,
    boundary_flags: frozenset[str],
    adapter_input: CameraAdapterInput,
    minimum_track_confidence: float,
) -> dict[str, Any]:
    first = observations[0]
    media = adapter_input.media_sidecar
    quality_flags = set(boundary_flags)
    if media["camera_motion_state"] == "not_checked":
        quality_flags.add("camera_motion_not_verified")
    elif media["camera_motion_state"] == "moved":
        quality_flags.add("camera_moved")
    identity = {
        "scope": list(first.scope_key),
        "segment_index": segment_index,
        "first_frame": observations[0].frame_id,
        "last_frame": observations[-1].frame_id,
        "source_tracking_sha256": adapter_input.source_tracking_sha256,
    }
    tracklet_id = "bbox-tracklet-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return {
        "schema_version": BBOX_TRACKLET_SCHEMA_VERSION,
        "tracklet_id": tracklet_id,
        "segment_index": segment_index,
        "source_video_id": first.source_video_id,
        "source_group_id": first.source_group_id,
        "device_id": first.device_id,
        "setup_id": first.setup_id,
        "stream_epoch": first.stream_epoch,
        "track_id": first.track_id,
        "frame_width": media["video_width"],
        "frame_height": media["video_height"],
        "source_fps": media["nominal_fps"],
        "frame_indices": [item.frame_id for item in observations],
        "point_times_sec": [item.timestamp_sec for item in observations],
        "bbox_xyxy_norm": [list(item.bbox_xyxy_norm) for item in observations],
        "bbox_bottom_points": [list(item.bbox_bottom_point) for item in observations],
        "bbox_heights": [item.bbox_height for item in observations],
        "detection_confidence": [item.track_confidence for item in observations],
        "observed_mask": [int(item.track_confidence >= minimum_track_confidence) for item in observations],
        "interpolated_mask": [0 for _ in observations],
        "quality_flags": sorted(quality_flags),
        "media_ref": media["media_ref"],
        "source_sha256": media["source_sha256"],
        "tracking_jsonl_sha256": adapter_input.source_tracking_sha256,
    }


def _window_segment(
    observations: Sequence[CameraObservation],
    *,
    tracklet: Mapping[str, Any],
    boundary_flags: frozenset[str],
    media: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[CameraWindowInput]]:
    sampling = config["sampling"]
    qc = config["camera_qc"]
    compensation = config["height_compensation"]
    model_input = config["model_input"]
    bucket_seconds = float(sampling["bucket_seconds"])
    buckets = weighted_bucket_observations(
        observations,
        minimum_track_confidence=float(sampling["minimum_track_confidence"]),
        bucket_seconds=bucket_seconds,
    )
    by_index = {item.bucket_index: item for item in buckets}
    raw_bucket_indices = [
        int(math.floor(observation.timestamp_sec / bucket_seconds))
        for observation in observations
    ]
    minimum_bucket = min(raw_bucket_indices)
    maximum_bucket = max(raw_bucket_indices)
    stride = int(sampling["stride_buckets"])
    window_size = int(sampling["window_buckets"])
    start = (minimum_bucket // stride) * stride
    starts: list[int] = []
    while start + window_size - 1 <= maximum_bucket:
        starts.append(start)
        start += stride

    records: list[Mapping[str, Any]] = []
    ready_inputs: list[CameraWindowInput] = []
    for start_bucket in starts:
        stop_bucket = start_bucket + window_size
        bucket_values = [by_index.get(index) for index in range(start_bucket, stop_bucket)]
        observed = np.asarray([item is not None for item in bucket_values], dtype=bool)
        observed_count = int(observed.sum())
        raw_detection_count = sum(item.raw_detection_count for item in bucket_values if item is not None)
        confidence_values = [item.mean_detection_confidence for item in bucket_values if item is not None]
        longest_gap_buckets = _longest_false_run(observed)
        reasons: list[str] = []
        if media["camera_motion_state"] == "moved":
            reasons = ["camera_moved"]
        else:
            reasons.extend(sorted(boundary_flags & _SEVERE_BOUNDARY_REASONS))
            if observed_count / window_size < float(qc["minimum_observed_ratio"]):
                reasons.append("insufficient_observed_ratio")
            if raw_detection_count < int(qc["minimum_raw_detections"]):
                reasons.append("too_few_raw_detections")
            if bool(qc["require_observed_window_edges"]) and (not observed[0] or not observed[-1]):
                reasons.append("unobserved_window_edge")
            if longest_gap_buckets > int(qc["maximum_internal_gap_buckets"]):
                reasons.append("long_internal_gap")
        reason_codes = list(dict.fromkeys(reasons))
        window_id = _window_id(
            tracklet_id=str(tracklet["tracklet_id"]),
            segment_index=int(tracklet["segment_index"]),
            start_bucket=start_bucket,
            stop_bucket=stop_bucket,
            tracking_sha256=str(tracklet["tracking_jsonl_sha256"]),
        )
        base = _window_base(
            tracklet=tracklet,
            window_id=window_id,
            start_bucket=start_bucket,
            stop_bucket=stop_bucket,
            bucket_seconds=bucket_seconds,
            raw_detection_count=raw_detection_count,
            observed_count=observed_count,
            longest_gap_buckets=longest_gap_buckets,
            mean_confidence=(float(np.mean(confidence_values)) if confidence_values else 0.0),
            config=config,
        )
        if reason_codes:
            records.append(_unavailable_window(base, reason_codes=reason_codes, quality_flags=tracklet["quality_flags"]))
            continue

        points, heights, interpolated = _interpolate_window(bucket_values, observed)
        quality = np.where(
            observed,
            float(model_input["point_quality_observed"]),
            float(model_input["point_quality_interpolated"]),
        )
        height_result = compensate_bbox_height(
            points,
            heights,
            quality,
            rolling_window=int(compensation["rolling_median_window"]),
            gain_clip=compensation["gain_clip"],
        )
        motion_extent = _motion_extent_body(
            height_result.corrected_points,
            height_result.smoothed_heights,
        )
        if motion_extent < float(qc["minimum_motion_extent_body_heights"]):
            records.append(
                _unavailable_window(
                    base,
                    reason_codes=["insufficient_motion"],
                    quality_flags=tracklet["quality_flags"],
                )
            )
            continue
        ready_record = {
            **base,
            "window_status": "ready",
            "reason_codes": [],
            "quality_flags": list(tracklet["quality_flags"]),
            "interpolated_bucket_count": int(interpolated.sum()),
            "bbox_bottom_points": points.tolist(),
            "bbox_heights": heights.tolist(),
            "smoothed_bbox_heights": height_result.smoothed_heights.tolist(),
            "corrected_points": height_result.corrected_points.tolist(),
            "observed_mask": observed.astype(np.int8).tolist(),
            "interpolated_mask": interpolated.astype(np.int8).tolist(),
            "point_quality": height_result.point_quality.tolist(),
            "shape_normalized_points": None,
            "point_mask": None,
            "raw_features": None,
            "model_features": None,
            "topology": None,
        }
        records.append(ready_record)
        ready_inputs.append(
            CameraWindowInput(
                window_record=ready_record,
                corrected_points=height_result.corrected_points,
                point_quality=height_result.point_quality,
            )
        )
    return records, ready_inputs


def _window_base(
    *,
    tracklet: Mapping[str, Any],
    window_id: str,
    start_bucket: int,
    stop_bucket: int,
    bucket_seconds: float,
    raw_detection_count: int,
    observed_count: int,
    longest_gap_buckets: int,
    mean_confidence: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_WINDOW_SCHEMA_VERSION,
        "window_id": window_id,
        "parent_tracklet_id": tracklet["tracklet_id"],
        "segment_index": tracklet["segment_index"],
        "source_group_id": tracklet["source_group_id"],
        "source_video_id": tracklet["source_video_id"],
        "device_id": tracklet["device_id"],
        "setup_id": tracklet["setup_id"],
        "stream_epoch": tracklet["stream_epoch"],
        "track_id": tracklet["track_id"],
        "window_start_sec": start_bucket * bucket_seconds,
        "window_end_sec": stop_bucket * bucket_seconds,
        "raw_detection_count": raw_detection_count,
        "observed_bucket_count": observed_count,
        "interpolated_bucket_count": 0,
        "observed_ratio": observed_count / int(config["sampling"]["window_buckets"]),
        "longest_gap_seconds": longest_gap_buckets * bucket_seconds,
        "mean_detection_confidence": mean_confidence,
        "preprocessing_config_sha256": config["trust_roots"]["preprocessing_config"]["sha256"],
        "feature_stats_sha256": config["trust_roots"]["preprocessing_feature_stats"]["sha256"],
        "model_training_split_sha256": MODEL_TRAINING_SPLIT_SHA256,
    }


def _unavailable_window(
    base: Mapping[str, Any],
    *,
    reason_codes: Sequence[str],
    quality_flags: Sequence[str],
) -> dict[str, Any]:
    return {
        **base,
        "window_status": "unavailable",
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "quality_flags": list(quality_flags),
        "bbox_bottom_points": None,
        "bbox_heights": None,
        "smoothed_bbox_heights": None,
        "corrected_points": None,
        "observed_mask": None,
        "interpolated_mask": None,
        "point_quality": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "model_features": None,
        "topology": None,
    }


def _interpolate_window(
    buckets: Sequence[Any],
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(buckets)
    points = np.full((count, 2), np.nan, dtype=np.float64)
    heights = np.full(count, np.nan, dtype=np.float64)
    for index, bucket in enumerate(buckets):
        if bucket is not None:
            points[index] = bucket.bbox_bottom_point
            heights[index] = bucket.bbox_height
    interpolated = np.zeros(count, dtype=bool)
    index = 0
    while index < count:
        if observed[index]:
            index += 1
            continue
        start = index
        while index < count and not observed[index]:
            index += 1
        stop = index
        if start == 0 or stop == count:
            raise CameraQCError("window edges cannot be extrapolated")
        gap = stop - start
        left_point, right_point = points[start - 1], points[stop]
        left_height, right_height = heights[start - 1], heights[stop]
        for offset, position in enumerate(range(start, stop), start=1):
            alpha = offset / (gap + 1)
            points[position] = (1.0 - alpha) * left_point + alpha * right_point
            heights[position] = (1.0 - alpha) * left_height + alpha * right_height
            interpolated[position] = True
    if not np.isfinite(points).all() or not np.isfinite(heights).all():
        raise CameraQCError("short-gap interpolation did not produce finite values")
    return points, heights, interpolated


def _motion_extent_body(points: np.ndarray, smoothed_heights: np.ndarray) -> float:
    low = np.quantile(points, 0.05, axis=0, method="linear")
    high = np.quantile(points, 0.95, axis=0, method="linear")
    extent = float(np.linalg.norm(high - low))
    body_height = float(np.median(smoothed_heights))
    if not math.isfinite(extent) or not math.isfinite(body_height) or body_height <= 0.0:
        raise CameraQCError("motion extent inputs are invalid")
    return extent / body_height


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _window_id(
    *,
    tracklet_id: str,
    segment_index: int,
    start_bucket: int,
    stop_bucket: int,
    tracking_sha256: str,
) -> str:
    payload = {
        "parent_tracklet_id": tracklet_id,
        "segment_index": segment_index,
        "start_bucket": start_bucket,
        "stop_bucket": stop_bucket,
        "tracking_sha256": tracking_sha256,
    }
    return "camera-window-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "CAMERA_WINDOW_SCHEMA_VERSION",
    "CameraQCError",
    "CameraQCResult",
    "CameraWindowInput",
    "HeightCompensationResult",
    "compensate_bbox_height",
    "rolling_median_valid",
    "run_camera_qc",
]
