"""Lightweight home-development trajectory geometry head and micro-bout policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    WeightedBucket,
    weighted_bucket_observations,
)


GEOMETRY_HEAD_SCHEMA_VERSION = "wandering-camera-geometry-head-v1"
GEOMETRY_HEAD_MODEL_ID = "home-camera-geometry-logistic-v1"
GEOMETRY_FEATURE_NAMES = (
    "log_duration_seconds",
    "log_bucket_count",
    "path_body_heights",
    "path_body_heights_per_second",
    "extent_body_heights",
    "net_body_heights",
    "net_to_path_ratio",
    "closure_to_extent_ratio",
    "minor_major_axis_ratio",
    "pause_ratio_010",
    "pause_ratio_015",
    "pause_ratio_020",
    "pause_ratio_030",
    "reversal_count_per_minute",
    "reversal_step_fraction",
    "mean_absolute_turn_radians",
    "large_turn_fraction",
    "radial_distance_cv",
)
FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")


class CameraGeometryHeadError(ValueError):
    """Geometry artifact or trusted trajectory interval is invalid."""


def load_camera_geometry_head(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraGeometryHeadError("cannot read camera geometry head") from exc
    required = {
        "schema_version",
        "model_id",
        "validation_scope",
        "minimum_track_confidence",
        "bucket_seconds",
        "feature_names",
        "class_order",
        "scaler_mean",
        "scaler_scale",
        "coefficients",
        "intercepts",
        "training_sample_count",
        "training_roles",
        "behavior_truth_consumed_by_training",
        "behavior_truth_consumed_at_inference",
        "cvat_rectangle_consumed_as_tracking_truth",
        "cross_validation",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CameraGeometryHeadError("camera geometry head fields drifted")
    if (
        value["schema_version"] != GEOMETRY_HEAD_SCHEMA_VERSION
        or value["model_id"] != GEOMETRY_HEAD_MODEL_ID
        or value["validation_scope"] != "same_participant_home_development"
        or value["minimum_track_confidence"] != 0.70
        or value["bucket_seconds"] != 0.5
        or value["feature_names"] != list(GEOMETRY_FEATURE_NAMES)
        or set(value["class_order"]) != set(FOUR_CLASS_ORDER)
        or value["behavior_truth_consumed_by_training"] is not True
        or value["behavior_truth_consumed_at_inference"] is not False
        or value["cvat_rectangle_consumed_as_tracking_truth"] is not False
    ):
        raise CameraGeometryHeadError("camera geometry head contract drifted")
    feature_count = len(GEOMETRY_FEATURE_NAMES)
    class_count = len(FOUR_CLASS_ORDER)
    arrays = {
        "scaler_mean": np.asarray(value["scaler_mean"], dtype=np.float64),
        "scaler_scale": np.asarray(value["scaler_scale"], dtype=np.float64),
        "coefficients": np.asarray(value["coefficients"], dtype=np.float64),
        "intercepts": np.asarray(value["intercepts"], dtype=np.float64),
    }
    if (
        arrays["scaler_mean"].shape != (feature_count,)
        or arrays["scaler_scale"].shape != (feature_count,)
        or arrays["coefficients"].shape != (class_count, feature_count)
        or arrays["intercepts"].shape != (class_count,)
        or not all(np.isfinite(array).all() for array in arrays.values())
        or np.any(arrays["scaler_scale"] <= 0.0)
    ):
        raise CameraGeometryHeadError("camera geometry head arrays are invalid")
    return value


def predict_home_camera_geometry(
    adapter: CameraAdapterInput,
    *,
    track_id: int,
    start_sec: float,
    end_sec: float,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict parent geometry and aggregate short-stop motion bouts."""

    parent_features, buckets = _geometry_features_and_buckets(
        adapter,
        track_id=track_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    if parent_features is None:
        raise CameraGeometryHeadError("trusted geometry interval is too short")
    parent = _predict_features(parent_features, artifact)
    bout_intervals = _motion_bout_intervals(
        buckets,
        interval_start=start_sec,
        interval_end=end_sec,
    )
    bouts = []
    for bout_start, bout_end in bout_intervals:
        features, _unused = _geometry_features_and_buckets(
            adapter,
            track_id=track_id,
            start_sec=bout_start,
            end_sec=bout_end,
        )
        if features is None:
            continue
        bouts.append(
            {
                "start_sec": bout_start,
                "end_sec_exclusive": bout_end,
                **_predict_features(features, artifact),
            }
        )
    final = parent
    policy_reason = "parent_behavior_geometry"
    if len(bouts) >= 2:
        direct_probabilities = [
            float(row["probabilities"]["direct"]) for row in bouts
        ]
        direct_vote_ratio = sum(
            row["predicted_pattern"] == "direct" for row in bouts
        ) / len(bouts)
        if (
            direct_vote_ratio >= 0.75
            and float(np.mean(direct_probabilities)) >= 0.55
            and float(parent["probabilities"]["pacing"]) < 0.70
        ):
            averaged = {
                name: float(np.mean([row["probabilities"][name] for row in bouts]))
                for name in FOUR_CLASS_ORDER
            }
            final = {
                "predicted_pattern": "direct",
                "probabilities": _normalize_probabilities(averaged),
            }
            policy_reason = "micro_bout_direct_consensus"
    return {
        "schema_version": "wandering-camera-geometry-prediction-v1",
        "model_id": artifact["model_id"],
        "parent": parent,
        "motion_bouts": bouts,
        "motion_bout_count": len(bouts),
        "aggregation_reason": policy_reason,
        "predicted_pattern": final["predicted_pattern"],
        "probabilities": final["probabilities"],
        "behavior_truth_consumed_at_inference": False,
        "minimum_track_confidence": 0.70,
    }


def _predict_features(
    features: list[float], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    vector = np.asarray(features, dtype=np.float64)
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    intercepts = np.asarray(artifact["intercepts"], dtype=np.float64)
    logits = coefficients @ ((vector - mean) / scale) + intercepts
    logits -= float(logits.max())
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    by_class = {
        str(name): float(value)
        for name, value in zip(artifact["class_order"], probabilities, strict=True)
    }
    return {
        "predicted_pattern": max(by_class, key=by_class.__getitem__),
        "probabilities": {
            name: by_class[name] for name in FOUR_CLASS_ORDER
        },
    }


def _geometry_features_and_buckets(
    adapter: CameraAdapterInput,
    *,
    track_id: int,
    start_sec: float,
    end_sec: float,
) -> tuple[list[float] | None, tuple[WeightedBucket, ...]]:
    selected = tuple(
        row
        for row in adapter.observations
        if row.track_id == track_id
        and start_sec <= row.timestamp_sec < end_sec
        and row.track_confidence >= 0.70
    )
    buckets = weighted_bucket_observations(
        selected,
        minimum_track_confidence=0.70,
        bucket_seconds=0.5,
    )
    if len(buckets) < 3:
        return None, buckets
    height = float(np.median([row.bbox_height for row in buckets]))
    if not math.isfinite(height) or height <= 0.0:
        return None, buckets
    points = np.asarray([row.bbox_bottom_point for row in buckets], dtype=np.float64)
    points = (points - points[0]) / height
    vectors = np.diff(points, axis=0)
    steps = np.linalg.norm(vectors, axis=1)
    path = float(steps.sum())
    duration = max(0.5, float(end_sec - start_sec))
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    extent = float(pairwise.max())
    net = float(np.linalg.norm(points[-1] - points[0]))
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection_steps = np.diff(centered @ major_vector)
    active_projection = projection_steps[np.abs(projection_steps) >= 0.02]
    reversal_count = (
        int(np.sum(np.sign(active_projection[1:]) != np.sign(active_projection[:-1])))
        if len(active_projection) >= 2
        else 0
    )
    active_vectors = vectors[steps >= 0.01]
    turns = []
    for left, right in zip(active_vectors, active_vectors[1:]):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            continue
        turns.append(
            math.acos(float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)))
        )
    turns_array = np.asarray(turns, dtype=np.float64)
    radial = np.linalg.norm(centered, axis=1)
    radial_mean = float(radial.mean())
    features = [
        math.log1p(duration),
        math.log1p(len(buckets)),
        path,
        path / duration,
        extent,
        net,
        net / max(path, 1e-9),
        net / max(extent, 1e-9),
        float(eigenvalues.min() / max(eigenvalues.max(), 1e-9)),
        float(np.mean(steps <= 0.010)),
        float(np.mean(steps <= 0.015)),
        float(np.mean(steps <= 0.020)),
        float(np.mean(steps <= 0.030)),
        reversal_count * 60.0 / duration,
        reversal_count / max(1, len(active_projection) - 1),
        float(turns_array.mean()) if len(turns_array) else 0.0,
        float(np.mean(turns_array >= 2.0)) if len(turns_array) else 0.0,
        float(radial.std() / max(radial_mean, 1e-9)),
    ]
    return features, buckets


def _motion_bout_intervals(
    buckets: tuple[WeightedBucket, ...],
    *,
    interval_start: float,
    interval_end: float,
) -> list[tuple[float, float]]:
    if len(buckets) < 3:
        return []
    height = float(np.median([row.bbox_height for row in buckets]))
    points = np.asarray([row.bbox_bottom_point for row in buckets], dtype=np.float64)
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    projected_steps = np.diff(centered @ major_vector) / height
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) / height
    active = steps > 0.015
    intervals: list[tuple[float, float]] = []
    run_start: int | None = None

    def append_run(run_end: int) -> None:
        nonlocal run_start
        if run_start is None or run_end < run_start:
            run_start = None
            return
        path = float(steps[run_start : run_end + 1].sum())
        start = max(interval_start, buckets[run_start].bucket_index * 0.5)
        end = min(interval_end, (buckets[run_end + 1].bucket_index + 1) * 0.5)
        if path >= 0.06 and end - start >= 1.0:
            intervals.append((start, end))
        run_start = None

    for index, is_active in enumerate(active):
        if is_active and run_start is None:
            run_start = index
        if not is_active:
            append_run(index - 1)
            continue
        next_reverses = (
            index + 1 < len(active)
            and active[index + 1]
            and abs(float(projected_steps[index])) >= 0.02
            and abs(float(projected_steps[index + 1])) >= 0.02
            and np.sign(projected_steps[index])
            != np.sign(projected_steps[index + 1])
        )
        if next_reverses or index == len(active) - 1:
            append_run(index)
    return intervals


def _normalize_probabilities(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(values[name])) for name in FOUR_CLASS_ORDER)
    if total <= 0.0:
        raise CameraGeometryHeadError("geometry probabilities have zero mass")
    return {
        name: max(0.0, float(values[name])) / total for name in FOUR_CLASS_ORDER
    }


__all__ = [
    "CameraGeometryHeadError",
    "GEOMETRY_FEATURE_NAMES",
    "load_camera_geometry_head",
    "predict_home_camera_geometry",
]
