"""Frozen 26-dimensional handcrafted feature schema for wandering RF v1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.topology import (
    compute_topology,
    compute_turn_geometry,
)


FEATURE_SCHEMA_VERSION = "wandering-handcrafted-features-v1"
EPSILON = 1e-8
QUANTILE_METHOD = "linear"
STD_DDOF = 0
REVISIT_INDEX_GAP = 8
NEAREST_FAR_DISTANCE_CLIP = 0.10
QUALITY_CHANNEL_INDEX = 13

FEATURE_NAMES = (
    "path_length",
    "net_displacement",
    "path_efficiency",
    "step_length_mean",
    "step_length_std",
    "step_length_max",
    "heading_resultant_length",
    "abs_turn_mean",
    "abs_turn_std",
    "abs_turn_max",
    "abs_turn_p90",
    "cumulative_abs_turn",
    "turn_direction_coherence",
    "absolute_winding",
    "reversal_count",
    "reversal_rate",
    "reversal_angle_mean_deg",
    "revisit_pair_count",
    "revisit_pair_density",
    "nearest_far_distance_clipped",
    "convex_hull_area",
    "radius_of_gyration",
    "spatial_anisotropy",
    "valid_point_ratio",
    "mean_quality",
    "min_quality",
)

_FEATURE_DEFINITIONS = (
    "sum of valid displacement lengths",
    "Euclidean distance between first and last valid shape points",
    "net_displacement / max(path_length, epsilon)",
    "mean valid displacement length",
    "population standard deviation of valid displacement length",
    "maximum valid displacement length",
    "magnitude of mean valid unit-heading vector",
    "mean absolute valid turn angle",
    "population standard deviation of absolute valid turn angle",
    "maximum absolute valid turn angle",
    "linear 90th percentile of absolute valid turn angle",
    "sum of absolute valid turn angles",
    "abs(sum(turn)) / max(sum(abs(turn)), epsilon)",
    "step-4 topology absolute_winding",
    "number of step-4 reversal events",
    "reversal_count / max(valid turn count, 1)",
    "mean step-4 reversal event angle in degrees, or zero",
    "step-4 topology revisit pair count",
    "revisit count / valid point pairs with j-i >= 8",
    "minimum distance over valid point pairs with j-i >= 8, clipped at 0.10",
    "deterministic monotonic-chain convex hull area",
    "root mean squared distance to valid-point centroid",
    "population covariance eigenvalue anisotropy",
    "valid point count / all point count",
    "mean of the unstandardized quality channel",
    "minimum of the unstandardized quality channel",
)


class HandcraftedFeatureError(ValueError):
    """A ready record cannot produce the exact finite v1 feature vector."""


def feature_schema_document() -> dict[str, Any]:
    """Return the canonical, model-facing feature schema description."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": 26,
        "dtype": "float64",
        "epsilon": EPSILON,
        "quantile_method": QUANTILE_METHOD,
        "std_ddof": STD_DDOF,
        "revisit_index_gap": REVISIT_INDEX_GAP,
        "nearest_far_distance_clip": NEAREST_FAR_DISTANCE_CLIP,
        "source_fields": ["shape_normalized_points", "point_mask", "raw_features", "topology"],
        "forbidden_model_inputs": [
            "source_dataset",
            "sample_id",
            "input_point_count",
            "resampled_source_points",
            "image_normalized_points",
            "absolute_xy_mean",
            "model_features",
            "normalized_dt",
            "time_available",
            "image_canvas",
        ],
        "features": [
            {"index": index, "name": name, "definition": definition}
            for index, (name, definition) in enumerate(
                zip(FEATURE_NAMES, _FEATURE_DEFINITIONS, strict=True), start=1
            )
        ],
    }


def extract_handcrafted_features(record: Mapping[str, Any]) -> np.ndarray:
    """Extract v1 features from one ready step-4 record, never raw TrajectorySample."""

    if not isinstance(record, Mapping):
        raise HandcraftedFeatureError("preprocessed record must be a mapping")
    if record.get("schema_version") != "wandering-preprocessed-sample-v1":
        raise HandcraftedFeatureError("input must be wandering-preprocessed-sample-v1")
    if record.get("preprocess_status") != "ready":
        raise HandcraftedFeatureError("only ready preprocessed records have features")
    points = _numeric_array(record.get("shape_normalized_points"), ndim=2, name="shape_normalized_points")
    mask = _validated_mask(record.get("point_mask"), len(points))
    raw = _numeric_array(record.get("raw_features"), ndim=2, name="raw_features")
    if points.shape != (80, 2) or mask.shape != (80,) or raw.shape != (80, 14):
        raise HandcraftedFeatureError("ready v1 shape/mask/raw_features must be [80,2]/[80]/[80,14]")
    if not np.array_equal(raw[:, 12], mask.astype(np.float64)):
        raise HandcraftedFeatureError("raw feature mask channel does not match point_mask")
    topology = record.get("topology")
    return _extract_feature_vector(points, mask, raw[:, QUALITY_CHANNEL_INDEX], topology)


def extract_features_from_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    point_mask: Sequence[int] | np.ndarray | None = None,
    quality: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Formula fixture helper using step-4 topology defaults on shape points."""

    array = _numeric_array(points, ndim=2, name="points")
    if array.shape[1:] != (2,) or len(array) < 2:
        raise HandcraftedFeatureError("points must have finite shape [N,2], N >= 2")
    mask = (
        np.ones(len(array), dtype=np.int8)
        if point_mask is None
        else _validated_mask(point_mask, len(array))
    )
    quality_array = (
        mask.astype(np.float64)
        if quality is None
        else _numeric_array(quality, ndim=1, name="quality")
    )
    if quality_array.shape != (len(array),):
        raise HandcraftedFeatureError("quality must have shape [N]")
    topology = compute_topology(
        array,
        mask,
        step_epsilon=EPSILON,
        reversal_direction_span=2,
        reversal_angle_deg=120.0,
        reversal_merge_gap=2,
        revisit_index_gap=REVISIT_INDEX_GAP,
        revisit_radius=NEAREST_FAR_DISTANCE_CLIP,
        max_revisit_links=5,
    )
    return _extract_feature_vector(array, mask, quality_array, topology)


def _extract_feature_vector(
    points: np.ndarray,
    point_mask: np.ndarray,
    quality: np.ndarray,
    topology: Any,
) -> np.ndarray:
    if not isinstance(topology, Mapping):
        raise HandcraftedFeatureError("topology must be a mapping")
    array = np.asarray(points, dtype=np.float64)
    mask = np.asarray(point_mask, dtype=bool)
    quality_array = np.asarray(quality, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) != len(mask) or len(array) != len(quality_array):
        raise HandcraftedFeatureError("points, mask, and quality shapes do not align")
    if not np.isfinite(array).all() or not np.isfinite(quality_array).all():
        raise HandcraftedFeatureError("ready data must be finite; imputation is forbidden")
    if np.count_nonzero(mask) < 2:
        raise HandcraftedFeatureError("at least two valid shape points are required")

    geometry = compute_turn_geometry(array, mask.astype(np.int8), step_epsilon=EPSILON)
    displacement = np.zeros_like(array)
    displacement[1:] = array[1:] - array[:-1]
    lengths = np.linalg.norm(displacement, axis=1)
    valid_lengths = lengths[geometry.valid_displacement]
    headings = geometry.headings[geometry.valid_displacement]
    turns = geometry.turn_angles[geometry.valid_turn]
    abs_turns = np.abs(turns)

    path_length = float(valid_lengths.sum())
    valid_indices = np.flatnonzero(mask)
    net_displacement = float(np.linalg.norm(array[valid_indices[-1]] - array[valid_indices[0]]))
    if len(valid_lengths):
        step_mean = float(valid_lengths.mean())
        step_std = float(valid_lengths.std(ddof=STD_DDOF))
        step_max = float(valid_lengths.max())
        heading_resultant = float(
            np.linalg.norm(np.column_stack((np.cos(headings), np.sin(headings))).mean(axis=0))
        )
    else:
        step_mean = step_std = step_max = heading_resultant = 0.0
    if len(abs_turns):
        abs_turn_mean = float(abs_turns.mean())
        abs_turn_std = float(abs_turns.std(ddof=STD_DDOF))
        abs_turn_max = float(abs_turns.max())
        abs_turn_p90 = float(np.quantile(abs_turns, 0.90, method=QUANTILE_METHOD))
        cumulative_abs_turn = float(abs_turns.sum())
        turn_direction_coherence = abs(float(turns.sum())) / max(cumulative_abs_turn, EPSILON)
    else:
        abs_turn_mean = abs_turn_std = abs_turn_max = abs_turn_p90 = 0.0
        cumulative_abs_turn = turn_direction_coherence = 0.0

    absolute_winding = _finite_nonnegative(topology.get("absolute_winding"), "absolute_winding")
    reversal_events = topology.get("reversal_events")
    if not isinstance(reversal_events, Sequence):
        raise HandcraftedFeatureError("topology.reversal_events must be a sequence")
    reversal_angles: list[float] = []
    for event in reversal_events:
        if not isinstance(event, Mapping):
            raise HandcraftedFeatureError("reversal event must be a mapping")
        reversal_angles.append(_finite_nonnegative(event.get("angle_deg"), "reversal angle"))
    reversal_count = len(reversal_angles)
    reversal_rate = reversal_count / max(len(turns), 1)
    reversal_angle_mean = float(np.mean(reversal_angles)) if reversal_angles else 0.0
    revisit_pair_count_value = topology.get("revisit_pair_count")
    if isinstance(revisit_pair_count_value, bool) or not isinstance(revisit_pair_count_value, int) or revisit_pair_count_value < 0:
        raise HandcraftedFeatureError("topology.revisit_pair_count must be a nonnegative integer")
    far_pair_count, nearest_far_distance = _far_pair_statistics(array, mask)
    revisit_pair_density = revisit_pair_count_value / max(far_pair_count, 1)

    valid_points = array[mask]
    hull_area = _convex_hull_area(valid_points)
    centered = valid_points - valid_points.mean(axis=0)
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    spatial_anisotropy = float(
        (eigenvalues[-1] - eigenvalues[0]) / max(float(eigenvalues.sum()), EPSILON)
    )

    values = np.asarray(
        [
            path_length,
            net_displacement,
            net_displacement / max(path_length, EPSILON),
            step_mean,
            step_std,
            step_max,
            heading_resultant,
            abs_turn_mean,
            abs_turn_std,
            abs_turn_max,
            abs_turn_p90,
            cumulative_abs_turn,
            turn_direction_coherence,
            absolute_winding,
            float(reversal_count),
            float(reversal_rate),
            reversal_angle_mean,
            float(revisit_pair_count_value),
            float(revisit_pair_density),
            nearest_far_distance,
            hull_area,
            radius_of_gyration,
            spatial_anisotropy,
            float(mask.mean()),
            float(quality_array.mean()),
            float(quality_array.min()),
        ],
        dtype=np.float64,
    )
    if values.shape != (26,) or not np.isfinite(values).all():
        raise HandcraftedFeatureError("feature extraction produced non-finite or wrong-size output")
    return values


def _far_pair_statistics(points: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    count = 0
    nearest = math.inf
    for left in range(len(points)):
        if not mask[left]:
            continue
        for right in range(left + REVISIT_INDEX_GAP, len(points)):
            if not mask[right]:
                continue
            count += 1
            nearest = min(nearest, float(np.linalg.norm(points[right] - points[left])))
    clipped = NEAREST_FAR_DISTANCE_CLIP if not math.isfinite(nearest) else min(nearest, NEAREST_FAR_DISTANCE_CLIP)
    return count, float(clipped)


def _convex_hull_area(points: np.ndarray) -> float:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    twice_area = sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )
    return abs(float(twice_area)) / 2.0


def _numeric_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HandcraftedFeatureError(f"{name} must be numeric") from exc
    if array.ndim != ndim or not np.isfinite(array).all():
        raise HandcraftedFeatureError(f"{name} must be finite with ndim={ndim}")
    return array


def _validated_mask(value: Any, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or not np.all(np.isin(array, [0, 1])):
        raise HandcraftedFeatureError("point_mask must have shape [N] and values 0/1")
    return array.astype(np.int8, copy=False)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise HandcraftedFeatureError(f"{name} must be finite and nonnegative")
    return float(value)


__all__ = [
    "EPSILON",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "HandcraftedFeatureError",
    "extract_features_from_points",
    "extract_handcrafted_features",
    "feature_schema_document",
]
