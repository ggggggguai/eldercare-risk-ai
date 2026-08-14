"""Frozen public-trajectory preprocessing bundle for wandering step 4.

The builder in this module consumes only the two reviewed development JSONL
files and the frozen step-3 split/assignments.  It never opens SmartCare's
official validation JSONL and it does not connect to camera or model code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.anchorless_normalization import (
    AnchorlessNormalizationError,
    robust_isotropic_normalize,
)
from elderly_monitoring.modules.mental_health.wandering.datasets import (
    TrajectoryDatasetError,
    load_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.manifests import (
    ManifestValidationError,
    SplitManifest,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import TrajectorySample
from elderly_monitoring.modules.mental_health.wandering.topology import (
    TopologyError,
    compute_topology,
    compute_turn_geometry,
)


PREPROCESSING_CONFIG_SCHEMA_VERSION = "wandering-preprocessing-config-v1"
PREPROCESSED_SAMPLE_SCHEMA_VERSION = "wandering-preprocessed-sample-v1"
FEATURE_STATS_SCHEMA_VERSION = "wandering-feature-stats-v1"
PREPROCESSING_REPORT_SCHEMA_VERSION = "wandering-preprocessing-report-v1"
PREPROCESSING_MANIFEST_SCHEMA_VERSION = "wandering-preprocessing-manifest-v1"

_INPUT_ROLES = (
    "wandering_patterns_samples",
    "smartcare_train_pool",
    "split_json",
    "split_sha256_file",
    "assignments",
    "split_config",
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "inputs",
        "split_sha256",
        "target_points",
        "min_valid_points",
        "max_index_gap",
        "interpolated_quality",
        "quantile_method",
        "scale_quantiles",
        "scale_epsilon",
        "step_epsilon",
        "iqr_epsilon",
        "numeric_tolerance",
        "temporal_features_enabled",
        "feature_channels",
        "topology",
        "diagnostics",
        "output_schemas",
    }
)
_INPUT_DESCRIPTOR_FIELDS = frozenset({"path", "sha256"})
_TOPOLOGY_FIELDS = frozenset(
    {
        "reversal_direction_span",
        "reversal_angle_deg",
        "reversal_merge_gap",
        "revisit_index_gap",
        "revisit_radius",
        "max_revisit_links",
    }
)
_OUTPUT_SCHEMA_FIELDS = frozenset(
    {"samples", "feature_stats", "preprocessing_report", "manifest"}
)
_FEATURE_CHANNELS = (
    "x",
    "y",
    "dx",
    "dy",
    "step_length",
    "sin_heading",
    "cos_heading",
    "sin_turn",
    "cos_turn",
    "abs_curvature",
    "normalized_dt",
    "time_available",
    "point_mask",
    "quality_score",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSIGNMENT_SCHEMA_VERSION = "wandering-split-assignments-v1"
_MACHINE_FILES = (
    "samples.jsonl",
    "feature_stats.json",
    "preprocessing_report.json",
    "manifest.json",
)


class PreprocessingDataError(ValueError):
    """Invalid configuration, frozen input, or bundle invariant."""


class PreprocessingConfigError(PreprocessingDataError):
    """The strict step-4 configuration is invalid."""


class InputHashMismatchError(PreprocessingDataError):
    """A frozen source/split file or canonical split digest has drifted."""


class PreprocessingUnavailable(PreprocessingDataError):
    """A legal sample cannot safely pass the quality gate."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        valid_input_point_count: int,
        trimmed_edge_point_count: int = 0,
        interpolated_input_point_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.valid_input_point_count = valid_input_point_count
        self.trimmed_edge_point_count = trimmed_edge_point_count
        self.interpolated_input_point_count = interpolated_input_point_count


@dataclass(frozen=True)
class PreprocessingInput:
    """Internal, non-persistent input after source-specific gap handling."""

    sample_id: str
    source_dataset: str
    input_role: str
    split: str
    binary_label: int | None
    pattern_label: str
    binary_supervision_eligible: bool
    pattern_supervision_eligible: bool
    coordinate_system: str
    points: np.ndarray
    point_mask: np.ndarray
    point_times_sec: np.ndarray | None
    point_quality: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise PreprocessingDataError("sample_id must be a non-empty string")
        if self.source_dataset not in ("wandering_patterns", "smartcare"):
            raise PreprocessingDataError("unsupported public source_dataset")
        if self.input_role not in ("samples", "train_pool"):
            raise PreprocessingDataError("unsupported public input_role")
        if self.split not in ("train", "validation", "test"):
            raise PreprocessingDataError("preprocessing input split must be train/validation/test")
        if self.coordinate_system not in ("source_native", "image_normalized"):
            raise PreprocessingDataError("unsupported public coordinate_system")
        points, mask, quality = _validated_points_mask_quality(
            self.points,
            self.point_mask,
            self.point_quality,
        )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "point_mask", mask)
        object.__setattr__(self, "point_quality", quality)
        if self.point_times_sec is not None:
            times = np.asarray(self.point_times_sec, dtype=np.float64)
            if times.ndim != 1 or len(times) != len(points) or not np.isfinite(times).all():
                raise PreprocessingDataError("point_times_sec must be null or finite shape [N]")
            if np.any(np.diff(times) <= 0.0):
                raise PreprocessingDataError("point_times_sec must be strictly increasing")
            object.__setattr__(self, "point_times_sec", times)


@dataclass(frozen=True)
class RepairedPublicTrajectory:
    points: np.ndarray
    quality: np.ndarray
    valid_input_point_count: int
    trimmed_edge_point_count: int
    interpolated_input_point_count: int


@dataclass(frozen=True)
class ArcLengthResampleResult:
    points: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class WanderingPreprocessingBuildResult:
    output_dir: Path
    report: Mapping[str, Any]
    manifest: Mapping[str, Any]


def load_preprocessing_config(path: str | Path) -> dict[str, Any]:
    """Load the exact step-4 config without accepting silent fields or drift."""

    config_path = Path(path)
    if not config_path.is_file():
        raise PreprocessingConfigError(f"preprocessing config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PreprocessingConfigError(f"cannot read preprocessing config: {config_path}") from exc
    _require_exact_fields(raw, _CONFIG_FIELDS, "preprocessing config")
    if raw["schema_version"] != PREPROCESSING_CONFIG_SCHEMA_VERSION:
        raise PreprocessingConfigError(
            f"schema_version must be {PREPROCESSING_CONFIG_SCHEMA_VERSION!r}"
        )
    _validate_inputs_config(raw["inputs"])
    _validate_sha256(raw["split_sha256"], "split_sha256")
    _require_int(raw["target_points"], "target_points", expected=80)
    _require_int(raw["min_valid_points"], "min_valid_points", expected=8)
    _require_int(raw["max_index_gap"], "max_index_gap", expected=2)
    _require_float(raw["interpolated_quality"], "interpolated_quality", expected=0.5)
    if raw["quantile_method"] != "linear":
        raise PreprocessingConfigError("quantile_method must remain 'linear'")
    if raw["scale_quantiles"] != [0.05, 0.95]:
        raise PreprocessingConfigError("scale_quantiles must remain [0.05, 0.95]")
    for field, expected in (
        ("scale_epsilon", 1e-8),
        ("step_epsilon", 1e-8),
        ("iqr_epsilon", 1e-8),
        ("numeric_tolerance", 1e-6),
    ):
        _require_float(raw[field], field, expected=expected)
    if raw["temporal_features_enabled"] is not False:
        raise PreprocessingConfigError("temporal_features_enabled must remain false")
    if raw["feature_channels"] != list(_FEATURE_CHANNELS):
        raise PreprocessingConfigError("feature_channels order has drifted")
    _validate_topology_config(raw["topology"])
    if raw["diagnostics"] != {"seed": 20260803, "samples_per_class": 8}:
        raise PreprocessingConfigError("diagnostics must remain seed=20260803 and 8 per class")
    _require_exact_fields(raw["output_schemas"], _OUTPUT_SCHEMA_FIELDS, "output_schemas")
    expected_schemas = {
        "samples": PREPROCESSED_SAMPLE_SCHEMA_VERSION,
        "feature_stats": FEATURE_STATS_SCHEMA_VERSION,
        "preprocessing_report": PREPROCESSING_REPORT_SCHEMA_VERSION,
        "manifest": PREPROCESSING_MANIFEST_SCHEMA_VERSION,
    }
    if raw["output_schemas"] != expected_schemas:
        raise PreprocessingConfigError("output schema versions have drifted")
    return raw


def repair_public_index_gaps(
    points: np.ndarray,
    point_mask: np.ndarray,
    *,
    min_valid_points: int = 8,
    max_index_gap: int = 2,
    interpolated_quality: float = 0.5,
) -> RepairedPublicTrajectory:
    """Trim unobserved edges and repair only internal gaps of at most two indices."""

    array, mask, _quality = _validated_points_mask_quality(
        points,
        point_mask,
        np.asarray(point_mask, dtype=np.float64),
    )
    valid_count = int(mask.sum())
    if valid_count < min_valid_points:
        raise PreprocessingUnavailable(
            "too_few_valid_points",
            f"requires at least {min_valid_points} original observations, got {valid_count}",
            valid_input_point_count=valid_count,
        )
    valid_indices = np.flatnonzero(mask)
    start = int(valid_indices[0])
    stop = int(valid_indices[-1]) + 1
    trimmed_count = start + (len(mask) - stop)
    repaired_points = array[start:stop].copy()
    repaired_mask = mask[start:stop].copy()
    quality = repaired_mask.astype(np.float64)
    interpolated_count = 0
    index = 0
    while index < len(repaired_mask):
        if repaired_mask[index]:
            index += 1
            continue
        gap_start = index
        while index < len(repaired_mask) and not repaired_mask[index]:
            index += 1
        gap_stop = index
        gap_length = gap_stop - gap_start
        if gap_length > max_index_gap:
            raise PreprocessingUnavailable(
                "unresolved_internal_gap",
                f"internal index gap length {gap_length} exceeds {max_index_gap}",
                valid_input_point_count=valid_count,
                trimmed_edge_point_count=trimmed_count,
                interpolated_input_point_count=interpolated_count,
            )
        left = repaired_points[gap_start - 1]
        right = repaired_points[gap_stop]
        for offset, missing_index in enumerate(range(gap_start, gap_stop), start=1):
            alpha = offset / (gap_length + 1)
            repaired_points[missing_index] = (1.0 - alpha) * left + alpha * right
            quality[missing_index] = interpolated_quality
            repaired_mask[missing_index] = True
            interpolated_count += 1
    return RepairedPublicTrajectory(
        points=repaired_points,
        quality=quality,
        valid_input_point_count=valid_count,
        trimmed_edge_point_count=trimmed_count,
        interpolated_input_point_count=interpolated_count,
    )


def arc_length_resample(
    points: np.ndarray,
    point_quality: np.ndarray,
    *,
    target_points: int = 80,
    step_epsilon: float = 1e-8,
) -> ArcLengthResampleResult:
    """Resample by cumulative arc length and propagate minimum endpoint quality."""

    array = np.asarray(points, dtype=np.float64)
    quality = np.asarray(point_quality, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise PreprocessingDataError("points must have shape [N,2] with N >= 2")
    if quality.ndim != 1 or len(quality) != len(array):
        raise PreprocessingDataError("point_quality must have shape [N]")
    if not np.isfinite(array).all() or not np.isfinite(quality).all():
        raise PreprocessingDataError("resampling inputs must be finite")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise PreprocessingDataError("point_quality must be within [0,1]")
    if isinstance(target_points, bool) or not isinstance(target_points, int) or target_points < 2:
        raise PreprocessingDataError("target_points must be an integer >= 2")
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= float(step_epsilon):
        raise PreprocessingUnavailable(
            "insufficient_motion",
            "trajectory arc length is degenerate",
            valid_input_point_count=len(array),
        )
    keep = np.concatenate(([True], np.diff(cumulative) > float(step_epsilon)))
    unique_distances = cumulative[keep]
    unique_points = array[keep]
    unique_quality = quality[keep]
    targets = np.linspace(0.0, total_length, target_points, dtype=np.float64)
    resampled_points = np.column_stack(
        (
            np.interp(targets, unique_distances, unique_points[:, 0]),
            np.interp(targets, unique_distances, unique_points[:, 1]),
        )
    )
    resampled_quality = np.empty(target_points, dtype=np.float64)
    for target_index, target in enumerate(targets):
        if target_index == 0:
            resampled_quality[target_index] = unique_quality[0]
            continue
        if target_index == target_points - 1:
            resampled_quality[target_index] = unique_quality[-1]
            continue
        right = int(np.searchsorted(unique_distances, target, side="right"))
        right = min(max(right, 1), len(unique_distances) - 1)
        left = right - 1
        resampled_quality[target_index] = min(
            float(unique_quality[left]),
            float(unique_quality[right]),
        )
    return ArcLengthResampleResult(points=resampled_points, quality=resampled_quality)


def build_raw_features(
    shape_points: np.ndarray,
    point_mask: np.ndarray,
    point_quality: np.ndarray,
    *,
    point_times_sec: np.ndarray | None,
    step_epsilon: float = 1e-8,
    temporal_features_enabled: bool = False,
) -> np.ndarray:
    """Construct the exact 14 channels and clear every masked position."""

    points, mask, quality = _validated_points_mask_quality(
        shape_points,
        point_mask,
        point_quality,
    )
    count = len(points)
    features = np.zeros((count, 14), dtype=np.float64)
    features[:, 0:2] = points
    geometry = compute_turn_geometry(points, mask, step_epsilon=step_epsilon)
    displacements = np.zeros_like(points)
    if count > 1:
        displacements[1:] = points[1:] - points[:-1]
    for index in range(1, count):
        if not geometry.valid_displacement[index]:
            continue
        dx, dy = displacements[index]
        features[index, 2] = dx
        features[index, 3] = dy
        features[index, 4] = math.hypot(float(dx), float(dy))
        features[index, 5] = math.sin(float(geometry.headings[index]))
        features[index, 6] = math.cos(float(geometry.headings[index]))
    for index in range(2, count):
        if not geometry.valid_turn[index]:
            continue
        turn = float(geometry.turn_angles[index])
        features[index, 7] = math.sin(turn)
        features[index, 8] = math.cos(turn)
        features[index, 9] = abs(turn)

    if temporal_features_enabled:
        if point_times_sec is None:
            pass
        else:
            times = np.asarray(point_times_sec, dtype=np.float64)
            if times.ndim != 1 or len(times) != count or not np.isfinite(times).all():
                raise PreprocessingDataError("point_times_sec must be finite shape [N]")
            intervals = np.diff(times)
            if np.any(intervals <= 0.0):
                raise PreprocessingDataError("point_times_sec must be strictly increasing")
            reliable = mask[1:] & mask[:-1]
            positive = intervals[reliable]
            if len(positive):
                median_positive = float(np.median(positive))
                for index in range(1, count):
                    if reliable[index - 1]:
                        features[index, 10] = float(
                            np.clip(intervals[index - 1] / median_positive, 0.0, 4.0)
                        )
                        features[index, 11] = 1.0
    features[:, 12] = mask.astype(np.float64)
    features[:, 13] = quality
    features[~mask] = 0.0
    if not np.isfinite(features).all():
        raise PreprocessingDataError("raw feature construction produced non-finite values")
    return features


def preprocess_input(
    sample: PreprocessingInput,
    *,
    target_points: int = 80,
    quantile_method: str = "linear",
    scale_quantiles: Sequence[float] = (0.05, 0.95),
    scale_epsilon: float = 1e-8,
    step_epsilon: float = 1e-8,
    temporal_features_enabled: bool = False,
    topology_parameters: Mapping[str, Any] | None = None,
    split_sha256: str = "",
    preprocessing_config_sha256: str = "",
    source_input_point_count: int | None = None,
    valid_input_point_count: int | None = None,
    trimmed_edge_point_count: int = 0,
    interpolated_input_point_count: int = 0,
) -> dict[str, Any]:
    """Run the shared core after source-specific gap handling is complete."""

    if not isinstance(sample, PreprocessingInput):
        raise PreprocessingDataError("preprocess_input expects PreprocessingInput")
    if not np.all(sample.point_mask):
        raise PreprocessingDataError(
            "source-specific gap handling must complete before PreprocessingInput"
        )
    base = _base_record(
        sample,
        split_sha256=split_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        input_point_count=(
            len(sample.points)
            if source_input_point_count is None
            else source_input_point_count
        ),
    )
    valid_count = (
        int(sample.point_mask.sum())
        if valid_input_point_count is None
        else valid_input_point_count
    )
    try:
        resampled = arc_length_resample(
            sample.points,
            sample.point_quality,
            target_points=target_points,
            step_epsilon=step_epsilon,
        )
        normalized = robust_isotropic_normalize(
            resampled.points,
            quantile_low=float(scale_quantiles[0]),
            quantile_high=float(scale_quantiles[1]),
            quantile_method=quantile_method,
            scale_epsilon=scale_epsilon,
        )
    except PreprocessingUnavailable as exc:
        return _unavailable_record(
            base,
            reason_code=exc.reason_code,
            valid_input_point_count=valid_count,
            trimmed_edge_point_count=trimmed_edge_point_count,
            interpolated_input_point_count=interpolated_input_point_count,
        )
    except AnchorlessNormalizationError as exc:
        return _unavailable_record(
            base,
            reason_code=exc.reason_code,
            valid_input_point_count=valid_count,
            trimmed_edge_point_count=trimmed_edge_point_count,
            interpolated_input_point_count=interpolated_input_point_count,
        )

    final_mask = np.ones(target_points, dtype=np.int8)
    raw_features = build_raw_features(
        normalized.points,
        final_mask,
        resampled.quality,
        point_times_sec=None,
        step_epsilon=step_epsilon,
        temporal_features_enabled=temporal_features_enabled,
    )
    topology_kwargs = dict(topology_parameters or {})
    topology = compute_topology(
        normalized.points,
        final_mask,
        step_epsilon=step_epsilon,
        **topology_kwargs,
    )
    image_points = (
        resampled.points.tolist()
        if sample.coordinate_system == "image_normalized"
        else None
    )
    return {
        **base,
        "preprocess_status": "ready",
        "reason_codes": [],
        "valid_input_point_count": valid_count,
        "trimmed_edge_point_count": trimmed_edge_point_count,
        "interpolated_input_point_count": interpolated_input_point_count,
        "resample_mode": "cumulative_arc_length_linear",
        "resampled_source_points": resampled.points.tolist(),
        "image_normalized_points": image_points,
        "shape_normalized_points": normalized.points.tolist(),
        "point_mask": final_mask.tolist(),
        "raw_features": raw_features.tolist(),
        "model_features": raw_features.tolist(),
        "topology": topology,
    }


def preprocess_public_trajectory(
    sample: TrajectorySample,
    assignment: Mapping[str, Any],
    *,
    target_points: int = 80,
    min_valid_points: int = 8,
    max_index_gap: int = 2,
    interpolated_quality: float = 0.5,
    quantile_method: str = "linear",
    scale_quantiles: Sequence[float] = (0.05, 0.95),
    scale_epsilon: float = 1e-8,
    step_epsilon: float = 1e-8,
    temporal_features_enabled: bool = False,
    topology_parameters: Mapping[str, Any] | None = None,
    split_sha256: str = "",
    preprocessing_config_sha256: str = "",
) -> dict[str, Any]:
    """Apply public index-gap QC, then adapt into the shared core contract."""

    if not isinstance(sample, TrajectorySample):
        raise PreprocessingDataError("preprocess_public_trajectory expects TrajectorySample")
    base = _base_public_record(
        sample,
        assignment,
        split_sha256=split_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
    )
    try:
        repaired = repair_public_index_gaps(
            np.asarray(sample.points, dtype=np.float64),
            np.asarray(sample.point_mask, dtype=np.int8),
            min_valid_points=min_valid_points,
            max_index_gap=max_index_gap,
            interpolated_quality=interpolated_quality,
        )
    except PreprocessingUnavailable as exc:
        return _unavailable_record(
            base,
            reason_code=exc.reason_code,
            valid_input_point_count=exc.valid_input_point_count,
            trimmed_edge_point_count=exc.trimmed_edge_point_count,
            interpolated_input_point_count=exc.interpolated_input_point_count,
        )
    if sample.point_times_sec is not None:
        raise PreprocessingDataError(
            "frozen public step-4 inputs must not contain point_times_sec"
        )
    internal = PreprocessingInput(
        sample_id=sample.sample_id,
        source_dataset=sample.source_dataset,
        input_role=assignment["input_role"],
        split=assignment["split"],
        binary_label=sample.binary_label,
        pattern_label=sample.pattern_label.value,
        binary_supervision_eligible=assignment["binary_supervision_eligible"],
        pattern_supervision_eligible=assignment["pattern_supervision_eligible"],
        coordinate_system=sample.coordinate_system.value,
        points=repaired.points,
        point_mask=np.ones(len(repaired.points), dtype=np.int8),
        point_times_sec=None,
        point_quality=repaired.quality,
    )
    return preprocess_input(
        internal,
        target_points=target_points,
        quantile_method=quantile_method,
        scale_quantiles=scale_quantiles,
        scale_epsilon=scale_epsilon,
        step_epsilon=step_epsilon,
        temporal_features_enabled=temporal_features_enabled,
        topology_parameters=topology_parameters,
        split_sha256=split_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        source_input_point_count=len(sample.points),
        valid_input_point_count=repaired.valid_input_point_count,
        trimmed_edge_point_count=repaired.trimmed_edge_point_count,
        interpolated_input_point_count=repaired.interpolated_input_point_count,
    )


def compute_train_feature_stats(
    records: Iterable[Mapping[str, Any]],
    *,
    feature_channels: Sequence[str],
    quantile_method: str,
    iqr_epsilon: float,
    binding_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Fit channel 1-10 median/IQR using only ready train mask=1 positions."""

    if tuple(feature_channels) != _FEATURE_CHANNELS:
        raise PreprocessingDataError("feature channel order has drifted")
    if quantile_method != "linear":
        raise PreprocessingDataError("quantile_method must remain 'linear'")
    matrices: list[np.ndarray] = []
    ready_train_count = 0
    for record in records:
        if record.get("preprocess_status") != "ready" or record.get("split") != "train":
            continue
        raw = np.asarray(record.get("raw_features"), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 14 or not np.isfinite(raw).all():
            raise PreprocessingDataError("ready train raw_features must be finite [T,14]")
        valid = raw[:, 12] == 1.0
        matrices.append(raw[valid, :10])
        ready_train_count += 1
    if not matrices:
        raise PreprocessingDataError("no ready train positions are available for statistics")
    values = np.concatenate(matrices, axis=0)
    quantiles = np.quantile(
        values,
        [0.25, 0.50, 0.75],
        axis=0,
        method=quantile_method,
    )
    channels: list[dict[str, Any]] = []
    zero_iqr_channels: list[str] = []
    for index, name in enumerate(feature_channels[:10]):
        q25 = float(quantiles[0, index])
        q50 = float(quantiles[1, index])
        q75 = float(quantiles[2, index])
        iqr = q75 - q25
        denominator = iqr if iqr > float(iqr_epsilon) else 1.0
        if iqr <= float(iqr_epsilon):
            zero_iqr_channels.append(name)
        channels.append(
            {
                "index": index,
                "name": name,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "iqr": iqr,
                "denominator": denominator,
            }
        )
    normalized_hashes = dict(sorted(binding_hashes.items()))
    for name, digest in normalized_hashes.items():
        _validate_sha256(digest, f"binding_hashes.{name}")
    return {
        "schema_version": FEATURE_STATS_SCHEMA_VERSION,
        "feature_channels": list(feature_channels),
        "standardized_channel_indices": list(range(10)),
        "unchanged_channel_indices": [10, 11, 12, 13],
        "quantile_method": quantile_method,
        "iqr_epsilon": float(iqr_epsilon),
        "ready_train_sample_count": ready_train_count,
        "valid_position_count": int(len(values)),
        "channels": channels,
        "zero_iqr_channels": zero_iqr_channels,
        "temporal_features_enabled": False,
        "binding_hashes": normalized_hashes,
    }


def apply_feature_stats(raw_features: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Apply train-only channel 1-10 statistics, then enforce mask clearing."""

    raw = np.asarray(raw_features, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 14 or not np.isfinite(raw).all():
        raise PreprocessingDataError("raw_features must be finite [T,14]")
    channels = stats.get("channels")
    if not isinstance(channels, Sequence) or len(channels) != 10:
        raise PreprocessingDataError("feature stats must define exactly 10 channels")
    model = raw.copy()
    for index, channel in enumerate(channels):
        if channel.get("index") != index:
            raise PreprocessingDataError("feature stats channel index mismatch")
        denominator = float(channel["denominator"])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise PreprocessingDataError("feature stats denominator must be positive")
        model[:, index] = (raw[:, index] - float(channel["q50"])) / denominator
    model[:, 10:14] = raw[:, 10:14]
    model[raw[:, 12] == 0.0] = 0.0
    if not np.isfinite(model).all():
        raise PreprocessingDataError("model feature transformation produced non-finite values")
    return model


def build_wandering_preprocessing_from_files(
    *,
    config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> WanderingPreprocessingBuildResult:
    """Verify frozen inputs, build four canonical artifacts, and commit atomically."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"wandering preprocessing output already exists: {output}")
    config_file = Path(config_path)
    config = load_preprocessing_config(config_file)
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise PreprocessingDataError(f"project_root is not a directory: {root}")
    input_paths, input_hashes = _verify_all_input_hashes(config["inputs"], root)
    config_hash = _sha256_file(config_file)
    split = _load_and_verify_split(
        input_paths["split_json"],
        input_paths["split_sha256_file"],
        expected_split_sha256=config["split_sha256"],
    )
    assignments = _load_assignments(input_paths["assignments"])
    wp_samples = load_trajectory_jsonl(input_paths["wandering_patterns_samples"])
    smartcare_samples = load_trajectory_jsonl(input_paths["smartcare_train_pool"])
    development_rows, sealed_count = _validate_and_join_inputs(
        split=split,
        assignments=assignments,
        wp_samples=wp_samples,
        smartcare_samples=smartcare_samples,
    )

    records: list[dict[str, Any]] = []
    topology_parameters = dict(config["topology"])
    for sample, assignment in development_rows:
        records.append(
            preprocess_public_trajectory(
                sample,
                assignment,
                target_points=config["target_points"],
                min_valid_points=config["min_valid_points"],
                max_index_gap=config["max_index_gap"],
                interpolated_quality=config["interpolated_quality"],
                quantile_method=config["quantile_method"],
                scale_quantiles=config["scale_quantiles"],
                scale_epsilon=config["scale_epsilon"],
                step_epsilon=config["step_epsilon"],
                temporal_features_enabled=config["temporal_features_enabled"],
                topology_parameters=topology_parameters,
                split_sha256=split.split_sha256,
                preprocessing_config_sha256=config_hash,
            )
        )
    records.sort(key=lambda row: row["sample_id"])
    _validate_fixed_build_counts(records, sealed_count=sealed_count)

    binding_hashes = {
        **input_hashes,
        "split_sha256": split.split_sha256,
        "preprocessing_config": config_hash,
    }
    feature_stats = compute_train_feature_stats(
        records,
        feature_channels=config["feature_channels"],
        quantile_method=config["quantile_method"],
        iqr_epsilon=config["iqr_epsilon"],
        binding_hashes=binding_hashes,
    )
    for record in records:
        if record["preprocess_status"] == "ready":
            record["model_features"] = apply_feature_stats(
                np.asarray(record["raw_features"], dtype=np.float64),
                feature_stats,
            ).tolist()

    diagnostics = _select_diagnostic_ids(records, config["diagnostics"])
    report = _build_report(
        records=records,
        input_hashes=input_hashes,
        split_sha256=split.split_sha256,
        config_hash=config_hash,
        sealed_count=sealed_count,
        diagnostics=diagnostics,
    )
    samples_bytes = _canonical_jsonl_bytes(records)
    stats_bytes = _canonical_json_bytes(feature_stats)
    report_bytes = _canonical_json_bytes(report)
    artifact_bytes = {
        "samples.jsonl": samples_bytes,
        "feature_stats.json": stats_bytes,
        "preprocessing_report.json": report_bytes,
    }
    manifest = {
        "schema_version": PREPROCESSING_MANIFEST_SCHEMA_VERSION,
        "preprocessing_config_sha256": config_hash,
        "split_sha256": split.split_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "generation_order": [
            "samples.jsonl",
            "feature_stats.json",
            "preprocessing_report.json",
            "manifest.json",
        ],
        "artifacts": {
            name: {
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in artifact_bytes.items()
        },
        "manifest_self_hash_embedded": False,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    files = {**artifact_bytes, "manifest.json": manifest_bytes}
    _commit_new_output_directory(output, files)
    return WanderingPreprocessingBuildResult(
        output_dir=output,
        report=report,
        manifest=manifest,
    )


def _base_record(
    sample: PreprocessingInput,
    *,
    split_sha256: str,
    preprocessing_config_sha256: str,
    input_point_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": PREPROCESSED_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "parent_sample_id": sample.sample_id,
        "source_dataset": sample.source_dataset,
        "input_role": sample.input_role,
        "split": sample.split,
        "binary_label": sample.binary_label,
        "pattern_label": sample.pattern_label,
        "binary_supervision_eligible": sample.binary_supervision_eligible,
        "pattern_supervision_eligible": sample.pattern_supervision_eligible,
        "input_coordinate_system": sample.coordinate_system,
        "input_point_count": input_point_count,
        "split_sha256": split_sha256,
        "preprocessing_config_sha256": preprocessing_config_sha256,
    }


def _base_public_record(
    sample: TrajectorySample,
    assignment: Mapping[str, Any],
    *,
    split_sha256: str,
    preprocessing_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREPROCESSED_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "parent_sample_id": sample.sample_id,
        "source_dataset": sample.source_dataset,
        "input_role": assignment["input_role"],
        "split": assignment["split"],
        "binary_label": sample.binary_label,
        "pattern_label": sample.pattern_label.value,
        "binary_supervision_eligible": assignment["binary_supervision_eligible"],
        "pattern_supervision_eligible": assignment["pattern_supervision_eligible"],
        "input_coordinate_system": sample.coordinate_system.value,
        "input_point_count": len(sample.points),
        "split_sha256": split_sha256,
        "preprocessing_config_sha256": preprocessing_config_sha256,
    }


def _unavailable_record(
    base: Mapping[str, Any],
    *,
    reason_code: str,
    valid_input_point_count: int,
    trimmed_edge_point_count: int,
    interpolated_input_point_count: int,
) -> dict[str, Any]:
    return {
        **base,
        "preprocess_status": "unavailable",
        "reason_codes": [reason_code],
        "valid_input_point_count": valid_input_point_count,
        "trimmed_edge_point_count": trimmed_edge_point_count,
        "interpolated_input_point_count": interpolated_input_point_count,
        "resample_mode": None,
        "resampled_source_points": None,
        "image_normalized_points": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "model_features": None,
        "topology": None,
    }


def _validate_and_join_inputs(
    *,
    split: SplitManifest,
    assignments: Sequence[Mapping[str, Any]],
    wp_samples: Sequence[TrajectorySample],
    smartcare_samples: Sequence[TrajectorySample],
) -> tuple[list[tuple[TrajectorySample, Mapping[str, Any]]], int]:
    assignment_index = {row["sample_id"]: row for row in assignments}
    split_partitions = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
        "sealed_external_test": set(split.sealed_external_test),
    }
    for partition, expected_ids in split_partitions.items():
        assigned_ids = {
            row["sample_id"] for row in assignments if row["split"] == partition
        }
        if assigned_ids != expected_ids:
            raise PreprocessingDataError(f"assignments do not match split partition {partition}")
    samples = tuple(wp_samples) + tuple(smartcare_samples)
    if len(wp_samples) != 1600 or len(smartcare_samples) != 190:
        raise PreprocessingDataError("public development inputs must remain 1600 WP + 190 SmartCare")
    sample_ids = [sample.sample_id for sample in samples]
    development_ids = split_partitions["train"] | split_partitions["validation"] | split_partitions["test"]
    if len(sample_ids) != 1790 or len(set(sample_ids)) != 1790 or set(sample_ids) != development_ids:
        raise PreprocessingDataError("public source IDs must equal the 1790 non-sealed split IDs")
    if set(sample_ids) & split_partitions["sealed_external_test"]:
        raise PreprocessingDataError("sealed sample ID entered public preprocessing sources")
    joined: list[tuple[TrajectorySample, Mapping[str, Any]]] = []
    for sample in samples:
        row = assignment_index[sample.sample_id]
        expected_source = "wandering_patterns" if sample.source_dataset == "wandering_patterns" else "smartcare"
        expected_role = "samples" if expected_source == "wandering_patterns" else "train_pool"
        if row["source_name"] != expected_source or row["input_role"] != expected_role:
            raise PreprocessingDataError(f"assignment source/role mismatch for {sample.sample_id}")
        if row.get("binary_label") != sample.binary_label or row.get("pattern_label") != sample.pattern_label.value:
            raise PreprocessingDataError(f"assignment label mismatch for {sample.sample_id}")
        expected_pattern_eligible = expected_source == "wandering_patterns"
        if row.get("binary_supervision_eligible") is not True:
            raise PreprocessingDataError(f"binary supervision mismatch for {sample.sample_id}")
        if row.get("pattern_supervision_eligible") is not expected_pattern_eligible:
            raise PreprocessingDataError(f"pattern supervision mismatch for {sample.sample_id}")
        joined.append((sample, row))
    joined.sort(key=lambda item: item[0].sample_id)
    return joined, len(split.sealed_external_test)


def _load_assignments(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise PreprocessingDataError(f"blank assignments line {line_number}")
                row = json.loads(raw_line)
                if not isinstance(row, dict) or row.get("schema_version") != _ASSIGNMENT_SCHEMA_VERSION:
                    raise PreprocessingDataError(f"invalid assignment row {line_number}")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                    raise PreprocessingDataError(f"invalid or duplicate assignment sample_id at line {line_number}")
                if row.get("split") not in ("train", "validation", "test", "sealed_external_test"):
                    raise PreprocessingDataError(f"invalid assignment split at line {line_number}")
                ids.add(sample_id)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessingDataError(f"cannot load assignments: {path}") from exc
    if len(rows) != 1810:
        raise PreprocessingDataError(f"assignments must contain 1810 rows, got {len(rows)}")
    return tuple(rows)


def _load_and_verify_split(
    split_path: Path,
    split_sha_path: Path,
    *,
    expected_split_sha256: str,
) -> SplitManifest:
    try:
        split = SplitManifest.from_dict(json.loads(split_path.read_text(encoding="utf-8")))
        checksum = split_sha_path.read_text(encoding="ascii")
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestValidationError) as exc:
        raise InputHashMismatchError("cannot load the frozen canonical split") from exc
    if split.split_sha256 != expected_split_sha256:
        raise InputHashMismatchError(
            f"canonical split SHA-256 drift: expected {expected_split_sha256}, got {split.split_sha256}"
        )
    if checksum != f"{expected_split_sha256}\n":
        raise InputHashMismatchError("split.sha256 content does not equal canonical split SHA-256")
    return split


def _verify_all_input_hashes(
    descriptors: Mapping[str, Mapping[str, str]],
    root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in _INPUT_ROLES:
        descriptor = descriptors[role]
        path = (root / PurePosixPath(descriptor["path"])).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PreprocessingConfigError(f"input {role} escapes project_root") from exc
        if not path.is_file():
            raise InputHashMismatchError(f"input {role} is missing: {descriptor['path']}")
        actual = _sha256_file(path)
        expected = descriptor["sha256"]
        if actual != expected:
            raise InputHashMismatchError(
                f"input {role} SHA-256 drift: expected {expected}, got {actual}"
            )
        paths[role] = path
        hashes[role] = actual
    return paths, hashes


def _validate_fixed_build_counts(records: Sequence[Mapping[str, Any]], *, sealed_count: int) -> None:
    ids = [row["sample_id"] for row in records]
    if len(ids) != 1790 or len(set(ids)) != 1790 or ids != sorted(ids):
        raise PreprocessingDataError("preprocessing must produce 1790 unique sorted records")
    if sealed_count != 20:
        raise PreprocessingDataError("exactly 20 sealed IDs must be excluded")
    statuses = Counter(row["preprocess_status"] for row in records)
    if statuses != Counter({"ready": 1775, "unavailable": 15}):
        raise PreprocessingDataError(f"preprocessing status counts drifted: {statuses}")
    ready_splits = Counter(
        row["split"] for row in records if row["preprocess_status"] == "ready"
    )
    if ready_splits != Counter({"train": 1257, "validation": 278, "test": 240}):
        raise PreprocessingDataError(f"ready split counts drifted: {ready_splits}")
    unavailable = [row for row in records if row["preprocess_status"] == "unavailable"]
    if any(
        row["source_dataset"] != "smartcare"
        or row["split"] != "train"
        or row["reason_codes"] != ["too_few_valid_points"]
        for row in unavailable
    ):
        raise PreprocessingDataError("only fixed SmartCare train short samples may be unavailable")
    if Counter(row["binary_label"] for row in unavailable) != Counter({0: 10, 1: 5}):
        raise PreprocessingDataError("SmartCare unavailable label counts drifted")


def _select_diagnostic_ids(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, int],
) -> dict[str, Any]:
    seed = config["seed"]
    count = config["samples_per_class"]
    ready_train = [
        row for row in records if row["preprocess_status"] == "ready" and row["split"] == "train"
    ]

    def select(group_name: str, candidates: Sequence[Mapping[str, Any]]) -> list[str]:
        if len(candidates) < count:
            raise PreprocessingDataError(f"diagnostic group {group_name} has fewer than {count} samples")
        ranked = sorted(
            candidates,
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{group_name}\0{row['sample_id']}".encode("utf-8")).hexdigest(),
                row["sample_id"],
            ),
        )
        return sorted(row["sample_id"] for row in ranked[:count])

    wp_groups = {
        label: select(
            f"wandering_patterns:{label}",
            [row for row in ready_train if row["source_dataset"] == "wandering_patterns" and row["pattern_label"] == label],
        )
        for label in ("direct", "pacing", "lapping", "random")
    }
    smartcare_groups = {
        name: select(
            f"smartcare:{name}",
            [row for row in ready_train if row["source_dataset"] == "smartcare" and row["binary_label"] == label],
        )
        for name, label in (("normal", 0), ("wandering_like", 1))
    }
    unavailable_ids = sorted(
        row["sample_id"] for row in records if row["preprocess_status"] == "unavailable"
    )
    return {
        "seed": seed,
        "samples_per_class": count,
        "ready_train_sample_ids": {
            "wandering_patterns": wp_groups,
            "smartcare": smartcare_groups,
        },
        "unavailable_sample_ids": unavailable_ids,
    }


def _build_report(
    *,
    records: Sequence[Mapping[str, Any]],
    input_hashes: Mapping[str, str],
    split_sha256: str,
    config_hash: str,
    sealed_count: int,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(row["preprocess_status"] for row in records)
    original_splits = Counter(row["split"] for row in records)
    ready_splits = Counter(
        row["split"] for row in records if row["preprocess_status"] == "ready"
    )
    unavailable_reasons = Counter(
        reason
        for row in records
        if row["preprocess_status"] == "unavailable"
        for reason in row["reason_codes"]
    )
    source_status = Counter(
        (row["source_dataset"], row["preprocess_status"]) for row in records
    )
    return {
        "schema_version": PREPROCESSING_REPORT_SCHEMA_VERSION,
        "preprocessing_config_sha256": config_hash,
        "split_sha256": split_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "counts": {
            "nonsealed_processed": len(records),
            "sealed_excluded": sealed_count,
            "ready": status_counts["ready"],
            "unavailable": status_counts["unavailable"],
            "original_by_split": {
                name: original_splits[name] for name in ("train", "validation", "test")
            },
            "ready_by_split": {
                name: ready_splits[name] for name in ("train", "validation", "test")
            },
            "source_status": {
                source: {
                    status: source_status[(source, status)]
                    for status in ("ready", "unavailable")
                }
                for source in ("wandering_patterns", "smartcare")
            },
            "unavailable_reason_codes": dict(sorted(unavailable_reasons.items())),
        },
        "diagnostics": diagnostics,
        "constraints": {
            "sample_id_equals_parent_sample_id": all(
                row["sample_id"] == row["parent_sample_id"] for row in records
            ),
            "sealed_source_file_opened": False,
            "runtime_random_split_used": False,
            "temporal_features_enabled": False,
            "bbox_height_compensation_used": False,
            "camera_inputs_used": False,
        },
    }


def _validated_points_mask_quality(
    points: np.ndarray,
    point_mask: np.ndarray,
    point_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        array = np.asarray(points, dtype=np.float64)
        quality = np.asarray(point_quality, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PreprocessingDataError("points and quality must be finite numeric values") from exc
    mask_array = np.asarray(point_mask)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise PreprocessingDataError("points must have shape [N,2] with N >= 2")
    if not np.isfinite(array).all():
        raise PreprocessingDataError("points must be finite")
    if mask_array.ndim != 1 or len(mask_array) != len(array) or not np.all(np.isin(mask_array, [0, 1])):
        raise PreprocessingDataError("point_mask must be binary shape [N]")
    mask = mask_array.astype(bool, copy=False)
    if quality.ndim != 1 or len(quality) != len(array) or not np.isfinite(quality).all():
        raise PreprocessingDataError("point_quality must be finite shape [N]")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise PreprocessingDataError("point_quality must be within [0,1]")
    if np.any(quality[~mask] != 0.0):
        raise PreprocessingDataError("masked input point quality must be zero")
    return array, mask, quality


def _validate_inputs_config(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(_INPUT_ROLES):
        raise PreprocessingConfigError(f"inputs must contain exactly: {', '.join(_INPUT_ROLES)}")
    for role in _INPUT_ROLES:
        descriptor = value[role]
        _require_exact_fields(descriptor, _INPUT_DESCRIPTOR_FIELDS, f"inputs.{role}")
        _validate_relative_path(descriptor["path"], f"inputs.{role}.path")
        _validate_sha256(descriptor["sha256"], f"inputs.{role}.sha256")


def _validate_topology_config(value: Any) -> None:
    _require_exact_fields(value, _TOPOLOGY_FIELDS, "topology")
    expected = {
        "reversal_direction_span": 2,
        "reversal_angle_deg": 120.0,
        "reversal_merge_gap": 2,
        "revisit_index_gap": 8,
        "revisit_radius": 0.10,
        "max_revisit_links": 5,
    }
    if value != expected:
        raise PreprocessingConfigError("topology parameters have drifted")


def _require_exact_fields(value: Any, expected: frozenset[str], field: str) -> None:
    if not isinstance(value, dict):
        raise PreprocessingConfigError(f"{field} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown fields: {', '.join(unknown)}")
        raise PreprocessingConfigError(f"{field} has {'; '.join(parts)}")


def _require_int(value: Any, field: str, *, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise PreprocessingConfigError(f"{field} must remain {expected}")


def _require_float(value: Any, field: str, *, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
        raise PreprocessingConfigError(f"{field} must remain {expected}")


def _validate_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PreprocessingConfigError(f"{field} must be a portable relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise PreprocessingConfigError(f"{field} must be a portable relative POSIX path")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PreprocessingConfigError(f"{field} must be a lowercase 64-character SHA-256")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreprocessingDataError("artifact is not canonical finite JSON data") from exc


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputHashMismatchError(f"cannot hash input file: {path}") from exc
    return digest.hexdigest()


def _commit_new_output_directory(output: Path, files: Mapping[str, bytes]) -> None:
    """Write a complete bundle to a temp sibling, then atomically rename it."""

    if output.exists():
        raise FileExistsError(f"wandering preprocessing output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    committed = False
    try:
        for name, payload in files.items():
            path = temp_dir / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_dir, output)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(temp_dir, ignore_errors=True)
