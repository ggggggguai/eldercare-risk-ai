"""Deterministic synthetic camera corruption for wandering step 8.

This module deliberately generates only synthetic camera observations from the
frozen development trajectories.  It imports the step-7 adapter/QC/inference
path instead of maintaining an augmentation-specific copy of those rules.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraObservation,
    canonical_jsonl_bytes,
    validate_media_sidecar,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    prepare_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCResult,
    run_camera_qc,
)


AUGMENTATION_CONFIG_SCHEMA_VERSION_V1 = "wandering-augmentation-config-v1"
AUGMENTATION_CONFIG_SCHEMA_VERSION_V2 = "wandering-augmentation-config-v2"
AUGMENTATION_CONFIG_SCHEMA_VERSION_V3 = "wandering-augmentation-config-v3"
AUGMENTATION_CONFIG_SCHEMA_VERSION = AUGMENTATION_CONFIG_SCHEMA_VERSION_V1
AUGMENTATION_CONFIG_SHA256_V1 = "db0834ed96977511c292158886e25231461505991a83684c0c2e137e58cb2e1e"
AUGMENTATION_CONFIG_SHA256_V2 = "11e184e450eea8bd82966979bc32765863cbe843c5ef0cdea934b6e3e26905ff"
AUGMENTATION_CONFIG_SHA256_V3 = "b6f6b682b2c4b908a83c60db6da04f39cfbab4e92fe748b8e6acaf84752c89d8"
AUGMENTATION_CONFIG_SHA256 = AUGMENTATION_CONFIG_SHA256_V1
ATTEMPT_SCHEMA_VERSION = "wandering-augmentation-attempt-v1"

_TOP_LEVEL_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "purpose",
        "global_seed",
        "trust_roots",
        "split_policy",
        "generation",
        "profiles",
        "semantic_check",
        "fault_suite",
        "compatibility",
        "visual_review",
        "output_schemas",
    }
)
_TOP_LEVEL_FIELDS_V2 = _TOP_LEVEL_FIELDS_V1 | {"diagnostic_predecessor"}
_TOP_LEVEL_FIELDS_V3 = _TOP_LEVEL_FIELDS_V1 | {"diagnostic_predecessors", "quota_gate"}
_TRUST_ROOT_NAMES = (
    "preprocessing_config",
    "preprocessing_samples",
    "preprocessing_feature_stats",
    "preprocessing_manifest",
    "camera_config",
    "camera_adapter_source",
    "camera_qc_source",
    "camera_inference_source",
    "rf_config",
    "rf_development_manifest",
    "tcn_config",
    "tcn_development_manifest",
)
_PROFILE_FIELDS = frozenset(
    {
        "rotation_degrees",
        "translation_xy",
        "uniform_scale",
        "anisotropic_scale",
        "shear",
        "projective",
        "crop_fraction",
        "local_shape_control_offset",
        "ar1_rho",
        "position_sigma",
        "random_missing_fraction",
        "contiguous_missing_buckets",
        "bbox_bottom_offset_fraction",
        "bbox_height_noise_sigma",
        "timestamp_jitter_seconds",
        "quantized_heights",
    }
)
_FORBIDDEN_PATH_MARKERS = (
    "official_validation",
    "sealed_external_test",
    "public_shape_benchmark",
    "wanderingpatterns/test",
)


class CameraCorruptionError(ValueError):
    """The frozen corruption contract, input, or generated view is invalid."""


@dataclass(frozen=True)
class CorruptedCameraView:
    """One deterministic corruption attempt and its step-7 result."""

    attempt_record: Mapping[str, Any]
    adapter_input: CameraAdapterInput | None
    qc_result: CameraQCResult | None
    prepared_window: Mapping[str, Any] | None


def load_augmentation_config(path: str | Path) -> dict[str, Any]:
    """Load only one byte-frozen Step-8 v1/v2/v3 augmentation config."""

    config_path = Path(path)
    try:
        payload = config_path.read_bytes()
    except OSError as exc:
        raise CameraCorruptionError(f"cannot read augmentation config: {config_path}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    schema_for_hash = {
        AUGMENTATION_CONFIG_SHA256_V1: AUGMENTATION_CONFIG_SCHEMA_VERSION_V1,
        AUGMENTATION_CONFIG_SHA256_V2: AUGMENTATION_CONFIG_SCHEMA_VERSION_V2,
        AUGMENTATION_CONFIG_SHA256_V3: AUGMENTATION_CONFIG_SCHEMA_VERSION_V3,
    }.get(actual_sha256)
    if schema_for_hash is None:
        raise CameraCorruptionError("augmentation config fields or frozen values have drifted")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise CameraCorruptionError("augmentation config is not valid UTF-8 YAML") from exc
    expected_fields = {
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V1: _TOP_LEVEL_FIELDS_V1,
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V2: _TOP_LEVEL_FIELDS_V2,
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V3: _TOP_LEVEL_FIELDS_V3,
    }[schema_for_hash]
    if not isinstance(value, dict) or frozenset(value) != expected_fields:
        raise CameraCorruptionError("augmentation top-level fields have drifted")
    if value["schema_version"] != schema_for_hash:
        raise CameraCorruptionError("augmentation schema/hash binding has drifted")
    if value["purpose"] != "semantic_preserving_synthetic_camera_corruption":
        raise CameraCorruptionError("augmentation v1 purpose has drifted")
    if value["global_seed"] != 20260805:
        raise CameraCorruptionError("augmentation v1 global seed has drifted")
    if schema_for_hash == AUGMENTATION_CONFIG_SCHEMA_VERSION_V2:
        if value.get("diagnostic_predecessor") != {
            "status": "diagnostic_invalid_pacing_gate",
            "config_sha256": AUGMENTATION_CONFIG_SHA256_V1,
            "manifest_sha256": "e733a99c4d17eb4f6d3295c2dea19196fece8b26026a140768bf2b155995474a",
            "report_sha256": "ee1e0b1afadc08693b9017802fda207e2b8edcd5c269816ae17c25e7b94005d7",
            "output_reuse_allowed": False,
        }:
            raise CameraCorruptionError("v2 diagnostic predecessor binding has drifted")
    if schema_for_hash == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        if value.get("diagnostic_predecessors") != {
            "v1": {
                "status": "diagnostic_invalid_pacing_gate",
                "config_sha256": AUGMENTATION_CONFIG_SHA256_V1,
                "manifest_sha256": "e733a99c4d17eb4f6d3295c2dea19196fece8b26026a140768bf2b155995474a",
                "report_sha256": "ee1e0b1afadc08693b9017802fda207e2b8edcd5c269816ae17c25e7b94005d7",
                "output_reuse_allowed": False,
            },
            "v2": {
                "status": "diagnostic_quota_blocked_precommit",
                "config_sha256": AUGMENTATION_CONFIG_SHA256_V2,
                "formal_output_created": False,
                "output_reuse_allowed": False,
                "observed_by_read_only_replay": {
                    "wp_direct_selected": {"low": 137, "medium": 34},
                    "wp_pacing_selected": {"low": 124, "medium": 26},
                },
            },
        }:
            raise CameraCorruptionError("v3 diagnostic predecessor bindings have drifted")
        if value.get("quota_gate") != {
            "checked_before_atomic_commit": True,
            "wp_min_selected_per_class_per_severity": 50,
            "smartcare_min_selected_per_binary_class_per_severity": 10,
            "on_failure": "fail_closed_without_output",
            "report_all_fields": [
                "parent_count",
                "at_least_once_qc_ready_count",
                "semantic_accepted_count",
                "selected_count",
                "qc_reason_counts",
                "semantic_rejection_reason_counts",
            ],
        }:
            raise CameraCorruptionError("v3 quota gate has drifted")
    roots = value.get("trust_roots")
    if not isinstance(roots, dict) or tuple(roots) != _TRUST_ROOT_NAMES:
        raise CameraCorruptionError("augmentation trust-root roles or order have drifted")
    for role, descriptor in roots.items():
        if not isinstance(descriptor, dict) or frozenset(descriptor) != {"path", "sha256"}:
            raise CameraCorruptionError(f"augmentation trust-root fields drifted: {role}")
        _validate_relative_path(descriptor["path"], f"trust_roots.{role}.path")
        _validate_sha256(descriptor["sha256"], f"trust_roots.{role}.sha256")
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or tuple(profiles) != ("low", "medium", "high"):
        raise CameraCorruptionError("augmentation severity profiles have drifted")
    for severity, profile in profiles.items():
        if not isinstance(profile, dict) or frozenset(profile) != _PROFILE_FIELDS:
            raise CameraCorruptionError(f"augmentation profile fields drifted: {severity}")
    if value["split_policy"]["forbidden_splits"] != ["test", "sealed_external_test"]:
        raise CameraCorruptionError("augmentation forbidden split policy has drifted")
    if value["split_policy"]["official_source_allowed"] is not False:
        raise CameraCorruptionError("official source must remain forbidden in step 8")
    if value["generation"]["model_temporal_features_enabled"] is not False:
        raise CameraCorruptionError("step-8 model temporal channels must remain disabled")
    attempts = value["generation"]["max_train_attempts_per_severity"]
    if schema_for_hash == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        if attempts != {"low": 4, "medium": 16}:
            raise CameraCorruptionError("v3 train attempt budget has drifted")
    elif attempts != 4:
        raise CameraCorruptionError("v1/v2 train attempt budget has drifted")
    if value["visual_review"]["initial_status"] != "pending_human_review":
        raise CameraCorruptionError("step-8 human-review status must start pending")
    return value


def verify_augmentation_trust_roots(
    config: Mapping[str, Any], project_root: str | Path
) -> dict[str, Path]:
    """Verify every frozen dependency before any sample bundle is opened."""

    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise CameraCorruptionError("project_root must be an existing directory")
    if config.get("schema_version") in {
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V2,
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V3,
    }:
        _verify_diagnostic_predecessors(config, root)
    output: dict[str, Path] = {}
    for role in _TRUST_ROOT_NAMES:
        descriptor = config["trust_roots"][role]
        relative = str(descriptor["path"])
        lowered = relative.replace("\\", "/").lower()
        if any(marker in lowered for marker in _FORBIDDEN_PATH_MARKERS):
            raise CameraCorruptionError(f"forbidden step-8 trust-root path: {role}")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CameraCorruptionError(f"trust root escapes project_root: {role}") from exc
        if not candidate.is_file():
            raise CameraCorruptionError(f"trust root is not a file: {role}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != descriptor["sha256"]:
            raise CameraCorruptionError(f"trust-root SHA-256 mismatch: {role}")
        output[role] = candidate
    return output


def derive_child_identity(
    *,
    parent_sample_id: str,
    split: str,
    severity: str,
    view_role: str,
    attempt_index: int,
    global_seed: int,
) -> tuple[str, int]:
    """Derive the order-independent child identifier and PCG64 seed."""

    if not all(isinstance(value, str) and value for value in (parent_sample_id, split, severity, view_role)):
        raise CameraCorruptionError("child identity tokens must be non-empty strings")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        raise CameraCorruptionError("attempt_index must be a nonnegative integer")
    if global_seed != 20260805:
        raise CameraCorruptionError("child seed must use the frozen global seed")
    identity = (
        f"wandering-augmentation-v1|{parent_sample_id}|{split}|"
        f"{severity}|{view_role}|{attempt_index}"
    )
    child_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    seed_digest = hashlib.sha256(f"{global_seed}|{child_id}".encode("utf-8")).digest()
    return child_id, int.from_bytes(seed_digest[:8], byteorder="big", signed=False)


def derive_versioned_child_identity(
    *,
    parent_sample_id: str,
    split: str,
    severity: str,
    view_role: str,
    attempt_index: int,
    config_schema_version: str,
    global_seed: int,
) -> tuple[str, str, int]:
    """Separate the v2 artifact namespace from the frozen v1 RNG replay key."""

    replay_key, replay_seed = derive_child_identity(
        parent_sample_id=parent_sample_id,
        split=split,
        severity=severity,
        view_role=view_role,
        attempt_index=attempt_index,
        global_seed=global_seed,
    )
    if config_schema_version == AUGMENTATION_CONFIG_SCHEMA_VERSION_V1:
        return replay_key, replay_key, replay_seed
    namespace = {
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V2: "wandering-augmentation-v2",
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V3: "wandering-augmentation-v3",
    }.get(config_schema_version)
    if namespace is None:
        raise CameraCorruptionError("unsupported augmentation config schema")
    identity = (
        f"{namespace}|{parent_sample_id}|{split}|"
        f"{severity}|{view_role}|{attempt_index}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), replay_key, replay_seed


def derive_versioned_fault_identity(
    *,
    parent_sample_id: str,
    fault_or_control_type: str,
    control: bool,
    config_schema_version: str,
    global_seed: int,
) -> tuple[str, str, int]:
    """Use v2 fault IDs while replaying the corresponding v1 fault RNG key."""

    v1_prefix = "wandering-qc-control-v1" if control else "wandering-qc-fault-v1"
    replay_key = hashlib.sha256(
        f"{v1_prefix}|{parent_sample_id}|{fault_or_control_type}".encode("utf-8")
    ).hexdigest()
    seed = int.from_bytes(
        hashlib.sha256(f"{global_seed}|{replay_key}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    if config_schema_version == AUGMENTATION_CONFIG_SCHEMA_VERSION_V1:
        return replay_key, replay_key, seed
    version = {
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V2: "v2",
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V3: "v3",
    }.get(config_schema_version)
    if version is None:
        raise CameraCorruptionError("unsupported augmentation config schema")
    artifact_prefix = (
        f"wandering-qc-control-{version}" if control else f"wandering-qc-fault-{version}"
    )
    artifact_id = hashlib.sha256(
        f"{artifact_prefix}|{parent_sample_id}|{fault_or_control_type}".encode("utf-8")
    ).hexdigest()
    return artifact_id, replay_key, seed


def derive_fault_identity(
    *, parent_sample_id: str, fault_or_control_type: str, control: bool, global_seed: int
) -> tuple[str, int]:
    """Derive a fault/control ID without reusing an augmentation attempt ID."""

    prefix = "wandering-qc-control-v1" if control else "wandering-qc-fault-v1"
    value = hashlib.sha256(
        f"{prefix}|{parent_sample_id}|{fault_or_control_type}".encode("utf-8")
    ).hexdigest()
    seed = int.from_bytes(
        hashlib.sha256(f"{global_seed}|{value}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return value, seed


def fit_virtual_canvas(
    points: Sequence[Sequence[float]] | np.ndarray,
    point_mask: Sequence[int] | np.ndarray,
    *,
    extent: float,
    epsilon: float,
) -> np.ndarray:
    """Apply the frozen positive isotropic center/extent fit."""

    array = _points(points)
    mask = _mask(point_mask, len(array))
    valid = array[mask]
    if len(valid) < 2:
        raise CameraCorruptionError("parent must contain at least two valid points")
    low = valid.min(axis=0)
    high = valid.max(axis=0)
    span = float(np.max(high - low))
    if span <= float(epsilon):
        raise CameraCorruptionError("degenerate_parent_span")
    center = (low + high) / 2.0
    return (array - center) * (float(extent) / span) + np.asarray([0.5, 0.5])


def protected_point_indices(parent: Mapping[str, Any], *, target_points: int) -> frozenset[int]:
    """Protect high-curvature, reversal-neighborhood, and revisit endpoints."""

    topology = parent.get("topology")
    if not isinstance(topology, Mapping):
        raise CameraCorruptionError("ready parent topology is missing")
    curvature = np.asarray(topology.get("abs_curvature"), dtype=np.float64)
    if curvature.shape != (target_points,) or not np.isfinite(curvature).all():
        raise CameraCorruptionError("parent topology curvature is invalid")
    keep = int(math.ceil(target_points * 0.20))
    ranked = sorted(range(target_points), key=lambda index: (-float(curvature[index]), index))
    protected = set(ranked[:keep])
    for event in topology.get("reversal_events", []):
        center = int(event["index"])
        protected.update(range(max(0, center - 2), min(target_points, center + 3)))
    for link in topology.get("selected_revisit_links", []):
        protected.update((int(link["i"]), int(link["j"])))
    return frozenset(protected)


def sample_local_shape_offsets(
    rng: np.random.Generator,
    *,
    control_indices: Sequence[int],
    protected_indices: frozenset[int],
    magnitude_range: Sequence[float],
) -> np.ndarray:
    """Sample uniform magnitude then uniform angle in control-index order."""

    low, high = _numeric_range(magnitude_range, "local_shape_control_offset")
    offsets = np.zeros((len(control_indices), 2), dtype=np.float64)
    for position, index in enumerate(control_indices):
        if int(index) in protected_indices:
            continue
        magnitude = float(rng.uniform(low, high))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        offsets[position] = (magnitude * math.cos(angle), magnitude * math.sin(angle))
    return offsets


def interpolate_local_offsets(
    *,
    point_count: int,
    control_indices: Sequence[int],
    control_offsets: Sequence[Sequence[float]] | np.ndarray,
    protected_indices: set[int] | frozenset[int],
) -> np.ndarray:
    """Linearly interpolate, replicate-pad, average over five, then protect."""

    indices = np.asarray(control_indices, dtype=np.int64)
    offsets = np.asarray(control_offsets, dtype=np.float64)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or not np.array_equal(indices, np.unique(indices))
        or indices[0] != 0
        or indices[-1] != point_count - 1
        or offsets.shape != (len(indices), 2)
        or not np.isfinite(offsets).all()
    ):
        raise CameraCorruptionError("local-shape control arrays are invalid")
    locations = np.arange(point_count, dtype=np.float64)
    interpolated = np.column_stack(
        [np.interp(locations, indices, offsets[:, axis]) for axis in range(2)]
    )
    padded = np.pad(interpolated, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.stack(
        [padded[index : index + 5].mean(axis=0) for index in range(point_count)]
    )
    protected = sorted(int(value) for value in protected_indices)
    if any(value < 0 or value >= point_count for value in protected):
        raise CameraCorruptionError("protected local-shape index is out of bounds")
    smoothed[protected] = 0.0
    return smoothed


def compose_projective_matrix(
    *,
    theta_degrees: float,
    uniform_scale: float,
    anisotropic_x: float,
    anisotropic_y: float,
    shear_x: float,
    shear_y: float,
    projective_x: float,
    projective_y: float,
    translation_x: float,
    translation_y: float,
) -> np.ndarray:
    """Compose the unique frozen column-vector 3x3 matrix."""

    values = np.asarray(
        [
            theta_degrees,
            uniform_scale,
            anisotropic_x,
            anisotropic_y,
            shear_x,
            shear_y,
            projective_x,
            projective_y,
            translation_x,
            translation_y,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or min(uniform_scale, anisotropic_x, anisotropic_y) <= 0.0:
        raise CameraCorruptionError("projective parameters must be finite with positive scales")
    theta = math.radians(float(theta_degrees))
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    uniform = np.diag([uniform_scale, uniform_scale, 1.0])
    anisotropic = np.diag([anisotropic_x, anisotropic_y, 1.0])
    shear = np.asarray([[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]])
    projective = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [projective_x, projective_y, 1.0]]
    )
    center = _translation(-0.5, -0.5)
    restore = _translation(0.5 + translation_x, 0.5 + translation_y)
    return restore @ projective @ shear @ anisotropic @ uniform @ rotation @ center


def apply_projective(
    points: Sequence[Sequence[float]] | np.ndarray,
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    denominator_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H and reject zero or sign-changing projective denominators."""

    array = _points(points)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise CameraCorruptionError("projection matrix must be finite shape [3,3]")
    homogeneous = np.column_stack((array, np.ones(len(array))))
    transformed = (transform @ homogeneous.T).T
    denominator = transformed[:, 2]
    if (
        np.any(np.abs(denominator) < float(denominator_epsilon))
        or (np.any(denominator > 0.0) and np.any(denominator < 0.0))
    ):
        raise CameraCorruptionError("projection_denominator_invalid")
    projected = transformed[:, :2] / denominator[:, None]
    if not np.isfinite(projected).all():
        raise CameraCorruptionError("projection_denominator_invalid")
    return projected, denominator


def sample_ar1_noise(
    rng: Any,
    *,
    count: int,
    rho: float,
    stationary_sigma: float,
    dimensions: int,
) -> np.ndarray:
    """Sample the frozen zero-state AR(1) process with clipped innovations."""

    if count < 1 or dimensions < 1 or not 0.0 <= rho < 1.0 or stationary_sigma < 0.0:
        raise CameraCorruptionError("AR(1) parameters are invalid")
    innovation_sigma = float(stationary_sigma) * math.sqrt(1.0 - float(rho) ** 2)
    if innovation_sigma == 0.0:
        innovations = np.zeros((count, dimensions), dtype=np.float64)
    else:
        draws = np.asarray(
            rng.normal(0.0, innovation_sigma, size=(count, dimensions)), dtype=np.float64
        )
        innovations = np.clip(draws, -3.0 * innovation_sigma, 3.0 * innovation_sigma)
    output = np.empty_like(innovations)
    prior = np.zeros(dimensions, dtype=np.float64)
    for index in range(count):
        prior = float(rho) * prior + innovations[index]
        output[index] = prior
    return output


def quantize_half_up(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Implement q=floor(pixel+0.5), not banker's rounding."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise CameraCorruptionError("quantization input must be finite")
    return np.floor(array + 0.5).astype(np.int64)


def generate_corrupted_camera_view(
    parent: Mapping[str, Any],
    *,
    severity: str,
    view_role: str,
    attempt_index: int,
    config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
) -> CorruptedCameraView:
    """Generate one child, then call the frozen step-7 QC and step-4 core."""

    generation = config["generation"]
    target = int(generation["target_points"])
    _validate_parent(parent, target=target, config=config)
    child_id, corruption_replay_key, child_seed = derive_versioned_child_identity(
        parent_sample_id=str(parent["sample_id"]),
        split=str(parent["split"]),
        severity=severity,
        view_role=view_role,
        attempt_index=attempt_index,
        config_schema_version=str(config["schema_version"]),
        global_seed=int(config["global_seed"]),
    )
    base = _attempt_base(
        parent,
        child_id,
        corruption_replay_key,
        child_seed,
        severity,
        view_role,
        attempt_index,
        config,
    )
    profile = config["profiles"].get(severity)
    if not isinstance(profile, Mapping):
        raise CameraCorruptionError(f"unknown severity: {severity!r}")
    rng = np.random.Generator(np.random.PCG64(child_seed))
    try:
        clean = np.asarray(parent["shape_normalized_points"], dtype=np.float64)
        mask = np.asarray(parent["point_mask"], dtype=np.int8)
        fitted = fit_virtual_canvas(
            clean,
            mask,
            extent=float(generation["base_fit_extent"]),
            epsilon=float(generation["base_span_epsilon"]),
        )
        protected = protected_point_indices(parent, target_points=target)
        controls = tuple(int(value) for value in generation["local_shape_control_indices"])
        control_offsets = sample_local_shape_offsets(
            rng,
            control_indices=controls,
            protected_indices=protected,
            magnitude_range=profile["local_shape_control_offset"],
        )
        local_offsets = interpolate_local_offsets(
            point_count=target,
            control_indices=controls,
            control_offsets=control_offsets,
            protected_indices=protected,
        )
        local_points = fitted + local_offsets
        projection_parameters = _sample_projection_parameters(rng, profile)
        matrix = compose_projective_matrix(**projection_parameters)
        projected, denominator = apply_projective(
            local_points,
            matrix,
            denominator_epsilon=float(generation["base_span_epsilon"]),
        )
        adapter_input, corruption_parameters, realized = _corrupt_observations(
            projected,
            denominator,
            corruption_replay_key=corruption_replay_key,
            rng=rng,
            profile=profile,
            generation=generation,
            camera_config=camera_config,
        )
    except CameraCorruptionError as exc:
        failed = {
            **base,
            "projection_matrix": None,
            "projection_parameters": None,
            "corruption_parameters": None,
            "realized_missing_indices": [],
            "contiguous_gap": None,
            "crop_edge": None,
            "frame_ids": [],
            "qc_status": "unavailable",
            "qc_reason_codes": [str(exc)],
            "semantic_status": "not_run_qc_unavailable",
            "semantic_reason_codes": [],
            "clean_metrics": None,
            "corrupted_metrics": None,
            "selected_for_training": False,
        }
        return CorruptedCameraView(failed, None, None, None)

    qc_result = run_camera_qc(adapter_input, camera_config)
    prepared: Mapping[str, Any] | None = None
    if len(qc_result.ready_inputs) == 1:
        prepared = prepare_camera_window(
            qc_result.ready_inputs[0], camera_config, preprocessing_config, feature_stats
        )
        qc_status = "ready"
        qc_reasons: list[str] = []
    else:
        qc_status = "unavailable"
        qc_reasons = _qc_reason_codes(qc_result)
    attempt = {
        **base,
        "projection_matrix": matrix.tolist(),
        "projection_parameters": projection_parameters,
        "corruption_parameters": {
            **corruption_parameters,
            "control_indices": list(controls),
            "control_offsets": control_offsets.tolist(),
            "protected_indices": sorted(protected),
            "smoothed_local_offsets": local_offsets.tolist(),
        },
        "realized_missing_indices": realized["missing_indices"],
        "contiguous_gap": realized["contiguous_gap"],
        "crop_edge": realized["crop_edge"],
        "frame_ids": realized["frame_ids"],
        "qc_status": qc_status,
        "qc_reason_codes": qc_reasons,
        "semantic_status": "pending" if prepared is not None else "not_run_qc_unavailable",
        "semantic_reason_codes": [],
        "clean_metrics": None,
        "corrupted_metrics": None,
        "selected_for_training": False,
    }
    return CorruptedCameraView(attempt, adapter_input, qc_result, prepared)


def build_camera_adapter_input(
    points: np.ndarray,
    heights: np.ndarray,
    *,
    record_id: str,
    camera_config: Mapping[str, Any],
    observed_mask: np.ndarray | None = None,
    camera_motion_state: str = "not_checked",
    track_id: int = 1,
) -> CameraAdapterInput:
    """Build an in-memory synthetic adapter input for corruption/fault tests."""

    point_array = _points(points)
    height_array = np.asarray(heights, dtype=np.float64)
    if height_array.shape != (len(point_array),) or np.any(height_array <= 0.0) or not np.isfinite(height_array).all():
        raise CameraCorruptionError("synthetic bbox heights are invalid")
    if observed_mask is None:
        observed = np.ones(len(point_array), dtype=bool)
    else:
        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != (len(point_array),):
            raise CameraCorruptionError("synthetic observed mask is invalid")
    width, height = 1920, 1080
    rows: list[dict[str, Any]] = []
    observations: list[CameraObservation] = []
    scope = ("synthetic-step8", record_id[:64], "virtual-camera", "step8-v1", "epoch-0", track_id)
    for index, (point, bbox_height) in enumerate(zip(point_array, height_array, strict=True)):
        if not observed[index]:
            continue
        bbox_width = 0.40 * float(bbox_height)
        bbox_norm = (
            float(point[0] - bbox_width / 2.0),
            float(point[1] - bbox_height),
            float(point[0] + bbox_width / 2.0),
            float(point[1]),
        )
        if not (0.0 <= bbox_norm[0] < bbox_norm[2] <= 1.0 and 0.0 <= bbox_norm[1] < bbox_norm[3] <= 1.0):
            raise CameraCorruptionError("synthetic fault bbox left the virtual canvas")
        bbox_pixel = (
            bbox_norm[0] * width,
            bbox_norm[1] * height,
            bbox_norm[2] * width,
            bbox_norm[3] * height,
        )
        timestamp = 0.25 + 0.5 * index
        frame_id = int(math.floor(30.0 * timestamp + 0.5))
        row = {
            "bbox": [float(value) for value in bbox_pixel],
            "frame_id": frame_id,
            "timestamp_sec": frame_id / 30.0,
            "track_confidence": 1.0,
            "track_id": track_id,
        }
        rows.append(row)
        observations.append(
            CameraObservation(
                scope_key=scope,
                source_group_id=scope[0],
                source_video_id=scope[1],
                device_id=scope[2],
                setup_id=scope[3],
                stream_epoch=scope[4],
                track_id=track_id,
                frame_id=frame_id,
                timestamp_sec=frame_id / 30.0,
                bbox_xyxy_pixel=tuple(bbox_pixel),
                bbox_xyxy_norm=bbox_norm,
                bbox_bottom_point=(float(point[0]), float(point[1])),
                bbox_height=float(bbox_height),
                track_confidence=1.0,
            )
        )
    if not rows:
        raise CameraCorruptionError("synthetic adapter input contains no observations")
    payload = canonical_jsonl_bytes(rows)
    tracking_sha = hashlib.sha256(payload).hexdigest()
    media = _synthetic_sidecar(
        record_id=record_id,
        tracking_sha256=tracking_sha,
        camera_motion_state=camera_motion_state,
    )
    media = validate_media_sidecar(media, camera_config)
    return CameraAdapterInput(
        media_sidecar=media,
        observations=tuple(observations),
        normalized_rows=tuple(rows),
        source_tracking_sha256=tracking_sha,
        normalized_tracking_sha256=tracking_sha,
    )


def _sample_projection_parameters(
    rng: np.random.Generator, profile: Mapping[str, Any]
) -> dict[str, float]:
    rotation = _uniform(rng, profile["rotation_degrees"])
    uniform = _uniform(rng, profile["uniform_scale"])
    anisotropic_x = _uniform(rng, profile["anisotropic_scale"])
    anisotropic_y = _uniform(rng, profile["anisotropic_scale"])
    shear_x = _uniform(rng, profile["shear"])
    shear_y = _uniform(rng, profile["shear"])
    projective_x = _uniform(rng, profile["projective"])
    projective_y = _uniform(rng, profile["projective"])
    translation_x = _uniform(rng, profile["translation_xy"])
    translation_y = _uniform(rng, profile["translation_xy"])
    return {
        "theta_degrees": rotation,
        "uniform_scale": uniform,
        "anisotropic_x": anisotropic_x,
        "anisotropic_y": anisotropic_y,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "projective_x": projective_x,
        "projective_y": projective_y,
        "translation_x": translation_x,
        "translation_y": translation_y,
    }


def _corrupt_observations(
    projected: np.ndarray,
    denominator: np.ndarray,
    *,
    corruption_replay_key: str,
    rng: np.random.Generator,
    profile: Mapping[str, Any],
    generation: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> tuple[CameraAdapterInput, dict[str, Any], dict[str, Any]]:
    count = len(projected)
    rho = _uniform(rng, profile["ar1_rho"])
    position_sigma = _uniform(rng, profile["position_sigma"])
    position_noise = sample_ar1_noise(
        rng, count=count, rho=rho, stationary_sigma=position_sigma, dimensions=2
    )
    positions = projected + position_noise
    height_low, height_high = (float(value) for value in generation["virtual_height_clip"])
    projective_heights = np.clip(
        float(generation["virtual_reference_height"]) / np.abs(denominator),
        height_low,
        height_high,
    )
    height_sigma = _uniform(rng, profile["bbox_height_noise_sigma"])
    height_noise = sample_ar1_noise(
        rng, count=count, rho=rho, stationary_sigma=height_sigma, dimensions=1
    )[:, 0]
    noisy_heights = np.clip(projective_heights + height_noise, height_low, height_high)
    bottom_offset = _uniform(rng, profile["bbox_bottom_offset_fraction"])
    bottom_sign = -1 if int(rng.integers(0, 2)) == 0 else 1
    positions[:, 1] += bottom_sign * bottom_offset * noisy_heights

    missing_fraction = _uniform(rng, profile["random_missing_fraction"])
    missing_count = int(math.floor(count * missing_fraction + 0.5))
    random_missing = (
        sorted(int(value) for value in rng.choice(count, size=missing_count, replace=False))
        if missing_count
        else []
    )
    gap_low, gap_high = (int(value) for value in profile["contiguous_missing_buckets"])
    gap_length = int(rng.integers(gap_low, gap_high + 1))
    gap_start = int(rng.integers(0, count - gap_length + 1))
    gap_indices = list(range(gap_start, gap_start + gap_length))
    missing = sorted(set(random_missing) | set(gap_indices))

    crop_fraction = _uniform(rng, profile["crop_fraction"])
    crop_index = int(rng.integers(0, 4))
    crop_edge = ("left", "right", "top", "bottom")[crop_index]
    quantized_height = int(rng.choice(np.asarray(profile["quantized_heights"], dtype=np.int64)))
    quantized_width = int(16 * quantized_height / 9)
    jitter_low, jitter_high = _numeric_range(profile["timestamp_jitter_seconds"], "timestamp_jitter_seconds")
    jitters = np.asarray(rng.uniform(jitter_low, jitter_high, size=count), dtype=np.float64)
    nominal_times = 0.25 + float(generation["synthetic_bucket_seconds"]) * np.arange(count)
    frame_ids = quantize_half_up(float(generation["synthetic_video_fps"]) * (nominal_times + jitters))
    timestamps = frame_ids.astype(np.float64) / float(generation["synthetic_video_fps"])
    if (
        np.any(timestamps < 0.0)
        or np.any(timestamps >= float(generation["synthetic_video_duration_seconds"]))
        or np.any(np.diff(frame_ids) <= 0)
        or not np.array_equal(
            np.floor(timestamps / float(generation["synthetic_bucket_seconds"])).astype(int),
            np.arange(count),
        )
    ):
        raise CameraCorruptionError("synthetic_timestamp_invalid")

    rows: list[dict[str, Any]] = []
    observations: list[CameraObservation] = []
    omitted_degenerate: list[int] = []
    omitted_out_of_view: list[int] = []
    scope = (
        "synthetic-step8",
        corruption_replay_key,
        "virtual-camera",
        "step8-v1",
        "epoch-0",
        1,
    )
    for index in range(count):
        if index in missing:
            continue
        bbox_height = float(noisy_heights[index])
        bbox_width = float(generation["bbox_width_to_height"]) * bbox_height
        x, y = (float(value) for value in positions[index])
        bbox = np.asarray([x - bbox_width / 2.0, y - bbox_height, x + bbox_width / 2.0, y])
        pixels = bbox * np.asarray(
            [quantized_width, quantized_height, quantized_width, quantized_height]
        )
        quantized = quantize_half_up(pixels).astype(np.float64)
        normalized = quantized / np.asarray(
            [quantized_width, quantized_height, quantized_width, quantized_height]
        )
        x1, y1, x2, y2 = (float(value) for value in normalized)
        if x1 >= x2 or y1 >= y2:
            omitted_degenerate.append(index)
            continue
        if not _bbox_visible((x1, y1, x2, y2), crop_fraction=crop_fraction, crop_edge=crop_edge):
            omitted_out_of_view.append(index)
            continue
        virtual_pixel = (x1 * 1920.0, y1 * 1080.0, x2 * 1920.0, y2 * 1080.0)
        bottom = ((x1 + x2) / 2.0, y2)
        quantized_bbox_height = y2 - y1
        row = {
            "bbox": [float(value) for value in virtual_pixel],
            "frame_id": int(frame_ids[index]),
            "timestamp_sec": float(timestamps[index]),
            "track_confidence": float(generation["observed_track_confidence"]),
            "track_id": 1,
        }
        rows.append(row)
        observations.append(
            CameraObservation(
                scope_key=scope,
                source_group_id=scope[0],
                source_video_id=scope[1],
                device_id=scope[2],
                setup_id=scope[3],
                stream_epoch=scope[4],
                track_id=1,
                frame_id=int(frame_ids[index]),
                timestamp_sec=float(timestamps[index]),
                bbox_xyxy_pixel=virtual_pixel,
                bbox_xyxy_norm=(x1, y1, x2, y2),
                bbox_bottom_point=bottom,
                bbox_height=quantized_bbox_height,
                track_confidence=float(generation["observed_track_confidence"]),
            )
        )
    if not rows:
        raise CameraCorruptionError("all_tracking_rows_omitted")
    tracking_payload = canonical_jsonl_bytes(rows)
    tracking_sha = hashlib.sha256(tracking_payload).hexdigest()
    media = validate_media_sidecar(
        _synthetic_sidecar(
            record_id=corruption_replay_key,
            tracking_sha256=tracking_sha,
            camera_motion_state="not_checked",
        ),
        camera_config,
    )
    adapter = CameraAdapterInput(
        media_sidecar=media,
        observations=tuple(observations),
        normalized_rows=tuple(rows),
        source_tracking_sha256=tracking_sha,
        normalized_tracking_sha256=tracking_sha,
    )
    parameters = {
        "rho": rho,
        "position_sigma": position_sigma,
        "position_noise": position_noise.tolist(),
        "projective_bbox_heights": projective_heights.tolist(),
        "bbox_height_noise_sigma": height_sigma,
        "bbox_height_noise": height_noise.tolist(),
        "noisy_bbox_heights": noisy_heights.tolist(),
        "bbox_bottom_offset_fraction": bottom_offset,
        "bbox_bottom_offset_sign": bottom_sign,
        "random_missing_fraction": missing_fraction,
        "random_missing_indices": random_missing,
        "contiguous_missing_indices": gap_indices,
        "crop_fraction": crop_fraction,
        "quantized_height": quantized_height,
        "quantized_width": quantized_width,
        "timestamp_jitter_seconds": jitters.tolist(),
        "omitted_quantized_degenerate_indices": omitted_degenerate,
        "omitted_out_of_view_indices": omitted_out_of_view,
        "synthetic_bucket_time": True,
        "temporal_features_enabled": False,
    }
    realized = {
        "missing_indices": missing,
        "contiguous_gap": {"start": gap_start, "length": gap_length},
        "crop_edge": crop_edge,
        "frame_ids": frame_ids.astype(int).tolist(),
    }
    return adapter, parameters, realized


def _attempt_base(
    parent: Mapping[str, Any],
    child_id: str,
    corruption_replay_key: str,
    child_seed: int,
    severity: str,
    view_role: str,
    attempt_index: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    labels = {
        "four_class": parent["pattern_label"] if parent["pattern_supervision_eligible"] else None,
        "binary": int(parent["binary_label"]) if parent["binary_supervision_eligible"] else None,
    }
    return {
        "schema_version": config["output_schemas"]["attempt"],
        "attempt_id": f"attempt-{child_id}",
        "child_id": child_id,
        "parent_sample_id": parent["sample_id"],
        "source_dataset": parent["source_dataset"],
        "split": parent["split"],
        "labels": labels,
        "view_role": view_role,
        "severity": severity,
        "attempt_index": attempt_index,
        "corruption_replay_key": corruption_replay_key,
        "child_seed": child_seed,
        "preprocessing_manifest_sha256": config["trust_roots"]["preprocessing_manifest"]["sha256"],
        "camera_config_sha256": config["trust_roots"]["camera_config"]["sha256"],
        "augmentation_config_sha256": augmentation_config_sha256(config),
        "validation_scope": "synthetic_camera_corruption",
    }


def _validate_parent(parent: Mapping[str, Any], *, target: int, config: Mapping[str, Any]) -> None:
    if parent.get("preprocess_status") != "ready":
        raise CameraCorruptionError("unavailable parent cannot be augmented")
    if parent.get("split") not in config["split_policy"]["development_splits"]:
        raise CameraCorruptionError("parent split is outside development scope")
    if parent.get("split") in config["split_policy"]["forbidden_splits"]:
        raise CameraCorruptionError("test or sealed parent is forbidden")
    if parent.get("source_dataset") not in {"wandering_patterns", "smartcare"}:
        raise CameraCorruptionError("unknown public development source")
    points = np.asarray(parent.get("shape_normalized_points"), dtype=np.float64)
    mask = np.asarray(parent.get("point_mask"))
    raw = np.asarray(parent.get("raw_features"), dtype=np.float64)
    if points.shape != (target, 2) or mask.shape != (target,) or raw.shape != (target, 14):
        raise CameraCorruptionError("ready parent arrays have drifted")
    if not np.isfinite(points).all() or not np.isfinite(raw).all() or not np.array_equal(mask, np.ones(target)):
        raise CameraCorruptionError("ready parent arrays are non-finite or masked")
    if not np.array_equal(raw[:, 13], np.ones(target)):
        raise CameraCorruptionError("ready parent quality channel has drifted")


def _qc_reason_codes(result: CameraQCResult) -> list[str]:
    reasons: list[str] = []
    for row in result.tracklet_records:
        reasons.extend(str(value) for value in row.get("quality_flags", []))
    for row in result.window_records:
        reasons.extend(str(value) for value in row.get("reason_codes", []))
    if not result.window_records:
        reasons.append("no_complete_window")
    return list(dict.fromkeys(reasons))


def _synthetic_sidecar(
    *, record_id: str, tracking_sha256: str, camera_motion_state: str
) -> dict[str, Any]:
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": record_id[:64],
        "source_group_id": "synthetic-step8",
        "device_id": "virtual-camera",
        "setup_id": "step8-v1",
        "stream_epoch": "epoch-0",
        "media_ref": f"synthetic/{record_id}.mp4",
        "source_sha256": hashlib.sha256(f"source|{record_id}".encode()).hexdigest(),
        "tracking_jsonl_sha256": tracking_sha256,
        "video_width": 1920,
        "video_height": 1080,
        "nominal_fps": 30.0,
        "duration_sec": 40.0,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {"backend": "synthetic", "model": "step8-fixture", "version": "v1"},
        "tracker": {"backend": "synthetic", "config": "step8-fixture", "version": "v1"},
        "fixed_camera_assumed": True,
        "camera_motion_state": camera_motion_state,
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic_no_person_data",
    }


def _bbox_visible(
    bbox: tuple[float, float, float, float], *, crop_fraction: float, crop_edge: str
) -> bool:
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return False
    if crop_edge == "left":
        return x1 >= crop_fraction
    if crop_edge == "right":
        return x2 <= 1.0 - crop_fraction
    if crop_edge == "top":
        return y1 >= crop_fraction
    if crop_edge == "bottom":
        return y2 <= 1.0 - crop_fraction
    raise CameraCorruptionError("unknown crop edge")


def _translation(x: float, y: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _uniform(rng: np.random.Generator, value: Sequence[float]) -> float:
    low, high = _numeric_range(value, "profile range")
    return float(rng.uniform(low, high))


def _numeric_range(value: Sequence[float], field: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise CameraCorruptionError(f"{field} must be a two-number range")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low + high) or low > high:
        raise CameraCorruptionError(f"{field} range is invalid")
    return low, high


def _points(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 1 or not np.isfinite(array).all():
        raise CameraCorruptionError("points must be finite shape [N,2], N>=1")
    return array


def _mask(value: Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or not np.all(np.isin(array, [0, 1])):
        raise CameraCorruptionError("point mask must be binary shape [N]")
    return array.astype(bool)


def _validate_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CameraCorruptionError(f"{field} must be a portable relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or any(part in ("", ".", "..") for part in posix.parts):
        raise CameraCorruptionError(f"{field} must be a safe relative path")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CameraCorruptionError(f"{field} must be a lowercase SHA-256")


def augmentation_config_sha256(config: Mapping[str, Any]) -> str:
    schema = config.get("schema_version")
    if schema == AUGMENTATION_CONFIG_SCHEMA_VERSION_V1:
        return AUGMENTATION_CONFIG_SHA256_V1
    if schema == AUGMENTATION_CONFIG_SCHEMA_VERSION_V2:
        return AUGMENTATION_CONFIG_SHA256_V2
    if schema == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        return AUGMENTATION_CONFIG_SHA256_V3
    raise CameraCorruptionError("unsupported augmentation config schema")


def _verify_diagnostic_predecessors(config: Mapping[str, Any], root: Path) -> None:
    schema = config["schema_version"]
    predecessor = (
        config["diagnostic_predecessor"]
        if schema == AUGMENTATION_CONFIG_SCHEMA_VERSION_V2
        else config["diagnostic_predecessors"]["v1"]
    )
    paths = {
        "config_sha256": root / "configs/modules/wandering_augmentation_v1.yaml",
        "manifest_sha256": root / "data/processed/wandering/augmentation/v1/manifest.json",
        "report_sha256": root / "data/processed/wandering/augmentation/v1/augmentation_report.json",
    }
    for field, path in paths.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != predecessor[field]:
            raise CameraCorruptionError(f"diagnostic v1 predecessor hash mismatch: {field}")
    if predecessor["output_reuse_allowed"] is not False:
        raise CameraCorruptionError("diagnostic v1 output reuse must remain forbidden")
    if schema == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        v2 = config["diagnostic_predecessors"]["v2"]
        v2_config = root / "configs/modules/wandering_augmentation_v2.yaml"
        if (
            not v2_config.is_file()
            or hashlib.sha256(v2_config.read_bytes()).hexdigest() != v2["config_sha256"]
        ):
            raise CameraCorruptionError("diagnostic v2 predecessor config hash mismatch")
        if (root / "data/processed/wandering/augmentation/v2").exists():
            raise CameraCorruptionError("diagnostic v2 formal output must remain absent")
        if v2["formal_output_created"] is not False or v2["output_reuse_allowed"] is not False:
            raise CameraCorruptionError("diagnostic v2 output flags have drifted")


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "AUGMENTATION_CONFIG_SCHEMA_VERSION",
    "AUGMENTATION_CONFIG_SCHEMA_VERSION_V1",
    "AUGMENTATION_CONFIG_SCHEMA_VERSION_V2",
    "AUGMENTATION_CONFIG_SCHEMA_VERSION_V3",
    "AUGMENTATION_CONFIG_SHA256",
    "AUGMENTATION_CONFIG_SHA256_V1",
    "AUGMENTATION_CONFIG_SHA256_V2",
    "AUGMENTATION_CONFIG_SHA256_V3",
    "CameraCorruptionError",
    "CorruptedCameraView",
    "apply_projective",
    "augmentation_config_sha256",
    "build_camera_adapter_input",
    "compose_projective_matrix",
    "derive_child_identity",
    "derive_fault_identity",
    "derive_versioned_child_identity",
    "derive_versioned_fault_identity",
    "fit_virtual_canvas",
    "generate_corrupted_camera_view",
    "interpolate_local_offsets",
    "load_augmentation_config",
    "protected_point_indices",
    "quantize_half_up",
    "sample_ar1_noise",
    "sample_local_shape_offsets",
    "verify_augmentation_trust_roots",
]
