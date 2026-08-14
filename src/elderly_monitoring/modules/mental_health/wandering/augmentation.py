"""Semantic gates and deterministic bundle construction for wandering step 8."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    AUGMENTATION_CONFIG_SCHEMA_VERSION_V2,
    AUGMENTATION_CONFIG_SCHEMA_VERSION_V3,
    CameraCorruptionError,
    augmentation_config_sha256,
    build_camera_adapter_input,
    derive_versioned_fault_identity,
    fit_virtual_canvas,
    generate_corrupted_camera_view,
    load_augmentation_config,
    verify_augmentation_trust_roots,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import run_camera_qc
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.topology import (
    compute_topology,
    compute_turn_geometry,
)


AUGMENTED_PAIR_SCHEMA_VERSION = "wandering-augmented-pair-v1"
PRESSURE_SCHEMA_VERSION = "wandering-corruption-pressure-v1"
FAULT_SCHEMA_VERSION = "wandering-qc-fault-v1"
REPORT_SCHEMA_VERSION = "wandering-augmentation-report-v1"
MANIFEST_SCHEMA_VERSION = "wandering-augmentation-manifest-v1"

_MACHINE_FILE_ORDER = (
    "augmentation_attempts.jsonl",
    "train_pairs.jsonl",
    "validation_pressure.jsonl",
    "qc_faults.jsonl",
    "augmentation_report.json",
)
_WP_LABELS = ("direct", "pacing", "lapping", "random")
_SEVERITIES = ("low", "medium", "high")
_STEP8_SOURCE_PATHS = (
    "src/elderly_monitoring/modules/mental_health/wandering/camera_corruption.py",
    "src/elderly_monitoring/modules/mental_health/wandering/augmentation.py",
    "src/elderly_monitoring/modules/mental_health/wandering/compatibility_report.py",
)


class AugmentationBuildError(ValueError):
    """The step-8 bundle violates a frozen input, count, or output contract."""


@dataclass(frozen=True)
class SemanticCheckResult:
    status: str
    reason_codes: tuple[str, ...]
    clean_metrics: Mapping[str, Any] | None
    corrupted_metrics: Mapping[str, Any] | None
    assigned_subtype: None = None


@dataclass(frozen=True)
class AugmentationBuildResult:
    output_dir: Path
    manifest_sha256: str
    attempt_count: int
    train_pair_count: int
    validation_pressure_count: int
    gated_fault_count: int
    limitation_control_count: int


def coverage_entropy_pair(
    clean_points: Sequence[Sequence[float]] | np.ndarray,
    clean_mask: Sequence[int] | np.ndarray,
    corrupted_points: Sequence[Sequence[float]] | np.ndarray,
    corrupted_mask: Sequence[int] | np.ndarray,
    *,
    grid_size: int,
    epsilon: float,
) -> tuple[float, float]:
    """Compute two normalized entropies using one clean/corrupted boundary."""

    clean, clean_valid = _valid_points(clean_points, clean_mask)
    corrupted, corrupted_valid = _valid_points(corrupted_points, corrupted_mask)
    joint = np.vstack((clean[clean_valid], corrupted[corrupted_valid]))
    low = joint.min(axis=0)
    span = joint.max(axis=0) - low

    def entropy(points: np.ndarray, valid: np.ndarray) -> float:
        values = points[valid]
        bins = np.zeros_like(values, dtype=np.int64)
        for axis in range(2):
            if span[axis] <= epsilon:
                bins[:, axis] = 0
            else:
                scaled = (values[:, axis] - low[axis]) / span[axis]
                bins[:, axis] = np.minimum(
                    np.floor(scaled * grid_size).astype(np.int64), grid_size - 1
                )
        flat = bins[:, 1] * grid_size + bins[:, 0]
        counts = np.bincount(flat, minlength=grid_size * grid_size)
        probability = counts[counts > 0].astype(np.float64) / len(values)
        return float(-(probability * np.log(probability)).sum() / math.log(grid_size * grid_size))

    return entropy(clean, clean_valid), entropy(corrupted, corrupted_valid)


def endpoint_repeat_rate(
    points: Sequence[Sequence[float]] | np.ndarray,
    point_mask: Sequence[int] | np.ndarray,
    *,
    reference_points: int,
    query_points: int,
    radius: float,
) -> float:
    """Compare the final ten valid points only with the first ten valid points."""

    array, valid = _valid_points(points, point_mask)
    selected = array[valid]
    if len(selected) < reference_points + query_points:
        raise AugmentationBuildError("endpoint repeat sets must be non-overlapping")
    reference = selected[:reference_points]
    query = selected[-query_points:]
    minimum = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    return float(np.mean(minimum <= float(radius)))


def semantic_check(
    parent: Mapping[str, Any],
    corrupted_points: Sequence[Sequence[float]] | np.ndarray,
    corrupted_mask: Sequence[int] | np.ndarray,
    *,
    config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    qc_ready: bool,
) -> SemanticCheckResult:
    """Apply only the frozen common and source-label-specific semantic gates."""

    if not qc_ready:
        return SemanticCheckResult(
            "not_run_qc_unavailable", ("camera_qc_unavailable",), None, None
        )
    semantic = config["semantic_check"]
    clean = np.asarray(parent.get("shape_normalized_points"), dtype=np.float64)
    clean_mask = np.asarray(parent.get("point_mask"))
    corrupted = np.asarray(corrupted_points, dtype=np.float64)
    mask = np.asarray(corrupted_mask)
    if clean.shape != (80, 2) or corrupted.shape != (80, 2):
        return SemanticCheckResult("rejected", ("invalid_shape",), None, None)
    if (
        clean_mask.shape != (80,)
        or mask.shape != (80,)
        or not np.all(np.isin(clean_mask, [0, 1]))
        or not np.all(np.isin(mask, [0, 1]))
        or not np.isfinite(clean).all()
        or not np.isfinite(corrupted).all()
    ):
        return SemanticCheckResult("rejected", ("invalid_finite_mask",), None, None)
    clean_entropy, corrupted_entropy = coverage_entropy_pair(
        clean,
        clean_mask,
        corrupted,
        mask,
        grid_size=int(semantic["coverage_grid_size"]),
        epsilon=float(semantic["metric_epsilon"]),
    )
    clean_metrics = _trajectory_metrics(
        clean,
        clean_mask,
        coverage_entropy=clean_entropy,
        semantic=semantic,
        preprocessing_config=preprocessing_config,
    )
    corrupted_metrics = _trajectory_metrics(
        corrupted,
        mask,
        coverage_entropy=corrupted_entropy,
        semantic=semantic,
        preprocessing_config=preprocessing_config,
    )
    reasons = _common_semantic_reasons(clean_metrics, corrupted_metrics, semantic)
    source = parent.get("source_dataset")
    if source == "wandering_patterns":
        label = parent.get("pattern_label")
        if label not in _WP_LABELS or parent.get("pattern_supervision_eligible") is not True:
            reasons.append("invalid_wp_parent_label")
        else:
            reasons.extend(_wp_semantic_reasons(label, clean_metrics, corrupted_metrics, semantic))
    elif source == "smartcare":
        if parent.get("pattern_label") != "unknown" or parent.get("pattern_supervision_eligible") is not False:
            reasons.append("smartcare_pseudo_subtype_forbidden")
    else:
        reasons.append("unknown_source_dataset")
    unique = tuple(dict.fromkeys(reasons))
    return SemanticCheckResult(
        "accepted" if not unique else "rejected",
        unique,
        clean_metrics,
        corrupted_metrics,
    )


def pacing_reversal_reason_codes(
    clean_reversals: int,
    corrupted_reversals: int,
    *,
    max_lost_from_clean: int,
    max_added_to_clean: int,
    lost_reason_code: str = "pacing_reversal_lost",
    added_reason_code: str = "pacing_excess_reversals",
) -> tuple[str, ...]:
    """Apply the v2 parent-relative pacing reversal gate."""

    values = (
        clean_reversals,
        corrupted_reversals,
        max_lost_from_clean,
        max_added_to_clean,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise AugmentationBuildError("pacing reversal counts/deltas must be nonnegative integers")
    if not isinstance(lost_reason_code, str) or not lost_reason_code:
        raise AugmentationBuildError("pacing lost reason code is invalid")
    if not isinstance(added_reason_code, str) or not added_reason_code:
        raise AugmentationBuildError("pacing added reason code is invalid")
    lower = max(0, clean_reversals - max_lost_from_clean)
    upper = clean_reversals + max_added_to_clean
    reasons: list[str] = []
    if corrupted_reversals < lower:
        reasons.append(lost_reason_code)
    if corrupted_reversals > upper:
        reasons.append(added_reason_code)
    return tuple(reasons)


def wp_train_pacing_reversal_histogram(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Recompute the frozen Step-4 WP-train pacing compatibility fact."""

    counts: Counter[int] = Counter()
    for row in rows:
        if (
            row.get("source_dataset") != "wandering_patterns"
            or row.get("split") != "train"
            or row.get("pattern_label") != "pacing"
            or row.get("pattern_supervision_eligible") is not True
            or row.get("preprocess_status") != "ready"
        ):
            raise AugmentationBuildError("pacing histogram input contains a non-WP-train pacing parent")
        topology = row.get("topology")
        events = topology.get("reversal_events") if isinstance(topology, Mapping) else None
        if not isinstance(events, list):
            raise AugmentationBuildError("pacing parent reversal_events are invalid")
        counts[len(events)] += 1
    return {str(key): int(value) for key, value in sorted(counts.items())}


def commit_new_output_directory(output_dir: str | Path, files: Mapping[str, bytes]) -> None:
    """Atomically create one fresh output directory and refuse all overwrites."""

    output = Path(output_dir)
    if output.exists():
        raise AugmentationBuildError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    committed = False
    try:
        for relative_name, payload in files.items():
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
                raise AugmentationBuildError("output artifact path is not a safe relative POSIX path")
            if not isinstance(payload, bytes):
                raise AugmentationBuildError("output artifact payload must be bytes")
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        os.replace(temporary, output)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def build_augmentation_bundle(
    *,
    config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> AugmentationBuildResult:
    """Build the train pairs, frozen pressure views, and QC fault suite."""

    output = Path(output_dir)
    if output.exists():
        raise AugmentationBuildError(f"output directory already exists: {output}")
    try:
        config = load_augmentation_config(config_path)
        _validate_output_target(output, Path(project_root), config)
        trusted = verify_augmentation_trust_roots(config, project_root)
        camera_config = load_camera_config(trusted["camera_config"])
        preprocessing_config = load_preprocessing_config(trusted["preprocessing_config"])
        feature_stats = _load_finite_json(trusted["preprocessing_feature_stats"])
        bundle = load_preprocessing_bundle(
            rf_config_path=trusted["rf_config"],
            project_root=project_root,
            mode=BUNDLE_MODE_DEVELOPMENT,
        )
    except (CameraCorruptionError, OSError, ValueError) as exc:
        raise AugmentationBuildError("step-8 trust chain or development bundle failed") from exc

    train = tuple(sorted(bundle.records_for_split("train"), key=lambda row: str(row["sample_id"])))
    validation = tuple(
        sorted(bundle.records_for_split("validation"), key=lambda row: str(row["sample_id"]))
    )
    _validate_parent_cohorts(train, validation, config)
    if config["schema_version"] in {
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V2,
        AUGMENTATION_CONFIG_SCHEMA_VERSION_V3,
    }:
        pacing_parents = [
            row
            for row in train
            if row["source_dataset"] == "wandering_patterns"
            and row["pattern_label"] == "pacing"
        ]
        actual_histogram = wp_train_pacing_reversal_histogram(pacing_parents)
        expected_histogram = config["semantic_check"]["pacing_reversal_gate"][
            "expected_wp_train_parent_histogram"
        ]
        if actual_histogram != expected_histogram or sum(actual_histogram.values()) != 280:
            raise AugmentationBuildError("WP train pacing reversal histogram drifted")
    else:
        actual_histogram = None

    attempts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pressure: list[dict[str, Any]] = []
    for parent in train:
        for severity in config["generation"]["train_severities"]:
            selected = False
            for attempt_index in range(train_attempt_budget(config, str(severity))):
                view = generate_corrupted_camera_view(
                    parent,
                    severity=str(severity),
                    view_role="train_pair",
                    attempt_index=attempt_index,
                    config=config,
                    camera_config=camera_config,
                    preprocessing_config=preprocessing_config,
                    feature_stats=feature_stats,
                )
                attempt, semantic = _evaluate_view_semantics(
                    parent,
                    view.attempt_record,
                    view.prepared_window,
                    config=config,
                    preprocessing_config=preprocessing_config,
                )
                if semantic.status == "accepted" and view.prepared_window is not None:
                    attempt["selected_for_training"] = True
                    pairs.append(
                        _pair_record(parent, attempt, view.prepared_window, config=config)
                    )
                    selected = True
                attempts.append(attempt)
                if selected:
                    break

    for parent in validation:
        for severity in config["generation"]["validation_severities"]:
            view = generate_corrupted_camera_view(
                parent,
                severity=str(severity),
                view_role="validation_pressure",
                attempt_index=0,
                config=config,
                camera_config=camera_config,
                preprocessing_config=preprocessing_config,
                feature_stats=feature_stats,
            )
            attempt, _semantic = _evaluate_view_semantics(
                parent,
                view.attempt_record,
                view.prepared_window,
                config=config,
                preprocessing_config=preprocessing_config,
            )
            attempts.append(attempt)
            pressure.append(
                _pressure_record(parent, attempt, view.prepared_window, config=config)
            )

    fault_records = _build_fault_suite(validation, config=config, camera_config=camera_config)
    _validate_output_counts(train, validation, pairs, pressure, fault_records, config)
    train_strata = _train_label_severity_counts(
        train=train, attempts=attempts, pairs=pairs, config=config
    )
    if config["schema_version"] == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        validate_train_selection_quotas(train_strata, config=config)
    elif config["schema_version"] == AUGMENTATION_CONFIG_SCHEMA_VERSION_V2:
        _validate_v2_pacing_selection_floor(pairs)
    attempts.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["parent_sample_id"]),
            str(row["view_role"]),
            _SEVERITIES.index(str(row["severity"])),
            int(row["attempt_index"]),
        )
    )
    pairs.sort(key=lambda row: (str(row["parent_sample_id"]), str(row["severity"])))
    pressure.sort(
        key=lambda row: (str(row["parent_sample_id"]), _SEVERITIES.index(str(row["severity"])))
    )
    fault_records.sort(
        key=lambda row: (
            str(row["parent_sample_id"]),
            0 if row["record_role"] == "gated_fault" else 1,
            str(row["fault_or_control_type"]),
        )
    )

    report = _augmentation_report(
        attempts=attempts,
        pairs=pairs,
        pressure=pressure,
        faults=fault_records,
        train=train,
        validation=validation,
        config=config,
        pacing_parent_histogram=actual_histogram,
        train_strata=train_strata,
    )
    files: dict[str, bytes] = {
        "augmentation_attempts.jsonl": canonical_jsonl_bytes(attempts),
        "train_pairs.jsonl": canonical_jsonl_bytes(pairs),
        "validation_pressure.jsonl": canonical_jsonl_bytes(pressure),
        "qc_faults.jsonl": canonical_jsonl_bytes(fault_records),
        "augmentation_report.json": canonical_json_bytes(report),
    }
    manifest = {
        "schema_version": config["output_schemas"]["manifest"],
        "purpose": "semantic_preserving_synthetic_camera_corruption",
        "validation_scope": "synthetic_camera_corruption",
        "augmentation_config_sha256": augmentation_config_sha256(config),
        "trust_roots": {name: dict(config["trust_roots"][name]) for name in config["trust_roots"]},
        "step8_source_hashes": _step8_source_hashes(Path(project_root)),
        "artifacts": _artifact_descriptors(files),
        "counts": {
            "development_train_parents": len(train),
            "development_validation_parents": len(validation),
            "attempts": len(attempts),
            "train_pairs": len(pairs),
            "validation_pressure": len(pressure),
            "gated_faults": sum(row["record_role"] == "gated_fault" for row in fault_records),
            "limitation_controls": sum(
                row["record_role"] == "limitation_control" for row in fault_records
            ),
        },
        "human_review_status": "pending_human_review",
        "official_source_opened": False,
        "test_split_opened": False,
        "model_training_performed": False,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    commit_new_output_directory(output, files)
    return AugmentationBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(files["manifest.json"]).hexdigest(),
        attempt_count=len(attempts),
        train_pair_count=len(pairs),
        validation_pressure_count=len(pressure),
        gated_fault_count=sum(row["record_role"] == "gated_fault" for row in fault_records),
        limitation_control_count=sum(
            row["record_role"] == "limitation_control" for row in fault_records
        ),
    )


def _validate_output_target(
    output: Path, project_root: Path, config: Mapping[str, Any]
) -> None:
    if config["schema_version"] != AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        raise AugmentationBuildError("v1/v2 augmentation configs are diagnostic and read-only")
    forbidden = {
        (project_root / f"data/processed/wandering/augmentation/{version}").resolve(
            strict=False
        )
        for version in ("v1", "v2")
    }
    if output.resolve(strict=False) in forbidden:
        raise AugmentationBuildError("v3 builder refuses diagnostic v1/v2 output targets")


def train_attempt_budget(config: Mapping[str, Any], severity: str) -> int:
    value = config["generation"]["max_train_attempts_per_severity"]
    if isinstance(value, Mapping):
        budget = value.get(severity)
    else:
        budget = value
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise AugmentationBuildError(f"invalid train attempt budget for {severity}")
    return budget


def _evaluate_view_semantics(
    parent: Mapping[str, Any],
    attempt_value: Mapping[str, Any],
    prepared: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
) -> tuple[dict[str, Any], SemanticCheckResult]:
    result = semantic_check(
        parent,
        prepared["shape_normalized_points"] if prepared is not None else parent["shape_normalized_points"],
        prepared["point_mask"] if prepared is not None else parent["point_mask"],
        config=config,
        preprocessing_config=preprocessing_config,
        qc_ready=prepared is not None,
    )
    attempt = {
        **dict(attempt_value),
        "semantic_status": result.status,
        "semantic_reason_codes": list(result.reason_codes),
        "clean_metrics": result.clean_metrics,
        "corrupted_metrics": result.corrupted_metrics,
        "selected_for_training": False,
    }
    required_fields = {
        "schema_version",
        "attempt_id",
        "child_id",
        "parent_sample_id",
        "source_dataset",
        "split",
        "labels",
        "view_role",
        "severity",
        "attempt_index",
        "corruption_replay_key",
        "child_seed",
        "projection_matrix",
        "projection_parameters",
        "corruption_parameters",
        "realized_missing_indices",
        "contiguous_gap",
        "crop_edge",
        "frame_ids",
        "qc_status",
        "qc_reason_codes",
        "semantic_status",
        "semantic_reason_codes",
        "clean_metrics",
        "corrupted_metrics",
        "selected_for_training",
        "preprocessing_manifest_sha256",
        "camera_config_sha256",
        "augmentation_config_sha256",
    }
    if not required_fields <= frozenset(attempt):
        raise AugmentationBuildError("augmentation attempt required fields are missing")
    return attempt, result


def _pair_record(
    parent: Mapping[str, Any],
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if attempt["split"] != "train" or attempt["severity"] not in {"low", "medium"}:
        raise AugmentationBuildError("only low/medium train views may become paired training data")
    if attempt["qc_status"] != "ready" or attempt["semantic_status"] != "accepted":
        raise AugmentationBuildError("selected train pair did not pass QC and semantics")
    payload = _prepared_payload(prepared)
    return {
        "schema_version": config["output_schemas"]["pair"],
        "child_id": attempt["child_id"],
        "parent_sample_id": attempt["parent_sample_id"],
        "source_dataset": attempt["source_dataset"],
        "split": "train",
        "labels": attempt["labels"],
        "severity": attempt["severity"],
        "attempt_index": attempt["attempt_index"],
        "corruption_replay_key": attempt["corruption_replay_key"],
        "child_seed": attempt["child_seed"],
        "clean_shape_normalized_points": parent["shape_normalized_points"],
        "clean_point_mask": parent["point_mask"],
        "clean_raw_features": parent["raw_features"],
        "clean_model_features": parent["model_features"],
        "clean_topology": parent["topology"],
        **payload,
        "synthetic_bucket_time": True,
        "temporal_features_enabled": False,
        "validation_scope": "synthetic_camera_corruption",
        "attempt_id": attempt["attempt_id"],
        "preprocessing_manifest_sha256": attempt["preprocessing_manifest_sha256"],
        "camera_config_sha256": attempt["camera_config_sha256"],
        "augmentation_config_sha256": attempt["augmentation_config_sha256"],
    }


def _pressure_record(
    parent: Mapping[str, Any],
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if attempt["split"] != "validation" or attempt["attempt_index"] != 0:
        raise AugmentationBuildError("pressure views must be one-shot validation records")
    record = {
        **dict(attempt),
        "schema_version": config["output_schemas"]["pressure"],
        "clean_shape_normalized_points": parent["shape_normalized_points"],
        "clean_point_mask": parent["point_mask"],
        "clean_raw_features": parent["raw_features"],
        "clean_model_features": parent["model_features"],
        "clean_topology": parent["topology"],
        "synthetic_bucket_time": True,
        "temporal_features_enabled": False,
    }
    record.update(_prepared_payload(prepared) if prepared is not None else _null_prepared_payload())
    return record


def _prepared_payload(prepared: Mapping[str, Any]) -> dict[str, Any]:
    raw = np.asarray(prepared["raw_features"], dtype=np.float64)
    model = np.asarray(prepared["model_features"], dtype=np.float64)
    if raw.shape != (80, 14) or model.shape != (80, 14):
        raise AugmentationBuildError("prepared step-4 feature shape drifted")
    if not np.array_equal(raw[:, 10:12], np.zeros((80, 2))) or not np.array_equal(
        model[:, 10:12], np.zeros((80, 2))
    ):
        raise AugmentationBuildError("step-8 temporal model channels must remain zero")
    return {
        "virtual_image_points": prepared["bbox_bottom_points"],
        "virtual_bbox_heights": prepared["bbox_heights"],
        "observed_mask": prepared["observed_mask"],
        "interpolated_mask": prepared["interpolated_mask"],
        "point_quality": prepared["point_quality"],
        "shape_normalized_points": prepared["shape_normalized_points"],
        "point_mask": prepared["point_mask"],
        "raw_features": prepared["raw_features"],
        "model_features": prepared["model_features"],
        "topology": prepared["topology"],
        "window_id": prepared["window_id"],
        "quality_flags": prepared["quality_flags"],
    }


def _null_prepared_payload() -> dict[str, Any]:
    return {
        "virtual_image_points": None,
        "virtual_bbox_heights": None,
        "observed_mask": None,
        "interpolated_mask": None,
        "point_quality": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "model_features": None,
        "topology": None,
        "window_id": None,
        "quality_flags": [],
    }


def _validate_parent_cohorts(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    policy = config["split_policy"]
    if len(train) != int(policy["expected_ready_train"]) or len(validation) != int(
        policy["expected_ready_validation"]
    ):
        raise AugmentationBuildError("development parent counts drifted")
    all_rows = tuple(train) + tuple(validation)
    if any(row["split"] not in {"train", "validation"} for row in all_rows):
        raise AugmentationBuildError("test/sealed record entered the step-8 parent cohort")
    if any(row["preprocess_status"] != "ready" for row in all_rows):
        raise AugmentationBuildError("unavailable record entered the step-8 parent cohort")
    wp_train = Counter(
        row["pattern_label"] for row in train if row["source_dataset"] == "wandering_patterns"
    )
    wp_validation = Counter(
        row["pattern_label"]
        for row in validation
        if row["source_dataset"] == "wandering_patterns"
    )
    expected_train = int(policy["expected_wp_four_class_train_per_class"])
    expected_validation = int(policy["expected_wp_four_class_validation_per_class"])
    if wp_train != Counter({label: expected_train for label in _WP_LABELS}):
        raise AugmentationBuildError("WP train class counts drifted")
    if wp_validation != Counter({label: expected_validation for label in _WP_LABELS}):
        raise AugmentationBuildError("WP validation class counts drifted")
    smart_train = Counter(
        int(row["binary_label"]) for row in train if row["source_dataset"] == "smartcare"
    )
    smart_validation = Counter(
        int(row["binary_label"])
        for row in validation
        if row["source_dataset"] == "smartcare"
    )
    if smart_train != Counter(
        {index: int(count) for index, count in enumerate(policy["expected_smartcare_train_binary"])}
    ):
        raise AugmentationBuildError("SmartCare train binary counts drifted")
    if smart_validation != Counter(
        {
            index: int(count)
            for index, count in enumerate(policy["expected_smartcare_validation_binary"])
        }
    ):
        raise AugmentationBuildError("SmartCare validation binary counts drifted")


def _build_fault_suite(
    validation: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    wp = [row for row in validation if row["source_dataset"] == "wandering_patterns"]
    smart = [row for row in validation if row["source_dataset"] == "smartcare"]
    for label in _WP_LABELS:
        selected.extend(
            _hash_order(row for row in wp if row["pattern_label"] == label)[
                : int(config["fault_suite"]["parents_per_wp_class"])
            ]
        )
    for label in (0, 1):
        selected.extend(
            _hash_order(row for row in smart if int(row["binary_label"]) == label)[
                : int(config["fault_suite"]["parents_per_smartcare_binary_class"])
            ]
        )
    if len(selected) != 48:
        raise AugmentationBuildError("fault-suite parent selection did not produce 48 records")
    records: list[dict[str, Any]] = []
    for parent in selected:
        donor = _select_donor(parent, validation)
        for fault_type in config["fault_suite"]["fault_types"]:
            records.append(
                _fault_record(
                    parent,
                    donor,
                    fault_type=str(fault_type),
                    config=config,
                    camera_config=camera_config,
                )
            )
        records.append(
            _fault_record(
                parent,
                donor,
                fault_type=str(config["fault_suite"]["limitation_control_types"][0]),
                config=config,
                camera_config=camera_config,
                control=True,
            )
        )
    for fault_type in config["fault_suite"]["fault_types"]:
        group = [
            row
            for row in records
            if row["record_role"] == "gated_fault"
            and row["fault_or_control_type"] == fault_type
        ]
        rejection_rate = sum(row["observed_qc_status"] == "unavailable" for row in group) / len(group)
        if rejection_rate < float(config["fault_suite"]["minimum_qc_rejection_rate_per_fault"]):
            raise AugmentationBuildError(f"QC rejection target failed for {fault_type}")
    return records


def build_qc_fault_suite(
    validation: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Public deterministic fault-suite entry point used by tests and the builder."""

    return _build_fault_suite(validation, config=config, camera_config=camera_config)


def _fault_record(
    parent: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    fault_type: str,
    config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    control: bool = False,
) -> dict[str, Any]:
    record_id, corruption_replay_key, seed = derive_versioned_fault_identity(
        parent_sample_id=str(parent["sample_id"]),
        fault_or_control_type=fault_type,
        control=control,
        config_schema_version=str(config["schema_version"]),
        global_seed=int(config["global_seed"]),
    )
    parent_points = _fault_canvas(parent)
    donor_points = _fault_canvas(donor)
    points = parent_points.copy()
    heights = np.full(80, 0.06, dtype=np.float64)
    observed = np.ones(80, dtype=bool)
    motion = "not_checked"
    expected: str | None
    donor_id: str | None = None
    boundary: dict[str, Any] | None = None
    if fault_type == "long_gap_4_buckets":
        observed[38:42] = False
        expected = "long_internal_gap"
    elif fault_type == "hard_jump_4_body_heights_per_second":
        points, direction, tail_scale = _translate_tail_to_body_step(
            points, points, body_step=2.0, height=0.06
        )
        expected = "suspected_id_switch"
        boundary = {"height_ratio": 1.0, "body_step": 2.0, "step_rate": 4.0, "direction": direction, "tail_scale": tail_scale}
    elif fault_type == "observable_id_splice_two_parent_halves":
        points, direction, tail_scale = _translate_tail_to_body_step(
            parent_points, donor_points, body_step=2.0, height=0.06
        )
        expected = "suspected_id_switch"
        donor_id = str(donor["sample_id"])
        boundary = {"height_ratio": 1.0, "body_step": 2.0, "step_rate": 4.0, "direction": direction, "tail_scale": tail_scale}
    elif fault_type == "height_ratio_2_with_step_1_body_height":
        heights[40:] = 0.12
        points, direction, tail_scale = _translate_tail_to_body_step(
            parent_points, parent_points, body_step=1.0, height=0.09
        )
        expected = "height_position_discontinuity"
        boundary = {"height_ratio": 2.0, "body_step": 1.0, "direction": direction, "tail_scale": tail_scale}
    elif fault_type == "camera_motion_state_moved":
        motion = "moved"
        expected = "camera_moved"
    elif fault_type == "geometrically_continuous_splice_control":
        points = _continuous_splice(parent_points, donor_points)
        expected = None
        donor_id = str(donor["sample_id"])
        boundary = _continuous_boundary_evidence(points, heights)
    else:
        raise AugmentationBuildError(f"unknown fault/control type: {fault_type}")
    try:
        adapter = build_camera_adapter_input(
            points,
            heights,
            record_id=corruption_replay_key,
            camera_config=camera_config,
            observed_mask=observed,
            camera_motion_state=motion,
            track_id=1,
        )
        qc = run_camera_qc(adapter, camera_config)
    except (CameraCorruptionError, ValueError) as exc:
        raise AugmentationBuildError(f"cannot construct fault/control {fault_type}") from exc
    reasons = _fault_qc_reasons(qc)
    status = "ready" if qc.ready_inputs else "unavailable"
    if fault_type == "observable_id_splice_two_parent_halves" and expected not in reasons:
        raise AugmentationBuildError("observable splice did not yield suspected_id_switch")
    return {
        "schema_version": config["output_schemas"]["fault"],
        "record_role": "limitation_control" if control else "gated_fault",
        "fault_or_control_id": record_id,
        "fault_or_control_type": fault_type,
        "parent_sample_id": parent["sample_id"],
        "donor_parent_sample_id": donor_id,
        "source_dataset": parent["source_dataset"],
        "split": "validation",
        "labels": {
            "four_class": parent["pattern_label"]
            if parent["pattern_supervision_eligible"]
            else None,
            "binary": int(parent["binary_label"]),
        },
        "track_id": 1,
        "splice_index": 40 if donor_id is not None else None,
        "corruption_replay_key": corruption_replay_key,
        "seed": seed,
        "expected_qc_reason": expected,
        "observed_qc_status": status,
        "observed_qc_reason_codes": reasons,
        "boundary_evidence": boundary,
        "virtual_image_points": points.tolist(),
        "virtual_bbox_heights": heights.tolist(),
        "observed_mask": observed.astype(np.int8).tolist(),
        "training_allowed": False,
        "validation_scope": "synthetic_camera_corruption",
        "capability_boundary": (
            "bbox-only QC can reject observable ID discontinuities but cannot identify geometrically continuous identity switches"
            if control
            else None
        ),
    }


def _fault_canvas(parent: Mapping[str, Any]) -> np.ndarray:
    fitted = fit_virtual_canvas(
        parent["shape_normalized_points"],
        parent["point_mask"],
        extent=0.10,
        epsilon=1e-6,
    )
    return fitted


def _translate_tail_to_body_step(
    parent_points: np.ndarray,
    tail_source: np.ndarray,
    *,
    body_step: float,
    height: float,
) -> tuple[np.ndarray, str, float]:
    prefix = parent_points[:40]
    tail = tail_source[40:]
    directions = (
        ("+x", np.asarray([1.0, 0.0])),
        ("-x", np.asarray([-1.0, 0.0])),
        ("+y", np.asarray([0.0, 1.0])),
        ("-y", np.asarray([0.0, -1.0])),
    )
    candidates: list[tuple[float, int, str, np.ndarray, np.ndarray]] = []
    for order, (name, vector) in enumerate(directions):
        target = prefix[-1] + vector * body_step * height
        moved = tail + (target - tail[0])
        combined = np.vstack((prefix, moved))
        margin = _bbox_margin(combined, np.full(80, height))
        candidates.append((margin, -order, name, combined, target))
    _margin, _order, name, points, target = max(candidates, key=lambda item: (item[0], item[1]))
    if _margin >= -1e-12:
        return points, name, 1.0
    centered = tail - tail[0]
    if _bbox_margin(np.vstack((prefix, np.repeat(target[None, :], len(tail), axis=0))), np.full(80, height)) < 0.0:
        raise AugmentationBuildError("hard-jump target cannot hold a valid bbox")
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        candidate_tail = target + centered * middle
        candidate = np.vstack((prefix, candidate_tail))
        if _bbox_margin(candidate, np.full(80, height)) >= 0.0:
            low = middle
        else:
            high = middle
    if low <= 1e-9:
        raise AugmentationBuildError("positive uniform tail scale is infeasible")
    scaled = np.vstack((prefix, target + centered * low))
    return scaled, name, low


def _continuous_splice(parent_points: np.ndarray, donor_points: np.ndarray) -> np.ndarray:
    p38, p39 = parent_points[38], parent_points[39]
    d40, d41 = donor_points[40], donor_points[41]
    v = p39 - p38
    u = d41 - d40
    denominator = float(np.dot(u, u))
    if float(np.linalg.norm(v)) <= 1e-6 or denominator <= 1e-12:
        raise AugmentationBuildError("continuous splice boundary direction is degenerate")
    a = float(np.dot(u, v) / denominator)
    b = float((u[0] * v[1] - u[1] * v[0]) / denominator)
    similarity = np.asarray([[a, -b], [b, a]])
    target = p39 + v
    transformed = (similarity @ (donor_points[40:] - d40).T).T + target
    combined = np.vstack((parent_points[:40], transformed))
    if _bbox_margin(combined, np.full(80, 0.06)) < -1e-12:
        raise AugmentationBuildError("continuous splice bbox left the virtual canvas")
    if not np.allclose(combined[40], p39 + v, rtol=0.0, atol=1e-12):
        raise AugmentationBuildError("continuous splice position alignment failed")
    if not np.allclose(combined[41] - combined[40], v, rtol=0.0, atol=1e-12):
        raise AugmentationBuildError("continuous splice direction/step alignment failed")
    return combined


def _continuous_boundary_evidence(points: np.ndarray, heights: np.ndarray) -> dict[str, Any]:
    v = points[39] - points[38]
    return {
        "mapped_d40": points[40].tolist(),
        "expected_d40": (points[39] + v).tolist(),
        "boundary_step": (points[40] - points[39]).tolist(),
        "donor_first_step": (points[41] - points[40]).tolist(),
        "parent_last_step": v.tolist(),
        "boundary_height_ratio": float(heights[40] / heights[39]),
    }


def _bbox_margin(points: np.ndarray, heights: np.ndarray) -> float:
    half_width = 0.20 * heights
    values = np.concatenate(
        (
            points[:, 0] - half_width,
            1.0 - (points[:, 0] + half_width),
            points[:, 1] - heights,
            1.0 - points[:, 1],
        )
    )
    return float(values.min())


def _select_donor(
    parent: Mapping[str, Any], validation: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    if parent["source_dataset"] == "wandering_patterns":
        candidates = [
            row
            for row in validation
            if row["source_dataset"] == parent["source_dataset"]
            and row["pattern_label"] != parent["pattern_label"]
        ]
    else:
        candidates = [
            row
            for row in validation
            if row["source_dataset"] == parent["source_dataset"]
            and int(row["binary_label"]) != int(parent["binary_label"])
        ]
    ordered = _hash_order(candidates)
    if not ordered:
        raise AugmentationBuildError("fault-suite donor set is empty")
    parent_key = _sample_hash(str(parent["sample_id"]))
    return next((row for row in ordered if _sample_hash(str(row["sample_id"])) > parent_key), ordered[0])


def _hash_order(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (_sample_hash(str(row["sample_id"])), str(row["sample_id"])))


def _sample_hash(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def _fault_qc_reasons(qc: Any) -> list[str]:
    reasons: list[str] = []
    for row in qc.tracklet_records:
        reasons.extend(str(item) for item in row.get("quality_flags", []))
    for row in qc.window_records:
        reasons.extend(str(item) for item in row.get("reason_codes", []))
    if not qc.window_records:
        reasons.append("no_complete_window")
    return list(dict.fromkeys(reasons))


def _validate_output_counts(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    pressure: Sequence[Mapping[str, Any]],
    faults: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    if len(pressure) != len(validation) * 3 or len(pressure) != 834:
        raise AugmentationBuildError("validation pressure count must be exactly 834")
    if len(pairs) > len(train) * 2 or any(
        row["split"] != "train" or row["severity"] not in {"low", "medium"} for row in pairs
    ):
        raise AugmentationBuildError("train pair count/scope is invalid")
    roles = Counter(row["record_role"] for row in faults)
    if roles != Counter(
        gated_fault=int(config["fault_suite"]["expected_gated_fault_records"]),
        limitation_control=int(config["fault_suite"]["expected_limitation_control_records"]),
    ):
        raise AugmentationBuildError("QC fault/control counts drifted")


def _validate_v2_pacing_selection_floor(pairs: Sequence[Mapping[str, Any]]) -> None:
    selected = Counter(
        str(row["severity"])
        for row in pairs
        if row["source_dataset"] == "wandering_patterns"
        and row["labels"]["four_class"] == "pacing"
    )
    for severity in ("low", "medium"):
        if selected[severity] < 50:
            raise AugmentationBuildError(
                f"v2 pacing {severity} selected count is below the frozen floor of 50"
            )


def _train_label_severity_counts(
    *,
    train: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    severities = tuple(str(value) for value in config["generation"]["train_severities"])

    def key_from_parent(row: Mapping[str, Any], severity: str) -> tuple[str, str, str]:
        source = str(row["source_dataset"])
        label = (
            str(row["pattern_label"])
            if source == "wandering_patterns"
            else f"binary_{int(row['binary_label'])}"
        )
        return source, label, severity

    def key_from_child(row: Mapping[str, Any]) -> tuple[str, str, str]:
        source = str(row["source_dataset"])
        labels = row["labels"]
        label = (
            str(labels["four_class"])
            if source == "wandering_patterns"
            else f"binary_{int(labels['binary'])}"
        )
        return source, label, str(row["severity"])

    parent_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    qc_ready_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    accepted_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    selected_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    qc_reasons: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    semantic_reasons: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for parent in train:
        for severity in severities:
            parent_ids[key_from_parent(parent, severity)].add(str(parent["sample_id"]))
    for attempt in attempts:
        if attempt["split"] != "train":
            continue
        key = key_from_child(attempt)
        parent_id = str(attempt["parent_sample_id"])
        if attempt["qc_status"] == "ready":
            qc_ready_ids[key].add(parent_id)
        if attempt["semantic_status"] == "accepted":
            accepted_ids[key].add(parent_id)
        qc_reasons[key].update(str(reason) for reason in attempt.get("qc_reason_codes", []))
        semantic_reasons[key].update(
            str(reason) for reason in attempt.get("semantic_reason_codes", [])
        )
    for pair in pairs:
        selected_ids[key_from_child(pair)].add(str(pair["parent_sample_id"]))
    return [
        {
            "source_dataset": key[0],
            "label": key[1],
            "severity": key[2],
            "parent_count": len(parent_ids[key]),
            "at_least_once_qc_ready_count": len(qc_ready_ids[key]),
            "semantic_accepted_count": len(accepted_ids[key]),
            "selected_count": len(selected_ids[key]),
            "qc_reason_counts": dict(sorted(qc_reasons[key].items())),
            "semantic_rejection_reason_counts": dict(
                sorted(semantic_reasons[key].items())
            ),
        }
        for key in sorted(parent_ids)
    ]


def validate_train_selection_quotas(
    strata: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> None:
    """Fail closed with every v3 train visual-review quota shortfall."""

    gate = config.get("quota_gate")
    if not isinstance(gate, Mapping):
        raise AugmentationBuildError("v3 quota gate is missing")
    severities = ("low", "medium")
    expected = {
        ("wandering_patterns", label, severity)
        for label in _WP_LABELS
        for severity in severities
    } | {
        ("smartcare", label, severity)
        for label in ("binary_0", "binary_1")
        for severity in severities
    }
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    required_report_fields = set(str(value) for value in gate["report_all_fields"])
    for row in strata:
        if not required_report_fields <= set(row):
            raise AugmentationBuildError("train quota stratum report fields are missing")
        key = (
            str(row.get("source_dataset")),
            str(row.get("label")),
            str(row.get("severity")),
        )
        if key in by_key:
            raise AugmentationBuildError("duplicate train quota stratum")
        by_key[key] = row
    unexpected = sorted(set(by_key) - expected)
    missing = sorted(expected - set(by_key))
    failures: list[dict[str, Any]] = []
    for key in sorted(set(by_key) & expected):
        row = by_key[key]
        minimum = int(
            gate["wp_min_selected_per_class_per_severity"]
            if key[0] == "wandering_patterns"
            else gate["smartcare_min_selected_per_binary_class_per_severity"]
        )
        selected = row.get("selected_count")
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise AugmentationBuildError("train quota selected_count is invalid")
        if selected < minimum:
            failures.append({**dict(row), "required_selected_count": minimum})
    if failures or missing or unexpected:
        payload = {
            "failing_strata": failures,
            "missing_strata": [list(key) for key in missing],
            "unexpected_strata": [list(key) for key in unexpected],
        }
        raise AugmentationBuildError(
            "v3 train selection quota failed: "
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )


def _validation_pressure_status_counts(
    attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in attempts:
        if row["split"] != "validation":
            continue
        source = str(row["source_dataset"])
        label = (
            str(row["labels"]["four_class"])
            if source == "wandering_patterns"
            else f"binary_{int(row['labels']['binary'])}"
        )
        counts[
            (
                source,
                label,
                str(row["severity"]),
                str(row["qc_status"]),
                str(row["semantic_status"]),
            )
        ] += 1
    if sum(counts.values()) != 834:
        raise AugmentationBuildError("validation pressure status count must total 834")
    return [
        {
            "source_dataset": key[0],
            "label": key[1],
            "severity": key[2],
            "qc_status": key[3],
            "semantic_status": key[4],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _augmentation_report(
    *,
    attempts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    pressure: Sequence[Mapping[str, Any]],
    faults: Sequence[Mapping[str, Any]],
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    pacing_parent_histogram: Mapping[str, int] | None,
    train_strata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts_by_status = Counter(
        (str(row["split"]), str(row["severity"]), str(row["qc_status"]), str(row["semantic_status"]))
        for row in attempts
    )
    fault_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in faults:
        fault_groups[str(row["fault_or_control_type"])].append(row)
    rejection_reasons = Counter(
        reason for row in attempts for reason in row.get("semantic_reason_codes", [])
    )
    if (
        config["schema_version"]
        in {AUGMENTATION_CONFIG_SCHEMA_VERSION_V2, AUGMENTATION_CONFIG_SCHEMA_VERSION_V3}
        and config["semantic_check"]["pacing_reversal_gate"]["forbidden_reason_code"]
        in rejection_reasons
    ):
        raise AugmentationBuildError("v2 emitted the forbidden absolute pacing reason")
    return {
        "schema_version": config["output_schemas"]["report"],
        "validation_scope": "synthetic_camera_corruption",
        "parent_counts": {"train": len(train), "validation": len(validation)},
        "wp_train_pacing_reversal_histogram": (
            dict(pacing_parent_histogram) if pacing_parent_histogram is not None else None
        ),
        "attempt_count": len(attempts),
        "selected_train_pair_count": len(pairs),
        "validation_pressure_count": len(pressure),
        "train_label_severity_counts": [dict(row) for row in train_strata],
        "validation_pressure_status_counts": _validation_pressure_status_counts(attempts),
        "attempt_status_counts": [
            {
                "split": key[0],
                "severity": key[1],
                "qc_status": key[2],
                "semantic_status": key[3],
                "count": count,
            }
            for key, count in sorted(attempts_by_status.items())
        ],
        "semantic_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "qc_reason_counts": dict(
            sorted(
                Counter(
                    reason for row in attempts for reason in row.get("qc_reason_codes", [])
                ).items()
            )
        ),
        "fault_results": [
            {
                "fault_or_control_type": name,
                "record_role": rows[0]["record_role"],
                "count": len(rows),
                "unavailable_count": sum(row["observed_qc_status"] == "unavailable" for row in rows),
                "ready_count": sum(row["observed_qc_status"] == "ready" for row in rows),
                "qc_rejection_rate": (
                    sum(row["observed_qc_status"] == "unavailable" for row in rows) / len(rows)
                    if rows[0]["record_role"] == "gated_fault"
                    else None
                ),
            }
            for name, rows in sorted(fault_groups.items())
        ],
        "human_review_status": "pending_human_review",
        "capability_statements": [
            "general synthetic corruption is not a measured target-camera error distribution",
            "bbox-only QC can reject observable ID discontinuities but cannot identify geometrically continuous identity switches",
        ],
        "training_performed": False,
        "test_or_official_used": False,
    }


def _artifact_descriptors(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if tuple(files) != _MACHINE_FILE_ORDER:
        raise AugmentationBuildError("augmentation artifact construction order drifted")
    return {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
        for name, payload in files.items()
    }


def _step8_source_hashes(project_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative in _STEP8_SOURCE_PATHS:
        path = project_root / PurePosixPath(relative)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise AugmentationBuildError(f"cannot hash Step-8 source: {relative}") from exc
        result[Path(relative).name] = {"path": relative, "sha256": digest}
    return result


def _load_finite_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                AugmentationBuildError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AugmentationBuildError(f"cannot parse trusted JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AugmentationBuildError("trusted JSON root must be an object")
    return value


def _trajectory_metrics(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    coverage_entropy: float,
    semantic: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
) -> dict[str, Any]:
    array, valid = _valid_points(points, mask)
    selected = array[valid]
    displacement = np.diff(selected, axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    path_length = float(lengths.sum())
    net_displacement = float(np.linalg.norm(selected[-1] - selected[0]))
    epsilon = float(semantic["metric_epsilon"])
    path_efficiency = net_displacement / path_length if path_length > epsilon else 0.0
    topology = compute_topology(
        array,
        valid.astype(np.int8),
        step_epsilon=float(preprocessing_config["step_epsilon"]),
        **dict(preprocessing_config["topology"]),
    )
    geometry = compute_turn_geometry(
        array,
        valid.astype(np.int8),
        step_epsilon=float(preprocessing_config["step_epsilon"]),
    )
    valid_curvature = geometry.abs_curvature[geometry.valid_turn]
    mean_abs_curvature = float(valid_curvature.mean()) if len(valid_curvature) else 0.0
    endpoint_repeat = endpoint_repeat_rate(
        array,
        valid.astype(np.int8),
        reference_points=int(semantic["endpoint_reference_points"]),
        query_points=int(semantic["endpoint_query_points"]),
        radius=float(semantic["endpoint_repeat_radius"]),
    )
    return {
        "path_length": path_length,
        "net_displacement": net_displacement,
        "path_efficiency": path_efficiency,
        "mean_absolute_curvature": mean_abs_curvature,
        "reversal_count": len(topology["reversal_events"]),
        "absolute_winding": float(topology["absolute_winding"]),
        "revisit_pair_count": int(topology["revisit_pair_count"]),
        "closure_distance": net_displacement,
        "coverage_entropy": float(coverage_entropy),
        "endpoint_repeat_rate": endpoint_repeat,
        "structure_event_count": len(topology["reversal_events"])
        + min(int(topology["revisit_pair_count"]), 5),
    }


def _common_semantic_reasons(
    clean: Mapping[str, Any], corrupted: Mapping[str, Any], semantic: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    epsilon = float(semantic["metric_epsilon"])
    clean_length = float(clean["path_length"])
    corrupted_length = float(corrupted["path_length"])
    if clean_length <= epsilon:
        reasons.append("clean_path_length_near_zero")
    else:
        ratio = corrupted_length / clean_length
        low, high = (float(value) for value in semantic["path_length_ratio"])
        if not low <= ratio <= high:
            reasons.append("path_length_ratio_out_of_range")
    if abs(float(corrupted["path_efficiency"]) - float(clean["path_efficiency"])) > float(
        semantic["path_efficiency_max_abs_delta"]
    ):
        reasons.append("path_efficiency_changed")
    clean_curvature = float(clean["mean_absolute_curvature"])
    corrupted_curvature = float(corrupted["mean_absolute_curvature"])
    if clean_curvature > epsilon:
        ratio = corrupted_curvature / clean_curvature
        low, high = (float(value) for value in semantic["curvature_ratio"])
        if not low <= ratio <= high:
            reasons.append("curvature_ratio_out_of_range")
    elif abs(corrupted_curvature - clean_curvature) > float(
        semantic["near_zero_curvature_max_abs_delta"]
    ):
        reasons.append("near_zero_curvature_changed")
    if abs(float(corrupted["coverage_entropy"]) - float(clean["coverage_entropy"])) > float(
        semantic["coverage_entropy_max_abs_delta"]
    ):
        reasons.append("coverage_entropy_changed")
    return reasons


def _wp_semantic_reasons(
    label: str,
    clean: Mapping[str, Any],
    corrupted: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    clean_reversals = int(clean["reversal_count"])
    corrupted_reversals = int(corrupted["reversal_count"])
    if label == "direct":
        if corrupted_reversals - clean_reversals > int(semantic["direct_max_added_reversals"]):
            reasons.append("direct_added_reversal")
        if corrupted_reversals > int(semantic["direct_max_total_reversals"]):
            reasons.append("direct_too_many_reversals")
        if float(corrupted["absolute_winding"]) - float(clean["absolute_winding"]) > float(
            semantic["direct_max_added_winding"]
        ):
            reasons.append("direct_added_winding")
    elif label == "pacing":
        gate = semantic.get("pacing_reversal_gate")
        if isinstance(gate, Mapping):
            reasons.extend(
                pacing_reversal_reason_codes(
                    clean_reversals,
                    corrupted_reversals,
                    max_lost_from_clean=int(gate["max_lost_from_clean"]),
                    max_added_to_clean=int(gate["max_added_to_clean"]),
                    lost_reason_code=str(gate["lost_reason_code"]),
                    added_reason_code=str(gate["added_reason_code"]),
                )
            )
            if gate.get("absolute_minimum") is not None:
                raise AugmentationBuildError("v2 pacing gate absolute_minimum must be null")
        else:
            if corrupted_reversals < int(semantic["pacing_min_reversals"]):
                reasons.append("pacing_too_few_reversals")
            if clean_reversals - corrupted_reversals > int(semantic["pacing_max_lost_reversals"]):
                reasons.append("pacing_reversal_lost")
        if abs(float(corrupted["endpoint_repeat_rate"]) - float(clean["endpoint_repeat_rate"])) > float(
            semantic["pacing_endpoint_repeat_max_abs_delta"]
        ):
            reasons.append("pacing_endpoint_repeat_changed")
    elif label == "lapping":
        clean_winding = float(clean["absolute_winding"])
        corrupted_winding = float(corrupted["absolute_winding"])
        if corrupted_winding < float(semantic["lapping_min_absolute_winding"]):
            reasons.append("lapping_winding_too_low")
        if clean_winding > float(semantic["metric_epsilon"]) and corrupted_winding < float(
            semantic["lapping_min_winding_retention"]
        ) * clean_winding:
            reasons.append("lapping_winding_lost")
        if float(corrupted["closure_distance"]) - float(clean["closure_distance"]) > float(
            semantic["lapping_max_added_closure_distance"]
        ):
            reasons.append("lapping_closure_lost")
        if int(clean["revisit_pair_count"]) - int(corrupted["revisit_pair_count"]) > int(
            semantic["lapping_max_lost_revisit_pairs"]
        ):
            reasons.append("lapping_revisit_lost")
    elif label == "random":
        if float(corrupted["path_efficiency"]) - float(clean["path_efficiency"]) > float(
            semantic["random_max_added_path_efficiency"]
        ):
            reasons.append("random_became_too_direct")
        if int(clean["structure_event_count"]) - int(corrupted["structure_event_count"]) > int(
            semantic["random_max_lost_structure_events"]
        ):
            reasons.append("random_structure_lost")
        if abs(float(corrupted["coverage_entropy"]) - float(clean["coverage_entropy"])) > float(
            semantic["coverage_entropy_max_abs_delta"]
        ):
            reasons.append("random_coverage_entropy_changed")
    return reasons


def _valid_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    point_mask: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=np.float64)
    mask = np.asarray(point_mask)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or mask.shape != (len(array),)
        or not np.all(np.isin(mask, [0, 1]))
        or not np.isfinite(array).all()
    ):
        raise AugmentationBuildError("semantic points/mask are invalid")
    valid = mask.astype(bool)
    if int(valid.sum()) < 1:
        raise AugmentationBuildError("semantic metrics require at least one valid point")
    return array, valid


__all__ = [
    "AUGMENTED_PAIR_SCHEMA_VERSION",
    "AugmentationBuildError",
    "AugmentationBuildResult",
    "FAULT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PRESSURE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SemanticCheckResult",
    "build_augmentation_bundle",
    "build_qc_fault_suite",
    "commit_new_output_directory",
    "coverage_entropy_pair",
    "endpoint_repeat_rate",
    "pacing_reversal_reason_codes",
    "semantic_check",
    "train_attempt_budget",
    "validate_train_selection_quotas",
    "wp_train_pacing_reversal_histogram",
]
