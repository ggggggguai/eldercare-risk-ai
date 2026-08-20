"""Trusted offline RF/TCN inference bundle for wandering camera step 7."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    group_observations,
    load_camera_inputs,
    validation_scope_for_authorization_status,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCError,
    CameraWindowInput,
    run_camera_qc,
)
from elderly_monitoring.modules.mental_health.wandering.handcrafted_features import (
    extract_features_from_points,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    apply_feature_stats,
    arc_length_resample,
    build_raw_features,
    load_preprocessing_config,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import load_rf_config
from elderly_monitoring.modules.mental_health.wandering.rf_baseline import (
    BINARY_CLASS_NAMES as RF_BINARY_CLASS_NAMES,
    FOUR_CLASS_NAMES as RF_FOUR_CLASS_NAMES,
    PRIMARY_SEED as RF_PRIMARY_SEED,
    TASK_BINARY as RF_TASK_BINARY,
    TASK_FOUR_CLASS as RF_TASK_FOUR_CLASS,
    safe_load_development_model,
)
from elderly_monitoring.modules.mental_health.wandering.tcn_baseline import (
    BINARY_CLASS_NAMES as TCN_BINARY_CLASS_NAMES,
    FOUR_CLASS_NAMES as TCN_FOUR_CLASS_NAMES,
    PRIMARY_SEED as TCN_PRIMARY_SEED,
    TASK_BINARY as TCN_TASK_BINARY,
    TASK_FOUR_CLASS as TCN_TASK_FOUR_CLASS,
    four_class_probabilities,
    load_tcn_config,
    safe_load_tcn_model,
    validate_model_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.topology import compute_topology


CAMERA_CONFIG_SCHEMA_VERSION = "wandering-camera-config-v1"
CAMERA_PREDICTION_SCHEMA_VERSION = "wandering-camera-prediction-v1"
CAMERA_RUN_MANIFEST_SCHEMA_VERSION = "wandering-camera-run-manifest-v1"
CAMERA_QC_SUMMARY_SCHEMA_VERSION = "wandering-camera-qc-summary-v1"
CAMERA_MODEL_BINDINGS_SCHEMA_VERSION = "wandering-camera-model-bindings-v1"

_RF_MANIFEST_SHA256 = "fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2"
_TCN_MANIFEST_SHA256 = "0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e"
_BANNED_TRUST_PATH_MARKERS = ("official_validation", "sealed_external_test", "public_shape_benchmark")


class CameraInferenceError(ValueError):
    """The camera config, trust chain, or inference result failed closed."""


class _TrustedModelInferenceError(CameraInferenceError):
    """One trusted model failed during forward execution or output validation."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class TrustedCameraModels:
    rf_four_class: Any
    rf_binary: Any
    tcn_four_class: Any
    tcn_binary: Any


@dataclass(frozen=True)
class CameraInferenceBuildResult:
    output_dir: Path
    manifest_sha256: str
    observation_count: int
    tracklet_count: int
    window_count: int
    ready_window_count: int
    unavailable_window_count: int
    inference_error_count: int


def load_camera_config(path: str | Path) -> dict[str, Any]:
    """Load only the exact frozen step-7 configuration."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraInferenceError(f"cannot read camera config: {config_path}") from exc
    expected = _expected_camera_config()
    if value != expected:
        raise CameraInferenceError("camera v1 config fields or frozen values have drifted")
    gap_seconds = float(value["camera_qc"]["maximum_internal_gap_seconds"])
    gap_buckets = int(value["camera_qc"]["maximum_internal_gap_buckets"])
    bucket_seconds = float(value["sampling"]["bucket_seconds"])
    if gap_seconds != gap_buckets * bucket_seconds:
        raise CameraInferenceError("camera gap seconds and bucket facts have drifted")
    return value


def load_trusted_camera_models(
    *,
    config: Mapping[str, Any],
    project_root: str | Path,
    rf_development_dir: str | Path,
    expected_rf_development_manifest_sha256: str,
    tcn_development_dir: str | Path,
    expected_tcn_development_manifest_sha256: str,
    rf_loader: Callable[..., Any] = safe_load_development_model,
    tcn_loader: Callable[..., Any] = safe_load_tcn_model,
) -> TrustedCameraModels:
    """Verify external roots, then load only the four primary-seed models safely."""

    if expected_rf_development_manifest_sha256 != config["trust_roots"]["rf_development_manifest"]["sha256"]:
        raise CameraInferenceError("external RF development manifest SHA-256 does not match camera config")
    if expected_tcn_development_manifest_sha256 != config["trust_roots"]["tcn_development_manifest"]["sha256"]:
        raise CameraInferenceError("external TCN development manifest SHA-256 does not match camera config")
    root = Path(project_root).resolve(strict=True)
    trusted = _verify_trust_roots(root, config)
    seed = int(config["models"]["primary_seed"])
    if seed != RF_PRIMARY_SEED or seed != TCN_PRIMARY_SEED:
        raise CameraInferenceError("camera primary seed is not the frozen RF/TCN primary seed")
    common_rf = {
        "seed": seed,
        "expected_manifest_sha256": expected_rf_development_manifest_sha256,
    }
    common_tcn = {
        "seed": seed,
        "expected_manifest_sha256": expected_tcn_development_manifest_sha256,
        "tcn_config_path": trusted["tcn_config"],
        "project_root": root,
    }
    return TrustedCameraModels(
        rf_four_class=rf_loader(rf_development_dir, task=RF_TASK_FOUR_CLASS, **common_rf),
        rf_binary=rf_loader(rf_development_dir, task=RF_TASK_BINARY, **common_rf),
        tcn_four_class=tcn_loader(tcn_development_dir, task=TCN_TASK_FOUR_CLASS, **common_tcn),
        tcn_binary=tcn_loader(tcn_development_dir, task=TCN_TASK_BINARY, **common_tcn),
    )


def prepare_camera_window(
    camera_input: CameraWindowInput,
    config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the frozen step-4 pure functions after camera-specific QC."""

    if not isinstance(camera_input, CameraWindowInput):
        raise CameraInferenceError("prepare_camera_window expects CameraWindowInput")
    if camera_input.window_record.get("window_status") != "ready":
        raise CameraInferenceError("only ready camera windows may enter shared preprocessing")
    if camera_input.window_record.get("preprocessing_config_sha256") != feature_stats.get(
        "binding_hashes", {}
    ).get("preprocessing_config"):
        raise CameraInferenceError("camera window preprocessing binding does not match feature stats")
    if camera_input.window_record.get("model_training_split_sha256") != feature_stats.get(
        "binding_hashes", {}
    ).get("split_sha256"):
        raise CameraInferenceError("camera window training split binding does not match feature stats")
    try:
        resampled = arc_length_resample(
            camera_input.corrected_points,
            camera_input.point_quality,
            target_points=int(config["model_input"]["target_points"]),
            step_epsilon=float(preprocessing_config["step_epsilon"]),
        )
        from elderly_monitoring.modules.mental_health.wandering.anchorless_normalization import (
            robust_isotropic_normalize,
        )

        normalized = robust_isotropic_normalize(
            resampled.points,
            quantile_low=float(preprocessing_config["scale_quantiles"][0]),
            quantile_high=float(preprocessing_config["scale_quantiles"][1]),
            quantile_method=str(preprocessing_config["quantile_method"]),
            scale_epsilon=float(preprocessing_config["scale_epsilon"]),
        )
        mask = np.ones(int(config["model_input"]["target_points"]), dtype=np.int8)
        raw_features = build_raw_features(
            normalized.points,
            mask,
            resampled.quality,
            point_times_sec=None,
            step_epsilon=float(preprocessing_config["step_epsilon"]),
            temporal_features_enabled=False,
        )
        topology = compute_topology(
            normalized.points,
            mask,
            step_epsilon=float(preprocessing_config["step_epsilon"]),
            **dict(preprocessing_config["topology"]),
        )
        model_features = apply_feature_stats(raw_features, feature_stats)
    except Exception as exc:  # shared-core errors invalidate this trusted camera window
        raise CameraInferenceError("shared step-4 camera preprocessing failed") from exc
    _validate_prepared_arrays(normalized.points, mask, raw_features, model_features)
    return {
        **dict(camera_input.window_record),
        "shape_normalized_points": normalized.points.tolist(),
        "point_mask": mask.tolist(),
        "raw_features": raw_features.tolist(),
        "model_features": model_features.tolist(),
        "topology": topology,
    }


def predict_camera_window(
    prepared_window: Mapping[str, Any],
    models: TrustedCameraModels,
    *,
    validation_scope: str,
) -> dict[str, Any]:
    """Produce four separate, uncalibrated comparison-only predictions."""

    if prepared_window.get("window_status") != "ready":
        return _unavailable_prediction(prepared_window, validation_scope=validation_scope)
    points = np.asarray(prepared_window["shape_normalized_points"], dtype=np.float64)
    mask = np.asarray(prepared_window["point_mask"], dtype=np.int8)
    raw = np.asarray(prepared_window["raw_features"], dtype=np.float64)
    features = np.asarray(prepared_window["model_features"], dtype=np.float32)
    handcrafted = extract_features_from_points(
        points,
        point_mask=mask,
        quality=raw[:, 13],
    )
    rf_four = _rf_probabilities(models.rf_four_class, handcrafted, width=4, role="rf_four_class")
    rf_binary = _rf_probabilities(models.rf_binary, handcrafted, width=2, role="rf_binary")
    tensor = torch.as_tensor(features[None, :, :], dtype=torch.float32)
    tensor_mask = torch.as_tensor(mask[None, :], dtype=torch.float32)
    validate_model_inputs(tensor, tensor_mask)
    tcn_four = _tcn_four_probabilities(models.tcn_four_class, tensor, tensor_mask)
    tcn_binary = _tcn_binary_probabilities(models.tcn_binary, tensor, tensor_mask)
    predictions = {
        "rf_four_class": _named_prediction(rf_four, RF_FOUR_CLASS_NAMES, binary=False),
        "rf_binary": _named_prediction(rf_binary, RF_BINARY_CLASS_NAMES, binary=True),
        "tcn_four_class": _named_prediction(tcn_four, TCN_FOUR_CLASS_NAMES, binary=False),
        "tcn_binary": _named_prediction(tcn_binary, TCN_BINARY_CLASS_NAMES, binary=True),
    }
    return {
        "schema_version": CAMERA_PREDICTION_SCHEMA_VERSION,
        "window_id": prepared_window["window_id"],
        "window_status": "ready",
        "reason_codes": [],
        "quality_flags": list(prepared_window["quality_flags"]),
        "model_purpose": "comparison_only",
        "validation_scope": validation_scope,
        "probability_calibrated": False,
        "fusion_performed": False,
        "primary_seed": RF_PRIMARY_SEED,
        "predictions": predictions,
        "model_invocation_skipped": False,
    }


def build_camera_inference_bundle(
    *,
    camera_config_path: str | Path,
    project_root: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    rf_development_dir: str | Path,
    expected_rf_development_manifest_sha256: str,
    tcn_development_dir: str | Path,
    expected_tcn_development_manifest_sha256: str,
    output_dir: str | Path,
) -> CameraInferenceBuildResult:
    """Build one deterministic, atomically committed camera prediction bundle."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera output already exists: {output}")
    config = load_camera_config(camera_config_path)
    root = Path(project_root).resolve(strict=True)
    expected_config_path = (root / "configs/modules/wandering_camera_v1.yaml").resolve(strict=False)
    if Path(camera_config_path).resolve(strict=False) != expected_config_path:
        raise CameraInferenceError("camera config must be the fixed repository path")
    if expected_rf_development_manifest_sha256 != config["trust_roots"]["rf_development_manifest"]["sha256"]:
        raise CameraInferenceError("external RF development manifest SHA-256 mismatch")
    if expected_tcn_development_manifest_sha256 != config["trust_roots"]["tcn_development_manifest"]["sha256"]:
        raise CameraInferenceError("external TCN development manifest SHA-256 mismatch")
    trusted_paths = _verify_trust_roots(root, config)
    preprocessing_config = load_preprocessing_config(trusted_paths["preprocessing_config"])
    feature_stats = _load_and_verify_feature_stats(trusted_paths, config)
    models = load_trusted_camera_models(
        config=config,
        project_root=root,
        rf_development_dir=rf_development_dir,
        expected_rf_development_manifest_sha256=expected_rf_development_manifest_sha256,
        tcn_development_dir=tcn_development_dir,
        expected_tcn_development_manifest_sha256=expected_tcn_development_manifest_sha256,
    )
    try:
        adapter = load_camera_inputs(tracking_jsonl_path, media_sidecar_path, config)
        qc = run_camera_qc(adapter, config)
    except (CameraAdapterError, CameraQCError) as exc:
        raise CameraInferenceError("camera adapter/QC failed closed") from exc
    validation_scope = _validation_scope(adapter.media_sidecar)

    prepared_by_id: dict[str, Mapping[str, Any]] = {}
    for item in qc.ready_inputs:
        prepared = prepare_camera_window(item, config, preprocessing_config, feature_stats)
        prepared_by_id[str(prepared["window_id"])] = prepared
    window_records = [
        dict(prepared_by_id.get(str(record["window_id"]), record)) for record in qc.window_records
    ]
    predictions: list[Mapping[str, Any]] = []
    for record in window_records:
        if record["window_status"] != "ready":
            predictions.append(_unavailable_prediction(record, validation_scope=validation_scope))
            continue
        try:
            predictions.append(predict_camera_window(record, models, validation_scope=validation_scope))
        except _TrustedModelInferenceError as exc:
            predictions.append(_inference_error_prediction(record, validation_scope=validation_scope, exc=exc))

    qc_summary = _qc_summary(adapter, qc.tracklet_records, window_records, predictions, validation_scope)
    model_bindings = _model_bindings(config, expected_rf_development_manifest_sha256, expected_tcn_development_manifest_sha256)
    files: dict[str, bytes] = {
        "media_sidecar.json": canonical_json_bytes(dict(adapter.media_sidecar)),
        "tracking_input.jsonl": canonical_jsonl_bytes(adapter.normalized_rows),
        "bbox_tracklets.jsonl": canonical_jsonl_bytes(qc.tracklet_records),
        "window_records.jsonl": canonical_jsonl_bytes(window_records),
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "qc_summary.json": canonical_json_bytes(qc_summary),
        "model_bindings.json": canonical_json_bytes(model_bindings),
    }
    manifest = {
        "schema_version": CAMERA_RUN_MANIFEST_SCHEMA_VERSION,
        "artifacts": _artifact_descriptors(files),
        "camera_config_sha256": _sha256_file(expected_config_path),
        "trust_roots": {
            name: dict(config["trust_roots"][name]) for name in sorted(config["trust_roots"])
        },
        "expected_rf_development_manifest_sha256": expected_rf_development_manifest_sha256,
        "expected_tcn_development_manifest_sha256": expected_tcn_development_manifest_sha256,
        "source_tracking_sha256": adapter.source_tracking_sha256,
        "normalized_tracking_sha256": adapter.normalized_tracking_sha256,
        "source_media_sha256": adapter.media_sidecar["source_sha256"],
        "code_hashes": _source_hashes(root),
        "environment": _environment_versions(),
        "validation_scope": validation_scope,
        "models_retrained": False,
        "official_source_opened": False,
        "fusion_performed": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    status_counts = Counter(row["window_status"] for row in predictions)
    return CameraInferenceBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        observation_count=len(adapter.observations),
        tracklet_count=len(qc.tracklet_records),
        window_count=len(predictions),
        ready_window_count=status_counts["ready"],
        unavailable_window_count=status_counts["unavailable"],
        inference_error_count=status_counts["inference_error"],
    )


def _verify_trust_roots(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name, descriptor in config["trust_roots"].items():
        relative = descriptor["path"]
        lowered = str(relative).lower()
        if any(marker in lowered for marker in _BANNED_TRUST_PATH_MARKERS):
            raise CameraInferenceError(f"camera trust root points to forbidden test/official data: {name}")
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CameraInferenceError(f"camera trust-root path escapes project root: {name}") from exc
        if not path.is_file() or _sha256_file(path) != descriptor["sha256"]:
            raise CameraInferenceError(f"camera trust-root file/hash mismatch: {name}")
        output[name] = path
    load_rf_config(output["rf_config"])
    load_tcn_config(output["tcn_config"])
    return output


def _load_and_verify_feature_stats(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_canonical_json(paths["preprocessing_manifest"])
    stats = _load_canonical_json(paths["preprocessing_feature_stats"])
    if manifest.get("schema_version") != "wandering-preprocessing-manifest-v1":
        raise CameraInferenceError("preprocessing manifest schema drifted")
    descriptor = manifest.get("artifacts", {}).get("feature_stats.json")
    stats_path = paths["preprocessing_feature_stats"]
    if descriptor != {"byte_count": stats_path.stat().st_size, "sha256": _sha256_file(stats_path)}:
        raise CameraInferenceError("preprocessing manifest no longer binds feature stats")
    if manifest.get("preprocessing_config_sha256") != config["trust_roots"]["preprocessing_config"]["sha256"]:
        raise CameraInferenceError("preprocessing config binding drifted")
    if stats.get("schema_version") != "wandering-feature-stats-v1":
        raise CameraInferenceError("feature stats schema drifted")
    if stats.get("temporal_features_enabled") is not False:
        raise CameraInferenceError("camera v1 requires temporal_features_enabled=false stats")
    if stats.get("binding_hashes", {}).get("preprocessing_config") != config["trust_roots"]["preprocessing_config"]["sha256"]:
        raise CameraInferenceError("feature stats preprocessing binding drifted")
    if stats.get("binding_hashes", {}).get("split_sha256") != manifest.get("split_sha256"):
        raise CameraInferenceError("feature stats model training split binding drifted")
    return stats


def _load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraInferenceError(f"cannot parse trusted JSON: {path.name}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise CameraInferenceError(f"trusted JSON is not canonical: {path.name}")
    return value


def _validate_prepared_arrays(
    points: np.ndarray,
    mask: np.ndarray,
    raw: np.ndarray,
    model: np.ndarray,
) -> None:
    if points.shape != (80, 2) or raw.shape != (80, 14) or model.shape != (80, 14) or mask.shape != (80,):
        raise CameraInferenceError("prepared camera arrays do not match [80,2]/[80,14]")
    if not np.isfinite(points).all() or not np.isfinite(raw).all() or not np.isfinite(model).all():
        raise CameraInferenceError("prepared camera arrays must be finite")
    if not np.array_equal(mask, np.ones(80, dtype=np.int8)):
        raise CameraInferenceError("ready camera point_mask must be all ones")
    if not np.array_equal(raw[:, 10:12], np.zeros((80, 2))) or not np.array_equal(model[:, 10:12], np.zeros((80, 2))):
        raise CameraInferenceError("camera v1 time channels must remain zero")
    if not np.array_equal(raw[:, 12], mask) or not np.array_equal(model[:, 12], mask):
        raise CameraInferenceError("camera mask channel must equal point_mask")
    if np.any((raw[:, 13] < 0.0) | (raw[:, 13] > 1.0)) or not np.array_equal(raw[:, 13], model[:, 13]):
        raise CameraInferenceError("camera quality channel semantics drifted")


def _rf_probabilities(model: Any, vector: np.ndarray, *, width: int, role: str) -> np.ndarray:
    try:
        values = model.predict_proba(vector.reshape(1, -1))
    except Exception as exc:
        raise _TrustedModelInferenceError(
            f"{role} forward failed",
            reason_code="trusted_model_forward_failed",
        ) from exc
    try:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (1, width):
            raise ValueError(f"{role} returned an invalid probability shape")
        return _probability_vector(array[0], width, role)
    except _TrustedModelInferenceError:
        raise
    except Exception as exc:
        raise _TrustedModelInferenceError(
            f"{role} output is invalid",
            reason_code="trusted_model_output_invalid",
        ) from exc


def _probability_vector(value: Sequence[float] | np.ndarray, width: int, role: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (width,) or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError(f"{role} probabilities must be finite and non-negative")
        if not math.isclose(float(array.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{role} probabilities must sum to one")
        return array
    except _TrustedModelInferenceError:
        raise
    except Exception as exc:
        raise _TrustedModelInferenceError(
            f"{role} output is invalid",
            reason_code="trusted_model_output_invalid",
        ) from exc


def _tcn_four_probabilities(
    model: Any,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    output = _tcn_forward(model, features, mask, role="tcn_four_class")
    try:
        tensor = four_class_probabilities(
            output["gate_logit"],
            output["subtype_logits"],
        )[0]
        return _probability_vector(tensor.detach().cpu().numpy(), 4, "tcn_four_class")
    except _TrustedModelInferenceError:
        raise
    except Exception as exc:
        raise _TrustedModelInferenceError(
            "tcn_four_class output is invalid",
            reason_code="trusted_model_output_invalid",
        ) from exc


def _tcn_binary_probabilities(
    model: Any,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    output = _tcn_forward(model, features, mask, role="tcn_binary")
    try:
        p = torch.sigmoid(output["binary_logit"]).reshape(-1)[0]
        tensor = torch.stack((1.0 - p, p))
        return _probability_vector(tensor.detach().cpu().numpy(), 2, "tcn_binary")
    except _TrustedModelInferenceError:
        raise
    except Exception as exc:
        raise _TrustedModelInferenceError(
            "tcn_binary output is invalid",
            reason_code="trusted_model_output_invalid",
        ) from exc


def _tcn_forward(
    model: Any,
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    role: str,
) -> Any:
    try:
        with torch.no_grad():
            return model.forward(features, mask)
    except Exception as exc:
        raise _TrustedModelInferenceError(
            f"{role} forward failed",
            reason_code="trusted_model_forward_failed",
        ) from exc


def _named_prediction(
    probabilities: np.ndarray,
    class_order: Sequence[str],
    *,
    binary: bool,
) -> dict[str, Any]:
    if binary:
        predicted_index = 1 if float(probabilities[1]) >= 0.5 else 0
    else:
        predicted_index = int(np.argmax(probabilities))
    return {
        "class_order": list(class_order),
        "probabilities": probabilities.tolist(),
        "predicted_label": class_order[predicted_index],
    }


def _unavailable_prediction(record: Mapping[str, Any], *, validation_scope: str) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_PREDICTION_SCHEMA_VERSION,
        "window_id": record["window_id"],
        "window_status": "unavailable",
        "reason_codes": list(record["reason_codes"]),
        "quality_flags": list(record["quality_flags"]),
        "model_purpose": "comparison_only",
        "validation_scope": validation_scope,
        "probability_calibrated": False,
        "fusion_performed": False,
        "primary_seed": RF_PRIMARY_SEED,
        "predictions": None,
        "model_invocation_skipped": True,
    }


def _inference_error_prediction(
    record: Mapping[str, Any],
    *,
    validation_scope: str,
    exc: _TrustedModelInferenceError,
) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_PREDICTION_SCHEMA_VERSION,
        "window_id": record["window_id"],
        "window_status": "inference_error",
        "reason_codes": [exc.reason_code],
        "quality_flags": list(record["quality_flags"]),
        "model_purpose": "comparison_only",
        "validation_scope": validation_scope,
        "probability_calibrated": False,
        "fusion_performed": False,
        "primary_seed": RF_PRIMARY_SEED,
        "predictions": None,
        "model_invocation_skipped": False,
    }


def _validation_scope(media: Mapping[str, Any]) -> str:
    return validation_scope_for_authorization_status(media.get("authorization_status"))


def _qc_summary(
    adapter: Any,
    tracklets: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    validation_scope: str,
) -> dict[str, Any]:
    reasons = Counter(reason for row in predictions for reason in row["reason_codes"])
    statuses = Counter(row["window_status"] for row in predictions)
    return {
        "schema_version": CAMERA_QC_SUMMARY_SCHEMA_VERSION,
        "validation_scope": validation_scope,
        "observation_count": len(adapter.observations),
        "normalized_tracking_row_count": len(adapter.normalized_rows),
        "input_track_scope_count": len(group_observations(adapter.observations)),
        "tracklet_count": len(tracklets),
        "window_count": len(windows),
        "window_status_counts": {name: statuses.get(name, 0) for name in ("ready", "unavailable", "inference_error")},
        "reason_counts": {name: reasons[name] for name in sorted(reasons)},
        "camera_motion_state": adapter.media_sidecar["camera_motion_state"],
    }


def _model_bindings(
    config: Mapping[str, Any],
    rf_manifest_sha256: str,
    tcn_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_MODEL_BINDINGS_SCHEMA_VERSION,
        "model_purpose": "comparison_only",
        "primary_seed": config["models"]["primary_seed"],
        "rf_development_manifest_sha256": rf_manifest_sha256,
        "tcn_development_manifest_sha256": tcn_manifest_sha256,
        "four_class_order": list(config["models"]["four_class_order"]),
        "binary_class_order": list(config["models"]["binary_class_order"]),
        "loaded_models": ["rf_four_class", "rf_binary", "tcn_four_class", "tcn_binary"],
        "probability_calibrated": False,
        "fusion_performed": False,
    }


def _artifact_descriptors(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(files.items())
    }


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/elderly_monitoring/modules/mental_health/wandering/camera_adapter.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_qc.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_inference.py",
        "scripts/wandering/run_camera_inference.py",
    )
    output: dict[str, str] = {}
    for relative in paths:
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CameraInferenceError("camera source path escapes project root") from exc
        if not path.is_file():
            raise CameraInferenceError(f"camera source file is missing: {relative}")
        output[relative] = _sha256_file(path)
    return output


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
    }


def _commit_new_output_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"camera output already exists: {output}")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_camera_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_CONFIG_SCHEMA_VERSION,
        "purpose": "comparison_only_offline_camera_chain",
        "sampling": {
            "minimum_video_fps": 15.0,
            "bucket_seconds": 0.5,
            "trajectory_hz": 2.0,
            "window_seconds": 40.0,
            "window_buckets": 80,
            "stride_seconds": 20.0,
            "stride_buckets": 40,
            "minimum_track_confidence": 0.25,
        },
        "camera_qc": {
            "minimum_observed_ratio": 0.75,
            "maximum_internal_gap_seconds": 1.5,
            "maximum_internal_gap_buckets": 3,
            "minimum_raw_detections": 8,
            "require_observed_window_edges": True,
            "minimum_motion_extent_body_heights": 0.25,
            "hard_jump_body_heights_per_second": 3.0,
            "height_ratio_discontinuity": 1.8,
            "concurrent_step_body_heights": 0.75,
        },
        "height_compensation": {"rolling_median_window": 5, "gain_clip": [0.5, 2.0]},
        "model_input": {
            "target_points": 80,
            "input_channels": 14,
            "temporal_features_enabled": False,
            "point_quality_observed": 1.0,
            "point_quality_interpolated": 0.5,
        },
        "trust_roots": {
            "preprocessing_config": {
                "path": "configs/data/wandering_preprocessing_v1.yaml",
                "sha256": "5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45",
            },
            "preprocessing_manifest": {
                "path": "data/processed/wandering/preprocessing/v1/manifest.json",
                "sha256": "242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593",
            },
            "preprocessing_feature_stats": {
                "path": "data/processed/wandering/preprocessing/v1/feature_stats.json",
                "sha256": "249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d",
            },
            "rf_config": {
                "path": "configs/modules/wandering_rf_v1.yaml",
                "sha256": "d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35",
            },
            "rf_development_manifest": {
                "path": "reports/mental_health/wandering_step5/development/v1/manifest.json",
                "sha256": _RF_MANIFEST_SHA256,
            },
            "tcn_config": {
                "path": "configs/modules/wandering_tcn_v1.yaml",
                "sha256": "8a8ab3a051dd6a00356df507b4fa3f06d4a7ed7ec7ecca8f04a3ff28dfd22cca",
            },
            "tcn_development_manifest": {
                "path": "reports/mental_health/wandering_step6/development/v1/manifest.json",
                "sha256": _TCN_MANIFEST_SHA256,
            },
        },
        "models": {
            "primary_seed": 20260731,
            "four_class_order": ["direct", "pacing", "lapping", "random"],
            "binary_class_order": ["direct_or_non_wandering", "wandering_like"],
            "run_rf_four_class": True,
            "run_rf_binary": True,
            "run_tcn_four_class": True,
            "run_tcn_binary": True,
            "fusion_performed": False,
        },
        "output_schemas": {
            "media": "wandering-media-v1",
            "bbox_tracklet": "wandering-bbox-tracklet-v1",
            "window": "wandering-camera-window-v1",
            "prediction": CAMERA_PREDICTION_SCHEMA_VERSION,
            "manifest": CAMERA_RUN_MANIFEST_SCHEMA_VERSION,
        },
    }


__all__ = [
    "CAMERA_CONFIG_SCHEMA_VERSION",
    "CAMERA_PREDICTION_SCHEMA_VERSION",
    "CAMERA_RUN_MANIFEST_SCHEMA_VERSION",
    "CameraInferenceBuildResult",
    "CameraInferenceError",
    "TrustedCameraModels",
    "build_camera_inference_bundle",
    "load_camera_config",
    "load_trusted_camera_models",
    "predict_camera_window",
    "prepare_camera_window",
]
