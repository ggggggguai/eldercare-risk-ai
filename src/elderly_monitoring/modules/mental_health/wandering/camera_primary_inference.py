"""TopoWander-MPT primary inference on trusted Step-7 camera windows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering import model as _MODEL_SOURCE_MODULE
from elderly_monitoring.modules.mental_health.wandering import release as _RELEASE_SOURCE_MODULE
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    group_observations,
    load_camera_inputs,
    validate_media_sidecar,
    validation_scope_for_authorization_status,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    aggregate_episode_candidates,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
    prepare_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCError,
    run_camera_qc,
)
from elderly_monitoring.modules.mental_health.wandering.model import (
    hierarchical_four_class_probabilities,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)
from elderly_monitoring.modules.mental_health.wandering.release import (
    CandidateManifestError,
    CandidateRuntime,
    load_candidate_from_manifest,
)


CAMERA_PRIMARY_CONFIG_SCHEMA_VERSION = "wandering-camera-primary-config-v1"
CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION = "wandering-camera-primary-prediction-v1"
CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION = "wandering-camera-primary-run-manifest-v1"
CAMERA_PRIMARY_MODEL_BINDINGS_SCHEMA_VERSION = "wandering-camera-primary-model-bindings-v1"
CAMERA_PRIMARY_EXECUTION_SCHEMA_VERSION = "wandering-camera-primary-execution-v1"
CAMERA_PRIMARY_QC_SUMMARY_SCHEMA_VERSION = "wandering-camera-primary-qc-summary-v1"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7"
EXPECTED_MODEL_STATE_SHA256 = "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"
STATUS = "wandering_m0cam_primary_camera_engineering_ready"
EVIDENCE_SCOPE = "synthetic_contract_only"
FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")
SUBTYPE_ORDER = ("pacing", "lapping", "random")
BINARY_CLASS_ORDER = ("direct_or_non_wandering", "wandering_like")
_CONFIG_RELATIVE = Path("configs/modules/wandering_camera_primary_v1.yaml")
_MODEL_SOURCE_RELATIVE = Path(
    "src/elderly_monitoring/modules/mental_health/wandering/model.py"
)
_RELEASE_SOURCE_RELATIVE = Path(
    "src/elderly_monitoring/modules/mental_health/wandering/release.py"
)
_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
)


class PrimaryCameraInferenceError(ValueError):
    """Primary camera trust, preprocessing, or build contract failed closed."""


class _TrustedPrimaryInferenceError(PrimaryCameraInferenceError):
    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class PrimaryCameraBuildResult:
    output_dir: Path
    manifest_sha256: str
    observation_count: int
    tracklet_count: int
    window_count: int
    ready_window_count: int
    unavailable_window_count: int
    inference_error_count: int
    episode_candidate_count: int


def load_primary_camera_config(path: str | Path) -> dict[str, Any]:
    """Load the exact M0-CAM-E primary-camera engineering configuration."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PrimaryCameraInferenceError(f"cannot read primary camera config: {config_path}") from exc
    if value != _expected_primary_camera_config():
        raise PrimaryCameraInferenceError("primary camera config fields or fixed identities have drifted")
    return value


def load_primary_camera_runtime(
    *,
    config: Mapping[str, Any],
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    candidate_loader: Callable[..., CandidateRuntime] = load_candidate_from_manifest,
) -> CandidateRuntime:
    """Load only the externally hashed fixed candidate and verify camera bindings."""

    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping):
        raise PrimaryCameraInferenceError("primary camera candidate config is invalid")
    if expected_manifest_sha256 != candidate["manifest_sha256"]:
        raise PrimaryCameraInferenceError("external candidate manifest SHA-256 does not match primary config")
    path = Path(manifest_path)
    try:
        if path.stat().st_size != int(candidate["manifest_size_bytes"]):
            raise PrimaryCameraInferenceError("candidate manifest byte count has drifted")
        if _sha256_file(path) != expected_manifest_sha256:
            raise PrimaryCameraInferenceError("candidate manifest bytes do not match external SHA-256")
    except OSError as exc:
        raise PrimaryCameraInferenceError(f"cannot read candidate manifest: {path}") from exc
    try:
        runtime = candidate_loader(
            path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except (CandidateManifestError, ValueError, OSError) as exc:
        raise PrimaryCameraInferenceError("fixed candidate failed manifest-bound safe loading") from exc
    _validate_runtime_identity(runtime, config)
    return runtime


def predict_primary_camera_window(
    prepared_window: Mapping[str, Any],
    runtime: CandidateRuntime,
    *,
    validation_scope: str,
) -> tuple[dict[str, Any], float | None]:
    """Directly forward one ready camera window as a CPU float32 batch of one."""

    _validate_window_envelope(prepared_window)
    if prepared_window["window_status"] != "ready":
        return _empty_prediction(
            prepared_window,
            runtime,
            validation_scope=validation_scope,
            status="unavailable",
            reason_codes=list(prepared_window["reason_codes"]),
            model_invocation_skipped=True,
        ), None
    features, points, mask = _camera_tensors(prepared_window)
    started = time.perf_counter_ns()
    try:
        runtime.model.to(device="cpu", dtype=torch.float32)
        runtime.model.eval()
        with torch.inference_mode():
            output = runtime.model(features, points, mask)
    except Exception as exc:  # trusted candidate forward becomes a per-window result
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        error = _TrustedPrimaryInferenceError(
            "trusted primary model forward failed",
            reason_code="trusted_model_forward_failed",
        )
        error.__cause__ = exc
        return _empty_prediction(
            prepared_window,
            runtime,
            validation_scope=validation_scope,
            status="inference_error",
            reason_codes=[error.reason_code],
            model_invocation_skipped=False,
        ), elapsed
    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    try:
        binary_logit, subtype_logits = _validated_logits(output)
        four = hierarchical_four_class_probabilities(binary_logit, subtype_logits)
        wandering = torch.sigmoid(binary_logit)
        subtype = torch.softmax(subtype_logits, dim=1)
        binary_probabilities = torch.cat((1.0 - wandering, wandering), dim=1)[0].cpu().numpy()
        subtype_probabilities = subtype[0].cpu().numpy()
        four_probabilities = four[0].cpu().numpy()
        _validate_probability_vector(binary_probabilities, len(BINARY_CLASS_ORDER), "binary")
        _validate_probability_vector(subtype_probabilities, len(SUBTYPE_ORDER), "subtype")
        _validate_probability_vector(four_probabilities, len(FOUR_CLASS_ORDER), "four_class")
    except Exception as exc:
        error = _TrustedPrimaryInferenceError(
            "trusted primary model output is invalid",
            reason_code="trusted_model_output_invalid",
        )
        error.__cause__ = exc
        return _empty_prediction(
            prepared_window,
            runtime,
            validation_scope=validation_scope,
            status="inference_error",
            reason_codes=[error.reason_code],
            model_invocation_skipped=False,
        ), elapsed

    binary_index = 1 if float(binary_probabilities[1]) >= 0.5 else 0
    subtype_index = int(np.argmax(subtype_probabilities))
    four_index = int(np.argmax(four_probabilities))
    return {
        **_prediction_envelope(prepared_window, runtime, validation_scope=validation_scope),
        "window_status": "ready",
        "reason_codes": [],
        "binary": _named_probabilities(binary_probabilities, BINARY_CLASS_ORDER, binary_index),
        "subtype": _named_probabilities(subtype_probabilities, SUBTYPE_ORDER, subtype_index),
        "four_class": _named_probabilities(four_probabilities, FOUR_CLASS_ORDER, four_index),
        "model_invocation_skipped": False,
    }, elapsed


def build_primary_camera_inference_bundle(
    *,
    camera_config_path: str | Path,
    project_root: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    episode_merge_gap_seconds: float,
    output_dir: str | Path,
    candidate_loader: Callable[..., CandidateRuntime] = load_candidate_from_manifest,
) -> PrimaryCameraBuildResult:
    """Build a fresh, non-overwriting, atomically committed M0-CAM-E bundle."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"primary camera output already exists: {output}")
    config = load_primary_camera_config(camera_config_path)
    root = Path(project_root).resolve(strict=True)
    expected_config_path = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(camera_config_path).resolve(strict=False) != expected_config_path:
        raise PrimaryCameraInferenceError("primary camera config must use the fixed repository path")
    manifest = Path(manifest_path).resolve(strict=False)
    expected_manifest_path = (root / config["candidate"]["manifest_path"]).resolve(strict=False)
    if manifest != expected_manifest_path:
        raise PrimaryCameraInferenceError("candidate manifest must use the fixed repository path")

    camera_config_path_bound = _verify_project_descriptor(root, config["camera_chain"], "camera chain config")
    try:
        camera_config = load_camera_config(camera_config_path_bound)
    except (CameraInferenceError, ValueError, OSError) as exc:
        raise PrimaryCameraInferenceError("camera preprocessing trust chain failed closed") from exc
    validation_scope = _preflight_synthetic_media_sidecar(media_sidecar_path, camera_config)
    source_identity = _preflight_active_source_identity(
        root=root,
        manifest_path=manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        config=config,
    )
    try:
        preprocessing_paths = _verify_preprocessing_roots(root, camera_config)
        preprocessing_config = load_preprocessing_config(preprocessing_paths["preprocessing_config"])
        feature_stats = _load_feature_stats(preprocessing_paths, camera_config)
    except (CameraInferenceError, ValueError, OSError) as exc:
        raise PrimaryCameraInferenceError("camera preprocessing trust chain failed closed") from exc
    runtime = load_primary_camera_runtime(
        config=config,
        manifest_path=manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        candidate_loader=candidate_loader,
    )

    try:
        adapter = load_camera_inputs(tracking_jsonl_path, media_sidecar_path, camera_config)
        qc = run_camera_qc(adapter, camera_config)
    except (CameraAdapterError, CameraQCError) as exc:
        raise PrimaryCameraInferenceError("camera adapter/QC input failed closed") from exc
    observed_validation_scope = validation_scope_for_authorization_status(
        adapter.media_sidecar.get("authorization_status")
    )
    if observed_validation_scope != validation_scope:
        raise PrimaryCameraInferenceError("synthetic media sidecar authorization changed during input loading")

    # Finish all schema and preprocessing validation before the first model forward.
    prepared_by_id: dict[str, Mapping[str, Any]] = {}
    try:
        for item in qc.ready_inputs:
            prepared = prepare_camera_window(item, camera_config, preprocessing_config, feature_stats)
            _camera_tensors(prepared)
            prepared_by_id[str(prepared["window_id"])] = prepared
    except (CameraInferenceError, KeyError, TypeError, ValueError) as exc:
        raise PrimaryCameraInferenceError("camera preprocessing input failed before model forward") from exc
    window_records = [
        dict(prepared_by_id.get(str(record["window_id"]), record)) for record in qc.window_records
    ]
    window_records.sort(key=_window_sort_key)

    predictions: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    runtime_observation = _configure_cpu_runtime(config["runtime"])
    try:
        for record in window_records:
            prediction, elapsed = predict_primary_camera_window(
                record,
                runtime,
                validation_scope=validation_scope,
            )
            predictions.append(prediction)
            if elapsed is not None:
                latency_ms.append(elapsed)
    finally:
        torch.set_num_threads(int(runtime_observation["previous_intra_op"]))
    episodes = aggregate_episode_candidates(
        predictions,
        merge_gap_seconds=episode_merge_gap_seconds,
    )
    status_counts = Counter(row["window_status"] for row in predictions)
    model_bindings = _model_bindings(config, runtime, source_identity=source_identity)
    execution = _execution_record(
        config=config,
        runtime_observation=runtime_observation,
        source_identity=source_identity,
        validation_scope=validation_scope,
        episode_merge_gap_seconds=episode_merge_gap_seconds,
        predictions=predictions,
        latency_ms=latency_ms,
    )
    qc_summary = {
        "schema_version": CAMERA_PRIMARY_QC_SUMMARY_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": validation_scope,
        "observation_count": len(adapter.observations),
        "normalized_tracking_row_count": len(adapter.normalized_rows),
        "input_track_scope_count": len(group_observations(adapter.observations)),
        "tracklet_count": len(qc.tracklet_records),
        "window_count": len(predictions),
        "window_status_counts": {
            name: status_counts[name] for name in ("ready", "unavailable", "inference_error")
        },
        "reason_counts": dict(
            sorted(Counter(reason for row in predictions for reason in row["reason_codes"]).items())
        ),
        "episode_candidate_count": len(episodes),
        "camera_motion_state": adapter.media_sidecar["camera_motion_state"],
    }
    files: dict[str, bytes] = {
        "media_sidecar.json": canonical_json_bytes(dict(adapter.media_sidecar)),
        "tracking_input.jsonl": canonical_jsonl_bytes(adapter.normalized_rows),
        "bbox_tracklets.jsonl": canonical_jsonl_bytes(qc.tracklet_records),
        "window_records.jsonl": canonical_jsonl_bytes(window_records),
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "episode_candidates.jsonl": canonical_jsonl_bytes(episodes),
        "qc_summary.json": canonical_json_bytes(qc_summary),
        "model_bindings.json": canonical_json_bytes(model_bindings),
        "execution.json": canonical_json_bytes(execution),
    }
    run_manifest = {
        "schema_version": CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "artifacts": _artifact_descriptors(files),
        "primary_camera_config_sha256": _sha256_file(expected_config_path),
        "camera_chain_config_sha256": config["camera_chain"]["sha256"],
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "model_state_sha256": config["candidate"]["model_state_sha256"],
        "source_tracking_sha256": adapter.source_tracking_sha256,
        "normalized_tracking_sha256": adapter.normalized_tracking_sha256,
        "source_media_sha256": adapter.media_sidecar["source_sha256"],
        "validation_scope": validation_scope,
        "code_hashes": _source_hashes(root),
        "data_access": dict(config["data_access"]),
        "models_retrained": False,
        "wp_prefix_padding_used": False,
        "algorithm_event_emitted": False,
        "risk_or_alert_decision_emitted": False,
    }
    manifest_bytes = canonical_json_bytes(run_manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    return PrimaryCameraBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        observation_count=len(adapter.observations),
        tracklet_count=len(qc.tracklet_records),
        window_count=len(predictions),
        ready_window_count=status_counts["ready"],
        unavailable_window_count=status_counts["unavailable"],
        inference_error_count=status_counts["inference_error"],
        episode_candidate_count=len(episodes),
    )


def _validate_runtime_identity(runtime: CandidateRuntime, config: Mapping[str, Any]) -> None:
    manifest = runtime.manifest
    candidate = config["candidate"]
    if runtime.manifest_sha256 != candidate["manifest_sha256"]:
        raise PrimaryCameraInferenceError("loaded runtime manifest identity has drifted")
    if manifest.get("candidate_id") != candidate["candidate_id"]:
        raise PrimaryCameraInferenceError("loaded candidate ID has drifted")
    artifact = manifest.get("artifacts", {}).get("model_state", {})
    if artifact.get("sha256") != candidate["model_state_sha256"]:
        raise PrimaryCameraInferenceError("loaded model-state identity has drifted")
    training = manifest.get("training_candidate_identity", {}).get("identity", {}).get("candidate", {})
    if (
        training.get("seed") != candidate["primary_seed"]
        or training.get("best_epoch") != candidate["best_epoch"]
        or training.get("retrained_for_release") is not False
        or training.get("ensemble") is not False
    ):
        raise PrimaryCameraInferenceError("loaded candidate seed/epoch selection has drifted")
    contract = manifest.get("inference_contract", {})
    if contract.get("four_class_order") != list(FOUR_CLASS_ORDER) or contract.get(
        "subtype_order"
    ) != list(SUBTYPE_ORDER):
        raise PrimaryCameraInferenceError("loaded candidate class order has drifted")
    if (
        config.get("inference", {}).get("binary_decision_threshold") != 0.5
        or contract.get("binary_decision") != "sigmoid >= 0.5"
    ):
        raise PrimaryCameraInferenceError("loaded candidate binary decision contract has drifted")
    expected_runtime = {
        "device": "cpu",
        "intra_op_threads": 8,
        "inter_op_threads": 1,
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": False,
    }
    if contract.get("runtime") != expected_runtime:
        raise PrimaryCameraInferenceError("loaded candidate CPU runtime contract has drifted")
    if runtime.model.training or any(parameter.device.type != "cpu" for parameter in runtime.model.parameters()):
        raise PrimaryCameraInferenceError("loaded primary model must be CPU eval")


def _validate_window_envelope(record: Mapping[str, Any]) -> None:
    required = {
        "window_id",
        "window_status",
        "parent_tracklet_id",
        *_SCOPE_FIELDS,
        "window_start_sec",
        "window_end_sec",
        "reason_codes",
        "quality_flags",
    }
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise PrimaryCameraInferenceError("primary camera window envelope is incomplete")
    if record["window_status"] not in {"ready", "unavailable"}:
        raise PrimaryCameraInferenceError("primary camera input window status is invalid")
    if not isinstance(record["window_id"], str) or not record["window_id"]:
        raise PrimaryCameraInferenceError("primary camera window_id is invalid")


def _camera_tensors(record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if record.get("window_status") != "ready":
        raise PrimaryCameraInferenceError("only ready camera windows may become model tensors")
    try:
        features_array = np.asarray(record["model_features"], dtype=np.float32)
        points_array = np.asarray(record["shape_normalized_points"], dtype=np.float32)
        mask_array = np.asarray(record["point_mask"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise PrimaryCameraInferenceError("primary camera prepared arrays must be numeric") from exc
    if features_array.shape != (80, 14) or points_array.shape != (80, 2) or mask_array.shape != (80,):
        raise PrimaryCameraInferenceError("primary camera prepared array shape has drifted")
    if not all(np.isfinite(value).all() for value in (features_array, points_array, mask_array)):
        raise PrimaryCameraInferenceError("primary camera prepared arrays must be finite")
    if not np.all((mask_array == 0.0) | (mask_array == 1.0)):
        raise PrimaryCameraInferenceError("primary camera point mask must contain exact 0/1")
    if not np.array_equal(features_array[:, 12], mask_array):
        raise PrimaryCameraInferenceError("primary camera model mask channel has drifted")
    return (
        torch.from_numpy(np.ascontiguousarray(features_array[None, :, :])),
        torch.from_numpy(np.ascontiguousarray(points_array[None, :, :])),
        torch.from_numpy(np.ascontiguousarray(mask_array[None, :])),
    )


def _validated_logits(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, Mapping):
        raise _TrustedPrimaryInferenceError(
            "trusted primary output must be a mapping",
            reason_code="trusted_model_output_invalid",
        )
    binary = output.get("binary_logit")
    subtype = output.get("subtype_logits")
    if not isinstance(binary, torch.Tensor) or not isinstance(subtype, torch.Tensor):
        raise _TrustedPrimaryInferenceError(
            "trusted primary logits must be tensors",
            reason_code="trusted_model_output_invalid",
        )
    if binary.shape != (1, 1) or subtype.shape != (1, 3):
        raise _TrustedPrimaryInferenceError(
            "trusted primary logit shapes have drifted",
            reason_code="trusted_model_output_invalid",
        )
    if binary.dtype != torch.float32 or subtype.dtype != torch.float32:
        raise _TrustedPrimaryInferenceError(
            "trusted primary logits must be float32",
            reason_code="trusted_model_output_invalid",
        )
    if binary.device.type != "cpu" or subtype.device.type != "cpu":
        raise _TrustedPrimaryInferenceError(
            "trusted primary logits must remain on CPU",
            reason_code="trusted_model_output_invalid",
        )
    if not torch.isfinite(binary).all() or not torch.isfinite(subtype).all():
        raise _TrustedPrimaryInferenceError(
            "trusted primary logits must be finite",
            reason_code="trusted_model_output_invalid",
        )
    return binary, subtype


def _validate_probability_vector(value: np.ndarray, width: int, role: str) -> None:
    if (
        value.shape != (width,)
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or not math.isclose(float(value.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise _TrustedPrimaryInferenceError(
            f"trusted primary {role} probabilities are invalid",
            reason_code="trusted_model_output_invalid",
        )


def _prediction_envelope(
    record: Mapping[str, Any], runtime: CandidateRuntime, *, validation_scope: str
) -> dict[str, Any]:
    training = runtime.manifest["training_candidate_identity"]["identity"]["candidate"]
    return {
        "schema_version": CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION,
        "window_id": record["window_id"],
        **{name: record[name] for name in _SCOPE_FIELDS},
        "parent_tracklet_id": record["parent_tracklet_id"],
        "window_start_sec": float(record["window_start_sec"]),
        "window_end_sec": float(record["window_end_sec"]),
        "quality_flags": list(record["quality_flags"]),
        "model_purpose": "primary_candidate_camera_engineering",
        "validation_scope": validation_scope,
        "evidence_scope": EVIDENCE_SCOPE,
        "candidate_id": runtime.manifest["candidate_id"],
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "model_state_sha256": runtime.manifest["artifacts"]["model_state"]["sha256"],
        "primary_seed": training["seed"],
        "best_epoch": training["best_epoch"],
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "wp_prefix_padding_used": False,
    }


def _empty_prediction(
    record: Mapping[str, Any],
    runtime: CandidateRuntime,
    *,
    validation_scope: str,
    status: str,
    reason_codes: Sequence[str],
    model_invocation_skipped: bool,
) -> dict[str, Any]:
    return {
        **_prediction_envelope(record, runtime, validation_scope=validation_scope),
        "window_status": status,
        "reason_codes": list(reason_codes),
        "binary": None,
        "subtype": None,
        "four_class": None,
        "model_invocation_skipped": model_invocation_skipped,
    }


def _named_probabilities(
    probabilities: np.ndarray, order: Sequence[str], predicted_index: int
) -> dict[str, Any]:
    return {
        "class_order": list(order),
        "probabilities": probabilities.astype(np.float64, copy=False).tolist(),
        "predicted_label": order[predicted_index],
    }


def _verify_preprocessing_roots(
    root: Path, camera_config: Mapping[str, Any]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name in ("preprocessing_config", "preprocessing_manifest", "preprocessing_feature_stats"):
        output[name] = _verify_project_descriptor(root, camera_config["trust_roots"][name], name)
    return output


def _load_feature_stats(
    paths: Mapping[str, Path], camera_config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _load_canonical_json(paths["preprocessing_manifest"])
    stats = _load_canonical_json(paths["preprocessing_feature_stats"])
    descriptor = manifest.get("artifacts", {}).get("feature_stats.json")
    stats_path = paths["preprocessing_feature_stats"]
    if descriptor != {"byte_count": stats_path.stat().st_size, "sha256": _sha256_file(stats_path)}:
        raise PrimaryCameraInferenceError("preprocessing manifest no longer binds feature stats")
    if manifest.get("preprocessing_config_sha256") != camera_config["trust_roots"]["preprocessing_config"]["sha256"]:
        raise PrimaryCameraInferenceError("preprocessing config binding has drifted")
    if stats.get("schema_version") != "wandering-feature-stats-v1" or stats.get(
        "temporal_features_enabled"
    ) is not False:
        raise PrimaryCameraInferenceError("feature stats contract has drifted")
    if stats.get("binding_hashes", {}).get("preprocessing_config") != camera_config[
        "trust_roots"
    ]["preprocessing_config"]["sha256"]:
        raise PrimaryCameraInferenceError("feature stats preprocessing binding has drifted")
    if stats.get("binding_hashes", {}).get("split_sha256") != manifest.get("split_sha256"):
        raise PrimaryCameraInferenceError("feature stats training split binding has drifted")
    return stats


def _load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrimaryCameraInferenceError(f"cannot parse trusted JSON: {path.name}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise PrimaryCameraInferenceError(f"trusted JSON is not canonical: {path.name}")
    return value


def _preflight_synthetic_media_sidecar(
    path: str | Path,
    camera_config: Mapping[str, Any],
) -> str:
    sidecar_path = Path(path)
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        media = validate_media_sidecar(raw, camera_config)
        validation_scope = validation_scope_for_authorization_status(
            media["authorization_status"]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CameraAdapterError) as exc:
        raise PrimaryCameraInferenceError("synthetic media sidecar preflight failed closed") from exc
    if (
        media["authorization_status"] != "synthetic_fixture"
        or validation_scope != "synthetic_camera_contract"
    ):
        raise PrimaryCameraInferenceError(
            "primary camera inference entry is synthetic-only; non-synthetic authorization is rejected"
        )
    return validation_scope


def _preflight_active_source_identity(
    *,
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping):
        raise PrimaryCameraInferenceError("active source identity candidate config is invalid")
    expected_manifest = {
        "size_bytes": int(candidate["manifest_size_bytes"]),
        "sha256": str(candidate["manifest_sha256"]),
    }
    try:
        observed_manifest = {
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
        }
    except OSError as exc:
        raise PrimaryCameraInferenceError("active source identity manifest is unreadable") from exc
    if (
        expected_manifest_sha256 != expected_manifest["sha256"]
        or observed_manifest != expected_manifest
    ):
        raise PrimaryCameraInferenceError("active source identity manifest path/size/SHA has drifted")
    manifest = _load_canonical_json(manifest_path)
    release_files = (
        manifest.get("release_implementation_identity", {})
        .get("identity", {})
        .get("files")
    )
    training_files = (
        manifest.get("training_candidate_identity", {})
        .get("identity", {})
        .get("training_source_bundle", {})
        .get("files")
    )
    if not isinstance(release_files, Mapping) or not isinstance(training_files, Mapping):
        raise PrimaryCameraInferenceError("active source identity file maps are invalid")

    model_key = _MODEL_SOURCE_RELATIVE.as_posix()
    release_key = _RELEASE_SOURCE_RELATIVE.as_posix()
    release_model = _source_descriptor(release_files.get(model_key), "release model.py")
    training_model = _source_descriptor(training_files.get(model_key), "training model.py")
    release_source = _source_descriptor(release_files.get(release_key), "release release.py")
    if training_model != release_model:
        raise PrimaryCameraInferenceError(
            "active source identity training/release model.py descriptors disagree"
        )

    model_observed = _observed_imported_source_descriptor(
        root=root,
        relative=_MODEL_SOURCE_RELATIVE,
        module=_MODEL_SOURCE_MODULE,
        role="model.py",
    )
    release_observed = _observed_imported_source_descriptor(
        root=root,
        relative=_RELEASE_SOURCE_RELATIVE,
        module=_RELEASE_SOURCE_MODULE,
        role="release.py",
    )
    if model_observed != release_model or release_observed != release_source:
        raise PrimaryCameraInferenceError("active source identity descriptor mismatch")
    return {
        "passed": True,
        "verified_before_candidate_loader": True,
        "candidate_manifest": {
            "path": str(candidate["manifest_path"]),
            "expected": expected_manifest,
            "observed": observed_manifest,
        },
        "training_release_model_descriptor_equal": True,
        "sources": {
            "model": {
                "path": model_key,
                "expected": release_model,
                "observed": model_observed,
                "training_expected": training_model,
                "release_expected": release_model,
            },
            "release": {
                "path": release_key,
                "expected": release_source,
                "observed": release_observed,
            },
        },
    }


def _source_descriptor(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "size_bytes"}:
        raise PrimaryCameraInferenceError(f"active source identity {role} descriptor is invalid")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise PrimaryCameraInferenceError(f"active source identity {role} descriptor is invalid")
    return {"sha256": sha256, "size_bytes": size_bytes}


def _observed_imported_source_descriptor(
    *,
    root: Path,
    relative: Path,
    module: Any,
    role: str,
) -> dict[str, Any]:
    imported_file = getattr(module, "__file__", None)
    if not isinstance(imported_file, str) or not imported_file:
        raise PrimaryCameraInferenceError(f"active source identity {role} import path is unavailable")
    try:
        imported_path = Path(imported_file).resolve(strict=True)
        expected_path = (root / relative).resolve(strict=True)
        imported_path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PrimaryCameraInferenceError(
            f"active source identity {role} import path escapes the current project root"
        ) from exc
    if imported_path != expected_path:
        raise PrimaryCameraInferenceError(
            f"active source identity {role} import path is not the fixed repository source"
        )
    return {
        "sha256": _sha256_file(imported_path),
        "size_bytes": imported_path.stat().st_size,
    }


def _verify_project_descriptor(root: Path, descriptor: Any, role: str) -> Path:
    if not isinstance(descriptor, Mapping) or set(descriptor) not in (
        {"path", "sha256"},
        {"path", "sha256", "size_bytes"},
    ):
        raise PrimaryCameraInferenceError(f"{role} descriptor is invalid")
    relative = Path(str(descriptor["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PrimaryCameraInferenceError(f"{role} path escapes project root")
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PrimaryCameraInferenceError(f"{role} path escapes project root") from exc
    if not path.is_file():
        raise PrimaryCameraInferenceError(f"{role} file is missing")
    size_matches = "size_bytes" not in descriptor or path.stat().st_size == descriptor["size_bytes"]
    if not size_matches or _sha256_file(path) != descriptor["sha256"]:
        raise PrimaryCameraInferenceError(f"{role} file/hash/size mismatch")
    return path


def _configure_cpu_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    requested_intra = int(runtime["intra_op_threads"])
    requested_inter = int(runtime["inter_op_threads"])
    previous_intra = torch.get_num_threads()
    torch.set_num_threads(requested_intra)
    interop_set = True
    try:
        torch.set_num_interop_threads(requested_inter)
    except RuntimeError:
        interop_set = False
    return {
        "requested": {"intra_op": requested_intra, "inter_op": requested_inter},
        "observed": {
            "intra_op": torch.get_num_threads(),
            "inter_op": torch.get_num_interop_threads(),
        },
        "previous_intra_op": previous_intra,
        "interop_set_in_process": interop_set,
    }


def _execution_record(
    *,
    config: Mapping[str, Any],
    runtime_observation: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    validation_scope: str,
    episode_merge_gap_seconds: float,
    predictions: Sequence[Mapping[str, Any]],
    latency_ms: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(latency_ms, dtype=np.float64)
    latency = {
        "scope": "engineering_cpu_model_forward_only",
        "sample_count": int(values.size),
        "p50": float(np.percentile(values, 50)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "product_threshold": None,
    }
    return {
        "schema_version": CAMERA_PRIMARY_EXECUTION_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": validation_scope,
        "active_source_identity_preflight": dict(source_identity),
        "runtime": {
            "device": "cpu",
            "input_dtype": "float32",
            "cpu_threads": {
                "intra_op": config["runtime"]["intra_op_threads"],
                "inter_op": config["runtime"]["inter_op_threads"],
            },
            "observed_cpu_threads": dict(runtime_observation["observed"]),
            "interop_set_in_process": runtime_observation["interop_set_in_process"],
            "maximum_batch_size": config["runtime"]["maximum_batch_size"],
            "observed_batch_sizes": [1 for row in predictions if row["model_invocation_skipped"] is False],
            "model_mode": "eval",
            "autograd_mode": "torch.inference_mode",
        },
        "latency_ms": latency,
        "model_forward_invocation_count": sum(
            row["model_invocation_skipped"] is False for row in predictions
        ),
        "wp_prefix_padding_used": False,
        "candidate_runtime_predict_logits_used": False,
        "episode_policy_status": "development_unfrozen",
        "episode_merge_gap_seconds": float(episode_merge_gap_seconds),
        "probability_calibrated": False,
        "models_retrained": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def _model_bindings(
    config: Mapping[str, Any],
    runtime: CandidateRuntime,
    *,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_PRIMARY_MODEL_BINDINGS_SCHEMA_VERSION,
        "candidate_id": runtime.manifest["candidate_id"],
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "model_state_sha256": config["candidate"]["model_state_sha256"],
        "primary_seed": config["candidate"]["primary_seed"],
        "best_epoch": config["candidate"]["best_epoch"],
        "four_class_order": list(FOUR_CLASS_ORDER),
        "subtype_order": list(SUBTYPE_ORDER),
        "binary_class_order": list(BINARY_CLASS_ORDER),
        "binary_decision_threshold": 0.5,
        "active_source_identity_preflight": dict(source_identity),
        "probability_calibrated": False,
        "direct_model_forward": True,
        "candidate_runtime_predict_logits_used": False,
        "wp_prefix_padding_used": False,
        "retrained": False,
        "ensemble": False,
    }


def _window_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in _SCOPE_FIELDS) + (
        row["parent_tracklet_id"],
        float(row["window_start_sec"]),
        float(row["window_end_sec"]),
        row["window_id"],
    )


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
        "src/elderly_monitoring/modules/mental_health/wandering/camera_primary_inference.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_episode.py",
        "scripts/wandering/run_topowander_camera_inference.py",
    )
    output: dict[str, str] = {}
    for relative in paths:
        path = (root / relative).resolve(strict=False)
        if not path.is_file():
            raise PrimaryCameraInferenceError(f"primary camera source file is missing: {relative}")
        output[relative] = _sha256_file(path)
    return output


def _commit_new_output_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"primary camera output already exists: {output}")
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
            raise FileExistsError(f"primary camera output already exists: {output}")
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


def _expected_primary_camera_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_PRIMARY_CONFIG_SCHEMA_VERSION,
        "purpose": "topowander_m0cam_primary_camera_engineering",
        "evidence_scope": EVIDENCE_SCOPE,
        "camera_chain": {
            "path": "configs/modules/wandering_camera_v1.yaml",
            "sha256": "08bbec6ef263dd45fef3262c407ce675107f87ea584584f7ab61142cd734ae8e",
            "size_bytes": 2500,
        },
        "candidate": {
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "manifest_path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
            "manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "manifest_size_bytes": 12642,
            "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
            "primary_seed": 20260731,
            "best_epoch": 5,
            "retrained": False,
            "ensemble": False,
        },
        "runtime": {
            "device": "cpu",
            "intra_op_threads": 8,
            "inter_op_threads": 1,
            "maximum_batch_size": 64,
        },
        "inference": {
            "direct_model_forward": True,
            "use_candidate_runtime_predict_logits": False,
            "wp_prefix_padding": False,
            "autograd_mode": "torch.inference_mode",
            "model_mode": "eval",
            "input_dtype": "float32",
            "probability_calibrated": False,
            "binary_decision_threshold": 0.5,
        },
        "class_order": {
            "binary": list(BINARY_CLASS_ORDER),
            "subtype": list(SUBTYPE_ORDER),
            "four_class": list(FOUR_CLASS_ORDER),
        },
        "episode_policy": {
            "policy_status": "development_unfrozen",
            "merge_gap_must_be_explicit": True,
            "alert_decision": None,
        },
        "output_schemas": {
            "prediction": CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION,
            "episode_candidate": "wandering-camera-episode-candidate-v1",
            "run_manifest": CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION,
        },
        "data_access": {
            "synthetic_fixture": True,
            "real_human_media": False,
            "wp_raw": False,
            "wp_public_holdout": False,
            "smartcare_official_or_raw": False,
            "sealed_camera": False,
        },
    }


__all__ = [
    "CAMERA_PRIMARY_CONFIG_SCHEMA_VERSION",
    "CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION",
    "CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION",
    "EXPECTED_CANDIDATE_MANIFEST_SHA256",
    "PrimaryCameraBuildResult",
    "PrimaryCameraInferenceError",
    "build_primary_camera_inference_bundle",
    "load_primary_camera_config",
    "load_primary_camera_runtime",
    "predict_primary_camera_window",
]
