"""Session-level product evidence derived from the fixed primary camera bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    build_primary_camera_inference_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    CameraEpisodeError,
    aggregate_episode_candidates,
)


SESSION_EVIDENCE_SCHEMA_VERSION = "wandering-session-evidence-v1"
PRODUCT_MANIFEST_SCHEMA_VERSION = "wandering-camera-product-manifest-v1"
PRODUCT_STAGE = "session_evidence_prototype"
PRODUCT_STATUS = "wandering_m0cam_session_evidence_prototype"
EVIDENCE_SCOPE = "synthetic_contract_only"
VALIDATION_SCOPE = "synthetic_camera_contract"
FIXED_CANDIDATE_ID = "topowander-m0s-seed20260731-epoch0005"
FIXED_MODEL_STATE_SHA256 = (
    "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"
)
FIXED_PRIMARY_SEED = 20260731
FIXED_BEST_EPOCH = 5
FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")
BINARY_CLASS_ORDER = ("direct_or_non_wandering", "wandering_like")
SUBTYPE_ORDER = ("pacing", "lapping", "random")
_PRIMARY_CONFIG_RELATIVE = Path("configs/modules/wandering_camera_primary_v1.yaml")
_PRIMARY_MANIFEST_RELATIVE = Path(
    "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1"
    "/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json"
)
_SOURCE_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
)
_PRIMARY_ARTIFACTS = frozenset(
    {
        "media_sidecar.json",
        "tracking_input.jsonl",
        "bbox_tracklets.jsonl",
        "window_records.jsonl",
        "predictions.jsonl",
        "episode_candidates.jsonl",
        "qc_summary.json",
        "model_bindings.json",
        "execution.json",
    }
)
_MISSING_CONDITIONS = (
    "stable_person_binding",
    "absolute_time_and_presence",
    "daily_evidence_and_personal_baseline",
    "frozen_risk_policy",
)
_MEDIA_FIELDS = frozenset(
    {
        "schema_version",
        "source_video_id",
        "source_group_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "media_ref",
        "source_sha256",
        "tracking_jsonl_sha256",
        "video_width",
        "video_height",
        "nominal_fps",
        "duration_sec",
        "capture_started_at",
        "timezone",
        "coordinate_system",
        "detector",
        "tracker",
        "fixed_camera_assumed",
        "camera_motion_state",
        "authorization_status",
        "deidentification_status",
    }
)
_DETECTOR_FIELDS = frozenset({"backend", "model", "version"})
_TRACKER_FIELDS = frozenset({"backend", "config", "version"})
_TRACKING_FIELDS = frozenset(
    {"bbox", "frame_id", "timestamp_sec", "track_confidence", "track_id"}
)
_TRACKLET_FIELDS = frozenset(
    {
        "schema_version",
        "tracklet_id",
        "segment_index",
        "source_video_id",
        "source_group_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "frame_width",
        "frame_height",
        "source_fps",
        "frame_indices",
        "point_times_sec",
        "bbox_xyxy_norm",
        "bbox_bottom_points",
        "bbox_heights",
        "detection_confidence",
        "observed_mask",
        "interpolated_mask",
        "quality_flags",
        "media_ref",
        "source_sha256",
        "tracking_jsonl_sha256",
    }
)
_WINDOW_FIELDS = frozenset(
    {
        "schema_version",
        "window_id",
        "parent_tracklet_id",
        "segment_index",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "window_start_sec",
        "window_end_sec",
        "window_status",
        "reason_codes",
        "quality_flags",
        "raw_detection_count",
        "observed_bucket_count",
        "interpolated_bucket_count",
        "observed_ratio",
        "longest_gap_seconds",
        "mean_detection_confidence",
        "bbox_bottom_points",
        "bbox_heights",
        "smoothed_bbox_heights",
        "corrected_points",
        "observed_mask",
        "interpolated_mask",
        "point_quality",
        "shape_normalized_points",
        "point_mask",
        "raw_features",
        "model_features",
        "topology",
        "preprocessing_config_sha256",
        "feature_stats_sha256",
        "model_training_split_sha256",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "window_id",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "parent_tracklet_id",
        "window_start_sec",
        "window_end_sec",
        "quality_flags",
        "model_purpose",
        "validation_scope",
        "evidence_scope",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "binary_decision_threshold",
        "probability_calibrated",
        "wp_prefix_padding_used",
        "window_status",
        "reason_codes",
        "binary",
        "subtype",
        "four_class",
        "model_invocation_skipped",
    }
)
_NAMED_PROBABILITY_FIELDS = frozenset(
    {"class_order", "probabilities", "predicted_label"}
)
_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_candidate_id",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "parent_tracklet_id",
        "episode_start_sec",
        "episode_end_sec_exclusive",
        "contributing_window_ids",
        "predicted_pattern",
        "binary_probability_summary",
        "four_class_probability_summary",
        "prediction_status",
        "evidence_scope",
        "policy_status",
        "merge_gap_seconds",
        "probability_calibrated",
        "alert_decision",
    }
)
_PROBABILITY_SUMMARY_FIELDS = frozenset({"class_order", "mean_probabilities"})
_QC_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_scope",
        "validation_scope",
        "observation_count",
        "normalized_tracking_row_count",
        "input_track_scope_count",
        "tracklet_count",
        "window_count",
        "window_status_counts",
        "reason_counts",
        "episode_candidate_count",
        "camera_motion_state",
    }
)
_WINDOW_STATUS_COUNT_FIELDS = frozenset(
    {"ready", "unavailable", "inference_error"}
)
_MODEL_BINDINGS_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "four_class_order",
        "subtype_order",
        "binary_class_order",
        "binary_decision_threshold",
        "active_source_identity_preflight",
        "probability_calibrated",
        "direct_model_forward",
        "candidate_runtime_predict_logits_used",
        "wp_prefix_padding_used",
        "retrained",
        "ensemble",
    }
)
_SOURCE_PREFLIGHT_FIELDS = frozenset(
    {
        "passed",
        "verified_before_candidate_loader",
        "candidate_manifest",
        "training_release_model_descriptor_equal",
        "sources",
    }
)
_PREFLIGHT_CANDIDATE_MANIFEST_FIELDS = frozenset({"path", "expected", "observed"})
_PREFLIGHT_SOURCE_FIELDS = frozenset({"model", "release"})
_PREFLIGHT_MODEL_FIELDS = frozenset(
    {"path", "expected", "observed", "training_expected", "release_expected"}
)
_PREFLIGHT_RELEASE_FIELDS = frozenset({"path", "expected", "observed"})
_SOURCE_DESCRIPTOR_FIELDS = frozenset({"sha256", "size_bytes"})
_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "evidence_scope",
        "validation_scope",
        "active_source_identity_preflight",
        "runtime",
        "latency_ms",
        "model_forward_invocation_count",
        "wp_prefix_padding_used",
        "candidate_runtime_predict_logits_used",
        "episode_policy_status",
        "episode_merge_gap_seconds",
        "probability_calibrated",
        "models_retrained",
        "environment",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "device",
        "input_dtype",
        "cpu_threads",
        "observed_cpu_threads",
        "interop_set_in_process",
        "maximum_batch_size",
        "observed_batch_sizes",
        "model_mode",
        "autograd_mode",
    }
)
_CPU_THREAD_FIELDS = frozenset({"intra_op", "inter_op"})
_LATENCY_FIELDS = frozenset(
    {"scope", "sample_count", "p50", "p95", "product_threshold"}
)
_ENVIRONMENT_FIELDS = frozenset({"python", "numpy", "torch", "pyyaml"})
_PRIMARY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "evidence_scope",
        "artifacts",
        "primary_camera_config_sha256",
        "camera_chain_config_sha256",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "source_tracking_sha256",
        "normalized_tracking_sha256",
        "source_media_sha256",
        "validation_scope",
        "code_hashes",
        "data_access",
        "models_retrained",
        "wp_prefix_padding_used",
        "algorithm_event_emitted",
        "risk_or_alert_decision_emitted",
    }
)
_PRIMARY_SOURCE_HASH_FIELDS = frozenset(
    {
        "scripts/wandering/run_topowander_camera_inference.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_adapter.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_qc.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_inference.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_primary_inference.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_episode.py",
    }
)
_PRIMARY_DATA_ACCESS = {
    "synthetic_fixture": True,
    "real_human_media": False,
    "wp_raw": False,
    "wp_public_holdout": False,
    "smartcare_official_or_raw": False,
    "sealed_camera": False,
}
_SESSION_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "product_stage",
        "session_status",
        "degraded",
        "evidence_scope",
        "validation_scope",
        "source_scope",
        "person_id",
        "person_binding_verified",
        "observation_count",
        "tracklet_count",
        "window_count",
        "ready_window_count",
        "unavailable_window_count",
        "inference_error_window_count",
        "ready_window_ratio",
        "qc_reason_counts",
        "quality_flag_counts",
        "camera_motion_state",
        "episode_candidate_count",
        "episode_type_counts",
        "direct_duration_sum_seconds",
        "pacing_duration_sum_seconds",
        "lapping_duration_sum_seconds",
        "random_duration_sum_seconds",
        "episode_candidate_duration_sum_seconds",
        "duration_semantics",
        "episode_policy_status",
        "episode_merge_gap_seconds",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "probability_calibrated",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
        "algorithm_event_status",
        "missing_conditions",
    }
)
_PRODUCT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "product_stage",
        "evidence_scope",
        "validation_scope",
        "session_status",
        "degraded",
        "artifacts",
        "primary_manifest_schema_version",
        "wandering_evidence_schema_version",
        "primary_builder_invocation_count",
        "algorithm_event_emitted",
        "risk_or_alert_decision_emitted",
        "real_human_media_consumed",
        "m0cam_d_started",
    }
)
_PRODUCT_TOP_LEVEL = frozenset(
    {"primary", "wandering_evidence.json", "manifest.json"}
)
_FORBIDDEN_PRIMARY_DECISION_FIELDS = frozenset(
    {
        "risk",
        "risk_level",
        "risk_score",
        "diagnosis",
        "medical_diagnosis",
        "action",
        "recommended_action",
        "algorithm_event",
        "AlgorithmEvent",
        "alert_decision",
    }
)


class WanderingCameraProductError(ValueError):
    """A primary bundle cannot be promoted to session-level evidence safely."""


@dataclass(frozen=True)
class WanderingCameraProductBuildResult:
    output_dir: Path
    manifest_sha256: str
    session_status: str
    degraded: bool
    observation_count: int
    window_count: int
    episode_candidate_count: int


@dataclass(frozen=True)
class ValidatedWanderingCameraProduct:
    """Read-only view of a fully revalidated MVP-1 final product."""

    output_dir: Path
    manifest_sha256: str
    manifest_bytes: bytes
    product_manifest: Mapping[str, Any]
    evidence: Mapping[str, Any]
    media: Mapping[str, Any]
    tracking: tuple[Mapping[str, Any], ...]
    tracklets: tuple[Mapping[str, Any], ...]
    windows: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]
    episodes: tuple[Mapping[str, Any], ...]
    qc_summary: Mapping[str, Any]
    model_bindings: Mapping[str, Any]
    execution: Mapping[str, Any]
    primary_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _ValidatedPrimaryBundle:
    media: Mapping[str, Any]
    tracking: tuple[Mapping[str, Any], ...]
    tracklets: tuple[Mapping[str, Any], ...]
    windows: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]
    episodes: tuple[Mapping[str, Any], ...]
    qc_summary: Mapping[str, Any]
    model_bindings: Mapping[str, Any]
    execution: Mapping[str, Any]
    manifest: Mapping[str, Any]


def build_wandering_camera_product(
    *,
    project_root: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    episode_merge_gap_seconds: float,
    output_dir: str | Path,
) -> WanderingCameraProductBuildResult:
    """Run the fixed synthetic primary once and atomically add session evidence."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"wandering camera product output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        primary_dir = staging / "primary"
        primary_result = build_primary_camera_inference_bundle(
            camera_config_path=root / _PRIMARY_CONFIG_RELATIVE,
            project_root=root,
            tracking_jsonl_path=tracking_jsonl_path,
            media_sidecar_path=media_sidecar_path,
            manifest_path=root / _PRIMARY_MANIFEST_RELATIVE,
            expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
            episode_merge_gap_seconds=episode_merge_gap_seconds,
            output_dir=primary_dir,
        )
        if Path(primary_result.output_dir).resolve(strict=False) != primary_dir.resolve(strict=False):
            raise WanderingCameraProductError("primary builder returned an unexpected output path")

        evidence = summarize_primary_camera_bundle(primary_dir)
        _validate_primary_result_counts(primary_result, evidence)
        evidence_bytes = _canonical_json(evidence, "wandering evidence")
        primary_manifest_bytes = (primary_dir / "manifest.json").read_bytes()
        product_manifest = _product_manifest(
            evidence=evidence,
            primary_manifest_bytes=primary_manifest_bytes,
            evidence_bytes=evidence_bytes,
        )
        manifest_bytes = _canonical_json(product_manifest, "product manifest")
        _write_new_file(staging / "wandering_evidence.json", evidence_bytes)
        _write_new_file(staging / "manifest.json", manifest_bytes)
        _verify_product_staging(staging, evidence_bytes, manifest_bytes)
        if output.exists():
            raise FileExistsError(f"wandering camera product output already exists: {output}")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return WanderingCameraProductBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        session_status=str(evidence["session_status"]),
        degraded=bool(evidence["degraded"]),
        observation_count=int(evidence["observation_count"]),
        window_count=int(evidence["window_count"]),
        episode_candidate_count=int(evidence["episode_candidate_count"]),
    )


def summarize_primary_camera_bundle(primary_dir: str | Path) -> dict[str, Any]:
    """Validate a complete fixed-primary bundle and return canonicalizable evidence."""

    bundle = _load_and_validate_primary_bundle(Path(primary_dir))
    return _summarize_validated_primary_bundle(bundle)


def load_validated_wandering_camera_product(
    product_dir: str | Path,
) -> ValidatedWanderingCameraProduct:
    """Revalidate an exact MVP-1 final and expose its already-validated rows."""

    product = Path(product_dir)
    if not product.is_dir():
        raise WanderingCameraProductError("product bundle directory is missing")
    entries = {path.name: path for path in product.iterdir()}
    if (
        set(entries) != _PRODUCT_TOP_LEVEL
        or not entries["primary"].is_dir()
        or not entries["wandering_evidence.json"].is_file()
        or not entries["manifest.json"].is_file()
    ):
        raise WanderingCameraProductError("product top-level collection has drifted")

    evidence_path = product / "wandering_evidence.json"
    manifest_path = product / "manifest.json"
    evidence_bytes = evidence_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    evidence = _load_canonical_json(evidence_path, "wandering evidence")
    manifest = _load_canonical_json(manifest_path, "product manifest")
    _reject_nonempty_primary_decisions(evidence, "wandering evidence")
    _reject_nonempty_primary_decisions(manifest, "product manifest")
    _require_exact_schema(
        evidence,
        SESSION_EVIDENCE_SCHEMA_VERSION,
        _SESSION_EVIDENCE_FIELDS,
        "wandering evidence",
    )
    _require_exact_schema(
        manifest,
        PRODUCT_MANIFEST_SCHEMA_VERSION,
        _PRODUCT_MANIFEST_FIELDS,
        "product manifest",
    )
    _require_exact_fields(
        evidence.get("source_scope"),
        frozenset(_SOURCE_SCOPE_FIELDS),
        "wandering evidence source_scope",
    )
    _require_exact_fields(
        evidence.get("episode_type_counts"),
        frozenset(FOUR_CLASS_ORDER),
        "wandering evidence episode_type_counts",
    )

    primary = _load_and_validate_primary_bundle(product / "primary")
    recomputed_evidence = _summarize_validated_primary_bundle(primary)
    if evidence != recomputed_evidence:
        raise WanderingCameraProductError(
            "wandering evidence does not match the fully revalidated primary bundle"
        )
    expected_manifest = _product_manifest(
        evidence=evidence,
        primary_manifest_bytes=(product / "primary/manifest.json").read_bytes(),
        evidence_bytes=evidence_bytes,
    )
    if manifest != expected_manifest:
        raise WanderingCameraProductError(
            "product manifest does not match the revalidated product collection"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise WanderingCameraProductError("product manifest artifacts are invalid")
    for relative, descriptor in artifacts.items():
        _verify_artifact(product / relative, descriptor, relative)

    return ValidatedWanderingCameraProduct(
        output_dir=product,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_bytes=manifest_bytes,
        product_manifest=manifest,
        evidence=evidence,
        media=primary.media,
        tracking=primary.tracking,
        tracklets=primary.tracklets,
        windows=primary.windows,
        predictions=primary.predictions,
        episodes=primary.episodes,
        qc_summary=primary.qc_summary,
        model_bindings=primary.model_bindings,
        execution=primary.execution,
        primary_manifest=primary.manifest,
    )


def _summarize_validated_primary_bundle(
    bundle: _ValidatedPrimaryBundle,
) -> dict[str, Any]:
    predictions = bundle.predictions
    episodes = bundle.episodes
    statuses = Counter(str(row["window_status"]) for row in predictions)
    ready_count = statuses["ready"]
    unavailable_count = statuses["unavailable"]
    error_count = statuses["inference_error"]
    window_count = len(predictions)
    if ready_count:
        session_status = "ready"
    elif error_count:
        session_status = "inference_error"
    else:
        session_status = "unavailable"
    degraded = bool(ready_count and (unavailable_count or error_count))

    reason_counts = Counter(
        str(reason)
        for row in predictions
        for reason in _string_list(row.get("reason_codes"), "prediction reason_codes")
    )
    quality_flag_counts = Counter(
        str(flag)
        for row in predictions
        for flag in _string_list(row.get("quality_flags"), "prediction quality_flags")
    )
    type_counts = Counter(str(row["predicted_pattern"]) for row in episodes)
    duration_sums = {name: 0.0 for name in FOUR_CLASS_ORDER}
    for episode in episodes:
        pattern = str(episode["predicted_pattern"])
        duration_sums[pattern] += float(episode["episode_end_sec_exclusive"]) - float(
            episode["episode_start_sec"]
        )

    bindings = bundle.model_bindings
    execution = bundle.execution
    return {
        "schema_version": SESSION_EVIDENCE_SCHEMA_VERSION,
        "product_stage": PRODUCT_STAGE,
        "session_status": session_status,
        "degraded": degraded,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "source_scope": {name: bundle.media[name] for name in _SOURCE_SCOPE_FIELDS},
        "person_id": None,
        "person_binding_verified": False,
        "observation_count": int(bundle.qc_summary["observation_count"]),
        "tracklet_count": len(bundle.tracklets),
        "window_count": window_count,
        "ready_window_count": ready_count,
        "unavailable_window_count": unavailable_count,
        "inference_error_window_count": error_count,
        "ready_window_ratio": ready_count / window_count if window_count else 0.0,
        "qc_reason_counts": {name: reason_counts[name] for name in sorted(reason_counts)},
        "quality_flag_counts": {
            name: quality_flag_counts[name] for name in sorted(quality_flag_counts)
        },
        "camera_motion_state": bundle.qc_summary["camera_motion_state"],
        "episode_candidate_count": len(episodes),
        "episode_type_counts": {name: type_counts[name] for name in FOUR_CLASS_ORDER},
        **{
            f"{name}_duration_sum_seconds": duration_sums[name]
            for name in FOUR_CLASS_ORDER
        },
        "episode_candidate_duration_sum_seconds": sum(duration_sums.values()),
        "duration_semantics": "candidate_interval_sum_not_presence_time",
        "episode_policy_status": execution["episode_policy_status"],
        "episode_merge_gap_seconds": float(execution["episode_merge_gap_seconds"]),
        "candidate_id": bindings["candidate_id"],
        "candidate_manifest_sha256": bindings["candidate_manifest_sha256"],
        "model_state_sha256": bindings["model_state_sha256"],
        "primary_seed": bindings["primary_seed"],
        "best_epoch": bindings["best_epoch"],
        "probability_calibrated": False,
        "risk_level": None,
        "risk_score": None,
        "recommended_action": None,
        "alert_decision": None,
        "medical_diagnosis": None,
        "algorithm_event": None,
        "algorithm_event_status": "not_ready_session_only",
        "missing_conditions": list(_MISSING_CONDITIONS),
    }


def _product_manifest(
    *,
    evidence: Mapping[str, Any],
    primary_manifest_bytes: bytes,
    evidence_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCT_MANIFEST_SCHEMA_VERSION,
        "status": PRODUCT_STATUS,
        "product_stage": PRODUCT_STAGE,
        "evidence_scope": evidence["evidence_scope"],
        "validation_scope": evidence["validation_scope"],
        "session_status": evidence["session_status"],
        "degraded": evidence["degraded"],
        "artifacts": {
            "primary/manifest.json": _payload_descriptor(primary_manifest_bytes),
            "wandering_evidence.json": _payload_descriptor(evidence_bytes),
        },
        "primary_manifest_schema_version": "wandering-camera-primary-run-manifest-v1",
        "wandering_evidence_schema_version": SESSION_EVIDENCE_SCHEMA_VERSION,
        "primary_builder_invocation_count": 1,
        "algorithm_event_emitted": False,
        "risk_or_alert_decision_emitted": False,
        "real_human_media_consumed": False,
        "m0cam_d_started": False,
    }


def _load_and_validate_primary_bundle(primary: Path) -> _ValidatedPrimaryBundle:
    if not primary.is_dir():
        raise WanderingCameraProductError("primary bundle directory is missing")
    observed_names = {path.name for path in primary.iterdir() if path.is_file()}
    expected_names = set(_PRIMARY_ARTIFACTS) | {"manifest.json"}
    if observed_names != expected_names or any(path.is_dir() for path in primary.iterdir()):
        raise WanderingCameraProductError("primary bundle file set is incomplete or unexpected")

    manifest = _load_canonical_json(primary / "manifest.json", "primary manifest")
    _reject_nonempty_primary_decisions(manifest, "primary manifest")
    _require_exact_schema(
        manifest,
        "wandering-camera-primary-run-manifest-v1",
        _PRIMARY_MANIFEST_FIELDS,
        "primary manifest",
    )
    _validate_primary_manifest_structure(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _PRIMARY_ARTIFACTS:
        raise WanderingCameraProductError("primary manifest artifact set has drifted")
    for name in sorted(_PRIMARY_ARTIFACTS):
        _verify_artifact(primary / name, artifacts[name], name)

    media = _load_canonical_json(primary / "media_sidecar.json", "media sidecar")
    tracking = _load_canonical_jsonl(primary / "tracking_input.jsonl", "tracking input")
    tracklets = _load_canonical_jsonl(primary / "bbox_tracklets.jsonl", "bbox tracklets")
    windows = _load_canonical_jsonl(primary / "window_records.jsonl", "window records")
    predictions = _load_canonical_jsonl(primary / "predictions.jsonl", "predictions")
    episodes = _load_canonical_jsonl(primary / "episode_candidates.jsonl", "episode candidates")
    qc_summary = _load_canonical_json(primary / "qc_summary.json", "QC summary")
    model_bindings = _load_canonical_json(primary / "model_bindings.json", "model bindings")
    execution = _load_canonical_json(primary / "execution.json", "execution")

    for role, value in (
        ("media sidecar", media),
        ("tracking input", tracking),
        ("bbox tracklets", tracklets),
        ("window records", windows),
        ("predictions", predictions),
        ("episode candidates", episodes),
        ("QC summary", qc_summary),
        ("model bindings", model_bindings),
        ("execution", execution),
    ):
        _reject_nonempty_primary_decisions(value, role)

    _require_exact_schema(media, "wandering-media-v1", _MEDIA_FIELDS, "media sidecar")
    _require_exact_fields(media.get("detector"), _DETECTOR_FIELDS, "media detector")
    _require_exact_fields(media.get("tracker"), _TRACKER_FIELDS, "media tracker")
    _require_rows_exact_fields(tracking, _TRACKING_FIELDS, "normalized tracking rows")
    _require_rows_exact_schema(
        tracklets,
        "wandering-bbox-tracklet-v1",
        _TRACKLET_FIELDS,
        "bbox tracklets",
    )
    _require_rows_exact_schema(
        windows,
        "wandering-camera-window-v1",
        _WINDOW_FIELDS,
        "window records",
    )
    _require_rows_exact_schema(
        predictions,
        "wandering-camera-primary-prediction-v1",
        _PREDICTION_FIELDS,
        "predictions",
    )
    _require_rows_exact_schema(
        episodes,
        "wandering-camera-episode-candidate-v1",
        _EPISODE_FIELDS,
        "episode candidates",
    )
    _require_exact_schema(
        qc_summary,
        "wandering-camera-primary-qc-summary-v1",
        _QC_SUMMARY_FIELDS,
        "QC summary",
    )
    _require_exact_fields(
        qc_summary.get("window_status_counts"),
        _WINDOW_STATUS_COUNT_FIELDS,
        "QC window_status_counts",
    )
    _require_exact_schema(
        model_bindings,
        "wandering-camera-primary-model-bindings-v1",
        _MODEL_BINDINGS_FIELDS,
        "model bindings",
    )
    _require_exact_schema(
        execution,
        "wandering-camera-primary-execution-v1",
        _EXECUTION_FIELDS,
        "execution",
    )
    _validate_source_preflight(
        model_bindings.get("active_source_identity_preflight"),
        "model bindings active source preflight",
    )
    _validate_execution_structure(execution)
    for row in predictions:
        _validate_prediction_probability_contract(row)
    for row in episodes:
        _validate_episode_probability_contract(row)
    _validate_primary_scopes(
        media=media,
        tracklets=tracklets,
        windows=windows,
        predictions=predictions,
        episodes=episodes,
    )
    _validate_primary_counts(
        tracking=tracking,
        tracklets=tracklets,
        windows=windows,
        predictions=predictions,
        episodes=episodes,
        qc_summary=qc_summary,
    )
    _validate_primary_semantics(
        media=media,
        tracking=tracking,
        windows=windows,
        predictions=predictions,
        episodes=episodes,
        qc_summary=qc_summary,
        model_bindings=model_bindings,
        execution=execution,
        manifest=manifest,
    )
    return _ValidatedPrimaryBundle(
        media=media,
        tracking=tracking,
        tracklets=tracklets,
        windows=windows,
        predictions=predictions,
        episodes=episodes,
        qc_summary=qc_summary,
        model_bindings=model_bindings,
        execution=execution,
        manifest=manifest,
    )


def _validate_primary_scopes(
    *,
    media: Mapping[str, Any],
    tracklets: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    source_scope = tuple(media.get(name) for name in _SOURCE_SCOPE_FIELDS)
    if any(not isinstance(value, str) or not value for value in source_scope):
        raise WanderingCameraProductError("primary media source scope is invalid")
    for role, rows in (
        ("tracklet", tracklets),
        ("window", windows),
        ("prediction", predictions),
        ("episode", episodes),
    ):
        for row in rows:
            observed = tuple(row.get(name) for name in _SOURCE_SCOPE_FIELDS)
            if observed != source_scope:
                raise WanderingCameraProductError(f"primary {role} source scope mismatch")
            track_id = row.get("track_id")
            if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
                raise WanderingCameraProductError(f"primary {role} track_id is invalid")


def _validate_primary_counts(
    *,
    tracking: Sequence[Mapping[str, Any]],
    tracklets: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    qc_summary: Mapping[str, Any],
) -> None:
    for row in tracking:
        if set(row) != {"bbox", "frame_id", "timestamp_sec", "track_confidence", "track_id"}:
            raise WanderingCameraProductError("normalized tracking row schema has drifted")
    expected_counts = {
        "observation_count": len(tracking),
        "normalized_tracking_row_count": len(tracking),
        "input_track_scope_count": len({row["track_id"] for row in tracking}),
        "tracklet_count": len(tracklets),
        "window_count": len(predictions),
        "episode_candidate_count": len(episodes),
    }
    for name, expected in expected_counts.items():
        value = qc_summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise WanderingCameraProductError(f"primary QC {name} count has drifted")
    if len(windows) != len(predictions):
        raise WanderingCameraProductError("primary window/prediction count mismatch")
    window_ids = [row.get("window_id") for row in windows]
    prediction_ids = [row.get("window_id") for row in predictions]
    if (
        any(not isinstance(value, str) or not value for value in window_ids + prediction_ids)
        or len(set(window_ids)) != len(window_ids)
        or len(set(prediction_ids)) != len(prediction_ids)
        or set(window_ids) != set(prediction_ids)
    ):
        raise WanderingCameraProductError("primary window/prediction IDs are inconsistent")
    statuses = Counter(row.get("window_status") for row in predictions)
    expected_status_counts = {
        name: statuses[name] for name in ("ready", "unavailable", "inference_error")
    }
    if set(statuses) - set(expected_status_counts):
        raise WanderingCameraProductError("primary prediction status is invalid")
    if qc_summary.get("window_status_counts") != expected_status_counts:
        raise WanderingCameraProductError("primary QC window status counts have drifted")
    reason_counts = Counter(
        reason
        for row in predictions
        for reason in _string_list(row.get("reason_codes"), "prediction reason_codes")
    )
    if qc_summary.get("reason_counts") != {
        name: reason_counts[name] for name in sorted(reason_counts)
    }:
        raise WanderingCameraProductError("primary QC reason counts have drifted")


def _validate_primary_semantics(
    *,
    media: Mapping[str, Any],
    tracking: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    qc_summary: Mapping[str, Any],
    model_bindings: Mapping[str, Any],
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    for role, value in (
        ("primary manifest evidence_scope", manifest.get("evidence_scope")),
        ("QC evidence_scope", qc_summary.get("evidence_scope")),
        ("execution evidence_scope", execution.get("evidence_scope")),
    ):
        if value != EVIDENCE_SCOPE:
            raise WanderingCameraProductError(f"{role} has drifted")
    for role, value in (
        ("primary manifest validation_scope", manifest.get("validation_scope")),
        ("QC validation_scope", qc_summary.get("validation_scope")),
        ("execution validation_scope", execution.get("validation_scope")),
    ):
        if value != VALIDATION_SCOPE:
            raise WanderingCameraProductError(f"{role} has drifted")
    if media.get("authorization_status") != "synthetic_fixture":
        raise WanderingCameraProductError("primary media is not a synthetic fixture")
    if qc_summary.get("camera_motion_state") != media.get("camera_motion_state"):
        raise WanderingCameraProductError("primary camera motion state has drifted")
    normalized_payload = _canonical_jsonl(tracking, "normalized tracking")
    if manifest.get("normalized_tracking_sha256") != hashlib.sha256(normalized_payload).hexdigest():
        raise WanderingCameraProductError("primary normalized tracking hash has drifted")
    if manifest.get("source_tracking_sha256") != media.get("tracking_jsonl_sha256"):
        raise WanderingCameraProductError("primary source tracking binding has drifted")
    if manifest.get("source_media_sha256") != media.get("source_sha256"):
        raise WanderingCameraProductError("primary source media binding has drifted")

    expected_identity = {
        "candidate_id": FIXED_CANDIDATE_ID,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "model_state_sha256": FIXED_MODEL_STATE_SHA256,
        "primary_seed": FIXED_PRIMARY_SEED,
        "best_epoch": FIXED_BEST_EPOCH,
    }
    for name, expected in expected_identity.items():
        if model_bindings.get(name) != expected:
            raise WanderingCameraProductError(f"primary fixed model identity {name} has drifted")
    if manifest.get("candidate_manifest_sha256") != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise WanderingCameraProductError("primary manifest candidate identity has drifted")
    if manifest.get("model_state_sha256") != FIXED_MODEL_STATE_SHA256:
        raise WanderingCameraProductError("primary manifest model state identity has drifted")
    if (
        model_bindings.get("four_class_order") != list(FOUR_CLASS_ORDER)
        or model_bindings.get("binary_decision_threshold") != 0.5
        or model_bindings.get("probability_calibrated") is not False
        or model_bindings.get("retrained") is not False
        or model_bindings.get("ensemble") is not False
    ):
        raise WanderingCameraProductError("primary model binding semantics have drifted")
    if (
        execution.get("episode_policy_status") != "development_unfrozen"
        or not _finite_nonnegative(execution.get("episode_merge_gap_seconds"))
        or execution.get("probability_calibrated") is not False
        or execution.get("models_retrained") is not False
    ):
        raise WanderingCameraProductError("primary execution policy semantics have drifted")
    if (
        manifest.get("status") != "wandering_m0cam_primary_camera_engineering_ready"
        or execution.get("status") != "wandering_m0cam_primary_camera_engineering_ready"
        or execution.get("model_forward_invocation_count")
        != sum(row.get("window_status") in {"ready", "inference_error"} for row in predictions)
    ):
        raise WanderingCameraProductError("primary execution status or forward count has drifted")
    source_preflight = model_bindings.get("active_source_identity_preflight")
    if (
        not isinstance(source_preflight, Mapping)
        or source_preflight.get("passed") is not True
        or source_preflight.get("verified_before_candidate_loader") is not True
        or execution.get("active_source_identity_preflight") != source_preflight
    ):
        raise WanderingCameraProductError("primary active source identity preflight has drifted")
    if (
        manifest.get("models_retrained") is not False
        or manifest.get("wp_prefix_padding_used") is not False
        or manifest.get("algorithm_event_emitted") is not False
        or manifest.get("risk_or_alert_decision_emitted") is not False
    ):
        raise WanderingCameraProductError("primary manifest product boundary has drifted")

    ready_ids: set[str] = set()
    for row in predictions:
        if row.get("validation_scope") != VALIDATION_SCOPE or row.get("evidence_scope") != EVIDENCE_SCOPE:
            raise WanderingCameraProductError("primary prediction scope has drifted")
        if row.get("probability_calibrated") is not False:
            raise WanderingCameraProductError("primary prediction calibration status has drifted")
        for name, expected in expected_identity.items():
            if row.get(name) != expected:
                raise WanderingCameraProductError(f"primary prediction {name} has drifted")
        status = row["window_status"]
        if status == "ready":
            ready_ids.add(str(row["window_id"]))
            if any(row.get(name) is None for name in ("binary", "subtype", "four_class")):
                raise WanderingCameraProductError("ready primary prediction is incomplete")
        elif any(row.get(name) is not None for name in ("binary", "subtype", "four_class")):
            raise WanderingCameraProductError("non-ready primary prediction retains probabilities")

    prediction_by_id = {str(row["window_id"]): row for row in predictions}
    episode_ids: set[str] = set()
    contributed_window_ids: set[str] = set()
    for row in episodes:
        episode_id = row.get("episode_candidate_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
            raise WanderingCameraProductError("primary episode candidate ID is invalid")
        episode_ids.add(episode_id)
        pattern = row.get("predicted_pattern")
        if pattern not in FOUR_CLASS_ORDER:
            raise WanderingCameraProductError("primary episode pattern is invalid")
        contributors = _string_list(row.get("contributing_window_ids"), "episode window IDs")
        if (
            not contributors
            or len(set(contributors)) != len(contributors)
            or any(window_id not in ready_ids for window_id in contributors)
            or contributed_window_ids.intersection(contributors)
        ):
            raise WanderingCameraProductError("primary episode includes a non-ready window")
        contributor_rows = [prediction_by_id[window_id] for window_id in contributors]
        if any(
            prediction.get("four_class", {}).get("predicted_label") != pattern
            or prediction.get("parent_tracklet_id") != row.get("parent_tracklet_id")
            or prediction.get("track_id") != row.get("track_id")
            for prediction in contributor_rows
        ):
            raise WanderingCameraProductError("primary episode contributor semantics have drifted")
        contributed_window_ids.update(contributors)
        if (
            row.get("evidence_scope") != EVIDENCE_SCOPE
            or row.get("policy_status") != "development_unfrozen"
            or row.get("merge_gap_seconds") != execution.get("episode_merge_gap_seconds")
            or row.get("probability_calibrated") is not False
            or row.get("alert_decision") is not None
        ):
            raise WanderingCameraProductError("primary episode policy semantics have drifted")
        start = row.get("episode_start_sec")
        end = row.get("episode_end_sec_exclusive")
        if not _finite_number(start) or not _finite_number(end) or float(end) <= float(start):
            raise WanderingCameraProductError("primary episode interval is invalid")
        if float(start) != min(float(item["window_start_sec"]) for item in contributor_rows) or float(
            end
        ) != max(float(item["window_end_sec"]) for item in contributor_rows):
            raise WanderingCameraProductError("primary episode interval/contributor binding has drifted")

    if contributed_window_ids != ready_ids:
        raise WanderingCameraProductError("primary ready-window episode coverage has drifted")

    try:
        expected_episodes = aggregate_episode_candidates(
            predictions,
            merge_gap_seconds=float(execution["episode_merge_gap_seconds"]),
        )
    except CameraEpisodeError as exc:
        raise WanderingCameraProductError(
            "primary prediction probability contract cannot be aggregated"
        ) from exc
    if _canonical_jsonl(expected_episodes, "recomputed episode candidates") != _canonical_jsonl(
        episodes,
        "primary episode candidates",
    ):
        raise WanderingCameraProductError(
            "primary episode probability summary or aggregation has drifted"
        )

    window_by_id = {str(row["window_id"]): row for row in windows}
    for prediction in predictions:
        window = window_by_id[str(prediction["window_id"])]
        expected_window_status = "ready" if prediction["window_status"] == "inference_error" else prediction["window_status"]
        if window.get("window_status") != expected_window_status:
            raise WanderingCameraProductError("primary window/prediction status pairing is invalid")


def _validate_primary_result_counts(result: Any, evidence: Mapping[str, Any]) -> None:
    expected = {
        "observation_count": evidence["observation_count"],
        "tracklet_count": evidence["tracklet_count"],
        "window_count": evidence["window_count"],
        "ready_window_count": evidence["ready_window_count"],
        "unavailable_window_count": evidence["unavailable_window_count"],
        "inference_error_count": evidence["inference_error_window_count"],
        "episode_candidate_count": evidence["episode_candidate_count"],
    }
    for name, value in expected.items():
        if getattr(result, name, None) != value:
            raise WanderingCameraProductError(f"primary builder result {name} count has drifted")


def _verify_product_staging(staging: Path, evidence_bytes: bytes, manifest_bytes: bytes) -> None:
    entries = {path.name: path for path in staging.iterdir()}
    if (
        set(entries) != _PRODUCT_TOP_LEVEL
        or not entries["primary"].is_dir()
        or not entries["wandering_evidence.json"].is_file()
        or not entries["manifest.json"].is_file()
    ):
        raise WanderingCameraProductError("product top-level collection has drifted")
    observed_evidence = (staging / "wandering_evidence.json").read_bytes()
    observed_manifest = (staging / "manifest.json").read_bytes()
    if observed_evidence != evidence_bytes or observed_manifest != manifest_bytes:
        raise WanderingCameraProductError("product staging bytes changed before commit")
    validated = load_validated_wandering_camera_product(staging)
    if validated.manifest_bytes != manifest_bytes:
        raise WanderingCameraProductError("product staging manifest changed during validation")


def _load_canonical_json(path: Path, role: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingCameraProductError(f"cannot parse {role}") from exc
    if not isinstance(value, dict) or raw != _canonical_json(value, role):
        raise WanderingCameraProductError(f"{role} is not canonical JSON")
    return value


def _load_canonical_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
        rows = tuple(json.loads(line.decode("utf-8")) for line in raw.splitlines())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingCameraProductError(f"cannot parse {role}") from exc
    if any(not isinstance(row, dict) for row in rows) or raw != _canonical_jsonl(rows, role):
        raise WanderingCameraProductError(f"{role} is not canonical JSONL")
    return rows


def _canonical_json(value: Any, role: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CameraAdapterError as exc:
        raise WanderingCameraProductError(f"{role} is not finite JSON") from exc


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]], role: str) -> bytes:
    try:
        return canonical_jsonl_bytes(rows)
    except CameraAdapterError as exc:
        raise WanderingCameraProductError(f"{role} is not finite JSONL") from exc


def _verify_artifact(path: Path, descriptor: Any, role: str) -> None:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"byte_count", "sha256"}:
        raise WanderingCameraProductError(f"{role} descriptor is invalid")
    byte_count = descriptor.get("byte_count")
    sha256 = descriptor.get("sha256")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise WanderingCameraProductError(f"{role} descriptor is invalid")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise WanderingCameraProductError(f"{role} artifact is missing or unreadable") from exc
    if len(payload) != byte_count or hashlib.sha256(payload).hexdigest() != sha256:
        raise WanderingCameraProductError(f"{role} artifact size or hash mismatch")


def _require_exact_fields(value: Any, expected: frozenset[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise WanderingCameraProductError(f"{role} field set has drifted")


def _require_exact_schema(
    value: Any,
    expected_schema: str,
    expected_fields: frozenset[str],
    role: str,
) -> None:
    _require_exact_fields(value, expected_fields, role)
    if value.get("schema_version") != expected_schema:
        raise WanderingCameraProductError(f"{role} schema has drifted")


def _require_rows_exact_fields(
    rows: Sequence[Mapping[str, Any]], expected_fields: frozenset[str], role: str
) -> None:
    for row in rows:
        _require_exact_fields(row, expected_fields, role)


def _require_rows_exact_schema(
    rows: Sequence[Mapping[str, Any]],
    expected_schema: str,
    expected_fields: frozenset[str],
    role: str,
) -> None:
    for row in rows:
        _require_exact_schema(row, expected_schema, expected_fields, role)


def _reject_nonempty_primary_decisions(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if name in _FORBIDDEN_PRIMARY_DECISION_FIELDS and nested is not None:
                raise WanderingCameraProductError(
                    f"{role} contains a non-empty primary decision field: {name}"
                )
            _reject_nonempty_primary_decisions(nested, role)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonempty_primary_decisions(nested, role)


def _validate_hex_digest(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WanderingCameraProductError(f"{role} is not a SHA-256 digest")


def _validate_primary_manifest_structure(manifest: Mapping[str, Any]) -> None:
    code_hashes = manifest.get("code_hashes")
    _require_exact_fields(code_hashes, _PRIMARY_SOURCE_HASH_FIELDS, "primary code_hashes")
    for relative, digest in code_hashes.items():
        _validate_hex_digest(digest, f"primary code hash {relative}")
    data_access = manifest.get("data_access")
    if not isinstance(data_access, Mapping) or dict(data_access) != _PRIMARY_DATA_ACCESS:
        raise WanderingCameraProductError("primary manifest data_access has drifted")


def _validate_source_descriptor(value: Any, role: str) -> None:
    _require_exact_fields(value, _SOURCE_DESCRIPTOR_FIELDS, role)
    _validate_hex_digest(value.get("sha256"), f"{role} sha256")
    size_bytes = value.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise WanderingCameraProductError(f"{role} size_bytes is invalid")


def _validate_source_preflight(value: Any, role: str) -> None:
    _require_exact_fields(value, _SOURCE_PREFLIGHT_FIELDS, role)
    candidate = value.get("candidate_manifest")
    _require_exact_fields(candidate, _PREFLIGHT_CANDIDATE_MANIFEST_FIELDS, f"{role} candidate_manifest")
    _validate_source_descriptor(candidate.get("expected"), f"{role} candidate expected")
    _validate_source_descriptor(candidate.get("observed"), f"{role} candidate observed")

    sources = value.get("sources")
    _require_exact_fields(sources, _PREFLIGHT_SOURCE_FIELDS, f"{role} sources")
    model = sources.get("model")
    _require_exact_fields(model, _PREFLIGHT_MODEL_FIELDS, f"{role} model")
    for name in ("expected", "observed", "training_expected", "release_expected"):
        _validate_source_descriptor(model.get(name), f"{role} model {name}")
    release = sources.get("release")
    _require_exact_fields(release, _PREFLIGHT_RELEASE_FIELDS, f"{role} release")
    for name in ("expected", "observed"):
        _validate_source_descriptor(release.get(name), f"{role} release {name}")


def _validate_execution_structure(execution: Mapping[str, Any]) -> None:
    _validate_source_preflight(
        execution.get("active_source_identity_preflight"),
        "execution active source preflight",
    )
    runtime = execution.get("runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, "execution runtime")
    _require_exact_fields(runtime.get("cpu_threads"), _CPU_THREAD_FIELDS, "execution cpu_threads")
    _require_exact_fields(
        runtime.get("observed_cpu_threads"),
        _CPU_THREAD_FIELDS,
        "execution observed_cpu_threads",
    )
    _require_exact_fields(execution.get("latency_ms"), _LATENCY_FIELDS, "execution latency_ms")
    _require_exact_fields(
        execution.get("environment"),
        _ENVIRONMENT_FIELDS,
        "execution environment",
    )


def _probability_vector(value: Any, width: int, role: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not _finite_nonnegative(item) for item in value)
    ):
        raise WanderingCameraProductError(f"{role} probability vector is invalid")
    probabilities = [float(item) for item in value]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise WanderingCameraProductError(f"{role} probability vector is invalid")
    return probabilities


def _validate_named_probability(
    value: Any,
    order: tuple[str, ...],
    role: str,
    *,
    binary: bool = False,
) -> list[float]:
    _require_exact_fields(value, _NAMED_PROBABILITY_FIELDS, f"{role} probability")
    if value.get("class_order") != list(order):
        raise WanderingCameraProductError(f"{role} probability class order has drifted")
    probabilities = _probability_vector(value.get("probabilities"), len(order), role)
    predicted_index = (
        1 if binary and probabilities[1] >= 0.5 else 0
    ) if binary else max(range(len(order)), key=probabilities.__getitem__)
    if value.get("predicted_label") != order[predicted_index]:
        raise WanderingCameraProductError(f"{role} probability predicted label is inconsistent")
    return probabilities


def _validate_prediction_probability_contract(prediction: Mapping[str, Any]) -> None:
    status = prediction.get("window_status")
    if status != "ready":
        if any(prediction.get(name) is not None for name in ("binary", "subtype", "four_class")):
            raise WanderingCameraProductError(
                "non-ready primary prediction retains probabilities"
            )
        return
    if prediction.get("binary_decision_threshold") != 0.5:
        raise WanderingCameraProductError("primary binary probability threshold has drifted")
    binary = _validate_named_probability(
        prediction.get("binary"),
        BINARY_CLASS_ORDER,
        "primary binary",
        binary=True,
    )
    subtype = _validate_named_probability(
        prediction.get("subtype"),
        SUBTYPE_ORDER,
        "primary subtype",
    )
    four = _validate_named_probability(
        prediction.get("four_class"),
        FOUR_CLASS_ORDER,
        "primary four_class",
    )
    expected_four = [binary[0], *(binary[1] * value for value in subtype)]
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6)
        for observed, expected in zip(four, expected_four, strict=True)
    ):
        raise WanderingCameraProductError(
            "primary four_class probability composition has drifted"
        )


def _validate_probability_summary(
    value: Any, order: tuple[str, ...], role: str
) -> list[float]:
    _require_exact_fields(value, _PROBABILITY_SUMMARY_FIELDS, f"{role} probability summary")
    if value.get("class_order") != list(order):
        raise WanderingCameraProductError(f"{role} probability summary class order has drifted")
    return _probability_vector(value.get("mean_probabilities"), len(order), role)


def _validate_episode_probability_contract(episode: Mapping[str, Any]) -> None:
    _validate_probability_summary(
        episode.get("binary_probability_summary"),
        BINARY_CLASS_ORDER,
        "primary episode binary",
    )
    four = _validate_probability_summary(
        episode.get("four_class_probability_summary"),
        FOUR_CLASS_ORDER,
        "primary episode four_class",
    )
    predicted_index = max(range(len(FOUR_CLASS_ORDER)), key=four.__getitem__)
    if episode.get("predicted_pattern") != FOUR_CLASS_ORDER[predicted_index]:
        raise WanderingCameraProductError(
            "primary episode probability summary and pattern are inconsistent"
        )


def _require_schema(value: Mapping[str, Any], expected: str, role: str) -> None:
    if value.get("schema_version") != expected:
        raise WanderingCameraProductError(f"{role} schema has drifted")


def _require_rows_schema(
    rows: Sequence[Mapping[str, Any]], expected: str, role: str
) -> None:
    if any(row.get("schema_version") != expected for row in rows):
        raise WanderingCameraProductError(f"{role} schema has drifted")


def _string_list(value: Any, role: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise WanderingCameraProductError(f"{role} must be a list of non-empty strings")
    return list(value)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _payload_descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "PRODUCT_MANIFEST_SCHEMA_VERSION",
    "SESSION_EVIDENCE_SCHEMA_VERSION",
    "ValidatedWanderingCameraProduct",
    "WanderingCameraProductBuildResult",
    "WanderingCameraProductError",
    "build_wandering_camera_product",
    "load_validated_wandering_camera_product",
    "summarize_primary_camera_bundle",
]
