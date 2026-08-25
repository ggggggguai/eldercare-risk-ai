"""Episode-first oracle-boundary shape inference for camera trajectories."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    group_observations,
    load_camera_inputs,
    validation_scope_for_authorization_status,
    weighted_bucket_observations,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
    CameraEpisodeImportError,
    load_episode_boundaries,
    validate_episode_boundary,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
    prepare_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EVIDENCE_SCOPE,
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_STATE_SHA256,
    FOUR_CLASS_ORDER,
    PrimaryCameraInferenceError,
    _configure_cpu_runtime,
    _load_feature_stats,
    _predict_primary_camera_window_with_provenance,
    _verify_preprocessing_roots,
    load_primary_camera_config,
    load_primary_camera_runtime,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCError,
    CameraWindowInput,
    _interpolate_window,
    _motion_extent_body,
    _split_observations,
    compensate_bbox_height,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)


CAMERA_EPISODE_CONFIG_SCHEMA_VERSION = "wandering-camera-episode-config-v1"
CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION = "wandering-camera-episode-prediction-v1"
CAMERA_EPISODE_RUN_SUMMARY_SCHEMA_VERSION = "wandering-camera-episode-run-summary-v1"
ORACLE_BOUNDARY_EVALUATION_NAME = "oracle-boundary shape classification"
_CONFIG_RELATIVE = Path("configs/modules/wandering_camera_episode_v1.yaml")
_INTERVAL_INPUT_QUALITY_FLAGS = {
    "accepted_boundary": "oracle_boundary",
    "automatic_proposal": "automatic_boundary_proposal",
}
_INTERVAL_PREPARATION_FIELDS = (
    "source_observation_count",
    "accepted_observation_count",
    "source_bucket_count",
    "observed_bucket_count",
    "interpolated_bucket_count",
    "observed_coverage_ratio",
    "interpolated_coverage_ratio",
    "maximum_gap_seconds",
    "motion_extent_body_heights",
    "qc_status",
    "qc_reason_codes",
    "quality_flags",
    "model_features",
    "shape_normalized_points",
    "point_mask",
    "raw_features",
    "topology",
)


class CameraEpisodeInferenceError(ValueError):
    """Episode boundary, QC, preprocessing, or frozen inference failed closed."""


@dataclass(frozen=True)
class CameraEpisodeBuildResult:
    output_dir: Path
    episode_count: int
    ready_count: int
    unavailable_count: int
    boundary_uncertain_count: int
    inference_error_count: int


def load_camera_episode_config(path: str | Path) -> dict[str, Any]:
    """Load the exact EP1A config without altering the legacy 40 s config."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeInferenceError("cannot read camera episode config") from exc
    if value != _expected_episode_config():
        raise CameraEpisodeInferenceError("camera episode config fields or fixed values drifted")
    return value


def build_whole_clip_boundary(
    media: Mapping[str, Any],
    *,
    episode_id: str,
    target_track_id: int,
) -> dict[str, Any]:
    """Declare one explicitly selected track over a short whole-clip episode."""

    value = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_video_id": media.get("source_video_id"),
        "target_track_id": target_track_id,
        "start_sec": 0.0,
        "end_sec_exclusive": media.get("duration_sec"),
        "boundary_source": "whole_clip",
        "boundary_status": "ready",
        "boundary_reason_codes": [],
        "cvat_track_id": None,
    }
    try:
        return validate_episode_boundary(value, media)
    except CameraEpisodeImportError as exc:
        raise CameraEpisodeInferenceError(str(exc)) from exc


def prepare_camera_episode(
    adapter_input: CameraAdapterInput,
    boundary: Mapping[str, Any],
    *,
    camera_config: Mapping[str, Any],
    episode_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an accepted boundary, then apply the shared interval preparation."""

    if not isinstance(adapter_input, CameraAdapterInput):
        raise CameraEpisodeInferenceError("prepare_camera_episode expects CameraAdapterInput")
    _validate_config_bindings(camera_config, episode_config)
    try:
        normalized_boundary = validate_episode_boundary(
            boundary, adapter_input.media_sidecar
        )
    except CameraEpisodeImportError as exc:
        raise CameraEpisodeInferenceError(str(exc)) from exc
    base = _prepared_base(normalized_boundary, adapter_input.media_sidecar)
    if normalized_boundary["boundary_status"] == "boundary_uncertain":
        return _unavailable_preparation(
            base,
            status="boundary_uncertain",
            reason_codes=normalized_boundary["boundary_reason_codes"],
        )
    prepared_interval = prepare_camera_interval(
        adapter_input,
        interval_id=str(normalized_boundary["episode_id"]),
        source_video_id=str(normalized_boundary["source_video_id"]),
        target_track_id=int(normalized_boundary["target_track_id"]),
        start_sec=float(normalized_boundary["start_sec"]),
        end_sec_exclusive=float(normalized_boundary["end_sec_exclusive"]),
        input_kind="accepted_boundary",
        camera_config=camera_config,
        episode_config=episode_config,
        preprocessing_config=preprocessing_config,
        feature_stats=feature_stats,
    )
    return {
        **base,
        **{
            name: prepared_interval[name]
            for name in _INTERVAL_PREPARATION_FIELDS
        },
    }


def prepare_camera_interval(
    adapter_input: CameraAdapterInput,
    *,
    interval_id: str,
    source_video_id: str,
    target_track_id: int,
    start_sec: float,
    end_sec_exclusive: float,
    input_kind: str,
    camera_config: Mapping[str, Any],
    episode_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
    minimum_motion_extent_body_heights: float | None = None,
) -> dict[str, Any]:
    """Prepare one validated interval without assigning boundary acceptance semantics."""

    if not isinstance(adapter_input, CameraAdapterInput):
        raise CameraEpisodeInferenceError("prepare_camera_interval expects CameraAdapterInput")
    _validate_config_bindings(camera_config, episode_config)
    if input_kind not in _INTERVAL_INPUT_QUALITY_FLAGS:
        raise CameraEpisodeInferenceError("camera episode interval input kind is invalid")
    configured_motion_threshold = float(
        episode_config["episode_qc"]["minimum_motion_extent_body_heights"]
    )
    motion_threshold = configured_motion_threshold
    if minimum_motion_extent_body_heights is not None:
        override = float(minimum_motion_extent_body_heights)
        if (
            input_kind != "automatic_proposal"
            or not np.isfinite(override)
            or override <= 0.0
            or override > configured_motion_threshold
        ):
            raise CameraEpisodeInferenceError(
                "automatic proposal motion-evidence override is invalid"
            )
        motion_threshold = override
    if not isinstance(interval_id, str) or not interval_id:
        raise CameraEpisodeInferenceError("camera episode interval ID is invalid")
    if (
        not isinstance(source_video_id, str)
        or source_video_id != adapter_input.media_sidecar["source_video_id"]
    ):
        raise CameraEpisodeInferenceError("camera episode interval source video differs")
    if not isinstance(target_track_id, int) or isinstance(target_track_id, bool):
        raise CameraEpisodeInferenceError("camera episode interval track ID is invalid")
    start = float(start_sec)
    end = float(end_sec_exclusive)
    duration = float(adapter_input.media_sidecar["duration_sec"])
    if (
        not np.isfinite(start + end)
        or start < 0.0
        or start >= end
        or end > duration + 1e-9
    ):
        raise CameraEpisodeInferenceError("camera episode interval endpoints are invalid")

    target_scope = (
        str(adapter_input.media_sidecar["source_group_id"]),
        source_video_id,
        str(adapter_input.media_sidecar["device_id"]),
        str(adapter_input.media_sidecar["setup_id"]),
        str(adapter_input.media_sidecar["stream_epoch"]),
        target_track_id,
    )
    base = _interval_prepared_base(
        interval_id=interval_id,
        target_scope=target_scope,
        start=start,
        end=end,
    )
    observations = group_observations(adapter_input.observations).get(target_scope)
    if observations is None:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=["target_track_not_found"],
        )
    selected = tuple(item for item in observations if start <= item.timestamp_sec < end)
    base["source_observation_count"] = len(selected)
    if not selected:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=["no_target_observations_in_boundary"],
        )
    minimum_track_confidence = float(
        episode_config["sampling"]["minimum_track_confidence"]
    )
    accepted = tuple(
        item for item in selected if item.track_confidence >= minimum_track_confidence
    )
    base["accepted_observation_count"] = len(accepted)
    if not accepted:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=[
                "too_few_raw_detections",
                "too_few_observed_buckets",
                "boundary_tracking_gap",
            ],
        )

    reasons: list[str] = []
    try:
        segments = _split_observations(
            accepted,
            sampling=camera_config["sampling"],
            qc=camera_config["camera_qc"],
        )
    except CameraQCError as exc:
        raise CameraEpisodeInferenceError("technical tracklet splitting failed") from exc
    boundary_flags = sorted({flag for _segment, flags in segments for flag in flags})
    severe_boundary_flags = {
        "long_internal_gap",
        "suspected_id_switch",
        "height_position_discontinuity",
    }.intersection(boundary_flags)
    if len(segments) != 1 or severe_boundary_flags:
        reasons.append("track_break_within_episode")
    reasons.extend(boundary_flags)

    sampling = episode_config["sampling"]
    qc = episode_config["episode_qc"]
    bucket_seconds = float(sampling["bucket_seconds"])
    buckets = weighted_bucket_observations(
        accepted,
        minimum_track_confidence=minimum_track_confidence,
        bucket_seconds=bucket_seconds,
    )
    base["observed_bucket_count"] = len(buckets)
    if buckets:
        by_index = {item.bucket_index: item for item in buckets}
        first_index = min(by_index)
        last_index = max(by_index)
        bucket_values = [by_index.get(index) for index in range(first_index, last_index + 1)]
        observed = np.asarray([item is not None for item in bucket_values], dtype=bool)
        observed_count = int(observed.sum())
        source_bucket_count = len(bucket_values)
        longest_gap = _longest_false_run(observed)
    else:
        bucket_values = []
        observed_count = source_bucket_count = longest_gap = 0
    base.update(
        {
            "source_bucket_count": source_bucket_count,
            "observed_bucket_count": observed_count,
            "interpolated_bucket_count": source_bucket_count - observed_count,
            "observed_coverage_ratio": (
                observed_count / source_bucket_count if source_bucket_count else 0.0
            ),
            "interpolated_coverage_ratio": (
                (source_bucket_count - observed_count) / source_bucket_count
                if source_bucket_count
                else 0.0
            ),
            "maximum_gap_seconds": longest_gap * bucket_seconds,
        }
    )
    if adapter_input.media_sidecar["camera_motion_state"] == "moved":
        reasons.append("camera_moved")
    if len(accepted) < int(qc["minimum_raw_detections"]):
        reasons.append("too_few_raw_detections")
    if observed_count < int(qc["minimum_observed_buckets"]):
        reasons.append("too_few_observed_buckets")
    if base["observed_coverage_ratio"] < float(qc["minimum_observed_ratio"]):
        reasons.append("insufficient_observed_ratio")
    if longest_gap > int(qc["maximum_internal_gap_buckets"]):
        reasons.append("long_internal_gap")
    leading_gap = max(0.0, float(accepted[0].timestamp_sec) - start)
    trailing_gap = max(0.0, end - float(accepted[-1].timestamp_sec))
    if max(leading_gap, trailing_gap) > float(qc["maximum_boundary_gap_seconds"]):
        reasons.append("boundary_tracking_gap")
    if reasons:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=list(dict.fromkeys(reasons)),
        )

    try:
        points, heights, interpolated = _interpolate_window(bucket_values, observed)
        quality = np.where(
            observed,
            float(camera_config["model_input"]["point_quality_observed"]),
            float(camera_config["model_input"]["point_quality_interpolated"]),
        )
        height_result = compensate_bbox_height(
            points,
            heights,
            quality,
            rolling_window=int(
                camera_config["height_compensation"]["rolling_median_window"]
            ),
            gain_clip=camera_config["height_compensation"]["gain_clip"],
        )
        motion_extent = _motion_extent_body(
            height_result.corrected_points,
            height_result.smoothed_heights,
        )
    except CameraQCError as exc:
        raise CameraEpisodeInferenceError("episode bucket interpolation/QC failed") from exc
    base["motion_extent_body_heights"] = motion_extent
    if motion_extent < motion_threshold:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=["insufficient_motion"],
        )

    quality_flags = ["episode_first", _INTERVAL_INPUT_QUALITY_FLAGS[input_kind]]
    if adapter_input.media_sidecar["camera_motion_state"] == "not_checked":
        quality_flags.append("camera_motion_not_verified")
    window_record = {
        "schema_version": "wandering-camera-window-v1",
        "window_id": f"episode-input-{interval_id}",
        "parent_tracklet_id": (
            f"episode-track-{source_video_id}-"
            f"{target_track_id}"
        ),
        "source_group_id": target_scope[0],
        "source_video_id": target_scope[1],
        "device_id": target_scope[2],
        "setup_id": target_scope[3],
        "stream_epoch": target_scope[4],
        "track_id": target_scope[5],
        "window_start_sec": start,
        "window_end_sec": end,
        "window_status": "ready",
        "reason_codes": [],
        "quality_flags": quality_flags,
        "preprocessing_config_sha256": feature_stats["binding_hashes"][
            "preprocessing_config"
        ],
        "model_training_split_sha256": feature_stats["binding_hashes"][
            "split_sha256"
        ],
    }
    try:
        prepared_window = prepare_camera_window(
            CameraWindowInput(
                window_record=window_record,
                corrected_points=height_result.corrected_points,
                point_quality=height_result.point_quality,
            ),
            camera_config,
            preprocessing_config,
            feature_stats,
        )
    except CameraInferenceError as exc:
        return _unavailable_preparation(
            base,
            status="unavailable",
            reason_codes=["episode_preprocessing_unavailable"],
        )
    return {
        **base,
        "qc_status": "ready",
        "qc_reason_codes": [],
        "quality_flags": quality_flags,
        "model_features": prepared_window["model_features"],
        "shape_normalized_points": prepared_window["shape_normalized_points"],
        "point_mask": prepared_window["point_mask"],
        "raw_features": prepared_window["raw_features"],
        "topology": prepared_window["topology"],
    }


def predict_camera_episode(
    prepared_episode: Mapping[str, Any],
    runtime: Any,
    *,
    validation_scope: str,
    evidence_scope: str,
) -> dict[str, Any]:
    """Map one truth-free prepared episode through the fixed TopoWander forward."""

    if prepared_episode.get("schema_version") != CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION:
        raise CameraEpisodeInferenceError("prepared episode schema is invalid")
    qc_status = prepared_episode.get("qc_status")
    if qc_status not in {"ready", "unavailable", "boundary_uncertain"}:
        raise CameraEpisodeInferenceError("prepared episode QC status is invalid")
    if qc_status != "ready":
        return _prediction_output(
            prepared_episode,
            runtime,
            prediction_status=str(qc_status),
            prediction_reason_codes=list(prepared_episode["qc_reason_codes"]),
            prediction=None,
        )
    prediction = predict_camera_interval(
        {
            **prepared_episode,
            "interval_id": prepared_episode["episode_id"],
        },
        runtime,
        validation_scope=validation_scope,
        evidence_scope=evidence_scope,
    )
    return _prediction_output(
        prepared_episode,
        runtime,
        prediction_status=str(prediction["window_status"]),
        prediction_reason_codes=list(prediction["reason_codes"]),
        prediction=prediction,
    )


def predict_camera_interval(
    prepared_interval: Mapping[str, Any],
    runtime: Any,
    *,
    validation_scope: str,
    evidence_scope: str,
) -> dict[str, Any]:
    """Run the existing trusted frozen forward for one ready prepared interval."""

    if prepared_interval.get("qc_status") != "ready":
        raise CameraEpisodeInferenceError("only ready camera intervals may invoke the model")
    interval_id = prepared_interval.get("interval_id")
    if not isinstance(interval_id, str) or not interval_id:
        raise CameraEpisodeInferenceError("prepared camera interval ID is invalid")
    window_record = {
        "window_id": f"episode-input-{interval_id}",
        "window_status": "ready",
        "parent_tracklet_id": (
            f"episode-track-{prepared_interval['source_video_id']}-"
            f"{prepared_interval['track_id']}"
        ),
        "source_group_id": prepared_interval["source_group_id"],
        "source_video_id": prepared_interval["source_video_id"],
        "device_id": prepared_interval["device_id"],
        "setup_id": prepared_interval["setup_id"],
        "stream_epoch": prepared_interval["stream_epoch"],
        "track_id": prepared_interval["track_id"],
        "window_start_sec": prepared_interval["start_sec"],
        "window_end_sec": prepared_interval["end_sec_exclusive"],
        "reason_codes": [],
        "quality_flags": prepared_interval["quality_flags"],
        "model_features": prepared_interval["model_features"],
        "shape_normalized_points": prepared_interval["shape_normalized_points"],
        "point_mask": prepared_interval["point_mask"],
    }
    try:
        prediction, _latency_ms = _predict_primary_camera_window_with_provenance(
            window_record,
            runtime,
            validation_scope=validation_scope,
            evidence_scope=evidence_scope,
        )
    except PrimaryCameraInferenceError as exc:
        raise CameraEpisodeInferenceError("fixed episode model forward failed closed") from exc
    return prediction


def build_camera_episode_inference_bundle(
    *,
    project_root: str | Path,
    episode_config_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    candidate_manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
    episode_boundaries_path: str | Path | None = None,
    whole_clip_episode_id: str | None = None,
    target_track_id: int | None = None,
) -> CameraEpisodeBuildResult:
    """Build a fresh EP1A prediction bundle from boundaries or whole-clip mode."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera episode output already exists: {output}")
    boundary_file_mode = episode_boundaries_path is not None
    whole_clip_mode = whole_clip_episode_id is not None or target_track_id is not None
    if boundary_file_mode == whole_clip_mode:
        raise CameraEpisodeInferenceError(
            "choose exactly one of boundary JSONL or whole-clip episode mode"
        )
    if whole_clip_mode and (
        whole_clip_episode_id is None or target_track_id is None
    ):
        raise CameraEpisodeInferenceError(
            "whole-clip mode requires episode_id and target_track_id"
        )
    root = Path(project_root).resolve(strict=True)
    fixed_config = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(episode_config_path).resolve(strict=False) != fixed_config:
        raise CameraEpisodeInferenceError("episode config must use the fixed repository path")
    episode_config = load_camera_episode_config(fixed_config)
    camera_config_path = _verify_descriptor(root, episode_config["camera_chain"])
    primary_config_path = _verify_descriptor(root, episode_config["primary_inference"])
    camera_config = load_camera_config(camera_config_path)
    primary_config = load_primary_camera_config(primary_config_path)
    _validate_config_bindings(camera_config, episode_config)
    if (
        primary_config["candidate"]["manifest_sha256"]
        != episode_config["candidate"]["manifest_sha256"]
        or primary_config["candidate"]["model_state_sha256"]
        != episode_config["candidate"]["model_state_sha256"]
    ):
        raise CameraEpisodeInferenceError("episode and primary candidate identity differ")
    expected_manifest_path = (root / primary_config["candidate"]["manifest_path"]).resolve(
        strict=False
    )
    manifest_path = Path(candidate_manifest_path).resolve(strict=False)
    if manifest_path != expected_manifest_path:
        raise CameraEpisodeInferenceError("candidate manifest must use the fixed path")
    adapter = load_camera_inputs(
        tracking_jsonl_path,
        media_sidecar_path,
        camera_config,
    )
    if boundary_file_mode:
        boundaries = load_episode_boundaries(
            episode_boundaries_path, adapter.media_sidecar
        )
    else:
        boundaries = [
            build_whole_clip_boundary(
                adapter.media_sidecar,
                episode_id=str(whole_clip_episode_id),
                target_track_id=int(target_track_id),
            )
        ]
    preprocessing_paths = _verify_preprocessing_roots(root, camera_config)
    preprocessing_config = load_preprocessing_config(
        preprocessing_paths["preprocessing_config"]
    )
    feature_stats = _load_feature_stats(preprocessing_paths, camera_config)
    runtime = load_primary_camera_runtime(
        config=primary_config,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    validation_scope = validation_scope_for_authorization_status(
        str(adapter.media_sidecar["authorization_status"])
    )
    evidence_scope = (
        EVIDENCE_SCOPE
        if validation_scope == "synthetic_camera_contract"
        else "authorized_development_smoke"
    )
    runtime_observation = _configure_cpu_runtime(primary_config["runtime"])
    predictions: list[dict[str, Any]] = []
    try:
        for boundary in boundaries:
            prepared = prepare_camera_episode(
                adapter,
                boundary,
                camera_config=camera_config,
                episode_config=episode_config,
                preprocessing_config=preprocessing_config,
                feature_stats=feature_stats,
            )
            predictions.append(
                predict_camera_episode(
                    prepared,
                    runtime,
                    validation_scope=validation_scope,
                    evidence_scope=evidence_scope,
                )
            )
    finally:
        torch.set_num_threads(int(runtime_observation["previous_intra_op"]))
    status_counts = Counter(str(row["prediction_status"]) for row in predictions)
    summary = {
        "schema_version": CAMERA_EPISODE_RUN_SUMMARY_SCHEMA_VERSION,
        "status": "wandering_m0cam_ep1a_oracle_boundary_ready",
        "evaluation_name": ORACLE_BOUNDARY_EVALUATION_NAME,
        "source_video_id": adapter.media_sidecar["source_video_id"],
        "episode_count": len(predictions),
        "prediction_status_counts": {
            name: status_counts.get(name, 0)
            for name in (
                "ready",
                "unavailable",
                "boundary_uncertain",
                "inference_error",
            )
        },
        "predicted_pattern_counts": dict(
            sorted(
                Counter(
                    str(row["predicted_pattern"])
                    for row in predictions
                    if row["predicted_pattern"] is not None
                ).items()
            )
        ),
        "candidate_id": runtime.manifest["candidate_id"],
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "model_state_sha256": runtime.manifest["artifacts"]["model_state"]["sha256"],
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "models_retrained": False,
        "truth_labels_consumed_by_inference": False,
        "automatic_boundary_inference": False,
        "legacy_40_second_diagnostic_included": False,
        "validation_scope": validation_scope,
        "evidence_scope": evidence_scope,
    }
    _commit_new_directory(
        output,
        {
            "episode_predictions.jsonl": canonical_jsonl_bytes(predictions),
            "summary.json": canonical_json_bytes(summary),
        },
    )
    return CameraEpisodeBuildResult(
        output_dir=output,
        episode_count=len(predictions),
        ready_count=status_counts.get("ready", 0),
        unavailable_count=status_counts.get("unavailable", 0),
        boundary_uncertain_count=status_counts.get("boundary_uncertain", 0),
        inference_error_count=status_counts.get("inference_error", 0),
    )


def _interval_prepared_base(
    *,
    interval_id: str,
    target_scope: tuple[str, str, str, str, str, int],
    start: float,
    end: float,
) -> dict[str, Any]:
    return {
        "interval_id": interval_id,
        "source_group_id": target_scope[0],
        "source_video_id": target_scope[1],
        "device_id": target_scope[2],
        "setup_id": target_scope[3],
        "stream_epoch": target_scope[4],
        "track_id": target_scope[5],
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "source_observation_count": 0,
        "accepted_observation_count": 0,
        "source_bucket_count": 0,
        "observed_bucket_count": 0,
        "interpolated_bucket_count": 0,
        "observed_coverage_ratio": 0.0,
        "interpolated_coverage_ratio": 0.0,
        "maximum_gap_seconds": 0.0,
        "motion_extent_body_heights": None,
        "qc_status": None,
        "qc_reason_codes": [],
        "quality_flags": [],
        "model_features": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "topology": None,
    }


def _prepared_base(
    boundary: Mapping[str, Any], media: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION,
        "episode_id": boundary["episode_id"],
        "source_group_id": media["source_group_id"],
        "source_video_id": boundary["source_video_id"],
        "device_id": media["device_id"],
        "setup_id": media["setup_id"],
        "stream_epoch": media["stream_epoch"],
        "track_id": boundary["target_track_id"],
        "cvat_track_id": boundary["cvat_track_id"],
        "start_sec": boundary["start_sec"],
        "end_sec_exclusive": boundary["end_sec_exclusive"],
        "duration_sec": boundary["end_sec_exclusive"] - boundary["start_sec"],
        "boundary_source": boundary["boundary_source"],
        "boundary_status": boundary["boundary_status"],
        "boundary_reason_codes": list(boundary["boundary_reason_codes"]),
        "source_observation_count": 0,
        "accepted_observation_count": 0,
        "source_bucket_count": 0,
        "observed_bucket_count": 0,
        "interpolated_bucket_count": 0,
        "observed_coverage_ratio": 0.0,
        "interpolated_coverage_ratio": 0.0,
        "maximum_gap_seconds": 0.0,
        "motion_extent_body_heights": None,
        "qc_status": None,
        "qc_reason_codes": [],
        "quality_flags": [],
        "model_features": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "topology": None,
    }


def _unavailable_preparation(
    base: Mapping[str, Any],
    *,
    status: str,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        **dict(base),
        "qc_status": status,
        "qc_reason_codes": list(dict.fromkeys(reason_codes)),
        "model_features": None,
        "shape_normalized_points": None,
        "point_mask": None,
        "raw_features": None,
        "topology": None,
    }


def _prediction_output(
    prepared: Mapping[str, Any],
    runtime: Any,
    *,
    prediction_status: str,
    prediction_reason_codes: Sequence[str],
    prediction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    binary = prediction.get("binary") if prediction is not None else None
    subtype = prediction.get("subtype") if prediction is not None else None
    four_class = prediction.get("four_class") if prediction is not None else None
    return {
        "schema_version": CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION,
        **{
            name: prepared[name]
            for name in (
                "episode_id",
                "source_group_id",
                "source_video_id",
                "device_id",
                "setup_id",
                "stream_epoch",
                "track_id",
                "cvat_track_id",
                "start_sec",
                "end_sec_exclusive",
                "duration_sec",
                "boundary_source",
                "boundary_status",
                "boundary_reason_codes",
                "source_observation_count",
                "accepted_observation_count",
                "source_bucket_count",
                "observed_bucket_count",
                "interpolated_bucket_count",
                "observed_coverage_ratio",
                "interpolated_coverage_ratio",
                "maximum_gap_seconds",
                "motion_extent_body_heights",
                "qc_status",
                "qc_reason_codes",
                "quality_flags",
            )
        },
        "prediction_status": prediction_status,
        "prediction_reason_codes": list(prediction_reason_codes),
        "binary": binary,
        "subtype": subtype,
        "four_class": four_class,
        "predicted_pattern": (
            four_class["predicted_label"] if four_class is not None else None
        ),
        "candidate_id": runtime.manifest["candidate_id"],
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "model_state_sha256": runtime.manifest["artifacts"]["model_state"]["sha256"],
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "model_invocation_skipped": prediction_status in {
            "unavailable",
            "boundary_uncertain",
        },
    }


def _validate_config_bindings(
    camera_config: Mapping[str, Any], episode_config: Mapping[str, Any]
) -> None:
    if episode_config != _expected_episode_config():
        raise CameraEpisodeInferenceError("episode config is not the fixed EP1A config")
    if episode_config["sampling"] != {
        "bucket_seconds": camera_config["sampling"]["bucket_seconds"],
        "minimum_track_confidence": camera_config["sampling"][
            "minimum_track_confidence"
        ],
    }:
        raise CameraEpisodeInferenceError("episode sampling drifted from camera v1")
    qc = episode_config["episode_qc"]
    for name in (
        "minimum_observed_ratio",
        "maximum_internal_gap_seconds",
        "maximum_internal_gap_buckets",
        "minimum_raw_detections",
        "minimum_motion_extent_body_heights",
    ):
        if qc[name] != camera_config["camera_qc"][name]:
            raise CameraEpisodeInferenceError("episode QC drifted from camera v1")
    if episode_config["model_input"] != {
        "target_points": camera_config["model_input"]["target_points"],
        "input_channels": camera_config["model_input"]["input_channels"],
        "temporal_features_enabled": camera_config["model_input"][
            "temporal_features_enabled"
        ],
    }:
        raise CameraEpisodeInferenceError("episode model input drifted from camera v1")


def _verify_descriptor(root: Path, descriptor: Mapping[str, Any]) -> Path:
    relative = Path(str(descriptor["path"]))
    path = (root / relative).resolve(strict=False)
    if not path.is_file():
        raise CameraEpisodeInferenceError(f"episode dependency is missing: {relative}")
    if (
        path.stat().st_size != descriptor["size_bytes"]
        or _sha256_file(path) != descriptor["sha256"]
    ):
        raise CameraEpisodeInferenceError(f"episode dependency drifted: {relative}")
    return path


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera episode output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
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


def _expected_episode_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_CONFIG_SCHEMA_VERSION,
        "purpose": "topowander_oracle_boundary_shape_classification",
        "camera_chain": {
            "path": "configs/modules/wandering_camera_v1.yaml",
            "sha256": "08bbec6ef263dd45fef3262c407ce675107f87ea584584f7ab61142cd734ae8e",
            "size_bytes": 2500,
        },
        "primary_inference": {
            "path": "configs/modules/wandering_camera_primary_v1.yaml",
            "sha256": "79f8e06e57ca355da926044828fc7a8a4c7d0e123d59767a2c86ef6c620ca2f5",
            "size_bytes": 1767,
        },
        "candidate": {
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        },
        "sampling": {"bucket_seconds": 0.5, "minimum_track_confidence": 0.25},
        "episode_qc": {
            "minimum_observed_ratio": 0.75,
            "maximum_internal_gap_seconds": 1.5,
            "maximum_internal_gap_buckets": 3,
            "maximum_boundary_gap_seconds": 1.5,
            "minimum_raw_detections": 8,
            "minimum_observed_buckets": 2,
            "minimum_motion_extent_body_heights": 0.25,
        },
        "model_input": {
            "target_points": 80,
            "input_channels": 14,
            "temporal_features_enabled": False,
        },
        "inference": {
            "probability_calibrated": False,
            "binary_decision_threshold": 0.5,
        },
        "class_order": {
            "binary": ["direct_or_non_wandering", "wandering_like"],
            "subtype": ["pacing", "lapping", "random"],
            "four_class": list(FOUR_CLASS_ORDER),
        },
        "boundary_sources": ["simplified_jsonl", "cvat_xml", "whole_clip"],
        "output_schemas": {
            "boundary": CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
            "truth": "wandering-camera-episode-truth-v1",
            "prediction": CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION,
            "summary": CAMERA_EPISODE_RUN_SUMMARY_SCHEMA_VERSION,
        },
        "data_access": {
            "synthetic_fixture": True,
            "authorized_camera_development": True,
            "sealed_camera": False,
            "wp_raw": False,
            "smartcare_official_or_raw": False,
        },
    }


__all__ = [
    "CAMERA_EPISODE_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_PREDICTION_SCHEMA_VERSION",
    "CAMERA_EPISODE_RUN_SUMMARY_SCHEMA_VERSION",
    "CameraEpisodeBuildResult",
    "CameraEpisodeInferenceError",
    "ORACLE_BOUNDARY_EVALUATION_NAME",
    "build_camera_episode_inference_bundle",
    "build_whole_clip_boundary",
    "load_camera_episode_config",
    "predict_camera_interval",
    "predict_camera_episode",
    "prepare_camera_interval",
    "prepare_camera_episode",
]
