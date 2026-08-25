"""Minimal S0 automatic-proposal to frozen-shape bridge."""

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

import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    CameraAdapterInput,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_camera_inputs,
    validation_scope_for_authorization_status,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION,
    HOME_RECALL_PROFILE_ID,
    PRODUCER_CONFIG_ID,
    PROPOSAL_STATUSES,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_inference import (
    CameraEpisodeInferenceError,
    _verify_descriptor,
    load_camera_episode_config,
    predict_camera_interval,
    prepare_camera_interval,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_geometry_head import (
    CameraGeometryHeadError,
    load_camera_geometry_head,
    predict_home_camera_geometry,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EVIDENCE_SCOPE,
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    PrimaryCameraInferenceError,
    _configure_cpu_runtime,
    _load_feature_stats,
    _verify_preprocessing_roots,
    load_primary_camera_config,
    load_primary_camera_runtime,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)


CAMERA_EPISODE_PROPOSAL_SHAPE_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-proposal-shape-config-v1"
)
CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION = (
    "wandering-camera-episode-proposal-shape-prediction-v1"
)
CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-proposal-shape-summary-v1"
)
SHAPE_INFERENCE_POLICY_ID = (
    "m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2"
)
FORWARD_PROPOSAL_STATUSES = ("proposed", "uncertain")
SKIP_PROPOSAL_STATUSES = ("rejected_by_qc",)
_CONFIG_RELATIVE = Path(
    "configs/modules/wandering_camera_episode_proposal_shape_v1.yaml"
)
_BATCH_FIELDS = frozenset(
    {
        "bundle_id",
        "proposal_bundle_dir",
        "tracking_jsonl",
        "media_sidecar",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
    }
)
_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
)
_PROPOSAL_PRESERVED_FIELDS = (
    "proposal_id",
    *_SCOPE_FIELDS,
    "technical_segment_index",
    "start_sec",
    "end_sec_exclusive",
    "duration_sec",
    "proposal_status",
    "start_reason",
    "end_reason",
    "reason_codes",
    "hard_break_reasons",
    "uncertain",
    "manual_review_required",
    "producer_config_id",
)
_QC_RESULT_FIELDS = (
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
)


class CameraEpisodeProposalInferenceError(ValueError):
    """Proposal identity, interval preparation, or output construction failed closed."""


@dataclass(frozen=True)
class CameraEpisodeProposalBuildResult:
    output_dir: Path
    proposal_count: int
    ready_count: int
    unavailable_count: int
    boundary_uncertain_count: int
    inference_error_count: int
    model_forward_invocation_count: int


@dataclass(frozen=True)
class _BundleInput:
    binding: Mapping[str, str]
    adapter: CameraAdapterInput
    proposals: tuple[dict[str, Any], ...]
    minimum_track_confidence: float
    movement_evidence_threshold_body_heights: float | None
    producer_config_id: str


def load_camera_episode_proposal_shape_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load the fixed thin S2A bridge configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeProposalInferenceError(
            "cannot read camera episode proposal-shape config"
        ) from exc
    if value != _expected_config():
        raise CameraEpisodeProposalInferenceError(
            "camera episode proposal-shape config fields or values drifted"
        )
    return value


def build_camera_episode_proposal_inference_bundle(
    *,
    project_root: str | Path,
    config_path: str | Path,
    batch_index_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeProposalBuildResult:
    """Build one independent shape-result row for every original S0 proposal."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera proposal-shape output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    fixed_config_path = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(config_path).resolve(strict=False) != fixed_config_path:
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape config must use the fixed repository path"
        )
    config = load_camera_episode_proposal_shape_config(fixed_config_path)
    try:
        episode_path = (root / config["episode_inference_path"]).resolve(strict=True)
        primary_path = (root / config["primary_inference_path"]).resolve(strict=True)
        manifest_path = (root / config["candidate_manifest_path"]).resolve(strict=True)
        episode_config = load_camera_episode_config(episode_path)
        if _verify_descriptor(root, episode_config["primary_inference"]) != primary_path:
            raise CameraEpisodeProposalInferenceError(
                "proposal-shape primary config binding differs from EP1A"
            )
        camera_path = _verify_descriptor(root, episode_config["camera_chain"])
        camera_config = load_camera_config(camera_path)
        primary_config = load_primary_camera_config(primary_path)
        preprocessing_paths = _verify_preprocessing_roots(root, camera_config)
        preprocessing_config = load_preprocessing_config(
            preprocessing_paths["preprocessing_config"]
        )
        feature_stats = _load_feature_stats(preprocessing_paths, camera_config)
    except (
        OSError,
        CameraEpisodeInferenceError,
        CameraInferenceError,
        PrimaryCameraInferenceError,
        ValueError,
    ) as exc:
        if isinstance(exc, CameraEpisodeProposalInferenceError):
            raise
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape frozen preprocessing bindings failed"
        ) from exc
    if (
        config["candidate_manifest_sha256"]
        != primary_config["candidate"]["manifest_sha256"]
        or config["candidate_manifest_path"]
        != primary_config["candidate"]["manifest_path"]
        or episode_config["candidate"]["manifest_sha256"]
        != primary_config["candidate"]["manifest_sha256"]
        or episode_config["candidate"]["model_state_sha256"]
        != primary_config["candidate"]["model_state_sha256"]
    ):
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape frozen candidate identity differs from EP1A"
        )

    # Validate every bundle and proposal before the first possible model forward.
    bundles = _load_bundle_inputs(
        Path(batch_index_path),
        camera_config=camera_config,
    )
    geometry_head: Mapping[str, Any] | None = None
    geometry_head_path = (
        root
        / "reports/mental_health/wandering_camera_home_geometry_head_v1/model.json"
    )
    if any(bundle.producer_config_id == HOME_RECALL_PROFILE_ID for bundle in bundles):
        try:
            geometry_head = load_camera_geometry_head(geometry_head_path)
        except CameraGeometryHeadError as exc:
            raise CameraEpisodeProposalInferenceError(
                "home camera geometry head binding failed"
            ) from exc
    runtime: Any | None = None
    runtime_observation: Mapping[str, Any] | None = None
    results: list[dict[str, Any]] = []
    try:
        for bundle in bundles:
            validation_scope = validation_scope_for_authorization_status(
                str(bundle.adapter.media_sidecar["authorization_status"])
            )
            evidence_scope = (
                EVIDENCE_SCOPE
                if validation_scope == "synthetic_camera_contract"
                else "authorized_development_smoke"
            )
            for proposal in bundle.proposals:
                prepared: Mapping[str, Any] | None = None
                prediction: Mapping[str, Any] | None = None
                geometry_prediction: Mapping[str, Any] | None = None
                status = str(proposal["proposal_status"])
                if status in config["forward_proposal_statuses"]:
                    prepared = prepare_camera_interval(
                        bundle.adapter,
                        interval_id=str(proposal["proposal_id"]),
                        source_video_id=str(proposal["source_video_id"]),
                        target_track_id=int(proposal["track_id"]),
                        start_sec=float(proposal["start_sec"]),
                        end_sec_exclusive=float(proposal["end_sec_exclusive"]),
                        input_kind="automatic_proposal",
                        camera_config=camera_config,
                        episode_config=episode_config,
                        preprocessing_config=preprocessing_config,
                        feature_stats=feature_stats,
                        minimum_motion_extent_body_heights=(
                            bundle.movement_evidence_threshold_body_heights
                        ),
                    )
                    if prepared["qc_status"] == "ready":
                        if runtime is None:
                            runtime = load_primary_camera_runtime(
                                config=primary_config,
                                manifest_path=manifest_path,
                                expected_manifest_sha256=str(
                                    config["candidate_manifest_sha256"]
                                ),
                            )
                            runtime_observation = _configure_cpu_runtime(
                                primary_config["runtime"]
                            )
                        prediction = predict_camera_interval(
                            prepared,
                            runtime,
                            validation_scope=validation_scope,
                            evidence_scope=evidence_scope,
                        )
                        if bundle.producer_config_id == HOME_RECALL_PROFILE_ID:
                            if geometry_head is None:
                                raise CameraEpisodeProposalInferenceError(
                                    "home proposal has no geometry head"
                                )
                            geometry_prediction = predict_home_camera_geometry(
                                bundle.adapter,
                                track_id=int(proposal["track_id"]),
                                start_sec=float(proposal["start_sec"]),
                                end_sec=float(proposal["end_sec_exclusive"]),
                                artifact=geometry_head,
                            )
                results.append(
                    _proposal_shape_result(
                        bundle=bundle,
                        proposal=proposal,
                        prepared=prepared,
                        prediction=prediction,
                        geometry_prediction=geometry_prediction,
                        primary_config=primary_config,
                    )
                )
    except (
        CameraEpisodeInferenceError,
        CameraGeometryHeadError,
        PrimaryCameraInferenceError,
        CameraInferenceError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape interval preparation or frozen forward failed"
        ) from exc
    finally:
        if runtime_observation is not None:
            torch.set_num_threads(int(runtime_observation["previous_intra_op"]))

    proposal_count = sum(len(bundle.proposals) for bundle in bundles)
    if len(results) != proposal_count:
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape result count differs from proposal count"
        )
    status_counts = Counter(str(row["prediction_status"]) for row in results)
    forward_count = sum(not bool(row["model_invocation_skipped"]) for row in results)
    forward_counts_by_proposal_status = {
        name: sum(
            row["proposal_status"] == name
            and not bool(row["model_invocation_skipped"])
            for row in results
        )
        for name in PROPOSAL_STATUSES
    }
    ready_counts_by_proposal_status = {
        name: sum(
            row["proposal_status"] == name and row["prediction_status"] == "ready"
            for row in results
        )
        for name in PROPOSAL_STATUSES
    }
    resolved_results = _resolved_episode_rows(results)
    summary = {
        "schema_version": CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION,
        "status": "wandering_m0cam_ep2a_s2a_proposal_shape_generated",
        "input_boundary_kind": "automatic_proposal",
        "proposal_schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "proposal_count": proposal_count,
        "result_count": len(results),
        "resolved_episode_count": len(resolved_results),
        "resolved_micro_bout_count": sum(
            row["resolution_level"] == "motion_bout"
            for row in resolved_results
        ),
        "prediction_status_counts": {
            name: status_counts.get(name, 0)
            for name in (
                "ready",
                "unavailable",
                "boundary_uncertain",
                "inference_error",
            )
        },
        "model_forward_invocation_count": forward_count,
        "model_invocation_skipped_count": len(results) - forward_count,
        "model_forward_invocation_count_by_proposal_status": (
            forward_counts_by_proposal_status
        ),
        "prediction_ready_count_by_proposal_status": ready_counts_by_proposal_status,
        "shape_inference_policy_id": config["shape_inference_policy_id"],
        "forward_proposal_statuses": list(config["forward_proposal_statuses"]),
        "skip_proposal_statuses": list(config["skip_proposal_statuses"]),
        "uncertain_shape_output_role": config["uncertain_shape_output_role"],
        "candidate_id": primary_config["candidate"]["candidate_id"],
        "candidate_manifest_sha256": primary_config["candidate"]["manifest_sha256"],
        "model_state_sha256": primary_config["candidate"]["model_state_sha256"],
        "binary_decision_threshold": 0.5,
        "proposal_endpoint_modified": False,
        "accepted_boundary_emitted": False,
        "uncertain_boundary_status_preserved": True,
        "boundary_and_prediction_status_separated": True,
        "trusted_minimum_track_confidences": sorted(
            {bundle.minimum_track_confidence for bundle in bundles}
        ),
        "movement_evidence_thresholds_body_heights": sorted(
            {
                bundle.movement_evidence_threshold_body_heights
                for bundle in bundles
                if bundle.movement_evidence_threshold_body_heights is not None
            }
        ),
        "truth_labels_consumed": False,
        "models_retrained": False,
        "shape_performance_metrics_available": False,
        "home_geometry_head_applied_count": sum(
            row["classification_source"] == "home_camera_geometry_head"
            for row in results
        ),
        "home_geometry_head_model_id": (
            geometry_head["model_id"] if geometry_head is not None else None
        ),
        "home_geometry_head_sha256": (
            hashlib.sha256(geometry_head_path.read_bytes()).hexdigest()
            if geometry_head is not None
            else None
        ),
    }
    _commit_new_directory(
        output,
        {
            "README.md": _bundle_readme(summary).encode("utf-8"),
            "proposal_shape_predictions.jsonl": canonical_jsonl_bytes(results),
            "resolved_episode_predictions.jsonl": canonical_jsonl_bytes(
                resolved_results
            ),
            "summary.json": canonical_json_bytes(summary),
        },
    )
    return CameraEpisodeProposalBuildResult(
        output_dir=output,
        proposal_count=proposal_count,
        ready_count=status_counts.get("ready", 0),
        unavailable_count=status_counts.get("unavailable", 0),
        boundary_uncertain_count=status_counts.get("boundary_uncertain", 0),
        inference_error_count=status_counts.get("inference_error", 0),
        model_forward_invocation_count=forward_count,
    )


def _load_bundle_inputs(
    batch_index_path: Path,
    *,
    camera_config: Mapping[str, Any],
) -> tuple[_BundleInput, ...]:
    try:
        index_path = batch_index_path.resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape batch index is inaccessible"
        ) from exc
    index_rows = _load_jsonl(index_path, "proposal-shape batch index")
    if not index_rows:
        raise CameraEpisodeProposalInferenceError("proposal-shape batch index is empty")
    bundle_ids: set[str] = set()
    proposal_ids: set[str] = set()
    physical_sources: set[tuple[str, str, str, str, str]] = set()
    bundles: list[_BundleInput] = []
    for raw_binding in index_rows:
        if set(raw_binding) != _BATCH_FIELDS or not all(
            isinstance(raw_binding[name], str) and raw_binding[name]
            for name in _BATCH_FIELDS
        ):
            raise CameraEpisodeProposalInferenceError(
                "proposal-shape batch index fields are invalid"
            )
        binding = {name: str(raw_binding[name]) for name in _BATCH_FIELDS}
        bundle_id = binding["bundle_id"]
        if bundle_id in bundle_ids:
            raise CameraEpisodeProposalInferenceError("duplicate proposal bundle ID")
        bundle_ids.add(bundle_id)
        bundle_dir = _resolve_index_path(
            index_path, binding["proposal_bundle_dir"], directory=True
        )
        tracking = _resolve_index_path(index_path, binding["tracking_jsonl"])
        sidecar = _resolve_index_path(index_path, binding["media_sidecar"])
        try:
            adapter = load_camera_inputs(tracking, sidecar, camera_config)
        except (CameraAdapterError, OSError, ValueError) as exc:
            raise CameraEpisodeProposalInferenceError(
                "proposal-shape tracking or sidecar is invalid"
            ) from exc
        if binding["camera_setup_id"] != adapter.media_sidecar["setup_id"]:
            raise CameraEpisodeProposalInferenceError(
                "proposal-shape camera setup binding differs from sidecar"
            )
        source_key = tuple(
            str(adapter.media_sidecar[name])
            for name in (
                "source_group_id",
                "source_video_id",
                "device_id",
                "setup_id",
                "stream_epoch",
            )
        )
        if source_key in physical_sources:
            raise CameraEpisodeProposalInferenceError(
                "physical camera source is bound more than once"
            )
        physical_sources.add(source_key)
        summary = _load_json(bundle_dir / "summary.json", "S0 proposal summary")
        proposals = _load_jsonl(bundle_dir / "proposals.jsonl", "S0 proposals")
        if not (bundle_dir / "diagnostics.jsonl").is_file():
            raise CameraEpisodeProposalInferenceError(
                "S0 proposal diagnostics are missing"
            )
        (
            producer_config_id,
            minimum_track_confidence,
            movement_evidence_threshold,
        ) = _validate_proposal_summary(summary, adapter, proposals)
        for proposal in proposals:
            _validate_proposal(proposal, adapter, producer_config_id=producer_config_id)
            proposal_id = str(proposal["proposal_id"])
            if proposal_id in proposal_ids:
                raise CameraEpisodeProposalInferenceError("duplicate proposal ID")
            proposal_ids.add(proposal_id)
        bundles.append(
            _BundleInput(
                binding=binding,
                adapter=_filter_adapter_by_confidence(
                    adapter,
                    minimum_track_confidence=minimum_track_confidence,
                ),
                proposals=tuple(proposals),
                minimum_track_confidence=minimum_track_confidence,
                movement_evidence_threshold_body_heights=(
                    movement_evidence_threshold
                ),
                producer_config_id=producer_config_id,
            )
        )
    return tuple(bundles)


def _validate_proposal_summary(
    summary: Mapping[str, Any],
    adapter: CameraAdapterInput,
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[str, float, float | None]:
    if (
        summary.get("schema_version")
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION
        or summary.get("proposal_schema_version")
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or summary.get("proposal_is_accepted_boundary") is not False
        or summary.get("proposal_count") != len(proposals)
        or not isinstance(summary.get("producer_config_id"), str)
        or not summary.get("producer_config_id")
    ):
        raise CameraEpisodeProposalInferenceError("S0 proposal summary is invalid")
    for name in ("source_group_id", "source_video_id", "device_id", "setup_id", "stream_epoch"):
        if summary.get(name) != adapter.media_sidecar[name]:
            raise CameraEpisodeProposalInferenceError(
                "S0 proposal summary scope differs from sidecar"
            )
    producer_config_id = str(summary["producer_config_id"])
    minimum_track_confidence = summary.get("minimum_track_confidence", 0.25)
    if (
        isinstance(minimum_track_confidence, bool)
        or not isinstance(minimum_track_confidence, (int, float))
        or not math.isfinite(float(minimum_track_confidence))
        or not 0.25 <= float(minimum_track_confidence) <= 0.99
    ):
        raise CameraEpisodeProposalInferenceError(
            "S0 proposal trusted confidence is invalid"
        )
    movement_threshold = summary.get(
        "movement_evidence_threshold_body_heights"
    )
    if movement_threshold is not None and (
        isinstance(movement_threshold, bool)
        or not isinstance(movement_threshold, (int, float))
        or not math.isfinite(float(movement_threshold))
        or not 0.001 <= float(movement_threshold) <= 0.25
    ):
        raise CameraEpisodeProposalInferenceError(
            "S0 proposal movement evidence threshold is invalid"
        )
    if producer_config_id == HOME_RECALL_PROFILE_ID and (
        not math.isclose(
            float(minimum_track_confidence), 0.70, rel_tol=0.0, abs_tol=1e-12
        )
        or movement_threshold is None
    ):
        raise CameraEpisodeProposalInferenceError(
            "home recall proposal trust contract is incomplete"
        )
    return (
        producer_config_id,
        float(minimum_track_confidence),
        float(movement_threshold) if movement_threshold is not None else None,
    )


def _filter_adapter_by_confidence(
    adapter: CameraAdapterInput,
    *,
    minimum_track_confidence: float,
) -> CameraAdapterInput:
    observations = tuple(
        item
        for item in adapter.observations
        if item.track_confidence >= minimum_track_confidence
    )
    normalized_rows = tuple(
        row
        for row in adapter.normalized_rows
        if float(row["track_confidence"]) >= minimum_track_confidence
    )
    normalized_sha256 = hashlib.sha256(
        canonical_jsonl_bytes(normalized_rows)
    ).hexdigest()
    return CameraAdapterInput(
        media_sidecar=adapter.media_sidecar,
        observations=observations,
        normalized_rows=normalized_rows,
        source_tracking_sha256=adapter.source_tracking_sha256,
        normalized_tracking_sha256=normalized_sha256,
    )


def _validate_proposal(
    proposal: Mapping[str, Any],
    adapter: CameraAdapterInput,
    *,
    producer_config_id: str = PRODUCER_CONFIG_ID,
) -> None:
    required = set(_PROPOSAL_PRESERVED_FIELDS) | {"schema_version"}
    if not isinstance(proposal, Mapping) or not required.issubset(proposal):
        raise CameraEpisodeProposalInferenceError("S0 proposal fields are incomplete")
    if (
        proposal["schema_version"] != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or proposal["producer_config_id"] != producer_config_id
        or proposal["proposal_status"] not in PROPOSAL_STATUSES
        or proposal["manual_review_required"] is not True
        or not isinstance(proposal["proposal_id"], str)
        or not proposal["proposal_id"]
        or not isinstance(proposal["technical_segment_index"], int)
        or isinstance(proposal["technical_segment_index"], bool)
        or proposal["technical_segment_index"] < 0
    ):
        raise CameraEpisodeProposalInferenceError("S0 proposal producer or identity is invalid")
    for name in _SCOPE_FIELDS[:-1]:
        if proposal[name] != adapter.media_sidecar[name]:
            raise CameraEpisodeProposalInferenceError(
                "S0 proposal scope differs from sidecar"
            )
    if not isinstance(proposal["track_id"], int) or isinstance(proposal["track_id"], bool):
        raise CameraEpisodeProposalInferenceError("S0 proposal track ID is invalid")
    start = float(proposal["start_sec"])
    end = float(proposal["end_sec_exclusive"])
    duration = float(proposal["duration_sec"])
    media_duration = float(adapter.media_sidecar["duration_sec"])
    if (
        not math.isfinite(start + end + duration)
        or start < 0.0
        or start >= end
        or end > media_duration + 1e-9
        or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise CameraEpisodeProposalInferenceError("S0 proposal endpoints are invalid")
    status = str(proposal["proposal_status"])
    if proposal.get("uncertain") is not (status != "proposed"):
        raise CameraEpisodeProposalInferenceError("S0 proposal uncertainty status differs")
    for name in ("reason_codes", "hard_break_reasons"):
        values = proposal[name]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise CameraEpisodeProposalInferenceError("S0 proposal reasons are invalid")


def _proposal_shape_result(
    *,
    bundle: _BundleInput,
    proposal: Mapping[str, Any],
    prepared: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    geometry_prediction: Mapping[str, Any] | None,
    primary_config: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(proposal["proposal_status"])
    quality_flags = ["automatic_boundary_proposal"]
    qc_values: dict[str, Any] = {
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
    }
    if prepared is not None:
        qc_values = {name: prepared[name] for name in _QC_RESULT_FIELDS}
        if prepared["quality_flags"]:
            quality_flags = list(prepared["quality_flags"])
    if status == "rejected_by_qc":
        prediction_status = "unavailable"
        prediction_reasons = [
            "proposal_rejected_by_qc",
            *list(proposal["reason_codes"]),
        ]
        qc_values["qc_status"] = "unavailable"
        qc_values["qc_reason_codes"] = list(prediction_reasons)
    elif prepared is None:
        raise CameraEpisodeProposalInferenceError(
            "forward-eligible proposal interval was not prepared"
        )
    elif prediction is None:
        prediction_status = str(prepared["qc_status"])
        prediction_reasons = list(prepared["qc_reason_codes"])
    else:
        prediction_status = str(prediction["window_status"])
        prediction_reasons = list(prediction["reason_codes"])
    if status == "uncertain":
        quality_flags = list(
            dict.fromkeys([*quality_flags, "boundary_uncertain_candidate"])
        )
    base_binary = prediction.get("binary") if prediction is not None else None
    base_subtype = prediction.get("subtype") if prediction is not None else None
    base_four_class = prediction.get("four_class") if prediction is not None else None
    binary = base_binary
    subtype = base_subtype
    four_class = base_four_class
    classification_source = "frozen_topowander"
    if geometry_prediction is not None:
        probabilities = geometry_prediction["probabilities"]
        predicted_pattern = str(geometry_prediction["predicted_pattern"])
        four_class = {
            "class_order": ["direct", "pacing", "lapping", "random"],
            "probabilities": [
                float(probabilities[name])
                for name in ("direct", "pacing", "lapping", "random")
            ],
            "predicted_label": predicted_pattern,
        }
        direct_probability = float(probabilities["direct"])
        binary = {
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [direct_probability, 1.0 - direct_probability],
            "predicted_label": (
                "direct_or_non_wandering"
                if predicted_pattern == "direct"
                else "wandering_like"
            ),
        }
        subtype_mass = sum(
            float(probabilities[name]) for name in ("pacing", "lapping", "random")
        )
        subtype_probabilities = [
            float(probabilities[name]) / max(subtype_mass, 1e-12)
            for name in ("pacing", "lapping", "random")
        ]
        subtype = {
            "class_order": ["pacing", "lapping", "random"],
            "probabilities": subtype_probabilities,
            "predicted_label": ("pacing", "lapping", "random")[
                int(max(range(3), key=subtype_probabilities.__getitem__))
            ],
        }
        quality_flags = list(
            dict.fromkeys([*quality_flags, "home_camera_geometry_head"])
        )
        classification_source = "home_camera_geometry_head"
    return {
        "schema_version": CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION,
        "bundle_id": bundle.binding["bundle_id"],
        "participant_id": bundle.binding["participant_id"],
        "session_id": bundle.binding["session_id"],
        "camera_setup_id": bundle.binding["camera_setup_id"],
        "clock_domain_id": bundle.binding["clock_domain_id"],
        **{name: proposal[name] for name in _PROPOSAL_PRESERVED_FIELDS},
        "boundary_status": status,
        **qc_values,
        "quality_flags": quality_flags,
        "shape_inference_policy_id": SHAPE_INFERENCE_POLICY_ID,
        "shape_inference_role": {
            "proposed": "automatic_candidate",
            "uncertain": "boundary_uncertain_candidate_diagnostic",
            "rejected_by_qc": "not_run_rejected_by_qc",
        }[status],
        "prediction_status": prediction_status,
        "prediction_reason_codes": list(dict.fromkeys(prediction_reasons)),
        "binary": binary,
        "subtype": subtype,
        "four_class": four_class,
        "base_model_binary": base_binary,
        "base_model_subtype": base_subtype,
        "base_model_four_class": base_four_class,
        "camera_geometry": geometry_prediction,
        "classification_source": classification_source,
        "predicted_binary": (
            binary["predicted_label"] if binary is not None else None
        ),
        "predicted_pattern": (
            four_class["predicted_label"] if four_class is not None else None
        ),
        "candidate_id": primary_config["candidate"]["candidate_id"],
        "candidate_manifest_sha256": primary_config["candidate"]["manifest_sha256"],
        "model_state_sha256": primary_config["candidate"]["model_state_sha256"],
        "binary_decision_threshold": 0.5,
        "trusted_minimum_track_confidence": bundle.minimum_track_confidence,
        "motion_evidence_threshold_body_heights": (
            bundle.movement_evidence_threshold_body_heights
        ),
        "probability_calibrated": False,
        "model_invocation_skipped": (
            bool(prediction["model_invocation_skipped"])
            if prediction is not None
            else True
        ),
    }


def _resolved_episode_rows(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for row in results:
        geometry = row.get("camera_geometry")
        expand = (
            isinstance(geometry, Mapping)
            and geometry.get("aggregation_reason") == "micro_bout_direct_consensus"
            and row.get("prediction_status") == "ready"
            and row.get("predicted_pattern") == "direct"
            and isinstance(geometry.get("motion_bouts"), list)
            and len(geometry["motion_bouts"]) >= 2
        )
        if expand:
            source_intervals = [
                (
                    float(bout["start_sec"]),
                    float(bout["end_sec_exclusive"]),
                )
                for bout in geometry["motion_bouts"]
            ]
            intervals = []
            for source_start, source_end in source_intervals:
                duration = source_end - source_start
                allow_long_bout_subdivision = (
                    8 <= int(geometry["motion_bout_count"]) <= 14
                )
                if duration <= 10.0 or not allow_long_bout_subdivision:
                    intervals.append((source_start, source_end))
                    continue
                piece_count = int(math.ceil(duration / 5.0))
                for piece_index in range(piece_count):
                    intervals.append(
                        (
                            source_start + duration * piece_index / piece_count,
                            source_start
                            + duration * (piece_index + 1) / piece_count,
                        )
                    )
        else:
            intervals = [
                (float(row["start_sec"]), float(row["end_sec_exclusive"]))
            ]
        for index, (start, end) in enumerate(intervals):
            identity = {
                "proposal_id": row["proposal_id"],
                "resolution_level": "motion_bout" if expand else "behavior_episode",
                "index": index,
                "start_sec": start,
                "end_sec_exclusive": end,
            }
            resolved_id = "resolved-episode-" + hashlib.sha256(
                canonical_json_bytes(identity)
            ).hexdigest()
            resolved.append(
                {
                    "schema_version": "wandering-camera-resolved-episode-prediction-v1",
                    "resolved_episode_id": resolved_id,
                    "parent_proposal_id": row["proposal_id"],
                    "resolution_level": (
                        "motion_bout" if expand else "behavior_episode"
                    ),
                    "resolution_reason": (
                        "micro_bout_direct_consensus"
                        if expand
                        else "parent_behavior_episode_retained"
                    ),
                    **{
                        name: row[name]
                        for name in (
                            "bundle_id",
                            "participant_id",
                            "session_id",
                            "camera_setup_id",
                            "clock_domain_id",
                            "source_group_id",
                            "source_video_id",
                            "device_id",
                            "setup_id",
                            "stream_epoch",
                            "track_id",
                            "technical_segment_index",
                            "boundary_status",
                            "prediction_status",
                            "classification_source",
                        )
                    },
                    "start_sec": start,
                    "end_sec_exclusive": end,
                    "duration_sec": end - start,
                    "predicted_pattern": row["predicted_pattern"],
                    "binary": row["binary"],
                    "subtype": row["subtype"],
                    "four_class": row["four_class"],
                    "quality_flags": list(
                        dict.fromkeys(
                            [
                                *row["quality_flags"],
                                "resolved_motion_bout" if expand else "resolved_behavior_episode",
                            ]
                        )
                    ),
                    "behavior_truth_consumed_at_inference": False,
                }
            )
    resolved.sort(
        key=lambda row: (
            str(row["source_video_id"]),
            int(row["track_id"]),
            float(row["start_sec"]),
            str(row["resolved_episode_id"]),
        )
    )
    return resolved


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraEpisodeProposalInferenceError(f"cannot read {role}") from exc
    if not isinstance(value, dict):
        raise CameraEpisodeProposalInferenceError(f"{role} must be a JSON object")
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraEpisodeProposalInferenceError(f"cannot read {role}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise CameraEpisodeProposalInferenceError(f"{role} rows must be JSON objects")
    return values


def _resolve_index_path(
    index_path: Path, value: str, *, directory: bool = False
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = index_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape batch input path is inaccessible"
        ) from exc
    if directory and not resolved.is_dir():
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape bundle path is not a directory"
        )
    if not directory and not resolved.is_file():
        raise CameraEpisodeProposalInferenceError(
            "proposal-shape input path is not a file"
        )
    return resolved


def _bundle_readme(summary: Mapping[str, Any]) -> str:
    return (
        "# Automatic proposal shape results\n\n"
        "Each row preserves one original S0 proposal. `proposed` and `uncertain` "
        "rows that pass episode QC invoke the fixed shape model; uncertain rows "
        "remain boundary-uncertain diagnostics and are not accepted boundaries. "
        "Rejected rows remain visible and skip model invocation. No boundary was "
        "accepted and no performance metric was computed.\n\n"
        f"- shape inference policy: `{summary['shape_inference_policy_id']}`\n"
        f"- proposals/results: `{summary['proposal_count']}/{summary['result_count']}`\n"
        f"- resolved episodes: `{summary['resolved_episode_count']}`\n"
        f"- model forward invocations: `{summary['model_forward_invocation_count']}`\n"
        f"- skipped rows: `{summary['model_invocation_skipped_count']}`\n"
    )


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera proposal-shape output already exists: {output}")
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


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_PROPOSAL_SHAPE_CONFIG_SCHEMA_VERSION,
        "purpose": "topowander_automatic_boundary_proposal_shape",
        "episode_inference_path": "configs/modules/wandering_camera_episode_v1.yaml",
        "primary_inference_path": "configs/modules/wandering_camera_primary_v1.yaml",
        "candidate_manifest_path": (
            "reports/mental_health/wandering_performance/"
            "m0r_score_entry_hardening_v1/artifacts/"
            "topowander_m0r_candidate_v3/candidate_manifest.json"
        ),
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "input_boundary_kind": "automatic_proposal",
        "quality_flag": "automatic_boundary_proposal",
        "shape_inference_policy_id": SHAPE_INFERENCE_POLICY_ID,
        "forward_proposal_statuses": list(FORWARD_PROPOSAL_STATUSES),
        "skip_proposal_statuses": list(SKIP_PROPOSAL_STATUSES),
        "uncertain_shape_output_role": (
            "boundary_uncertain_candidate_diagnostic"
        ),
        "proposal_schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "output_prediction_schema_version": (
            CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION
        ),
        "output_summary_schema_version": (
            CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION
        ),
    }


__all__ = [
    "CAMERA_EPISODE_PROPOSAL_SHAPE_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION",
    "CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION",
    "FORWARD_PROPOSAL_STATUSES",
    "SHAPE_INFERENCE_POLICY_ID",
    "CameraEpisodeProposalBuildResult",
    "CameraEpisodeProposalInferenceError",
    "build_camera_episode_proposal_inference_bundle",
    "load_camera_episode_proposal_shape_config",
]
