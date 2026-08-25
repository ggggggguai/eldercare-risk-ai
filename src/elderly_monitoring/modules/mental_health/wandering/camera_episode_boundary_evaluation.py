"""Synthetic-first evaluation for continuous camera boundary proposals.

The evaluator is read-only. It joins S0 proposal rows to an independent
boundary-only view, delegates one-to-one assignment to the existing camera
development matcher, and never invokes shape inference or rewrites either
input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    validate_media_sidecar,
    validation_scope_for_authorization_status,
)
from elderly_monitoring.modules.mental_health.wandering.camera_development import (
    EPISODE_SCOPE_FIELDS,
    CameraDevelopmentError,
    match_episodes_one_to_one,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION,
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION,
    PROPOSAL_STATUSES,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
    CameraEpisodeImportError,
    load_episode_boundaries,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
)


CAMERA_EPISODE_BOUNDARY_EVAL_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-config-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-summary-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_METRICS_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-metrics-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_MATCH_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-match-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_UNMATCHED_TRUTH_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-unmatched-truth-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_UNMATCHED_PROPOSAL_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-unmatched-proposal-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_DIAGNOSTIC_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-diagnostic-v1"
)
CAMERA_EPISODE_BOUNDARY_EVAL_FAILURE_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-eval-failure-v1"
)

BOUNDARY_EVALUATION_INDEX_FIELDS = frozenset(
    {
        "bundle_id",
        "proposal_bundle_dir",
        "human_boundary_jsonl",
        "media_sidecar",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
        "truth_source",
    }
)
_PROPOSAL_FIELDS = frozenset(
    {
        "schema_version",
        "proposal_id",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
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
        "source_observation_count",
        "accepted_observation_count",
        "source_bucket_count",
        "producer_config_id",
    }
)
_PROPOSAL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "technical_segment_index",
        "segment_start_observation_sec",
        "segment_end_observation_sec",
        "source_observation_count",
        "accepted_observation_count",
        "source_bucket_count",
        "observed_bucket_count",
        "accepted_observation_coverage_ratio",
        "hard_break_reasons",
        "state_sequence",
        "proposal_ids",
        "proposal_status_counts",
        "reason_codes",
        "manual_review_required",
    }
)
_PROPOSAL_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "evidence_scope",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "validation_scope",
        "producer_config_id",
        "proposal_schema_version",
        "accepted_boundary_schema_version",
        "proposal_is_accepted_boundary",
        "proposal_count",
        "locomotion_proposal_count",
        "proposal_status_counts",
        "scope_count",
        "technical_segment_count",
        "technical_hard_break_segment_count",
        "source_observation_count",
        "accepted_observation_count",
        "accepted_observation_coverage_ratio",
        "source_bucket_count",
        "observed_bucket_count",
        "observed_bucket_coverage_ratio",
        "technical_segment_locomotion_coverage_ratio",
        "proposal_duration_seconds",
        "uncertain_count",
        "manual_review_required_count",
        "reason_counts",
        "stationary_dwell_candidate_seconds",
        "stationary_dwell_candidate_basis",
        "stationary_dwell_candidate_validated",
        "movement_start_candidate_validated",
        "truth_labels_consumed_by_producer",
        "shape_predictions_consumed_by_producer",
        "frozen_model_loaded",
        "models_trained_or_updated",
        "automatic_boundary_validated",
        "boundary_metrics_available",
        "shape_metrics_available",
        "manual_acceptance_required",
    }
)
_PROPOSAL_DEVELOPMENT_SUMMARY_FIELDS = frozenset(
    {
        "minimum_track_confidence",
        "movement_evidence_threshold_body_heights",
        "segmenter_strategy",
    }
)
_TRUTH_SOURCES = frozenset({"synthetic_fixture", "independent_human"})
_VIEW_ORDER = ("all_locomotion_candidates", "proposed_only_conditional")
_GROUP_FIELDS = (
    ("participant", "participant_id"),
    ("session", "session_id"),
    ("camera_setup", "camera_setup_id"),
    ("clock_domain", "clock_domain_id"),
)
_NOT_COMPUTABLE = "not_computable"
_CONFIG_RELATIVE = Path(
    "configs/modules/wandering_camera_episode_boundary_eval_v1.yaml"
)


class CameraEpisodeBoundaryEvaluationError(ValueError):
    """S1A input identity, schema, or metric contract failed closed."""

    def __init__(self, message: str, *, code: str = "invalid_boundary_evaluation_input"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CameraEpisodeBoundaryEvaluationBuildResult:
    output_dir: Path
    ready_truth_count: int
    uncertain_truth_count: int
    proposal_count: int
    match_count: int
    failure_count: int


@dataclass(frozen=True)
class _LoadedInputs:
    proposals: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    coverage: Mapping[str, Any]
    truth_source: str
    batch_count: int
    input_snapshots: Mapping[Path, bytes]


def load_camera_episode_boundary_evaluation_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load the fixed, explicitly unvalidated S1A evaluator policy."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            "cannot read camera episode boundary evaluation config"
        ) from exc
    if value != _expected_config():
        raise CameraEpisodeBoundaryEvaluationError(
            "camera episode boundary evaluation config fields or values drifted"
        )
    return value


def build_camera_episode_boundary_evaluation_bundle(
    *,
    project_root: str | Path,
    config_path: str | Path,
    batch_index_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeBoundaryEvaluationBuildResult:
    """Load immutable inputs, evaluate them, and atomically write a fresh bundle."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera boundary evaluation output exists: {output}")
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeBoundaryEvaluationError("project root is inaccessible") from exc
    fixed_config = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(config_path).resolve(strict=False) != fixed_config:
        raise CameraEpisodeBoundaryEvaluationError(
            "boundary evaluation config must use the fixed repository path"
        )
    config = load_camera_episode_boundary_evaluation_config(fixed_config)
    camera_path = (root / str(config["camera_chain"]["path"])).resolve(strict=False)
    if not camera_path.is_file():
        raise CameraEpisodeBoundaryEvaluationError("camera chain config is missing")
    try:
        camera_config = load_camera_config(camera_path)
    except CameraInferenceError as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            "camera chain config is invalid"
        ) from exc
    try:
        index_path = Path(batch_index_path).resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            "boundary evaluation batch index is inaccessible"
        ) from exc
    index_rows = _load_batch_index(index_path)
    loaded = _load_inputs(index_path, index_rows, camera_config)
    artifacts = evaluate_camera_episode_boundaries(
        loaded.proposals,
        loaded.boundaries,
        config=config,
        input_coverage=loaded.coverage,
    )
    summary = _build_output_summary(loaded, artifacts, config)
    metrics = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_METRICS_SCHEMA_VERSION,
        "truth_source": loaded.truth_source,
        "evidence_scope": summary["evidence_scope"],
        **artifacts["metrics"],
    }
    files = {
        "summary.json": canonical_json_bytes(summary),
        "metrics.json": canonical_json_bytes(metrics),
        "matches.jsonl": canonical_jsonl_bytes(artifacts["matches"]),
        "unmatched_truth.jsonl": canonical_jsonl_bytes(
            artifacts["unmatched_truth"]
        ),
        "unmatched_proposals.jsonl": canonical_jsonl_bytes(
            artifacts["unmatched_proposals"]
        ),
        "split_merge_diagnostics.jsonl": canonical_jsonl_bytes(
            artifacts["diagnostics"]
        ),
        "failures.jsonl": canonical_jsonl_bytes(artifacts["failures"]),
        "README.md": _readme_bytes(summary, metrics),
    }
    _assert_input_snapshots_unchanged(loaded.input_snapshots)
    _commit_new_directory(output, files)
    main_matches = sum(
        row["view"] == "all_locomotion_candidates"
        for row in artifacts["matches"]
    )
    ready_count = sum(
        row["boundary_status"] == "ready" for row in loaded.boundaries
    )
    uncertain_count = len(loaded.boundaries) - ready_count
    return CameraEpisodeBoundaryEvaluationBuildResult(
        output_dir=output,
        ready_truth_count=ready_count,
        uncertain_truth_count=uncertain_count,
        proposal_count=len(loaded.proposals),
        match_count=main_matches,
        failure_count=len(artifacts["failures"]),
    )


def evaluate_camera_episode_boundaries(
    proposals: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    input_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate already-bound rows while delegating all matching decisions."""

    if config != _expected_config():
        raise CameraEpisodeBoundaryEvaluationError(
            "boundary evaluation requires the fixed S1A config"
        )
    proposal_rows = [_validate_bound_proposal(row) for row in proposals]
    boundary_rows = [_validate_bound_boundary(row) for row in boundaries]
    _unique_text_ids(proposal_rows, "proposal_id")
    _unique_text_ids(boundary_rows, "episode_id")
    ready_truth = [
        row for row in boundary_rows if row["boundary_status"] == "ready"
    ]
    uncertain_truth = [
        row
        for row in boundary_rows
        if row["boundary_status"] == "boundary_uncertain"
    ]
    view_statuses = config["evaluation_views"]
    view_internal: dict[str, dict[str, Any]] = {}
    all_matches: list[dict[str, Any]] = []
    unmatched_truth: list[dict[str, Any]] = []
    unmatched_proposals: list[dict[str, Any]] = []
    view_metrics: dict[str, Any] = {}
    for view in _VIEW_ORDER:
        candidates = [
            row
            for row in proposal_rows
            if row["proposal_status"] in view_statuses[view]
        ]
        result = _match_view(
            view,
            candidates,
            ready_truth,
            matching_policy=config["matching_policy"],
        )
        view_internal[view] = result
        all_matches.extend(result["matches"])
        unmatched_truth.extend(result["unmatched_truth"])
        unmatched_proposals.extend(result["unmatched_proposals"])
        view_metrics[view] = _view_metric_row(
            ready_truth,
            candidates,
            result["matches"],
        )
    for truth in uncertain_truth:
        unmatched_truth.append(
            _unmatched_truth_row(
                truth,
                view="truth_coverage_only",
                disposition="boundary_uncertain_excluded_from_determinate_metrics",
            )
        )
    rejected = [
        row for row in proposal_rows if row["proposal_status"] == "rejected_by_qc"
    ]
    for proposal in rejected:
        unmatched_proposals.append(
            _unmatched_proposal_row(
                proposal,
                view="coverage_only",
                disposition="rejected_by_qc_not_a_locomotion_candidate",
            )
        )
    coverage = _coverage_metrics(
        proposal_rows,
        boundary_rows,
        input_coverage=input_coverage,
    )
    diagnostics = _split_merge_diagnostics(
        proposal_rows,
        ready_truth,
        config=config,
    )
    group_support = _group_support(
        proposal_rows,
        ready_truth,
        view_internal,
    )
    duration_support = _duration_support(
        proposal_rows,
        ready_truth,
        view_internal,
        config=config,
    )
    failures = _failure_rows(
        proposal_rows,
        ready_truth,
        uncertain_truth,
        view_internal,
        coverage,
        diagnostics,
        config=config,
    )
    return {
        "metrics": {
            "primary_view": "all_locomotion_candidates",
            "conditional_view": "proposed_only_conditional",
            "views": view_metrics,
            "coverage": coverage,
            "group_support": group_support,
            "duration_support": duration_support,
            "matching_policy": dict(config["matching_policy"]),
            "failure_thresholds": dict(config["failure_thresholds"]),
            "shape_metrics_available": False,
            "automatic_shape_evaluated": False,
        },
        "matches": sorted(all_matches, key=_match_sort_key),
        "unmatched_truth": sorted(unmatched_truth, key=_unmatched_truth_sort_key),
        "unmatched_proposals": sorted(
            unmatched_proposals, key=_unmatched_proposal_sort_key
        ),
        "diagnostics": sorted(diagnostics, key=_diagnostic_sort_key),
        "failures": sorted(failures, key=_failure_sort_key),
    }


def _match_view(
    view: str,
    candidates: Sequence[Mapping[str, Any]],
    ready_truth: Sequence[Mapping[str, Any]],
    *,
    matching_policy: Mapping[str, Any],
) -> dict[str, Any]:
    matcher_policy = {
        "policy_id": matching_policy["policy_id"],
        "minimum_temporal_iou": matching_policy["minimum_temporal_iou"],
        "maximum_onset_delta_sec": matching_policy["maximum_onset_delta_sec"],
    }
    prediction_rows = [
        {
            "prediction_id": row["proposal_id"],
            **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
            "start_sec": row["start_sec"],
            "end_sec_exclusive": row["end_sec_exclusive"],
        }
        for row in candidates
    ]
    annotation_rows = [
        {
            "annotation_id": row["episode_id"],
            **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
            "start_sec": row["start_sec"],
            "end_sec_exclusive": row["end_sec_exclusive"],
        }
        for row in ready_truth
    ]
    try:
        matched = match_episodes_one_to_one(
            prediction_rows,
            annotation_rows,
            matching_policy=matcher_policy,
        )
    except CameraDevelopmentError as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            "shared episode matching failed"
        ) from exc
    proposal_by_id = {str(row["proposal_id"]): row for row in candidates}
    truth_by_id = {str(row["episode_id"]): row for row in ready_truth}
    match_rows = [
        _match_row(
            view,
            proposal_by_id[str(row["prediction_id"])],
            truth_by_id[str(row["annotation_id"])],
            matching_policy_id=str(matcher_policy["policy_id"]),
        )
        for row in matched["matches"]
    ]
    unmatched_truth_rows = [
        _unmatched_truth_row(
            truth_by_id[episode_id],
            view=view,
            disposition="unmatched_ready_truth",
        )
        for episode_id in matched["unmatched_annotation_ids"]
    ]
    unmatched_proposal_rows = [
        _unmatched_proposal_row(
            proposal_by_id[proposal_id],
            view=view,
            disposition="unmatched_locomotion_candidate",
        )
        for proposal_id in matched["unmatched_prediction_ids"]
    ]
    return {
        "candidates": list(candidates),
        "matches": sorted(match_rows, key=_match_sort_key),
        "unmatched_truth": sorted(
            unmatched_truth_rows, key=_unmatched_truth_sort_key
        ),
        "unmatched_proposals": sorted(
            unmatched_proposal_rows, key=_unmatched_proposal_sort_key
        ),
        "optimization_order": list(matched["optimization_order"]),
    }


def _match_row(
    view: str,
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    matching_policy_id: str,
) -> dict[str, Any]:
    p_start, p_end = _interval(proposal)
    t_start, t_end = _interval(truth)
    tiou = _temporal_iou(proposal, truth)
    onset_signed = p_start - t_start
    offset_signed = p_end - t_end
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_MATCH_SCHEMA_VERSION,
        "view": view,
        "matching_policy_id": matching_policy_id,
        "proposal_id": proposal["proposal_id"],
        "episode_id": truth["episode_id"],
        **{field: proposal[field] for field in EPISODE_SCOPE_FIELDS},
        "proposal_status": proposal["proposal_status"],
        "boundary_status": truth["boundary_status"],
        "proposal_start_sec": p_start,
        "proposal_end_sec_exclusive": p_end,
        "truth_start_sec": t_start,
        "truth_end_sec_exclusive": t_end,
        "proposal_duration_sec": p_end - p_start,
        "truth_duration_sec": t_end - t_start,
        "temporal_iou": tiou,
        "onset_signed_error_sec": onset_signed,
        "onset_absolute_error_sec": abs(onset_signed),
        "offset_signed_error_sec": offset_signed,
        "offset_absolute_error_sec": abs(offset_signed),
    }


def _unmatched_truth_row(
    truth: Mapping[str, Any],
    *,
    view: str,
    disposition: str,
) -> dict[str, Any]:
    start, end = _interval(truth)
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_UNMATCHED_TRUTH_SCHEMA_VERSION,
        "view": view,
        "disposition": disposition,
        "episode_id": truth["episode_id"],
        **{field: truth[field] for field in EPISODE_SCOPE_FIELDS},
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "boundary_status": truth["boundary_status"],
        "boundary_reason_codes": list(truth["boundary_reason_codes"]),
    }


def _unmatched_proposal_row(
    proposal: Mapping[str, Any],
    *,
    view: str,
    disposition: str,
) -> dict[str, Any]:
    start, end = _interval(proposal)
    return {
        "schema_version": (
            CAMERA_EPISODE_BOUNDARY_EVAL_UNMATCHED_PROPOSAL_SCHEMA_VERSION
        ),
        "view": view,
        "disposition": disposition,
        "proposal_id": proposal["proposal_id"],
        **{field: proposal[field] for field in EPISODE_SCOPE_FIELDS},
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "proposal_status": proposal["proposal_status"],
        "reason_codes": list(proposal["reason_codes"]),
        "hard_break_reasons": list(proposal["hard_break_reasons"]),
        "manual_review_required": bool(proposal["manual_review_required"]),
    }


def _view_metric_row(
    ready_truth: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    detection = _detection_metrics(
        truth_support=len(ready_truth),
        candidate_support=len(candidates),
        matched_count=len(matches),
    )
    return {
        "detection": detection,
        "localization": _localization_metrics(matches),
        "optimization_order": [
            "maximum_match_count",
            "maximum_total_temporal_iou",
            "minimum_total_onset_offset_delta",
            "stable_prediction_annotation_id",
        ],
    }


def _detection_metrics(
    *,
    truth_support: int,
    candidate_support: int,
    matched_count: int,
) -> dict[str, Any]:
    precision: float | str = (
        matched_count / candidate_support
        if candidate_support
        else _NOT_COMPUTABLE
    )
    recall: float | str = (
        matched_count / truth_support if truth_support else _NOT_COMPUTABLE
    )
    f1 = _f1(precision, recall)
    return {
        "ready_truth_support": truth_support,
        "candidate_support": candidate_support,
        "matched_count": matched_count,
        "unmatched_truth_count": truth_support - matched_count,
        "unmatched_candidate_count": candidate_support - matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _localization_metrics(
    matches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        field: _numeric_summary([float(row[field]) for row in matches])
        for field in (
            "temporal_iou",
            "onset_signed_error_sec",
            "onset_absolute_error_sec",
            "offset_signed_error_sec",
            "offset_absolute_error_sec",
        )
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": _NOT_COMPUTABLE,
            "median": _NOT_COMPUTABLE,
            "p90": _NOT_COMPUTABLE,
            "max": _NOT_COMPUTABLE,
        }
    ordered = sorted(float(value) for value in values)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _f1(precision: float | str, recall: float | str) -> float | str:
    if not isinstance(precision, float) or not isinstance(recall, float):
        return _NOT_COMPUTABLE
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _group_support(
    proposals: Sequence[Mapping[str, Any]],
    ready_truth: Sequence[Mapping[str, Any]],
    view_internal: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group_name, field in _GROUP_FIELDS:
        values = sorted(
            {
                str(row[field])
                for row in [*proposals, *ready_truth]
            }
        )
        output[group_name] = {}
        for value in values:
            row: dict[str, Any] = {"views": {}}
            for view in _VIEW_ORDER:
                candidates = [
                    item
                    for item in view_internal[view]["candidates"]
                    if item[field] == value
                ]
                truths = [item for item in ready_truth if item[field] == value]
                matches = [
                    item
                    for item in view_internal[view]["matches"]
                    if item[field] == value
                ]
                row["views"][view] = _view_metric_row(
                    truths,
                    candidates,
                    matches,
                )
            output[group_name][value] = row
    return output


def _duration_support(
    proposals: Sequence[Mapping[str, Any]],
    ready_truth: Sequence[Mapping[str, Any]],
    view_internal: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for band in config["duration_bands"]:
        name = str(band["name"])
        truths = [
            row
            for row in ready_truth
            if _duration_band(float(row["duration_sec"]), config) == name
        ]
        band_row: dict[str, Any] = {
            "ready_truth_support": len(truths),
            "views": {},
        }
        for view in _VIEW_ORDER:
            candidates = [
                row
                for row in view_internal[view]["candidates"]
                if _duration_band(float(row["duration_sec"]), config) == name
            ]
            matches = view_internal[view]["matches"]
            matched_truth_count = sum(
                _duration_band(float(row["truth_duration_sec"]), config) == name
                for row in matches
            )
            matched_candidate_count = sum(
                _duration_band(float(row["proposal_duration_sec"]), config) == name
                for row in matches
            )
            same_duration_band_match_count = sum(
                _duration_band(float(row["truth_duration_sec"]), config) == name
                and _duration_band(float(row["proposal_duration_sec"]), config)
                == name
                for row in matches
            )
            cross_duration_band_match_count_by_truth = (
                matched_truth_count - same_duration_band_match_count
            )
            cross_duration_band_match_count_by_proposal = (
                matched_candidate_count - same_duration_band_match_count
            )
            precision: float | str = (
                matched_candidate_count / len(candidates)
                if candidates
                else _NOT_COMPUTABLE
            )
            recall: float | str = (
                matched_truth_count / len(truths)
                if truths
                else _NOT_COMPUTABLE
            )
            cross_band = (
                cross_duration_band_match_count_by_truth > 0
                or cross_duration_band_match_count_by_proposal > 0
            )
            band_row["views"][view] = {
                "candidate_support": len(candidates),
                "matched_truth_count": matched_truth_count,
                "matched_candidate_count": matched_candidate_count,
                "same_duration_band_match_count": (
                    same_duration_band_match_count
                ),
                "cross_duration_band_match_count_by_truth": (
                    cross_duration_band_match_count_by_truth
                ),
                "cross_duration_band_match_count_by_proposal": (
                    cross_duration_band_match_count_by_proposal
                ),
                "unmatched_truth_count": len(truths) - matched_truth_count,
                "unmatched_candidate_count": (
                    len(candidates) - matched_candidate_count
                ),
                "precision": precision,
                "recall": recall,
                "f1": _NOT_COMPUTABLE if cross_band else _f1(precision, recall),
                "f1_reason": (
                    "precision_and_recall_use_distinct_duration_cohorts_with_cross_band_matches"
                    if cross_band
                    else None
                ),
                "precision_population": "proposal_duration_band",
                "recall_population": "truth_duration_band",
                "localization_by_truth_duration": _localization_metrics(
                    [
                        row
                        for row in matches
                        if _duration_band(
                            float(row["truth_duration_sec"]), config
                        )
                        == name
                    ]
                ),
            }
        output[name] = band_row
    return output


def _coverage_metrics(
    proposals: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    *,
    input_coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    proposal_counts = Counter(str(row["proposal_status"]) for row in proposals)
    boundary_counts = Counter(str(row["boundary_status"]) for row in boundaries)
    ready_truth = [
        row for row in boundaries if row["boundary_status"] == "ready"
    ]
    overlap_counts = Counter(
        _ready_truth_overlap_status(truth, proposals) for truth in ready_truth
    )
    technical = {
        "bundle_count": 0,
        "technical_segment_count": 0,
        "source_observation_count": 0,
        "accepted_observation_count": 0,
        "accepted_observation_coverage_ratio": _NOT_COMPUTABLE,
        "source_bucket_count": 0,
        "observed_bucket_count": 0,
        "observed_bucket_coverage_ratio": _NOT_COMPUTABLE,
    }
    if input_coverage is not None:
        technical.update(dict(input_coverage))
    return {
        "human_boundary_status_counts": {
            "ready": boundary_counts["ready"],
            "boundary_uncertain": boundary_counts["boundary_uncertain"],
        },
        "proposal_status_counts": {
            name: proposal_counts[name] for name in PROPOSAL_STATUSES
        },
        "manual_review_required_count": sum(
            bool(row["manual_review_required"]) for row in proposals
        ),
        "ready_truth_overlap_status_counts": {
            "proposed_overlap": overlap_counts["proposed_overlap"],
            "uncertain_only_overlap": overlap_counts[
                "uncertain_only_overlap"
            ],
            "rejected_by_qc_only_overlap": overlap_counts[
                "rejected_by_qc_only_overlap"
            ],
            "no_proposal_overlap": overlap_counts["no_proposal_overlap"],
        },
        "technical_input_coverage": technical,
        "coverage_is_not_boundary_accuracy": True,
    }


def _ready_truth_overlap_status(
    truth: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
) -> str:
    overlapping = [
        row
        for row in proposals
        if _positive_scoped_overlap(row, truth)
    ]
    statuses = {str(row["proposal_status"]) for row in overlapping}
    if "proposed" in statuses:
        return "proposed_overlap"
    if "uncertain" in statuses:
        return "uncertain_only_overlap"
    if "rejected_by_qc" in statuses:
        return "rejected_by_qc_only_overlap"
    return "no_proposal_overlap"


def _split_merge_diagnostics(
    proposals: Sequence[Mapping[str, Any]],
    ready_truth: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    locomotion = [
        row for row in proposals if row["proposal_status"] != "rejected_by_qc"
    ]
    output: list[dict[str, Any]] = []
    for truth in ready_truth:
        overlapping = [
            proposal
            for proposal in locomotion
            if _positive_scoped_overlap(proposal, truth)
        ]
        if len(overlapping) > 1:
            output.append(
                _diagnostic_row(
                    "split_candidate",
                    overlapping,
                    [truth],
                    reason_codes=["one_ready_truth_overlaps_multiple_candidates"],
                    matching_policy_validated=bool(
                        config["matching_policy"]["validated"]
                    ),
                )
            )
    for proposal in locomotion:
        overlapping = [
            truth
            for truth in ready_truth
            if _positive_scoped_overlap(proposal, truth)
        ]
        if len(overlapping) > 1:
            output.append(
                _diagnostic_row(
                    "merge_candidate",
                    [proposal],
                    overlapping,
                    reason_codes=["one_candidate_overlaps_multiple_ready_truths"],
                    matching_policy_validated=bool(
                        config["matching_policy"]["validated"]
                    ),
                )
            )
        for truth in overlapping:
            if not _pair_passes_policy(proposal, truth, config["matching_policy"]):
                output.append(
                    _diagnostic_row(
                        "subthreshold_overlap",
                        [proposal],
                        [truth],
                        reason_codes=[
                            (
                                "positive_overlap_below_matching_policy"
                                if config["matching_policy"]["validated"]
                                else "positive_overlap_below_unvalidated_matching_policy"
                            )
                        ],
                        matching_policy_validated=bool(
                            config["matching_policy"]["validated"]
                        ),
                    )
                )
    by_scope: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for proposal in locomotion:
        by_scope.setdefault(_technical_segment_key(proposal), []).append(
            proposal
        )
    thresholds = config["failure_thresholds"]
    for rows in by_scope.values():
        ordered = sorted(rows, key=_entity_sort_key)
        for left, right in zip(ordered, ordered[1:]):
            left_start, left_end = _interval(left)
            right_start, right_end = _interval(right)
            gap = right_start - left_end
            if gap < -1e-9:
                continue
            if (
                min(left_end - left_start, right_end - right_start)
                <= float(thresholds["fragmentation_max_duration_sec"]) + 1e-9
                or gap
                <= float(thresholds["fragmentation_max_gap_sec"]) + 1e-9
            ):
                output.append(
                    _diagnostic_row(
                        "fragmentation_candidate",
                        [left, right],
                        [],
                        reason_codes=[
                            "adjacent_candidate_gap_or_duration_below_unvalidated_threshold"
                        ],
                        matching_policy_validated=bool(
                            config["matching_policy"]["validated"]
                        ),
                    )
                )
    return _unique_diagnostics(output)


def _diagnostic_row(
    diagnostic_type: str,
    proposals: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    *,
    reason_codes: Sequence[str],
    matching_policy_validated: bool,
) -> dict[str, Any]:
    scope_source = proposals[0] if proposals else truths[0]
    proposal_rows = sorted(proposals, key=_entity_sort_key)
    truth_rows = sorted(truths, key=_entity_sort_key)
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_type": diagnostic_type,
        **{field: scope_source[field] for field in EPISODE_SCOPE_FIELDS},
        "proposal_ids": [str(row["proposal_id"]) for row in proposal_rows],
        "episode_ids": [str(row["episode_id"]) for row in truth_rows],
        "proposal_intervals": [
            [float(row["start_sec"]), float(row["end_sec_exclusive"])]
            for row in proposal_rows
        ],
        "truth_intervals": [
            [float(row["start_sec"]), float(row["end_sec_exclusive"])]
            for row in truth_rows
        ],
        "proposal_statuses": [
            str(row["proposal_status"]) for row in proposal_rows
        ],
        "reason_codes": list(reason_codes),
        "matching_policy_validated": matching_policy_validated,
        "diagnostic_only": True,
    }


def _unique_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for row in rows:
        payload = canonical_json_bytes(row)
        if payload in seen:
            continue
        seen.add(payload)
        output.append(dict(row))
    return output


def _failure_rows(
    proposals: Sequence[Mapping[str, Any]],
    ready_truth: Sequence[Mapping[str, Any]],
    uncertain_truth: Sequence[Mapping[str, Any]],
    view_internal: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for view in _VIEW_ORDER:
        for row in view_internal[view]["unmatched_truth"]:
            failures.append(
                _failure_from_entities(
                    "unmatched_ready_truth",
                    view=view,
                    proposal=None,
                    truth=_truth_from_unmatched(row),
                    reason_codes=["ready_truth_unmatched_in_view"],
                )
            )
        for row in view_internal[view]["unmatched_proposals"]:
            proposal = _proposal_from_unmatched(row)
            failure_type = (
                "unmatched_proposed_candidate"
                if proposal["proposal_status"] == "proposed"
                else "unmatched_uncertain_candidate"
            )
            failures.append(
                _failure_from_entities(
                    failure_type,
                    view=view,
                    proposal=proposal,
                    truth=None,
                    reason_codes=["locomotion_candidate_unmatched_in_view"],
                )
            )
    for truth in uncertain_truth:
        failures.append(
            _failure_from_entities(
                "human_boundary_uncertain",
                view="truth_coverage_only",
                proposal=None,
                truth=truth,
                reason_codes=list(truth["boundary_reason_codes"]),
            )
        )
    for truth in ready_truth:
        coverage_status = _ready_truth_overlap_status(truth, proposals)
        if coverage_status == "uncertain_only_overlap":
            failure_type = "uncertain_only_truth_coverage"
        elif coverage_status == "rejected_by_qc_only_overlap":
            failure_type = "rejected_by_qc_truth_coverage"
        else:
            continue
        overlapping = [
            row for row in proposals if _positive_scoped_overlap(row, truth)
        ]
        failures.append(
            _failure_from_entities(
                failure_type,
                view="coverage_only",
                proposal=sorted(overlapping, key=_entity_sort_key)[0],
                truth=truth,
                reason_codes=[coverage_status],
            )
        )
    for proposal in proposals:
        if proposal["proposal_status"] != "rejected_by_qc":
            continue
        if any(_positive_scoped_overlap(proposal, truth) for truth in ready_truth):
            continue
        failures.append(
            _failure_from_entities(
                "rejected_by_qc_proposal",
                view="coverage_only",
                proposal=proposal,
                truth=None,
                reason_codes=list(proposal["reason_codes"]),
            )
        )
    for diagnostic in diagnostics:
        failures.append(_failure_from_diagnostic(diagnostic))
    thresholds = config["failure_thresholds"]
    for match in view_internal["all_locomotion_candidates"]["matches"]:
        for failure_type, field, threshold_name in (
            ("large_onset_error", "onset_absolute_error_sec", "large_onset_error_sec"),
            ("large_offset_error", "offset_absolute_error_sec", "large_offset_error_sec"),
        ):
            if float(match[field]) > float(thresholds[threshold_name]):
                failures.append(
                    _failure_from_match(
                        failure_type,
                        match,
                        reason_codes=[
                            f"{field}_above_unvalidated_threshold"
                        ],
                    )
                )
        if float(match["temporal_iou"]) < float(thresholds["low_tiou"]):
            failures.append(
                _failure_from_match(
                    "low_tiou_match",
                    match,
                    reason_codes=["tiou_below_unvalidated_failure_threshold"],
                )
            )
    return _unique_failures(failures)


def _failure_from_entities(
    failure_type: str,
    *,
    view: str,
    proposal: Mapping[str, Any] | None,
    truth: Mapping[str, Any] | None,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    source = proposal if proposal is not None else truth
    if source is None:
        raise CameraEpisodeBoundaryEvaluationError("failure row lost its scope")
    row = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_FAILURE_SCHEMA_VERSION,
        "failure_type": failure_type,
        "view": view,
        **{field: source[field] for field in EPISODE_SCOPE_FIELDS},
        "proposal_id": None if proposal is None else proposal["proposal_id"],
        "episode_id": None if truth is None else truth["episode_id"],
        "proposal_ids": [] if proposal is None else [proposal["proposal_id"]],
        "episode_ids": [] if truth is None else [truth["episode_id"]],
        "proposal_intervals": (
            []
            if proposal is None
            else [[proposal["start_sec"], proposal["end_sec_exclusive"]]]
        ),
        "truth_intervals": (
            []
            if truth is None
            else [[truth["start_sec"], truth["end_sec_exclusive"]]]
        ),
        "proposal_status": (
            None if proposal is None else proposal["proposal_status"]
        ),
        "boundary_status": None if truth is None else truth["boundary_status"],
        "reason_codes": list(dict.fromkeys(str(item) for item in reason_codes)),
        "thresholds_validated": False,
    }
    row["failure_id"] = _stable_row_id("boundary-failure", row)
    return row


def _failure_from_diagnostic(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_FAILURE_SCHEMA_VERSION,
        "failure_type": diagnostic["diagnostic_type"],
        "view": "diagnostic_only",
        **{field: diagnostic[field] for field in EPISODE_SCOPE_FIELDS},
        "proposal_id": (
            diagnostic["proposal_ids"][0]
            if len(diagnostic["proposal_ids"]) == 1
            else None
        ),
        "episode_id": (
            diagnostic["episode_ids"][0]
            if len(diagnostic["episode_ids"]) == 1
            else None
        ),
        "proposal_ids": list(diagnostic["proposal_ids"]),
        "episode_ids": list(diagnostic["episode_ids"]),
        "proposal_intervals": list(diagnostic["proposal_intervals"]),
        "truth_intervals": list(diagnostic["truth_intervals"]),
        "proposal_status": (
            diagnostic["proposal_statuses"][0]
            if len(diagnostic["proposal_statuses"]) == 1
            else None
        ),
        "boundary_status": "ready" if diagnostic["episode_ids"] else None,
        "reason_codes": list(diagnostic["reason_codes"]),
        "thresholds_validated": False,
    }
    row["failure_id"] = _stable_row_id("boundary-failure", row)
    return row


def _failure_from_match(
    failure_type: str,
    match: Mapping[str, Any],
    *,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    row = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_FAILURE_SCHEMA_VERSION,
        "failure_type": failure_type,
        "view": match["view"],
        **{field: match[field] for field in EPISODE_SCOPE_FIELDS},
        "proposal_id": match["proposal_id"],
        "episode_id": match["episode_id"],
        "proposal_ids": [match["proposal_id"]],
        "episode_ids": [match["episode_id"]],
        "proposal_intervals": [
            [match["proposal_start_sec"], match["proposal_end_sec_exclusive"]]
        ],
        "truth_intervals": [
            [match["truth_start_sec"], match["truth_end_sec_exclusive"]]
        ],
        "proposal_status": match["proposal_status"],
        "boundary_status": match["boundary_status"],
        "reason_codes": list(reason_codes),
        "thresholds_validated": False,
    }
    row["failure_id"] = _stable_row_id("boundary-failure", row)
    return row


def _unique_failures(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        output[str(row["failure_id"])] = dict(row)
    return list(output.values())


def _load_batch_index(path: Path) -> list[dict[str, str]]:
    rows = _load_jsonl(path, "boundary evaluation batch index")
    output: list[dict[str, str]] = []
    bundle_ids: set[str] = set()
    for value in rows:
        if frozenset(value) != BOUNDARY_EVALUATION_INDEX_FIELDS:
            raise CameraEpisodeBoundaryEvaluationError(
                "boundary evaluation batch index fields are invalid"
            )
        row = {field: _nonempty_text(value[field], field) for field in value}
        if row["truth_source"] not in _TRUTH_SOURCES:
            raise CameraEpisodeBoundaryEvaluationError(
                "batch truth_source is invalid"
            )
        if row["bundle_id"] in bundle_ids:
            raise CameraEpisodeBoundaryEvaluationError(
                "duplicate bundle_id in boundary evaluation index"
            )
        bundle_ids.add(row["bundle_id"])
        output.append(row)
    if not output:
        raise CameraEpisodeBoundaryEvaluationError(
            "boundary evaluation batch index is empty"
        )
    return sorted(output, key=lambda row: row["bundle_id"])


def _load_inputs(
    index_path: Path,
    index_rows: Sequence[Mapping[str, str]],
    camera_config: Mapping[str, Any],
) -> _LoadedInputs:
    proposals: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    truth_sources: set[str] = set()
    snapshots: dict[Path, bytes] = {index_path: index_path.read_bytes()}
    proposal_ids: set[str] = set()
    episode_ids: set[str] = set()
    proposal_identities: set[tuple[Any, ...]] = set()
    truth_identities: set[tuple[Any, ...]] = set()
    ready_intervals: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    source_bindings: dict[tuple[str, str, str, str, str], tuple[str, ...]] = {}
    source_video_bindings: dict[str, tuple[Any, ...]] = {}
    coverage_totals = Counter()
    validation_scopes: set[str] = set()
    for index_row in index_rows:
        bundle = _resolve_input(index_path.parent, index_row["proposal_bundle_dir"])
        boundary_path = _resolve_input(
            index_path.parent, index_row["human_boundary_jsonl"]
        )
        sidecar_path = _resolve_input(index_path.parent, index_row["media_sidecar"])
        if not bundle.is_dir():
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal bundle directory is missing"
            )
        for path in (
            bundle / "summary.json",
            bundle / "proposals.jsonl",
            bundle / "diagnostics.jsonl",
            boundary_path,
            sidecar_path,
        ):
            if not path.is_file():
                raise CameraEpisodeBoundaryEvaluationError(
                    "boundary evaluation input file is missing"
                )
            snapshots[path] = path.read_bytes()
        media = _load_media_sidecar(sidecar_path, camera_config)
        truth_source = str(index_row["truth_source"])
        truth_sources.add(truth_source)
        if (
            truth_source == "synthetic_fixture"
            and media["authorization_status"] != "synthetic_fixture"
        ):
            raise CameraEpisodeBoundaryEvaluationError(
                "synthetic truth_source requires a synthetic media sidecar"
            )
        if (
            truth_source == "independent_human"
            and media["authorization_status"]
            != "authorized_camera_labeled_evaluation"
        ):
            raise CameraEpisodeBoundaryEvaluationError(
                "independent human boundary requires labeled-evaluation authorization"
            )
        summary = _load_json(bundle / "summary.json", "proposal summary")
        proposal_rows = _load_jsonl(
            bundle / "proposals.jsonl", "proposal JSONL"
        )
        diagnostic_rows = _load_jsonl(
            bundle / "diagnostics.jsonl", "proposal diagnostics"
        )
        proposal_rows, diagnostic_rows = _validate_proposal_bundle(
            summary,
            proposal_rows,
            diagnostic_rows,
            media=media,
        )
        try:
            boundary_rows = load_episode_boundaries(boundary_path, media)
        except CameraEpisodeImportError as exc:
            raise CameraEpisodeBoundaryEvaluationError(str(exc)) from exc
        if truth_source == "independent_human" and any(
            row["boundary_source"] == "whole_clip" for row in boundary_rows
        ):
            raise CameraEpisodeBoundaryEvaluationError(
                "independent continuous boundary cannot use whole_clip"
            )
        binding = {
            "participant_id": _nonempty_text(
                index_row["participant_id"], "participant_id"
            ),
            "session_id": _nonempty_text(index_row["session_id"], "session_id"),
            "camera_setup_id": _nonempty_text(
                index_row["camera_setup_id"], "camera_setup_id"
            ),
            "clock_domain_id": _nonempty_text(
                index_row["clock_domain_id"], "clock_domain_id"
            ),
        }
        source_key = (
            str(media["source_group_id"]),
            str(media["source_video_id"]),
            str(media["device_id"]),
            str(media["setup_id"]),
            str(media["stream_epoch"]),
        )
        group_binding = tuple(binding[field] for _, field in _GROUP_FIELDS)
        if source_key in source_bindings:
            raise CameraEpisodeBoundaryEvaluationError(
                "cross-batch physical source identity is duplicated"
            )
        source_bindings[source_key] = group_binding
        source_video_id = str(media["source_video_id"])
        source_video_binding = (*source_key, *group_binding)
        prior_source = source_video_bindings.setdefault(
            source_video_id, source_video_binding
        )
        if prior_source != source_video_binding:
            raise CameraEpisodeBoundaryEvaluationError(
                "source_video_id identity drifted across batch rows"
            )
        for proposal in proposal_rows:
            bound = {**proposal, **binding}
            proposal_id = str(bound["proposal_id"])
            if proposal_id in proposal_ids:
                raise CameraEpisodeBoundaryEvaluationError(
                    "duplicate proposal_id across boundary evaluation batches"
                )
            proposal_ids.add(proposal_id)
            physical = (
                *_scope_key(bound),
                float(bound["start_sec"]),
                float(bound["end_sec_exclusive"]),
            )
            if physical in proposal_identities:
                raise CameraEpisodeBoundaryEvaluationError(
                    "duplicate physical proposal identity across batches"
                )
            proposal_identities.add(physical)
            proposals.append(bound)
        for boundary in boundary_rows:
            bound = {
                **boundary,
                "source_group_id": media["source_group_id"],
                "device_id": media["device_id"],
                "setup_id": media["setup_id"],
                "stream_epoch": media["stream_epoch"],
                "track_id": boundary["target_track_id"],
                "duration_sec": (
                    float(boundary["end_sec_exclusive"])
                    - float(boundary["start_sec"])
                ),
                **binding,
            }
            episode_id = str(bound["episode_id"])
            if episode_id in episode_ids:
                raise CameraEpisodeBoundaryEvaluationError(
                    "duplicate episode_id across boundary evaluation batches"
                )
            episode_ids.add(episode_id)
            physical = (
                *_scope_key(bound),
                float(bound["start_sec"]),
                float(bound["end_sec_exclusive"]),
            )
            if physical in truth_identities:
                raise CameraEpisodeBoundaryEvaluationError(
                    "duplicate physical human boundary identity across batches"
                )
            truth_identities.add(physical)
            if bound["boundary_status"] == "ready":
                key = _scope_key(bound)
                interval = _interval(bound)
                if any(
                    interval[0] < prior[1] and interval[1] > prior[0]
                    for prior in ready_intervals.setdefault(key, [])
                ):
                    raise CameraEpisodeBoundaryEvaluationError(
                        "overlapping ready human boundaries are forbidden"
                    )
                ready_intervals[key].append(interval)
            boundaries.append(bound)
        coverage_totals["bundle_count"] += 1
        for field in (
            "technical_segment_count",
            "technical_hard_break_segment_count",
            "source_observation_count",
            "accepted_observation_count",
            "source_bucket_count",
            "observed_bucket_count",
        ):
            coverage_totals[field] += int(summary[field])
        validation_scopes.add(str(summary["validation_scope"]))
    if len(truth_sources) != 1:
        raise CameraEpisodeBoundaryEvaluationError(
            "one evaluation bundle cannot mix truth_source values"
        )
    source_observations = coverage_totals["source_observation_count"]
    accepted_observations = coverage_totals["accepted_observation_count"]
    source_buckets = coverage_totals["source_bucket_count"]
    observed_buckets = coverage_totals["observed_bucket_count"]
    coverage = {
        **dict(coverage_totals),
        "accepted_observation_coverage_ratio": (
            accepted_observations / source_observations
            if source_observations
            else _NOT_COMPUTABLE
        ),
        "observed_bucket_coverage_ratio": (
            observed_buckets / source_buckets
            if source_buckets
            else _NOT_COMPUTABLE
        ),
        "validation_scopes": sorted(validation_scopes),
    }
    return _LoadedInputs(
        proposals=tuple(sorted(proposals, key=_entity_sort_key)),
        boundaries=tuple(sorted(boundaries, key=_entity_sort_key)),
        coverage=coverage,
        truth_source=next(iter(truth_sources)),
        batch_count=len(index_rows),
        input_snapshots=snapshots,
    )


def _load_media_sidecar(
    path: Path,
    camera_config: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_json(path, "media sidecar")
    try:
        return validate_media_sidecar(value, camera_config)
    except CameraAdapterError as exc:
        raise CameraEpisodeBoundaryEvaluationError(str(exc)) from exc


def _validate_proposal_bundle(
    summary_value: Mapping[str, Any],
    proposal_values: Sequence[Mapping[str, Any]],
    diagnostic_values: Sequence[Mapping[str, Any]],
    *,
    media: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _validate_proposal_summary(summary_value, media=media)
    proposals = [
        _validate_proposal_row(row, media=media, producer_config_id=summary["producer_config_id"])
        for row in proposal_values
    ]
    diagnostics = [
        _validate_proposal_diagnostic(row, media=media)
        for row in diagnostic_values
    ]
    if not diagnostics:
        raise CameraEpisodeBoundaryEvaluationError("proposal diagnostics are empty")
    _unique_text_ids(proposals, "proposal_id")
    intervals: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for row in sorted(proposals, key=_raw_proposal_sort_key):
        key = (
            str(row["source_group_id"]),
            str(row["source_video_id"]),
            str(row["device_id"]),
            str(row["setup_id"]),
            str(row["stream_epoch"]),
            int(row["track_id"]),
        )
        interval = _interval(row)
        if any(
            interval[0] < prior[1] and interval[1] > prior[0]
            for prior in intervals.setdefault(key, [])
        ):
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal intervals overlap within one track scope"
            )
        intervals[key].append(interval)
    diagnostic_keys: set[tuple[Any, ...]] = set()
    proposals_by_segment: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for proposal in proposals:
        key = _technical_segment_key(proposal)
        proposals_by_segment.setdefault(key, []).append(proposal)
    for diagnostic in diagnostics:
        key = _technical_segment_key(diagnostic)
        if key in diagnostic_keys:
            raise CameraEpisodeBoundaryEvaluationError(
                "duplicate proposal diagnostic technical segment"
            )
        diagnostic_keys.add(key)
        expected = sorted(
            str(row["proposal_id"]) for row in proposals_by_segment.get(key, [])
        )
        if sorted(diagnostic["proposal_ids"]) != expected:
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal diagnostic IDs do not match proposal rows"
            )
        expected_counts = Counter(
            str(row["proposal_status"]) for row in proposals_by_segment.get(key, [])
        )
        observed_counts = {
            name: int(count)
            for name, count in diagnostic["proposal_status_counts"].items()
        }
        if observed_counts != {
            name: expected_counts[name]
            for name in PROPOSAL_STATUSES
            if expected_counts[name]
        }:
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal diagnostic status counts drifted"
            )
        if not expected and not {
            "stationary_track_no_episode",
            "below_trusted_confidence",
            "insufficient_trusted_buckets",
        }.intersection(diagnostic["reason_codes"]):
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal-free diagnostic has no trusted no-episode reason"
            )
    if not set(proposals_by_segment).issubset(diagnostic_keys):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal technical segment has no diagnostic"
        )
    _validate_proposal_summary_counts(summary, proposals, diagnostics)
    return (
        sorted(proposals, key=_raw_proposal_sort_key),
        sorted(diagnostics, key=_diagnostic_input_sort_key),
    )


def _validate_proposal_summary(
    value: Mapping[str, Any],
    *,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if fields not in {
        _PROPOSAL_SUMMARY_FIELDS,
        _PROPOSAL_SUMMARY_FIELDS | _PROPOSAL_DEVELOPMENT_SUMMARY_FIELDS,
    }:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary fields must match S0 v1 or its declared development extension"
        )
    row = dict(value)
    if (
        row["schema_version"]
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION
        or row["status"]
        != "wandering_m0cam_ep2a_s0_boundary_proposal_generated"
        or row["proposal_schema_version"]
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or row["accepted_boundary_schema_version"]
        != CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary schema or status is invalid"
        )
    _validate_media_identity(row, media, role="proposal summary")
    expected_scope = validation_scope_for_authorization_status(
        str(media["authorization_status"])
    )
    if row["validation_scope"] != expected_scope:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary validation scope differs from media"
        )
    _nonempty_text(row["producer_config_id"], "producer_config_id")
    if _PROPOSAL_DEVELOPMENT_SUMMARY_FIELDS.issubset(row):
        minimum_track_confidence = _finite_nonnegative(
            row["minimum_track_confidence"], "minimum_track_confidence"
        )
        movement_threshold = _finite_nonnegative(
            row["movement_evidence_threshold_body_heights"],
            "movement_evidence_threshold_body_heights",
        )
        if (
            not 0.25 <= minimum_track_confidence <= 0.99
            or not 0.001 <= movement_threshold <= 0.25
            or row["segmenter_strategy"]
            not in {
                "home_recall_hysteresis",
                "legacy_path_displacement_state_machine",
            }
        ):
            raise CameraEpisodeBoundaryEvaluationError(
                "proposal development trust fields are invalid"
            )
    for field in (
        "proposal_count",
        "locomotion_proposal_count",
        "scope_count",
        "technical_segment_count",
        "technical_hard_break_segment_count",
        "source_observation_count",
        "accepted_observation_count",
        "source_bucket_count",
        "observed_bucket_count",
        "uncertain_count",
        "manual_review_required_count",
    ):
        row[field] = _nonnegative_int(row[field], field)
    for field in (
        "accepted_observation_coverage_ratio",
        "observed_bucket_coverage_ratio",
        "technical_segment_locomotion_coverage_ratio",
    ):
        row[field] = _finite_ratio(row[field], field)
    row["proposal_duration_seconds"] = _finite_nonnegative(
        row["proposal_duration_seconds"], "proposal_duration_seconds"
    )
    row["stationary_dwell_candidate_seconds"] = _finite_nonnegative(
        row["stationary_dwell_candidate_seconds"],
        "stationary_dwell_candidate_seconds",
    )
    if row["proposal_status_counts"] != {
        name: _nonnegative_int(row["proposal_status_counts"].get(name), name)
        for name in PROPOSAL_STATUSES
    }:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary status counts are invalid"
        )
    if not isinstance(row["reason_counts"], Mapping) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in row["reason_counts"].items()
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary reason_counts are invalid"
        )
    required_false = (
        "proposal_is_accepted_boundary",
        "truth_labels_consumed_by_producer",
        "shape_predictions_consumed_by_producer",
        "frozen_model_loaded",
        "models_trained_or_updated",
        "automatic_boundary_validated",
        "boundary_metrics_available",
        "shape_metrics_available",
    )
    if any(row[field] is not False for field in required_false):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary evidence-boundary flags are invalid"
        )
    if any(
        not isinstance(row[field], bool)
        for field in (
            "stationary_dwell_candidate_validated",
            "movement_start_candidate_validated",
        )
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal development-validation flags are invalid"
        )
    if row["manual_acceptance_required"] is not True:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary must require manual acceptance"
        )
    return row


def _validate_proposal_row(
    value: Mapping[str, Any],
    *,
    media: Mapping[str, Any],
    producer_config_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _PROPOSAL_FIELDS:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal fields must match the exact S0 v1 set"
        )
    row = dict(value)
    if row["schema_version"] != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal schema_version is invalid"
        )
    _validate_media_identity(row, media, role="proposal")
    row["proposal_id"] = _nonempty_text(row["proposal_id"], "proposal_id")
    row["track_id"] = _nonnegative_int(row["track_id"], "track_id")
    row["technical_segment_index"] = _nonnegative_int(
        row["technical_segment_index"], "technical_segment_index"
    )
    row["start_sec"] = _finite_nonnegative(row["start_sec"], "start_sec")
    row["end_sec_exclusive"] = _finite_nonnegative(
        row["end_sec_exclusive"], "end_sec_exclusive"
    )
    if not (
        row["start_sec"]
        < row["end_sec_exclusive"]
        <= float(media["duration_sec"]) + 1e-9
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal interval is outside media"
        )
    row["end_sec_exclusive"] = min(
        row["end_sec_exclusive"], float(media["duration_sec"])
    )
    row["duration_sec"] = _finite_nonnegative(
        row["duration_sec"], "duration_sec"
    )
    if not math.isclose(
        row["duration_sec"],
        row["end_sec_exclusive"] - row["start_sec"],
        abs_tol=1e-9,
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal duration is inconsistent"
        )
    status = row["proposal_status"]
    if not isinstance(status, str) or status not in PROPOSAL_STATUSES:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal_status is invalid"
        )
    row["start_reason"] = _nonempty_text(
        row["start_reason"], "start_reason"
    )
    row["end_reason"] = _nonempty_text(row["end_reason"], "end_reason")
    row["reason_codes"] = _string_list(row["reason_codes"], "reason_codes")
    row["hard_break_reasons"] = _string_list(
        row["hard_break_reasons"], "hard_break_reasons"
    )
    if not isinstance(row["uncertain"], bool) or row["uncertain"] != (
        status != "proposed"
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal uncertain flag is inconsistent with status"
        )
    if row["manual_review_required"] is not True:
        raise CameraEpisodeBoundaryEvaluationError(
            "every proposal must require manual review"
        )
    for field in (
        "source_observation_count",
        "accepted_observation_count",
        "source_bucket_count",
    ):
        row[field] = _nonnegative_int(row[field], field)
    if row["accepted_observation_count"] > row["source_observation_count"]:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal accepted observations exceed source observations"
        )
    if row["producer_config_id"] != producer_config_id:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal producer identity differs from summary"
        )
    return row


def _validate_proposal_diagnostic(
    value: Mapping[str, Any],
    *,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _PROPOSAL_DIAGNOSTIC_FIELDS:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal diagnostic fields must match the exact S0 v1 set"
        )
    row = dict(value)
    if (
        row["schema_version"]
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal diagnostic schema_version is invalid"
        )
    _validate_media_identity(row, media, role="proposal diagnostic")
    row["track_id"] = _nonnegative_int(row["track_id"], "track_id")
    row["technical_segment_index"] = _nonnegative_int(
        row["technical_segment_index"], "technical_segment_index"
    )
    start = _finite_nonnegative(
        row["segment_start_observation_sec"],
        "segment_start_observation_sec",
    )
    end = _finite_nonnegative(
        row["segment_end_observation_sec"], "segment_end_observation_sec"
    )
    if start > end or end > float(media["duration_sec"]) + 1e-9:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal diagnostic observation interval is invalid"
        )
    row["segment_start_observation_sec"] = start
    row["segment_end_observation_sec"] = min(
        end, float(media["duration_sec"])
    )
    for field in (
        "source_observation_count",
        "accepted_observation_count",
        "source_bucket_count",
        "observed_bucket_count",
    ):
        row[field] = _nonnegative_int(row[field], field)
    if row["accepted_observation_count"] > row["source_observation_count"]:
        raise CameraEpisodeBoundaryEvaluationError(
            "diagnostic accepted observations exceed source observations"
        )
    if row["observed_bucket_count"] > row["source_bucket_count"]:
        raise CameraEpisodeBoundaryEvaluationError(
            "diagnostic observed buckets exceed source buckets"
        )
    row["accepted_observation_coverage_ratio"] = _finite_ratio(
        row["accepted_observation_coverage_ratio"],
        "accepted_observation_coverage_ratio",
    )
    expected_coverage = (
        row["accepted_observation_count"] / row["source_observation_count"]
        if row["source_observation_count"]
        else 0.0
    )
    if not math.isclose(
        row["accepted_observation_coverage_ratio"],
        expected_coverage,
        abs_tol=1e-9,
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "diagnostic accepted-observation coverage is inconsistent"
        )
    row["hard_break_reasons"] = _string_list(
        row["hard_break_reasons"], "hard_break_reasons"
    )
    row["state_sequence"] = _string_list(
        row["state_sequence"], "state_sequence"
    )
    row["proposal_ids"] = _string_list(row["proposal_ids"], "proposal_ids")
    if len(row["proposal_ids"]) != len(set(row["proposal_ids"])):
        raise CameraEpisodeBoundaryEvaluationError(
            "duplicate proposal_id in diagnostic"
        )
    counts = row["proposal_status_counts"]
    if not isinstance(counts, Mapping) or any(
        name not in PROPOSAL_STATUSES
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for name, count in counts.items()
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal diagnostic status counts are invalid"
        )
    row["reason_codes"] = _string_list(row["reason_codes"], "reason_codes")
    if row["manual_review_required"] is not bool(row["proposal_ids"]):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal diagnostic manual-review flag is inconsistent"
        )
    return row


def _validate_proposal_summary_counts(
    summary: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    status_counts = Counter(str(row["proposal_status"]) for row in proposals)
    reason_counts = Counter(
        str(reason) for row in proposals for reason in row["reason_codes"]
    )
    expected = {
        "proposal_count": len(proposals),
        "locomotion_proposal_count": sum(
            row["proposal_status"] != "rejected_by_qc" for row in proposals
        ),
        "scope_count": len(
            {
                (
                    row["source_group_id"],
                    row["source_video_id"],
                    row["device_id"],
                    row["setup_id"],
                    row["stream_epoch"],
                    row["track_id"],
                )
                for row in diagnostics
            }
        ),
        "technical_segment_count": len(diagnostics),
        "technical_hard_break_segment_count": sum(
            bool(row["hard_break_reasons"]) for row in diagnostics
        ),
        "accepted_observation_count": sum(
            int(row["accepted_observation_count"]) for row in diagnostics
        ),
        "source_bucket_count": sum(
            int(row["source_bucket_count"]) for row in diagnostics
        ),
        "observed_bucket_count": sum(
            int(row["observed_bucket_count"]) for row in diagnostics
        ),
        "uncertain_count": status_counts["uncertain"],
        "manual_review_required_count": sum(
            bool(row["manual_review_required"]) for row in proposals
        ),
    }
    for field, value in expected.items():
        if int(summary[field]) != value:
            raise CameraEpisodeBoundaryEvaluationError(
                f"proposal summary {field} does not match rows"
            )
    diagnostic_source_observation_count = sum(
        int(row["source_observation_count"]) for row in diagnostics
    )
    if int(summary["source_observation_count"]) < diagnostic_source_observation_count:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary source observations are below diagnostic rows"
        )
    if summary["proposal_status_counts"] != {
        name: status_counts[name] for name in PROPOSAL_STATUSES
    }:
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary status counts do not match rows"
        )
    if dict(summary["reason_counts"]) != dict(sorted(reason_counts.items())):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary reason counts do not match rows"
        )
    proposal_duration = sum(
        float(row["duration_sec"])
        for row in proposals
        if row["proposal_status"] != "rejected_by_qc"
    )
    if not math.isclose(
        float(summary["proposal_duration_seconds"]),
        proposal_duration,
        abs_tol=1e-9,
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            "proposal summary duration does not match rows"
        )
    accepted_coverage = (
        expected["accepted_observation_count"]
        / int(summary["source_observation_count"])
        if int(summary["source_observation_count"])
        else 0.0
    )
    bucket_coverage = (
        expected["observed_bucket_count"] / expected["source_bucket_count"]
        if expected["source_bucket_count"]
        else 0.0
    )
    locomotion_segment_coverage = (
        sum(
            any(
                proposal["proposal_status"] != "rejected_by_qc"
                for proposal in proposals
                if _technical_segment_key(proposal)
                == _technical_segment_key(diagnostic)
            )
            for diagnostic in diagnostics
        )
        / len(diagnostics)
        if diagnostics
        else 0.0
    )
    for field, expected_value in (
        ("accepted_observation_coverage_ratio", accepted_coverage),
        ("observed_bucket_coverage_ratio", bucket_coverage),
        (
            "technical_segment_locomotion_coverage_ratio",
            locomotion_segment_coverage,
        ),
    ):
        if not math.isclose(float(summary[field]), expected_value, abs_tol=1e-9):
            raise CameraEpisodeBoundaryEvaluationError(
                f"proposal summary {field} is inconsistent"
            )


def _validate_bound_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound proposal must be a mapping"
        )
    required = {
        "proposal_id",
        "proposal_status",
        "technical_segment_index",
        "start_sec",
        "end_sec_exclusive",
        "duration_sec",
        "reason_codes",
        "hard_break_reasons",
        "manual_review_required",
        *EPISODE_SCOPE_FIELDS,
    }
    if not required.issubset(value):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound proposal fields are incomplete"
        )
    row = dict(value)
    row["proposal_id"] = _nonempty_text(row["proposal_id"], "proposal_id")
    status = row["proposal_status"]
    if not isinstance(status, str) or status not in PROPOSAL_STATUSES:
        raise CameraEpisodeBoundaryEvaluationError(
            "bound proposal status is invalid"
        )
    row["technical_segment_index"] = _nonnegative_int(
        row["technical_segment_index"], "technical_segment_index"
    )
    _validate_full_scope(row)
    start, end = _interval(row)
    row["start_sec"] = start
    row["end_sec_exclusive"] = end
    row["duration_sec"] = _finite_nonnegative(
        row["duration_sec"], "proposal duration_sec"
    )
    if not math.isclose(row["duration_sec"], end - start, abs_tol=1e-9):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound proposal duration is inconsistent"
        )
    row["reason_codes"] = _string_list(row["reason_codes"], "reason_codes")
    row["hard_break_reasons"] = _string_list(
        row["hard_break_reasons"], "hard_break_reasons"
    )
    if row["manual_review_required"] is not True:
        raise CameraEpisodeBoundaryEvaluationError(
            "bound proposal must require manual review"
        )
    return row


def _validate_bound_boundary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound human boundary must be a mapping"
        )
    required = {
        "episode_id",
        "boundary_status",
        "boundary_reason_codes",
        "start_sec",
        "end_sec_exclusive",
        "duration_sec",
        *EPISODE_SCOPE_FIELDS,
    }
    if not required.issubset(value):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound human boundary fields are incomplete"
        )
    row = dict(value)
    row["episode_id"] = _nonempty_text(row["episode_id"], "episode_id")
    status = row["boundary_status"]
    if status not in {"ready", "boundary_uncertain"}:
        raise CameraEpisodeBoundaryEvaluationError(
            "bound human boundary status is invalid"
        )
    _validate_full_scope(row)
    start, end = _interval(row)
    row["start_sec"] = start
    row["end_sec_exclusive"] = end
    row["duration_sec"] = _finite_nonnegative(
        row["duration_sec"], "boundary duration_sec"
    )
    if not math.isclose(row["duration_sec"], end - start, abs_tol=1e-9):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound human boundary duration is inconsistent"
        )
    row["boundary_reason_codes"] = _string_list(
        row["boundary_reason_codes"], "boundary_reason_codes"
    )
    if (status == "ready") != (not row["boundary_reason_codes"]):
        raise CameraEpisodeBoundaryEvaluationError(
            "bound human boundary reasons are inconsistent"
        )
    return row


def _validate_full_scope(row: Mapping[str, Any]) -> None:
    for field in EPISODE_SCOPE_FIELDS:
        if field == "track_id":
            _nonnegative_int(row[field], field)
        else:
            _nonempty_text(row[field], field)


def _validate_media_identity(
    row: Mapping[str, Any],
    media: Mapping[str, Any],
    *,
    role: str,
) -> None:
    for field in (
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
    ):
        if row.get(field) != media.get(field):
            raise CameraEpisodeBoundaryEvaluationError(
                f"{role} identity differs from media sidecar"
            )


def _truth_from_unmatched(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
        "start_sec": row["start_sec"],
        "end_sec_exclusive": row["end_sec_exclusive"],
        "duration_sec": row["duration_sec"],
        "boundary_status": row["boundary_status"],
        "boundary_reason_codes": list(row["boundary_reason_codes"]),
    }


def _proposal_from_unmatched(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": row["proposal_id"],
        **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
        "start_sec": row["start_sec"],
        "end_sec_exclusive": row["end_sec_exclusive"],
        "duration_sec": row["duration_sec"],
        "proposal_status": row["proposal_status"],
        "reason_codes": list(row["reason_codes"]),
        "hard_break_reasons": list(row["hard_break_reasons"]),
        "manual_review_required": bool(row["manual_review_required"]),
    }


def _pair_passes_policy(
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    if not _positive_scoped_overlap(proposal, truth):
        return False
    p_start, _ = _interval(proposal)
    t_start, _ = _interval(truth)
    return (
        _temporal_iou(proposal, truth)
        >= float(policy["minimum_temporal_iou"])
        and abs(p_start - t_start)
        <= float(policy["maximum_onset_delta_sec"])
    )


def _temporal_iou(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    left_start, left_end = _interval(left)
    right_start, right_end = _interval(right)
    overlap = max(
        0.0, min(left_end, right_end) - max(left_start, right_start)
    )
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union > 0.0 else 0.0


def _positive_scoped_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if _scope_key(left) != _scope_key(right):
        return False
    left_start, left_end = _interval(left)
    right_start, right_end = _interval(right)
    return min(left_end, right_end) > max(left_start, right_start)


def _scope_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in EPISODE_SCOPE_FIELDS)


def _technical_segment_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_group_id"]),
        str(row["source_video_id"]),
        str(row["device_id"]),
        str(row["setup_id"]),
        str(row["stream_epoch"]),
        int(row["track_id"]),
        int(row["technical_segment_index"]),
    )


def _duration_band(duration: float, config: Mapping[str, Any]) -> str:
    for band in config["duration_bands"]:
        maximum = band["maximum_sec_exclusive"]
        if duration >= float(band["minimum_sec"]) and (
            maximum is None or duration < float(maximum)
        ):
            return str(band["name"])
    raise CameraEpisodeBoundaryEvaluationError(
        "duration does not fit a configured band"
    )


def _build_output_summary(
    loaded: _LoadedInputs,
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ready_count = sum(
        row["boundary_status"] == "ready" for row in loaded.boundaries
    )
    uncertain_count = len(loaded.boundaries) - ready_count
    main_matches = sum(
        row["view"] == "all_locomotion_candidates"
        for row in artifacts["matches"]
    )
    evaluation_completed = (
        loaded.truth_source == "independent_human"
        and config["matching_policy"]["validated"] is True
    )
    if loaded.truth_source == "synthetic_fixture":
        evidence_scope = "synthetic_contract_only"
    elif evaluation_completed:
        evidence_scope = "authorized_labeled_development_evaluated"
    else:
        evidence_scope = "development_unfrozen_boundary_diagnostic"
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_SUMMARY_SCHEMA_VERSION,
        "status": "wandering_m0cam_ep2a_s1a_boundary_evaluator_implemented",
        "study_design": (
            "frozen_policy_independent_human_boundary_evaluation"
            if evaluation_completed
            else "implementation_only"
        ),
        "evidence_scope": evidence_scope,
        "truth_source": loaded.truth_source,
        "batch_count": loaded.batch_count,
        "proposal_schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "human_boundary_schema_version": CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
        "matching_policy_id": config["matching_policy"]["policy_id"],
        "matching_policy_validated": config["matching_policy"]["validated"],
        "failure_thresholds_validated": config["failure_thresholds"]["validated"],
        "ready_truth_count": ready_count,
        "boundary_uncertain_count": uncertain_count,
        "proposal_count": len(loaded.proposals),
        "all_locomotion_candidate_match_count": main_matches,
        "failure_count": len(artifacts["failures"]),
        "primary_view": "all_locomotion_candidates",
        "conditional_view": "proposed_only_conditional",
        "proposal_modified": False,
        "human_boundary_modified": False,
        "shape_predictions_consumed": False,
        "shape_truth_consumed": False,
        "frozen_model_loaded": False,
        "automatic_shape_evaluated": False,
        "models_trained_or_updated": False,
        "automatic_boundary_evaluation_completed": evaluation_completed,
        "m0cam_ep2a_s1a_implementation_status": "completed",
        "m0cam_ep2a_human_boundary_status": (
            "pending"
            if loaded.truth_source == "synthetic_fixture"
            else "available_reviewed"
        ),
        "m0cam_ep2a_evaluation_status": (
            "boundary_evaluated_shape_pending"
            if evaluation_completed
            else "pending"
        ),
        "m0cam_ep2a_data_status": evidence_scope,
        "m0cam_ep2a_status": "not_completed",
    }


def _readme_bytes(
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> bytes:
    main = metrics["views"]["all_locomotion_candidates"]["detection"]
    conditional = metrics["views"]["proposed_only_conditional"]["detection"]
    introduction = (
        "This bundle evaluates frozen automatic-boundary proposals against "
        "independent human boundaries. Cohort interpretation remains external "
        "to this bundle and must be reported separately."
        if summary["automatic_boundary_evaluation_completed"]
        else "This S1A bundle evaluates a boundary-only contract. Synthetic metrics "
        "exercise software behavior and are not automatic-boundary performance."
    )
    metric_label = (
        "all-candidate TP/candidate/truth"
        if summary["automatic_boundary_evaluation_completed"]
        else "all-candidate synthetic TP/candidate/truth"
    )
    text = (
        "# Camera episode boundary evaluation\n\n"
        f"{introduction}\n\n"
        f"- evidence_scope: {summary['evidence_scope']}\n"
        f"- truth_source: {summary['truth_source']}\n"
        f"- matching_policy_id: {summary['matching_policy_id']}\n"
        f"- matching_policy_validated: {str(summary['matching_policy_validated']).lower()}\n"
        f"- ready/boundary_uncertain truth: "
        f"{summary['ready_truth_count']}/{summary['boundary_uncertain_count']}\n"
        f"- proposals: {summary['proposal_count']}\n"
        f"- {metric_label}: "
        f"{main['matched_count']}/{main['candidate_support']}/{main['ready_truth_support']}\n"
        f"- proposed-only synthetic TP/candidate/truth: "
        f"{conditional['matched_count']}/{conditional['candidate_support']}/"
        f"{conditional['ready_truth_support']}\n"
        "- shape inference: not run\n"
        "- proposal or human-boundary modification: not performed\n"
        "- M0-CAM-EP2A status: not_completed\n"
    )
    return text.encode("utf-8")


def _assert_input_snapshots_unchanged(snapshots: Mapping[Path, bytes]) -> None:
    for path, expected in snapshots.items():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise CameraEpisodeBoundaryEvaluationError(
                "boundary evaluation input disappeared during evaluation"
            ) from exc
        if observed != expected:
            raise CameraEpisodeBoundaryEvaluationError(
                "boundary evaluation input changed during evaluation"
            )


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera boundary evaluation output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(
                f"camera boundary evaluation output exists: {output}"
            )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        CameraEpisodeBoundaryEvaluationError,
    ) as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            f"cannot parse {role} as strict finite JSON"
        ) from exc
    if not isinstance(value, dict):
        raise CameraEpisodeBoundaryEvaluationError(
            f"{role} must be a JSON object"
        )
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            f"cannot read {role}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (
            json.JSONDecodeError,
            CameraEpisodeBoundaryEvaluationError,
        ) as exc:
            raise CameraEpisodeBoundaryEvaluationError(
                f"invalid {role} strict finite JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise CameraEpisodeBoundaryEvaluationError(
                f"{role} rows must be JSON objects"
            )
        rows.append(value)
    if not rows:
        raise CameraEpisodeBoundaryEvaluationError(f"{role} is empty")
    return rows


def _resolve_input(base: Path, value: str) -> Path:
    path = Path(value)
    try:
        return (
            path.resolve(strict=True)
            if path.is_absolute()
            else (base / path).resolve(strict=True)
        )
    except OSError as exc:
        raise CameraEpisodeBoundaryEvaluationError(
            "boundary evaluation input path is inaccessible"
        ) from exc


def _interval(row: Mapping[str, Any]) -> tuple[float, float]:
    start = _finite_nonnegative(row.get("start_sec"), "start_sec")
    end = _finite_nonnegative(row.get("end_sec_exclusive"), "end_sec_exclusive")
    if end <= start:
        raise CameraEpisodeBoundaryEvaluationError(
            "interval must be finite and non-empty"
        )
    return start, end


def _nonempty_text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in "\r\n\0")
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be a non-empty string"
        )
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str)
        or not item
        or any(char in item for char in "\r\n\0")
        for item in value
    ):
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be a string list"
        )
    return list(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be a non-negative integer"
        )
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be a finite non-negative number"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be a finite non-negative number"
        )
    return number


def _finite_ratio(value: Any, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number > 1.0:
        raise CameraEpisodeBoundaryEvaluationError(
            f"{field} must be within [0,1]"
        )
    return number


def _unique_text_ids(rows: Sequence[Mapping[str, Any]], field: str) -> None:
    values = [_nonempty_text(row.get(field), field) for row in rows]
    if len(values) != len(set(values)):
        raise CameraEpisodeBoundaryEvaluationError(f"duplicate {field}")


def _stable_row_id(prefix: str, value: Mapping[str, Any]) -> str:
    return prefix + "-" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise CameraEpisodeBoundaryEvaluationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _entity_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    identifier = row.get("proposal_id", row.get("episode_id", ""))
    return (
        *tuple(
            str(row[field])
            for field in EPISODE_SCOPE_FIELDS
            if field != "track_id"
        ),
        int(row["track_id"]),
        float(row["start_sec"]),
        float(row["end_sec_exclusive"]),
        str(identifier),
    )


def _raw_proposal_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_group_id"]),
        str(row["source_video_id"]),
        str(row["device_id"]),
        str(row["setup_id"]),
        str(row["stream_epoch"]),
        int(row["track_id"]),
        float(row["start_sec"]),
        float(row["end_sec_exclusive"]),
        str(row["proposal_id"]),
    )


def _diagnostic_input_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_group_id"]),
        str(row["source_video_id"]),
        str(row["device_id"]),
        str(row["setup_id"]),
        str(row["stream_epoch"]),
        int(row["track_id"]),
        int(row["technical_segment_index"]),
    )


def _match_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _VIEW_ORDER.index(str(row["view"])),
        *_entity_sort_key(
            {
                **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
                "start_sec": row["proposal_start_sec"],
                "end_sec_exclusive": row["proposal_end_sec_exclusive"],
                "proposal_id": row["proposal_id"],
            }
        ),
        str(row["episode_id"]),
    )


def _unmatched_truth_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    view_rank = (
        _VIEW_ORDER.index(str(row["view"]))
        if row["view"] in _VIEW_ORDER
        else len(_VIEW_ORDER)
    )
    return (
        view_rank,
        *_entity_sort_key(row),
    )


def _unmatched_proposal_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    view_rank = (
        _VIEW_ORDER.index(str(row["view"]))
        if row["view"] in _VIEW_ORDER
        else len(_VIEW_ORDER)
    )
    return (
        view_rank,
        *_entity_sort_key(row),
    )


def _diagnostic_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["diagnostic_type"]),
        *tuple(str(row[field]) for field in EPISODE_SCOPE_FIELDS),
        tuple(row["proposal_ids"]),
        tuple(row["episode_ids"]),
    )


def _failure_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["failure_type"]),
        str(row["view"]),
        *tuple(str(row[field]) for field in EPISODE_SCOPE_FIELDS),
        str(row["proposal_id"]),
        str(row["episode_id"]),
        str(row["failure_id"]),
    )


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_EVAL_CONFIG_SCHEMA_VERSION,
        "purpose": "boundary_evaluator_implementation_only",
        "camera_chain": {"path": "configs/modules/wandering_camera_v1.yaml"},
        "matching_policy": {
            "policy_id": "m0cam-ep2a-s1b-b01-development-frozen-v1",
            "minimum_temporal_iou": 0.25,
            "maximum_onset_delta_sec": 10.0,
            "validated": True,
        },
        "evaluation_views": {
            "all_locomotion_candidates": ["proposed", "uncertain"],
            "proposed_only_conditional": ["proposed"],
        },
        "duration_bands": [
            {
                "name": "short",
                "minimum_sec": 0.0,
                "maximum_sec_exclusive": 15.0,
            },
            {
                "name": "medium",
                "minimum_sec": 15.0,
                "maximum_sec_exclusive": 40.0,
            },
            {
                "name": "long",
                "minimum_sec": 40.0,
                "maximum_sec_exclusive": None,
            },
        ],
        "failure_thresholds": {
            "low_tiou": 0.5,
            "large_onset_error_sec": 2.0,
            "large_offset_error_sec": 2.0,
            "fragmentation_max_duration_sec": 2.0,
            "fragmentation_max_gap_sec": 1.0,
            "validated": False,
        },
        "data_access": {
            "shape_truth": False,
            "shape_predictions": False,
            "frozen_model": False,
            "training": False,
            "sealed_camera": False,
        },
    }


__all__ = [
    "BOUNDARY_EVALUATION_INDEX_FIELDS",
    "CAMERA_EPISODE_BOUNDARY_EVAL_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_BOUNDARY_EVAL_SUMMARY_SCHEMA_VERSION",
    "CameraEpisodeBoundaryEvaluationBuildResult",
    "CameraEpisodeBoundaryEvaluationError",
    "build_camera_episode_boundary_evaluation_bundle",
    "evaluate_camera_episode_boundaries",
    "load_camera_episode_boundary_evaluation_config",
]
