"""Truth-free continuous episode boundary proposal producer.

The producer deliberately stops at manual-review proposals. It consumes no
shape or purpose truth, loads no model, and cannot emit the accepted EP1A
boundary schema.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraObservation,
    WeightedBucket,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    group_observations,
    load_camera_inputs,
    validation_scope_for_authorization_status,
    weighted_bucket_observations,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCError,
    _split_observations,
)


CAMERA_EPISODE_BOUNDARY_PROPOSAL_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-proposal-config-v1"
)
CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-development-profile-v1"
)
CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-proposal-v1"
)
CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-proposal-summary-v1"
)
CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION = (
    "wandering-camera-episode-boundary-proposal-diagnostic-v1"
)
PRODUCER_CONFIG_ID = "m0cam-ep2a-s0-b01-development-frozen-v1"
HOME_RECALL_PROFILE_ID = "home-recall-confidence070-v1"
PROPOSAL_STATUSES = ("proposed", "uncertain", "rejected_by_qc")
TECHNICAL_HARD_BREAK_REASONS = frozenset(
    {
        "long_internal_gap",
        "suspected_id_switch",
        "height_position_discontinuity",
    }
)
_CONFIG_RELATIVE = Path(
    "configs/modules/wandering_camera_episode_boundary_proposal_v1.yaml"
)


class CameraEpisodeBoundaryProposalError(ValueError):
    """Tracking or proposal configuration cannot produce a valid v1 bundle."""


class EpisodeBoundaryState(str, Enum):
    IDLE = "idle"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class CameraEpisodeBoundaryProposalBuildResult:
    output_dir: Path
    proposal_count: int
    locomotion_proposal_count: int
    proposed_count: int
    uncertain_count: int
    rejected_by_qc_count: int


def load_camera_episode_boundary_proposal_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load the fixed lightweight S0 proposal configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeBoundaryProposalError(
            "cannot read camera episode boundary proposal config"
        ) from exc
    if value != _expected_config():
        raise CameraEpisodeBoundaryProposalError(
            "camera episode boundary proposal config fields or values drifted"
        )
    return value


def load_camera_episode_boundary_development_profile(
    path: str | Path,
) -> dict[str, Any]:
    """Load one declared B01+B02 development-search profile."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeBoundaryProposalError(
            "cannot read camera episode boundary development profile"
        ) from exc
    _validate_development_profile(value)
    return value


def propose_camera_episode_boundaries(
    adapter_input: CameraAdapterInput,
    *,
    proposal_config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one isolated state machine per technical camera segment."""

    if not isinstance(adapter_input, CameraAdapterInput):
        raise CameraEpisodeBoundaryProposalError(
            "propose_camera_episode_boundaries expects CameraAdapterInput"
        )
    _validate_config_bindings(proposal_config, camera_config)
    sampling = proposal_config["sampling"]
    threshold = float(sampling["minimum_track_confidence"])
    proposals: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    groups = group_observations(adapter_input.observations)
    for scope_key, observations in groups.items():
        if _uses_home_recall_strategy(proposal_config):
            trusted_observations = tuple(
                item for item in observations if item.track_confidence >= threshold
            )
            if not trusted_observations:
                diagnostics.append(
                    _diagnostic_row(
                        observations,
                        accepted=tuple(),
                        buckets=tuple(),
                        scope_key=scope_key,
                        technical_segment_index=0,
                        hard_break_reasons=tuple(),
                        state_sequence=(
                            EpisodeBoundaryState.IDLE.value,
                            EpisodeBoundaryState.CLOSED.value,
                        ),
                        proposals=tuple(),
                        reason_codes=("below_trusted_confidence",),
                    )
                )
                continue
            observations = trusted_observations
        try:
            segments = _split_observations(
                observations,
                sampling=camera_config["sampling"],
                qc=camera_config["camera_qc"],
            )
        except CameraQCError as exc:
            raise CameraEpisodeBoundaryProposalError(
                "technical camera observation splitting failed"
            ) from exc
        for segment_index, (segment, flags) in enumerate(segments):
            segment_start_bound = 0.0
            segment_end_bound = float(adapter_input.media_sidecar["duration_sec"])
            if segment_index > 0:
                segment_start_bound = _technical_split_time(
                    segments[segment_index - 1][0],
                    segment,
                    minimum_track_confidence=threshold,
                )
            if segment_index + 1 < len(segments):
                segment_end_bound = _technical_split_time(
                    segment,
                    segments[segment_index + 1][0],
                    minimum_track_confidence=threshold,
                )
            hard_break_reasons = sorted(TECHNICAL_HARD_BREAK_REASONS & flags)
            segment_proposals, diagnostic = _propose_segment(
                segment,
                scope_key=scope_key,
                technical_segment_index=segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                hard_break_reasons=hard_break_reasons,
                media=adapter_input.media_sidecar,
                proposal_config=proposal_config,
            )
            proposals.extend(segment_proposals)
            diagnostics.append(diagnostic)
    proposals.sort(key=_proposal_sort_key)
    diagnostics.sort(key=_diagnostic_sort_key)
    _validate_proposal_intervals(proposals)
    accepted_total = sum(
        item.track_confidence >= threshold for item in adapter_input.observations
    )
    if sum(int(row["accepted_observation_count"]) for row in diagnostics) != accepted_total:
        raise CameraEpisodeBoundaryProposalError(
            "technical segment accepted-observation accounting drifted"
        )
    return proposals, diagnostics


def build_camera_episode_boundary_proposal_bundle(
    *,
    project_root: str | Path,
    proposal_config_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeBoundaryProposalBuildResult:
    """Build a fresh proposal-only bundle without truth or model access."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera episode boundary proposal output exists: {output}")
    root = Path(project_root).resolve(strict=True)
    fixed_config = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(proposal_config_path).resolve(strict=False) != fixed_config:
        raise CameraEpisodeBoundaryProposalError(
            "proposal config must use the fixed repository path"
        )
    proposal_config = load_camera_episode_boundary_proposal_config(fixed_config)
    camera_path = (
        root / str(proposal_config["camera_chain"]["path"])
    ).resolve(strict=False)
    if not camera_path.is_file():
        raise CameraEpisodeBoundaryProposalError("camera chain config is missing")
    camera_config = load_camera_config(camera_path)
    _validate_config_bindings(proposal_config, camera_config)
    adapter = load_camera_inputs(
        tracking_jsonl_path,
        media_sidecar_path,
        camera_config,
    )
    proposals, diagnostics = propose_camera_episode_boundaries(
        adapter,
        proposal_config=proposal_config,
        camera_config=camera_config,
    )
    summary = _build_summary(adapter, proposals, diagnostics, proposal_config)
    readme = _bundle_readme(summary)
    _commit_new_directory(
        output,
        {
            "README.md": readme.encode("utf-8"),
            "diagnostics.jsonl": canonical_jsonl_bytes(diagnostics),
            "proposals.jsonl": canonical_jsonl_bytes(proposals),
            "summary.json": canonical_json_bytes(summary),
        },
    )
    status_counts = Counter(str(row["proposal_status"]) for row in proposals)
    return CameraEpisodeBoundaryProposalBuildResult(
        output_dir=output,
        proposal_count=len(proposals),
        locomotion_proposal_count=sum(
            row["proposal_status"] != "rejected_by_qc" for row in proposals
        ),
        proposed_count=status_counts["proposed"],
        uncertain_count=status_counts["uncertain"],
        rejected_by_qc_count=status_counts["rejected_by_qc"],
    )


def build_camera_episode_boundary_development_bundle(
    *,
    project_root: str | Path,
    development_profile_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeBoundaryProposalBuildResult:
    """Build a fresh proposal bundle from one declared development profile."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera episode boundary proposal output exists: {output}")
    root = Path(project_root).resolve(strict=True)
    profile_path = Path(development_profile_path).resolve(strict=True)
    if not profile_path.is_relative_to(root) or not profile_path.is_file():
        raise CameraEpisodeBoundaryProposalError(
            "development profile must be a repository file"
        )
    proposal_config = load_camera_episode_boundary_development_profile(profile_path)
    camera_path = (
        root / str(proposal_config["camera_chain"]["path"])
    ).resolve(strict=False)
    if not camera_path.is_file() or not camera_path.is_relative_to(root):
        raise CameraEpisodeBoundaryProposalError("camera chain config is missing")
    camera_config = load_camera_config(camera_path)
    _validate_config_bindings(proposal_config, camera_config)
    adapter = load_camera_inputs(
        tracking_jsonl_path,
        media_sidecar_path,
        camera_config,
    )
    proposals, diagnostics = propose_camera_episode_boundaries(
        adapter,
        proposal_config=proposal_config,
        camera_config=camera_config,
    )
    summary = _build_summary(adapter, proposals, diagnostics, proposal_config)
    _commit_new_directory(
        output,
        {
            "README.md": _bundle_readme(summary).encode("utf-8"),
            "diagnostics.jsonl": canonical_jsonl_bytes(diagnostics),
            "proposals.jsonl": canonical_jsonl_bytes(proposals),
            "summary.json": canonical_json_bytes(summary),
        },
    )
    status_counts = Counter(str(row["proposal_status"]) for row in proposals)
    return CameraEpisodeBoundaryProposalBuildResult(
        output_dir=output,
        proposal_count=len(proposals),
        locomotion_proposal_count=sum(
            row["proposal_status"] != "rejected_by_qc" for row in proposals
        ),
        proposed_count=status_counts["proposed"],
        uncertain_count=status_counts["uncertain"],
        rejected_by_qc_count=status_counts["rejected_by_qc"],
    )


def _propose_segment(
    segment: Sequence[CameraObservation],
    *,
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    segment_start_bound: float,
    segment_end_bound: float,
    hard_break_reasons: Sequence[str],
    media: Mapping[str, Any],
    proposal_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _uses_home_recall_strategy(proposal_config):
        return _propose_home_recall_segment(
            segment,
            scope_key=scope_key,
            technical_segment_index=technical_segment_index,
            segment_start_bound=segment_start_bound,
            segment_end_bound=segment_end_bound,
            hard_break_reasons=hard_break_reasons,
            media=media,
            proposal_config=proposal_config,
        )
    sampling = proposal_config["sampling"]
    machine = proposal_config["state_machine"]
    threshold = float(sampling["minimum_track_confidence"])
    bucket_seconds = float(sampling["bucket_seconds"])
    accepted = tuple(item for item in segment if item.track_confidence >= threshold)
    buckets = weighted_bucket_observations(
        segment,
        minimum_track_confidence=threshold,
        bucket_seconds=bucket_seconds,
    )
    state = EpisodeBoundaryState.IDLE
    state_sequence = [state.value]
    proposals: list[dict[str, Any]] = []
    start_index: int | None = None
    start_reason: str | None = None
    stationary_start_index: int | None = None
    stationary_max_drift = 0.0
    movement_candidates: list[int] = []
    movement_min_buckets = int(machine["movement_start_min_buckets"])
    movement_threshold = float(
        machine["movement_start_min_displacement_body_heights"]
    )
    stationary_threshold = float(
        machine["stationary_max_displacement_body_heights"]
    )
    dwell_seconds = float(machine["stationary_dwell_candidate_seconds"])
    producer_config_id = _producer_config_id(proposal_config)

    if media["camera_motion_state"] == "moved" or len(buckets) < movement_min_buckets:
        reasons = [
            "camera_moved"
            if media["camera_motion_state"] == "moved"
            else "insufficient_locomotion_evidence"
        ]
        reasons.extend(hard_break_reasons)
        _transition(state_sequence, EpisodeBoundaryState.UNCERTAIN)
        _transition(state_sequence, EpisodeBoundaryState.CLOSED)
        proposals.append(
            _rejected_segment_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                hard_break_reasons=hard_break_reasons,
                reason_codes=reasons,
                media=media,
                bucket_seconds=bucket_seconds,
                producer_config_id=producer_config_id,
            )
        )
        return proposals, _diagnostic_row(
            segment,
            accepted=accepted,
            buckets=buckets,
            scope_key=scope_key,
            technical_segment_index=technical_segment_index,
            hard_break_reasons=hard_break_reasons,
            state_sequence=state_sequence,
            proposals=proposals,
        )

    for index, bucket in enumerate(buckets):
        if state in {EpisodeBoundaryState.IDLE, EpisodeBoundaryState.CLOSED}:
            if movement_candidates and (
                bucket.bucket_index
                != buckets[movement_candidates[-1]].bucket_index + 1
            ):
                movement_candidates.clear()
            movement_candidates.append(index)
            movement_candidates = movement_candidates[-movement_min_buckets:]
            if len(movement_candidates) < movement_min_buckets:
                continue
            movement = _path_displacement_body_heights(
                [buckets[item] for item in movement_candidates]
            )
            if movement < movement_threshold:
                continue
            start_index = movement_candidates[0]
            start_reason = (
                "sustained_movement_at_track_start"
                if start_index == 0 and not proposals
                else "sustained_movement"
            )
            stationary_start_index = None
            movement_candidates.clear()
            _transition(state_sequence, EpisodeBoundaryState.OPEN)
            state = EpisodeBoundaryState.OPEN
            continue

        previous = buckets[index - 1]
        contiguous = bucket.bucket_index == previous.bucket_index + 1
        step = _body_step(previous, bucket)
        if state == EpisodeBoundaryState.OPEN:
            if contiguous and step <= stationary_threshold:
                stationary_start_index = index - 1
                stationary_max_drift = 0.0
                _transition(state_sequence, EpisodeBoundaryState.CLOSING)
                state = EpisodeBoundaryState.CLOSING
            continue

        if state == EpisodeBoundaryState.CLOSING:
            if stationary_start_index is None:
                raise CameraEpisodeBoundaryProposalError(
                    "closing state lost stationary candidate"
                )
            stationary_drift = _body_step(
                buckets[stationary_start_index], bucket
            )
            stationary_max_drift = max(stationary_max_drift, stationary_drift)
            if (
                not contiguous
                or step > stationary_threshold
                or stationary_max_drift >= movement_threshold
            ):
                stationary_start_index = None
                stationary_max_drift = 0.0
                _transition(state_sequence, EpisodeBoundaryState.OPEN)
                state = EpisodeBoundaryState.OPEN
                continue
            stationary_duration = (
                _bucket_start(bucket, bucket_seconds)
                - _bucket_start(buckets[stationary_start_index], bucket_seconds)
            )
            if stationary_duration + 1e-9 < dwell_seconds:
                continue
            if start_index is None or start_reason is None:
                raise CameraEpisodeBoundaryProposalError("open state lost movement onset")
            stationary_candidate_drift = (
                stationary_max_drift > stationary_threshold + 1e-9
            )
            reason_codes = [
                "sustained_movement",
                "sustained_stationary",
                "protocol_derived_unvalidated_parameters",
            ]
            if stationary_candidate_drift:
                reason_codes.append("stationary_candidate_drift")
            proposal = _locomotion_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                start_sec=_bucket_start(buckets[start_index], bucket_seconds),
                end_sec=_bucket_start(
                    buckets[stationary_start_index], bucket_seconds
                ),
                start_reason=start_reason,
                end_reason="sustained_stationary",
                hard_break_reasons=hard_break_reasons,
                reason_codes=reason_codes,
                media=media,
                proposal_config=proposal_config,
                force_uncertain=(
                    bool(hard_break_reasons) or stationary_candidate_drift
                ),
            )
            proposals.append(proposal)
            _transition(state_sequence, EpisodeBoundaryState.CLOSED)
            state = EpisodeBoundaryState.CLOSED
            start_index = None
            start_reason = None
            stationary_start_index = None
            stationary_max_drift = 0.0
            movement_candidates.clear()

    if state in {EpisodeBoundaryState.OPEN, EpisodeBoundaryState.CLOSING}:
        if start_index is None or start_reason is None:
            raise CameraEpisodeBoundaryProposalError("open segment lost movement onset")
        end_reason = (
            "technical_hard_break" if hard_break_reasons else "track_or_stream_end"
        )
        reasons = [
            "sustained_movement",
            "offset_requires_manual_review",
            "protocol_derived_unvalidated_parameters",
        ]
        reasons.extend(hard_break_reasons or ["track_or_stream_end"])
        _transition(state_sequence, EpisodeBoundaryState.UNCERTAIN)
        _transition(state_sequence, EpisodeBoundaryState.CLOSED)
        proposals.append(
            _locomotion_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                start_sec=_bucket_start(buckets[start_index], bucket_seconds),
                end_sec=min(
                    float(media["duration_sec"]),
                    _bucket_end(buckets[-1], bucket_seconds),
                ),
                start_reason=start_reason,
                end_reason=end_reason,
                hard_break_reasons=hard_break_reasons,
                reason_codes=reasons,
                media=media,
                proposal_config=proposal_config,
                force_uncertain=True,
            )
        )
    elif not proposals:
        _transition(state_sequence, EpisodeBoundaryState.UNCERTAIN)
        _transition(state_sequence, EpisodeBoundaryState.CLOSED)
        proposals.append(
            _rejected_segment_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                hard_break_reasons=hard_break_reasons,
                reason_codes=[
                    "insufficient_locomotion_evidence",
                    *hard_break_reasons,
                ],
                media=media,
                bucket_seconds=bucket_seconds,
                producer_config_id=producer_config_id,
            )
        )
    return proposals, _diagnostic_row(
        segment,
        accepted=accepted,
        buckets=buckets,
        scope_key=scope_key,
        technical_segment_index=technical_segment_index,
        hard_break_reasons=hard_break_reasons,
        state_sequence=state_sequence,
        proposals=proposals,
    )


def _propose_home_recall_segment(
    segment: Sequence[CameraObservation],
    *,
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    segment_start_bound: float,
    segment_end_bound: float,
    hard_break_reasons: Sequence[str],
    media: Mapping[str, Any],
    proposal_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recall-first home-development state machine over trusted observations only."""

    sampling = proposal_config["sampling"]
    machine = proposal_config["state_machine"]
    threshold = float(sampling["minimum_track_confidence"])
    bucket_seconds = float(sampling["bucket_seconds"])
    accepted = tuple(item for item in segment if item.track_confidence >= threshold)
    buckets = weighted_bucket_observations(
        accepted,
        minimum_track_confidence=threshold,
        bucket_seconds=bucket_seconds,
    )
    state = EpisodeBoundaryState.IDLE
    state_sequence = [state.value]
    proposals: list[dict[str, Any]] = []
    movement_candidates: list[int] = []
    start_index: int | None = None
    start_reason: str | None = None
    stationary_start_index: int | None = None
    movement_min_buckets = int(machine["movement_start_min_buckets"])
    movement_threshold = float(
        machine["movement_start_min_displacement_body_heights"]
    )
    stationary_threshold = float(
        machine["stationary_max_displacement_body_heights"]
    )
    dwell_seconds = float(machine["stationary_dwell_candidate_seconds"])

    if media["camera_motion_state"] == "moved":
        _transition(state_sequence, EpisodeBoundaryState.UNCERTAIN)
        _transition(state_sequence, EpisodeBoundaryState.CLOSED)
        proposals.append(
            _rejected_segment_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                hard_break_reasons=hard_break_reasons,
                reason_codes=["camera_moved", *hard_break_reasons],
                media=media,
                bucket_seconds=bucket_seconds,
                producer_config_id=_producer_config_id(proposal_config),
            )
        )
        return proposals, _diagnostic_row(
            segment,
            accepted=accepted,
            buckets=buckets,
            scope_key=scope_key,
            technical_segment_index=technical_segment_index,
            hard_break_reasons=hard_break_reasons,
            state_sequence=state_sequence,
            proposals=proposals,
        )

    for index, bucket in enumerate(buckets):
        if state in {EpisodeBoundaryState.IDLE, EpisodeBoundaryState.CLOSED}:
            if movement_candidates and (
                bucket.bucket_index
                != buckets[movement_candidates[-1]].bucket_index + 1
            ):
                movement_candidates.clear()
            movement_candidates.append(index)
            movement_candidates = movement_candidates[-movement_min_buckets:]
            if len(movement_candidates) < movement_min_buckets:
                continue
            candidate_buckets = [buckets[item] for item in movement_candidates]
            steps = [
                _body_step(left, right)
                for left, right in zip(candidate_buckets, candidate_buckets[1:])
            ]
            active_step_count = sum(
                step > stationary_threshold + 1e-12 for step in steps
            )
            if (
                _path_displacement_body_heights(candidate_buckets)
                < movement_threshold
                or active_step_count < max(1, movement_min_buckets - 1)
            ):
                continue
            start_index = movement_candidates[0]
            start_reason = (
                "sustained_movement_at_track_start"
                if start_index == 0 and not proposals
                else "sustained_movement"
            )
            stationary_start_index = None
            movement_candidates.clear()
            _transition(state_sequence, EpisodeBoundaryState.OPEN)
            state = EpisodeBoundaryState.OPEN
            continue

        previous = buckets[index - 1]
        contiguous = bucket.bucket_index == previous.bucket_index + 1
        step = _body_step(previous, bucket)
        if state == EpisodeBoundaryState.OPEN:
            if contiguous and step <= stationary_threshold:
                stationary_start_index = index - 1
                _transition(state_sequence, EpisodeBoundaryState.CLOSING)
                state = EpisodeBoundaryState.CLOSING
            continue

        if state == EpisodeBoundaryState.CLOSING:
            if stationary_start_index is None:
                raise CameraEpisodeBoundaryProposalError(
                    "home recall closing state lost stationary candidate"
                )
            if not contiguous or step > stationary_threshold:
                stationary_start_index = None
                _transition(state_sequence, EpisodeBoundaryState.OPEN)
                state = EpisodeBoundaryState.OPEN
                continue
            stationary_duration = (
                _bucket_start(bucket, bucket_seconds)
                - _bucket_start(buckets[stationary_start_index], bucket_seconds)
            )
            if stationary_duration + 1e-9 < dwell_seconds:
                continue
            if start_index is None or start_reason is None:
                raise CameraEpisodeBoundaryProposalError(
                    "home recall open state lost movement onset"
                )
            proposals.append(
                _locomotion_row(
                    segment,
                    accepted=accepted,
                    buckets=buckets,
                    scope_key=scope_key,
                    technical_segment_index=technical_segment_index,
                    segment_start_bound=segment_start_bound,
                    segment_end_bound=segment_end_bound,
                    start_sec=_bucket_start(buckets[start_index], bucket_seconds),
                    end_sec=_bucket_start(
                        buckets[stationary_start_index], bucket_seconds
                    ),
                    start_reason=start_reason,
                    end_reason="sustained_stationary",
                    hard_break_reasons=hard_break_reasons,
                    reason_codes=[
                        "sustained_movement",
                        "sustained_stationary",
                        "home_recall_hysteresis",
                    ],
                    media=media,
                    proposal_config=proposal_config,
                    force_uncertain=_home_blocking_hard_break(
                        hard_break_reasons
                    ),
                )
            )
            _transition(state_sequence, EpisodeBoundaryState.CLOSED)
            state = EpisodeBoundaryState.CLOSED
            start_index = None
            start_reason = None
            stationary_start_index = None
            movement_candidates.clear()

    if state in {EpisodeBoundaryState.OPEN, EpisodeBoundaryState.CLOSING}:
        if start_index is None or start_reason is None:
            raise CameraEpisodeBoundaryProposalError(
                "home recall open segment lost movement onset"
            )
        technical_break = _home_blocking_hard_break(hard_break_reasons)
        _transition(
            state_sequence,
            EpisodeBoundaryState.UNCERTAIN
            if technical_break
            else EpisodeBoundaryState.CLOSED,
        )
        if technical_break:
            _transition(state_sequence, EpisodeBoundaryState.CLOSED)
        proposals.append(
            _locomotion_row(
                segment,
                accepted=accepted,
                buckets=buckets,
                scope_key=scope_key,
                technical_segment_index=technical_segment_index,
                segment_start_bound=segment_start_bound,
                segment_end_bound=segment_end_bound,
                start_sec=_bucket_start(buckets[start_index], bucket_seconds),
                end_sec=min(
                    float(media["duration_sec"]),
                    _bucket_end(buckets[-1], bucket_seconds),
                ),
                start_reason=start_reason,
                end_reason=(
                    "technical_hard_break"
                    if technical_break
                    else "track_or_stream_end"
                ),
                hard_break_reasons=hard_break_reasons,
                reason_codes=[
                    "sustained_movement",
                    "home_recall_hysteresis",
                    *(hard_break_reasons or ["stream_end_clipped"]),
                ],
                media=media,
                proposal_config=proposal_config,
                force_uncertain=technical_break,
            )
        )
    elif state == EpisodeBoundaryState.IDLE and not proposals:
        _transition(state_sequence, EpisodeBoundaryState.CLOSED)

    diagnostic_reasons: tuple[str, ...] = tuple()
    if not proposals:
        diagnostic_reasons = (
            "insufficient_trusted_buckets"
            if len(buckets) < movement_min_buckets
            else "stationary_track_no_episode",
        )
    return proposals, _diagnostic_row(
        segment,
        accepted=accepted,
        buckets=buckets,
        scope_key=scope_key,
        technical_segment_index=technical_segment_index,
        hard_break_reasons=hard_break_reasons,
        state_sequence=state_sequence,
        proposals=proposals,
        reason_codes=diagnostic_reasons,
    )


def _locomotion_row(
    segment: Sequence[CameraObservation],
    *,
    accepted: Sequence[CameraObservation],
    buckets: Sequence[WeightedBucket],
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    segment_start_bound: float,
    segment_end_bound: float,
    start_sec: float,
    end_sec: float,
    start_reason: str,
    end_reason: str,
    hard_break_reasons: Sequence[str],
    reason_codes: Sequence[str],
    media: Mapping[str, Any],
    proposal_config: Mapping[str, Any],
    force_uncertain: bool,
) -> dict[str, Any]:
    refined_start, refined_end, refinement_reasons = _refine_locomotion_interval(
        accepted,
        coarse_start_sec=start_sec,
        coarse_end_sec=end_sec,
        segment_start_bound=segment_start_bound,
        segment_end_bound=segment_end_bound,
        proposal_config=proposal_config,
    )
    start_sec = refined_start
    end_sec = refined_end
    duration = end_sec - start_sec
    short = duration + 1e-9 < float(
        proposal_config["state_machine"]["minimum_episode_duration_seconds"]
    )
    uncertain = force_uncertain or short
    reasons = [*reason_codes, *refinement_reasons]
    if short:
        reasons.append("short_episode_candidate")
    return _proposal_row(
        segment,
        accepted=accepted,
        buckets=buckets,
        scope_key=scope_key,
        technical_segment_index=technical_segment_index,
        segment_start_bound=segment_start_bound,
        segment_end_bound=segment_end_bound,
        start_sec=start_sec,
        end_sec=end_sec,
        proposal_status="uncertain" if uncertain else "proposed",
        start_reason=start_reason,
        end_reason=end_reason,
        hard_break_reasons=hard_break_reasons,
        reason_codes=reasons,
        uncertain=uncertain,
        media=media,
        producer_config_id=_producer_config_id(proposal_config),
    )


def _rejected_segment_row(
    segment: Sequence[CameraObservation],
    *,
    accepted: Sequence[CameraObservation],
    buckets: Sequence[WeightedBucket],
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    segment_start_bound: float,
    segment_end_bound: float,
    hard_break_reasons: Sequence[str],
    reason_codes: Sequence[str],
    media: Mapping[str, Any],
    bucket_seconds: float,
    producer_config_id: str,
) -> dict[str, Any]:
    if buckets:
        start_sec = _bucket_start(buckets[0], bucket_seconds)
        end_sec = min(float(media["duration_sec"]), _bucket_end(buckets[-1], bucket_seconds))
    else:
        start_sec = max(0.0, float(segment[0].timestamp_sec) - bucket_seconds / 2.0)
        end_sec = min(
            float(media["duration_sec"]),
            max(start_sec + bucket_seconds, float(segment[-1].timestamp_sec) + bucket_seconds / 2.0),
        )
    return _proposal_row(
        segment,
        accepted=accepted,
        buckets=buckets,
        scope_key=scope_key,
        technical_segment_index=technical_segment_index,
        segment_start_bound=segment_start_bound,
        segment_end_bound=segment_end_bound,
        start_sec=start_sec,
        end_sec=end_sec,
        proposal_status="rejected_by_qc",
        start_reason="no_confirmed_locomotion_start",
        end_reason="insufficient_locomotion_evidence",
        hard_break_reasons=hard_break_reasons,
        reason_codes=reason_codes,
        uncertain=True,
        media=media,
        producer_config_id=producer_config_id,
    )


def _proposal_row(
    segment: Sequence[CameraObservation],
    *,
    accepted: Sequence[CameraObservation],
    buckets: Sequence[WeightedBucket],
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    segment_start_bound: float,
    segment_end_bound: float,
    start_sec: float,
    end_sec: float,
    proposal_status: str,
    start_reason: str,
    end_reason: str,
    hard_break_reasons: Sequence[str],
    reason_codes: Sequence[str],
    uncertain: bool,
    media: Mapping[str, Any],
    producer_config_id: str,
) -> dict[str, Any]:
    if proposal_status not in PROPOSAL_STATUSES:
        raise CameraEpisodeBoundaryProposalError("proposal status is invalid")
    start = max(0.0, float(segment_start_bound), float(start_sec))
    end = min(
        float(media["duration_sec"]),
        float(segment_end_bound),
        float(end_sec),
    )
    if not math.isfinite(start + end) or start >= end:
        raise CameraEpisodeBoundaryProposalError("proposal interval must be positive")
    identity = {
        "scope": list(scope_key),
        "technical_segment_index": technical_segment_index,
        "start_sec": start,
        "end_sec_exclusive": end,
        "proposal_status": proposal_status,
        "producer_config_id": producer_config_id,
    }
    proposal_id = "boundary-proposal-" + hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    source_bucket_count = (
        buckets[-1].bucket_index - buckets[0].bucket_index + 1 if buckets else 0
    )
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "source_group_id": scope_key[0],
        "source_video_id": scope_key[1],
        "device_id": scope_key[2],
        "setup_id": scope_key[3],
        "stream_epoch": scope_key[4],
        "track_id": scope_key[5],
        "technical_segment_index": technical_segment_index,
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "proposal_status": proposal_status,
        "start_reason": start_reason,
        "end_reason": end_reason,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "hard_break_reasons": sorted(set(hard_break_reasons)),
        "uncertain": bool(uncertain),
        "manual_review_required": True,
        "source_observation_count": len(segment),
        "accepted_observation_count": len(accepted),
        "source_bucket_count": source_bucket_count,
        "producer_config_id": producer_config_id,
    }


def _diagnostic_row(
    segment: Sequence[CameraObservation],
    *,
    accepted: Sequence[CameraObservation],
    buckets: Sequence[WeightedBucket],
    scope_key: tuple[str, str, str, str, str, int],
    technical_segment_index: int,
    hard_break_reasons: Sequence[str],
    state_sequence: Sequence[str],
    proposals: Sequence[Mapping[str, Any]],
    reason_codes: Sequence[str] = tuple(),
) -> dict[str, Any]:
    counts = Counter(str(row["proposal_status"]) for row in proposals)
    source_bucket_count = (
        buckets[-1].bucket_index - buckets[0].bucket_index + 1 if buckets else 0
    )
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION,
        "source_group_id": scope_key[0],
        "source_video_id": scope_key[1],
        "device_id": scope_key[2],
        "setup_id": scope_key[3],
        "stream_epoch": scope_key[4],
        "track_id": scope_key[5],
        "technical_segment_index": technical_segment_index,
        "segment_start_observation_sec": float(segment[0].timestamp_sec),
        "segment_end_observation_sec": float(segment[-1].timestamp_sec),
        "source_observation_count": len(segment),
        "accepted_observation_count": len(accepted),
        "source_bucket_count": source_bucket_count,
        "observed_bucket_count": len(buckets),
        "accepted_observation_coverage_ratio": (
            len(accepted) / len(segment) if segment else 0.0
        ),
        "hard_break_reasons": sorted(set(hard_break_reasons)),
        "state_sequence": list(state_sequence),
        "proposal_ids": [str(row["proposal_id"]) for row in proposals],
        "proposal_status_counts": {
            name: counts[name] for name in PROPOSAL_STATUSES if counts[name]
        },
        "reason_codes": sorted(
            {
                str(reason)
                for row in proposals
                for reason in row["reason_codes"]
            }
            | {str(reason) for reason in reason_codes}
        ),
        "manual_review_required": bool(proposals),
    }


def _build_summary(
    adapter: CameraAdapterInput,
    proposals: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    proposal_config: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(str(row["proposal_status"]) for row in proposals)
    reason_counts = Counter(
        str(reason) for row in proposals for reason in row["reason_codes"]
    )
    source_count = len(adapter.observations)
    accepted_count = sum(int(row["accepted_observation_count"]) for row in diagnostics)
    observed_bucket_count = sum(int(row["observed_bucket_count"]) for row in diagnostics)
    source_bucket_count = sum(int(row["source_bucket_count"]) for row in diagnostics)
    locomotion_count = sum(
        row["proposal_status"] != "rejected_by_qc" for row in proposals
    )
    summary = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION,
        "status": "wandering_m0cam_ep2a_s0_boundary_proposal_generated",
        "evidence_scope": "partial_truth_free_smoke_only",
        "source_group_id": adapter.media_sidecar["source_group_id"],
        "source_video_id": adapter.media_sidecar["source_video_id"],
        "device_id": adapter.media_sidecar["device_id"],
        "setup_id": adapter.media_sidecar["setup_id"],
        "stream_epoch": adapter.media_sidecar["stream_epoch"],
        "validation_scope": validation_scope_for_authorization_status(
            str(adapter.media_sidecar["authorization_status"])
        ),
        "producer_config_id": _producer_config_id(proposal_config),
        "proposal_schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "accepted_boundary_schema_version": "wandering-camera-episode-boundary-v1",
        "proposal_is_accepted_boundary": False,
        "proposal_count": len(proposals),
        "locomotion_proposal_count": locomotion_count,
        "proposal_status_counts": {
            name: status_counts[name] for name in PROPOSAL_STATUSES
        },
        "scope_count": len(group_observations(adapter.observations)),
        "technical_segment_count": len(diagnostics),
        "technical_hard_break_segment_count": sum(
            bool(row["hard_break_reasons"]) for row in diagnostics
        ),
        "source_observation_count": source_count,
        "accepted_observation_count": accepted_count,
        "accepted_observation_coverage_ratio": (
            accepted_count / source_count if source_count else 0.0
        ),
        "source_bucket_count": source_bucket_count,
        "observed_bucket_count": observed_bucket_count,
        "observed_bucket_coverage_ratio": (
            observed_bucket_count / source_bucket_count if source_bucket_count else 0.0
        ),
        "technical_segment_locomotion_coverage_ratio": (
            sum(
                any(
                    proposal["proposal_status"] != "rejected_by_qc"
                    for proposal in proposals
                    if proposal["source_group_id"] == diagnostic["source_group_id"]
                    and proposal["source_video_id"] == diagnostic["source_video_id"]
                    and proposal["device_id"] == diagnostic["device_id"]
                    and proposal["setup_id"] == diagnostic["setup_id"]
                    and proposal["stream_epoch"] == diagnostic["stream_epoch"]
                    and proposal["track_id"] == diagnostic["track_id"]
                    and proposal["technical_segment_index"]
                    == diagnostic["technical_segment_index"]
                )
                for diagnostic in diagnostics
            )
            / len(diagnostics)
            if diagnostics
            else 0.0
        ),
        "proposal_duration_seconds": sum(
            float(row["duration_sec"])
            for row in proposals
            if row["proposal_status"] != "rejected_by_qc"
        ),
        "uncertain_count": status_counts["uncertain"],
        "manual_review_required_count": sum(
            bool(row["manual_review_required"]) for row in proposals
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stationary_dwell_candidate_seconds": proposal_config["state_machine"][
            "stationary_dwell_candidate_seconds"
        ],
        "stationary_dwell_candidate_basis": proposal_config["protocol"][
            "stationary_dwell_candidate_basis"
        ],
        "stationary_dwell_candidate_validated": proposal_config["protocol"][
            "stationary_dwell_candidate_validated"
        ],
        "movement_start_candidate_validated": proposal_config["protocol"][
            "movement_start_candidate_validated"
        ],
        "truth_labels_consumed_by_producer": False,
        "shape_predictions_consumed_by_producer": False,
        "frozen_model_loaded": False,
        "models_trained_or_updated": False,
        "automatic_boundary_validated": False,
        "boundary_metrics_available": False,
        "shape_metrics_available": False,
        "manual_acceptance_required": True,
    }
    if proposal_config.get("schema_version") == (
        CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
    ):
        summary["minimum_track_confidence"] = float(
            proposal_config["sampling"]["minimum_track_confidence"]
        )
        summary["movement_evidence_threshold_body_heights"] = float(
            proposal_config["state_machine"][
                "movement_start_min_displacement_body_heights"
            ]
        )
        summary["segmenter_strategy"] = (
            "home_recall_hysteresis"
            if _uses_home_recall_strategy(proposal_config)
            else "legacy_path_displacement_state_machine"
        )
    return summary


def _bundle_readme(summary: Mapping[str, Any]) -> str:
    return (
        "# Camera episode boundary proposals\n\n"
        "This bundle contains truth-free, implementation-only boundary proposals. "
        "Every row requires manual review and is not an accepted EP1A boundary.\n\n"
        f"- source_video_id: `{summary['source_video_id']}`\n"
        f"- proposal_count: `{summary['proposal_count']}`\n"
        f"- locomotion_proposal_count: `{summary['locomotion_proposal_count']}`\n"
        f"- proposed/uncertain/rejected_by_qc: "
        f"`{summary['proposal_status_counts']['proposed']}/"
        f"{summary['proposal_status_counts']['uncertain']}/"
        f"{summary['proposal_status_counts']['rejected_by_qc']}`\n"
        f"- accepted_observation_coverage_ratio: "
        f"`{summary['accepted_observation_coverage_ratio']:.6f}`\n"
        "- boundary metrics: unavailable (no independent human boundary)\n"
        "- shape inference: not run\n"
        "- accepted boundary export: not performed\n"
    )


def _path_displacement_body_heights(buckets: Sequence[WeightedBucket]) -> float:
    return sum(_body_step(left, right) for left, right in zip(buckets, buckets[1:]))


def _technical_split_time(
    left_segment: Sequence[CameraObservation],
    right_segment: Sequence[CameraObservation],
    *,
    minimum_track_confidence: float,
) -> float:
    left_accepted = [
        item
        for item in left_segment
        if item.track_confidence >= minimum_track_confidence
    ]
    right_accepted = [
        item
        for item in right_segment
        if item.track_confidence >= minimum_track_confidence
    ]
    left = left_accepted[-1] if left_accepted else left_segment[-1]
    right = right_accepted[0] if right_accepted else right_segment[0]
    if left.timestamp_sec >= right.timestamp_sec:
        raise CameraEpisodeBoundaryProposalError(
            "technical segment timestamps do not define a split"
        )
    return (float(left.timestamp_sec) + float(right.timestamp_sec)) / 2.0


def _body_step(left: WeightedBucket, right: WeightedBucket) -> float:
    height = float(np.median([left.bbox_height, right.bbox_height]))
    if not math.isfinite(height) or height <= 0.0:
        raise CameraEpisodeBoundaryProposalError("bucket body height is invalid")
    displacement = float(
        np.linalg.norm(
            np.asarray(right.bbox_bottom_point, dtype=np.float64)
            - np.asarray(left.bbox_bottom_point, dtype=np.float64)
        )
    )
    if not math.isfinite(displacement):
        raise CameraEpisodeBoundaryProposalError("bucket displacement is invalid")
    return displacement / height


def _refine_locomotion_interval(
    accepted: Sequence[CameraObservation],
    *,
    coarse_start_sec: float,
    coarse_end_sec: float,
    segment_start_bound: float,
    segment_end_bound: float,
    proposal_config: Mapping[str, Any],
) -> tuple[float, float, list[str]]:
    """Refine a coarse interval on a local 0.25-second motion grid."""

    refinement = proposal_config.get("boundary_refinement")
    if not isinstance(refinement, Mapping) or refinement.get("enabled") is not True:
        return float(coarse_start_sec), float(coarse_end_sec), []
    _validate_development_profile(proposal_config)
    fine_seconds = float(refinement["bucket_seconds"])
    radius = float(refinement["search_radius_seconds"])
    fine_buckets = weighted_bucket_observations(
        accepted,
        minimum_track_confidence=float(
            proposal_config["sampling"]["minimum_track_confidence"]
        ),
        bucket_seconds=fine_seconds,
    )
    start = float(coarse_start_sec)
    end = float(coarse_end_sec)
    movement_threshold = float(
        refinement["movement_step_min_body_heights"]
    )
    start_low = start - radius - 1e-9
    start_high = start + radius + 1e-9
    for left, right in zip(fine_buckets, fine_buckets[1:]):
        candidate = _bucket_start(right, fine_seconds)
        if candidate < start_low or candidate > start_high:
            continue
        if right.bucket_index != left.bucket_index + 1:
            continue
        if _body_step(left, right) + 1e-12 >= movement_threshold:
            start = candidate
            break

    stationary_threshold = float(
        refinement["stationary_step_max_body_heights"]
    )
    confirmation = int(refinement["stationary_confirmation_buckets"])
    end_low = end - radius - 1e-9
    end_high = end + radius + 1e-9
    for index, bucket in enumerate(fine_buckets):
        candidate = _bucket_start(bucket, fine_seconds)
        if candidate < end_low or candidate > end_high:
            continue
        following = fine_buckets[index : index + confirmation + 1]
        if len(following) != confirmation + 1:
            continue
        if any(
            right.bucket_index != left.bucket_index + 1
            or _body_step(left, right) > stationary_threshold + 1e-12
            for left, right in zip(following, following[1:])
        ):
            continue
        end = candidate
        break

    start = max(float(segment_start_bound), start)
    end = min(float(segment_end_bound), end)
    if start >= end:
        start = max(float(segment_start_bound), float(coarse_start_sec))
        end = min(float(segment_end_bound), float(coarse_end_sec))
    reasons: list[str] = []
    if not math.isclose(start, float(coarse_start_sec), abs_tol=1e-9):
        reasons.append("start_refined_0p25s")
    if not math.isclose(end, float(coarse_end_sec), abs_tol=1e-9):
        reasons.append("end_refined_0p25s")
    return start, end, reasons


def _bucket_start(bucket: WeightedBucket, bucket_seconds: float) -> float:
    return bucket.bucket_index * bucket_seconds


def _bucket_end(bucket: WeightedBucket, bucket_seconds: float) -> float:
    return (bucket.bucket_index + 1) * bucket_seconds


def _transition(states: list[str], state: EpisodeBoundaryState) -> None:
    if states[-1] != state.value:
        states.append(state.value)


def _proposal_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_group_id"]),
        str(row["source_video_id"]),
        str(row["device_id"]),
        str(row["setup_id"]),
        str(row["stream_epoch"]),
        int(row["track_id"]),
        float(row["start_sec"]),
        str(row["proposal_id"]),
    )


def _diagnostic_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_group_id"]),
        str(row["source_video_id"]),
        str(row["device_id"]),
        str(row["setup_id"]),
        str(row["stream_epoch"]),
        int(row["track_id"]),
        int(row["technical_segment_index"]),
    )


def _validate_proposal_intervals(rows: Sequence[Mapping[str, Any]]) -> None:
    previous: dict[tuple[str, str, str, str, str, int], float] = {}
    ids: set[str] = set()
    for row in rows:
        proposal_id = str(row["proposal_id"])
        if proposal_id in ids:
            raise CameraEpisodeBoundaryProposalError("duplicate proposal_id")
        ids.add(proposal_id)
        key = (
            str(row["source_group_id"]),
            str(row["source_video_id"]),
            str(row["device_id"]),
            str(row["setup_id"]),
            str(row["stream_epoch"]),
            int(row["track_id"]),
        )
        start = float(row["start_sec"])
        if start < previous.get(key, -math.inf) - 1e-9:
            raise CameraEpisodeBoundaryProposalError(
                "proposal intervals overlap within one camera scope"
            )
        previous[key] = float(row["end_sec_exclusive"])


def _validate_config_bindings(
    proposal_config: Mapping[str, Any], camera_config: Mapping[str, Any]
) -> None:
    if proposal_config.get("schema_version") == (
        CAMERA_EPISODE_BOUNDARY_PROPOSAL_CONFIG_SCHEMA_VERSION
    ):
        if proposal_config != _expected_config():
            raise CameraEpisodeBoundaryProposalError("proposal config is not fixed v1")
    elif proposal_config.get("schema_version") == (
        CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
    ):
        _validate_development_profile(proposal_config)
    else:
        raise CameraEpisodeBoundaryProposalError("proposal config schema is unsupported")
    sampling = proposal_config["sampling"]
    if sampling["bucket_seconds"] != camera_config["sampling"]["bucket_seconds"]:
        raise CameraEpisodeBoundaryProposalError("proposal bucket grid drifted from camera QC")
    if float(sampling["minimum_track_confidence"]) < float(
        camera_config["sampling"]["minimum_track_confidence"]
    ):
        raise CameraEpisodeBoundaryProposalError(
            "proposal accepted-observation threshold is below camera QC"
        )
    if (
        sampling["maximum_internal_gap_buckets"]
        != camera_config["camera_qc"]["maximum_internal_gap_buckets"]
    ):
        raise CameraEpisodeBoundaryProposalError(
            "proposal technical gap threshold drifted from camera QC"
        )


def _producer_config_id(proposal_config: Mapping[str, Any]) -> str:
    if proposal_config.get("schema_version") == (
        CAMERA_EPISODE_BOUNDARY_PROPOSAL_CONFIG_SCHEMA_VERSION
    ):
        if proposal_config != _expected_config():
            raise CameraEpisodeBoundaryProposalError("proposal config is not fixed v1")
        return PRODUCER_CONFIG_ID
    _validate_development_profile(proposal_config)
    return str(proposal_config["profile_id"])


def _validate_development_profile(value: Any) -> None:
    expected = _expected_config()
    fields = {
        *expected,
        "profile_id",
        "base_producer_config_id",
        "boundary_refinement",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CameraEpisodeBoundaryProposalError(
            "development profile fields drifted"
        )
    if value.get("schema_version") != (
        CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
    ):
        raise CameraEpisodeBoundaryProposalError(
            "development profile schema drifted"
        )
    if value.get("purpose") != "b01_b02_development_parameter_search":
        raise CameraEpisodeBoundaryProposalError(
            "development profile purpose drifted"
        )
    profile_id = value.get("profile_id")
    if (
        not isinstance(profile_id, str)
        or not profile_id
        or len(profile_id) > 128
        or any(not (character.isalnum() or character in "._-") for character in profile_id)
    ):
        raise CameraEpisodeBoundaryProposalError("development profile_id is invalid")
    if value.get("base_producer_config_id") != PRODUCER_CONFIG_ID:
        raise CameraEpisodeBoundaryProposalError(
            "development base producer identity drifted"
        )
    for field in ("camera_chain", "output_schemas", "data_access"):
        if value.get(field) != expected[field]:
            raise CameraEpisodeBoundaryProposalError(
                f"development profile {field} drifted"
            )
    sampling = value.get("sampling")
    if not isinstance(sampling, Mapping) or set(sampling) != set(expected["sampling"]):
        raise CameraEpisodeBoundaryProposalError(
            "development profile sampling fields drifted"
        )
    if sampling.get("bucket_seconds") != 0.5:
        raise CameraEpisodeBoundaryProposalError("coarse bucket_seconds must remain 0.5")
    minimum_track_confidence = sampling.get("minimum_track_confidence")
    if (
        isinstance(minimum_track_confidence, bool)
        or not isinstance(minimum_track_confidence, (int, float))
        or not 0.25 <= float(minimum_track_confidence) <= 0.99
    ):
        raise CameraEpisodeBoundaryProposalError(
            "minimum_track_confidence must be in [0.25, 0.99]"
        )
    maximum_gap = sampling.get("maximum_internal_gap_buckets")
    if isinstance(maximum_gap, bool) or not isinstance(maximum_gap, int) or not 1 <= maximum_gap <= 3:
        raise CameraEpisodeBoundaryProposalError(
            "maximum_internal_gap_buckets must be an integer in [1, 3]"
        )
    machine = value.get("state_machine")
    if not isinstance(machine, Mapping) or set(machine) != set(expected["state_machine"]):
        raise CameraEpisodeBoundaryProposalError(
            "development profile state_machine fields drifted"
        )
    movement_buckets = machine.get("movement_start_min_buckets")
    if (
        isinstance(movement_buckets, bool)
        or not isinstance(movement_buckets, int)
        or not 2 <= movement_buckets <= 10
    ):
        raise CameraEpisodeBoundaryProposalError(
            "movement_start_min_buckets must be an integer in [2, 10]"
        )
    for field, minimum, maximum in (
        ("movement_start_min_displacement_body_heights", 0.01, 1.0),
        ("stationary_max_displacement_body_heights", 0.0, 0.5),
        ("stationary_dwell_candidate_seconds", 0.5, 60.0),
        ("minimum_episode_duration_seconds", 0.25, 30.0),
    ):
        number = machine.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not minimum <= float(number) <= maximum
        ):
            raise CameraEpisodeBoundaryProposalError(
                f"development profile {field} is invalid"
            )
    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping) or set(protocol) != set(expected["protocol"]):
        raise CameraEpisodeBoundaryProposalError(
            "development profile protocol fields drifted"
        )
    for field in (
        "movement_start_candidate_basis",
        "stationary_dwell_candidate_basis",
    ):
        if not isinstance(protocol.get(field), str) or not protocol[field]:
            raise CameraEpisodeBoundaryProposalError(
                f"development profile {field} is invalid"
            )
    for field in (
        "movement_start_candidate_validated",
        "stationary_dwell_candidate_validated",
    ):
        if not isinstance(protocol.get(field), bool):
            raise CameraEpisodeBoundaryProposalError(
                f"development profile {field} must be boolean"
            )
    refinement = value.get("boundary_refinement")
    refinement_fields = {
        "enabled",
        "bucket_seconds",
        "search_radius_seconds",
        "movement_step_min_body_heights",
        "stationary_step_max_body_heights",
        "stationary_confirmation_buckets",
    }
    if not isinstance(refinement, Mapping) or set(refinement) != refinement_fields:
        raise CameraEpisodeBoundaryProposalError(
            "development boundary_refinement fields drifted"
        )
    if not isinstance(refinement.get("enabled"), bool):
        raise CameraEpisodeBoundaryProposalError("refinement enabled must be boolean")
    if refinement.get("bucket_seconds") != 0.25:
        raise CameraEpisodeBoundaryProposalError(
            "development refinement bucket_seconds must remain 0.25"
        )
    radius = refinement.get("search_radius_seconds")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not 0.25 <= float(radius) <= 2.0
        or not math.isclose(float(radius) * 4.0, round(float(radius) * 4.0))
    ):
        raise CameraEpisodeBoundaryProposalError(
            "refinement search radius must use the 0.25-second grid"
        )
    for field, minimum, maximum in (
        ("movement_step_min_body_heights", 0.001, 0.5),
        ("stationary_step_max_body_heights", 0.0, 0.25),
    ):
        number = refinement.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not minimum <= float(number) <= maximum
        ):
            raise CameraEpisodeBoundaryProposalError(
                f"development refinement {field} is invalid"
            )
    confirmation = refinement.get("stationary_confirmation_buckets")
    if (
        isinstance(confirmation, bool)
        or not isinstance(confirmation, int)
        or not 1 <= confirmation <= 8
    ):
        raise CameraEpisodeBoundaryProposalError(
            "stationary_confirmation_buckets must be an integer in [1, 8]"
        )


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera episode boundary proposal output exists: {output}")
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


def _uses_home_recall_strategy(proposal_config: Mapping[str, Any]) -> bool:
    return (
        proposal_config.get("schema_version")
        == CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
        and proposal_config.get("profile_id") == HOME_RECALL_PROFILE_ID
    )


def _home_blocking_hard_break(reasons: Sequence[str]) -> bool:
    return bool(
        {"long_internal_gap", "height_position_discontinuity"}.intersection(
            reasons
        )
    )


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_CONFIG_SCHEMA_VERSION,
        "purpose": "continuous_episode_boundary_proposal_implementation_only",
        "camera_chain": {"path": "configs/modules/wandering_camera_v1.yaml"},
        "sampling": {
            "bucket_seconds": 0.5,
            "minimum_track_confidence": 0.25,
            "maximum_internal_gap_buckets": 3,
        },
        "state_machine": {
            "movement_start_min_buckets": 5,
            "movement_start_min_displacement_body_heights": 0.25,
            "stationary_max_displacement_body_heights": 0.08,
            "stationary_dwell_candidate_seconds": 15.0,
            "minimum_episode_duration_seconds": 2.0,
        },
        "protocol": {
            "movement_start_candidate_basis": (
                "b01_development_repeated_failure_analysis"
            ),
            "movement_start_candidate_validated": True,
            "stationary_dwell_candidate_basis": (
                "b01_development_dwell_comparison_retained_15_seconds"
            ),
            "stationary_dwell_candidate_validated": True,
        },
        "output_schemas": {
            "proposal": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
            "summary": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION,
            "diagnostic": CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION,
        },
        "data_access": {
            "shape_truth": False,
            "purpose_truth": False,
            "frozen_model": False,
            "training": False,
            "sealed_camera": False,
        },
    }


__all__ = [
    "CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION",
    "CAMERA_EPISODE_BOUNDARY_PROPOSAL_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_BOUNDARY_PROPOSAL_DIAGNOSTIC_SCHEMA_VERSION",
    "CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION",
    "CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION",
    "CameraEpisodeBoundaryProposalBuildResult",
    "CameraEpisodeBoundaryProposalError",
    "EpisodeBoundaryState",
    "build_camera_episode_boundary_development_bundle",
    "build_camera_episode_boundary_proposal_bundle",
    "load_camera_episode_boundary_development_profile",
    "load_camera_episode_boundary_proposal_config",
    "propose_camera_episode_boundaries",
]
