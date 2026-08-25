from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraObservation,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION,
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
    _refine_locomotion_interval,
    build_camera_episode_boundary_proposal_bundle,
    load_camera_episode_boundary_development_profile,
    load_camera_episode_boundary_proposal_config,
    propose_camera_episode_boundaries,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CameraEpisodeImportError,
    load_episode_boundaries,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG_PATH = ROOT / "configs/modules/wandering_camera_v1.yaml"
PROPOSAL_CONFIG_PATH = (
    ROOT / "configs/modules/wandering_camera_episode_boundary_proposal_v1.yaml"
)
HOME_PROFILE_PATH = (
    ROOT / "configs/modules/wandering_camera_episode_boundary_home_v1.yaml"
)


def _media(duration_sec: float) -> dict[str, object]:
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": "mixed-activity-001",
        "source_group_id": "session-episode-boundary-001",
        "device_id": "camera-001",
        "setup_id": "fixed-setup-001",
        "stream_epoch": "epoch-001",
        "media_ref": "authorized/mixed-activity-001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": "2" * 64,
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 20.0,
        "duration_sec": duration_sec,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {
            "backend": "ultralytics_yolo",
            "model": "yolov8n.pt",
            "version": "synthetic",
        },
        "tracker": {
            "backend": "bytetrack",
            "config": "bytetrack.yaml",
            "version": "synthetic",
        },
        "fixed_camera_assumed": True,
        "camera_motion_state": "stable",
        "authorization_status": "authorized_camera_engineering_smoke",
        "deidentification_status": "anonymous",
    }


def _observation(
    *,
    index: int,
    x: float,
    y: float = 0.75,
    height: float = 0.25,
    track_id: int = 1,
    confidence: float = 0.9,
    timestamp_sec: float | None = None,
) -> CameraObservation:
    scope = (
        "session-episode-boundary-001",
        "mixed-activity-001",
        "camera-001",
        "fixed-setup-001",
        "epoch-001",
        track_id,
    )
    timestamp = index * 0.5 + 0.1 if timestamp_sec is None else timestamp_sec
    return CameraObservation(
        scope_key=scope,
        source_group_id=scope[0],
        source_video_id=scope[1],
        device_id=scope[2],
        setup_id=scope[3],
        stream_epoch=scope[4],
        track_id=track_id,
        frame_id=index * 10 + track_id,
        timestamp_sec=timestamp,
        bbox_xyxy_pixel=(100.0, 100.0, 180.0, 360.0),
        bbox_xyxy_norm=(0.15625, 0.208333, 0.28125, 0.75),
        bbox_bottom_point=(x, y),
        bbox_height=height,
        track_confidence=confidence,
    )


def _adapter(
    tracks: dict[int, list[tuple[float, float, float, float]]],
    *,
    duration_sec: float | None = None,
) -> CameraAdapterInput:
    observations = [
        _observation(
            index=index,
            x=value[0],
            y=value[1],
            height=value[2],
            confidence=value[3],
            track_id=track_id,
        )
        for track_id, values in tracks.items()
        for index, value in enumerate(values)
    ]
    observations.sort(key=lambda item: (item.scope_key, item.timestamp_sec, item.frame_id))
    maximum_time = max(item.timestamp_sec for item in observations)
    return CameraAdapterInput(
        media_sidecar=_media(
            duration_sec if duration_sec is not None else maximum_time + 0.5
        ),
        observations=tuple(observations),
        normalized_rows=tuple(),
        source_tracking_sha256="2" * 64,
        normalized_tracking_sha256="2" * 64,
    )


def _value(
    x: float,
    y: float = 0.75,
    height: float = 0.25,
    confidence: float = 0.9,
) -> tuple[float, float, float, float]:
    return (x, y, height, confidence)


def _line(start: float, stop: float, count: int) -> list[tuple[float, float, float, float]]:
    return [
        _value(start + (stop - start) * index / max(1, count - 1))
        for index in range(count)
    ]


def _stationary(x: float, count: int) -> list[tuple[float, float, float, float]]:
    return [_value(x) for _ in range(count)]


def _configs() -> tuple[dict[str, object], dict[str, object]]:
    return (
        load_camera_episode_boundary_proposal_config(PROPOSAL_CONFIG_PATH),
        load_camera_config(CAMERA_CONFIG_PATH),
    )


def _development_profile() -> dict[str, object]:
    proposal_config, _camera_config = _configs()
    profile = deepcopy(proposal_config)
    profile["schema_version"] = (
        CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
    )
    profile["purpose"] = "b01_b02_development_parameter_search"
    profile["profile_id"] = "w5d01-refinement-test"
    profile["base_producer_config_id"] = (
        "m0cam-ep2a-s0-b01-development-frozen-v1"
    )
    profile["boundary_refinement"] = {
        "enabled": True,
        "bucket_seconds": 0.25,
        "search_radius_seconds": 0.5,
        "movement_step_min_body_heights": 0.05,
        "stationary_step_max_body_heights": 0.03,
        "stationary_confirmation_buckets": 2,
    }
    return profile


def _run(
    values: list[tuple[float, float, float, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    proposal_config, camera_config = _configs()
    return propose_camera_episode_boundaries(
        _adapter({1: values}),
        proposal_config=proposal_config,
        camera_config=camera_config,
    )


def _run_home(
    tracks: dict[int, list[tuple[float, float, float, float]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return propose_camera_episode_boundaries(
        _adapter(tracks),
        proposal_config=load_camera_episode_boundary_development_profile(
            HOME_PROFILE_PATH
        ),
        camera_config=load_camera_config(CAMERA_CONFIG_PATH),
    )


def _locomotion(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["proposal_status"] != "rejected_by_qc"]


def test_stationary_move_and_sustained_stationary_form_one_proposal() -> None:
    proposals, diagnostics = _run(
        _stationary(0.2, 4) + _line(0.2, 0.75, 14) + _stationary(0.75, 34)
    )

    assert CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION == (
        "wandering-camera-episode-boundary-proposal-v1"
    )
    assert len(_locomotion(proposals)) == 1
    proposal = _locomotion(proposals)[0]
    assert proposal["proposal_status"] == "proposed"
    assert proposal["start_reason"] == "sustained_movement"
    assert proposal["end_reason"] == "sustained_stationary"
    assert proposal["uncertain"] is False
    assert proposal["manual_review_required"] is True
    assert diagnostics[0]["state_sequence"] == ["idle", "open", "closing", "closed"]


def test_frozen_v1_representative_proposal_identity_and_endpoints_are_unchanged() -> None:
    proposals, _diagnostics = _run(
        _line(0.2, 0.75, 12) + _stationary(0.75, 34)
    )

    assert proposals == [
        {
            "schema_version": "wandering-camera-episode-boundary-proposal-v1",
            "proposal_id": (
                "boundary-proposal-"
                "7db6bb4b68e6bdaac71456b894899a00b238f386a2e44d8240be643ccd852406"
            ),
            "source_group_id": "session-episode-boundary-001",
            "source_video_id": "mixed-activity-001",
            "device_id": "camera-001",
            "setup_id": "fixed-setup-001",
            "stream_epoch": "epoch-001",
            "track_id": 1,
            "technical_segment_index": 0,
            "start_sec": 0.0,
            "end_sec_exclusive": 5.5,
            "duration_sec": 5.5,
            "proposal_status": "proposed",
            "start_reason": "sustained_movement_at_track_start",
            "end_reason": "sustained_stationary",
            "reason_codes": [
                "sustained_movement",
                "sustained_stationary",
                "protocol_derived_unvalidated_parameters",
            ],
            "hard_break_reasons": [],
            "uncertain": False,
            "manual_review_required": True,
            "source_observation_count": 46,
            "accepted_observation_count": 46,
            "source_bucket_count": 46,
            "producer_config_id": "m0cam-ep2a-s0-b01-development-frozen-v1",
        }
    ]


def test_development_refinement_uses_quarter_second_grid_and_reason_codes() -> None:
    xs = [0.20, 0.20, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.44, 0.44, 0.44]
    observations = tuple(
        _observation(index=index, timestamp_sec=index * 0.25 + 0.05, x=x)
        for index, x in enumerate(xs)
    )

    start, end, reasons = _refine_locomotion_interval(
        observations,
        coarse_start_sec=1.0,
        coarse_end_sec=2.5,
        segment_start_bound=0.0,
        segment_end_bound=3.0,
        proposal_config=_development_profile(),
    )

    assert start == 0.75
    assert end == 2.0
    assert reasons == ["start_refined_0p25s", "end_refined_0p25s"]
    assert start * 4 == pytest.approx(round(start * 4))
    assert end * 4 == pytest.approx(round(end * 4))


def test_development_refinement_is_clamped_to_technical_segment_bounds() -> None:
    xs = [0.20, 0.20, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.44, 0.44, 0.44]
    observations = tuple(
        _observation(index=index, timestamp_sec=index * 0.25 + 0.05, x=x)
        for index, x in enumerate(xs)
    )

    start, end, reasons = _refine_locomotion_interval(
        observations,
        coarse_start_sec=1.0,
        coarse_end_sec=2.5,
        segment_start_bound=0.9,
        segment_end_bound=1.9,
        proposal_config=_development_profile(),
    )

    assert start == 0.9
    assert end == 1.9
    assert reasons == ["start_refined_0p25s", "end_refined_0p25s"]


def test_short_pause_returns_closing_to_open_without_fragmenting() -> None:
    proposals, diagnostics = _run(
        _line(0.2, 0.55, 10)
        + _stationary(0.55, 8)
        + _line(0.55, 0.82, 10)
        + _stationary(0.82, 34)
    )

    assert len(_locomotion(proposals)) == 1
    assert diagnostics[0]["state_sequence"] == [
        "idle",
        "open",
        "closing",
        "open",
        "closing",
        "closed",
    ]


def test_sustained_slow_movement_uses_two_second_opening_window() -> None:
    proposals, diagnostics = _run(
        _line(0.2, 0.27, 5) + _stationary(0.27, 34)
    )

    assert len(_locomotion(proposals)) == 1
    proposal = _locomotion(proposals)[0]
    assert proposal["producer_config_id"] == (
        "m0cam-ep2a-s0-b01-development-frozen-v1"
    )
    assert proposal["start_reason"] == "sustained_movement_at_track_start"
    assert proposal["end_reason"] == "sustained_stationary"
    assert diagnostics[0]["state_sequence"] == [
        "idle",
        "open",
        "closing",
        "closed",
    ]


@pytest.mark.parametrize(
    ("name", "moving"),
    [
        (
            "pacing",
            _line(0.2, 0.8, 12)
            + _line(0.8, 0.2, 12)
            + _line(0.2, 0.8, 12),
        ),
        (
            "lapping",
            [
                _value(
                    0.5 + 0.23 * math.cos(4.0 * math.pi * index / 48.0),
                    0.6 + 0.16 * math.sin(4.0 * math.pi * index / 48.0),
                )
                for index in range(49)
            ],
        ),
        (
            "random",
            [
                _value(
                    left[0] + (right[0] - left[0]) * step / 3.0,
                    left[1] + (right[1] - left[1]) * step / 3.0,
                )
                for left, right in zip(
                    (
                        (0.2, 0.7),
                        (0.35, 0.55),
                        (0.62, 0.72),
                        (0.45, 0.42),
                        (0.78, 0.58),
                        (0.3, 0.75),
                        (0.68, 0.38),
                    ),
                    (
                        (0.35, 0.55),
                        (0.62, 0.72),
                        (0.45, 0.42),
                        (0.78, 0.58),
                        (0.3, 0.75),
                        (0.68, 0.38),
                        (0.82, 0.7),
                    ),
                    strict=True,
                )
                for step in range(3)
            ]
            + [_value(0.82, 0.7)],
        ),
    ],
)
def test_direction_reversal_loop_and_random_turns_do_not_cut(
    name: str, moving: list[tuple[float, float, float, float]]
) -> None:
    proposals, _diagnostics = _run(moving + _stationary(moving[-1][0], 34))

    assert len(_locomotion(proposals)) == 1, name


def test_two_movement_bouts_need_sustained_dwell_between_them() -> None:
    proposals, _diagnostics = _run(
        _line(0.2, 0.7, 12)
        + _stationary(0.7, 34)
        + _line(0.7, 0.3, 12)
        + _stationary(0.3, 34)
    )

    rows = _locomotion(proposals)
    assert len(rows) == 2
    assert rows[0]["end_sec_exclusive"] <= rows[1]["start_sec"]


def test_slow_translation_cannot_yield_confident_stationary_boundary() -> None:
    moving = _line(0.2, 0.55, 6)
    slow_translation = [_value(0.55 + 0.006 * index) for index in range(40)]
    proposals, _diagnostics = _run(
        moving
        + slow_translation
        + _stationary(slow_translation[-1][0], 34)
    )

    rows = _locomotion(proposals)
    assert len(rows) == 1
    assert rows[0]["end_reason"] == "sustained_stationary"
    assert rows[0]["proposal_status"] == "uncertain"
    assert "stationary_candidate_drift" in rows[0]["reason_codes"]


def test_long_gap_uses_shared_hard_break_and_never_splices() -> None:
    proposal_config, camera_config = _configs()
    values = _line(0.2, 0.55, 10) + _line(0.55, 0.85, 10)
    observations = [
        _observation(
            index=index if index < 10 else index + 5,
            timestamp_sec=(index if index < 10 else index + 5) * 0.5 + 0.1,
            x=value[0],
        )
        for index, value in enumerate(values)
    ]
    adapter = _adapter({1: values}, duration_sec=13.0)
    adapter = CameraAdapterInput(
        media_sidecar=adapter.media_sidecar,
        observations=tuple(observations),
        normalized_rows=tuple(),
        source_tracking_sha256=adapter.source_tracking_sha256,
        normalized_tracking_sha256=adapter.normalized_tracking_sha256,
    )

    proposals, diagnostics = propose_camera_episode_boundaries(
        adapter,
        proposal_config=proposal_config,
        camera_config=camera_config,
    )

    rows = _locomotion(proposals)
    assert len(rows) == 2
    assert rows[0]["end_sec_exclusive"] < rows[1]["start_sec"]
    assert all("long_internal_gap" in row["hard_break_reasons"] for row in rows)
    assert len(diagnostics) == 2


@pytest.mark.parametrize(
    ("name", "values", "reason"),
    [
        (
            "id-switch",
            _line(0.2, 0.42, 10) + _line(0.92, 0.97, 10),
            "suspected_id_switch",
        ),
        (
            "height-position",
            _line(0.2, 0.4, 10)
            + [_value(0.62 + 0.01 * index, height=0.10) for index in range(10)],
            "height_position_discontinuity",
        ),
    ],
)
def test_id_switch_and_height_position_discontinuity_are_hard_splits(
    name: str,
    values: list[tuple[float, float, float, float]],
    reason: str,
) -> None:
    proposals, diagnostics = _run(values)

    assert len(diagnostics) == 2, name
    assert all(reason in row["hard_break_reasons"] for row in diagnostics)
    assert all(reason in row["hard_break_reasons"] for row in proposals)


def test_same_bucket_hard_break_keeps_technical_intervals_non_overlapping() -> None:
    proposal_config, camera_config = _configs()
    base = _adapter({1: [_value(0.2), _value(0.8)]}, duration_sec=1.0)
    observations = (
        _observation(index=0, timestamp_sec=0.10, x=0.20),
        _observation(index=1, timestamp_sec=0.20, x=0.80),
    )
    adapter = CameraAdapterInput(
        media_sidecar=base.media_sidecar,
        observations=observations,
        normalized_rows=tuple(),
        source_tracking_sha256=base.source_tracking_sha256,
        normalized_tracking_sha256=base.normalized_tracking_sha256,
    )

    proposals, diagnostics = propose_camera_episode_boundaries(
        adapter,
        proposal_config=proposal_config,
        camera_config=camera_config,
    )

    assert len(diagnostics) == 2
    assert len(proposals) == 2
    assert all(
        "suspected_id_switch" in row["hard_break_reasons"]
        for row in proposals
    )
    assert proposals[0]["end_sec_exclusive"] <= proposals[1]["start_sec"]


def test_low_confidence_edges_cannot_open_or_extend_a_proposal() -> None:
    values = (
        [_value(0.02, confidence=0.1), _value(0.95, confidence=0.1)]
        + _line(0.2, 0.75, 12)
        + _stationary(0.75, 34)
        + [_value(0.02, confidence=0.1), _value(0.95, confidence=0.1)]
    )
    proposals, diagnostics = _run(values)

    proposal = _locomotion(proposals)[0]
    assert proposal["start_sec"] >= 1.0
    assert proposal["end_sec_exclusive"] < (len(values) - 2) * 0.5
    assert proposal["accepted_observation_count"] < proposal["source_observation_count"]
    assert diagnostics[0]["accepted_observation_count"] == len(values) - 4


def test_tracks_are_isolated_and_open_track_end_is_uncertain() -> None:
    proposal_config, camera_config = _configs()
    adapter = _adapter(
        {
            1: _line(0.2, 0.8, 14),
            2: _line(0.8, 0.2, 14),
        }
    )

    proposals, diagnostics = propose_camera_episode_boundaries(
        adapter,
        proposal_config=proposal_config,
        camera_config=camera_config,
    )

    assert {row["track_id"] for row in proposals} == {1, 2}
    assert len(diagnostics) == 2
    assert all(row["proposal_status"] == "uncertain" for row in proposals)
    assert all(row["uncertain"] is True for row in proposals)
    assert all(row["end_reason"] == "track_or_stream_end" for row in proposals)


def test_insufficient_locomotion_is_retained_as_rejected_by_qc() -> None:
    proposals, diagnostics = _run(_stationary(0.4, 20))

    assert len(proposals) == 1
    assert proposals[0]["proposal_status"] == "rejected_by_qc"
    assert proposals[0]["uncertain"] is True
    assert proposals[0]["manual_review_required"] is True
    assert "insufficient_locomotion_evidence" in proposals[0]["reason_codes"]
    assert diagnostics[0]["proposal_status_counts"] == {"rejected_by_qc": 1}


def test_home_profile_uses_only_confidence_070_tracks_for_formal_proposals() -> None:
    high_confidence_motion = _line(0.2, 0.55, 12)
    low_confidence_false_track = [
        _value(0.1 if index % 2 == 0 else 0.9, confidence=0.69)
        for index in range(20)
    ]

    proposals, diagnostics = _run_home(
        {1: high_confidence_motion, 9: low_confidence_false_track}
    )

    locomotion = _locomotion(proposals)
    assert locomotion
    assert {row["track_id"] for row in locomotion} == {1}
    assert all(row["accepted_observation_count"] > 0 for row in proposals)
    assert sum(row["accepted_observation_count"] for row in diagnostics) == len(
        high_confidence_motion
    )


def test_home_profile_slow_high_confidence_translation_is_forward_candidate() -> None:
    proposals, _diagnostics = _run_home({1: _line(0.2, 0.32, 14)})

    locomotion = _locomotion(proposals)
    assert len(locomotion) == 1
    assert locomotion[0]["start_reason"] == "sustained_movement_at_track_start"
    assert locomotion[0]["proposal_status"] in {"proposed", "uncertain"}


def test_home_profile_stationary_high_confidence_jitter_is_not_an_episode() -> None:
    jitter = [
        _value(0.4 + (0.001 if index % 2 else -0.001))
        for index in range(30)
    ]

    proposals, diagnostics = _run_home({1: jitter})

    assert proposals == []
    assert diagnostics[0]["proposal_status_counts"] == {}
    assert diagnostics[0]["reason_codes"] == ["stationary_track_no_episode"]


def test_home_profile_brief_pause_does_not_fragment_continuous_motion() -> None:
    proposals, _diagnostics = _run_home(
        {
            1: _line(0.2, 0.5, 10)
            + _stationary(0.5, 2)
            + _line(0.5, 0.8, 10)
        }
    )

    assert len(_locomotion(proposals)) == 1


def test_home_profile_same_track_position_jump_keeps_reason_without_forcing_uncertain() -> None:
    proposals, _diagnostics = _run_home(
        {1: _line(0.2, 0.4, 10) + _line(0.9, 0.98, 10)}
    )

    locomotion = _locomotion(proposals)
    assert len(locomotion) == 2
    assert all("suspected_id_switch" in row["hard_break_reasons"] for row in locomotion)
    assert all(row["proposal_status"] == "proposed" for row in locomotion)


def _write_input_pair(directory: Path) -> tuple[Path, Path, dict[str, object]]:
    values = _line(0.2, 0.75, 12) + _stationary(0.75, 34)
    rows = [
        {
            "frame_id": index * 10 + 1,
            "track_id": 1,
            "bbox": [
                value[0] * 640.0 - 40.0,
                100.0,
                value[0] * 640.0 + 40.0,
                360.0,
            ],
            "track_confidence": value[3],
            "timestamp_sec": index * 0.5 + 0.1,
        }
        for index, value in enumerate(values)
    ]
    tracking_bytes = canonical_jsonl_bytes(rows)
    tracking = directory / "tracking.jsonl"
    tracking.write_bytes(tracking_bytes)
    media = _media(len(values) * 0.5)
    media["tracking_jsonl_sha256"] = hashlib.sha256(tracking_bytes).hexdigest()
    sidecar = directory / "media_sidecar.json"
    sidecar.write_bytes(canonical_json_bytes(media))
    return tracking, sidecar, media


def test_builder_is_deterministic_non_overwriting_and_schema_is_not_accepted_boundary(
    tmp_path: Path,
) -> None:
    tracking, sidecar, media = _write_input_pair(tmp_path)
    output_a = tmp_path / "proposal-a"
    output_b = tmp_path / "proposal-b"

    first = build_camera_episode_boundary_proposal_bundle(
        project_root=ROOT,
        proposal_config_path=PROPOSAL_CONFIG_PATH,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        output_dir=output_a,
    )
    second = build_camera_episode_boundary_proposal_bundle(
        project_root=ROOT,
        proposal_config_path=PROPOSAL_CONFIG_PATH,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        output_dir=output_b,
    )

    assert first.proposal_count == second.proposal_count == 1
    for relative in ("proposals.jsonl", "summary.json", "diagnostics.jsonl", "README.md"):
        assert (output_a / relative).read_bytes() == (output_b / relative).read_bytes()
    summary = json.loads((output_a / "summary.json").read_text(encoding="utf-8"))
    assert summary["truth_labels_consumed_by_producer"] is False
    assert summary["shape_predictions_consumed_by_producer"] is False
    assert summary["frozen_model_loaded"] is False
    assert summary["automatic_boundary_validated"] is False
    assert summary["stationary_dwell_candidate_validated"] is True
    assert summary["movement_start_candidate_validated"] is True
    with pytest.raises(CameraEpisodeImportError, match="boundary fields"):
        load_episode_boundaries(output_a / "proposals.jsonl", media)
    with pytest.raises(FileExistsError):
        build_camera_episode_boundary_proposal_bundle(
            project_root=ROOT,
            proposal_config_path=PROPOSAL_CONFIG_PATH,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            output_dir=output_a,
        )


def test_boundary_proposal_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/propose_camera_episode_boundaries.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--tracking-jsonl" in completed.stdout
    assert "--media-sidecar" in completed.stdout
    assert "--output-dir" in completed.stdout
