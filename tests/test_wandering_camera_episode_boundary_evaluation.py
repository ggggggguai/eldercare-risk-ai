from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation import (
    BOUNDARY_EVALUATION_INDEX_FIELDS,
    CameraEpisodeBoundaryEvaluationError,
    build_camera_episode_boundary_evaluation_bundle,
    evaluate_camera_episode_boundaries,
    load_camera_episode_boundary_evaluation_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CameraEpisodeImportError,
    load_episode_boundaries,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/modules/wandering_camera_episode_boundary_eval_v1.yaml"
)


def _media(
    *,
    source_video_id: str = "synthetic-boundary-001",
    source_group_id: str = "synthetic-group-001",
    setup_id: str = "setup-001",
    duration_sec: float = 120.0,
    authorization_status: str = "synthetic_fixture",
) -> dict[str, object]:
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": source_video_id,
        "source_group_id": source_group_id,
        "device_id": "camera-001",
        "setup_id": setup_id,
        "stream_epoch": "epoch-001",
        "media_ref": f"synthetic/{source_video_id}.mp4",
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
        "authorization_status": authorization_status,
        "deidentification_status": "anonymous",
    }


def _proposal(
    proposal_id: str,
    start: float,
    end: float,
    status: str,
    *,
    media: dict[str, object],
    track_id: int = 1,
    technical_segment_index: int = 0,
) -> dict[str, object]:
    uncertain = status != "proposed"
    return {
        "schema_version": "wandering-camera-episode-boundary-proposal-v1",
        "proposal_id": proposal_id,
        "source_group_id": media["source_group_id"],
        "source_video_id": media["source_video_id"],
        "device_id": media["device_id"],
        "setup_id": media["setup_id"],
        "stream_epoch": media["stream_epoch"],
        "track_id": track_id,
        "technical_segment_index": technical_segment_index,
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "proposal_status": status,
        "start_reason": (
            "no_confirmed_locomotion_start"
            if status == "rejected_by_qc"
            else "sustained_movement"
        ),
        "end_reason": (
            "insufficient_locomotion_evidence"
            if status == "rejected_by_qc"
            else "sustained_stationary"
        ),
        "reason_codes": [
            "insufficient_locomotion_evidence"
            if status == "rejected_by_qc"
            else "protocol_derived_unvalidated_parameters"
        ],
        "hard_break_reasons": [],
        "uncertain": uncertain,
        "manual_review_required": True,
        "source_observation_count": 100,
        "accepted_observation_count": 100,
        "source_bucket_count": 240,
        "producer_config_id": "m0cam-ep2a-s0-protocol-derived-unvalidated-v1",
    }


def _boundary(
    episode_id: str,
    start: float,
    end: float,
    *,
    media: dict[str, object],
    track_id: int = 1,
    status: str = "ready",
) -> dict[str, object]:
    return {
        "schema_version": "wandering-camera-episode-boundary-v1",
        "episode_id": episode_id,
        "source_video_id": media["source_video_id"],
        "target_track_id": track_id,
        "start_sec": start,
        "end_sec_exclusive": end,
        "boundary_source": "simplified_jsonl",
        "boundary_status": status,
        "boundary_reason_codes": (
            [] if status == "ready" else ["synthetic_boundary_uncertain"]
        ),
        "cvat_track_id": None,
    }


def _diagnostic(
    proposals: list[dict[str, object]],
    *,
    media: dict[str, object],
    track_id: int,
    technical_segment_index: int = 0,
) -> dict[str, object]:
    rows = [
        row
        for row in proposals
        if row["track_id"] == track_id
        and row["technical_segment_index"] == technical_segment_index
    ]
    counts = {
        name: sum(row["proposal_status"] == name for row in rows)
        for name in ("proposed", "uncertain", "rejected_by_qc")
    }
    counts = {name: count for name, count in counts.items() if count}
    return {
        "schema_version": (
            "wandering-camera-episode-boundary-proposal-diagnostic-v1"
        ),
        "source_group_id": media["source_group_id"],
        "source_video_id": media["source_video_id"],
        "device_id": media["device_id"],
        "setup_id": media["setup_id"],
        "stream_epoch": media["stream_epoch"],
        "track_id": track_id,
        "technical_segment_index": technical_segment_index,
        "segment_start_observation_sec": 0.1,
        "segment_end_observation_sec": float(media["duration_sec"]) - 0.1,
        "source_observation_count": 100,
        "accepted_observation_count": 100,
        "source_bucket_count": 240,
        "observed_bucket_count": 240,
        "accepted_observation_coverage_ratio": 1.0,
        "hard_break_reasons": [],
        "state_sequence": ["idle", "open", "closing", "closed"],
        "proposal_ids": [str(row["proposal_id"]) for row in rows],
        "proposal_status_counts": counts,
        "reason_codes": sorted(
            {
                str(reason)
                for row in rows
                for reason in row["reason_codes"]
            }
        ),
        "manual_review_required": True,
    }


def _summary(
    proposals: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    *,
    media: dict[str, object],
) -> dict[str, object]:
    status_counts = {
        name: sum(row["proposal_status"] == name for row in proposals)
        for name in ("proposed", "uncertain", "rejected_by_qc")
    }
    reason_counts: dict[str, int] = {}
    for row in proposals:
        for reason in row["reason_codes"]:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    source_count = sum(int(row["source_observation_count"]) for row in diagnostics)
    accepted_count = sum(
        int(row["accepted_observation_count"]) for row in diagnostics
    )
    source_bucket_count = sum(int(row["source_bucket_count"]) for row in diagnostics)
    observed_bucket_count = sum(
        int(row["observed_bucket_count"]) for row in diagnostics
    )
    return {
        "schema_version": "wandering-camera-episode-boundary-proposal-summary-v1",
        "status": "wandering_m0cam_ep2a_s0_boundary_proposal_generated",
        "evidence_scope": "partial_truth_free_smoke_only",
        "source_group_id": media["source_group_id"],
        "source_video_id": media["source_video_id"],
        "device_id": media["device_id"],
        "setup_id": media["setup_id"],
        "stream_epoch": media["stream_epoch"],
        "validation_scope": (
            "synthetic_camera_contract"
            if media["authorization_status"] == "synthetic_fixture"
            else "authorized_camera_labeled_evaluation"
        ),
        "producer_config_id": "m0cam-ep2a-s0-protocol-derived-unvalidated-v1",
        "proposal_schema_version": (
            "wandering-camera-episode-boundary-proposal-v1"
        ),
        "accepted_boundary_schema_version": (
            "wandering-camera-episode-boundary-v1"
        ),
        "proposal_is_accepted_boundary": False,
        "proposal_count": len(proposals),
        "locomotion_proposal_count": sum(
            row["proposal_status"] != "rejected_by_qc" for row in proposals
        ),
        "proposal_status_counts": status_counts,
        "scope_count": len({int(row["track_id"]) for row in diagnostics}),
        "technical_segment_count": len(diagnostics),
        "technical_hard_break_segment_count": 0,
        "source_observation_count": source_count,
        "accepted_observation_count": accepted_count,
        "accepted_observation_coverage_ratio": accepted_count / source_count,
        "source_bucket_count": source_bucket_count,
        "observed_bucket_count": observed_bucket_count,
        "observed_bucket_coverage_ratio": (
            observed_bucket_count / source_bucket_count
        ),
        "technical_segment_locomotion_coverage_ratio": 1.0,
        "proposal_duration_seconds": sum(
            float(row["duration_sec"])
            for row in proposals
            if row["proposal_status"] != "rejected_by_qc"
        ),
        "uncertain_count": status_counts["uncertain"],
        "manual_review_required_count": len(proposals),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stationary_dwell_candidate_seconds": 15.0,
        "stationary_dwell_candidate_basis": (
            "protocol_derived_about_15_seconds"
        ),
        "stationary_dwell_candidate_validated": False,
        "movement_start_candidate_validated": False,
        "truth_labels_consumed_by_producer": False,
        "shape_predictions_consumed_by_producer": False,
        "frozen_model_loaded": False,
        "models_trained_or_updated": False,
        "automatic_boundary_validated": False,
        "boundary_metrics_available": False,
        "shape_metrics_available": False,
        "manual_acceptance_required": True,
    }


def _write_bundle(
    directory: Path,
    *,
    media: dict[str, object],
    proposals: list[dict[str, object]],
) -> Path:
    directory.mkdir(parents=True)
    keys = sorted(
        {
            (int(row["track_id"]), int(row["technical_segment_index"]))
            for row in proposals
        }
    )
    diagnostics = [
        _diagnostic(
            proposals,
            media=media,
            track_id=track_id,
            technical_segment_index=segment_index,
        )
        for track_id, segment_index in keys
    ]
    (directory / "proposals.jsonl").write_bytes(canonical_jsonl_bytes(proposals))
    (directory / "diagnostics.jsonl").write_bytes(
        canonical_jsonl_bytes(diagnostics)
    )
    (directory / "summary.json").write_bytes(
        canonical_json_bytes(_summary(proposals, diagnostics, media=media))
    )
    return directory


def _write_batch(
    root: Path,
    *,
    bundle_id: str,
    media: dict[str, object],
    proposals: list[dict[str, object]],
    boundaries: list[dict[str, object]],
    participant_id: str,
    session_id: str,
    camera_setup_id: str,
    truth_source: str = "synthetic_fixture",
) -> dict[str, str]:
    directory = root / bundle_id
    directory.mkdir(parents=True)
    bundle = _write_bundle(
        directory / "proposal-bundle", media=media, proposals=proposals
    )
    boundary_path = directory / "human-boundaries.jsonl"
    boundary_path.write_bytes(canonical_jsonl_bytes(boundaries))
    sidecar = directory / "media-sidecar.json"
    sidecar.write_bytes(canonical_json_bytes(media))
    return {
        "bundle_id": bundle_id,
        "proposal_bundle_dir": str(bundle.relative_to(root)),
        "human_boundary_jsonl": str(boundary_path.relative_to(root)),
        "media_sidecar": str(sidecar.relative_to(root)),
        "participant_id": participant_id,
        "session_id": session_id,
        "camera_setup_id": camera_setup_id,
        "clock_domain_id": f"clock-{session_id}",
        "truth_source": truth_source,
    }


def _complex_fixture(root: Path) -> Path:
    media_a = _media()
    proposals_a = [
        _proposal("p1", 0.5, 9.5, "proposed", media=media_a),
        _proposal("p2", 12.0, 22.0, "uncertain", media=media_a),
        _proposal("r1", 24.0, 34.0, "rejected_by_qc", media=media_a),
        _proposal("p3", 35.0, 41.0, "proposed", media=media_a),
        _proposal("p4", 41.0, 47.0, "proposed", media=media_a),
        _proposal("p5", 48.0, 58.0, "proposed", media=media_a),
        _proposal("p6", 79.0, 83.0, "proposed", media=media_a),
        _proposal("p7", 112.0, 116.0, "proposed", media=media_a),
        _proposal("p8", 5.0, 10.0, "uncertain", media=media_a, track_id=2),
    ]
    boundaries_a = [
        _boundary("t1", 0.0, 10.0, media=media_a),
        _boundary("t2", 12.0, 22.0, media=media_a),
        _boundary("t3", 24.0, 34.0, media=media_a),
        _boundary("t4", 36.0, 46.0, media=media_a),
        _boundary("t5", 48.0, 52.0, media=media_a),
        _boundary("t6", 54.0, 58.0, media=media_a),
        _boundary("t7", 70.0, 80.0, media=media_a),
        _boundary("t8", 90.0, 108.0, media=media_a),
        _boundary(
            "tu1", 5.0, 10.0, media=media_a, track_id=2,
            status="boundary_uncertain",
        ),
    ]
    row_a = _write_batch(
        root,
        bundle_id="bundle-a",
        media=media_a,
        proposals=proposals_a,
        boundaries=boundaries_a,
        participant_id="participant-a",
        session_id="session-a",
        camera_setup_id="camera-setup-a",
    )

    media_b = _media(
        source_video_id="synthetic-boundary-002",
        source_group_id="synthetic-group-002",
        setup_id="setup-002",
        duration_sec=60.0,
    )
    proposals_b = [_proposal("p9", 0.0, 50.0, "proposed", media=media_b)]
    boundaries_b = [_boundary("t9", 0.0, 50.0, media=media_b)]
    row_b = _write_batch(
        root,
        bundle_id="bundle-b",
        media=media_b,
        proposals=proposals_b,
        boundaries=boundaries_b,
        participant_id="participant-b",
        session_id="session-b",
        camera_setup_id="camera-setup-b",
    )
    index = root / "batch-index.jsonl"
    index.write_bytes(canonical_jsonl_bytes([row_a, row_b]))
    return index


def _full_scope(
    *,
    track_id: int = 1,
    session_id: str = "session-a",
) -> dict[str, object]:
    return {
        "source_group_id": "synthetic-group-001",
        "source_video_id": "synthetic-boundary-001",
        "device_id": "camera-001",
        "setup_id": "setup-001",
        "stream_epoch": "epoch-001",
        "track_id": track_id,
        "participant_id": "participant-a",
        "session_id": session_id,
        "camera_setup_id": "camera-setup-a",
        "clock_domain_id": "clock-session-a",
    }


def _bound_proposal(
    proposal_id: str, start: float, end: float, *, status: str = "proposed",
    technical_segment_index: int = 0,
    **scope: object,
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "proposal_status": status,
        "technical_segment_index": technical_segment_index,
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "reason_codes": ["synthetic"],
        "hard_break_reasons": [],
        "manual_review_required": True,
        **_full_scope(**scope),
    }


def _bound_truth(
    episode_id: str, start: float, end: float, *, status: str = "ready",
    **scope: object,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "boundary_status": status,
        "boundary_reason_codes": [] if status == "ready" else ["uncertain"],
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        **_full_scope(**scope),
    }


def test_config_and_exact_loaders_keep_proposal_separate_from_boundary(
    tmp_path: Path,
) -> None:
    config = load_camera_episode_boundary_evaluation_config(CONFIG_PATH)
    assert config["matching_policy"]["validated"] is True
    assert config["data_access"] == {
        "shape_truth": False,
        "shape_predictions": False,
        "frozen_model": False,
        "training": False,
        "sealed_camera": False,
    }
    assert BOUNDARY_EVALUATION_INDEX_FIELDS == frozenset(
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

    index = _complex_fixture(tmp_path)
    first_row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    proposal_path = tmp_path / first_row["proposal_bundle_dir"] / "proposals.jsonl"
    media = json.loads(
        (tmp_path / first_row["media_sidecar"]).read_text(encoding="utf-8")
    )
    with pytest.raises(CameraEpisodeImportError, match="boundary fields"):
        load_episode_boundaries(proposal_path, media)

    build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=index,
        output_dir=tmp_path / "output",
    )

    proposal_rows = [
        json.loads(line)
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
    ]
    proposal_rows[0]["observable_pattern"] = "direct"
    proposal_path.write_bytes(canonical_jsonl_bytes(proposal_rows))
    with pytest.raises(CameraEpisodeBoundaryEvaluationError, match="proposal fields"):
        build_camera_episode_boundary_evaluation_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=index,
            output_dir=tmp_path / "invalid-output",
        )


def test_independent_human_boundary_path_runs_and_rejects_whole_clip(
    tmp_path: Path,
) -> None:
    media = _media(
        source_video_id="human-boundary-001",
        source_group_id="human-group-001",
        authorization_status="authorized_camera_labeled_evaluation",
    )
    proposal = _proposal("human-p1", 1.0, 11.0, "proposed", media=media)
    boundary = _boundary("human-t1", 0.0, 10.0, media=media)
    row = _write_batch(
        tmp_path,
        bundle_id="human-bundle",
        media=media,
        proposals=[proposal],
        boundaries=[boundary],
        participant_id="participant-human",
        session_id="session-human",
        camera_setup_id="camera-setup-human",
        truth_source="independent_human",
    )
    index = tmp_path / "human-index.jsonl"
    index.write_bytes(canonical_jsonl_bytes([row]))
    output = tmp_path / "human-evaluation"

    build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=index,
        output_dir=output,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert summary["truth_source"] == "independent_human"
    assert summary["evidence_scope"] == "authorized_labeled_development_evaluated"
    assert summary["automatic_boundary_evaluation_completed"] is True
    assert summary["matching_policy_validated"] is True
    diagnostic_lines = (output / "split_merge_diagnostics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    diagnostics = [json.loads(line) for line in diagnostic_lines if line]
    assert all(row["matching_policy_validated"] is True for row in diagnostics)
    assert metrics["views"]["all_locomotion_candidates"]["detection"][
        "matched_count"
    ] == 1

    boundary_path = tmp_path / "human-bundle" / "human-boundaries.jsonl"
    whole_clip = dict(boundary)
    whole_clip["boundary_source"] = "whole_clip"
    boundary_path.write_bytes(canonical_jsonl_bytes([whole_clip]))
    with pytest.raises(
        CameraEpisodeBoundaryEvaluationError,
        match="continuous boundary cannot use whole_clip",
    ):
        build_camera_episode_boundary_evaluation_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=index,
            output_dir=tmp_path / "whole-clip-rejected",
        )


def test_two_views_metrics_localization_group_and_duration_support(
    tmp_path: Path,
) -> None:
    index = _complex_fixture(tmp_path)
    output = tmp_path / "evaluation"
    result = build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=index,
        output_dir=output,
    )

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    candidate = metrics["views"]["all_locomotion_candidates"]
    proposed = metrics["views"]["proposed_only_conditional"]
    assert result.ready_truth_count == 9
    assert result.uncertain_truth_count == 1
    assert candidate["detection"] == {
        "ready_truth_support": 9,
        "candidate_support": 9,
        "matched_count": 5,
        "unmatched_truth_count": 4,
        "unmatched_candidate_count": 4,
        "precision": pytest.approx(5 / 9),
        "recall": pytest.approx(5 / 9),
        "f1": pytest.approx(5 / 9),
    }
    assert proposed["detection"]["ready_truth_support"] == 9
    assert proposed["detection"]["candidate_support"] == 7
    assert proposed["detection"]["matched_count"] == 4
    assert proposed["detection"]["recall"] == pytest.approx(4 / 9)
    assert candidate["localization"]["temporal_iou"]["count"] == 5
    assert candidate["localization"]["onset_signed_error_sec"]["mean"] == (
        pytest.approx(-0.1)
    )
    assert candidate["localization"]["offset_signed_error_sec"]["mean"] == (
        pytest.approx(0.1)
    )
    assert set(metrics["group_support"]["participant"]) == {
        "participant-a",
        "participant-b",
    }
    assert metrics["group_support"]["participant"]["participant-b"][
        "views"
    ]["all_locomotion_candidates"]["detection"]["matched_count"] == 1
    assert metrics["duration_support"]["long"]["ready_truth_support"] == 1
    assert metrics["duration_support"]["medium"]["ready_truth_support"] == 1


def test_duration_band_f1_is_not_combined_across_distinct_cohorts() -> None:
    config = load_camera_episode_boundary_evaluation_config(CONFIG_PATH)
    proposals = [
        _bound_proposal("long-proposal", 0.0, 45.0),
        _bound_proposal("medium-proposal", 60.0, 90.0),
    ]
    boundaries = [
        _bound_truth("medium-truth", 0.0, 20.0),
        _bound_truth("long-truth", 60.0, 110.0),
    ]

    artifacts = evaluate_camera_episode_boundaries(
        proposals,
        boundaries,
        config=config,
    )

    medium = artifacts["metrics"]["duration_support"]["medium"]["views"][
        "all_locomotion_candidates"
    ]
    assert medium["precision"] == 1.0
    assert medium["recall"] == 1.0
    assert medium["same_duration_band_match_count"] == 0
    assert medium["cross_duration_band_match_count_by_truth"] == 1
    assert medium["cross_duration_band_match_count_by_proposal"] == 1
    assert medium["f1"] == "not_computable"
    assert medium["f1_reason"] == (
        "precision_and_recall_use_distinct_duration_cohorts_with_cross_band_matches"
    )


def test_fragmentation_diagnostic_does_not_cross_technical_hard_breaks() -> None:
    config = load_camera_episode_boundary_evaluation_config(CONFIG_PATH)
    left = _bound_proposal(
        "left-segment",
        0.0,
        10.0,
        technical_segment_index=0,
    )
    right = _bound_proposal(
        "right-segment",
        10.1,
        20.0,
        technical_segment_index=1,
    )
    left["hard_break_reasons"] = ["long_internal_gap"]
    right["hard_break_reasons"] = ["long_internal_gap"]

    artifacts = evaluate_camera_episode_boundaries(
        [left, right],
        [],
        config=config,
    )

    assert not any(
        row["diagnostic_type"] == "fragmentation_candidate"
        for row in artifacts["diagnostics"]
    )


def test_shared_maximum_cardinality_matcher_and_scope_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation as evaluator

    calls = 0
    shared = evaluator.match_episodes_one_to_one

    def spy(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return shared(*args, **kwargs)

    monkeypatch.setattr(evaluator, "match_episodes_one_to_one", spy)
    proposals = [
        _bound_proposal("P1", 0.0, 15.0),
        _bound_proposal("P2", 0.0, 5.1),
        _bound_proposal("cross-session", 0.0, 10.0, session_id="session-b"),
    ]
    boundaries = [
        _bound_truth("A1", 0.0, 10.0),
        _bound_truth("A2", 5.0, 15.0),
    ]
    config = load_camera_episode_boundary_evaluation_config(CONFIG_PATH)

    artifacts = evaluate_camera_episode_boundaries(
        proposals,
        boundaries,
        config=config,
    )

    main = artifacts["metrics"]["views"]["all_locomotion_candidates"]
    assert calls == 2
    assert main["detection"]["matched_count"] == 2
    assert main["detection"]["unmatched_candidate_count"] == 1
    assert {
        row["proposal_id"]
        for row in artifacts["unmatched_proposals"]
        if row["view"] == "all_locomotion_candidates"
    } == {"cross-session"}


def test_uncertain_rejected_and_human_uncertain_remain_visible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=_complex_fixture(tmp_path),
        output_dir=output,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    coverage = metrics["coverage"]
    assert coverage["human_boundary_status_counts"] == {
        "ready": 9,
        "boundary_uncertain": 1,
    }
    assert coverage["proposal_status_counts"] == {
        "proposed": 7,
        "uncertain": 2,
        "rejected_by_qc": 1,
    }
    assert coverage["ready_truth_overlap_status_counts"] == {
        "proposed_overlap": 6,
        "uncertain_only_overlap": 1,
        "rejected_by_qc_only_overlap": 1,
        "no_proposal_overlap": 1,
    }
    assert coverage["manual_review_required_count"] == 10
    failures = [
        json.loads(line)
        for line in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failure_types = {row["failure_type"] for row in failures}
    assert "human_boundary_uncertain" in failure_types
    assert "uncertain_only_truth_coverage" in failure_types
    assert "rejected_by_qc_truth_coverage" in failure_types
    assert any(
        row["failure_type"] == "unmatched_ready_truth"
        and row["view"] == "proposed_only_conditional"
        and row["episode_id"] == "t2"
        for row in failures
    )


def test_split_merge_subthreshold_fragmentation_and_failures_are_row_level(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=_complex_fixture(tmp_path),
        output_dir=output,
    )
    diagnostics = [
        json.loads(line)
        for line in (output / "split_merge_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    types = {row["diagnostic_type"] for row in diagnostics}
    assert {
        "split_candidate",
        "merge_candidate",
        "subthreshold_overlap",
        "fragmentation_candidate",
    }.issubset(types)
    assert any(
        row["diagnostic_type"] == "split_candidate"
        and row["episode_ids"] == ["t4"]
        and row["proposal_ids"] == ["p3", "p4"]
        for row in diagnostics
    )
    assert any(
        row["diagnostic_type"] == "merge_candidate"
        and row["proposal_ids"] == ["p5"]
        and row["episode_ids"] == ["t5", "t6"]
        for row in diagnostics
    )
    assert any(
        row["diagnostic_type"] == "subthreshold_overlap"
        and row["proposal_ids"] == ["p6"]
        and row["episode_ids"] == ["t7"]
        for row in diagnostics
    )


def test_empty_matched_cohort_is_not_computable() -> None:
    config = load_camera_episode_boundary_evaluation_config(CONFIG_PATH)
    artifacts = evaluate_camera_episode_boundaries(
        [_bound_proposal("P", 20.0, 25.0)],
        [_bound_truth("A", 0.0, 10.0)],
        config=config,
    )
    localization = artifacts["metrics"]["views"][
        "all_locomotion_candidates"
    ]["localization"]
    assert localization["temporal_iou"] == {
        "count": 0,
        "mean": "not_computable",
        "median": "not_computable",
        "p90": "not_computable",
        "max": "not_computable",
    }
    assert artifacts["metrics"]["views"]["all_locomotion_candidates"][
        "detection"
    ]["f1"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_proposal_id", "duplicate proposal_id"),
        ("duplicate_episode_id", "duplicate episode_id"),
        ("overlapping_ready_truth", "overlapping episode boundaries"),
        ("summary_identity_drift", "identity"),
        ("nonfinite_proposal", "finite"),
    ],
)
def test_invalid_identity_ids_intervals_and_numbers_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    index = _complex_fixture(tmp_path)
    row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    bundle = tmp_path / row["proposal_bundle_dir"]
    proposal_path = bundle / "proposals.jsonl"
    boundary_path = tmp_path / row["human_boundary_jsonl"]
    if mutation == "duplicate_proposal_id":
        proposals = [json.loads(line) for line in proposal_path.read_text().splitlines()]
        proposals[1]["proposal_id"] = proposals[0]["proposal_id"]
        proposal_path.write_bytes(canonical_jsonl_bytes(proposals))
    elif mutation == "duplicate_episode_id":
        boundaries = [json.loads(line) for line in boundary_path.read_text().splitlines()]
        boundaries[1]["episode_id"] = boundaries[0]["episode_id"]
        boundary_path.write_bytes(canonical_jsonl_bytes(boundaries))
    elif mutation == "overlapping_ready_truth":
        boundaries = [json.loads(line) for line in boundary_path.read_text().splitlines()]
        boundaries[1]["start_sec"] = 9.0
        boundary_path.write_bytes(canonical_jsonl_bytes(boundaries))
    elif mutation == "summary_identity_drift":
        summary_path = bundle / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["source_group_id"] = "drifted"
        summary_path.write_bytes(canonical_json_bytes(summary))
    elif mutation == "nonfinite_proposal":
        payload = proposal_path.read_text(encoding="utf-8")
        proposal_path.write_text(
            payload.replace('"start_sec":0.5', '"start_sec":NaN', 1),
            encoding="utf-8",
        )
    with pytest.raises(CameraEpisodeBoundaryEvaluationError, match=message):
        build_camera_episode_boundary_evaluation_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=index,
            output_dir=tmp_path / "invalid-output",
        )
    assert not (tmp_path / "invalid-output").exists()


def test_deterministic_non_overwrite_and_source_files_unchanged(
    tmp_path: Path,
) -> None:
    index = _complex_fixture(tmp_path)
    source_files = [index]
    for row in map(json.loads, index.read_text(encoding="utf-8").splitlines()):
        bundle = tmp_path / row["proposal_bundle_dir"]
        source_files.extend(
            [
                bundle / "summary.json",
                bundle / "proposals.jsonl",
                bundle / "diagnostics.jsonl",
                tmp_path / row["human_boundary_jsonl"],
                tmp_path / row["media_sidecar"],
            ]
        )
    before = {path: path.read_bytes() for path in source_files}
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    first = build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=index,
        output_dir=output_a,
    )
    second = build_camera_episode_boundary_evaluation_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=index,
        output_dir=output_b,
    )
    assert first == second.__class__(output_dir=output_a, **{
        name: getattr(second, name)
        for name in (
            "ready_truth_count",
            "uncertain_truth_count",
            "proposal_count",
            "match_count",
            "failure_count",
        )
    })
    for relative in (
        "summary.json",
        "metrics.json",
        "matches.jsonl",
        "unmatched_truth.jsonl",
        "unmatched_proposals.jsonl",
        "split_merge_diagnostics.jsonl",
        "failures.jsonl",
        "README.md",
    ):
        assert (output_a / relative).read_bytes() == (output_b / relative).read_bytes()
    assert before == {path: path.read_bytes() for path in source_files}
    with pytest.raises(FileExistsError):
        build_camera_episode_boundary_evaluation_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=index,
            output_dir=output_a,
        )
    summary = json.loads((output_a / "summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_scope"] == "synthetic_contract_only"
    assert summary["truth_source"] == "synthetic_fixture"
    assert summary["matching_policy_validated"] is True
    assert summary["shape_predictions_consumed"] is False
    assert summary["shape_truth_consumed"] is False
    assert summary["proposal_modified"] is False
    assert summary["human_boundary_modified"] is False
    assert summary["automatic_shape_evaluated"] is False
    assert summary["models_trained_or_updated"] is False


def test_boundary_evaluation_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/evaluate_camera_episode_boundaries.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--batch-index" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--config" in completed.stdout
