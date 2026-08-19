from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    CAMERA_EPISODE_PIPELINE_CONFIG_SCHEMA_VERSION,
    _build_episode_result,
    _evaluate_automatic_binary,
    _validate_episode_result,
    load_camera_episode_pipeline_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_episode_pipeline_v1.yaml"
CLI = ROOT / "scripts/wandering/run_camera_episode_pipeline.py"


def _prediction(*, proposal_id: str, status: str, prediction_status: str) -> dict[str, object]:
    binary = None
    four_class = None
    if prediction_status == "ready":
        binary = {
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [0.1, 0.9],
            "predicted_label": "wandering_like",
        }
        four_class = {
            "class_order": ["direct", "pacing", "lapping", "random"],
            "probabilities": [0.1, 0.2, 0.3, 0.4],
            "predicted_label": "random",
        }
    return {
        "proposal_id": proposal_id,
        "source_video_id": "video-001",
        "track_id": 1,
        "technical_segment_index": 2,
        "start_sec": 3.0,
        "end_sec_exclusive": 9.0,
        "duration_sec": 6.0,
        "proposal_status": status,
        "reason_codes": ["sustained_movement"],
        "hard_break_reasons": [],
        "prediction_status": prediction_status,
        "prediction_reason_codes": [],
        "qc_status": "ready" if prediction_status == "ready" else "unavailable",
        "qc_reason_codes": [],
        "binary": binary,
        "four_class": four_class,
        "quality_flags": ["automatic_boundary_proposal"],
        "candidate_id": "candidate-001",
        "model_state_sha256": "1" * 64,
        "binary_decision_threshold": 0.5,
        "producer_config_id": "closing-s008-d04",
        "shape_inference_policy_id": "proposal-shape-policy-v1",
    }


def _index_row() -> dict[str, object]:
    return {
        "batch_id": "B01",
        "participant_id": "person-001",
        "session_id": "session-001",
        "camera_setup_id": "setup-001",
        "source_video_id": "video-001",
        "timezone": "Asia/Shanghai",
        "artifacts": {
            "tracking": {"path": "/inputs/tracking.jsonl", "sha256": "2" * 64},
            "media_sidecar": {"path": "/inputs/media_sidecar.json", "sha256": "3" * 64},
            "truth": {"path": "/inputs/episode_truth.jsonl", "sha256": "4" * 64},
        },
    }


def test_episode_result_preserves_endpoint_identity_and_separates_run_decision() -> None:
    row = _build_episode_result(
        prediction=_prediction(
            proposal_id="proposal-001",
            status="uncertain",
            prediction_status="ready",
        ),
        index_row=_index_row(),
        config_id="w5d02-test",
        config_sha256="5" * 64,
        profile_id="closing-s008-d04",
        profile_sha256="6" * 64,
    )

    assert row["schema_version"] == "wandering-handoff-episode-result-v1"
    assert row["module"] == "mental_health"
    assert row["episode_id"] == "proposal-001"
    assert row["technical_segment_index"] == 2
    assert (row["start_sec"], row["end_sec_exclusive"], row["duration_seconds"]) == (
        3.0,
        9.0,
        6.0,
    )
    assert row["proposal_status"] == "uncertain"
    assert row["status"] == "uncertain"
    assert row["run_status"] == "uncertain"
    assert row["qc_status"] == "ready"
    assert row["binary"]["decision_threshold"] == 0.5
    assert row["four_class"]["decision_threshold"] is None
    assert "automatic_boundary_not_human_accepted" in row["quality_flags"]
    assert row["identity"] == {
        "model_id": "candidate-001",
        "model_sha256": "1" * 64,
        "config_id": "w5d02-test",
        "config_sha256": "5" * 64,
        "policy_id": "closing-s008-d04",
        "policy_sha256": "6" * 64,
    }
    _validate_episode_result(row)

    invalid = dict(row)
    invalid["unexpected"] = True
    try:
        _validate_episode_result(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("schema drift must be rejected")


def test_rejected_proposal_is_unavailable_and_keeps_null_predictions() -> None:
    row = _build_episode_result(
        prediction=_prediction(
            proposal_id="proposal-rejected",
            status="rejected_by_qc",
            prediction_status="unavailable",
        ),
        index_row=_index_row(),
        config_id="w5d02-test",
        config_sha256="5" * 64,
        profile_id="closing-s008-d04",
        profile_sha256="6" * 64,
    )

    assert row["status"] == "unavailable"
    assert row["run_status"] == "rejected"
    assert row["binary"] is None
    assert row["four_class"] is None


def test_automatic_binary_reports_classification_and_pipeline_miss_separately() -> None:
    truths = [
        {
            "batch_id": "B01",
            "episode_id": "truth-1",
            "source_video_id": "video-001",
            "target_track_id": 1,
            "start_sec": 3.0,
            "end_sec_exclusive": 9.0,
            "observable_pattern": "pacing",
        },
        {
            "batch_id": "B01",
            "episode_id": "truth-2",
            "source_video_id": "video-001",
            "target_track_id": 1,
            "start_sec": 20.0,
            "end_sec_exclusive": 25.0,
            "observable_pattern": "direct",
        },
    ]
    predictions = [
        _prediction(
            proposal_id="proposal-001",
            status="proposed",
            prediction_status="ready",
        )
    ]
    matches = [
        {
            "view": "all_locomotion_candidates",
            "episode_id": "truth-1",
            "proposal_id": "proposal-001",
        }
    ]

    metrics = _evaluate_automatic_binary(
        truth_rows=truths,
        proposal_shape_rows=predictions,
        match_rows=matches,
        minimum_prediction_coverage=0.85,
    )

    assert metrics["pooled"]["known_episode_support"] == 2
    assert metrics["pooled"]["ready_uncertain_prediction_overlap_count"] == 1
    assert metrics["pooled"]["ready_uncertain_prediction_coverage"] == 0.5
    assert metrics["pooled"]["prediction_coverage_gate_satisfied"] is False
    assert metrics["pooled"]["pipeline_miss_count"] == 1
    assert metrics["pooled"]["automatic_binary"]["support"] == 1
    assert metrics["pooled"]["automatic_binary"]["accuracy"] == 1.0


def test_production_config_binds_contract_profile_schema_and_fixed_primary() -> None:
    config = load_camera_episode_pipeline_config(CONFIG)

    assert config["schema_version"] == CAMERA_EPISODE_PIPELINE_CONFIG_SCHEMA_VERSION
    assert config["pipeline_id"] == "w5d02-b01-b02-automatic-episode-shape-v1"
    assert config["development"]["expected_batch_counts"] == {"B01": 36, "B02": 12}
    assert config["segmenter"]["profile_id"] == "closing-s008-d04"
    assert config["fixed_primary"]["candidate_id"] == (
        "topowander-m0s-seed20260731-epoch0005"
    )
    assert config["fixed_primary"]["binary_decision_threshold"] == 0.5
    assert config["gates"]["known_episode_prediction_coverage_minimum"] == 0.85


def test_cli_help_exposes_integrated_inputs_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--batch-index" not in completed.stdout
