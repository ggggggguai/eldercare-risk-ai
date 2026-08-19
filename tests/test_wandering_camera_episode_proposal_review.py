from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_review import (
    CameraEpisodeProposalReviewError,
    build_camera_episode_proposal_review_pack,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/modules/wandering_camera_episode_proposal_review_v1.yaml"
)
CLI_PATH = ROOT / "scripts/wandering/build_camera_episode_proposal_review_pack.py"
HUMAN_FIELDS = (
    "human_boundary_decision",
    "corrected_start_sec",
    "corrected_end_sec",
    "observable_pattern",
    "purpose_context",
    "review_notes",
)


def _proposal(
    proposal_id: str,
    *,
    start: float,
    end: float,
    status: str,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "wandering-camera-episode-boundary-proposal-v1",
        "proposal_id": proposal_id,
        "source_group_id": "synthetic-review-group",
        "source_video_id": "synthetic-review-video",
        "device_id": "camera-001",
        "setup_id": "setup-001",
        "stream_epoch": "epoch-001",
        "track_id": 1,
        "technical_segment_index": 0,
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_sec": end - start,
        "proposal_status": status,
        "start_reason": "sustained_movement",
        "end_reason": "manual_review" if status != "proposed" else "stationary",
        "reason_codes": reasons,
        "hard_break_reasons": [],
        "uncertain": status != "proposed",
        "manual_review_required": True,
        "source_observation_count": 20,
        "accepted_observation_count": 20,
        "source_bucket_count": int((end - start) / 0.5),
        "producer_config_id": "m0cam-ep2a-s0-b01-development-frozen-v1",
    }


def _prediction(
    proposal: dict[str, object],
    *,
    prediction_status: str,
) -> dict[str, object]:
    ready = prediction_status == "ready"
    binary = (
        {
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [0.2, 0.8],
            "predicted_label": "wandering_like",
        }
        if ready
        else None
    )
    four_class = (
        {
            "class_order": ["direct", "pacing", "lapping", "random"],
            "probabilities": [0.2, 0.5, 0.2, 0.1],
            "predicted_label": "pacing",
        }
        if ready
        else None
    )
    return {
        "schema_version": "wandering-camera-episode-proposal-shape-prediction-v1",
        "bundle_id": "synthetic-review-bundle",
        "participant_id": "synthetic-participant",
        "session_id": "synthetic-session",
        "camera_setup_id": "setup-001",
        "clock_domain_id": "synthetic-clock",
        **{
            key: value
            for key, value in proposal.items()
            if key != "schema_version"
        },
        "prediction_status": prediction_status,
        "prediction_reason_codes": [] if ready else list(proposal["reason_codes"]),
        "binary": binary,
        "subtype": None,
        "four_class": four_class,
        "predicted_binary": "wandering_like" if ready else None,
        "predicted_pattern": "pacing" if ready else None,
        "candidate_id": "topowander-m0s-seed20260731-epoch0005",
        "candidate_manifest_sha256": "3" * 64,
        "model_state_sha256": "9" * 64,
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "model_invocation_skipped": not ready,
    }


def _fixture(
    tmp_path: Path,
    *,
    statuses: tuple[str, str] = ("proposed", "uncertain"),
) -> tuple[Path, list[Path]]:
    rows = []
    for index in range(40):
        x = 80.0 + 400.0 * index / 39.0
        y = 300.0 + 40.0 * ((index % 8) / 7.0)
        rows.append(
            {
                "frame_id": index * 10,
                "person_id": "anonymous_001",
                "track_id": 1,
                "bbox": [x - 30.0, y - 120.0, x + 30.0, y],
                "scene_region": "office",
                "track_confidence": 0.9,
                "center": [x, y - 60.0],
                "speed_px_per_sec": None,
                "timestamp_sec": index * 0.5 + 0.1,
            }
        )
    tracking_bytes = canonical_jsonl_bytes(rows)
    tracking = tmp_path / "tracking.jsonl"
    tracking.write_bytes(tracking_bytes)
    media = {
        "schema_version": "wandering-media-v1",
        "source_video_id": "synthetic-review-video",
        "source_group_id": "synthetic-review-group",
        "device_id": "camera-001",
        "setup_id": "setup-001",
        "stream_epoch": "epoch-001",
        "media_ref": "synthetic/review-video.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": hashlib.sha256(tracking_bytes).hexdigest(),
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 20.0,
        "duration_sec": 20.0,
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
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "anonymous",
    }
    sidecar = tmp_path / "media_sidecar.json"
    sidecar.write_bytes(canonical_json_bytes(media))

    proposals = [
        _proposal(
            "proposal-first",
            start=0.0,
            end=10.0,
            status=statuses[0],
            reasons=["sustained_movement", "sustained_stationary"],
        ),
        _proposal(
            "proposal-second",
            start=10.0,
            end=20.0,
            status=statuses[1],
            reasons=["offset_requires_manual_review"],
        ),
    ]
    proposal_bundle = tmp_path / "proposal-bundle"
    proposal_bundle.mkdir()
    (proposal_bundle / "proposals.jsonl").write_bytes(canonical_jsonl_bytes(proposals))
    (proposal_bundle / "diagnostics.jsonl").write_bytes(canonical_jsonl_bytes([]))
    (proposal_bundle / "summary.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "wandering-camera-episode-boundary-proposal-summary-v1",
                "proposal_schema_version": "wandering-camera-episode-boundary-proposal-v1",
                "proposal_is_accepted_boundary": False,
                "source_group_id": media["source_group_id"],
                "source_video_id": media["source_video_id"],
                "device_id": media["device_id"],
                "setup_id": media["setup_id"],
                "stream_epoch": media["stream_epoch"],
                "proposal_count": len(proposals),
            }
        )
    )

    predictions = []
    for proposal in proposals:
        status = str(proposal["proposal_status"])
        prediction_status = {
            "proposed": "ready",
            "uncertain": "ready",
            "rejected_by_qc": "unavailable",
        }[status]
        predictions.append(
            _prediction(proposal, prediction_status=prediction_status)
        )
    proposal_shape_bundle = tmp_path / "proposal-shape-bundle"
    proposal_shape_bundle.mkdir()
    (proposal_shape_bundle / "proposal_shape_predictions.jsonl").write_bytes(
        canonical_jsonl_bytes(predictions)
    )
    (proposal_shape_bundle / "summary.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "wandering-camera-episode-proposal-shape-summary-v1",
                "input_boundary_kind": "automatic_proposal",
                "proposal_schema_version": "wandering-camera-episode-boundary-proposal-v1",
                "proposal_count": len(proposals),
                "result_count": len(predictions),
                "model_forward_invocation_count": sum(
                    not row["model_invocation_skipped"] for row in predictions
                ),
                "model_invocation_skipped_count": sum(
                    row["model_invocation_skipped"] for row in predictions
                ),
                "proposal_endpoint_modified": False,
                "accepted_boundary_emitted": False,
                "truth_labels_consumed": False,
                "shape_performance_metrics_available": False,
            }
        )
    )
    (proposal_shape_bundle / "README.md").write_text(
        "# Synthetic proposal shape\n", encoding="utf-8"
    )

    index = tmp_path / "review_batch_index.jsonl"
    index.write_bytes(
        canonical_jsonl_bytes(
            [
                {
                    "bundle_id": "synthetic-review-bundle",
                    "proposal_bundle_dir": str(proposal_bundle),
                    "proposal_shape_bundle_dir": str(proposal_shape_bundle),
                    "tracking_jsonl": str(tracking),
                    "media_sidecar": str(sidecar),
                    "participant_id": "synthetic-participant",
                    "session_id": "synthetic-session",
                    "camera_setup_id": "setup-001",
                    "clock_domain_id": "synthetic-clock",
                }
            ]
        )
    )
    return index, [proposal_bundle / "proposals.jsonl", proposal_shape_bundle / "proposal_shape_predictions.jsonl", tracking, sidecar]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_builds_two_proposal_review_pack_with_blank_human_template(
    tmp_path: Path,
) -> None:
    batch_index, input_paths = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in input_paths}
    output = tmp_path / "review-pack"

    result = build_camera_episode_proposal_review_pack(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=batch_index,
        output_dir=output,
    )

    assert result.proposal_count == result.trajectory_plot_count == 2
    assert result.model_forward_invocation_count == 2
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "human_review_template.csv",
        "per_proposal_trajectory_plots",
        "per_video_summary.md",
        "review_index.csv",
    }
    rows = _read_csv(output / "review_index.csv")
    assert len(rows) == 2
    assert rows[0]["source_video_id"] == "synthetic-review-video"
    assert rows[0]["proposal_id"] == "proposal-first"
    assert rows[0]["start_sec"] == "0.000000"
    assert rows[0]["end_sec_exclusive"] == "10.000000"
    assert rows[0]["prediction_status"] == "ready"
    assert rows[0]["predicted_binary"] == "wandering_like"
    assert rows[0]["predicted_pattern"] == "pacing"
    assert json.loads(rows[0]["binary_probabilities"]) == [0.2, 0.8]
    assert rows[1]["proposal_status"] == "uncertain"
    assert rows[1]["prediction_status"] == "ready"
    assert json.loads(rows[1]["binary_probabilities"]) == [0.2, 0.8]
    assert all(row["needs_human_review"] == "true" for row in rows)
    for row in rows:
        plot = output / row["trajectory_plot_path"]
        assert plot.is_file()
        svg = plot.read_text(encoding="utf-8")
        assert row["source_video_id"] in svg
        assert row["proposal_id"] in svg
        assert row["proposal_status"] in svg
        assert "marker-end=\"url(#direction-arrow)\"" in svg

    template = _read_csv(output / "human_review_template.csv")
    assert len(template) == 2
    assert all(row[field] == "" for row in template for field in HUMAN_FIELDS)
    assert template[0]["observable_pattern"] != rows[0]["predicted_pattern"]
    assert all(path.read_bytes() == before[path] for path in input_paths)
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file()
    )
    assert "oracle_boundary" not in serialized
    assert "wandering-camera-episode-boundary-v1" not in serialized
    assert "performance metrics: not computed" in (
        output / "README.md"
    ).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_camera_episode_proposal_review_pack(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=batch_index,
            output_dir=output,
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_scope"])
def test_proposal_shape_join_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    batch_index, input_paths = _fixture(tmp_path)
    prediction_path = input_paths[1]
    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    if failure == "missing":
        predictions.pop()
    elif failure == "duplicate":
        predictions[1] = dict(predictions[0])
    else:
        predictions[1]["source_video_id"] = "wrong-video"
    prediction_path.write_bytes(canonical_jsonl_bytes(predictions))
    output = tmp_path / "rejected-review-pack"

    with pytest.raises(CameraEpisodeProposalReviewError):
        build_camera_episode_proposal_review_pack(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=batch_index,
            output_dir=output,
        )

    assert not output.exists()


def test_uncertain_candidate_keeps_probabilities_while_rejected_stays_skipped(
    tmp_path: Path,
) -> None:
    batch_index, _inputs = _fixture(
        tmp_path,
        statuses=("uncertain", "rejected_by_qc"),
    )
    output = tmp_path / "skip-review-pack"

    result = build_camera_episode_proposal_review_pack(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=batch_index,
        output_dir=output,
    )

    rows = _read_csv(output / "review_index.csv")
    assert [row["prediction_status"] for row in rows] == [
        "ready",
        "unavailable",
    ]
    assert json.loads(rows[0]["binary_probabilities"]) == [0.2, 0.8]
    assert rows[1]["binary_probabilities"] == ""
    assert all((output / row["trajectory_plot_path"]).is_file() for row in rows)
    assert result.model_forward_invocation_count == 1


def test_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--batch-index" in completed.stdout
    assert "--output-dir" in completed.stdout
