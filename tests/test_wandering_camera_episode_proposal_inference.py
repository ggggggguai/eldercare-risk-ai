from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from elderly_monitoring.modules.mental_health.wandering import (
    camera_episode_proposal_inference as proposal_inference_module,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION,
    build_camera_episode_proposal_inference_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_STATE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/modules/wandering_camera_episode_proposal_shape_v1.yaml"
)
CLI_PATH = ROOT / "scripts/wandering/run_camera_episode_proposal_inference.py"


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        model_features: torch.Tensor,
        shape_normalized_points: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert model_features.shape == (1, 80, 14)
        assert shape_normalized_points.shape == (1, 80, 2)
        assert point_mask.shape == (1, 80)
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        return {
            "binary_logit": torch.tensor([[2.0]], dtype=torch.float32),
            "subtype_logits": torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32),
            "projection_embedding": torch.zeros((1, 64), dtype=torch.float32),
        }


def _runtime(model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        manifest={
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "training_candidate_identity": {
                "identity": {"candidate": {"seed": 20260731, "best_epoch": 5}}
            },
            "artifacts": {"model_state": {"sha256": EXPECTED_MODEL_STATE_SHA256}},
        },
        manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )


def _proposal(
    proposal_id: str,
    *,
    start: float,
    end: float,
    status: str,
    reasons: list[str],
    producer_config_id: str = "m0cam-ep2a-s0-b01-development-frozen-v1",
) -> dict[str, object]:
    return {
        "schema_version": "wandering-camera-episode-boundary-proposal-v1",
        "proposal_id": proposal_id,
        "source_group_id": "synthetic-proposal-group",
        "source_video_id": "synthetic-proposal-video",
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
        "end_reason": (
            "sustained_stationary" if status == "proposed" else "manual_review"
        ),
        "reason_codes": reasons,
        "hard_break_reasons": [],
        "uncertain": status != "proposed",
        "manual_review_required": True,
        "source_observation_count": 20,
        "accepted_observation_count": 20,
        "source_bucket_count": int((end - start) / 0.5),
        "producer_config_id": producer_config_id,
    }


def _fixture(
    tmp_path: Path,
    *,
    producer_config_id: str = "m0cam-ep2a-s0-b01-development-frozen-v1",
) -> tuple[Path, list[dict[str, object]]]:
    rows = []
    for index in range(60):
        x = 128.0 + 320.0 * index / 59.0
        rows.append(
            {
                "frame_id": index * 10,
                "person_id": "anonymous_001",
                "track_id": 1,
                "bbox": [x - 40.0, 240.0, x + 40.0, 360.0],
                "scene_region": "office",
                "track_confidence": 0.9,
                "center": [x, 300.0],
                "speed_px_per_sec": None,
                "timestamp_sec": index * 0.5 + 0.1,
            }
        )
    tracking_bytes = canonical_jsonl_bytes(rows)
    tracking = tmp_path / "tracking.jsonl"
    tracking.write_bytes(tracking_bytes)
    media = {
        "schema_version": "wandering-media-v1",
        "source_video_id": "synthetic-proposal-video",
        "source_group_id": "synthetic-proposal-group",
        "device_id": "camera-001",
        "setup_id": "setup-001",
        "stream_epoch": "epoch-001",
        "media_ref": "synthetic/proposal-video.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": hashlib.sha256(tracking_bytes).hexdigest(),
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 20.0,
        "duration_sec": 30.0,
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
            "proposal-ready",
            start=0.0,
            end=15.0,
            status="proposed",
            reasons=["sustained_movement", "sustained_stationary"],
            producer_config_id=producer_config_id,
        ),
        _proposal(
            "proposal-uncertain",
            start=15.0,
            end=22.0,
            status="uncertain",
            reasons=["offset_requires_manual_review"],
            producer_config_id=producer_config_id,
        ),
        _proposal(
            "proposal-rejected",
            start=22.0,
            end=30.0,
            status="rejected_by_qc",
            reasons=["insufficient_locomotion_evidence"],
            producer_config_id=producer_config_id,
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
                "producer_config_id": producer_config_id,
                "proposal_is_accepted_boundary": False,
                "source_group_id": media["source_group_id"],
                "source_video_id": media["source_video_id"],
                "device_id": media["device_id"],
                "setup_id": media["setup_id"],
                "stream_epoch": media["stream_epoch"],
                "proposal_count": len(proposals),
                "proposal_status_counts": {
                    "proposed": 1,
                    "uncertain": 1,
                    "rejected_by_qc": 1,
                },
            }
        )
    )
    index = tmp_path / "batch_index.jsonl"
    index.write_bytes(
        canonical_jsonl_bytes(
            [
                {
                    "bundle_id": "synthetic-bundle",
                    "proposal_bundle_dir": str(proposal_bundle),
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
    return index, proposals


def test_proposed_and_uncertain_invoke_model_without_accepting_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_index, proposals = _fixture(tmp_path)
    model = _CountingModel()
    monkeypatch.setattr(
        proposal_inference_module,
        "load_primary_camera_runtime",
        lambda **_kwargs: _runtime(model),
    )
    output = tmp_path / "proposal-shape"

    result = build_camera_episode_proposal_inference_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=batch_index,
        output_dir=output,
    )

    predictions = [
        json.loads(line)
        for line in (output / "proposal_shape_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.proposal_count == len(proposals) == len(predictions)
    assert model.calls == 2
    assert [row["prediction_status"] for row in predictions] == [
        "ready",
        "ready",
        "unavailable",
    ]
    assert [row["model_invocation_skipped"] for row in predictions] == [
        False,
        False,
        True,
    ]
    for source, observed in zip(proposals, predictions, strict=True):
        for name in (
            "proposal_id",
            "technical_segment_index",
            "start_sec",
            "end_sec_exclusive",
            "duration_sec",
            "proposal_status",
            "start_reason",
            "end_reason",
            "reason_codes",
            "hard_break_reasons",
            "manual_review_required",
        ):
            assert observed[name] == source[name]
        assert observed["schema_version"] == (
            CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION
        )
        assert "automatic_boundary_proposal" in observed["quality_flags"]
    assert predictions[0]["binary"]["predicted_label"] == "wandering_like"
    assert predictions[1]["proposal_status"] == "uncertain"
    assert predictions[1]["uncertain"] is True
    assert predictions[1]["binary"]["predicted_label"] == "wandering_like"
    assert "boundary_uncertain_candidate" in predictions[1]["quality_flags"]
    assert "proposal_rejected_by_qc" in predictions[2]["prediction_reason_codes"]
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    assert "oracle_boundary" not in serialized
    assert "wandering-camera-episode-boundary-v1" not in serialized
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "proposal_shape_predictions.jsonl",
        "summary.json",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["proposal_count"] == summary["result_count"] == 3
    assert summary["model_forward_invocation_count"] == 2
    assert summary["shape_inference_policy_id"] == (
        "m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2"
    )
    assert summary["forward_proposal_statuses"] == ["proposed", "uncertain"]
    assert summary["model_forward_invocation_count_by_proposal_status"] == {
        "proposed": 1,
        "uncertain": 1,
        "rejected_by_qc": 0,
    }
    assert summary["shape_performance_metrics_available"] is False

    with pytest.raises(FileExistsError):
        build_camera_episode_proposal_inference_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=batch_index,
            output_dir=output,
        )


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


def test_bundle_summary_can_bind_the_selected_w5d01_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_index, _ = _fixture(tmp_path, producer_config_id="closing-s008-d04")
    model = _CountingModel()
    monkeypatch.setattr(
        proposal_inference_module,
        "load_primary_camera_runtime",
        lambda **_kwargs: _runtime(model),
    )

    result = build_camera_episode_proposal_inference_bundle(
        project_root=ROOT,
        config_path=CONFIG_PATH,
        batch_index_path=batch_index,
        output_dir=tmp_path / "selected-profile-shape",
    )

    assert result.proposal_count == 3
    assert result.model_forward_invocation_count == 2


def test_bundle_rejects_producer_id_that_differs_from_its_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_index, _ = _fixture(tmp_path, producer_config_id="closing-s008-d04")
    proposal_path = tmp_path / "proposal-bundle/proposals.jsonl"
    proposals = [
        json.loads(line)
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
    ]
    proposals[0]["producer_config_id"] = "different-profile"
    proposal_path.write_bytes(canonical_jsonl_bytes(proposals))
    load_calls = 0

    def _unexpected_runtime(**_kwargs: object) -> SimpleNamespace:
        nonlocal load_calls
        load_calls += 1
        return _runtime(_CountingModel())

    monkeypatch.setattr(
        proposal_inference_module,
        "load_primary_camera_runtime",
        _unexpected_runtime,
    )

    with pytest.raises(
        proposal_inference_module.CameraEpisodeProposalInferenceError,
        match="producer",
    ):
        build_camera_episode_proposal_inference_bundle(
            project_root=ROOT,
            config_path=CONFIG_PATH,
            batch_index_path=batch_index,
            output_dir=tmp_path / "producer-mismatch",
        )
    assert load_calls == 0
