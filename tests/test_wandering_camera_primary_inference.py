from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering import (
    camera_primary_inference as primary_inference_module,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION,
    CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION,
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    PrimaryCameraInferenceError,
    build_primary_camera_inference_bundle,
    load_primary_camera_config,
    load_primary_camera_runtime,
    predict_primary_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.model import (
    hierarchical_four_class_probabilities,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_primary_v1.yaml"
MANIFEST = (
    ROOT
    / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1"
    / "artifacts/topowander_m0r_candidate_v3/candidate_manifest.json"
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _tracking_row(bucket: int, *, track_id: int = 1, moving: bool = True) -> dict[str, object]:
    phase = bucket / 79.0
    center_x = 220.0 + (140.0 * np.sin(phase * 2.0 * np.pi) if moving else 0.0)
    center_y = 300.0 + (35.0 * np.sin(phase * 4.0 * np.pi) if moving else 0.0)
    return {
        "frame_id": bucket * 10 + track_id,
        "track_id": track_id,
        "bbox": [center_x - 20.0, center_y - 100.0, center_x + 20.0, center_y],
        "track_confidence": 0.9,
        "timestamp_sec": bucket * 0.5 + 0.1,
    }


def _write_pair(directory: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    tracking = directory / "tracking.jsonl"
    payload = b"".join(_canonical_json(row) for row in rows)
    tracking.write_bytes(payload)
    sidecar = {
        "schema_version": "wandering-media-v1",
        "source_video_id": "video-0001",
        "source_group_id": "session-0001",
        "device_id": "camera-0001",
        "setup_id": "fixed-setup-0001",
        "stream_epoch": "epoch-0001",
        "media_ref": "synthetic/video-0001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 25.0,
        "duration_sec": 40.0,
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
        "camera_motion_state": "not_checked",
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic",
    }
    sidecar_path = directory / "sidecar.json"
    sidecar_path.write_bytes(_canonical_json(sidecar))
    return tracking, sidecar_path


def _prepared() -> dict[str, object]:
    features = np.zeros((80, 14), dtype=np.float32)
    points = np.stack(
        (np.linspace(-1.0, 1.0, 80), np.sin(np.linspace(0.0, 2.0 * np.pi, 80))),
        axis=1,
    ).astype(np.float32)
    mask = np.ones(80, dtype=np.float32)
    features[:, :2] = points
    features[:, 12] = mask
    features[:, 13] = 1.0
    return {
        "schema_version": "wandering-camera-window-v1",
        "window_id": "window-0001",
        "parent_tracklet_id": "tracklet-0001",
        "source_group_id": "session-0001",
        "source_video_id": "video-0001",
        "device_id": "camera-0001",
        "setup_id": "fixed-setup-0001",
        "stream_epoch": "epoch-0001",
        "track_id": 1,
        "window_start_sec": 0.0,
        "window_end_sec": 40.0,
        "window_status": "ready",
        "reason_codes": [],
        "quality_flags": ["camera_motion_not_verified"],
        "model_features": features.tolist(),
        "shape_normalized_points": points.tolist(),
        "point_mask": mask.tolist(),
    }


class _FakeModel(torch.nn.Module):
    def __init__(self, output: dict[str, torch.Tensor] | None = None, *, failure: bool = False) -> None:
        super().__init__()
        self.output = output
        self.failure = failure
        self.calls = 0

    def forward(
        self,
        model_features: torch.Tensor,
        shape_normalized_points: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        assert model_features.device.type == "cpu"
        assert model_features.dtype == shape_normalized_points.dtype == point_mask.dtype == torch.float32
        assert model_features.shape == (1, 80, 14)
        assert shape_normalized_points.shape == (1, 80, 2)
        assert point_mask.shape == (1, 80)
        self.calls += 1
        if self.failure:
            raise RuntimeError("synthetic trusted forward failure")
        if self.output is not None:
            return self.output
        return {
            "binary_logit": torch.tensor([[math.log(4.0)]], dtype=torch.float32),
            "subtype_logits": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
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
            "artifacts": {
                "model_state": {
                    "sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"
                }
            },
            "inference_contract": {
                "four_class_order": ["direct", "pacing", "lapping", "random"],
                "subtype_order": ["pacing", "lapping", "random"],
            },
        },
        manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )


def test_config_and_manifest_loader_bind_only_the_fixed_primary_candidate() -> None:
    config = load_primary_camera_config(CONFIG)
    assert config["evidence_scope"] == "synthetic_contract_only"
    assert config["candidate"]["manifest_sha256"] == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert config["candidate"]["model_state_sha256"] == (
        "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"
    )
    assert config["runtime"] == {
        "device": "cpu",
        "intra_op_threads": 8,
        "inter_op_threads": 1,
        "maximum_batch_size": 64,
    }
    runtime = load_primary_camera_runtime(
        config=config,
        manifest_path=MANIFEST,
        expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    assert runtime.model.training is False
    assert {parameter.device.type for parameter in runtime.model.parameters()} == {"cpu"}
    assert not hasattr(runtime, "camera_predict_logits")


def test_config_and_manifest_or_class_order_drift_fail_before_loader_or_forward(tmp_path: Path) -> None:
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bad = copy.deepcopy(base)
    bad["class_order"]["four_class"] = ["direct", "lapping", "pacing", "random"]
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad, sort_keys=False), encoding="utf-8")
    with pytest.raises(PrimaryCameraInferenceError):
        load_primary_camera_config(bad_path)

    config = load_primary_camera_config(CONFIG)
    loader = mock.Mock()
    with pytest.raises(PrimaryCameraInferenceError):
        load_primary_camera_runtime(
            config=config,
            manifest_path=MANIFEST,
            expected_manifest_sha256="0" * 64,
            candidate_loader=loader,
        )
    loader.assert_not_called()


def test_binary_decision_contract_drift_fails_before_model_forward() -> None:
    config = load_primary_camera_config(CONFIG)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["inference_contract"]["binary_decision"] = "sigmoid > 0.5"
    model = _FakeModel()
    runtime = SimpleNamespace(
        model=model,
        manifest=manifest,
        manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    loader = mock.Mock(return_value=runtime)

    with pytest.raises(PrimaryCameraInferenceError, match="binary decision"):
        load_primary_camera_runtime(
            config=config,
            manifest_path=MANIFEST,
            expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
            candidate_loader=loader,
        )

    loader.assert_called_once()
    assert model.calls == 0


def test_ready_direct_forward_recomputes_exact_uncalibrated_probabilities() -> None:
    model = _FakeModel()
    prediction, latency_ms = predict_primary_camera_window(
        _prepared(),
        _runtime(model),
        validation_scope="synthetic_camera_contract",
    )
    assert prediction["schema_version"] == CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION
    assert prediction["window_status"] == "ready"
    assert prediction["model_invocation_skipped"] is False
    assert prediction["probability_calibrated"] is False
    assert prediction["four_class"]["class_order"] == ["direct", "pacing", "lapping", "random"]
    expected = hierarchical_four_class_probabilities(
        torch.tensor([[math.log(4.0)]], dtype=torch.float32),
        torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
    )[0].tolist()
    assert prediction["four_class"]["probabilities"] == pytest.approx(expected)
    assert prediction["binary"]["probabilities"] == pytest.approx([0.2, 0.8])
    assert prediction["binary"]["predicted_label"] == "wandering_like"
    assert prediction["four_class"]["predicted_label"] == "pacing"
    assert model.calls == 1
    assert latency_ms >= 0.0


def test_binary_probability_tie_uses_frozen_greater_than_or_equal_rule() -> None:
    model = _FakeModel(
        {
            "binary_logit": torch.tensor([[0.0]], dtype=torch.float32),
            "subtype_logits": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
        }
    )
    prediction, _ = predict_primary_camera_window(
        _prepared(),
        _runtime(model),
        validation_scope="synthetic_camera_contract",
    )
    assert prediction["binary"]["probabilities"] == pytest.approx([0.5, 0.5])
    assert prediction["binary"]["predicted_label"] == "wandering_like"
    assert model.calls == 1


def test_primary_source_never_calls_wp_specific_runtime_predict_logits() -> None:
    source = (
        ROOT
        / "src/elderly_monitoring/modules/mental_health/wandering/camera_primary_inference.py"
    ).read_text(encoding="utf-8")
    assert "runtime.predict_logits(" not in source
    assert "fixed_cohort_wp_prefix_padding" not in source


def test_fake_forward_failure_is_a_per_window_inference_error() -> None:
    model = _FakeModel(failure=True)
    prediction, latency_ms = predict_primary_camera_window(
        _prepared(),
        _runtime(model),
        validation_scope="synthetic_camera_contract",
    )
    assert prediction["window_status"] == "inference_error"
    assert prediction["reason_codes"] == ["trusted_model_forward_failed"]
    assert prediction["binary"] is prediction["subtype"] is prediction["four_class"] is None
    assert prediction["model_invocation_skipped"] is False
    assert model.calls == 1
    assert latency_ms >= 0.0


@pytest.mark.parametrize(
    "output",
    (
        {"binary_logit": torch.zeros((1, 2)), "subtype_logits": torch.zeros((1, 3))},
        {"binary_logit": torch.tensor([[float("nan")]]), "subtype_logits": torch.zeros((1, 3))},
        {"binary_logit": torch.zeros((1, 1)), "subtype_logits": torch.tensor([[0.0, float("inf"), 0.0]])},
    ),
)
def test_bad_shape_nan_and_inf_are_per_window_inference_errors(
    output: dict[str, torch.Tensor],
) -> None:
    prediction, latency_ms = predict_primary_camera_window(
        _prepared(),
        _runtime(_FakeModel(output)),
        validation_scope="synthetic_camera_contract",
    )
    assert prediction["window_status"] == "inference_error"
    assert prediction["reason_codes"] == ["trusted_model_output_invalid"]
    assert prediction["binary"] is prediction["subtype"] is prediction["four_class"] is None
    assert prediction["model_invocation_skipped"] is False
    assert latency_ms >= 0.0


def test_unavailable_window_has_full_scope_null_probabilities_and_zero_forward() -> None:
    unavailable = _prepared()
    unavailable.update(
        window_status="unavailable",
        reason_codes=["insufficient_motion"],
        model_features=None,
        shape_normalized_points=None,
        point_mask=None,
    )
    model = _FakeModel()
    prediction, latency_ms = predict_primary_camera_window(
        unavailable,
        _runtime(model),
        validation_scope="synthetic_camera_contract",
    )
    assert prediction["window_status"] == "unavailable"
    assert prediction["parent_tracklet_id"] == "tracklet-0001"
    assert prediction["window_start_sec"] == 0.0
    assert prediction["window_end_sec"] == 40.0
    assert prediction["binary"] is prediction["subtype"] is prediction["four_class"] is None
    assert prediction["model_invocation_skipped"] is True
    assert model.calls == 0
    assert latency_ms is None


@pytest.mark.parametrize("window_status", ("ready", "unavailable"))
@pytest.mark.parametrize(
    ("validation_scope", "evidence_scope"),
    (
        ("authorized_camera_engineering_smoke", "synthetic_contract_only"),
        ("authorized_camera_labeled_evaluation", "synthetic_contract_only"),
        ("other_validation_scope", "synthetic_contract_only"),
        ("synthetic_camera_contract", "authorized_development_smoke"),
        ("synthetic_camera_contract", "authorized_labeled_development_evaluated"),
        ("synthetic_camera_contract", "test_fixture_only"),
        ("synthetic_camera_contract", "other_evidence_scope"),
    ),
)
def test_public_predictor_rejects_every_non_synthetic_scope_before_forward(
    window_status: str,
    validation_scope: str,
    evidence_scope: str,
) -> None:
    prepared = _prepared()
    if window_status == "unavailable":
        prepared.update(
            window_status="unavailable",
            reason_codes=["insufficient_motion"],
            model_features=None,
            shape_normalized_points=None,
            point_mask=None,
        )
    model = _FakeModel()

    with pytest.raises(PrimaryCameraInferenceError, match="synthetic"):
        predict_primary_camera_window(
            prepared,
            _runtime(model),
            validation_scope=validation_scope,
            evidence_scope=evidence_scope,
        )

    assert model.calls == 0


def test_provenance_aware_forward_is_private_and_not_exported() -> None:
    name = "_predict_primary_camera_window_with_provenance"
    assert hasattr(primary_inference_module, name)
    assert name not in primary_inference_module.__all__


def test_fresh_actual_candidate_bundle_is_atomic_and_descriptor_complete(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    tracking, sidecar = _write_pair(
        tmp_path / "input", [_tracking_row(index) for index in range(80)]
    )
    output = tmp_path / "output"
    result = build_primary_camera_inference_bundle(
        camera_config_path=CONFIG,
        project_root=ROOT,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        manifest_path=MANIFEST,
        expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
        episode_merge_gap_seconds=0.0,
        output_dir=output,
    )
    expected_names = {
        "media_sidecar.json",
        "tracking_input.jsonl",
        "bbox_tracklets.jsonl",
        "window_records.jsonl",
        "predictions.jsonl",
        "episode_candidates.jsonl",
        "qc_summary.json",
        "model_bindings.json",
        "execution.json",
        "manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_names
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CAMERA_PRIMARY_RUN_MANIFEST_SCHEMA_VERSION
    assert set(manifest["artifacts"]) == expected_names - {"manifest.json"}
    for name, descriptor in manifest["artifacts"].items():
        payload = (output / name).read_bytes()
        assert descriptor == {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert manifest["candidate_manifest_sha256"] == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert manifest["evidence_scope"] == "synthetic_contract_only"
    assert manifest["status"] == "wandering_m0cam_primary_camera_engineering_ready"
    assert result.ready_window_count == 1
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    assert execution["runtime"]["cpu_threads"] == {"intra_op": 8, "inter_op": 1}
    assert execution["latency_ms"]["sample_count"] == 1
    assert execution["latency_ms"]["p50"] >= 0.0
    assert execution["latency_ms"]["p95"] >= 0.0
    source_identity = execution["active_source_identity_preflight"]
    assert source_identity["verified_before_candidate_loader"] is True
    assert source_identity["passed"] is True
    assert source_identity["sources"]["model"]["expected"] == source_identity["sources"]["model"]["observed"]
    assert source_identity["sources"]["release"]["expected"] == source_identity["sources"]["release"]["observed"]
    bindings = json.loads((output / "model_bindings.json").read_text(encoding="utf-8"))
    assert bindings["active_source_identity_preflight"] == source_identity
    assert torch.get_num_threads() == 1
    with pytest.raises(FileExistsError):
        build_primary_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            manifest_path=MANIFEST,
            expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
            episode_merge_gap_seconds=0.0,
            output_dir=output,
        )
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_input_failure_and_manifest_drift_leave_no_output_and_no_forward(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(
        tmp_path / "input", [_tracking_row(index) for index in range(80)]
    )
    output = tmp_path / "output"
    loader = mock.Mock()
    with pytest.raises(PrimaryCameraInferenceError):
        build_primary_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            manifest_path=MANIFEST,
            expected_manifest_sha256="0" * 64,
            episode_merge_gap_seconds=0.0,
            output_dir=output,
            candidate_loader=loader,
        )
    loader.assert_not_called()
    assert not output.exists()

    bad_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    bad_sidecar["tracking_jsonl_sha256"] = "0" * 64
    sidecar.write_bytes(_canonical_json(bad_sidecar))
    fake_model = _FakeModel()
    with mock.patch(
        "elderly_monitoring.modules.mental_health.wandering.camera_primary_inference.load_primary_camera_runtime",
        return_value=_runtime(fake_model),
    ):
        with pytest.raises(PrimaryCameraInferenceError):
            build_primary_camera_inference_bundle(
                camera_config_path=CONFIG,
                project_root=ROOT,
                tracking_jsonl_path=tracking,
                media_sidecar_path=sidecar,
                manifest_path=MANIFEST,
                expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
                episode_merge_gap_seconds=0.0,
                output_dir=output,
            )
    assert fake_model.calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "authorization_status",
    ("authorized_camera_engineering_smoke", "authorized_camera_labeled_evaluation"),
)
def test_primary_entry_rejects_non_synthetic_authorization_before_loader_tracking_or_forward(
    tmp_path: Path,
    authorization_status: str,
) -> None:
    tracking, sidecar = _write_pair(
        tmp_path / "input", [_tracking_row(index) for index in range(80)]
    )
    sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_value["authorization_status"] = authorization_status
    sidecar.write_bytes(_canonical_json(sidecar_value))
    tracking.unlink()
    output = tmp_path / "output"
    model = _FakeModel()
    loader = mock.Mock(return_value=_runtime(model))

    with pytest.raises(PrimaryCameraInferenceError, match="synthetic"):
        build_primary_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            manifest_path=MANIFEST,
            expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
            episode_merge_gap_seconds=0.0,
            output_dir=output,
            candidate_loader=loader,
        )

    loader.assert_not_called()
    assert model.calls == 0
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


@pytest.mark.parametrize(
    "module_attribute",
    ("_MODEL_SOURCE_MODULE", "_RELEASE_SOURCE_MODULE"),
)
def test_active_source_identity_drift_fails_before_loader_tracking_or_forward(
    tmp_path: Path,
    module_attribute: str,
) -> None:
    tracking, sidecar = _write_pair(
        tmp_path / "input", [_tracking_row(index) for index in range(80)]
    )
    tracking.unlink()
    drift_source = tmp_path / f"drift-{module_attribute}.py"
    drift_source.write_text("# deliberate source identity drift\n", encoding="utf-8")
    output = tmp_path / "output"
    model = _FakeModel()
    loader = mock.Mock(return_value=_runtime(model))

    with mock.patch.object(
        primary_inference_module,
        module_attribute,
        SimpleNamespace(__file__=str(drift_source)),
        create=True,
    ):
        with pytest.raises(PrimaryCameraInferenceError, match="source identity"):
            build_primary_camera_inference_bundle(
                camera_config_path=CONFIG,
                project_root=ROOT,
                tracking_jsonl_path=tracking,
                media_sidecar_path=sidecar,
                manifest_path=MANIFEST,
                expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
                episode_merge_gap_seconds=0.0,
                output_dir=output,
                candidate_loader=loader,
            )

    loader.assert_not_called()
    assert model.calls == 0
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_cli_help_and_fresh_tracking_sidecar_end_to_end(tmp_path: Path) -> None:
    cli = ROOT / "scripts/wandering/run_topowander_camera_inference.py"
    help_result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for required in (
        "--tracking-jsonl",
        "--media-sidecar",
        "--manifest",
        "--expected-manifest-sha256",
        "--episode-merge-gap-seconds",
        "--output",
    ):
        assert required in help_result.stdout
    for forbidden in ("wp-public", "official", "calibrate", "risk", "algorithm-event"):
        assert forbidden not in help_result.stdout.lower()

    tracking, sidecar = _write_pair(
        tmp_path / "input", [_tracking_row(index) for index in range(80)]
    )
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--config",
            str(CONFIG),
            "--project-root",
            str(ROOT),
            "--tracking-jsonl",
            str(tracking),
            "--media-sidecar",
            str(sidecar),
            "--manifest",
            str(MANIFEST),
            "--expected-manifest-sha256",
            EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "--episode-merge-gap-seconds",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ready=1" in completed.stdout
    assert "inference_error=0" in completed.stdout
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    assert execution["runtime"]["observed_cpu_threads"] == {"intra_op": 8, "inter_op": 1}
    assert execution["candidate_runtime_predict_logits_used"] is False
