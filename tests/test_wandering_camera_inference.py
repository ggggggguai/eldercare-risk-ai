from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CAMERA_PREDICTION_SCHEMA_VERSION,
    CAMERA_RUN_MANIFEST_SCHEMA_VERSION,
    CameraInferenceError,
    build_camera_inference_bundle,
    load_camera_config,
    load_trusted_camera_models,
    prepare_camera_window,
    predict_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.handcrafted_features import (
    extract_features_from_points,
    extract_handcrafted_features,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import load_preprocessing_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_v1.yaml"
RF_DEVELOPMENT = ROOT / "reports/mental_health/wandering_step5/development/v1"
TCN_DEVELOPMENT = ROOT / "reports/mental_health/wandering_step6/development/v1"
RF_MANIFEST_SHA256 = "fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2"
TCN_MANIFEST_SHA256 = "0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _tracking_row(bucket: int, *, track_id: int = 1) -> dict[str, object]:
    phase = bucket / 79.0
    center_x = 220.0 + 140.0 * np.sin(phase * 2.0 * np.pi)
    center_y = 300.0 + 35.0 * np.sin(phase * 4.0 * np.pi)
    return {
        "frame_id": bucket * 10 + track_id,
        "track_id": track_id,
        "bbox": [center_x - 20.0, center_y - 100.0, center_x + 20.0, center_y],
        "track_confidence": 0.9,
        "timestamp_sec": bucket * 0.5 + 0.1,
    }


def _write_pair(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    authorization_status: str = "synthetic_fixture",
) -> tuple[Path, Path]:
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
        "detector": {"backend": "ultralytics_yolo", "model": "yolov8n.pt", "version": "synthetic"},
        "tracker": {"backend": "bytetrack", "config": "bytetrack.yaml", "version": "synthetic"},
        "fixed_camera_assumed": True,
        "camera_motion_state": "not_checked",
        "authorization_status": authorization_status,
        "deidentification_status": "synthetic",
    }
    sidecar_path = directory / "sidecar.json"
    sidecar_path.write_bytes(_canonical_json(sidecar))
    return tracking, sidecar_path


def test_config_freezes_exact_fields_values_trust_roots_and_has_no_future_inputs() -> None:
    config = load_camera_config(CONFIG)
    assert config["schema_version"] == "wandering-camera-config-v1"
    assert config["purpose"] == "comparison_only_offline_camera_chain"
    assert config["sampling"] == {
        "minimum_video_fps": 15.0,
        "bucket_seconds": 0.5,
        "trajectory_hz": 2.0,
        "window_seconds": 40.0,
        "window_buckets": 80,
        "stride_seconds": 20.0,
        "stride_buckets": 40,
        "minimum_track_confidence": 0.25,
    }
    assert config["camera_qc"]["maximum_internal_gap_seconds"] == 1.5
    assert config["camera_qc"]["maximum_internal_gap_buckets"] == 3
    assert config["model_input"]["temporal_features_enabled"] is False
    assert config["models"]["primary_seed"] == 20260731
    assert config["models"]["fusion_performed"] is False
    text = CONFIG.read_text(encoding="utf-8")
    for forbidden in ("official_validation", "public_shape_benchmark", "calibration", "episode", "risk"):
        assert forbidden not in text


def test_config_rejects_extensions_path_escape_hash_drift_public_test_and_existing_output(tmp_path: Path) -> None:
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutations = []
    extra = copy.deepcopy(base)
    extra["unexpected"] = True
    mutations.append(extra)
    escape = copy.deepcopy(base)
    escape["trust_roots"]["rf_config"]["path"] = "../wandering_rf_v1.yaml"
    mutations.append(escape)
    hash_drift = copy.deepcopy(base)
    hash_drift["trust_roots"]["preprocessing_feature_stats"]["sha256"] = "0" * 64
    mutations.append(hash_drift)
    public_test = copy.deepcopy(base)
    public_test["trust_roots"]["rf_development_manifest"]["path"] = "reports/mental_health/wandering_step5/public_shape_benchmark/v1/manifest.json"
    mutations.append(public_test)
    official = copy.deepcopy(base)
    official["trust_roots"]["official_validation"] = {"path": "official_validation.jsonl", "sha256": "0" * 64}
    mutations.append(official)
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"bad_{index}.yaml"
        path.write_text(yaml.safe_dump(mutation, sort_keys=False), encoding="utf-8")
        with pytest.raises(CameraInferenceError):
            load_camera_config(path)

    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=output,
        )


def test_shared_preprocessing_functions_apply_frozen_stats_and_preserve_channel_semantics(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path, [_tracking_row(i) for i in range(80)])
    config = load_camera_config(CONFIG)
    from elderly_monitoring.modules.mental_health.wandering.camera_adapter import load_camera_inputs
    from elderly_monitoring.modules.mental_health.wandering.camera_qc import run_camera_qc

    qc = run_camera_qc(load_camera_inputs(tracking, sidecar, config), config)
    assert len(qc.ready_inputs) == 1
    stats = json.loads((ROOT / config["trust_roots"]["preprocessing_feature_stats"]["path"]).read_text(encoding="utf-8"))
    preprocessing_config = load_preprocessing_config(ROOT / config["trust_roots"]["preprocessing_config"]["path"])
    prepared = prepare_camera_window(qc.ready_inputs[0], config, preprocessing_config, stats)
    assert np.asarray(prepared["shape_normalized_points"]).shape == (80, 2)
    assert np.asarray(prepared["raw_features"]).shape == (80, 14)
    assert np.asarray(prepared["model_features"]).shape == (80, 14)
    raw = np.asarray(prepared["raw_features"])
    model = np.asarray(prepared["model_features"])
    assert np.array_equal(raw[:, 10:12], np.zeros((80, 2)))
    assert np.array_equal(model[:, 10:12], np.zeros((80, 2)))
    assert np.array_equal(model[:, 12], np.ones(80))
    assert np.all((model[:, 13] >= 0.0) & (model[:, 13] <= 1.0))
    assert not np.array_equal(raw[:, :10], model[:, :10])


def test_step4_ready_fixture_has_identical_26_features_through_points_helper() -> None:
    samples = ROOT / "data/processed/wandering/preprocessing/v1/samples.jsonl"
    record = None
    with samples.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["preprocess_status"] == "ready":
                record = candidate
                break
    assert record is not None
    direct = extract_handcrafted_features(record)
    through_camera_formula = extract_features_from_points(
        record["shape_normalized_points"],
        point_mask=record["point_mask"],
        quality=np.asarray(record["raw_features"])[:, 13],
    )
    np.testing.assert_allclose(direct, through_camera_formula, rtol=0.0, atol=0.0)


def test_model_loading_uses_only_safe_loaders_primary_seed_and_external_manifest_roots() -> None:
    config = load_camera_config(CONFIG)
    rf_loader = mock.Mock(side_effect=["rf-four", "rf-binary"])
    tcn_loader = mock.Mock(side_effect=["tcn-four", "tcn-binary"])
    models = load_trusted_camera_models(
        config=config,
        project_root=ROOT,
        rf_development_dir=RF_DEVELOPMENT,
        expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
        tcn_development_dir=TCN_DEVELOPMENT,
        expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
        rf_loader=rf_loader,
        tcn_loader=tcn_loader,
    )
    assert models.rf_four_class == "rf-four"
    assert models.tcn_binary == "tcn-binary"
    assert [call.kwargs["seed"] for call in rf_loader.call_args_list] == [20260731, 20260731]
    assert [call.kwargs["seed"] for call in tcn_loader.call_args_list] == [20260731, 20260731]
    assert all(call.kwargs["expected_manifest_sha256"] == RF_MANIFEST_SHA256 for call in rf_loader.call_args_list)
    assert all(call.kwargs["expected_manifest_sha256"] == TCN_MANIFEST_SHA256 for call in tcn_loader.call_args_list)

    with pytest.raises(CameraInferenceError):
        load_trusted_camera_models(
            config=config,
            project_root=ROOT,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256="0" * 64,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            rf_loader=rf_loader,
            tcn_loader=tcn_loader,
        )


class _FakeRF:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray([probabilities], dtype=np.float64)
        self.calls = 0

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        assert values.shape == (1, 26)
        self.calls += 1
        return self.probabilities.copy()


class _FakeTCN(torch.nn.Module):
    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task
        self.calls = 0

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        assert features.shape == (1, 80, 14)
        assert mask.shape == (1, 80)
        self.calls += 1
        if self.task == "four_class":
            return {
                "gate_logit": torch.tensor([[np.log(4.0)]], dtype=torch.float32),
                "subtype_logits": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
            }
        return {"binary_logit": torch.tensor([[0.0]], dtype=torch.float32)}


def test_ready_prediction_has_exact_four_independent_uncalibrated_unfused_outputs(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path, [_tracking_row(i) for i in range(80)])
    config = load_camera_config(CONFIG)
    from elderly_monitoring.modules.mental_health.wandering.camera_adapter import load_camera_inputs
    from elderly_monitoring.modules.mental_health.wandering.camera_qc import run_camera_qc
    from elderly_monitoring.modules.mental_health.wandering.camera_inference import TrustedCameraModels

    qc = run_camera_qc(load_camera_inputs(tracking, sidecar, config), config)
    stats = json.loads((ROOT / config["trust_roots"]["preprocessing_feature_stats"]["path"]).read_text(encoding="utf-8"))
    preprocessing_config = load_preprocessing_config(ROOT / config["trust_roots"]["preprocessing_config"]["path"])
    prepared = prepare_camera_window(qc.ready_inputs[0], config, preprocessing_config, stats)
    models = TrustedCameraModels(
        rf_four_class=_FakeRF([0.1, 0.6, 0.2, 0.1]),
        rf_binary=_FakeRF([0.4, 0.6]),
        tcn_four_class=_FakeTCN("four_class"),
        tcn_binary=_FakeTCN("binary"),
    )
    prediction = predict_camera_window(prepared, models, validation_scope="synthetic_camera_contract")
    assert set(prediction) == {
        "schema_version", "window_id", "window_status", "reason_codes", "quality_flags",
        "model_purpose", "validation_scope", "probability_calibrated", "fusion_performed",
        "primary_seed", "predictions", "model_invocation_skipped",
    }
    assert prediction["schema_version"] == CAMERA_PREDICTION_SCHEMA_VERSION
    assert prediction["probability_calibrated"] is False
    assert prediction["fusion_performed"] is False
    assert prediction["model_invocation_skipped"] is False
    assert set(prediction["predictions"]) == {"rf_four_class", "rf_binary", "tcn_four_class", "tcn_binary"}
    assert prediction["predictions"]["rf_four_class"]["predicted_label"] == "pacing"
    assert prediction["predictions"]["rf_binary"]["predicted_label"] == "wandering_like"
    for result in prediction["predictions"].values():
        assert all(np.isfinite(result["probabilities"]))
        assert sum(result["probabilities"]) == pytest.approx(1.0)


def test_actual_end_to_end_bundle_is_atomic_non_overwriting_and_byte_deterministic(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    outputs = [tmp_path / "first", tmp_path / "second"]
    results = []
    for output in outputs:
        results.append(
            build_camera_inference_bundle(
                camera_config_path=CONFIG,
                project_root=ROOT,
                tracking_jsonl_path=tracking,
                media_sidecar_path=sidecar,
                rf_development_dir=RF_DEVELOPMENT,
                expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
                tcn_development_dir=TCN_DEVELOPMENT,
                expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
                output_dir=output,
            )
        )
    expected_names = {
        "media_sidecar.json", "tracking_input.jsonl", "bbox_tracklets.jsonl", "window_records.jsonl",
        "predictions.jsonl", "qc_summary.json", "model_bindings.json", "manifest.json",
    }
    assert {path.name for path in outputs[0].iterdir()} == expected_names
    assert {path.name for path in outputs[1].iterdir()} == expected_names
    for name in expected_names:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    assert results[0].manifest_sha256 == results[1].manifest_sha256
    manifest = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CAMERA_RUN_MANIFEST_SCHEMA_VERSION
    assert set(manifest) == {
        "schema_version", "artifacts", "camera_config_sha256", "trust_roots",
        "expected_rf_development_manifest_sha256", "expected_tcn_development_manifest_sha256",
        "source_tracking_sha256", "normalized_tracking_sha256", "source_media_sha256",
        "code_hashes", "environment", "validation_scope", "models_retrained",
        "official_source_opened", "fusion_performed",
    }
    assert set(manifest["artifacts"]) == expected_names - {"manifest.json"}
    assert all(set(descriptor) == {"byte_count", "sha256"} for descriptor in manifest["artifacts"].values())
    assert manifest["models_retrained"] is False
    assert manifest["official_source_opened"] is False
    assert manifest["fusion_performed"] is False
    assert manifest["validation_scope"] == "synthetic_camera_contract"
    assert results[0].ready_window_count == 1
    assert results[0].unavailable_window_count == 0
    prediction = json.loads((outputs[0] / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["window_status"] == "ready"
    assert len(prediction["predictions"]) == 4
    with pytest.raises(FileExistsError):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=outputs[0],
        )


def test_unavailable_windows_skip_every_model_and_no_future_outputs_exist(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path, [
        {
            "frame_id": i * 10 + 1,
            "track_id": 1,
            "bbox": [100.0, 200.0, 140.0, 300.0],
            "track_confidence": 0.9,
            "timestamp_sec": i * 0.5 + 0.1,
        }
        for i in range(80)
    ])
    rf_loader = mock.Mock()
    tcn_loader = mock.Mock()
    from elderly_monitoring.modules.mental_health.wandering.camera_inference import TrustedCameraModels

    rf_four = _FakeRF([0.25] * 4)
    rf_binary = _FakeRF([0.5, 0.5])
    tcn_four = _FakeTCN("four_class")
    tcn_binary = _FakeTCN("binary")
    models = TrustedCameraModels(rf_four, rf_binary, tcn_four, tcn_binary)
    output = tmp_path / "output"
    with mock.patch(
        "elderly_monitoring.modules.mental_health.wandering.camera_inference.load_trusted_camera_models",
        return_value=models,
    ):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=output,
        )
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["window_status"] == "unavailable"
    assert prediction["predictions"] is None
    assert prediction["model_invocation_skipped"] is True
    assert rf_four.calls == rf_binary.calls == tcn_four.calls == tcn_binary.calls == 0
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    for forbidden in ("official_validation", "calibration", "episode", "algorithm_event", "risk_level"):
        assert forbidden not in all_text.lower()


def test_trust_failure_happens_before_any_forward_and_leaves_no_output(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    output = tmp_path / "output"
    with pytest.raises(CameraInferenceError):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256="0" * 64,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=output,
        )
    assert not output.exists()


def test_single_forward_failure_keeps_no_partial_probabilities(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    from elderly_monitoring.modules.mental_health.wandering.camera_inference import TrustedCameraModels

    class FailingRF(_FakeRF):
        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            self.calls += 1
            raise RuntimeError("synthetic forward failure")

    models = TrustedCameraModels(
        _FakeRF([0.1, 0.6, 0.2, 0.1]),
        FailingRF([0.4, 0.6]),
        _FakeTCN("four_class"),
        _FakeTCN("binary"),
    )
    output = tmp_path / "output"
    with mock.patch(
        "elderly_monitoring.modules.mental_health.wandering.camera_inference.load_trusted_camera_models",
        return_value=models,
    ):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=output,
        )
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["window_status"] == "inference_error"
    assert prediction["reason_codes"] == ["trusted_model_forward_failed"]
    assert prediction["predictions"] is None
    assert prediction["model_invocation_skipped"] is False
    assert models.rf_four_class.calls == 1
    assert models.rf_binary.calls == 1
    assert models.tcn_four_class.calls == 0
    assert models.tcn_binary.calls == 0


def test_model_output_failure_is_a_per_window_inference_error(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    from elderly_monitoring.modules.mental_health.wandering.camera_inference import TrustedCameraModels

    models = TrustedCameraModels(
        _FakeRF([0.1, 0.6, 0.2]),
        _FakeRF([0.4, 0.6]),
        _FakeTCN("four_class"),
        _FakeTCN("binary"),
    )
    output = tmp_path / "output"
    with mock.patch(
        "elderly_monitoring.modules.mental_health.wandering.camera_inference.load_trusted_camera_models",
        return_value=models,
    ):
        build_camera_inference_bundle(
            camera_config_path=CONFIG,
            project_root=ROOT,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=RF_DEVELOPMENT,
            expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
            tcn_development_dir=TCN_DEVELOPMENT,
            expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
            output_dir=output,
        )
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["window_status"] == "inference_error"
    assert prediction["reason_codes"] == ["trusted_model_output_invalid"]
    assert prediction["predictions"] is None


def test_non_model_contract_error_aborts_bundle_without_output(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    output = tmp_path / "output"
    with mock.patch(
        "elderly_monitoring.modules.mental_health.wandering.camera_inference.extract_features_from_points",
        side_effect=KeyError("prepared-window-contract-drift"),
    ):
        with pytest.raises(KeyError, match="prepared-window-contract-drift"):
            build_camera_inference_bundle(
                camera_config_path=CONFIG,
                project_root=ROOT,
                tracking_jsonl_path=tracking,
                media_sidecar_path=sidecar,
                rf_development_dir=RF_DEVELOPMENT,
                expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
                tcn_development_dir=TCN_DEVELOPMENT,
                expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
                output_dir=output,
            )
    assert not output.exists()


def test_authorized_labeled_input_keeps_its_exact_validation_scope(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(
        tmp_path / "input",
        [_tracking_row(i) for i in range(80)],
        authorization_status="authorized_camera_labeled_evaluation",
    )
    output = tmp_path / "output"
    build_camera_inference_bundle(
        camera_config_path=CONFIG,
        project_root=ROOT,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        rf_development_dir=RF_DEVELOPMENT,
        expected_rf_development_manifest_sha256=RF_MANIFEST_SHA256,
        tcn_development_dir=TCN_DEVELOPMENT,
        expected_tcn_development_manifest_sha256=TCN_MANIFEST_SHA256,
        output_dir=output,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert manifest["validation_scope"] == "authorized_camera_labeled_evaluation"
    assert prediction["validation_scope"] == "authorized_camera_labeled_evaluation"


def test_cli_exposes_only_the_fixed_tracking_or_mp4_camera_entry() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/wandering/run_camera_inference.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    for required in (
        "--tracking-jsonl", "--media-sidecar", "--input-video", "--rf-development-dir",
        "--expected-rf-development-manifest-sha256", "--tcn-development-dir",
        "--expected-tcn-development-manifest-sha256", "--output",
    ):
        assert required in output
    for forbidden in ("official", "calibrate", "episode", "risk"):
        assert forbidden not in output.lower()


def test_tracking_sidecar_cli_builds_the_fixed_prediction_bundle(tmp_path: Path) -> None:
    tracking, sidecar = _write_pair(tmp_path / "input", [_tracking_row(i) for i in range(80)])
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/run_camera_inference.py"),
            "--config", str(CONFIG),
            "--project-root", str(ROOT),
            "--tracking-jsonl", str(tracking),
            "--media-sidecar", str(sidecar),
            "--rf-development-dir", str(RF_DEVELOPMENT),
            "--expected-rf-development-manifest-sha256", RF_MANIFEST_SHA256,
            "--tcn-development-dir", str(TCN_DEVELOPMENT),
            "--expected-tcn-development-manifest-sha256", TCN_MANIFEST_SHA256,
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ready=1" in completed.stdout
    assert "unavailable=0" in completed.stdout
    assert (output / "manifest.json").is_file()
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["window_status"] == "ready"
    assert set(prediction["predictions"]) == {
        "rf_four_class", "rf_binary", "tcn_four_class", "tcn_binary"
    }
