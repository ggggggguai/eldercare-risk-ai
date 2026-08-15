from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.wandering import camera_product as product_module
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_product import (
    PRODUCT_MANIFEST_SCHEMA_VERSION,
    SESSION_EVIDENCE_SCHEMA_VERSION,
    WanderingCameraProductError,
    build_wandering_camera_product,
    load_validated_wandering_camera_product,
    summarize_primary_camera_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    aggregate_episode_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/wandering/run_wandering_camera_product.py"
PRIMARY_MANIFEST = (
    ROOT
    / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1"
    / "artifacts/topowander_m0r_candidate_v3/candidate_manifest.json"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"
)
SOURCE_SCOPE = {
    "source_group_id": "session-0001",
    "source_video_id": "video-0001",
    "device_id": "camera-0001",
    "setup_id": "fixed-setup-0001",
    "stream_epoch": "epoch-0001",
}
PRIMARY_ARTIFACT_NAMES = {
    "media_sidecar.json",
    "tracking_input.jsonl",
    "bbox_tracklets.jsonl",
    "window_records.jsonl",
    "predictions.jsonl",
    "episode_candidates.jsonl",
    "qc_summary.json",
    "model_bindings.json",
    "execution.json",
}
PRIMARY_SOURCE_HASH_NAMES = {
    "scripts/wandering/run_topowander_camera_inference.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_adapter.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_episode.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_inference.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_primary_inference.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_qc.py",
}
PRIMARY_DATA_ACCESS = {
    "synthetic_fixture": True,
    "real_human_media": False,
    "wp_raw": False,
    "wp_public_holdout": False,
    "smartcare_official_or_raw": False,
    "sealed_camera": False,
}


def _source_descriptor(fill: str) -> dict[str, object]:
    return {"sha256": fill * 64, "size_bytes": 123}


def _source_preflight() -> dict[str, object]:
    manifest_descriptor = _source_descriptor("5")
    model_descriptor = _source_descriptor("6")
    release_descriptor = _source_descriptor("7")
    return {
        "passed": True,
        "verified_before_candidate_loader": True,
        "candidate_manifest": {
            "path": "reports/synthetic/candidate_manifest.json",
            "expected": copy.deepcopy(manifest_descriptor),
            "observed": copy.deepcopy(manifest_descriptor),
        },
        "training_release_model_descriptor_equal": True,
        "sources": {
            "model": {
                "path": "src/elderly_monitoring/modules/mental_health/wandering/model.py",
                "expected": copy.deepcopy(model_descriptor),
                "observed": copy.deepcopy(model_descriptor),
                "training_expected": copy.deepcopy(model_descriptor),
                "release_expected": copy.deepcopy(model_descriptor),
            },
            "release": {
                "path": "src/elderly_monitoring/modules/mental_health/wandering/release.py",
                "expected": copy.deepcopy(release_descriptor),
                "observed": copy.deepcopy(release_descriptor),
            },
        },
    }


def _prediction(index: int, status: str, *, track_id: int = 42) -> dict[str, object]:
    ready = status == "ready"
    reason_codes = [] if ready else [
        "insufficient_motion" if status == "unavailable" else "trusted_model_forward_failed"
    ]
    return {
        "schema_version": "wandering-camera-primary-prediction-v1",
        "window_id": f"window-{index:04d}",
        **SOURCE_SCOPE,
        "track_id": track_id,
        "parent_tracklet_id": f"tracklet-{track_id}",
        "window_start_sec": float(index * 40),
        "window_end_sec": float((index + 1) * 40),
        "quality_flags": ["camera_motion_not_verified"],
        "model_purpose": "primary_candidate_camera_engineering",
        "validation_scope": "synthetic_camera_contract",
        "evidence_scope": "synthetic_contract_only",
        "candidate_id": "topowander-m0s-seed20260731-epoch0005",
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "primary_seed": 20260731,
        "best_epoch": 5,
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "wp_prefix_padding_used": False,
        "window_status": status,
        "reason_codes": reason_codes,
        "binary": (
            {
                "class_order": ["direct_or_non_wandering", "wandering_like"],
                "probabilities": [0.2, 0.8],
                "predicted_label": "wandering_like",
            }
            if ready
            else None
        ),
        "subtype": (
            {
                "class_order": ["pacing", "lapping", "random"],
                "probabilities": [0.8, 0.1, 0.1],
                "predicted_label": "pacing",
            }
            if ready
            else None
        ),
        "four_class": (
            {
                "class_order": ["direct", "pacing", "lapping", "random"],
                "probabilities": [0.2, 0.64, 0.08, 0.08],
                "predicted_label": "pacing",
            }
            if ready
            else None
        ),
        "model_invocation_skipped": status == "unavailable",
    }


def _primary_files(statuses: list[str]) -> dict[str, bytes]:
    predictions = [_prediction(index, status) for index, status in enumerate(statuses)]
    windows = [
        {
            "schema_version": "wandering-camera-window-v1",
            "window_id": row["window_id"],
            **SOURCE_SCOPE,
            "track_id": row["track_id"],
            "parent_tracklet_id": row["parent_tracklet_id"],
            "window_start_sec": row["window_start_sec"],
            "window_end_sec": row["window_end_sec"],
            "window_status": row["window_status"] if row["window_status"] != "inference_error" else "ready",
            "reason_codes": row["reason_codes"] if row["window_status"] == "unavailable" else [],
            "quality_flags": row["quality_flags"],
            "segment_index": 0,
            "raw_detection_count": 80,
            "observed_bucket_count": 80,
            "interpolated_bucket_count": 0,
            "observed_ratio": 1.0,
            "longest_gap_seconds": 0.0,
            "mean_detection_confidence": 0.9,
            "bbox_bottom_points": None,
            "bbox_heights": None,
            "smoothed_bbox_heights": None,
            "corrected_points": None,
            "observed_mask": None,
            "interpolated_mask": None,
            "point_quality": None,
            "shape_normalized_points": None,
            "point_mask": None,
            "raw_features": None,
            "model_features": None,
            "topology": None,
            "preprocessing_config_sha256": "8" * 64,
            "feature_stats_sha256": "9" * 64,
            "model_training_split_sha256": "a" * 64,
        }
        for row in predictions
    ]
    episodes = aggregate_episode_candidates(predictions, merge_gap_seconds=0.0)
    observation_count = len(statuses)
    tracking = [
        {
            "bbox": [100.0, 100.0, 140.0, 220.0],
            "frame_id": index,
            "timestamp_sec": float(index),
            "track_confidence": 0.9,
            "track_id": 42,
        }
        for index in range(observation_count)
    ]
    tracklets = (
        [
            {
                "schema_version": "wandering-bbox-tracklet-v1",
                "tracklet_id": "tracklet-42",
                "segment_index": 0,
                **SOURCE_SCOPE,
                "track_id": 42,
                "frame_width": 640,
                "frame_height": 480,
                "source_fps": 25.0,
                "frame_indices": list(range(observation_count)),
                "point_times_sec": [float(index) for index in range(observation_count)],
                "bbox_xyxy_norm": [],
                "bbox_bottom_points": [],
                "bbox_heights": [],
                "detection_confidence": [],
                "observed_mask": [],
                "interpolated_mask": [],
                "quality_flags": ["camera_motion_not_verified"],
                "media_ref": "synthetic/video-0001.mp4",
                "source_sha256": "1" * 64,
                "tracking_jsonl_sha256": "2" * 64,
            }
        ]
        if statuses
        else []
    )
    sidecar = {
        "schema_version": "wandering-media-v1",
        **SOURCE_SCOPE,
        "media_ref": "synthetic/video-0001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": "2" * 64,
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 25.0,
        "duration_sec": 40.0,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {"backend": "synthetic", "model": "none", "version": "synthetic"},
        "tracker": {"backend": "synthetic", "config": "none", "version": "synthetic"},
        "fixed_camera_assumed": True,
        "camera_motion_state": "not_checked",
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic",
    }
    status_counts = {name: statuses.count(name) for name in ("ready", "unavailable", "inference_error")}
    reasons: dict[str, int] = {}
    for row in predictions:
        for reason in row["reason_codes"]:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    qc_summary = {
        "schema_version": "wandering-camera-primary-qc-summary-v1",
        "evidence_scope": "synthetic_contract_only",
        "validation_scope": "synthetic_camera_contract",
        "observation_count": observation_count,
        "normalized_tracking_row_count": observation_count,
        "input_track_scope_count": 1 if statuses else 0,
        "tracklet_count": len(tracklets),
        "window_count": len(statuses),
        "window_status_counts": status_counts,
        "reason_counts": reasons,
        "episode_candidate_count": len(episodes),
        "camera_motion_state": "not_checked",
    }
    source_preflight = _source_preflight()
    model_bindings = {
        "schema_version": "wandering-camera-primary-model-bindings-v1",
        "candidate_id": "topowander-m0s-seed20260731-epoch0005",
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "primary_seed": 20260731,
        "best_epoch": 5,
        "four_class_order": ["direct", "pacing", "lapping", "random"],
        "subtype_order": ["pacing", "lapping", "random"],
        "binary_class_order": ["direct_or_non_wandering", "wandering_like"],
        "binary_decision_threshold": 0.5,
        "active_source_identity_preflight": copy.deepcopy(source_preflight),
        "probability_calibrated": False,
        "direct_model_forward": True,
        "candidate_runtime_predict_logits_used": False,
        "wp_prefix_padding_used": False,
        "retrained": False,
        "ensemble": False,
    }
    execution = {
        "schema_version": "wandering-camera-primary-execution-v1",
        "status": "wandering_m0cam_primary_camera_engineering_ready",
        "evidence_scope": "synthetic_contract_only",
        "validation_scope": "synthetic_camera_contract",
        "active_source_identity_preflight": copy.deepcopy(source_preflight),
        "runtime": {
            "device": "cpu",
            "input_dtype": "float32",
            "cpu_threads": {"intra_op": 8, "inter_op": 1},
            "observed_cpu_threads": {"intra_op": 8, "inter_op": 1},
            "interop_set_in_process": True,
            "maximum_batch_size": 64,
            "observed_batch_sizes": [
                1 for status in statuses if status in {"ready", "inference_error"}
            ],
            "model_mode": "eval",
            "autograd_mode": "torch.inference_mode",
        },
        "latency_ms": {
            "scope": "engineering_cpu_model_forward_only",
            "sample_count": status_counts["ready"] + status_counts["inference_error"],
            "p50": 1.0 if status_counts["ready"] + status_counts["inference_error"] else None,
            "p95": 1.0 if status_counts["ready"] + status_counts["inference_error"] else None,
            "product_threshold": None,
        },
        "model_forward_invocation_count": status_counts["ready"] + status_counts["inference_error"],
        "wp_prefix_padding_used": False,
        "candidate_runtime_predict_logits_used": False,
        "episode_policy_status": "development_unfrozen",
        "episode_merge_gap_seconds": 0.0,
        "probability_calibrated": False,
        "models_retrained": False,
        "environment": {
            "python": "3.11.0",
            "numpy": "fixture",
            "torch": "fixture",
            "pyyaml": "fixture",
        },
    }
    files = {
        "media_sidecar.json": canonical_json_bytes(sidecar),
        "tracking_input.jsonl": canonical_jsonl_bytes(tracking),
        "bbox_tracklets.jsonl": canonical_jsonl_bytes(tracklets),
        "window_records.jsonl": canonical_jsonl_bytes(windows),
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "episode_candidates.jsonl": canonical_jsonl_bytes(episodes),
        "qc_summary.json": canonical_json_bytes(qc_summary),
        "model_bindings.json": canonical_json_bytes(model_bindings),
        "execution.json": canonical_json_bytes(execution),
    }
    manifest = {
        "schema_version": "wandering-camera-primary-run-manifest-v1",
        "status": "wandering_m0cam_primary_camera_engineering_ready",
        "evidence_scope": "synthetic_contract_only",
        "artifacts": {
            name: {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(files.items())
        },
        "primary_camera_config_sha256": "3" * 64,
        "camera_chain_config_sha256": "4" * 64,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "source_tracking_sha256": "2" * 64,
        "normalized_tracking_sha256": hashlib.sha256(
            files["tracking_input.jsonl"]
        ).hexdigest(),
        "source_media_sha256": "1" * 64,
        "validation_scope": "synthetic_camera_contract",
        "code_hashes": {name: "b" * 64 for name in sorted(PRIMARY_SOURCE_HASH_NAMES)},
        "data_access": copy.deepcopy(PRIMARY_DATA_ACCESS),
        "models_retrained": False,
        "wp_prefix_padding_used": False,
        "algorithm_event_emitted": False,
        "risk_or_alert_decision_emitted": False,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    return files


def _write_primary_bundle(directory: Path, statuses: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name, payload in _primary_files(statuses).items():
        (directory / name).write_bytes(payload)


def _refresh_primary_descriptor(directory: Path, name: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (directory / name).read_bytes()
    manifest["artifacts"][name] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _mutate_primary_artifact(
    directory: Path,
    name: str,
    mutation: object,
) -> None:
    path = directory / name
    if name.endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows, f"fixture artifact must contain a row: {name}"
        mutation(rows[0])  # type: ignore[operator]
        path.write_bytes(canonical_jsonl_bytes(rows))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)  # type: ignore[operator]
        path.write_bytes(canonical_json_bytes(value))
    if name != "manifest.json":
        _refresh_primary_descriptor(directory, name)


def _corrupting_primary_builder(
    statuses: list[str],
    calls: list[dict[str, object]],
    mutation: object,
):
    base = _fake_primary_builder(statuses, calls)

    def build(**kwargs: object) -> SimpleNamespace:
        result = base(**kwargs)
        mutation(Path(kwargs["output_dir"]))  # type: ignore[operator]
        return result

    return build


def _invoke_product(tmp_path: Path) -> object:
    return build_wandering_camera_product(
        project_root=ROOT,
        tracking_jsonl_path=tmp_path / "tracking.jsonl",
        media_sidecar_path=tmp_path / "sidecar.json",
        episode_merge_gap_seconds=0.0,
        output_dir=tmp_path / "product",
    )


def _assert_rejected_without_product_residue(tmp_path: Path, match: str) -> None:
    with pytest.raises(WanderingCameraProductError, match=match):
        _invoke_product(tmp_path)
    assert not (tmp_path / "product").exists()
    assert not list(tmp_path.glob(".product.tmp-*"))


def _fake_primary_builder(statuses: list[str], calls: list[dict[str, object]]):
    def build(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        output = Path(kwargs["output_dir"])
        _write_primary_bundle(output, statuses)
        counts = {name: statuses.count(name) for name in ("ready", "unavailable", "inference_error")}
        return SimpleNamespace(
            output_dir=output,
            manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
            observation_count=len(statuses),
            tracklet_count=1 if statuses else 0,
            window_count=len(statuses),
            ready_window_count=counts["ready"],
            unavailable_window_count=counts["unavailable"],
            inference_error_count=counts["inference_error"],
            episode_candidate_count=counts["ready"],
        )

    return build


def test_missing_prediction_model_purpose_with_refreshed_descriptor_is_rejected(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, ["ready"])
    _mutate_primary_artifact(
        primary,
        "predictions.jsonl",
        lambda row: row.pop("model_purpose"),
    )

    with pytest.raises(WanderingCameraProductError, match="prediction.*field set"):
        summarize_primary_camera_bundle(primary)


def test_nonempty_prediction_risk_and_algorithm_event_with_refreshed_descriptor_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def inject(primary: Path) -> None:
        def mutation(row: dict[str, object]) -> None:
            row["risk_level"] = "high"
            row["algorithm_event"] = {"module": "mental_health"}

        _mutate_primary_artifact(primary, "predictions.jsonl", mutation)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _corrupting_primary_builder(["ready"], calls, inject),
    )

    _assert_rejected_without_product_residue(tmp_path, "decision field")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("artifact", "required_field"),
    (
        ("manifest.json", "primary_camera_config_sha256"),
        ("media_sidecar.json", "deidentification_status"),
        ("tracking_input.jsonl", "track_confidence"),
        ("bbox_tracklets.jsonl", "source_fps"),
        ("window_records.jsonl", "topology"),
        ("predictions.jsonl", "model_purpose"),
        ("episode_candidates.jsonl", "prediction_status"),
        ("qc_summary.json", "normalized_tracking_row_count"),
        ("model_bindings.json", "binary_class_order"),
        ("execution.json", "runtime"),
    ),
)
@pytest.mark.parametrize("drift", ("missing", "extra"))
def test_every_primary_artifact_requires_its_exact_v1_field_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    required_field: str,
    drift: str,
) -> None:
    calls: list[dict[str, object]] = []

    def inject(primary: Path) -> None:
        def mutation(value: dict[str, object]) -> None:
            if drift == "missing":
                value.pop(required_field)
            else:
                value["rogue_v1_field"] = "descriptor-aware-attack"

        _mutate_primary_artifact(primary, artifact, mutation)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _corrupting_primary_builder(["ready"], calls, inject),
    )

    _assert_rejected_without_product_residue(tmp_path, "field set")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("artifact", "field_name", "value"),
    (
        ("predictions.jsonl", "risk_score", 0.9),
        ("episode_candidates.jsonl", "medical_diagnosis", "synthetic-diagnosis"),
        ("execution.json", "recommended_action", "escalate"),
        ("manifest.json", "risk_level", "high"),
    ),
)
def test_nonempty_primary_decision_fields_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field_name: str,
    value: object,
) -> None:
    calls: list[dict[str, object]] = []

    def inject(primary: Path) -> None:
        _mutate_primary_artifact(
            primary,
            artifact,
            lambda row: row.__setitem__(field_name, value),
        )

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _corrupting_primary_builder(["ready"], calls, inject),
    )

    _assert_rejected_without_product_residue(tmp_path, "decision field")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("artifact", "nested_name", "drift"),
    (
        ("predictions.jsonl", "binary", "missing"),
        ("predictions.jsonl", "subtype", "extra"),
        ("predictions.jsonl", "four_class", "class_order"),
        ("predictions.jsonl", "binary", "probabilities"),
        ("episode_candidates.jsonl", "binary_probability_summary", "missing"),
        ("episode_candidates.jsonl", "four_class_probability_summary", "extra"),
        ("episode_candidates.jsonl", "four_class_probability_summary", "class_order"),
        ("episode_candidates.jsonl", "binary_probability_summary", "probabilities"),
    ),
)
def test_nested_probability_and_summary_contract_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    nested_name: str,
    drift: str,
) -> None:
    calls: list[dict[str, object]] = []

    def inject(primary: Path) -> None:
        def mutation(row: dict[str, object]) -> None:
            nested = row[nested_name]
            assert isinstance(nested, dict)
            if drift == "missing":
                field = "mean_probabilities" if "summary" in nested_name else "predicted_label"
                nested.pop(field)
            elif drift == "extra":
                nested["rogue_probability_field"] = True
            elif drift == "class_order":
                nested["class_order"] = list(reversed(nested["class_order"]))
            elif "summary" in nested_name:
                nested["mean_probabilities"] = [0.8, 0.2]
            else:
                nested["probabilities"] = [0.8, 0.2]

        _mutate_primary_artifact(primary, artifact, mutation)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _corrupting_primary_builder(["ready"], calls, inject),
    )

    _assert_rejected_without_product_residue(tmp_path, "probability")
    assert len(calls) == 1


@pytest.mark.parametrize("drift", ("missing", "extra", "scope"))
def test_primary_manifest_data_access_must_match_fixed_synthetic_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    calls: list[dict[str, object]] = []

    def inject(primary: Path) -> None:
        def mutation(manifest: dict[str, object]) -> None:
            data_access = manifest["data_access"]
            assert isinstance(data_access, dict)
            if drift == "missing":
                data_access.pop("sealed_camera")
            elif drift == "extra":
                data_access["rogue_source"] = False
            else:
                data_access["synthetic_fixture"] = False
                data_access["real_human_media"] = True

        _mutate_primary_artifact(primary, "manifest.json", mutation)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _corrupting_primary_builder(["ready"], calls, inject),
    )

    _assert_rejected_without_product_residue(tmp_path, "data_access")
    assert len(calls) == 1


@pytest.mark.parametrize("drift", ("missing", "extra"))
def test_product_evidence_requires_exact_field_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    calls: list[dict[str, object]] = []
    original = product_module.summarize_primary_camera_bundle

    def drifting_summary(primary: Path) -> dict[str, object]:
        evidence = original(primary)
        if drift == "missing":
            evidence.pop("duration_semantics")
        else:
            evidence["rogue_product_evidence_field"] = True
        return evidence

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _fake_primary_builder(["ready"], calls),
    )
    monkeypatch.setattr(product_module, "summarize_primary_camera_bundle", drifting_summary)

    _assert_rejected_without_product_residue(tmp_path, "wandering evidence.*field set")
    assert len(calls) == 1


@pytest.mark.parametrize("drift", ("missing", "extra"))
def test_product_manifest_requires_exact_field_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    calls: list[dict[str, object]] = []
    original = product_module._canonical_json

    def drifting_canonical_json(value: object, role: str) -> bytes:
        if role == "product manifest":
            assert isinstance(value, dict)
            value = dict(value)
            if drift == "missing":
                value.pop("real_human_media_consumed", None)
            else:
                value["rogue_product_manifest_field"] = True
        return original(value, role)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _fake_primary_builder(["ready"], calls),
    )
    monkeypatch.setattr(product_module, "_canonical_json", drifting_canonical_json)

    _assert_rejected_without_product_residue(tmp_path, "product manifest.*field set")
    assert len(calls) == 1


def test_rogue_product_top_level_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = product_module._verify_product_staging

    def inject_rogue_file(staging: Path, evidence_bytes: bytes, manifest_bytes: bytes) -> None:
        (staging / "rogue.json").write_bytes(canonical_json_bytes({"rogue": True}))
        original(staging, evidence_bytes, manifest_bytes)

    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _fake_primary_builder(["ready"], calls),
    )
    monkeypatch.setattr(product_module, "_verify_product_staging", inject_rogue_file)

    _assert_rejected_without_product_residue(tmp_path, "top-level")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("statuses", "expected_status", "expected_degraded"),
    (
        (["ready"], "ready", False),
        (["ready", "unavailable", "inference_error"], "ready", True),
        (["inference_error"], "inference_error", False),
        (["unavailable"], "unavailable", False),
        ([], "unavailable", False),
    ),
)
def test_session_status_matrix_and_degraded_semantics(
    tmp_path: Path,
    statuses: list[str],
    expected_status: str,
    expected_degraded: bool,
) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, statuses)

    evidence = summarize_primary_camera_bundle(primary)

    assert evidence["schema_version"] == SESSION_EVIDENCE_SCHEMA_VERSION
    assert evidence["product_stage"] == "session_evidence_prototype"
    assert evidence["session_status"] == expected_status
    assert evidence["degraded"] is expected_degraded
    assert evidence["ready_window_count"] == statuses.count("ready")
    assert evidence["unavailable_window_count"] == statuses.count("unavailable")
    assert evidence["inference_error_window_count"] == statuses.count("inference_error")
    assert evidence["ready_window_ratio"] == pytest.approx(
        statuses.count("ready") / len(statuses) if statuses else 0.0
    )


def test_evidence_keeps_scope_qc_episode_and_fixed_null_boundaries(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, ["ready", "unavailable", "inference_error"])

    evidence = summarize_primary_camera_bundle(primary)

    assert evidence["source_scope"] == SOURCE_SCOPE
    assert evidence["evidence_scope"] == "synthetic_contract_only"
    assert evidence["validation_scope"] == "synthetic_camera_contract"
    assert evidence["person_id"] is None
    assert evidence["person_binding_verified"] is False
    assert evidence["qc_reason_counts"] == {
        "insufficient_motion": 1,
        "trusted_model_forward_failed": 1,
    }
    assert evidence["quality_flag_counts"] == {"camera_motion_not_verified": 3}
    assert evidence["episode_candidate_count"] == 1
    assert evidence["episode_type_counts"] == {
        "direct": 0,
        "pacing": 1,
        "lapping": 0,
        "random": 0,
    }
    assert evidence["pacing_duration_sum_seconds"] == 40.0
    assert evidence["direct_duration_sum_seconds"] == 0.0
    assert evidence["episode_candidate_duration_sum_seconds"] == 40.0
    assert evidence["duration_semantics"] == "candidate_interval_sum_not_presence_time"
    assert evidence["candidate_id"] == "topowander-m0s-seed20260731-epoch0005"
    assert evidence["candidate_manifest_sha256"] == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert evidence["model_state_sha256"] == EXPECTED_MODEL_STATE_SHA256
    assert evidence["primary_seed"] == 20260731
    assert evidence["best_epoch"] == 5
    assert evidence["probability_calibrated"] is False
    assert evidence["episode_policy_status"] == "development_unfrozen"
    assert evidence["episode_merge_gap_seconds"] == 0.0
    for name in (
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
    ):
        assert evidence[name] is None
    assert evidence["algorithm_event_status"] == "not_ready_session_only"
    assert evidence["missing_conditions"] == [
        "stable_person_binding",
        "absolute_time_and_presence",
        "daily_evidence_and_personal_baseline",
        "frozen_risk_policy",
    ]
    assert 42 not in evidence.values()


def test_failure_windows_never_become_direct_negative_or_wandering(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, ["unavailable", "inference_error"])

    evidence = summarize_primary_camera_bundle(primary)

    assert evidence["episode_candidate_count"] == 0
    assert evidence["episode_type_counts"] == {
        "direct": 0,
        "pacing": 0,
        "lapping": 0,
        "random": 0,
    }
    assert all(
        evidence[f"{name}_duration_sum_seconds"] == 0.0
        for name in ("direct", "pacing", "lapping", "random")
    )


def test_cross_source_scope_drift_fails_closed(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, ["ready"])
    rows = [json.loads(line) for line in (primary / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["setup_id"] = "other-setup"
    (primary / "predictions.jsonl").write_bytes(canonical_jsonl_bytes(rows))
    _refresh_primary_descriptor(primary, "predictions.jsonl")

    with pytest.raises(WanderingCameraProductError, match="scope"):
        summarize_primary_camera_bundle(primary)


@pytest.mark.parametrize("corruption", ("missing", "hash", "schema", "count"))
def test_primary_missing_hash_schema_and_count_drift_fail_without_product_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    calls: list[dict[str, object]] = []

    def corrupting_builder(**kwargs: object) -> SimpleNamespace:
        result = _fake_primary_builder(["ready"], calls)(**kwargs)
        primary = Path(kwargs["output_dir"])
        if corruption == "missing":
            (primary / "predictions.jsonl").unlink()
        elif corruption == "hash":
            (primary / "predictions.jsonl").write_bytes(b"{}\n")
        elif corruption == "schema":
            rows = [json.loads(line) for line in (primary / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["schema_version"] = "drifted-schema"
            (primary / "predictions.jsonl").write_bytes(canonical_jsonl_bytes(rows))
            _refresh_primary_descriptor(primary, "predictions.jsonl")
        else:
            summary = json.loads((primary / "qc_summary.json").read_text(encoding="utf-8"))
            summary["window_count"] = 2
            (primary / "qc_summary.json").write_bytes(canonical_json_bytes(summary))
            _refresh_primary_descriptor(primary, "qc_summary.json")
        return result

    monkeypatch.setattr(product_module, "build_primary_camera_inference_bundle", corrupting_builder)
    output = tmp_path / "product"

    with pytest.raises(WanderingCameraProductError):
        build_wandering_camera_product(
            project_root=ROOT,
            tracking_jsonl_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            episode_merge_gap_seconds=0.0,
            output_dir=output,
        )

    assert len(calls) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".product.tmp-*"))


def test_canonical_evidence_is_deterministic_for_a_fixed_primary_bundle(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    _write_primary_bundle(primary, ["ready", "unavailable"])

    first = canonical_json_bytes(summarize_primary_camera_bundle(primary))
    second = canonical_json_bytes(summarize_primary_camera_bundle(primary))

    assert first == second
    assert first == canonical_json_bytes(json.loads(first.decode("utf-8")))


def test_product_builder_calls_primary_once_binds_manifests_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        product_module,
        "build_primary_camera_inference_bundle",
        _fake_primary_builder(["ready", "unavailable"], calls),
    )
    output = tmp_path / "product"

    result = build_wandering_camera_product(
        project_root=ROOT,
        tracking_jsonl_path=tmp_path / "tracking.jsonl",
        media_sidecar_path=tmp_path / "sidecar.json",
        episode_merge_gap_seconds=0.0,
        output_dir=output,
    )

    assert len(calls) == 1
    validated = load_validated_wandering_camera_product(output)
    assert validated.output_dir == output
    assert validated.manifest_sha256 == result.manifest_sha256
    assert len(validated.predictions) == 2
    assert len(validated.episodes) == 1
    assert validated.evidence["session_status"] == "ready"
    assert validated.product_manifest["algorithm_event_emitted"] is False
    assert Path(calls[0]["output_dir"]).name == "primary"
    assert {path.name for path in output.iterdir()} == {
        "primary",
        "wandering_evidence.json",
        "manifest.json",
    }
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["schema_version"] == PRODUCT_MANIFEST_SCHEMA_VERSION
    assert manifest["product_stage"] == "session_evidence_prototype"
    assert manifest["evidence_scope"] == "synthetic_contract_only"
    assert manifest["algorithm_event_emitted"] is False
    assert set(manifest["artifacts"]) == {
        "primary/manifest.json",
        "wandering_evidence.json",
    }
    for relative, descriptor in manifest["artifacts"].items():
        payload = (output / relative).read_bytes()
        assert descriptor == {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert result.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert result.session_status == "ready"
    assert result.degraded is True
    with pytest.raises(FileExistsError):
        build_wandering_camera_product(
            project_root=ROOT,
            tracking_jsonl_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            episode_merge_gap_seconds=0.0,
            output_dir=output,
        )
    assert len(calls) == 1


def test_builder_failure_removes_owned_staging_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = tmp_path / ".product.tmp-unrelated"
    unrelated.mkdir()

    def failing_builder(**kwargs: object) -> None:
        primary = Path(kwargs["output_dir"])
        primary.mkdir()
        (primary / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic primary failure")

    monkeypatch.setattr(product_module, "build_primary_camera_inference_bundle", failing_builder)
    output = tmp_path / "product"

    with pytest.raises(RuntimeError, match="synthetic primary failure"):
        build_wandering_camera_product(
            project_root=ROOT,
            tracking_jsonl_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            episode_merge_gap_seconds=0.0,
            output_dir=output,
        )

    assert not output.exists()
    assert unrelated.is_dir()
    assert list(tmp_path.glob(".product.tmp-*")) == [unrelated]


def test_product_source_reuses_only_the_primary_builder() -> None:
    source_path = (
        ROOT
        / "src/elderly_monitoring/modules/mental_health/wandering/camera_product.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert source.count("build_primary_camera_inference_bundle(") == 1
    for forbidden in (
        "build_camera_inference_bundle(",
        "run_yolov8_bytetrack(",
        "run_camera_inference(",
        "safe_load_development_model(",
        "safe_load_tcn_model(",
    ):
        assert forbidden not in source


def _tracking_row(bucket: int) -> dict[str, object]:
    phase = bucket / 79.0
    center_x = 220.0 + 140.0 * np.sin(phase * 2.0 * np.pi)
    center_y = 300.0 + 35.0 * np.sin(phase * 4.0 * np.pi)
    return {
        "frame_id": bucket * 10 + 1,
        "track_id": 1,
        "bbox": [center_x - 20.0, center_y - 100.0, center_x + 20.0, center_y],
        "track_confidence": 0.9,
        "timestamp_sec": bucket * 0.5 + 0.1,
    }


def _write_synthetic_pair(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    tracking = directory / "tracking.jsonl"
    tracking_payload = canonical_jsonl_bytes(_tracking_row(index) for index in range(80))
    tracking.write_bytes(tracking_payload)
    sidecar = {
        "schema_version": "wandering-media-v1",
        **SOURCE_SCOPE,
        "media_ref": "synthetic/video-0001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": hashlib.sha256(tracking_payload).hexdigest(),
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
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic",
    }
    sidecar_path = directory / "sidecar.json"
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))
    return tracking, sidecar_path


def test_cli_help_parameter_errors_and_fixed_candidate_synthetic_e2e(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for required in (
        "--tracking-jsonl",
        "--media-sidecar",
        "--episode-merge-gap-seconds",
        "--output",
    ):
        assert required in help_result.stdout
    lowered = help_result.stdout.lower()
    for forbidden in (
        "candidate",
        "manifest",
        "threshold",
        "calibration",
        "risk",
        "download",
        "input-video",
        "tracker",
    ):
        assert forbidden not in lowered
    error_result = subprocess.run(
        [sys.executable, str(CLI)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert error_result.returncode == 2

    assert PRIMARY_MANIFEST.is_file()
    tracking, sidecar = _write_synthetic_pair(tmp_path / "input")
    output = tmp_path / "product"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--tracking-jsonl",
            str(tracking),
            "--media-sidecar",
            str(sidecar),
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
    assert "session_status=ready" in completed.stdout
    evidence = json.loads((output / "wandering_evidence.json").read_text(encoding="utf-8"))
    assert evidence["session_status"] == "ready"
    assert evidence["evidence_scope"] == "synthetic_contract_only"
    assert evidence["algorithm_event"] is None
    assert (output / "primary/manifest.json").is_file()
