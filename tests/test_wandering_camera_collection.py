from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering import camera_collection
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import load_camera_inputs
from elderly_monitoring.modules.mental_health.wandering.camera_collection import (
    CameraCollectionPreparationError,
    build_authorized_camera_tracking_pair,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict:
    return {
        "schema_version": "wandering-camera-authorization-receipt-v1",
        "receipt_id": "receipt-development-0001",
        "approval_status": "approved",
        "active": True,
        "purpose": "camera_development",
        "dataset_role": "development",
        "valid_from": "2020-01-01T00:00:00+00:00",
        "expires_at": "2999-01-01T00:00:00+00:00",
        "allowed_operations": ["prepare_session", "run_development"],
        "participant_ids": ["participant-0001"],
        "session_ids": ["session-0001"],
        "camera_setup_ids": ["setup-0001"],
        "source_group_ids": ["group-0001"],
        "governance": {
            "consent_confirmed": True,
            "deidentified_storage": True,
            "access_control_confirmed": True,
            "retention_and_deletion_defined": True,
            "withdrawal_process_defined": True,
            "audio_policy": "not_collected",
        },
    }


def _collection() -> dict:
    return {
        "schema_version": "wandering-camera-collection-v1",
        "collection_id": "collection-0001",
        "dataset_role": "development",
        "authorization_receipt_id": "receipt-development-0001",
        "participants": [{"participant_id": "participant-0001"}],
        "camera_setups": [
            {"camera_setup_id": "setup-0001", "setup_id": "adapter-setup-0001"}
        ],
        "sessions": [
            {
                "session_id": "session-0001",
                "participant_id": "participant-0001",
                "source_group_id": "group-0001",
                "camera_setup_ids": ["setup-0001"],
            }
        ],
        "sources": [
            {
                "source_video_id": "video-0001",
                "session_id": "session-0001",
                "source_group_id": "group-0001",
                "camera_setup_id": "setup-0001",
                "device_id": "device-0001",
                "setup_id": "adapter-setup-0001",
                "stream_epoch": "epoch-0001",
                "tracking_ref": "inputs/tracking.jsonl",
                "media_sidecar_ref": "inputs/media.json",
            }
        ],
        "tracklet_participant_bindings": [
            {
                "source_group_id": "group-0001",
                "source_video_id": "video-0001",
                "device_id": "device-0001",
                "setup_id": "adapter-setup-0001",
                "stream_epoch": "epoch-0001",
                "track_id": 7,
                "participant_id": "participant-0001",
                "session_id": "session-0001",
                "camera_setup_id": "setup-0001",
                "clock_domain_id": "clock-0001",
            }
        ],
        "participant_present_intervals": [
            {
                "participant_id": "participant-0001",
                "session_id": "session-0001",
                "clock_domain_id": "clock-0001",
                "start_sec": 0.0,
                "end_sec_exclusive": 60.0,
            }
        ],
        "clock_alignments": [
            {
                "clock_domain_id": "clock-0001",
                "session_id": "session-0001",
                "source_group_id": "group-0001",
                "camera_setup_ids": ["setup-0001"],
                "source_video_ids": ["video-0001"],
                "status": "single_camera",
            }
        ],
    }


def _write_inputs(tmp_path: Path, *, receipt: dict | None = None, collection: dict | None = None):
    receipt_path = tmp_path / "receipt.json"
    collection_path = tmp_path / "collection.json"
    if receipt is not None:
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    if collection is not None:
        collection_path.write_text(json.dumps(collection), encoding="utf-8")
    return receipt_path, collection_path


def _call(tmp_path: Path, receipt_path: Path, collection_path: Path, **updates: object):
    values = {
        "project_root": ROOT,
        "receipt_path": receipt_path,
        "collection_path": collection_path,
        "source_video_id": "video-0001",
        "input_video_path": tmp_path / "fake-video.mp4",
        "media_ref": "deidentified/session-0001/video-0001.mp4",
        "camera_motion_state": "stable",
        "deidentification_status": "deidentified",
        "capture_started_at": None,
        "timezone": None,
        "output_dir": tmp_path / "prepared-c1",
    }
    values.update(updates)
    return build_authorized_camera_tracking_pair(**values)


def _forbid_protected_work(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("protected")
        raise AssertionError("video/runtime/temp work must remain unreachable")

    for name in (
        "_resolve_portable_detector",
        "_verify_portable_detector",
        "_resolve_video",
        "_sha256_file",
        "_probe_video",
        "_load_tracking_runtime",
        "_prepare_authorized_camera_video_session",
    ):
        monkeypatch.setattr(camera_collection, name, forbidden)
    monkeypatch.setattr(camera_collection.tempfile, "TemporaryDirectory", forbidden)
    return calls


def _enable_portable_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[Path]]:
    detector = tmp_path / "portable-runtime" / "yolov8n.pt"
    detector.parent.mkdir(parents=True, exist_ok=True)
    detector.write_bytes(b"synthetic local detector fixture")
    verification_calls: list[Path] = []
    monkeypatch.setattr(
        camera_collection, "_resolve_portable_detector", lambda root: detector.resolve()
    )

    def verify(root: Path, observed: Path) -> Path:
        verification_calls.append(observed)
        assert observed.samefile(detector)
        return detector.resolve()

    monkeypatch.setattr(camera_collection, "_verify_portable_detector", verify)
    monkeypatch.setattr(camera_collection, "_portable_network_guard", nullcontext)
    return detector.resolve(), verification_calls


@pytest.mark.parametrize(
    ("receipt", "match"),
    [
        (None, "authorization receipt"),
        ({**_receipt(), "approval_status": "pending"}, "not approved"),
        ({**_receipt(), "active": False}, "not active"),
        ({**_receipt(), "expires_at": "2021-01-01T00:00:00+00:00"}, "expired"),
    ],
)
def test_receipt_failures_precede_all_video_runtime_and_output_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: dict | None,
    match: str,
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=receipt, collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match=match):
        _call(tmp_path, receipt_path, collection_path)
    assert calls == []
    assert not (tmp_path / "prepared-c1").exists()


def test_video_direct_invalid_receipt_has_role_aware_zero_later_path_touches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    receipt["active"] = False
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=receipt, collection=_collection()
    )
    roles: list[str] = []
    original_file = camera_collection._require_external_file
    original_destination = camera_collection._require_external_destination

    def spy_file(path, root, role):
        roles.append(f"file:{role}")
        return original_file(path, root, role)

    def spy_destination(path, root, role):
        roles.append(f"destination:{role}")
        return original_destination(path, root, role)

    monkeypatch.setattr(camera_collection, "_require_external_file", spy_file)
    monkeypatch.setattr(camera_collection, "_require_external_destination", spy_destination)
    with pytest.raises(CameraCollectionPreparationError, match="not active"):
        _call(tmp_path, receipt_path, collection_path)
    assert roles == ["file:authorization receipt"]


def test_video_direct_invalid_collection_has_role_aware_zero_later_path_touches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = _collection()
    collection["sources"][0]["session_id"] = "session-outside-collection"
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=collection
    )
    roles: list[str] = []
    original_file = camera_collection._require_external_file
    original_destination = camera_collection._require_external_destination

    def spy_file(path, root, role):
        roles.append(f"file:{role}")
        return original_file(path, root, role)

    def spy_destination(path, root, role):
        roles.append(f"destination:{role}")
        return original_destination(path, root, role)

    monkeypatch.setattr(camera_collection, "_require_external_file", spy_file)
    monkeypatch.setattr(camera_collection, "_require_external_destination", spy_destination)
    with pytest.raises(CameraCollectionPreparationError):
        _call(tmp_path, receipt_path, collection_path)
    assert roles == ["file:authorization receipt", "file:collection"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("operation", "requested operation"),
        ("dataset", "development-only"),
        ("purpose", "purpose"),
        ("scope", "scope mismatch"),
    ],
)
def test_receipt_operation_dataset_purpose_and_scope_mismatches_are_pre_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    receipt = _receipt()
    if mutation == "operation":
        receipt["allowed_operations"] = ["run_development"]
    elif mutation == "dataset":
        receipt["dataset_role"] = "sealed"
    elif mutation == "purpose":
        receipt["purpose"] = "camera_evaluation"
    else:
        receipt["participant_ids"] = ["participant-other"]
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=receipt, collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match=match):
        _call(tmp_path, receipt_path, collection_path)
    assert calls == []
    assert not (tmp_path / "prepared-c1").exists()


@pytest.mark.parametrize(("case", "match"), [("unknown", "exactly one"), ("duplicate", "duplicate")])
def test_unknown_or_duplicate_source_fails_before_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
) -> None:
    collection = _collection()
    if case == "duplicate":
        collection["sources"].append(copy.deepcopy(collection["sources"][0]))
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=collection
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match=match):
        _call(
            tmp_path,
            receipt_path,
            collection_path,
            source_video_id="unknown" if case == "unknown" else "video-0001",
        )
    assert calls == []
    assert not (tmp_path / "prepared-c1").exists()


def test_existing_final_output_fails_before_video_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    (tmp_path / "prepared-c1").mkdir()
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="already exists"):
        _call(tmp_path, receipt_path, collection_path)
    assert calls == []


def test_dangling_output_entry_is_refused_before_video_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    output = tmp_path / "prepared-c1"
    try:
        output.symlink_to(tmp_path / "missing-output-target", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="dangling"):
        _call(tmp_path, receipt_path, collection_path)
    assert calls == []
    assert output.is_symlink()
    assert not (tmp_path / "missing-output-target").exists()


def test_capture_time_pair_is_required_before_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="both null or both provided"):
        _call(tmp_path, receipt_path, collection_path, capture_started_at="2026-08-13T10:00:00+08:00")
    assert calls == []


def test_success_uses_fixed_shared_tracker_builds_bound_pair_and_cleans_raw_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    video = tmp_path / "fake-video.mp4"
    video.write_bytes(b"deterministic fake media, never decoded by this test")
    raw_parents: list[Path] = []
    tracker_calls: list[dict] = []
    detector_path, detector_verification_calls = _enable_portable_detector(
        tmp_path, monkeypatch
    )

    def fake_tracker(**kwargs) -> int:
        tracker_calls.append(kwargs)
        output_path = kwargs["output_path"]
        raw_parents.append(output_path.parent)
        output_path.write_text(
            '{ "person_id":"anonymous_track_007", "track_id":7, "frame_id":1, '
            '"scene_region":"camera_development", "bbox":[10,20,50,80], '
            '"track_confidence":0.91, "center":[30,50], '
            '"speed_px_per_sec":null, "timestamp_sec":0.04 }\n'
            '{"frame_id":2,"person_id":"anonymous_track_007","track_id":7,'
            '"bbox":[11,21,51,81],"scene_region":"camera_development",'
            '"track_confidence":0.92,"center":[31,51],"speed_px_per_sec":35.355,'
            '"timestamp_sec":0.08}\n',
            encoding="utf-8",
        )
        return 2

    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=250
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(
            run=fake_tracker,
            detector_backend="ultralytics_yolo",
            detector_version="8.3.0",
            tracker_backend="bytetrack",
            tracker_version="8.3.0",
        ),
    )

    result = _call(tmp_path, receipt_path, collection_path)

    output = tmp_path / "prepared-c1"
    sidecar_path = output / "media_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["authorization_status"] == "authorized_camera_engineering_smoke"
    assert sidecar["source_video_id"] == "video-0001"
    assert sidecar["source_group_id"] == "group-0001"
    assert sidecar["device_id"] == "device-0001"
    assert sidecar["setup_id"] == "adapter-setup-0001"
    assert sidecar["stream_epoch"] == "epoch-0001"
    assert sidecar["source_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert sidecar["tracking_jsonl_sha256"] == hashlib.sha256(
        (output / "tracking.jsonl").read_bytes()
    ).hexdigest()
    assert sidecar["media_ref"] == "deidentified/session-0001/video-0001.mp4"
    assert sidecar["camera_motion_state"] == "stable"
    assert sidecar["deidentification_status"] == "deidentified"
    assert sidecar["detector"] == {
        "backend": "ultralytics_yolo",
        "model": "yolov8n.pt",
        "version": "8.3.0",
    }
    assert sidecar["tracker"] == {
        "backend": "bytetrack",
        "config": "bytetrack.yaml",
        "version": "8.3.0",
    }
    assert result["tracking_execution"] == {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "max_frames": None,
        "person_class_id": 0,
        "person_id_prefix": "anonymous_track",
        "scene_region": "camera_development",
    }
    preparation_summary = json.loads(
        (output / "preparation_summary.json").read_text(encoding="utf-8")
    )
    assert preparation_summary["input_kind"] == "authorized_video_shared_tracking_and_sidecar"
    assert preparation_summary["controller_schema_version"] == (
        "wandering-camera-video-controller-v1"
    )
    assert preparation_summary["media_opened"] is True
    assert preparation_summary["detector_run"] is True
    assert preparation_summary["tracker_run"] is True
    assert preparation_summary["detector"] == sidecar["detector"]
    assert preparation_summary["tracker"] == sidecar["tracker"]
    assert preparation_summary["tracking_execution"] == result["tracking_execution"]
    assert preparation_summary["camera_qc_run"] is False
    assert preparation_summary["model_inference_run"] is False
    assert preparation_summary["m0cam_d_started"] is False
    assert "preparation_context" not in preparation_summary
    assert preparation_summary["authorization_binding"]["receipt_sha256"] == hashlib.sha256(
        json.dumps(
            _receipt(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    assert preparation_summary["collection_binding"]["collection_sha256"] == hashlib.sha256(
        json.dumps(
            _collection(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    assert preparation_summary["source_binding"] == {
        "source_video_id": "video-0001",
        "session_id": "session-0001",
        "source_group_id": "group-0001",
        "camera_setup_id": "setup-0001",
        "device_id": "device-0001",
        "setup_id": "adapter-setup-0001",
        "stream_epoch": "epoch-0001",
        "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_sha256_basis": "controller_observed_video_pre_and_post_tracking",
    }
    assert "source_sha256" not in preparation_summary["video_metadata"]
    assert "observed_source_sha256" not in preparation_summary["source_binding"]
    assert "binding_basis" not in preparation_summary["source_binding"]

    def assert_portable_artifact_tree(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert key.lower() not in {
                    "input_video_path",
                    "local_media_path",
                    "credentials",
                    "headers",
                    "test_fixture_only",
                }
                assert_portable_artifact_tree(item)
        elif isinstance(value, list):
            for item in value:
                assert_portable_artifact_tree(item)
        elif isinstance(value, str):
            lowered = value.lower()
            assert not any(
                marker in lowered
                for marker in (
                    "://",
                    "token=",
                    "authorization:",
                    "bearer ",
                    "password",
                    "secret",
                    "cookie:",
                    "test_fixture",
                )
            )
            assert not value.startswith(("/", "\\\\"))
            assert re.match(r"^[A-Za-z]:[\\/]", value) is None

    assert_portable_artifact_tree(preparation_summary)
    assert_portable_artifact_tree(sidecar)
    assert len(tracker_calls) == 1
    assert tracker_calls[0]["video_path"] == video.resolve(strict=True)
    assert Path(tracker_calls[0]["model_name"]).samefile(detector_path)
    assert tracker_calls[0]["tracker_config"] == "bytetrack.yaml"
    assert tracker_calls[0]["max_frames"] is None
    assert detector_verification_calls == [detector_path]
    assert all(not path.exists() for path in raw_parents)
    reloaded = load_camera_inputs(
        output / "tracking.jsonl",
        sidecar_path,
        load_camera_config(ROOT / "configs/modules/wandering_camera_v1.yaml"),
    )
    assert len(reloaded.observations) == 2
    assert result["authorized_camera_data_consumed"] is True
    assert result["m0cam_d_started"] is False


def test_raw_temp_is_cleaned_when_tracker_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    (tmp_path / "fake-video.mp4").write_bytes(b"fake")
    _enable_portable_detector(tmp_path, monkeypatch)
    raw_parents: list[Path] = []

    def failed_tracker(**kwargs) -> int:
        output_path = kwargs["output_path"]
        raw_parents.append(output_path.parent)
        output_path.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("tracker failed")

    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=25
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(
            run=failed_tracker,
            detector_backend="ultralytics_yolo",
            detector_version="test-version",
            tracker_backend="bytetrack",
            tracker_version="test-version",
        ),
    )
    with pytest.raises(RuntimeError, match="tracker failed"):
        _call(tmp_path, receipt_path, collection_path)
    assert raw_parents and all(not path.exists() for path in raw_parents)
    assert not (tmp_path / "prepared-c1").exists()


def test_network_attempt_inside_tracker_is_denied_and_mapped_to_collection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    (tmp_path / "fake-video.mp4").write_bytes(b"fake")
    detector = tmp_path / "portable-runtime" / "yolov8n.pt"
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b"synthetic local detector fixture")
    monkeypatch.setattr(
        camera_collection, "_resolve_portable_detector", lambda root: detector.resolve()
    )
    monkeypatch.setattr(
        camera_collection,
        "_verify_portable_detector",
        lambda root, observed: detector.resolve(),
    )

    def network_tracker(**kwargs) -> int:
        import socket

        socket.create_connection(("example.invalid", 443))
        raise AssertionError("network denial must stop the tracker")

    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=25
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(
            run=network_tracker,
            detector_backend="ultralytics_yolo",
            detector_version="test-version",
            tracker_backend="bytetrack",
            tracker_version="test-version",
        ),
    )
    with pytest.raises(
        CameraCollectionPreparationError, match="network guard"
    ):
        _call(tmp_path, receipt_path, collection_path)
    assert not (tmp_path / "prepared-c1").exists()


def test_swallowed_network_attempt_inside_tracker_still_blocks_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    (tmp_path / "fake-video.mp4").write_bytes(b"fake")
    detector = tmp_path / "portable-runtime" / "yolov8n.pt"
    detector.parent.mkdir(parents=True)
    detector.write_bytes(b"synthetic local detector fixture")
    token = object()
    monkeypatch.setattr(
        camera_collection, "_resolve_portable_detector", lambda root: token
    )
    monkeypatch.setattr(
        camera_collection,
        "_verify_portable_detector",
        lambda root, observed: detector.resolve(),
    )

    def swallowing_tracker(**kwargs) -> int:
        import socket
        from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
            NetworkAccessDeniedError,
        )

        try:
            socket.create_connection(("example.invalid", 443))
        except NetworkAccessDeniedError:
            pass
        kwargs["output_path"].write_text(
            '{"frame_id":1,"track_id":7,"bbox":[10,20,50,80],'
            '"track_confidence":0.91,"timestamp_sec":0.04}\n',
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=25
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(
            run=swallowing_tracker,
            detector_backend="ultralytics_yolo",
            detector_version="test-version",
            tracker_backend="bytetrack",
            tracker_version="test-version",
        ),
    )
    with pytest.raises(CameraCollectionPreparationError, match="network guard"):
        _call(tmp_path, receipt_path, collection_path)
    assert not (tmp_path / "prepared-c1").exists()


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("detector_backend", "https://backend.invalid"),
        ("detector_backend", "token=secret"),
        ("tracker_backend", ".."),
        ("detector_version", "1.0?token=secret"),
        ("tracker_version", "C:\\secret"),
        ("tracker_version", "bad\x01version"),
    ],
)
def test_runtime_component_attacks_fail_before_raw_temp_and_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: str,
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    (tmp_path / "fake-video.mp4").write_bytes(b"fake")
    _enable_portable_detector(tmp_path, monkeypatch)
    tracker_calls: list[str] = []
    temp_calls: list[str] = []

    def forbidden_tracker(**kwargs) -> int:
        tracker_calls.append("tracker")
        raise AssertionError("invalid component must fail before tracker")

    def forbidden_temp(*args, **kwargs):
        temp_calls.append("temp")
        raise AssertionError("invalid component must fail before raw temp")

    values = {
        "run": forbidden_tracker,
        "detector_backend": "ultralytics_yolo",
        "detector_version": "8.3.0",
        "tracker_backend": "bytetrack",
        "tracker_version": "8.3.0",
    }
    values[field] = bad
    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=25
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(**values),
    )
    monkeypatch.setattr(camera_collection.tempfile, "TemporaryDirectory", forbidden_temp)
    with pytest.raises(CameraCollectionPreparationError, match="cannot prepare authorized"):
        _call(tmp_path, receipt_path, collection_path)
    assert tracker_calls == []
    assert temp_calls == []
    assert not (tmp_path / "prepared-c1").exists()


def test_build_tracking_cli_help_has_no_authorization_scope_or_test_bypass_flags() -> None:
    script = ROOT / "scripts/wandering/build_camera_tracking_pair.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = completed.stdout.lower()
    assert "--receipt" in help_text
    assert "--collection" in help_text
    assert "--source-video-id" in help_text
    assert "--input-video" in help_text
    for forbidden in (
        "--authorization-status",
        "--validation-scope",
        "--evidence-scope",
        "--test-fixture",
        "--fake-tracker",
        "--candidate-loader",
        "--skip",
        "--bypass",
    ):
        assert forbidden not in help_text


def test_production_api_has_no_model_scope_test_or_fake_runtime_parameters() -> None:
    parameters = set(inspect.signature(build_authorized_camera_tracking_pair).parameters)
    assert parameters == {
        "project_root",
        "receipt_path",
        "collection_path",
        "source_video_id",
        "input_video_path",
        "media_ref",
        "camera_motion_state",
        "deidentification_status",
        "capture_started_at",
        "timezone",
        "output_dir",
    }


def test_camera_collection_exports_only_controller_and_error() -> None:
    assert camera_collection.__all__ == [
        "CameraCollectionPreparationError",
        "build_authorized_camera_tracking_pair",
    ]
    assert "TrackingRuntime" not in vars(camera_collection)
    assert "VideoMetadata" not in vars(camera_collection)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"media_ref": "C:drive-relative.mp4"}, "media_ref"),
        ({"media_ref": "a" * 257}, "media_ref"),
        ({"media_ref": "safe/../escape.mp4"}, "media_ref"),
        (
            {
                "capture_started_at": "2026-08-13T10:00:00+08:00",
                "timezone": "Not/A_Real_Zone",
            },
            "timezone",
        ),
        (
            {
                "capture_started_at": "2026-08-13T10:00:00+08:00",
                "timezone": "../UTC",
            },
            "timezone",
        ),
    ],
)
def test_bad_media_ref_or_timezone_fails_before_all_video_runtime_and_temp_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    match: str,
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match=match):
        _call(tmp_path, receipt_path, collection_path, **updates)
    assert calls == []
    assert not (tmp_path / "prepared-c1").exists()


def test_in_project_production_paths_fail_before_video_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="active checkout identity"):
        _call(
            tmp_path,
            receipt_path,
            collection_path,
            project_root=tmp_path,
        )
    assert calls == []


def test_external_symlink_resolving_into_project_fails_before_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_link = tmp_path / "receipt-link.json"
    try:
        receipt_link.symlink_to(ROOT / "README.md")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="outside project root"):
        _call(tmp_path, receipt_link, tmp_path / "collection.json")
    assert calls == []


def test_dotdot_and_output_symlink_resolving_into_active_checkout_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    output_alias = tmp_path / "checkout-output-alias"
    try:
        output_alias.symlink_to(ROOT, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="outside project root"):
        _call(
            tmp_path,
            receipt_path,
            collection_path,
            output_dir=output_alias / "reports" / ".." / ".forbidden-f2-output",
        )
    assert calls == []
    assert not (ROOT / ".forbidden-f2-output").exists()


def test_drvfs_case_alias_of_active_checkout_is_rejected_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias_text = str(ROOT).replace("/Users/", "/users/")
    alias = Path(alias_text)
    if alias == ROOT or not alias.exists():
        pytest.skip("case-insensitive DrvFS alias is unavailable")
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="outside project root"):
        _call(
            tmp_path,
            receipt_path,
            collection_path,
            output_dir=alias / ".forbidden-f2-case-output",
        )
    assert calls == []


def test_same_length_video_mutation_after_tracker_fails_and_cleans_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    video = tmp_path / "fake-video.mp4"
    video.write_bytes(b"original-bytes")
    _enable_portable_detector(tmp_path, monkeypatch)
    raw_parents: list[Path] = []

    def mutating_tracker(**kwargs) -> int:
        output_path = kwargs["output_path"]
        raw_parents.append(output_path.parent)
        output_path.write_text(
            '{"frame_id":1,"track_id":7,"bbox":[10,20,50,80],'
            '"track_confidence":0.91,"timestamp_sec":0.04}\n',
            encoding="utf-8",
        )
        video.write_bytes(b"mutated--bytes")
        assert len(b"original-bytes") == len(b"mutated--bytes")
        return 1

    monkeypatch.setattr(
        camera_collection,
        "_probe_video",
        lambda path: camera_collection._VideoMetadata(
            width=640, height=480, fps=25.0, frame_count=25
        ),
    )
    monkeypatch.setattr(
        camera_collection,
        "_load_tracking_runtime",
        lambda: camera_collection._TrackingRuntime(
            run=mutating_tracker,
            detector_backend="ultralytics_yolo",
            detector_version="test-version",
            tracker_backend="bytetrack",
            tracker_version="test-version",
        ),
    )
    with pytest.raises(CameraCollectionPreparationError, match="changed during tracking"):
        _call(tmp_path, receipt_path, collection_path)
    assert raw_parents and all(not path.exists() for path in raw_parents)
    assert not (tmp_path / "prepared-c1").exists()


def test_in_project_output_path_fails_before_video_runtime_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path = _write_inputs(
        tmp_path, receipt=_receipt(), collection=_collection()
    )
    calls = _forbid_protected_work(monkeypatch)
    with pytest.raises(CameraCollectionPreparationError, match="outside project root"):
        _call(
            tmp_path,
            receipt_path,
            collection_path,
            output_dir=ROOT / ".forbidden-c01-f-output",
        )
    assert calls == []
    assert not (ROOT / ".forbidden-c01-f-output").exists()
