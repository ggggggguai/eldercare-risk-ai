from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering import camera_dataset

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    load_camera_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    CameraDatasetError,
    build_camera_readiness,
    load_camera_collection_config,
    prepare_authorized_camera_session,
    prepare_camera_input_pair,
    validate_authorization_receipt,
    validate_collection_manifest,
    validate_episode_annotations,
    write_authorized_camera_annotations,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/wandering_camera_collection_v1.yaml"


def _config() -> dict:
    return load_camera_collection_config(CONFIG)


def _receipt() -> dict:
    return {
        "schema_version": "wandering-camera-authorization-receipt-v1",
        "receipt_id": "receipt-development-0001",
        "approval_status": "approved",
        "active": True,
        "purpose": "camera_development",
        "dataset_role": "development",
        "valid_from": "2026-08-01T00:00:00+08:00",
        "expires_at": "2026-09-01T00:00:00+08:00",
        "allowed_operations": ["prepare_session", "validate_annotations", "run_development"],
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


def _collection(*, dataset_role: str = "development") -> dict:
    return {
        "schema_version": "wandering-camera-collection-v1",
        "collection_id": "collection-0001",
        "dataset_role": dataset_role,
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
                "status": "aligned",
            }
        ],
    }


def _annotation(**updates: object) -> dict:
    value = {
        "annotation_id": "annotation-0001",
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
        "start_sec": 10.0,
        "end_sec_exclusive": 20.0,
        "observable_pattern": "pacing",
        "purpose_context": "purposeful",
        "purpose_evidence": "participant_report",
        "evaluation_role": "purposeful_hard_negative",
        "script_type": "phone_call",
        "visibility_quality": "good",
        "tracking_issue": "none",
        "annotation_status": "accepted",
        "annotator_id": "annotator-0001",
        "reviewed_by": "reviewer-0001",
    }
    value.update(updates)
    return value


def _media_sidecar(tracking_sha256: str) -> dict:
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": "synthetic-video-0001",
        "source_group_id": "synthetic-group-0001",
        "device_id": "synthetic-device-0001",
        "setup_id": "synthetic-setup-0001",
        "stream_epoch": "synthetic-epoch-0001",
        "media_ref": "synthetic/video-0001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": tracking_sha256,
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 25.0,
        "duration_sec": 5.0,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {
            "backend": "synthetic_detector",
            "model": "synthetic.pt",
            "version": "test-only",
        },
        "tracker": {
            "backend": "synthetic_tracker",
            "config": "synthetic.yaml",
            "version": "test-only",
        },
        "fixed_camera_assumed": True,
        "camera_motion_state": "not_checked",
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic",
    }


def _write_test_pair(tmp_path: Path) -> tuple[Path, Path]:
    tracking = tmp_path / "tracking.jsonl"
    tracking.write_text(
        '{"frame_id":1,"track_id":7,"bbox":[10,20,50,80],'
        '"track_confidence":0.91,"timestamp_sec":0.04}\n',
        encoding="utf-8",
    )
    sidecar = tmp_path / "media_sidecar.json"
    sidecar.write_text(
        json.dumps(_media_sidecar(hashlib.sha256(tracking.read_bytes()).hexdigest())),
        encoding="utf-8",
    )
    return tracking, sidecar


def _write_authorized_pair(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    receipt_path = tmp_path / "receipt.json"
    collection_path = tmp_path / "collection.json"
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    collection_path.write_text(json.dumps(_collection()), encoding="utf-8")
    tracking, sidecar_path = _write_test_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(
        {
            "source_video_id": "video-0001",
            "source_group_id": "group-0001",
            "device_id": "device-0001",
            "setup_id": "adapter-setup-0001",
            "stream_epoch": "epoch-0001",
            "authorization_status": "authorized_camera_engineering_smoke",
            "deidentification_status": "deidentified",
        }
    )
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return receipt_path, collection_path, tracking, sidecar_path


def test_fixed_collection_config_loads_exact_contract() -> None:
    config = _config()
    assert config["schema_version"] == "wandering-camera-collection-config-v1"
    assert config["dataset_roles"] == ["development", "sealed"]
    assert config["readiness_milestones"] == ["C0", "C1", "C2", "C3"]
    assert "matching_policy" not in config
    assert "uncertain_threshold" not in config


def test_receipt_requires_active_approved_development_scope() -> None:
    summary = validate_authorization_receipt(
        _receipt(),
        _config(),
        operation="run_development",
        participant_ids={"participant-0001"},
        session_ids={"session-0001"},
        camera_setup_ids={"setup-0001"},
        source_group_ids={"group-0001"},
        now="2026-08-13T12:00:00+08:00",
    )
    assert summary["receipt_id"] == "receipt-development-0001"
    assert summary["approved"] is True
    assert "participant_ids" not in summary
    assert len(summary["receipt_sha256"]) == 64

    for field, bad in (
        ("approval_status", "pending"),
        ("active", False),
        ("purpose", "camera_sealed_evaluation"),
        ("dataset_role", "sealed"),
    ):
        receipt = _receipt()
        receipt[field] = bad
        with pytest.raises(CameraDatasetError):
            validate_authorization_receipt(
                receipt,
                _config(),
                operation="run_development",
                now="2026-08-13T12:00:00+08:00",
            )

    with pytest.raises(CameraDatasetError, match="expired"):
        validate_authorization_receipt(
            _receipt(),
            _config(),
            operation="run_development",
            now="2026-09-02T00:00:00+08:00",
        )
    with pytest.raises(CameraDatasetError, match="scope"):
        validate_authorization_receipt(
            _receipt(),
            _config(),
            operation="run_development",
            participant_ids={"participant-9999"},
            now="2026-08-13T12:00:00+08:00",
        )


def test_collection_cross_references_group_isolation_and_privacy() -> None:
    value = validate_collection_manifest(_collection(), _config())
    assert value["sources"][0]["source_group_id"] == value["sessions"][0]["source_group_id"]

    bad = _collection()
    bad["sources"][0]["source_group_id"] = "group-crossed"
    with pytest.raises(CameraDatasetError, match="source_group"):
        validate_collection_manifest(bad, _config())

    bad = _collection()
    bad["participants"].append({"participant_id": "participant-0001"})
    with pytest.raises(CameraDatasetError, match="duplicate"):
        validate_collection_manifest(bad, _config())

    for forbidden in (
        "https://camera.example/live?token=secret",
        "C:/Users/operator/video.mp4",
        "/home/operator/video.mp4",
    ):
        bad = _collection()
        bad["sources"][0]["tracking_ref"] = forbidden
        with pytest.raises(CameraDatasetError):
            validate_collection_manifest(bad, _config())


def test_c3_binding_covers_full_tracking_scope_session_setup_and_clock() -> None:
    value = validate_collection_manifest(_collection(), _config())
    binding = value["tracklet_participant_bindings"][0]
    assert {
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
    } == set(binding)

    for field, bad_value in (
        ("session_id", "session-wrong"),
        ("camera_setup_id", "setup-wrong"),
        ("setup_id", "adapter-setup-wrong"),
        ("clock_domain_id", "clock-wrong"),
    ):
        bad = _collection()
        bad["tracklet_participant_bindings"][0][field] = bad_value
        with pytest.raises(CameraDatasetError, match="binding"):
            validate_collection_manifest(bad, _config())

    duplicate = _collection()
    duplicate["tracklet_participant_bindings"].append(
        copy.deepcopy(duplicate["tracklet_participant_bindings"][0])
    )
    with pytest.raises(CameraDatasetError, match="duplicate tracklet binding"):
        validate_collection_manifest(duplicate, _config())


def test_annotations_keep_shape_and_purpose_separate_and_preserve_truth_states() -> None:
    collection = validate_collection_manifest(_collection(), _config())
    rows = validate_episode_annotations(
        [
            _annotation(),
            _annotation(
                annotation_id="annotation-0002",
                start_sec=20.0,
                end_sec_exclusive=25.0,
                observable_pattern="unknown",
                purpose_context="unknown",
                purpose_evidence="unknown",
                evaluation_role="uncertain",
                annotation_status="uncertain",
            ),
            _annotation(
                annotation_id="annotation-0003",
                start_sec=25.0,
                end_sec_exclusive=30.0,
                observable_pattern="unknown",
                purpose_context="unknown",
                purpose_evidence="unknown",
                evaluation_role="excluded",
                annotation_status="excluded",
            ),
        ],
        collection,
    )
    assert rows[0]["observable_pattern"] == "pacing"
    assert rows[0]["purpose_context"] == "purposeful"
    assert rows[0]["evaluation_role"] == "purposeful_hard_negative"
    assert [row["annotation_status"] for row in rows[1:]] == ["uncertain", "excluded"]

    bad = _annotation(purpose_context="unknown", purpose_evidence="participant_report")
    with pytest.raises(CameraDatasetError, match="purpose"):
        validate_episode_annotations([bad], collection)
    bad = _annotation(annotation_id="annotation-duplicate")
    with pytest.raises(CameraDatasetError, match="overlap"):
        validate_episode_annotations([bad, copy.deepcopy(bad)], collection)

    accepted_unknown = _annotation(
        observable_pattern="unknown",
        purpose_context="unknown",
        purpose_evidence="unknown",
        evaluation_role="ordinary_negative",
    )
    with pytest.raises(CameraDatasetError, match="accepted truth.*scoreable shape"):
        validate_episode_annotations([accepted_unknown], collection)


def test_readiness_is_not_ready_without_real_c0_c1_c2_c3() -> None:
    report = build_camera_readiness(config=_config())
    assert report["readiness_status"] == "not_ready"
    assert report["authorized_camera_data_consumed"] is False
    assert report["milestones"] == {"C0": False, "C1": False, "C2": False, "C3": False}
    assert "metrics" not in report
    assert "f1" not in json.dumps(report).lower()


def test_prepare_pair_only_validates_and_writes_canonical_inputs(tmp_path: Path) -> None:
    tracking, sidecar = _write_test_pair(tmp_path)
    output = tmp_path / "canonical"
    result = prepare_camera_input_pair(
        tracking,
        sidecar,
        output,
        camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
    )
    assert result["input_kind"] == "synthetic_tracking_and_sidecar"
    assert result["media_opened"] is False
    assert result["detector_run"] is False
    assert result["tracker_run"] is False
    assert result["camera_qc_run"] is False
    assert result["model_inference_run"] is False
    assert result["m0cam_d_started"] is False
    assert result["authorization_status"] == "synthetic_fixture"
    assert result["evidence_scope"] == "synthetic_schema_contract_only"
    assert result["authorized_camera_data_consumed"] is False
    assert result["normalized_tracking_bytes"] == len((output / "tracking.jsonl").read_bytes())
    assert result == json.loads(
        (output / "preparation_summary.json").read_text(encoding="utf-8")
    )
    assert (output / "tracking.jsonl").is_file()
    assert (output / "media_sidecar.json").is_file()
    assert (output / "preparation_summary.json").is_file()
    assert hashlib.sha256((output / "tracking.jsonl").read_bytes()).hexdigest() == result[
        "normalized_tracking_sha256"
    ]
    with pytest.raises(CameraDatasetError, match="already exists"):
        prepare_camera_input_pair(
            tracking,
            sidecar,
            output,
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )


def test_prepare_pair_round_trips_shared_tracker_shape_and_rebinds_sidecar(
    tmp_path: Path,
) -> None:
    tracking = tmp_path / "shared-tracker.jsonl"
    tracking.write_bytes(
        (
            '{ "speed_px_per_sec" : null, "timestamp_sec" : 0.04, '
            '"center" : [30.0, 60.0], "scene_region" : "room", '
            '"bbox" : [10, 20, 50, 80], "person_id" : "elder_007", '
            '"track_confidence" : 0.91, "track_id" : 7, "frame_id" : 1 }\r\n'
            '{"track_id":7, "frame_id":2, "bbox":[11,21,51,81], '
            '"track_confidence":0.92, "timestamp_sec":0.08, '
            '"person_id":"elder_007", "scene_region":"room", '
            '"center":[31.0,61.0], "speed_px_per_sec":35.355}\r\n'
        ).encode("utf-8")
    )
    sidecar_value = _media_sidecar(hashlib.sha256(tracking.read_bytes()).hexdigest())
    sidecar = tmp_path / "source-sidecar.json"
    sidecar.write_text(json.dumps(sidecar_value, indent=3), encoding="utf-8", newline="\r\n")
    source_sidecar_bytes = sidecar.read_bytes()

    output = tmp_path / "canonical"
    result = prepare_camera_input_pair(
        tracking,
        sidecar,
        output,
        camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
    )

    assert sidecar.read_bytes() == source_sidecar_bytes
    normalized_tracking = output / "tracking.jsonl"
    output_sidecar = output / "media_sidecar.json"
    output_sidecar_value = json.loads(output_sidecar.read_text(encoding="utf-8"))
    normalized_sha256 = hashlib.sha256(normalized_tracking.read_bytes()).hexdigest()
    assert output_sidecar_value["tracking_jsonl_sha256"] == normalized_sha256
    assert result["source_sidecar_sha256"] == hashlib.sha256(source_sidecar_bytes).hexdigest()
    assert result["source_sidecar_bytes"] == len(source_sidecar_bytes)
    assert result["output_sidecar_sha256"] == hashlib.sha256(output_sidecar.read_bytes()).hexdigest()
    assert result["output_sidecar_bytes"] == len(output_sidecar.read_bytes())
    reloaded = load_camera_inputs(
        normalized_tracking,
        output_sidecar,
        load_camera_config(ROOT / "configs/modules/wandering_camera_v1.yaml"),
    )
    assert len(reloaded.observations) == 2
    assert all(
        set(row) == {"bbox", "frame_id", "timestamp_sec", "track_confidence", "track_id"}
        for row in reloaded.normalized_rows
    )


def test_prepare_pair_round_trip_failure_leaves_no_final_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking, sidecar = _write_test_pair(tmp_path)
    output = tmp_path / "canonical"
    real_loader = load_camera_inputs
    calls = 0

    def fail_round_trip(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_loader(*args, **kwargs)
        raise CameraAdapterError("forced staged round-trip failure")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset.load_camera_inputs",
        fail_round_trip,
    )
    with pytest.raises(CameraDatasetError, match="round-trip"):
        prepare_camera_input_pair(
            tracking,
            sidecar,
            output,
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == 2
    assert not output.exists()
    assert list(tmp_path.glob(".canonical.tmp-*")) == []


def test_prepare_session_is_receipt_first_and_zero_reads_on_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt()
    receipt["allowed_operations"] = ["validate_annotations"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("protected_input")
        raise AssertionError("protected input must not be read")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset.prepare_camera_input_pair",
        forbidden,
    )
    with pytest.raises(CameraDatasetError, match="requested operation"):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=tmp_path / "missing-collection.json",
            tracking_jsonl_path=tmp_path / "missing-tracking.jsonl",
            media_sidecar_path=tmp_path / "missing-sidecar.json",
            output_dir=tmp_path / "output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_invalid_receipt_touches_only_active_identity_configs_and_receipt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt()
    receipt["approval_status"] = "pending"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    roles: list[str] = []
    original_file = camera_dataset._require_external_file
    original_destination = camera_dataset._require_external_destination

    def spy_file(path, root, role):
        roles.append(f"file:{role}")
        return original_file(path, root, role)

    def spy_destination(path, root, role):
        roles.append(f"destination:{role}")
        return original_destination(path, root, role)

    monkeypatch.setattr(camera_dataset, "_require_external_file", spy_file)
    monkeypatch.setattr(camera_dataset, "_require_external_destination", spy_destination)
    with pytest.raises(CameraDatasetError, match="not approved"):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=tmp_path / "must-not-touch-collection.json",
            tracking_jsonl_path=tmp_path / "must-not-touch-tracking.jsonl",
            media_sidecar_path=tmp_path / "must-not-touch-sidecar.json",
            output_dir=tmp_path / "must-not-touch-output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert roles == ["file:authorization receipt"]


def test_invalid_collection_touches_no_later_pair_or_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    collection_path = tmp_path / "collection.json"
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    collection = _collection()
    collection["participants"][0]["participant_id"] = "participant-outside-receipt"
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    roles: list[str] = []
    original_file = camera_dataset._require_external_file
    original_destination = camera_dataset._require_external_destination

    def spy_file(path, root, role):
        roles.append(f"file:{role}")
        return original_file(path, root, role)

    def spy_destination(path, root, role):
        roles.append(f"destination:{role}")
        return original_destination(path, root, role)

    monkeypatch.setattr(camera_dataset, "_require_external_file", spy_file)
    monkeypatch.setattr(camera_dataset, "_require_external_destination", spy_destination)
    with pytest.raises(CameraDatasetError):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=tmp_path / "must-not-touch-tracking.jsonl",
            media_sidecar_path=tmp_path / "must-not-touch-sidecar.json",
            output_dir=tmp_path / "must-not-touch-output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert roles == ["file:authorization receipt", "file:collection"]


def test_active_checkout_and_fixed_config_identity_reject_copies_before_receipt_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path / "second-checkout"
    (fake_root / "src/elderly_monitoring").mkdir(parents=True)
    (fake_root / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    copied_collection_config = tmp_path / "copied-collection.yaml"
    copied_camera_config = tmp_path / "copied-camera.yaml"
    copied_collection_config.write_bytes(CONFIG.read_bytes())
    copied_camera_config.write_bytes(
        (ROOT / "configs/modules/wandering_camera_v1.yaml").read_bytes()
    )
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("receipt")
        raise AssertionError("receipt must remain untouched")

    monkeypatch.setattr(camera_dataset, "_require_external_file", forbidden)
    common = dict(
        receipt_path=tmp_path / "receipt.json",
        collection_path=tmp_path / "collection.json",
        tracking_jsonl_path=tmp_path / "tracking.jsonl",
        media_sidecar_path=tmp_path / "sidecar.json",
        output_dir=tmp_path / "output",
        collection_config_path=CONFIG,
        camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
    )
    with pytest.raises(CameraDatasetError, match="active checkout"):
        prepare_authorized_camera_session(project_root=fake_root, **common)
    for updates in (
        {"collection_config_path": copied_collection_config},
        {"camera_config_path": copied_camera_config},
    ):
        with pytest.raises(CameraDatasetError, match="fixed active checkout config"):
            prepare_authorized_camera_session(
                project_root=ROOT,
                **{**common, **updates},
            )
    assert calls == []


def test_samefile_checkout_and_config_aliases_are_accepted(tmp_path: Path) -> None:
    receipt_path, collection_path, tracking, sidecar = _write_authorized_pair(tmp_path)
    root_alias = tmp_path / "active-root-alias"
    collection_config_alias = tmp_path / "collection-config-alias.yaml"
    camera_config_alias = tmp_path / "camera-config-alias.yaml"
    try:
        root_alias.symlink_to(ROOT, target_is_directory=True)
        collection_config_alias.symlink_to(CONFIG)
        camera_config_alias.symlink_to(ROOT / "configs/modules/wandering_camera_v1.yaml")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    output = tmp_path / "alias-output"
    result = prepare_authorized_camera_session(
        project_root=root_alias / ".",
        receipt_path=receipt_path,
        collection_path=collection_path,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        output_dir=output,
        collection_config_path=collection_config_alias,
        camera_config_path=camera_config_alias,
    )
    assert result == json.loads((output / "preparation_summary.json").read_text(encoding="utf-8"))


def test_source_binding_tamper_fails_before_atomic_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path, tracking, sidecar = _write_authorized_pair(tmp_path)
    original = camera_dataset._source_binding

    def tampered(*args, **kwargs):
        value = original(*args, **kwargs)
        value["source_sha256_basis"] = "validated_sidecar_declaration_tampered"
        return value

    monkeypatch.setattr(camera_dataset, "_source_binding", tampered)
    output = tmp_path / "tampered-output"
    with pytest.raises(CameraDatasetError, match="source binding is inconsistent"):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            output_dir=output,
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert not output.exists()
    assert list(tmp_path.glob(".tampered-output.tmp-*")) == []


def test_annotation_builder_is_receipt_first_and_zero_reads_on_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt()
    receipt["active"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("annotations")
        raise AssertionError("annotations must not be read")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset._read_jsonl_objects",
        forbidden,
    )
    with pytest.raises(CameraDatasetError, match="not active"):
        write_authorized_camera_annotations(
            receipt_path=receipt_path,
            collection_path=tmp_path / "missing-collection.json",
            annotations_path=tmp_path / "missing-annotations.jsonl",
            output_path=tmp_path / "annotations.jsonl",
            collection_config_path=CONFIG,
        )
    assert calls == []
    assert not (tmp_path / "annotations.jsonl").exists()


def test_auxiliary_clis_require_receipt_and_expose_no_bypass_flags() -> None:
    prepare_text = (ROOT / "scripts/wandering/prepare_camera_session.py").read_text(
        encoding="utf-8"
    )
    annotation_text = (ROOT / "scripts/wandering/build_camera_annotations.py").read_text(
        encoding="utf-8"
    )
    for text in (prepare_text, annotation_text):
        assert "--receipt" in text
        assert "--collection" in text
        for forbidden in (
            "allow-synthetic",
            "fake-runtime",
            "bypass",
            "evidence-scope",
            "validation-scope",
            "test-fixture",
            "test-hooks",
            "candidate-loader",
        ):
            assert forbidden not in text


def test_public_preparation_signatures_have_no_provenance_context_or_hooks() -> None:
    pair_parameters = set(inspect.signature(prepare_camera_input_pair).parameters)
    session_parameters = set(inspect.signature(prepare_authorized_camera_session).parameters)
    assert pair_parameters == {
        "tracking_jsonl_path",
        "media_sidecar_path",
        "output_dir",
        "camera_config",
    }
    assert session_parameters == {
        "project_root",
        "receipt_path",
        "collection_path",
        "tracking_jsonl_path",
        "media_sidecar_path",
        "output_dir",
        "collection_config_path",
        "camera_config_path",
    }
    forbidden = ("context", "provenance", "status", "scope", "hook", "fake")
    assert not any(any(marker in name.lower() for marker in forbidden) for name in pair_parameters)
    assert not any(any(marker in name.lower() for marker in forbidden) for name in session_parameters)
    with pytest.raises(TypeError):
        prepare_camera_input_pair(
            "tracking",
            "sidecar",
            "output",
            camera_config="config",
            preparation_context={},
        )
    with pytest.raises(TypeError):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path="receipt",
            collection_path="collection",
            tracking_jsonl_path="tracking",
            media_sidecar_path="sidecar",
            output_dir="output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
            preparation_context={},
        )


def test_public_synthetic_helper_rejects_authorized_sidecar_before_tracking_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, tracking, sidecar = _write_authorized_pair(tmp_path)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("tracking")
        raise AssertionError("authorized sidecar must fail before tracking")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset.load_camera_inputs",
        forbidden,
    )
    output = tmp_path / "synthetic-output"
    with pytest.raises(CameraDatasetError, match="synthetic"):
        prepare_camera_input_pair(
            tracking,
            sidecar,
            output,
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not output.exists()


def test_authorized_pair_writes_fixed_false_provenance_and_safe_atomic_bindings(
    tmp_path: Path,
) -> None:
    receipt_path, collection_path, tracking, sidecar = _write_authorized_pair(tmp_path)
    output = tmp_path / "authorized-output"
    result = prepare_authorized_camera_session(
        project_root=ROOT,
        receipt_path=receipt_path,
        collection_path=collection_path,
        tracking_jsonl_path=tracking,
        media_sidecar_path=sidecar,
        output_dir=output,
        collection_config_path=CONFIG,
        camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
    )
    persisted = json.loads((output / "preparation_summary.json").read_text(encoding="utf-8"))
    assert result == persisted
    assert persisted["input_kind"] == "authorized_tracking_and_sidecar"
    for field in (
        "media_opened",
        "detector_run",
        "tracker_run",
        "camera_qc_run",
        "model_inference_run",
        "m0cam_d_started",
    ):
        assert persisted[field] is False
    assert persisted["authorized_camera_data_consumed"] is True
    assert persisted["normalized_tracking_bytes"] == len(
        (output / "tracking.jsonl").read_bytes()
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    collection = validate_collection_manifest(
        json.loads(collection_path.read_text(encoding="utf-8")), _config()
    )
    assert persisted["authorization_binding"]["receipt_sha256"] == hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    ).hexdigest()
    assert persisted["collection_binding"]["collection_sha256"] == hashlib.sha256(
        json.dumps(collection, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    ).hexdigest()
    persisted_sidecar = json.loads((output / "media_sidecar.json").read_text(encoding="utf-8"))
    assert persisted["source_binding"] == {
        "source_video_id": "video-0001",
        "session_id": "session-0001",
        "source_group_id": "group-0001",
        "camera_setup_id": "setup-0001",
        "device_id": "device-0001",
        "setup_id": "adapter-setup-0001",
        "stream_epoch": "epoch-0001",
        "source_sha256": persisted_sidecar["source_sha256"],
        "source_sha256_basis": "validated_sidecar_declaration",
    }
    assert "observed_source_sha256" not in persisted["source_binding"]
    assert "binding_basis" not in persisted["source_binding"]
    forbidden_keys = {
        "receipt",
        "collection",
        "participant_ids",
        "session_ids",
        "camera_setup_ids",
        "source_group_ids",
        "local_media_path",
        "input_video_path",
        "headers",
        "credentials",
        "token",
    }
    forbidden_markers = (
        "authorization:",
        "bearer ",
        "token=",
        "http://",
        "https://",
        "rtsp://",
    )

    def assert_safe_tree(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert key.lower() not in forbidden_keys
                if key.lower() in {
                    "participants",
                    "sessions",
                    "camera_setups",
                    "source_groups",
                }:
                    assert isinstance(item, int) and not isinstance(item, bool)
                assert_safe_tree(item)
        elif isinstance(value, list):
            for item in value:
                assert_safe_tree(item)
        elif isinstance(value, str):
            lowered = value.lower()
            assert not any(marker in lowered for marker in forbidden_markers)
            assert not value.startswith(("/", "\\\\"))
            assert re.match(r"^[A-Za-z]:[\\/]", value) is None

    assert_safe_tree(persisted)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("media_ref", "C:drive-relative.mp4"),
        ("media_ref", "safe/../escape.mp4"),
        ("media_ref", "https://camera.invalid/video.mp4"),
        ("media_ref", "a" * 257),
        ("timezone", "Not/A_Real_Zone"),
        ("timezone", "../UTC"),
    ],
)
def test_bad_portable_sidecar_metadata_fails_before_tracking_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: str,
) -> None:
    tracking, sidecar_path = _write_test_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = bad
    if field == "timezone":
        sidecar["capture_started_at"] = "2026-08-13T10:00:00+08:00"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("tracking")
        raise AssertionError("bad metadata must fail before tracking")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset.load_camera_inputs",
        forbidden,
    )
    output = tmp_path / "output"
    with pytest.raises(CameraDatasetError):
        prepare_camera_input_pair(
            tracking,
            sidecar_path,
            output,
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("mode", ["synthetic", "authorized"])
def test_shared_component_validator_precedes_tracking_for_both_pair_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if mode == "synthetic":
        tracking, sidecar_path = _write_test_pair(tmp_path)
        receipt_path = collection_path = None
    else:
        receipt_path, collection_path, tracking, sidecar_path = _write_authorized_pair(
            tmp_path
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["tracker"] = {**sidecar["tracker"], "version": "token=secret"}
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("tracking")
        raise AssertionError("component validation must precede tracking")

    monkeypatch.setattr(camera_dataset, "load_camera_inputs", forbidden)
    output = tmp_path / f"{mode}-component-output"
    with pytest.raises(CameraDatasetError):
        if mode == "synthetic":
            prepare_camera_input_pair(
                tracking,
                sidecar_path,
                output,
                camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
            )
        else:
            prepare_authorized_camera_session(
                project_root=ROOT,
                receipt_path=receipt_path,
                collection_path=collection_path,
                tracking_jsonl_path=tracking,
                media_sidecar_path=sidecar_path,
                output_dir=output,
                collection_config_path=CONFIG,
                camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
            )
    assert calls == []
    assert not output.exists()


def test_owner_markdown_template_json_is_formally_rejected() -> None:
    template_root = ROOT / "reports/mental_health/wandering_camera_c01_prep_v1/templates"
    receipt_text = (template_root / "authorization_receipt.template.md").read_text(
        encoding="utf-8"
    )
    collection_text = (template_root / "collection.template.md").read_text(encoding="utf-8")
    receipt = json.loads(re.search(r"```json\s*(.*?)\s*```", receipt_text, re.S).group(1))
    collection = json.loads(
        re.search(r"```json\s*(.*?)\s*```", collection_text, re.S).group(1)
    )
    with pytest.raises(CameraDatasetError):
        validate_authorization_receipt(receipt, _config(), operation="prepare_session")
    with pytest.raises(CameraDatasetError):
        validate_collection_manifest(collection, _config())


def _video_provenance(**updates: object) -> camera_dataset._VideoPreparationProvenance:
    values = {
        "controller": "receipt_first_shared_yolov8_bytetrack_c1_preparation",
        "controller_schema_version": "wandering-camera-video-controller-v1",
        "source_video_id": "video-0001",
        "source_sha256": "1" * 64,
        "detector_backend": "ultralytics_yolo",
        "detector_model": "yolov8n.pt",
        "detector_version": "test-version",
        "tracker_backend": "bytetrack",
        "tracker_config": "bytetrack.yaml",
        "tracker_version": "test-version",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "max_frames": None,
        "person_class_id": 0,
        "person_id_prefix": "anonymous_track",
        "scene_region": "camera_development",
        "raw_observation_count": 1,
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 25.0,
        "duration_sec": 5.0,
        "frame_count": 125,
    }
    values.update(updates)
    return camera_dataset._VideoPreparationProvenance(**values)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"controller": "other_controller"}, "controller mismatch"),
        ({"controller_schema_version": "other-v1"}, "controller schema mismatch"),
        ({"source_video_id": "video-other"}, "source mismatch"),
        ({"source_sha256": "2" * 64}, "source SHA-256 mismatch"),
        ({"detector_model": "other.pt"}, "detector mismatch"),
        ({"tracker_config": "other.yaml"}, "tracker mismatch"),
        ({"raw_observation_count": 2}, "observation count mismatch"),
    ],
)
def test_private_video_provenance_rejects_controller_source_component_and_count_mismatch(
    tmp_path: Path, updates: dict[str, object], match: str
) -> None:
    receipt_path, collection_path, tracking, sidecar_path = _write_authorized_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["detector"] = {
        "backend": "ultralytics_yolo",
        "model": "yolov8n.pt",
        "version": "test-version",
    }
    sidecar["tracker"] = {
        "backend": "bytetrack",
        "config": "bytetrack.yaml",
        "version": "test-version",
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    output = tmp_path / "private-output"
    with pytest.raises(CameraDatasetError, match=match):
        camera_dataset._prepare_authorized_camera_video_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar_path,
            output_dir=output,
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
            video_provenance=replace(_video_provenance(), **updates),
        )
    assert not output.exists()
    assert list(tmp_path.glob(".private-output.tmp-*")) == []


def test_private_video_provenance_carrier_has_exact_fields_and_rejects_extras() -> None:
    expected = {
        "controller",
        "controller_schema_version",
        "source_video_id",
        "source_sha256",
        "detector_backend",
        "detector_model",
        "detector_version",
        "tracker_backend",
        "tracker_config",
        "tracker_version",
        "confidence_threshold",
        "iou_threshold",
        "max_frames",
        "person_class_id",
        "person_id_prefix",
        "scene_region",
        "raw_observation_count",
        "video_width",
        "video_height",
        "nominal_fps",
        "duration_sec",
        "frame_count",
    }
    assert set(inspect.signature(camera_dataset._VideoPreparationProvenance).parameters) == expected
    values = {name: getattr(_video_provenance(), name) for name in expected}
    with pytest.raises(TypeError):
        camera_dataset._VideoPreparationProvenance(**values, arbitrary_extra="forbidden")


def test_private_video_summary_flag_tamper_leaves_no_final_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path, tracking, sidecar_path = _write_authorized_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["detector"] = {
        "backend": "ultralytics_yolo",
        "model": "yolov8n.pt",
        "version": "test-version",
    }
    sidecar["tracker"] = {
        "backend": "bytetrack",
        "config": "bytetrack.yaml",
        "version": "test-version",
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    original = camera_dataset._video_summary

    def tampered(provenance):
        value = original(provenance)
        value["tracker_run"] = False
        return value

    monkeypatch.setattr(camera_dataset, "_video_summary", tampered)
    output = tmp_path / "video-flag-tamper-output"
    with pytest.raises(CameraDatasetError, match="work flags are inconsistent"):
        camera_dataset._prepare_authorized_camera_video_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar_path,
            output_dir=output,
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
            video_provenance=_video_provenance(),
        )
    assert not output.exists()
    assert list(tmp_path.glob(".video-flag-tamper-output.tmp-*")) == []


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("media_ref", "/absolute/video.mp4"),
        ("media_ref", "server\\share\\video.mp4"),
        ("media_ref", "user:password@host/video.mp4"),
        ("media_ref", "bad\x1fref.mp4"),
        ("timezone", "https://timezone.invalid/UTC"),
        ("timezone", "user:password@UTC"),
        ("timezone", "bad\x1fzone"),
        ("timezone", "C:\\UTC"),
    ],
)
def test_additional_portable_metadata_attacks_fail_before_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: str,
) -> None:
    tracking, sidecar_path = _write_test_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = bad
    if field == "timezone":
        sidecar["capture_started_at"] = "2026-08-13T10:00:00+08:00"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("tracking")
        raise AssertionError("bad portable metadata must fail before tracking")

    monkeypatch.setattr(camera_dataset, "load_camera_inputs", forbidden)
    with pytest.raises(CameraDatasetError):
        prepare_camera_input_pair(
            tracking,
            sidecar_path,
            tmp_path / "output",
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("media_ref", "C:drive-relative.mp4"),
        ("timezone", "Not/A_Real_Zone"),
    ],
)
def test_authorized_pair_bad_metadata_fails_before_protected_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: str,
) -> None:
    receipt_path, collection_path, tracking, sidecar_path = _write_authorized_pair(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = bad
    if field == "timezone":
        sidecar["capture_started_at"] = "2026-08-13T10:00:00+08:00"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("tracking")
        raise AssertionError("bad authorized metadata must fail before protected tracking")

    monkeypatch.setattr(camera_dataset, "load_camera_inputs", forbidden)
    output = tmp_path / "authorized-output"
    with pytest.raises(CameraDatasetError):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar_path,
            output_dir=output,
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not output.exists()


def test_authorized_pair_in_project_tracking_path_fails_before_input_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, collection_path, _, sidecar_path = _write_authorized_pair(tmp_path)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("read")
        raise AssertionError("in-project tracking must fail before protected reads")

    monkeypatch.setattr(camera_dataset, "load_camera_inputs", forbidden)
    with pytest.raises(CameraDatasetError, match="outside project root"):
        prepare_authorized_camera_session(
            project_root=ROOT,
            receipt_path=receipt_path,
            collection_path=collection_path,
            tracking_jsonl_path=ROOT / "README.md",
            media_sidecar_path=sidecar_path,
            output_dir=tmp_path / "output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not (tmp_path / "output").exists()
