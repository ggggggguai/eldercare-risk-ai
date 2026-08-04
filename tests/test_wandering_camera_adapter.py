from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    BBOX_TRACKLET_SCHEMA_VERSION,
    MEDIA_SCHEMA_VERSION,
    CameraAdapterError,
    group_observations,
    load_camera_inputs,
    validate_media_sidecar,
    weighted_bucket_observations,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_v1.yaml"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _tracking_row(
    frame_id: int,
    timestamp_sec: float,
    *,
    track_id: int = 7,
    bbox: list[float] | None = None,
    confidence: float = 0.9,
) -> dict[str, object]:
    box = bbox or [10.0 + frame_id, 20.0, 30.0 + frame_id, 60.0]
    return {
        "frame_id": frame_id,
        "person_id": f"elder_{track_id:03d}",
        "track_id": track_id,
        "bbox": box,
        "scene_region": "home",
        "track_confidence": confidence,
        "center": [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
        "speed_px_per_sec": 12.0,
        "timestamp_sec": timestamp_sec,
    }


def _sidecar(tracking_sha256: str, **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": MEDIA_SCHEMA_VERSION,
        "source_video_id": "video-0001",
        "source_group_id": "session-0001",
        "device_id": "camera-0001",
        "setup_id": "fixed-setup-0001",
        "stream_epoch": "epoch-0001",
        "media_ref": "synthetic/video-0001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": tracking_sha256,
        "video_width": 100,
        "video_height": 100,
        "nominal_fps": 25.0,
        "duration_sec": 60.0,
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
    value.update(updates)
    return value


def _write_input_pair(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    sidecar_updates: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    tracking = directory / "tracking.jsonl"
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    tracking.write_bytes(payload)
    sidecar = _sidecar(hashlib.sha256(payload).hexdigest(), **(sidecar_updates or {}))
    sidecar_path = directory / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return tracking, sidecar_path


def test_declares_frozen_camera_schemas() -> None:
    assert MEDIA_SCHEMA_VERSION == "wandering-media-v1"
    assert BBOX_TRACKLET_SCHEMA_VERSION == "wandering-bbox-tracklet-v1"


def test_media_sidecar_exact_fields_enums_paths_and_numeric_contract() -> None:
    config = _config()
    valid = _sidecar("2" * 64)
    assert validate_media_sidecar(valid, config)["camera_motion_state"] == "not_checked"

    mutations: list[dict[str, object]] = []
    extra = dict(valid, unexpected=True)
    mutations.append(extra)
    missing = dict(valid)
    missing.pop("video_width")
    mutations.append(missing)
    mutations.append(dict(valid, media_ref="C:/secret/video.mp4"))
    mutations.append(dict(valid, media_ref="https://example.test/live?token=secret"))
    mutations.append(dict(valid, camera_motion_state="auto_stable"))
    mutations.append(dict(valid, coordinate_system="pixel_center"))
    mutations.append(dict(valid, nominal_fps=14.9))
    mutations.append(dict(valid, video_width=True))
    mutations.append(dict(valid, duration_sec=float("nan")))
    mutations.append(dict(valid, source_sha256="not-a-sha"))
    absolute_model = dict(valid)
    absolute_model["detector"] = {
        "backend": "ultralytics_yolo",
        "model": "C:/private/models/yolov8n.pt",
        "version": "synthetic",
    }
    mutations.append(absolute_model)
    for mutation in mutations:
        with pytest.raises(CameraAdapterError):
            validate_media_sidecar(mutation, config)


@pytest.mark.parametrize(
    "authorization_status",
    (
        "synthetic_fixture",
        "authorized_camera_engineering_smoke",
        "authorized_camera_labeled_evaluation",
    ),
)
def test_media_sidecar_accepts_only_frozen_authorization_statuses(
    authorization_status: str,
) -> None:
    valid = _sidecar("2" * 64, authorization_status=authorization_status)
    assert validate_media_sidecar(valid, _config())["authorization_status"] == authorization_status


@pytest.mark.parametrize(
    "authorization_status",
    ("denied", "authorized", "synthetic_fixtur", "authorized_camera_smoke"),
)
def test_media_sidecar_rejects_unknown_or_non_authorized_statuses(
    authorization_status: str,
) -> None:
    invalid = _sidecar("2" * 64, authorization_status=authorization_status)
    with pytest.raises(CameraAdapterError, match="authorization_status"):
        validate_media_sidecar(invalid, _config())


def test_tracking_hash_bbox_bottom_height_weighted_bucket_and_normalized_copy(tmp_path: Path) -> None:
    rows = [
        _tracking_row(1, 0.10, bbox=[10.0, 10.0, 30.0, 40.0], confidence=0.25),
        _tracking_row(2, 0.40, bbox=[30.0, 20.0, 50.0, 60.0], confidence=0.75),
    ]
    tracking, sidecar = _write_input_pair(tmp_path, rows)
    inputs = load_camera_inputs(tracking, sidecar, _config())

    assert tuple(inputs.normalized_rows[0]) == (
        "bbox",
        "frame_id",
        "timestamp_sec",
        "track_confidence",
        "track_id",
    )
    assert "person_id" not in inputs.normalized_rows[0]
    assert "center" not in inputs.normalized_rows[0]
    assert "speed_px_per_sec" not in inputs.normalized_rows[0]
    assert inputs.normalized_rows[0]["bbox"] == [10.0, 10.0, 30.0, 40.0]
    assert inputs.observations[0].bbox_bottom_point == pytest.approx((0.20, 0.40))
    assert inputs.observations[0].bbox_height == pytest.approx(0.30)

    buckets = weighted_bucket_observations(
        inputs.observations,
        minimum_track_confidence=0.25,
        bucket_seconds=0.5,
    )
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.bucket_index == 0
    assert bucket.point_time_sec == pytest.approx(0.25)
    assert bucket.bbox_bottom_point == pytest.approx((0.35, 0.55))
    assert bucket.bbox_height == pytest.approx(0.375)
    assert bucket.raw_detection_count == 2
    assert bucket.mean_detection_confidence == pytest.approx(0.625)


def test_full_scope_key_keeps_video_device_setup_epoch_and_track_isolated(tmp_path: Path) -> None:
    tracking, sidecar = _write_input_pair(tmp_path, [_tracking_row(1, 0.1)])
    observation = load_camera_inputs(tracking, sidecar, _config()).observations[0]
    variants = (
        observation,
        dataclasses.replace(observation, source_video_id="video-0002"),
        dataclasses.replace(observation, device_id="camera-0002"),
        dataclasses.replace(observation, setup_id="setup-0002"),
        dataclasses.replace(observation, stream_epoch="epoch-0002"),
        dataclasses.replace(observation, track_id=8),
    )
    grouped = group_observations(variants)
    assert len(grouped) == 6
    assert all(len(group) == 1 for group in grouped.values())


def test_tracking_rejects_bad_hash_bbox_nan_duplicates_and_non_increasing_order(tmp_path: Path) -> None:
    config = _config()
    cases = [
        [_tracking_row(1, 0.1, bbox=[10.0, 10.0, 101.0, 40.0])],
        [_tracking_row(1, 0.1, bbox=[10.0, 10.0, float("nan"), 40.0])],
        [_tracking_row(1, 0.1), _tracking_row(1, 0.2)],
        [_tracking_row(2, 0.2), _tracking_row(3, 0.1)],
        [_tracking_row(2, 0.1), _tracking_row(1, 0.2)],
        [_tracking_row(-1, 0.1)],
        [_tracking_row(1, 0.1, confidence=1.1)],
    ]
    for index, rows in enumerate(cases):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        tracking, sidecar = _write_input_pair(case_dir, rows)
        with pytest.raises(CameraAdapterError):
            load_camera_inputs(tracking, sidecar, config)

    valid_dir = tmp_path / "bad_hash"
    valid_dir.mkdir()
    tracking, sidecar = _write_input_pair(valid_dir, [_tracking_row(1, 0.1)])
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    value["tracking_jsonl_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CameraAdapterError):
        load_camera_inputs(tracking, sidecar, config)


def test_source_rows_can_be_noncanonical_but_normalized_rows_are_stably_sorted(tmp_path: Path) -> None:
    rows = [
        _tracking_row(1, 0.1, track_id=9),
        _tracking_row(1, 0.1, track_id=7),
    ]
    tracking, sidecar = _write_input_pair(tmp_path, rows)
    result = load_camera_inputs(tracking, sidecar, _config())
    assert [row["track_id"] for row in result.normalized_rows] == [7, 9]
    assert result.source_tracking_sha256 != result.normalized_tracking_sha256
