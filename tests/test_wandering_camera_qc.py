from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import load_camera_inputs
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CAMERA_WINDOW_SCHEMA_VERSION,
    CameraQCError,
    compensate_bbox_height,
    rolling_median_valid,
    run_camera_qc,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_v1.yaml"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write_pair(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    camera_motion_state: str = "not_checked",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    tracking = directory / "tracking.jsonl"
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
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
        "duration_sec": 120.0,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {"backend": "ultralytics_yolo", "model": "yolov8n.pt", "version": "synthetic"},
        "tracker": {"backend": "bytetrack", "config": "bytetrack.yaml", "version": "synthetic"},
        "fixed_camera_assumed": True,
        "camera_motion_state": camera_motion_state,
        "authorization_status": "synthetic_fixture",
        "deidentification_status": "synthetic",
    }
    sidecar_path = directory / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return tracking, sidecar_path


def _row(
    bucket: int,
    *,
    x: float | None = None,
    y2: float = 320.0,
    height: float = 100.0,
    track_id: int = 1,
    confidence: float = 0.9,
) -> dict[str, object]:
    center_x = 120.0 + bucket * 1.0 if x is None else x
    return {
        "frame_id": bucket * 10 + track_id,
        "track_id": track_id,
        "bbox": [center_x - 20.0, y2 - height, center_x + 20.0, y2],
        "track_confidence": confidence,
        "timestamp_sec": bucket * 0.5 + 0.1,
    }


def _run(tmp_path: Path, rows: list[dict[str, object]], *, state: str = "not_checked"):
    tracking, sidecar = _write_pair(tmp_path, rows, camera_motion_state=state)
    inputs = load_camera_inputs(tracking, sidecar, _config())
    return run_camera_qc(inputs, _config())


def test_declares_frozen_window_schema() -> None:
    assert CAMERA_WINDOW_SCHEMA_VERSION == "wandering-camera-window-v1"


def test_rolling_median_and_height_compensation_match_hand_calculation() -> None:
    heights = np.asarray([1.0, 1.0, 9.0, 1.0, 1.0], dtype=np.float64)
    smoothed = rolling_median_valid(heights, window=5)
    np.testing.assert_allclose(smoothed, np.ones(5))

    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    raw_heights = np.asarray([2.0, 1.0, 0.25], dtype=np.float64)
    quality = np.ones(3, dtype=np.float64)
    result = compensate_bbox_height(
        points,
        raw_heights,
        quality,
        rolling_window=1,
        gain_clip=(0.5, 2.0),
    )
    np.testing.assert_allclose(result.smoothed_heights, raw_heights)
    np.testing.assert_allclose(result.gains, [0.5, 1.0, 2.0])
    np.testing.assert_allclose(result.corrected_points, [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])


def test_global_2hz_grid_40_second_window_and_20_second_stride_are_exact(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row(bucket) for bucket in range(120)])
    assert len(result.tracklet_records) == 1
    assert len(result.window_records) == 2
    assert [row["window_start_sec"] for row in result.window_records] == [0.0, 20.0]
    assert [row["window_end_sec"] for row in result.window_records] == [40.0, 60.0]
    assert all(row["observed_bucket_count"] == 80 for row in result.window_records)
    assert all(row["interpolated_bucket_count"] == 0 for row in result.window_records)
    assert all(row["window_status"] == "ready" for row in result.window_records)
    assert all(len(row["bbox_bottom_points"]) == 80 for row in result.window_records)
    assert len({row["window_id"] for row in result.window_records}) == 2


def test_three_missing_buckets_are_interpolated_but_four_split_the_tracklet(tmp_path: Path) -> None:
    short_gap_rows = [_row(bucket) for bucket in range(80) if bucket not in {20, 21, 22}]
    short = _run(tmp_path / "short", short_gap_rows)
    assert len(short.tracklet_records) == 1
    assert len(short.window_records) == 1
    assert short.window_records[0]["window_status"] == "ready"
    assert short.window_records[0]["longest_gap_seconds"] == pytest.approx(1.5)
    assert short.window_records[0]["interpolated_bucket_count"] == 3
    assert short.window_records[0]["observed_mask"][20:23] == [0, 0, 0]
    assert short.window_records[0]["interpolated_mask"][20:23] == [1, 1, 1]
    assert short.window_records[0]["point_quality"][20:23] == [0.5, 0.5, 0.5]

    long_gap_rows = [_row(bucket) for bucket in range(120) if bucket not in {40, 41, 42, 43}]
    long = _run(tmp_path / "long", long_gap_rows)
    assert len(long.tracklet_records) == 2
    assert any("long_internal_gap" in row["quality_flags"] for row in long.tracklet_records)
    assert all("long_internal_gap" in row["reason_codes"] for row in long.window_records)
    assert all(row["window_status"] == "unavailable" for row in long.window_records)


def test_window_edges_are_not_extrapolated(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row(bucket) for bucket in range(1, 81)])
    assert len(result.window_records) == 1
    window = result.window_records[0]
    assert window["window_status"] == "unavailable"
    assert "unobserved_window_edge" in window["reason_codes"]
    assert window["corrected_points"] is None
    assert window["model_features"] is None


def test_low_confidence_edges_and_all_low_confidence_keep_unavailable_candidate_window(
    tmp_path: Path,
) -> None:
    low_tail_rows = [
        _row(bucket, confidence=0.1 if bucket == 79 else 0.9)
        for bucket in range(80)
    ]
    low_tail = _run(tmp_path / "low-tail", low_tail_rows)
    assert len(low_tail.window_records) == 1
    tail_window = low_tail.window_records[0]
    assert tail_window["window_start_sec"] == 0.0
    assert tail_window["window_end_sec"] == 40.0
    assert tail_window["window_status"] == "unavailable"
    assert tail_window["observed_bucket_count"] == 79
    assert "unobserved_window_edge" in tail_window["reason_codes"]

    all_low = _run(
        tmp_path / "all-low",
        [_row(bucket, confidence=0.1) for bucket in range(80)],
    )
    assert len(all_low.window_records) == 1
    all_low_window = all_low.window_records[0]
    assert all_low_window["window_start_sec"] == 0.0
    assert all_low_window["window_end_sec"] == 40.0
    assert all_low_window["window_status"] == "unavailable"
    assert all_low_window["observed_bucket_count"] == 0
    assert "unobserved_window_edge" in all_low_window["reason_codes"]


def test_hard_jump_height_discontinuity_and_moved_camera_are_rejected(tmp_path: Path) -> None:
    jump_rows = [_row(bucket) for bucket in range(120)]
    for bucket in range(60, 120):
        jump_rows[bucket] = _row(bucket, x=520.0 + (bucket - 60))
    jump = _run(tmp_path / "jump", jump_rows)
    assert len(jump.tracklet_records) == 2
    assert any("suspected_id_switch" in row["quality_flags"] for row in jump.tracklet_records)
    assert all(row["window_status"] == "unavailable" for row in jump.window_records)

    height_rows = [_row(bucket) for bucket in range(120)]
    for bucket in range(60, 120):
        height_rows[bucket] = _row(bucket, x=260.0 + (bucket - 60), height=40.0)
    height = _run(tmp_path / "height", height_rows)
    assert any("height_position_discontinuity" in row["quality_flags"] for row in height.tracklet_records)
    assert all(row["window_status"] == "unavailable" for row in height.window_records)

    moved = _run(tmp_path / "moved", [_row(bucket) for bucket in range(80)], state="moved")
    assert moved.window_records[0]["window_status"] == "unavailable"
    assert moved.window_records[0]["reason_codes"] == ["camera_moved"]
    assert moved.window_records[0]["corrected_points"] is None


def test_not_checked_camera_is_flagged_without_claiming_motion_detection(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row(bucket) for bucket in range(80)])
    assert result.window_records[0]["window_status"] == "ready"
    assert "camera_motion_not_verified" in result.window_records[0]["quality_flags"]
    assert "camera_moved" not in result.window_records[0]["reason_codes"]


def test_static_jitter_is_unavailable_before_shape_normalization(tmp_path: Path) -> None:
    rows = [_row(bucket, x=120.0 + (0.02 if bucket % 2 else 0.0)) for bucket in range(80)]
    result = _run(tmp_path, rows)
    assert result.window_records[0]["window_status"] == "unavailable"
    assert result.window_records[0]["reason_codes"] == ["insufficient_motion"]
    assert not result.ready_inputs
    assert result.window_records[0]["shape_normalized_points"] is None


def test_window_and_tracklet_schemas_are_exact_and_ids_are_deterministic(tmp_path: Path) -> None:
    rows = [_row(bucket) for bucket in range(80)]
    first = _run(tmp_path / "a", rows)
    second = _run(tmp_path / "b", rows)
    assert first.tracklet_records == second.tracklet_records
    assert first.window_records == second.window_records
    assert set(first.tracklet_records[0]) == {
        "schema_version", "tracklet_id", "segment_index", "source_video_id", "source_group_id",
        "device_id", "setup_id", "stream_epoch", "track_id", "frame_width", "frame_height",
        "source_fps", "frame_indices", "point_times_sec", "bbox_xyxy_norm", "bbox_bottom_points",
        "bbox_heights", "detection_confidence", "observed_mask", "interpolated_mask", "quality_flags",
        "media_ref", "source_sha256", "tracking_jsonl_sha256",
    }
    assert set(first.window_records[0]) == {
        "schema_version", "window_id", "parent_tracklet_id", "segment_index", "source_group_id",
        "source_video_id", "device_id", "setup_id", "stream_epoch", "track_id", "window_start_sec",
        "window_end_sec", "window_status", "reason_codes", "quality_flags", "raw_detection_count",
        "observed_bucket_count", "interpolated_bucket_count", "observed_ratio", "longest_gap_seconds",
        "mean_detection_confidence", "bbox_bottom_points", "bbox_heights", "smoothed_bbox_heights",
        "corrected_points", "observed_mask", "interpolated_mask", "point_quality",
        "shape_normalized_points", "point_mask", "raw_features", "model_features", "topology",
        "preprocessing_config_sha256", "feature_stats_sha256", "model_training_split_sha256",
    }


def test_invalid_qc_arguments_fail_closed() -> None:
    with pytest.raises(CameraQCError):
        rolling_median_valid(np.ones(3), window=4)
    with pytest.raises(CameraQCError):
        compensate_bbox_height(
            np.zeros((3, 2)),
            np.asarray([1.0, 0.0, 1.0]),
            np.ones(3),
            rolling_window=1,
            gain_clip=(0.5, 2.0),
        )


def test_reason_distribution_is_auditable(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row(bucket, x=120.0) for bucket in range(80)])
    reasons = Counter(reason for row in result.window_records for reason in row["reason_codes"])
    assert reasons == Counter({"insufficient_motion": 1})
