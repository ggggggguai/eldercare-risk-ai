from __future__ import annotations

import json
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraObservation,
)
from elderly_monitoring.modules.mental_health.wandering.camera_geometry_head import (
    CameraGeometryHeadError,
    load_camera_geometry_head,
    predict_home_camera_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "reports/mental_health/wandering_camera_home_geometry_head_v1/model.json"
)


def _adapter() -> CameraAdapterInput:
    xs = (
        [0.20 + 0.08 * index for index in range(6)]
        + [0.60, 0.60]
        + [0.60 - 0.08 * index for index in range(6)]
        + [0.20, 0.20]
        + [0.20 + 0.08 * index for index in range(6)]
    )
    scope = ("group", "video", "device", "setup", "epoch", 1)
    observations = tuple(
        CameraObservation(
            scope_key=scope,
            source_group_id=scope[0],
            source_video_id=scope[1],
            device_id=scope[2],
            setup_id=scope[3],
            stream_epoch=scope[4],
            track_id=1,
            frame_id=index * 10,
            timestamp_sec=index * 0.5 + 0.1,
            bbox_xyxy_pixel=(0.0, 0.0, 100.0, 200.0),
            bbox_xyxy_norm=(0.0, 0.0, 0.2, 0.5),
            bbox_bottom_point=(x, 0.75),
            bbox_height=0.25,
            track_confidence=0.9,
        )
        for index, x in enumerate(xs)
    )
    return CameraAdapterInput(
        media_sidecar={"source_video_id": "video"},
        observations=observations,
        normalized_rows=tuple(),
        source_tracking_sha256="1" * 64,
        normalized_tracking_sha256="2" * 64,
    )


def test_geometry_artifact_gates_and_micro_bouts_are_available() -> None:
    artifact = load_camera_geometry_head(MODEL_PATH)

    prediction = predict_home_camera_geometry(
        _adapter(),
        track_id=1,
        start_sec=0.0,
        end_sec=11.0,
        artifact=artifact,
    )

    cross_validation = artifact["cross_validation"]["best"]
    assert cross_validation["recall_by_class"]["direct"] >= 0.90
    assert cross_validation["recall_by_class"]["pacing"] >= 0.60
    assert prediction["motion_bout_count"] >= 2
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert prediction["behavior_truth_consumed_at_inference"] is False


def test_geometry_artifact_rejects_contract_drift(tmp_path: Path) -> None:
    value = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    value["minimum_track_confidence"] = 0.69
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CameraGeometryHeadError, match="contract drifted"):
        load_camera_geometry_head(drifted)
