from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from elderly_monitoring.modules.mental_health.wandering import (
    camera_episode_inference as episode_inference_module,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraObservation,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_inference import (
    CAMERA_EPISODE_CONFIG_SCHEMA_VERSION,
    CameraEpisodeInferenceError,
    build_whole_clip_boundary,
    load_camera_episode_config,
    predict_camera_episode,
    prepare_camera_episode,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)


ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG_PATH = ROOT / "configs/modules/wandering_camera_v1.yaml"
EPISODE_CONFIG_PATH = ROOT / "configs/modules/wandering_camera_episode_v1.yaml"
PREPROCESSING_CONFIG_PATH = ROOT / "configs/data/wandering_preprocessing_v1.yaml"
FEATURE_STATS_PATH = ROOT / "data/processed/wandering/preprocessing/v1/feature_stats.json"


def _configs() -> tuple[dict, dict, dict, dict]:
    return (
        load_camera_config(CAMERA_CONFIG_PATH),
        load_camera_episode_config(EPISODE_CONFIG_PATH),
        load_preprocessing_config(PREPROCESSING_CONFIG_PATH),
        json.loads(FEATURE_STATS_PATH.read_text(encoding="utf-8")),
    )


def _media(duration_sec: float) -> dict[str, object]:
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": "video-episode-001",
        "source_group_id": "session-episode-001",
        "device_id": "camera-001",
        "setup_id": "fixed-setup-001",
        "stream_epoch": "epoch-001",
        "media_ref": "authorized/video-episode-001.mp4",
        "source_sha256": "1" * 64,
        "tracking_jsonl_sha256": "2" * 64,
        "video_width": 640,
        "video_height": 480,
        "nominal_fps": 20.0,
        "duration_sec": duration_sec,
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
        "authorization_status": "authorized_camera_engineering_smoke",
        "deidentification_status": "anonymous",
    }


def _observation(
    *,
    index: int,
    count: int,
    track_id: int = 1,
    static: bool = False,
    jump: bool = False,
) -> CameraObservation:
    phase = index / max(1, count - 1)
    x = 0.2 if static else 0.2 + 0.55 * phase
    if jump:
        x = 0.98
    scope = (
        "session-episode-001",
        "video-episode-001",
        "camera-001",
        "fixed-setup-001",
        "epoch-001",
        track_id,
    )
    return CameraObservation(
        scope_key=scope,
        source_group_id=scope[0],
        source_video_id=scope[1],
        device_id=scope[2],
        setup_id=scope[3],
        stream_epoch=scope[4],
        track_id=track_id,
        frame_id=index * 10 + track_id,
        timestamp_sec=index * 0.5 + 0.1,
        bbox_xyxy_pixel=(100.0, 100.0, 180.0, 360.0),
        bbox_xyxy_norm=(0.15625, 0.208333, 0.28125, 0.75),
        bbox_bottom_point=(x, 0.75),
        bbox_height=0.25,
        track_confidence=0.9,
    )


def _adapter(
    duration_sec: float,
    *,
    missing: set[int] | None = None,
    static: bool = False,
    second_track: bool = False,
    jump_index: int | None = None,
) -> CameraAdapterInput:
    count = int(round(duration_sec / 0.5))
    missing = missing or set()
    observations = [
        _observation(
            index=index,
            count=count,
            static=static,
            jump=index == jump_index,
        )
        for index in range(count)
        if index not in missing
    ]
    if second_track:
        observations.extend(
            _observation(index=index, count=count, track_id=2)
            for index in range(count)
        )
    observations.sort(key=lambda item: (item.scope_key, item.timestamp_sec, item.frame_id))
    return CameraAdapterInput(
        media_sidecar=_media(duration_sec),
        observations=tuple(observations),
        normalized_rows=tuple(),
        source_tracking_sha256="2" * 64,
        normalized_tracking_sha256="2" * 64,
    )


def _boundary(adapter: CameraAdapterInput, *, track_id: int = 1) -> dict[str, object]:
    return build_whole_clip_boundary(
        adapter.media_sidecar,
        episode_id="episode-001",
        target_track_id=track_id,
    )


class _FakeModel(torch.nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
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
        if self.fail:
            raise RuntimeError("injected episode forward failure")
        return {
            "binary_logit": torch.tensor([[-4.0]], dtype=torch.float32),
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
            "artifacts": {"model_state": {"sha256": "94c3c22d" + "0" * 56}},
        },
        manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )


@pytest.mark.parametrize("duration_sec", [5.0, 10.0, 20.0, 45.0])
def test_variable_duration_episode_prepares_exact_finite_model_input(
    duration_sec: float,
) -> None:
    camera, episode, preprocessing, stats = _configs()
    adapter = _adapter(duration_sec)

    prepared = prepare_camera_episode(
        adapter,
        _boundary(adapter),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )

    assert prepared["qc_status"] == "ready"
    features = np.asarray(prepared["model_features"])
    points = np.asarray(prepared["shape_normalized_points"])
    assert features.shape == (80, 14)
    assert points.shape == (80, 2)
    assert np.isfinite(features).all()
    assert np.isfinite(points).all()
    assert prepared["duration_sec"] == duration_sec
    assert prepared["source_bucket_count"] != 80 or duration_sec == 45.0


def test_short_gap_interpolates_but_long_gap_and_static_episode_are_unavailable() -> None:
    camera, episode, preprocessing, stats = _configs()
    short_gap = _adapter(10.0, missing={7})
    prepared = prepare_camera_episode(
        short_gap,
        _boundary(short_gap),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    assert prepared["qc_status"] == "ready"
    assert prepared["interpolated_bucket_count"] == 1
    assert prepared["interpolated_coverage_ratio"] > 0.0
    assert np.min(np.asarray(prepared["raw_features"])[:, 13]) == 0.5

    for adapter, expected_reason in (
        (_adapter(10.0, missing={6, 7, 8, 9}), "track_break_within_episode"),
        (_adapter(10.0, static=True), "insufficient_motion"),
    ):
        result = prepare_camera_episode(
            adapter,
            _boundary(adapter),
            camera_config=camera,
            episode_config=episode,
            preprocessing_config=preprocessing,
            feature_stats=stats,
        )
        assert result["qc_status"] == "unavailable"
        assert expected_reason in result["qc_reason_codes"]
        assert result["model_features"] is None


def test_target_track_is_explicit_and_track_break_or_switch_is_never_spliced() -> None:
    camera, episode, preprocessing, stats = _configs()
    two_tracks = _adapter(10.0, second_track=True)
    selected = prepare_camera_episode(
        two_tracks,
        _boundary(two_tracks, track_id=1),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    assert selected["track_id"] == 1
    assert selected["source_observation_count"] == 20

    missing_target = prepare_camera_episode(
        two_tracks,
        _boundary(two_tracks, track_id=99),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    assert missing_target["qc_status"] == "unavailable"
    assert missing_target["qc_reason_codes"] == ["target_track_not_found"]

    switched = _adapter(10.0, jump_index=10)
    switched_result = prepare_camera_episode(
        switched,
        _boundary(switched),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    assert switched_result["qc_status"] == "unavailable"
    assert "track_break_within_episode" in switched_result["qc_reason_codes"]


def test_boundary_view_rejects_truth_fields_and_preprocessing_never_sees_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera, episode, preprocessing, stats = _configs()
    adapter = _adapter(5.0)
    contaminated = {**_boundary(adapter), "observable_pattern": "direct"}
    with pytest.raises(CameraEpisodeInferenceError, match="boundary fields"):
        prepare_camera_episode(
            adapter,
            contaminated,
            camera_config=camera,
            episode_config=episode,
            preprocessing_config=preprocessing,
            feature_stats=stats,
        )

    observed_records: list[dict[str, object]] = []
    real_prepare = episode_inference_module.prepare_camera_window

    def capture(camera_input, *args, **kwargs):
        observed_records.append(dict(camera_input.window_record))
        return real_prepare(camera_input, *args, **kwargs)

    monkeypatch.setattr(episode_inference_module, "prepare_camera_window", capture)
    prepare_camera_episode(
        adapter,
        _boundary(adapter),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    forbidden = {"observable_pattern", "purpose_context", "evaluation_role"}
    assert observed_records
    assert not forbidden.intersection(observed_records[0])


def test_ready_unavailable_boundary_uncertain_and_inference_error_are_distinct() -> None:
    camera, episode, preprocessing, stats = _configs()
    adapter = _adapter(5.0)
    prepared = prepare_camera_episode(
        adapter,
        _boundary(adapter),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    model = _FakeModel()
    ready = predict_camera_episode(
        prepared,
        _runtime(model),
        validation_scope="authorized_camera_engineering_smoke",
        evidence_scope="authorized_development_smoke",
    )
    assert ready["prediction_status"] == "ready"
    assert ready["predicted_pattern"] == "direct"
    assert ready["binary"]["class_order"] == [
        "direct_or_non_wandering",
        "wandering_like",
    ]
    assert ready["four_class"]["class_order"] == [
        "direct",
        "pacing",
        "lapping",
        "random",
    ]
    assert ready["binary_decision_threshold"] == 0.5
    assert ready["probability_calibrated"] is False
    assert model.calls == 1

    failing_model = _FakeModel(fail=True)
    failed = predict_camera_episode(
        prepared,
        _runtime(failing_model),
        validation_scope="authorized_camera_engineering_smoke",
        evidence_scope="authorized_development_smoke",
    )
    assert failed["prediction_status"] == "inference_error"
    assert failed["predicted_pattern"] is None

    static = _adapter(5.0, static=True)
    unavailable_prepared = prepare_camera_episode(
        static,
        _boundary(static),
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    skipped_model = _FakeModel()
    unavailable = predict_camera_episode(
        unavailable_prepared,
        _runtime(skipped_model),
        validation_scope="authorized_camera_engineering_smoke",
        evidence_scope="authorized_development_smoke",
    )
    assert unavailable["prediction_status"] == "unavailable"
    assert skipped_model.calls == 0

    uncertain_boundary = {
        **_boundary(adapter),
        "boundary_status": "boundary_uncertain",
        "boundary_reason_codes": ["operator_boundary_uncertain"],
    }
    uncertain_prepared = prepare_camera_episode(
        adapter,
        uncertain_boundary,
        camera_config=camera,
        episode_config=episode,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    uncertain_model = _FakeModel()
    uncertain = predict_camera_episode(
        uncertain_prepared,
        _runtime(uncertain_model),
        validation_scope="authorized_camera_engineering_smoke",
        evidence_scope="authorized_development_smoke",
    )
    assert uncertain["prediction_status"] == "boundary_uncertain"
    assert uncertain_model.calls == 0


def test_episode_config_does_not_change_legacy_window_or_frozen_candidate() -> None:
    episode = load_camera_episode_config(EPISODE_CONFIG_PATH)
    legacy = yaml.safe_load(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
    assert CAMERA_EPISODE_CONFIG_SCHEMA_VERSION == "wandering-camera-episode-config-v1"
    assert episode["model_input"] == {
        "target_points": 80,
        "input_channels": 14,
        "temporal_features_enabled": False,
    }
    assert episode["candidate"]["manifest_sha256"] == EXPECTED_CANDIDATE_MANIFEST_SHA256
    assert episode["inference"]["binary_decision_threshold"] == 0.5
    assert legacy["sampling"]["window_seconds"] == 40.0
    assert legacy["sampling"]["stride_seconds"] == 20.0

