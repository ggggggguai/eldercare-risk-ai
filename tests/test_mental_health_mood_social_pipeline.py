"""Focused integration tests for API-001 MoodSocialPipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    map_mood_social_features,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    DEFAULT_PACKAGE_DIRECTORY,
    MoodSocialPipeline,
    MoodSocialPipelineUnavailableError,
    _build_expert_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
)


def _target_date() -> str:
    return (
        datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
    ).isoformat()


def _payload() -> dict[str, object]:
    target = _target_date()
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "req_api001",
        "person_id": "elder-001",
        "target_date": target,
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera"],
        "profile": {},
        "current_daily_features": {
            "date": target,
            "activity": None,
            "sleep": None,
            "physiology": None,
            "social": None,
        },
        "history_daily_features": [],
        "history_attention_indices": [],
    }


def _activity() -> dict[str, object]:
    coverage = [0.0] * 24
    intensity: list[float | None] = [None] * 24
    coverage[6] = 60.0
    intensity[6] = 0.5
    return {
        "daytime_active_minutes": 10.0,
        "weighted_daytime_activity": 30.0,
        "valid_daytime_detection_minutes": 60.0,
        "low_activity_minutes": 40.0,
        "sedentary_bout_total_minutes": 30.0,
        "longest_sedentary_bout_minutes": 30.0,
        "activity_peak_minute_of_day": 390,
        "hourly_activity_intensity": intensity,
        "hourly_valid_detection_minutes": coverage,
        "camera_gait_metrics": [],
    }


@pytest.fixture(scope="module")
def pipeline() -> MoodSocialPipeline:
    return MoodSocialPipeline.from_package(DEFAULT_PACKAGE_DIRECTORY)


def test_package_loader_excludes_offline_model006(
    pipeline: MoodSocialPipeline,
) -> None:
    assert set(pipeline.package.online_models) == {
        "activity_expert",
        "sleep_expert",
        "activity_sleep_joint_expert",
        "physiology_expert",
        "social_context_expert",
        "personal_trend",
        "selected_fusion",
    }
    assert pipeline.package.offline_models == {}
    assert pipeline.package.manifest["model006_online"] is False


def test_missing_package_is_distinct_runtime_unavailability(tmp_path: Path) -> None:
    with pytest.raises(MoodSocialPipelineUnavailableError):
        MoodSocialPipeline.from_package(tmp_path / "missing-package")


def test_no_effective_evidence_returns_frozen_unavailable_shape(
    pipeline: MoodSocialPipeline,
) -> None:
    request = MoodSocialInferRequest.model_validate(_payload())
    response = pipeline.predict(request)
    assert response.available is False
    assert response.attention_index is None
    assert response.confidence == 0.0
    assert response.used_sources == []
    assert response.covered_domains == []
    assert "insufficient_data" in response.limitations


def test_activity_evidence_runs_expert_and_selected_fusion_deterministically(
    pipeline: MoodSocialPipeline,
) -> None:
    payload = _payload()
    payload["current_daily_features"]["activity"] = _activity()
    request = MoodSocialInferRequest.model_validate(payload)
    first = pipeline.predict(request)
    second = pipeline.predict(request)
    assert first == second
    assert first.available is True
    assert first.used_sources == ["camera"]
    assert first.covered_domains == ["activity"]
    assert first.domain_scores is not None
    assert first.domain_scores.activity is not None
    assert 0.0 <= first.attention_index <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.model_contributions == []
    assert first.diagnosis is False


def test_expert_frame_preserves_masks_and_never_contains_public_dataset_id() -> None:
    request = MoodSocialInferRequest.model_validate(_payload())
    mapped = map_mood_social_features(request)
    frame = _build_expert_frame(mapped)
    assert "dataset_id" not in frame.columns
    mask_columns = [name for name in frame if name.startswith("feature_mask.")]
    assert mask_columns
    for mask_name in mask_columns:
        feature_name = mask_name.removeprefix("feature_mask.")
        if int(frame.loc[0, mask_name]) == 0:
            assert frame.loc[0, feature_name] is None
