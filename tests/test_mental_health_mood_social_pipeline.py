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


def _sleep() -> dict[str, object]:
    return {
        "in_bed_minutes": 480.0,
        "sleep_minutes": 420.0,
        "sleep_efficiency": 0.875,
        "sleep_onset_minute_of_day": 1320,
        "wake_time_minute_of_day": 420,
        "awake_minutes": 60.0,
        "awakening_count": 2,
        "night_exit_count": 1,
        "night_exit_minutes": 10.0,
    }


def _social(*, connected: bool) -> dict[str, object]:
    if not connected:
        return {
            "call_log_observed": True,
            "incoming_call_opportunities": 0,
            "answered_call_count": 0,
            "missed_call_count": 0,
            "outgoing_call_count": 0,
            "connected_duration_minutes": 0.0,
            "active_contact_count": 0,
        }
    return {
        "call_log_observed": True,
        "incoming_call_opportunities": 2,
        "answered_call_count": 2,
        "missed_call_count": 0,
        "outgoing_call_count": 1,
        "connected_duration_minutes": 30.0,
        "active_contact_count": 2,
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


def test_camera_sleep_s10_evidence_uses_engineering_proxy_scope(
    pipeline: MoodSocialPipeline,
) -> None:
    payload = _payload()
    target = datetime.fromisoformat(str(payload["target_date"])).date()
    payload["request_id"] = "req_contract_02_cst"
    payload["available_sources"] = ["camera", "sleep_device", "s10"]
    payload["current_daily_features"].update(
        {
            "activity": _activity(),
            "sleep": _sleep(),
            "physiology": {"heart_rate_mean_bpm": 64.0},
            "social": _social(connected=False),
        }
    )
    payload["history_daily_features"] = [
        {
            "date": (target - timedelta(days=days_before)).isoformat(),
            "activity": None,
            "sleep": None,
            "physiology": None,
            "social": _social(connected=True),
        }
        for days_before in (3, 2, 1)
    ]
    request = MoodSocialInferRequest.model_validate(payload)

    response = pipeline.predict(request)

    assert response.available is True
    assert response.evidence_scope == "engineering_proxy"
    assert response.used_sources == ["camera", "sleep_device", "s10"]
    assert "personal_change_social" in response.covered_domains
    assert response.domain_scores is not None
    assert response.domain_scores.personal_change.social is not None


def test_configured_s10_without_effective_evidence_is_not_reported_as_used(
    pipeline: MoodSocialPipeline,
) -> None:
    payload = _payload()
    payload["available_sources"] = ["camera", "s10"]
    payload["current_daily_features"]["activity"] = _activity()
    payload["current_daily_features"]["social"] = {"call_log_observed": False}

    response = pipeline.predict(MoodSocialInferRequest.model_validate(payload))

    assert response.available is True
    assert response.used_sources == ["camera"]
    assert response.evidence_scope == "proxy_label_supported"
    assert "personal_change_social" not in response.covered_domains


def test_attention_trend_includes_current_and_filters_other_model_versions(
    pipeline: MoodSocialPipeline,
) -> None:
    payload = _payload()
    payload["current_daily_features"]["activity"] = _activity()
    first = pipeline.predict(MoodSocialInferRequest.model_validate(payload))
    assert first.attention_index is not None
    target = datetime.fromisoformat(str(payload["target_date"])).date()

    old_only = _payload()
    old_only["current_daily_features"]["activity"] = _activity()
    old_only["history_attention_indices"] = [
        {
            "date": (target - timedelta(days=days_before)).isoformat(),
            "attention_index": value,
            "model_version": "mood-fusion-v3.3.2",
        }
        for days_before, value in ((2, 0.0), (1, 1.0))
    ]
    old_response = pipeline.predict(MoodSocialInferRequest.model_validate(old_only))
    assert old_response.trend == "unknown"

    current = float(first.attention_index)
    if current >= 0.2:
        history_values = (current - 0.2, current - 0.1)
        expected_trend = "rising"
    else:
        history_values = (current + 0.2, current + 0.1)
        expected_trend = "falling"
    three_points = _payload()
    three_points["current_daily_features"]["activity"] = _activity()
    three_points["history_attention_indices"] = [
        {
            "date": (target - timedelta(days=days_before)).isoformat(),
            "attention_index": value,
            "model_version": "mood-fusion-v3.3.3",
        }
        for days_before, value in zip((2, 1), history_values, strict=True)
    ]
    trend_response = pipeline.predict(
        MoodSocialInferRequest.model_validate(three_points)
    )
    assert trend_response.trend == expected_trend


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
