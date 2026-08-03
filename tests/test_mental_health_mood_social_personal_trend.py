from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_DAILY_FEATURE_SPECS,
    SLEEP_DAILY_FEATURE_SPECS,
    CameraGaitDay,
)
from elderly_monitoring.modules.mental_health.mood_social.personal_trend import (
    BRANCH_ORDER,
    COMPONENT_ORDER,
    C_VALUES,
    MODEL006_MODEL_SHA256,
    TrendObservation,
    build_personal_trend_training_frame,
    compute_trend_components,
    load_personal_trend_config,
    load_personal_trend_inputs,
    personal_trend_sample_weights,
    _walking_speed_components,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_personal_trend_v3_3_3.yaml"
ACTIVITY_SPEC = (ACTIVITY_DAILY_FEATURE_SPECS[0],)
SLEEP_ONSET_SPEC = (SLEEP_DAILY_FEATURE_SPECS[3],)


def _observations(values: list[float | None]) -> tuple[TrendObservation, ...]:
    first = date(2026, 7, 1)
    return tuple(
        TrendObservation(
            position=float(index),
            calendar_date=first + timedelta(days=index),
            values={"observed_activity_intensity": value},
        )
        for index, value in enumerate(values)
    )


def test_config_freezes_three_branches_components_and_search() -> None:
    config = load_personal_trend_config(CONFIG, repository_root=ROOT)
    assert tuple(config.payload["branches"]) == BRANCH_ORDER
    assert tuple(config.payload["components"]["order"]) == COMPONENT_ORDER
    assert tuple(config.payload["logistic"]["c_values"]) == C_VALUES
    assert config.payload["branches"]["social"]["direct_s10_phq9_validation"] is False
    assert not any(config.payload["production_boundary"].values())
    assert (
        config.payload["upstream"]["offline_auxiliary"]["model_sha256"]
        == MODEL006_MODEL_SHA256
    )


def test_three_prior_days_are_required_for_mask() -> None:
    result = compute_trend_components(_observations([0.5, 0.5, 0.1]), ACTIVITY_SPEC)
    assert result.personal_change_mask == 0
    assert result.reliability == 0.0


def test_exactly_three_history_days_use_frozen_reliability() -> None:
    result = compute_trend_components(
        _observations([0.5, 0.5, 0.5, 0.1]),
        ACTIVITY_SPEC,
    )
    assert result.personal_change_mask == 1
    assert result.valid_history_days == 3
    assert result.values[-1] == pytest.approx(3 / 7)
    assert result.reliability == pytest.approx((3 / 7) * 0.5)


def test_abnormal_history_day_is_excluded_for_entire_domain() -> None:
    result = compute_trend_components(
        _observations([0.5, 0.5, 0.5, 0.0, 0.5, 0.5]),
        ACTIVITY_SPEC,
    )
    assert 3.0 in result.details["excluded_history_positions"]
    assert 3.0 not in result.accepted_history_positions


def test_missing_calendar_day_breaks_persistence() -> None:
    result = compute_trend_components(
        _observations([0.5, 0.5, 0.5, 0.1, None, 0.1]),
        ACTIVITY_SPEC,
    )
    assert result.values[4] == pytest.approx(1 / 7)


def test_circular_sleep_midnight_wrap_is_small_and_noon_is_large() -> None:
    near = tuple(
        TrendObservation(float(index), None, {"sleep_onset_minute_of_day": value})
        for index, value in enumerate([1430.0, 0.0, 10.0, 20.0])
    )
    far = (*near[:3], TrendObservation(3.0, None, {"sleep_onset_minute_of_day": 720.0}))
    assert compute_trend_components(near, SLEEP_ONSET_SPEC).values[0] < 0.5
    assert compute_trend_components(far, SLEEP_ONSET_SPEC).values[0] == pytest.approx(
        1.0
    )


def test_all_missing_current_evidence_forces_null_mask() -> None:
    result = compute_trend_components(
        _observations([0.5, 0.5, 0.5, None]),
        ACTIVITY_SPEC,
    )
    assert result.personal_change_mask == 0
    assert result.values == (0.0,) * len(COMPONENT_ORDER)


def test_walking_speed_uses_scene_q10_q90_then_same_day_median() -> None:
    days = tuple(date(2026, 7, 1) + timedelta(days=index) for index in range(4))
    gait = tuple(
        CameraGaitDay(
            day,
            camera,
            "scene-1",
            speed,
            None,
            None,
            None,
        )
        for camera, speeds in (
            ("camera-a", (1.0, 2.0, 3.0, 1.0)),
            ("camera-b", (1.0, 2.0, 3.0, 3.0)),
        )
        for day, speed in zip(days, speeds, strict=True)
    )
    result = _walking_speed_components(days, gait)
    assert result.personal_change_mask == 1
    assert result.details["current_normalized_scene_median"] == pytest.approx(0.5)
    assert result.values[0] == pytest.approx(0.5)
    assert result.details["current_available_scene_count"] == 2


def test_walking_speed_rejects_constant_denominator_and_new_scene_history() -> None:
    days = tuple(date(2026, 7, 1) + timedelta(days=index) for index in range(4))
    constant = tuple(
        CameraGaitDay(day, "camera-a", "scene-1", 1.0, None, None, None) for day in days
    )
    assert _walking_speed_components(days, constant).personal_change_mask == 0
    changed_scene = (
        *constant[:3],
        CameraGaitDay(days[-1], "camera-a", "scene-2", 2.0, None, None, None),
    )
    assert _walking_speed_components(days, changed_scene).personal_change_mask == 0


def test_sample_weights_equalize_class_participant_and_repeated_windows() -> None:
    frame = pd.DataFrame(
        {
            "global_participant_id": ["a", "a", "b", "c"],
            "target": [0, 0, 0, 1],
        }
    )
    weight = personal_trend_sample_weights(frame)
    assert weight.mean() == pytest.approx(1.0)
    assert weight[:3].sum() == pytest.approx(weight[3])
    assert weight[:2].sum() == pytest.approx(weight[2])


@pytest.fixture(scope="module")
def frozen_inputs():
    config = load_personal_trend_config(CONFIG, repository_root=ROOT)
    return load_personal_trend_inputs(config)


def test_frozen_inputs_preserve_split_schema_experts_eval_and_model006(
    frozen_inputs,
) -> None:
    assert len(frozen_inputs.frames["psyche_d"]) == 10866
    assert len(frozen_inputs.frames["shenzhen_elderly"]) == 5327
    assert len(frozen_inputs.upstream_protection["experts"]) == 5


def test_proxy_frames_have_expected_strict_availability(frozen_inputs) -> None:
    psyche_ids = set(
        frozen_inputs.assignments.loc[
            frozen_inputs.assignments["dataset_id"].astype(str).eq("psyche_d"),
            "global_participant_id",
        ].astype(str)
    )
    activity = build_personal_trend_training_frame(
        frozen_inputs,
        "activity",
        activity_ecdf_training_participant_ids=psyche_ids,
    )
    sleep = build_personal_trend_training_frame(frozen_inputs, "sleep")
    social = build_personal_trend_training_frame(frozen_inputs, "social")
    assert (len(activity), int(activity["available"].sum())) == (10866, 1390)
    assert (len(sleep), int(sleep["available"].sum())) == (10866, 1352)
    assert (len(social), int(social["available"].sum())) == (5327, 5327)
    assert activity.loc[activity["available"], "valid_history_days"].eq(3).all()
    assert sleep.loc[sleep["available"], "valid_history_days"].eq(3).all()
    assert social["timescale_semantics"].str.contains("proxy").all()


def test_proxy_component_columns_exclude_labels_and_model006_predictions(
    frozen_inputs,
) -> None:
    frame = build_personal_trend_training_frame(frozen_inputs, "social")
    risk_inputs = set(COMPONENT_ORDER)
    assert risk_inputs.issubset(frame.columns)
    assert not any(
        token in column.lower()
        for column in risk_inputs
        for token in ("phq", "grade", "model006", "target", "mask")
    )
    assert np.isfinite(frame.loc[:, list(COMPONENT_ORDER)].to_numpy()).all()
