from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social import (
    load_mood_social_config,
    map_mood_social_features,
)


TARGET_DATE = date(2026, 7, 28)


def _day(offset: int) -> date:
    return TARGET_DATE - timedelta(days=offset)


def _gait_metric(
    camera_id: str,
    scene_version: str,
    *,
    speed: float | None = None,
    sit_to_stand: float | None = None,
    turn: float | None = None,
    stability: float | None = None,
) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "scene_version": scene_version,
        "gait_speed_image_norm_per_sec": speed,
        "sit_to_stand_duration_seconds": sit_to_stand,
        "turn_duration_seconds": turn,
        "postural_stability": stability,
    }


def _activity(
    *,
    valid_minutes: float = 60.0,
    active_minutes: float | None = 30.0,
    low_minutes: float | None = 30.0,
    intensity: float = 0.5,
    sedentary_bout_total: float | None = None,
    longest_sedentary_bout: float | None = None,
    gait: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    hourly_coverage = [0.0] * 24
    hourly_intensity: list[float | None] = [None] * 24
    remaining = valid_minutes
    for hour in range(6, 18):
        if remaining <= 0.0:
            break
        covered = min(60.0, remaining)
        hourly_coverage[hour] = covered
        hourly_intensity[hour] = intensity
        remaining -= covered
    if remaining > 0.0:
        raise ValueError("activity helper only supports the 06:00-18:00 window")

    if low_minutes is None:
        default_bout = None
    elif low_minutes <= 0.0:
        default_bout = 0.0
    else:
        default_bout = min(30.0, low_minutes)
    bout_total = default_bout if sedentary_bout_total is None else sedentary_bout_total
    longest = bout_total if longest_sedentary_bout is None else longest_sedentary_bout
    return {
        "daytime_active_minutes": active_minutes,
        "weighted_daytime_activity": intensity * valid_minutes,
        "valid_daytime_detection_minutes": valid_minutes,
        "low_activity_minutes": low_minutes,
        "sedentary_bout_total_minutes": bout_total,
        "longest_sedentary_bout_minutes": longest,
        "activity_peak_minute_of_day": 360,
        "hourly_activity_intensity": hourly_intensity,
        "hourly_valid_detection_minutes": hourly_coverage,
        "camera_gait_metrics": gait or [],
    }


def _zero_coverage_activity() -> dict[str, object]:
    return {
        "daytime_active_minutes": None,
        "weighted_daytime_activity": None,
        "valid_daytime_detection_minutes": 0.0,
        "low_activity_minutes": None,
        "sedentary_bout_total_minutes": None,
        "longest_sedentary_bout_minutes": None,
        "activity_peak_minute_of_day": None,
        "hourly_activity_intensity": [None] * 24,
        "hourly_valid_detection_minutes": [0.0] * 24,
        "camera_gait_metrics": [],
    }


def _sleep(
    *,
    onset: int,
    wake: int,
    in_bed_minutes: float = 480.0,
    sleep_minutes: float = 420.0,
    awake_minutes: float = 60.0,
    awakening_count: int = 3,
    night_exit_count: int = 0,
    night_exit_minutes: float = 0.0,
) -> dict[str, object]:
    return {
        "in_bed_minutes": in_bed_minutes,
        "sleep_minutes": sleep_minutes,
        "sleep_efficiency": sleep_minutes / in_bed_minutes,
        "sleep_onset_minute_of_day": onset,
        "wake_time_minute_of_day": wake,
        "awake_minutes": awake_minutes,
        "awakening_count": awakening_count,
        "night_exit_count": night_exit_count,
        "night_exit_minutes": night_exit_minutes,
    }


def _physiology() -> dict[str, object]:
    return {
        "heart_rate_mean_bpm": 60.0,
        "snoring_minutes": 0.0,
    }


def _observed_social(
    *,
    answered: int = 0,
    missed: int = 0,
    outgoing: int = 0,
    duration: float | None = None,
    contacts: int | None = None,
) -> dict[str, object]:
    connected = answered + outgoing
    actual_duration = (
        (0.0 if connected == 0 else float(connected * 10))
        if duration is None
        else duration
    )
    actual_contacts = (0 if connected == 0 else 1) if contacts is None else contacts
    return {
        "call_log_observed": True,
        "incoming_call_opportunities": answered + missed,
        "answered_call_count": answered,
        "missed_call_count": missed,
        "outgoing_call_count": outgoing,
        "connected_duration_minutes": actual_duration,
        "active_contact_count": actual_contacts,
    }


def _unobserved_social() -> dict[str, object]:
    return {"call_log_observed": False}


def _daily_record(
    record_date: date,
    *,
    activity: dict[str, object] | None = None,
    sleep: dict[str, object] | None = None,
    physiology: dict[str, object] | None = None,
    social: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "date": record_date.isoformat(),
        "activity": activity,
        "sleep": sleep,
        "physiology": physiology,
        "social": social,
    }


def _request_payload(
    *,
    current: dict[str, object] | None = None,
    history: list[dict[str, object]] | None = None,
    profile: dict[str, object] | None = None,
) -> dict[str, object]:
    ordered_history = sorted(
        history or [],
        key=lambda item: str(item["date"]),
    )
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "req_mh_003_mapper",
        "person_id": "elder-001",
        "target_date": TARGET_DATE.isoformat(),
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera", "sleep_device", "s10"],
        "profile": profile or {},
        "current_daily_features": current or _daily_record(TARGET_DATE),
        "history_daily_features": ordered_history,
        "history_attention_indices": [],
    }


def _mapped_day(mapped: Any, record_date: date) -> Any:
    return next(item for item in mapped.daily_features if item.date == record_date)


class MoodSocialFeatureMapperTest(unittest.TestCase):
    def test_state_window_is_d_minus_6_through_d_only(self) -> None:
        history = [
            _daily_record(
                _day(28),
                activity=_activity(
                    active_minutes=0.0,
                    low_minutes=60.0,
                    intensity=0.0,
                    gait=[_gait_metric("camera-z", "scene-2", speed=0.1)],
                ),
            ),
            _daily_record(
                _day(7),
                activity=_activity(
                    active_minutes=0.0,
                    low_minutes=60.0,
                    intensity=0.0,
                    gait=[_gait_metric("camera-b", "scene-1", speed=0.2)],
                ),
            ),
            _daily_record(
                _day(6),
                activity=_activity(
                    active_minutes=60.0,
                    low_minutes=0.0,
                    intensity=1.0,
                ),
            ),
        ]
        current = _daily_record(
            TARGET_DATE,
            activity=_activity(
                active_minutes=60.0,
                low_minutes=0.0,
                intensity=1.0,
                gait=[_gait_metric("camera-a", "scene-1", speed=0.3)],
            ),
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertEqual(mapped.window_start_date, _day(6))
        self.assertEqual(mapped.window_end_date, TARGET_DATE)
        self.assertEqual(
            tuple(item.date for item in mapped.daily_features),
            tuple(_day(offset) for offset in range(6, -1, -1)),
        )
        self.assertEqual(mapped.activity.value("active_ratio"), 1.0)
        self.assertEqual(mapped.activity.value("valid_days"), 2)
        self.assertEqual(
            [item.date for item in mapped.trend_context.camera_gait_days],
            [_day(28), _day(7), TARGET_DATE],
        )

    def test_activity_ratios_pool_minutes_with_separate_valid_dates(self) -> None:
        history = [
            _daily_record(
                _day(2),
                activity=_activity(
                    valid_minutes=60.0,
                    active_minutes=0.0,
                    low_minutes=60.0,
                    intensity=0.0,
                ),
            ),
            _daily_record(
                _day(1),
                activity=_activity(
                    valid_minutes=120.0,
                    active_minutes=120.0,
                    low_minutes=0.0,
                    intensity=1.0,
                ),
            ),
        ]
        current = _daily_record(
            TARGET_DATE,
            activity=_activity(
                valid_minutes=180.0,
                active_minutes=None,
                low_minutes=90.0,
                intensity=0.5,
                sedentary_bout_total=30.0,
                longest_sedentary_bout=30.0,
            ),
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertAlmostEqual(
            float(mapped.activity.value("active_ratio")),
            120.0 / 180.0,
        )
        self.assertAlmostEqual(
            float(mapped.activity.value("sedentary_ratio")),
            150.0 / 360.0,
        )
        self.assertEqual(mapped.activity.value("valid_days"), 3)
        target = _mapped_day(mapped, TARGET_DATE)
        self.assertIsNone(target.activity.value("active_ratio"))
        self.assertEqual(target.activity.mask("active_ratio"), 0)
        self.assertEqual(target.activity.mask("sedentary_ratio"), 1)

    def test_null_zero_coverage_and_observed_zero_are_distinct(self) -> None:
        history = [
            _daily_record(_day(1), activity=_zero_coverage_activity()),
        ]
        current = _daily_record(
            TARGET_DATE,
            activity=_activity(
                active_minutes=0.0,
                low_minutes=0.0,
                intensity=0.5,
            ),
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        missing = _mapped_day(mapped, _day(2))
        zero_coverage = _mapped_day(mapped, _day(1))
        observed_zero = _mapped_day(mapped, TARGET_DATE)
        for item in (missing, zero_coverage):
            self.assertEqual(item.day_mask.activity, 0)
            self.assertTrue(all(value is None for value in item.activity.values))
            self.assertTrue(all(mask == 0 for mask in item.activity.feature_mask))

        self.assertEqual(observed_zero.day_mask.activity, 1)
        self.assertEqual(observed_zero.activity.value("active_ratio"), 0.0)
        self.assertEqual(observed_zero.activity.value("sedentary_ratio"), 0.0)
        self.assertEqual(observed_zero.activity.mask("active_ratio"), 1)
        self.assertEqual(observed_zero.activity.mask("sedentary_ratio"), 1)
        self.assertEqual(mapped.activity.value("active_ratio"), 0.0)
        self.assertEqual(mapped.activity.mask("active_ratio"), 1)

    def test_activity_rhythm_formulas_match_the_frozen_golden_values(self) -> None:
        hourly_intensity: list[float | None] = [None] * 24
        for hour in range(6, 18):
            hourly_intensity[hour] = float((hour - 6) % 2)

        first = _activity(
            valid_minutes=720.0,
            active_minutes=360.0,
            low_minutes=360.0,
            intensity=0.5,
            sedentary_bout_total=180.0,
            longest_sedentary_bout=180.0,
        )
        second = _activity(
            valid_minutes=720.0,
            active_minutes=360.0,
            low_minutes=360.0,
            intensity=0.5,
            sedentary_bout_total=240.0,
            longest_sedentary_bout=240.0,
        )
        first["hourly_activity_intensity"] = list(hourly_intensity)
        second["hourly_activity_intensity"] = list(hourly_intensity)

        mapped = map_mood_social_features(
            _request_payload(
                current=_daily_record(TARGET_DATE, activity=second),
                history=[_daily_record(_day(1), activity=first)],
            )
        )

        expected = {
            "longest_inactive_bout_norm": 1.0 / 3.0,
            "relative_amplitude": 1.0 / 9.0,
            "interdaily_stability": 1.0,
            "intradaily_variability": 4.0,
            "activity_variability": 0.0,
            "feature_coverage": 2.0 / 7.0,
        }
        for feature, value in expected.items():
            with self.subTest(feature=feature):
                self.assertAlmostEqual(
                    float(mapped.activity.value(feature)),
                    value,
                    places=12,
                )
                self.assertEqual(mapped.activity.mask(feature), 1)

    def test_each_domain_day_mask_is_independent(self) -> None:
        history = [
            _daily_record(_day(3), activity=_activity()),
            _daily_record(
                _day(2),
                sleep={"sleep_minutes": 400.0},
            ),
            _daily_record(_day(1), physiology=_physiology()),
        ]
        current = _daily_record(TARGET_DATE, social=_observed_social())

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        expected = {
            _day(3): (1, 0, 0, 0),
            _day(2): (0, 1, 0, 0),
            _day(1): (0, 0, 1, 0),
            TARGET_DATE: (0, 0, 0, 1),
        }
        for record_date, masks in expected.items():
            item = _mapped_day(mapped, record_date)
            with self.subTest(date=record_date):
                self.assertEqual(
                    (
                        item.day_mask.activity,
                        item.day_mask.sleep,
                        item.day_mask.physiology,
                        item.day_mask.social,
                    ),
                    masks,
                )

    def test_sleep_circular_midpoint_fragmentation_and_regularity(self) -> None:
        first_sleep = _sleep(onset=1380, wake=420)
        second_sleep = _sleep(
            onset=60,
            wake=540,
            awake_minutes=30.0,
            awakening_count=1,
        )
        history = [_daily_record(_day(1), sleep=first_sleep)]
        current = _daily_record(TARGET_DATE, sleep=second_sleep)

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        first_day = _mapped_day(mapped, _day(1))
        expected_first_fragmentation = 60.0 / 480.0 + 3.0 / 7.0
        expected_second_fragmentation = 30.0 / 480.0 + 1.0 / 7.0
        self.assertEqual(
            first_day.sleep.value("sleep_midpoint_minute_of_day"),
            180.0,
        )
        self.assertAlmostEqual(
            float(first_day.sleep.value("sleep_fragmentation")),
            expected_first_fragmentation,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_fragmentation")),
            (expected_first_fragmentation + expected_second_fragmentation) / 2.0,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_onset_sin")),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_onset_cos")),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("wake_time_sin")),
            math.sqrt(3.0) / 2.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("wake_time_cos")),
            -0.5,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_midpoint_sin")),
            math.sqrt(3.0) / 2.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_midpoint_cos")),
            0.5,
            places=12,
        )
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_regularity")),
            math.cos(math.pi / 12.0),
            places=12,
        )
        self.assertEqual(mapped.sleep.value("valid_nights"), 2)

    def test_sleep_efficiency_pools_real_minutes_instead_of_daily_ratios(self) -> None:
        history = [
            _daily_record(
                _day(1),
                sleep=_sleep(
                    onset=1320,
                    wake=1380,
                    in_bed_minutes=60.0,
                    sleep_minutes=60.0,
                    awake_minutes=0.0,
                ),
            )
        ]
        current = _daily_record(
            TARGET_DATE,
            sleep=_sleep(
                onset=1380,
                wake=540,
                in_bed_minutes=600.0,
                sleep_minutes=300.0,
                awake_minutes=300.0,
            ),
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_efficiency")),
            360.0 / 660.0,
        )
        self.assertNotAlmostEqual(
            float(mapped.sleep.value("sleep_efficiency")),
            (1.0 + 0.5) / 2.0,
        )

    def test_exactly_opposite_sleep_phases_do_not_choose_an_arbitrary_mean(
        self,
    ) -> None:
        history = [
            _daily_record(_day(1), sleep=_sleep(onset=0, wake=480)),
        ]
        current = _daily_record(
            TARGET_DATE,
            sleep=_sleep(onset=720, wake=1200),
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        for name in (
            "sleep_onset_sin",
            "sleep_onset_cos",
            "wake_time_sin",
            "wake_time_cos",
            "sleep_midpoint_sin",
            "sleep_midpoint_cos",
        ):
            with self.subTest(feature=name):
                self.assertIsNone(mapped.sleep.value(name))
                self.assertEqual(mapped.sleep.mask(name), 0)
        self.assertAlmostEqual(
            float(mapped.sleep.value("sleep_regularity")),
            0.0,
            places=12,
        )

    def test_sleep_efficiency_without_a_positive_denominator_uses_observed_ratio(
        self,
    ) -> None:
        current = _daily_record(
            TARGET_DATE,
            sleep={
                "in_bed_minutes": 0.0,
                "sleep_minutes": 0.0,
                "sleep_efficiency": 1.0,
            },
        )

        mapped = map_mood_social_features(_request_payload(current=current))

        target = _mapped_day(mapped, TARGET_DATE)
        self.assertEqual(target.sleep.value("sleep_efficiency"), 1.0)
        self.assertEqual(target.sleep.mask("sleep_efficiency"), 1)
        self.assertEqual(mapped.sleep.value("sleep_efficiency"), 1.0)

    def test_sleep_coverage_counts_real_field_day_cells(self) -> None:
        history = [
            _daily_record(_day(1), sleep={"sleep_minutes": 400.0}),
        ]
        current = _daily_record(
            TARGET_DATE,
            sleep={"night_exit_count": 0},
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        target = _mapped_day(mapped, TARGET_DATE)
        self.assertIsInstance(target.sleep.value("night_exit_count"), int)
        self.assertEqual(mapped.sleep.value("valid_nights"), 2)
        self.assertAlmostEqual(
            float(mapped.sleep.value("feature_coverage")),
            2.0 / (7.0 * 9.0),
        )

    def test_sleep_regularity_coverage_excludes_noncontributing_singletons(
        self,
    ) -> None:
        history = [
            _daily_record(_day(2), sleep={"sleep_onset_minute_of_day": 60}),
            _daily_record(_day(1), sleep={"sleep_onset_minute_of_day": 120}),
        ]
        current = _daily_record(
            TARGET_DATE,
            sleep={"wake_time_minute_of_day": 480},
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertIsNotNone(mapped.sleep.value("sleep_regularity"))
        self.assertAlmostEqual(
            float(mapped.sleep.value("feature_coverage")),
            5.0 / (7.0 * 9.0),
        )

    def test_physiology_missingness_is_independent_from_sleep(self) -> None:
        history = [
            _daily_record(
                _day(1),
                sleep={"sleep_minutes": 400.0},
            )
        ]
        current = _daily_record(TARGET_DATE, physiology=_physiology())

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        sleep_only = _mapped_day(mapped, _day(1))
        physiology_only = _mapped_day(mapped, TARGET_DATE)
        self.assertEqual(
            (sleep_only.day_mask.sleep, sleep_only.day_mask.physiology),
            (1, 0),
        )
        self.assertEqual(
            (
                physiology_only.day_mask.sleep,
                physiology_only.day_mask.physiology,
            ),
            (0, 1),
        )
        self.assertEqual(mapped.sleep.value("valid_nights"), 1)
        self.assertEqual(mapped.physiology.value("valid_nights"), 1)
        self.assertEqual(mapped.physiology.value("heart_rate_mean_bpm"), 60.0)
        self.assertEqual(mapped.physiology.mask("heart_rate_mean_bpm"), 1)
        self.assertEqual(mapped.physiology.value("snoring_minutes_norm"), 0.0)
        self.assertEqual(mapped.physiology.mask("snoring_minutes_norm"), 1)
        self.assertIsNone(mapped.physiology.value("hrv_sdnn_ms"))
        self.assertEqual(mapped.physiology.mask("hrv_sdnn_ms"), 0)

    def test_physiology_coverage_counts_each_field_night_once(self) -> None:
        history = [
            _daily_record(
                _day(6),
                physiology={"heart_rate_mean_bpm": 60.0},
            ),
            *(
                _daily_record(
                    _day(offset),
                    physiology={"snoring_minutes": 0.0},
                )
                for offset in range(5, 0, -1)
            ),
        ]
        current = _daily_record(
            TARGET_DATE,
            physiology={"snoring_minutes": 0.0},
        )

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertEqual(mapped.physiology.value("valid_nights"), 7)
        self.assertAlmostEqual(
            float(mapped.physiology.value("feature_coverage")),
            7.0 / (7.0 * 8.0),
        )

    def test_s10_observed_zero_is_evidence_but_zero_ratios_are_null(self) -> None:
        current = _daily_record(TARGET_DATE, social=_observed_social())

        mapped = map_mood_social_features(_request_payload(current=current))

        target = _mapped_day(mapped, TARGET_DATE)
        self.assertEqual(target.day_mask.social, 1)
        self.assertEqual(target.social.value("connected_call_count"), 0)
        self.assertEqual(target.social.mask("connected_call_count"), 1)
        self.assertIsNone(target.social.value("answer_rate"))
        self.assertEqual(target.social.mask("answer_rate"), 0)
        self.assertIsNone(target.social.value("mean_connected_duration_minutes"))
        self.assertEqual(
            target.social.mask("mean_connected_duration_minutes"),
            0,
        )
        self.assertEqual(mapped.social_contact.value("outgoing_call_count"), 0.0)
        self.assertEqual(mapped.social_contact.mask("outgoing_call_count"), 1)
        self.assertEqual(
            mapped.social_contact.value("no_effective_contact_days"),
            1,
        )

    def test_s10_window_uses_observed_day_means_and_pooled_ratios(self) -> None:
        history = [
            _daily_record(
                _day(1),
                social=_observed_social(
                    answered=1,
                    missed=1,
                    outgoing=1,
                    duration=20.0,
                    contacts=1,
                ),
            )
        ]
        current = _daily_record(TARGET_DATE, social=_observed_social())

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        expected = {
            "incoming_call_opportunities": 1.0,
            "answered_call_count": 0.5,
            "missed_call_count": 0.5,
            "outgoing_call_count": 0.5,
            "connected_call_count": 1.0,
            "connected_duration_minutes": 10.0,
            "active_contact_count": 0.5,
            "answer_rate": 0.5,
            "mean_connected_duration_minutes": 10.0,
        }
        for name, value in expected.items():
            with self.subTest(feature=name):
                self.assertEqual(mapped.social_contact.value(name), value)

    def test_s10_contact_today_resets_streak_and_unobserved_today_is_null(self) -> None:
        history = [
            _daily_record(_day(1), social=_observed_social()),
        ]
        contacted = _daily_record(
            TARGET_DATE,
            social=_observed_social(answered=1, duration=5.0, contacts=1),
        )
        contacted_result = map_mood_social_features(
            _request_payload(current=contacted, history=history)
        )
        self.assertEqual(
            contacted_result.social_contact.value("no_effective_contact_days"),
            0,
        )

        unobserved = _daily_record(TARGET_DATE, social=_unobserved_social())
        unobserved_result = map_mood_social_features(
            _request_payload(current=unobserved, history=history)
        )
        self.assertIsNone(
            unobserved_result.social_contact.value("no_effective_contact_days")
        )
        self.assertEqual(
            unobserved_result.social_contact.mask("no_effective_contact_days"),
            0,
        )

    def test_s10_missing_or_unobserved_gap_breaks_consecutive_days(self) -> None:
        for gap_record in (
            None,
            _daily_record(_day(2), social=_unobserved_social()),
        ):
            history = [
                _daily_record(_day(3), social=_observed_social()),
                _daily_record(_day(1), social=_observed_social()),
            ]
            if gap_record is not None:
                history.append(gap_record)
            current = _daily_record(TARGET_DATE, social=_observed_social())

            with self.subTest(gap="missing" if gap_record is None else "unobserved"):
                mapped = map_mood_social_features(
                    _request_payload(current=current, history=history)
                )
                self.assertEqual(
                    mapped.social_contact.value("no_effective_contact_days"),
                    2,
                )
                self.assertEqual(_mapped_day(mapped, _day(2)).day_mask.social, 0)

    def test_s10_consecutive_days_are_left_censored_at_29(self) -> None:
        history = [
            _daily_record(
                _day(offset),
                social=_observed_social(),
            )
            for offset in range(28, 0, -1)
        ]
        current = _daily_record(TARGET_DATE, social=_observed_social())

        mapped = map_mood_social_features(
            _request_payload(current=current, history=history)
        )

        self.assertEqual(len(history), 28)
        self.assertEqual(
            mapped.social_contact.value("no_effective_contact_days"),
            29,
        )
        self.assertEqual(
            _mapped_day(mapped, TARGET_DATE).social.value("no_effective_contact_days"),
            29,
        )

    def test_gait_context_is_sorted_unmerged_and_mapping_is_idempotent(self) -> None:
        history = [
            _daily_record(
                _day(28),
                activity=_activity(
                    gait=[_gait_metric("camera-z", "scene-2", speed=0.1)]
                ),
            ),
            _daily_record(
                _day(2),
                activity=_activity(
                    gait=[
                        _gait_metric("camera-z", "scene-1", turn=1.4),
                        _gait_metric("camera-a", "scene-2", stability=0.7),
                        _gait_metric("camera-a", "scene-1", sit_to_stand=2.1),
                    ]
                ),
            ),
        ]
        current = _daily_record(
            TARGET_DATE,
            activity=_activity(gait=[_gait_metric("camera-b", "scene-1", speed=0.3)]),
        )
        payload = _request_payload(current=current, history=history)
        original = copy.deepcopy(payload)

        first = map_mood_social_features(payload)
        second = map_mood_social_features(payload)

        self.assertEqual(payload, original)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            [
                (item.date, item.camera_id, item.scene_version)
                for item in first.trend_context.camera_gait_days
            ],
            [
                (_day(28), "camera-z", "scene-2"),
                (_day(2), "camera-a", "scene-1"),
                (_day(2), "camera-a", "scene-2"),
                (_day(2), "camera-z", "scene-1"),
                (TARGET_DATE, "camera-b", "scene-1"),
            ],
        )
        self.assertEqual(len(first.trend_context.camera_gait_days), 5)

    def test_gait_zero_values_remain_observed_without_trend_normalization(self) -> None:
        current = _daily_record(
            TARGET_DATE,
            activity=_activity(
                gait=[
                    _gait_metric(
                        "camera-a",
                        "scene-1",
                        speed=0.0,
                        stability=0.0,
                    )
                ]
            ),
        )

        mapped = map_mood_social_features(_request_payload(current=current))

        gait = mapped.trend_context.camera_gait_days[0]
        self.assertEqual(gait.gait_speed_image_norm_per_sec, 0.0)
        self.assertEqual(gait.postural_stability, 0.0)
        self.assertEqual(
            tuple(gait.to_dict()),
            (
                "date",
                "camera_id",
                "scene_version",
                "gait_speed_image_norm_per_sec",
                "sit_to_stand_duration_seconds",
                "turn_duration_seconds",
                "postural_stability",
            ),
        )
        for forbidden in (
            "q10",
            "q90",
            "walking_speed_norm",
            "walking_speed_norm_camera",
        ):
            self.assertFalse(hasattr(gait, forbidden), forbidden)

    def test_profile_zero_values_are_not_treated_as_missing(self) -> None:
        mapped = map_mood_social_features(
            _request_payload(
                profile={
                    "chronic_disease_count": 0,
                    "functional_limitation": 0,
                    "social_participation_days_per_week": 0,
                }
            )
        )

        for name in (
            "chronic_disease_count",
            "functional_limitation",
            "social_participation_days_per_week",
        ):
            with self.subTest(feature=name):
                self.assertEqual(mapped.social_context.value(name), 0)
                self.assertEqual(mapped.social_context.mask(name), 1)
        self.assertIsNone(mapped.social_context.value("age_group"))
        self.assertEqual(mapped.social_context.mask("age_group"), 0)

    def test_public_mapping_objects_reject_invalid_window_and_group_shapes(
        self,
    ) -> None:
        mapped = map_mood_social_features(_request_payload())
        target_day = mapped.daily_features[-1]

        with self.assertRaisesRegex(ValueError, "exact calendar days"):
            replace(mapped, daily_features=(target_day,) * 7)
        with self.assertRaisesRegex(ValueError, "activity vector"):
            replace(mapped, activity=mapped.sleep)
        with self.assertRaisesRegex(ValueError, "stable date/scene order"):
            gait_current = _daily_record(
                TARGET_DATE,
                activity=_activity(
                    gait=[_gait_metric("camera-b", "scene-1", speed=0.2)]
                ),
            )
            gait_history = [
                _daily_record(
                    _day(1),
                    activity=_activity(
                        gait=[_gait_metric("camera-a", "scene-1", speed=0.1)]
                    ),
                )
            ]
            gait_mapped = map_mood_social_features(
                _request_payload(current=gait_current, history=gait_history)
            )
            replace(
                gait_mapped.trend_context,
                camera_gait_days=tuple(
                    reversed(gait_mapped.trend_context.camera_gait_days)
                ),
            )

        observed = map_mood_social_features(
            _request_payload(current=_daily_record(TARGET_DATE, activity=_activity()))
        ).daily_features[-1]
        with self.assertRaisesRegex(ValueError, "day_mask is 0"):
            replace(
                observed,
                day_mask=replace(observed.day_mask, activity=0),
            )

    def test_mapper_rejects_config_that_changes_a_frozen_mapping_value(self) -> None:
        config = load_mood_social_config()
        invalid_configs = {
            "module": replace(config, module="legacy"),
            "version": replace(config, version="3.3.2"),
            "timezone": replace(
                config,
                runtime=replace(config.runtime, timezone="UTC"),
            ),
            "state_window": replace(
                config,
                runtime=replace(config.runtime, inference_window_days=14),
            ),
            "history_window": replace(
                config,
                baseline=replace(
                    config.baseline,
                    history_lookback_calendar_days=27,
                ),
            ),
            "camera_start": replace(
                config,
                camera=replace(config.camera, daytime_start="07:00"),
            ),
            "camera_end": replace(
                config,
                camera=replace(config.camera, daytime_end="19:00"),
            ),
            "camera_minutes": replace(
                config,
                camera=replace(config.camera, daytime_minutes=600),
            ),
        }

        for field, invalid in invalid_configs.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                map_mood_social_features(_request_payload(), config=invalid)


if __name__ == "__main__":
    unittest.main()
