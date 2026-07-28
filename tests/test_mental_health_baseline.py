from __future__ import annotations

import unittest
from datetime import date, timedelta

from elderly_monitoring.modules.mental_health.baseline import score_daily_mental_health
from elderly_monitoring.modules.mental_health.pipeline import MentalHealthRiskPipeline


def daily_record(
    day: str,
    *,
    person_id: str = "p01",
    activity_volume: float | None = 100.0,
    active_ratio: float | None = 0.5,
    nighttime_activity_ratio: float | None = 0.2,
    scene_transition_count: int | None = 4,
    observation_coverage: float | None = 1.0,
    sleep_onset_latency: float | None = 20.0,
    night_awakenings: int | None = 2,
    sleep_efficiency: float | None = 0.85,
    sleep_quality_score: float | None = 1.0,
    data_quality_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "person_id": person_id,
        "date": day,
        "start_time": f"{day}T08:00:00+08:00",
        "end_time": f"{day}T20:00:00+08:00",
        "valid_observation_seconds": 3600.0 if observation_coverage else 0.0,
        "activity_volume": activity_volume,
        "active_ratio": active_ratio,
        "nighttime_activity_ratio": nighttime_activity_ratio,
        "scene_transition_count": scene_transition_count,
        "observation_coverage": observation_coverage,
        "sleep_onset_latency": sleep_onset_latency,
        "night_awakenings": night_awakenings,
        "sleep_efficiency": sleep_efficiency,
        "quality_score": sleep_quality_score,
        "data_quality_flags": data_quality_flags or [],
    }


def day_range(start: str, count: int, **overrides: object) -> list[dict[str, object]]:
    first = date.fromisoformat(start)
    return [
        daily_record((first + timedelta(days=index)).isoformat(), **overrides)
        for index in range(count)
    ]


class MentalHealthBaselineTest(unittest.TestCase):
    def test_seven_stable_days_produce_ready_near_zero_baseline(self) -> None:
        history = day_range("2026-07-01", 7)

        result = score_daily_mental_health(history, [daily_record("2026-07-08")])[0]

        self.assertEqual(result["activity_drop_score"], 0.0)
        self.assertEqual(result["sleep_disturbance_score"], 0.0)
        self.assertEqual(result["routine_irregularity_score"], 0.0)
        self.assertEqual(result["persistent_abnormal_days"], 0)
        self.assertEqual(result["baseline_quality"], 1.0)
        self.assertEqual(result["feature_coverage"], 1.0)
        self.assertTrue(result["initial_baseline_ready"])
        self.assertTrue(result["stable_baseline_ready"])
        self.assertEqual(result["baseline_window"]["eligible_history_days"], 7)
        self.assertEqual(result["baseline_window"]["start_date"], "2026-07-01")
        self.assertEqual(result["baseline_window"]["end_date"], "2026-07-07")

    def test_zero_variance_history_detects_directional_changes(self) -> None:
        history = day_range("2026-07-01", 7)
        degraded = daily_record(
            "2026-07-08",
            activity_volume=40.0,
            active_ratio=0.2,
            nighttime_activity_ratio=0.8,
            scene_transition_count=12,
            sleep_onset_latency=60.0,
            night_awakenings=6,
            sleep_efficiency=0.5,
        )

        result = score_daily_mental_health(history, [degraded])[0]

        self.assertEqual(result["activity_drop_score"], 1.0)
        self.assertEqual(result["sleep_disturbance_score"], 1.0)
        self.assertEqual(result["routine_irregularity_score"], 1.0)
        detail = result["risk_factor_details"]["activity_drop_score"]["activity_volume"]
        self.assertEqual(detail["direction"], "decrease")
        self.assertEqual(detail["standardized_component"], 1.0)

    def test_deviation_components_follow_configured_deterministic_formula(self) -> None:
        history = day_range("2026-07-01", 7)

        result = score_daily_mental_health(
            history,
            [daily_record("2026-07-08", activity_volume=99.0)],
        )[0]

        detail = result["risk_factor_details"]["activity_drop_score"]["activity_volume"]
        self.assertEqual(detail["scale_floor"], 5.0)
        self.assertEqual(detail["standardized_component"], 0.1)
        self.assertEqual(detail["relative_component"], 0.02)
        self.assertEqual(detail["quantile_component"], 0.2)
        self.assertEqual(detail["score"], 0.2)

    def test_risk_directions_do_not_penalize_improvements(self) -> None:
        history = day_range("2026-07-01", 7)
        improved = daily_record(
            "2026-07-08",
            activity_volume=130.0,
            active_ratio=0.7,
            nighttime_activity_ratio=None,
            scene_transition_count=None,
            sleep_onset_latency=None,
            night_awakenings=None,
            sleep_efficiency=0.95,
        )

        result = score_daily_mental_health(history, [improved])[0]

        self.assertEqual(result["activity_drop_score"], 0.0)
        self.assertEqual(result["sleep_disturbance_score"], 0.0)
        self.assertIsNone(result["routine_irregularity_score"])

    def test_current_day_and_duplicate_history_days_do_not_pollute_baseline(self) -> None:
        history = day_range("2026-06-20", 16)
        history.append(daily_record("2026-06-25"))
        history.append(daily_record("2026-07-06", activity_volume=1.0, active_ratio=0.01))

        result = score_daily_mental_health(
            history,
            [daily_record("2026-07-06", activity_volume=100.0, active_ratio=0.5)],
        )[0]

        self.assertEqual(result["activity_drop_score"], 0.0)
        self.assertEqual(result["baseline_window"]["eligible_history_days"], 14)
        self.assertEqual(result["baseline_window"]["start_date"], "2026-06-22")
        self.assertEqual(result["baseline_window"]["end_date"], "2026-07-05")

    def test_persistence_uses_consecutive_qualified_calendar_days(self) -> None:
        stable = day_range("2026-07-01", 7)
        degraded = {
            "activity_volume": 20.0,
            "active_ratio": 0.1,
            "sleep_onset_latency": 80.0,
            "night_awakenings": 8,
            "sleep_efficiency": 0.4,
        }
        history = stable + day_range("2026-07-08", 2, **degraded)

        result = score_daily_mental_health(
            history,
            [daily_record("2026-07-10", **degraded)],
        )[0]

        self.assertEqual(result["persistent_abnormal_days"], 3)
        self.assertEqual(result["evidence_window"]["start_date"], "2026-07-08")
        self.assertEqual(result["evidence_window"]["end_date"], "2026-07-10")

    def test_abnormal_history_days_do_not_enter_rolling_baseline(self) -> None:
        stable = day_range("2026-07-01", 7)
        degraded = {
            "activity_volume": 20.0,
            "active_ratio": 0.1,
            "sleep_onset_latency": 80.0,
            "night_awakenings": 8,
            "sleep_efficiency": 0.4,
        }
        history = stable + day_range("2026-07-08", 10, **degraded)

        result = score_daily_mental_health(
            history,
            [daily_record("2026-07-18", **degraded)],
        )[0]

        self.assertEqual(result["activity_drop_score"], 1.0)
        self.assertEqual(result["sleep_disturbance_score"], 1.0)
        self.assertEqual(result["baseline_window"]["start_date"], "2026-07-01")
        self.assertEqual(result["baseline_window"]["end_date"], "2026-07-07")

    def test_missing_or_low_quality_day_breaks_persistence(self) -> None:
        degraded = {
            "activity_volume": 20.0,
            "active_ratio": 0.1,
            "sleep_onset_latency": None,
            "night_awakenings": None,
            "sleep_efficiency": None,
            "sleep_quality_score": None,
        }
        history = day_range("2026-07-01", 7)
        history.append(daily_record("2026-07-08", **degraded))
        history.append(
            daily_record(
                "2026-07-09",
                activity_volume=None,
                active_ratio=None,
                nighttime_activity_ratio=None,
                scene_transition_count=None,
                observation_coverage=0.0,
                sleep_onset_latency=None,
                night_awakenings=None,
                sleep_efficiency=None,
                sleep_quality_score=None,
            )
        )

        result = score_daily_mental_health(
            history,
            [daily_record("2026-07-10", **degraded)],
        )[0]

        self.assertEqual(result["persistent_abnormal_days"], 1)
        self.assertEqual(result["evidence_window"]["start_date"], "2026-07-10")

    def test_insufficient_history_keeps_baseline_scores_unavailable(self) -> None:
        result = score_daily_mental_health(
            day_range("2026-07-01", 2),
            [daily_record("2026-07-03", activity_volume=1.0, sleep_efficiency=0.1)],
        )[0]

        self.assertIsNone(result["activity_drop_score"])
        self.assertIsNone(result["sleep_disturbance_score"])
        self.assertIsNone(result["routine_irregularity_score"])
        self.assertEqual(result["feature_coverage"], 0.0)
        self.assertFalse(result["initial_baseline_ready"])
        self.assertEqual(result["persistent_abnormal_days"], 0)
        self.assertIn("insufficient_baseline_history", result["data_quality_flags"])

    def test_missing_sleep_is_not_a_zero_score_and_coverage_uses_expected_weights(self) -> None:
        history = day_range(
            "2026-07-01",
            7,
            sleep_onset_latency=None,
            night_awakenings=None,
            sleep_efficiency=None,
            sleep_quality_score=None,
        )
        current = daily_record(
            "2026-07-08",
            sleep_onset_latency=None,
            night_awakenings=None,
            sleep_efficiency=None,
            sleep_quality_score=None,
        )

        result = score_daily_mental_health(history, [current])[0]

        self.assertEqual(result["activity_drop_score"], 0.0)
        self.assertIsNone(result["sleep_disturbance_score"])
        self.assertEqual(result["routine_irregularity_score"], 0.0)
        self.assertAlmostEqual(result["feature_coverage"], 0.34 / 0.56, places=4)

    def test_sleep_only_history_uses_configured_missing_quality_default(self) -> None:
        history = day_range(
            "2026-07-01",
            7,
            activity_volume=None,
            active_ratio=None,
            nighttime_activity_ratio=None,
            scene_transition_count=None,
            observation_coverage=None,
            sleep_quality_score=None,
        )
        current = daily_record(
            "2026-07-08",
            activity_volume=None,
            active_ratio=None,
            nighttime_activity_ratio=None,
            scene_transition_count=None,
            observation_coverage=None,
            sleep_quality_score=None,
        )

        result = score_daily_mental_health(history, [current])[0]

        self.assertEqual(result["baseline_quality"], 0.5)
        self.assertEqual(result["sleep_disturbance_score"], 0.0)
        self.assertIn("missing_source_quality", result["data_quality_flags"])

    def test_people_are_scored_against_their_own_history(self) -> None:
        history = day_range("2026-07-01", 7, person_id="p01", activity_volume=100.0)
        history += day_range("2026-07-01", 7, person_id="p02", activity_volume=20.0)
        current = [
            daily_record("2026-07-08", person_id="p02", activity_volume=20.0),
            daily_record("2026-07-08", person_id="p01", activity_volume=20.0),
        ]

        results = score_daily_mental_health(history, current)

        self.assertEqual([item["person_id"] for item in results], ["p01", "p02"])
        self.assertEqual(results[0]["activity_drop_score"], 1.0)
        self.assertEqual(results[1]["activity_drop_score"], 0.0)

    def test_a_grade_daytime_fields_enter_unified_activity_and_routine_scores(self) -> None:
        history = [
            {
                "person_id": "p01",
                "date": f"2026-07-{index:02d}",
                "start_time": f"2026-07-{index:02d}T08:00:00+08:00",
                "end_time": f"2026-07-{index:02d}T20:00:00+08:00",
                "valid_daytime_detection_minutes": 600.0,
                "daytime_active_minutes": 160.0,
                "weighted_daytime_activity": 80.0,
                "bedroom_stay_ratio": 0.18,
                "outdoor_event_count": 2,
                "outdoor_total_duration_minutes": 70.0,
                "data_quality_flags": [],
            }
            for index in range(1, 8)
        ]
        current = {
            **history[-1],
            "date": "2026-07-08",
            "start_time": "2026-07-08T08:00:00+08:00",
            "end_time": "2026-07-08T20:00:00+08:00",
            "daytime_active_minutes": 40.0,
            "weighted_daytime_activity": 20.0,
            "bedroom_stay_ratio": 0.78,
            "outdoor_event_count": 0,
            "outdoor_total_duration_minutes": 0.0,
        }

        result = score_daily_mental_health(history, [current])[0]

        self.assertGreaterEqual(result["activity_drop_score"], 0.6)
        self.assertGreaterEqual(result["routine_irregularity_score"], 0.6)
        self.assertIn("daytime_active_minutes", result["risk_factor_details"]["activity_drop_score"])
        self.assertIn("outdoor_event_count", result["risk_factor_details"]["activity_drop_score"])
        self.assertIn("bedroom_stay_ratio", result["risk_factor_details"]["routine_irregularity_score"])

    def test_a_grade_sleep_drift_and_leave_bed_enter_unified_sleep_score(self) -> None:
        history = [
            daily_record(
                f"2026-07-{index:02d}",
                sleep_efficiency=0.88,
                sleep_onset_latency=18.0,
                night_awakenings=1,
            )
            | {
                "night_leave_bed_count": 1,
                "night_leave_bed_minutes": 5.0,
                "sleep_midpoint_minute_of_day": 210.0,
            }
            for index in range(1, 8)
        ]
        current = daily_record(
            "2026-07-08",
            sleep_efficiency=0.70,
            sleep_onset_latency=55.0,
            night_awakenings=4,
        ) | {
            "night_leave_bed_count": 5,
            "night_leave_bed_minutes": 35.0,
            "sleep_midpoint_minute_of_day": 330.0,
        }

        result = score_daily_mental_health(history, [current])[0]

        self.assertGreaterEqual(result["sleep_disturbance_score"], 0.6)
        sleep_details = result["risk_factor_details"]["sleep_disturbance_score"]
        self.assertIn("night_leave_bed_count", sleep_details)
        self.assertIn("night_leave_bed_minutes", sleep_details)
        self.assertIn("sleep_midpoint_minute_of_day", sleep_details)

    def test_a_grade_social_call_metrics_derive_social_score_for_pipeline(self) -> None:
        history = [
            {
                "person_id": "p01",
                "date": f"2026-07-{index:02d}",
                "start_time": f"2026-07-{index:02d}T08:00:00+08:00",
                "end_time": f"2026-07-{index:02d}T20:00:00+08:00",
                "call_count_7d": 9,
                "answered_call_count_7d": 8,
                "call_answer_rate_7d": 0.9,
                "call_duration_minutes_7d": 45.0,
                "active_call_count_7d": 4,
                "missed_call_count_7d": 1,
                "quality_score": 1.0,
            }
            for index in range(1, 8)
        ]
        current = {
            **history[-1],
            "date": "2026-07-08",
            "start_time": "2026-07-08T08:00:00+08:00",
            "end_time": "2026-07-08T20:00:00+08:00",
            "call_count_7d": 2,
            "answered_call_count_7d": 1,
            "call_answer_rate_7d": 0.5,
            "call_duration_minutes_7d": 4.0,
            "active_call_count_7d": 0,
            "missed_call_count_7d": 4,
        }

        [features] = score_daily_mental_health(history, [current])
        event = MentalHealthRiskPipeline().predict_from_features(features)

        self.assertGreaterEqual(features["social_withdrawal_score"], 0.6)
        self.assertIn("call_duration_minutes_7d", features["risk_factor_details"]["social_withdrawal_score"])
        self.assertIn("call_answer_rate_7d", features["risk_factor_details"]["social_withdrawal_score"])
        self.assertIn("social_interaction_decline", event.risk_factors)

    def test_end_to_end_degradation_reaches_level_three_after_three_days(self) -> None:
        degraded = {
            "activity_volume": 20.0,
            "active_ratio": 0.1,
            "sleep_onset_latency": 80.0,
            "night_awakenings": 8,
            "sleep_efficiency": 0.4,
        }
        history = day_range("2026-07-01", 7) + day_range("2026-07-08", 2, **degraded)
        features = score_daily_mental_health(
            history,
            [daily_record("2026-07-10", **degraded)],
        )[0]

        event = MentalHealthRiskPipeline().predict_from_features(features)

        self.assertGreaterEqual(event.risk_score, 0.65)
        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.recommended_action, "manual_review")


if __name__ == "__main__":
    unittest.main()
