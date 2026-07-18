from __future__ import annotations

import unittest
from datetime import date, timedelta

from elderly_monitoring.modules.mental_health.baseline import score_daily_mental_health
from elderly_monitoring.modules.mental_health.feature_extraction.movement_vitality import (
    build_movement_vitality_result,
    normalize_movement_vitality_daily,
    score_movement_vitality_day,
)
from elderly_monitoring.modules.mental_health.pipeline import MentalHealthRiskPipeline


def movement_day(
    day: str,
    *,
    person_id: str = "elder_001",
    gait_speed_norm_per_sec: float = 0.10,
    sit_stand_duration_seconds: float = 3.0,
    turn_duration_seconds: float = 2.0,
    turn_stability_score: float = 0.90,
    gait_cycle_stability_score: float = 0.88,
    pose_quality_coverage: float = 0.90,
) -> dict[str, object]:
    return {
        "person_id": person_id,
        "date": day,
        "start_time": f"{day}T08:00:00+08:00",
        "end_time": f"{day}T20:00:00+08:00",
        "gait_speed_norm_per_sec": gait_speed_norm_per_sec,
        "sit_stand_duration_seconds": sit_stand_duration_seconds,
        "turn_duration_seconds": turn_duration_seconds,
        "turn_stability_score": turn_stability_score,
        "gait_cycle_stability_score": gait_cycle_stability_score,
        "pose_quality_coverage": pose_quality_coverage,
    }


def movement_range(start: str, count: int, **overrides: object) -> list[dict[str, object]]:
    first = date.fromisoformat(start)
    return [
        movement_day((first + timedelta(days=index)).isoformat(), **overrides)
        for index in range(count)
    ]


class MentalHealthMovementVitalityTest(unittest.TestCase):
    def test_normalizer_accepts_product_facing_aliases(self) -> None:
        [daily] = normalize_movement_vitality_daily(
            [
                {
                    "person_id": "elder_001",
                    "date": "2026-07-15",
                    "walking_speed_norm_per_sec": 0.08,
                    "sit_to_stand_seconds_median": 4.8,
                    "turning_seconds_median": 3.1,
                    "gait_stability_score": 0.67,
                    "pose_quality_score": 0.86,
                }
            ]
        )

        self.assertEqual(daily["gait_speed_norm_per_sec"], 0.08)
        self.assertEqual(daily["sit_stand_duration_seconds"], 4.8)
        self.assertEqual(daily["turn_duration_seconds"], 3.1)
        self.assertEqual(daily["gait_cycle_stability_score"], 0.67)
        self.assertEqual(daily["pose_quality_coverage"], 0.86)

    def test_movement_vitality_scores_decline_against_personal_history(self) -> None:
        history = movement_range("2026-07-01", 7)
        current = movement_day(
            "2026-07-08",
            gait_speed_norm_per_sec=0.055,
            sit_stand_duration_seconds=5.2,
            turn_duration_seconds=3.8,
            turn_stability_score=0.62,
            gait_cycle_stability_score=0.58,
        )

        score = score_movement_vitality_day(current, history)

        self.assertIsNotNone(score.movement_vitality_score)
        self.assertGreaterEqual(score.movement_vitality_score or 0.0, 0.6)
        self.assertIn("walking_speed_decline", score.factors)
        self.assertIn("sit_to_stand_time_increase", score.factors)
        self.assertFalse(score.initial_baseline_ready is False)
        self.assertTrue(score.stable_baseline_ready)

    def test_low_pose_quality_caps_score_and_records_detail(self) -> None:
        history = movement_range("2026-07-01", 7)
        current = movement_day(
            "2026-07-08",
            gait_speed_norm_per_sec=0.02,
            sit_stand_duration_seconds=8.0,
            turn_duration_seconds=6.0,
            turn_stability_score=0.20,
            gait_cycle_stability_score=0.20,
            pose_quality_coverage=0.50,
        )

        score = score_movement_vitality_day(current, history)

        self.assertLess(score.movement_vitality_score or 0.0, 0.6)
        self.assertIn("quality_cap", score.risk_factor_details)

    def test_build_result_enriches_daily_features_for_algorithm_service(self) -> None:
        result = build_movement_vitality_result(
            person_id="elder_001",
            history_daily_features=movement_range("2026-07-01", 7),
            daily_features=[
                movement_day(
                    "2026-07-08",
                    gait_speed_norm_per_sec=0.055,
                    sit_stand_duration_seconds=5.2,
                    turn_duration_seconds=3.8,
                    turn_stability_score=0.62,
                    gait_cycle_stability_score=0.58,
                )
            ],
        )

        self.assertEqual(result["schema_version"], "movement_vitality_service_v1")
        [daily] = result["daily_features"]
        self.assertGreaterEqual(daily["movement_vitality_score"], 0.6)
        self.assertFalse("diagnosis" in daily)
        self.assertIn("not a medical diagnosis", result["medical_disclaimer"])

    def test_movement_vitality_enters_daily_baseline_and_pipeline(self) -> None:
        history = movement_range("2026-07-01", 7)
        current = movement_day(
            "2026-07-08",
            gait_speed_norm_per_sec=0.055,
            sit_stand_duration_seconds=5.2,
            turn_duration_seconds=3.8,
            turn_stability_score=0.62,
            gait_cycle_stability_score=0.58,
        )

        [features] = score_daily_mental_health(history, [current])

        self.assertGreaterEqual(features["movement_vitality_score"], 0.6)
        self.assertIn("movement_vitality_score", features["risk_factor_details"])
        self.assertIn(
            "turn_duration_seconds",
            features["risk_factor_details"]["movement_vitality_score"],
        )

        event = MentalHealthRiskPipeline().predict_from_features(features)
        self.assertIn("movement_vitality_decline", event.risk_factors)
        self.assertFalse(event.metadata["diagnosis"])


if __name__ == "__main__":
    unittest.main()
