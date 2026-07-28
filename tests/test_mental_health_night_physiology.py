from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from elderly_monitoring.modules.mental_health.baseline import score_daily_mental_health
from elderly_monitoring.modules.mental_health.feature_extraction.physiology import (
    build_night_physiology_result,
    normalize_night_physiology_daily,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


def physiology_day(
    day: str,
    *,
    mean_heart_rate: float = 68.0,
    mean_breath_rate: float = 16.0,
    heart_rate_std: float = 3.0,
    breath_rate_std: float = 1.0,
    heart_rate_range: float = 12.0,
    breath_rate_range: float = 4.0,
    abnormal_heart_rate_count: int = 0,
) -> dict[str, object]:
    return {
        "person_id": "elder_001",
        "date": day,
        "mean_heart_rate": mean_heart_rate,
        "mean_breath_rate": mean_breath_rate,
        "heart_rate_std": heart_rate_std,
        "breath_rate_std": breath_rate_std,
        "heart_rate_range": heart_rate_range,
        "breath_rate_range": breath_rate_range,
        "abnormal_heart_rate_count": abnormal_heart_rate_count,
        "data_quality": "valid",
        "quality_score": 1.0,
    }


class MentalHealthNightPhysiologyTest(unittest.TestCase):
    def test_normalization_accepts_sleep_device_aliases_and_derives_ranges(self) -> None:
        [daily] = normalize_night_physiology_daily(
            [
                {
                    "date": "2026-07-17",
                    "meanHeartRate": 70,
                    "meanBreathRate": 17,
                    "heartRateMin": 60,
                    "heartRateMax": 82,
                    "highCount": 1,
                    "lowCount": 0,
                    "lowHighCount": 0,
                }
            ],
            person_id="elder_001",
            device_serial="SLEEP001",
        )

        self.assertEqual(daily["mean_heart_rate"], 70)
        self.assertEqual(daily["mean_breath_rate"], 17)
        self.assertEqual(daily["heart_rate_range"], 22)
        self.assertEqual(daily["abnormal_heart_rate_count"], 1)
        self.assertTrue(daily["baseline_eligible"])

    def test_night_physiology_scores_against_personal_history(self) -> None:
        history = [physiology_day(f"2026-07-{10 + index:02d}") for index in range(7)]

        result = build_night_physiology_result(
            person_id="elder_001",
            daily_features=[
                physiology_day(
                    "2026-07-17",
                    mean_heart_rate=86,
                    mean_breath_rate=22,
                    heart_rate_std=9,
                    breath_rate_std=3.5,
                    heart_rate_range=34,
                    breath_rate_range=12,
                    abnormal_heart_rate_count=3,
                )
            ],
            history_daily_features=history,
        )

        [daily] = result["daily_features"]
        self.assertGreaterEqual(daily["night_physiology_score"], 0.6)
        self.assertEqual(daily["baseline_confidence"], "medium")
        self.assertIn("nighttime_heart_rate_shift", daily["night_physiology_factors"])
        self.assertIn("nighttime_breath_rate_shift", daily["night_physiology_factors"])

    def test_night_physiology_enters_daily_mental_health_baseline_as_auxiliary_score(self) -> None:
        history = [physiology_day(f"2026-07-{10 + index:02d}") for index in range(7)]
        current = physiology_day(
            "2026-07-17",
            mean_heart_rate=86,
            mean_breath_rate=22,
            heart_rate_std=9,
            breath_rate_std=3.5,
            heart_rate_range=34,
            breath_rate_range=12,
        )

        [result] = score_daily_mental_health(history, [current])

        self.assertGreaterEqual(result["night_physiology_score"], 0.6)
        self.assertIn("night_physiology_score", result["risk_factor_details"])
        self.assertEqual(result["activity_drop_score"], None)
        self.assertEqual(result["sleep_disturbance_score"], None)

    def test_night_physiology_endpoint(self) -> None:
        settings = ServiceSettings(api_token="api", model_path="missing.pt")
        client = TestClient(create_app(settings=settings))
        response = client.post(
            "/v1/mental-health/night-physiology",
            headers={"Authorization": "Bearer api"},
            json={
                "person_id": "elder_001",
                "daily_features": [physiology_day("2026-07-17")],
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["schema_version"], "night_physiology_service_v1")
        self.assertEqual(data["daily_features"][0]["date"], "2026-07-17")


if __name__ == "__main__":
    unittest.main()
