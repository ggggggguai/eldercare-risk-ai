from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from elderly_monitoring.modules.mental_health.feature_extraction.sleep import (
    build_sleep_rhythm_result,
    normalize_ep_sleep_reports,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


def ep_report(
    day: str,
    *,
    sleep_time: str,
    wake_time: str,
    in_bed: int = 28800,
    sleep: int = 25200,
    latency_minutes: int = 20,
    leave_bed_count: int = 1,
) -> dict[str, object]:
    bed_hour, bed_minute = sleep_time.split("T")[1][:5].split(":")
    bed_time = f"{sleep_time[:11]}{int(bed_hour):02d}:{max(int(bed_minute) - latency_minutes, 0):02d}:00+08:00"
    return {
        "date": day,
        "reportStartTime": bed_time,
        "reportEndTime": f"{day}T08:00:00+08:00",
        "sleepAnalysis": {
            "resultCode": 0,
            "timeOutput": [
                {"type": 1, "value": bed_time},
                {"type": 2, "value": sleep_time},
                {"type": 3, "value": wake_time},
                {"type": 4, "value": f"{day}T07:10:00+08:00"},
                {"type": 5, "value": in_bed},
                {"type": 6, "value": in_bed - sleep},
                {"type": 7, "value": int(sleep * 0.65)},
                {"type": 8, "value": int(sleep * 0.25)},
                {"type": 9, "value": sleep},
                {"type": 10, "value": leave_bed_count * 300},
            ],
            "leaveBedCount": leave_bed_count,
            "nightAwakenings": 2,
            "meanHeartFreqOutPut": {"value": 68},
            "meanBreathFreqOutPut": {"value": 16},
            "freqRecordOutput": [
                {"heartFreq": 64, "breathFreq": 15},
                {"heartFreq": 68, "breathFreq": 16},
                {"heartFreq": 72, "breathFreq": 18},
            ],
            "percentageOutPut": {"sleepPoint": 82},
            "moveOutPut": {"moveTimes": 22, "moveTimePerHour": 3.1},
        },
    }


class MentalHealthSleepRhythmTest(unittest.TestCase):
    def test_ep_sleep_report_is_normalized_to_daily_features(self) -> None:
        [result] = normalize_ep_sleep_reports(
            [ep_report("2026-07-17", sleep_time="2026-07-16T23:20:00+08:00", wake_time="2026-07-17T06:30:00+08:00")],
            person_id="elder_001",
            device_serial="SLEEP001",
            body_detect_messages=[
                {"messageTime": "2026-07-17T01:00:00+08:00", "messageType": 2},
                {"messageTime": "2026-07-17T14:00:00+08:00", "messageType": 2},
            ],
        )

        self.assertEqual(result["person_id"], "elder_001")
        self.assertEqual(result["date"], "2026-07-17")
        self.assertAlmostEqual(result["sleep_efficiency"], 0.875)
        self.assertEqual(result["sleep_latency_minutes"], 20)
        self.assertEqual(result["night_leave_bed_count"], 1)
        self.assertEqual(result["mean_heart_rate"], 68)
        self.assertEqual(result["heart_rate_range"], 8)
        self.assertEqual(result["breath_rate_range"], 3)
        self.assertTrue(result["baseline_eligible"])

    def test_sleep_rhythm_scores_against_personal_history(self) -> None:
        history = [
            {
                "person_id": "elder_001",
                "date": f"2026-07-{10 + index:02d}",
                "sleep_efficiency": 0.88,
                "sleep_latency_minutes": 20,
                "night_awakenings": 1,
                "night_leave_bed_count": 1,
                "awake_ratio": 0.12,
                "sleep_midpoint_minute_of_day": 210,
                "data_quality": "valid",
                "quality_score": 1.0,
            }
            for index in range(7)
        ]
        result = build_sleep_rhythm_result(
            person_id="elder_001",
            daily_features=[
                {
                    "person_id": "elder_001",
                    "date": "2026-07-17",
                    "sleep_efficiency": 0.62,
                    "sleep_latency_minutes": 70,
                    "night_awakenings": 5,
                    "night_leave_bed_count": 4,
                    "awake_ratio": 0.34,
                    "sleep_midpoint_minute_of_day": 300,
                    "data_quality": "valid",
                    "quality_score": 1.0,
                }
            ],
            history_daily_features=history,
        )

        [daily] = result["daily_features"]
        self.assertGreaterEqual(daily["sleep_disturbance_score"], 0.6)
        self.assertEqual(daily["baseline_confidence"], "medium")
        self.assertIn("sleep_efficiency_decline", daily["sleep_rhythm_factors"])
        self.assertIn("night_leave_bed_increase", daily["sleep_rhythm_factors"])

    def test_sleep_rhythm_endpoint(self) -> None:
        settings = ServiceSettings(api_token="api", model_path="missing.pt")
        client = TestClient(create_app(settings=settings))
        response = client.post(
            "/v1/mental-health/sleep-rhythm",
            headers={"Authorization": "Bearer api"},
            json={
                "person_id": "elder_001",
                "reports": [
                    ep_report(
                        "2026-07-17",
                        sleep_time="2026-07-16T23:20:00+08:00",
                        wake_time="2026-07-17T06:30:00+08:00",
                    )
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["schema_version"], "sleep_rhythm_service_v1")
        self.assertEqual(data["daily_features"][0]["date"], "2026-07-17")


if __name__ == "__main__":
    unittest.main()
