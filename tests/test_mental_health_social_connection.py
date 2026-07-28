from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from elderly_monitoring.modules.mental_health.feature_extraction.social import (
    aggregate_social_connection_daily,
    build_social_connection_result,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


def request_event(day: str, call_id: str, *, direction: str = "device") -> dict[str, object]:
    return {
        "person_id": "elder_001",
        "header": {"deviceId": "S10_001", "messageTime": f"{day}T09:00:00+08:00"},
        "body": {
            "action": "request",
            "callId": call_id,
            "account": "family_001",
            "timestamp": f"{day}T09:00:00+08:00",
        },
        "direction": direction,
    }


def status_event(day: str, call_id: str, *, duration: int) -> dict[str, object]:
    return {
        "person_id": "elder_001",
        "timestamp": f"{day}T09:05:00+08:00",
        "requestId": call_id,
        "result": {"duration": duration, "msg": "completed"},
    }


class MentalHealthSocialConnectionTest(unittest.TestCase):
    def test_ertc_events_are_aggregated_into_rolling_social_features(self) -> None:
        records = [
            request_event("2026-07-11", "c1"),
            status_event("2026-07-11", "c1", duration=300),
            request_event("2026-07-12", "c2", direction="family"),
            {"person_id": "elder_001", "timestamp": "2026-07-12T09:01:00+08:00", "callId": "c2", "action": "bellTimeout"},
            request_event("2026-07-17", "c3"),
            status_event("2026-07-17", "c3", duration=120),
        ]

        daily = aggregate_social_connection_daily(records, requested_date=None)
        by_date = {row["date"]: row for row in daily}

        self.assertEqual(by_date["2026-07-17"]["call_count_7d"], 3)
        self.assertEqual(by_date["2026-07-17"]["answered_call_count_7d"], 2)
        self.assertAlmostEqual(by_date["2026-07-17"]["call_answer_rate_7d"], 0.6667)
        self.assertAlmostEqual(by_date["2026-07-17"]["call_duration_minutes_7d"], 7.0)
        self.assertEqual(by_date["2026-07-17"]["active_call_count_7d"], 2)
        self.assertEqual(by_date["2026-07-17"]["bell_timeout_count_7d"], 1)

    def test_social_connection_scores_decline_against_personal_history(self) -> None:
        history = [
            {
                "person_id": "elder_001",
                "date": f"2026-07-{10 + index:02d}",
                "call_count_7d": 8,
                "answered_call_count_7d": 7,
                "call_answer_rate_7d": 0.875,
                "call_duration_minutes_7d": 42,
                "active_call_count_7d": 3,
                "missed_call_count_7d": 1,
                "data_quality": "valid",
                "quality_score": 1.0,
            }
            for index in range(7)
        ]
        result = build_social_connection_result(
            person_id="elder_001",
            daily_features=[
                {
                    "person_id": "elder_001",
                    "date": "2026-07-17",
                    "call_count_7d": 2,
                    "answered_call_count_7d": 1,
                    "call_answer_rate_7d": 0.5,
                    "call_duration_minutes_7d": 4,
                    "active_call_count_7d": 0,
                    "missed_call_count_7d": 3,
                    "data_quality": "valid",
                    "quality_score": 1.0,
                }
            ],
            history_daily_features=history,
        )

        [daily] = result["daily_features"]
        self.assertGreaterEqual(daily["social_withdrawal_score"], 0.6)
        self.assertEqual(daily["baseline_confidence"], "medium")
        self.assertIn("call_duration_decline", daily["social_connection_factors"])
        self.assertIn("call_frequency_decline", daily["social_connection_factors"])

    def test_social_connection_endpoint(self) -> None:
        settings = ServiceSettings(api_token="api", model_path="missing.pt")
        client = TestClient(create_app(settings=settings))

        response = client.post(
            "/v1/mental-health/social-connection",
            headers={"Authorization": "Bearer api"},
            json={
                "person_id": "elder_001",
                "date": "2026-07-17",
                "call_events": [
                    request_event("2026-07-17", "c1"),
                    status_event("2026-07-17", "c1", duration=180),
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["schema_version"], "social_connection_service_v1")
        self.assertEqual(data["daily_features"][0]["date"], "2026-07-17")
        self.assertEqual(data["daily_features"][0]["call_count_7d"], 1)
        self.assertIn("call content is not analyzed", data["privacy_boundary"])


if __name__ == "__main__":
    unittest.main()
