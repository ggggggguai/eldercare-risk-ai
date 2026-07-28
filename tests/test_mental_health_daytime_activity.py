from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import unittest

from fastapi.testclient import TestClient

from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    DaytimeActivityConfig,
    aggregate_activity_windows,
    aggregate_daytime_activity_from_windows,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


CORE_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def frame_record(second: int, *, x: float, zone: str = "living_room", room: str = "living_room") -> dict[str, object]:
    timestamp = datetime.fromisoformat("2026-07-16T08:00:00+08:00") + timedelta(seconds=second)
    return {
        "timestamp": timestamp.isoformat(),
        "person_id": "elder_001",
        "camera_id": "living_room_cam",
        "bbox": [x, 100.0, 50.0, 100.0],
        "bbox_confidence": 0.95,
        "keypoints": [
            {"name": name, "x": x + index, "y": 130.0 + index, "score": 0.95}
            for index, name in enumerate(CORE_KEYPOINTS)
        ],
        "keypoint_confidence": 0.95,
        "tracking_confidence": 0.95,
        "zone": zone,
        "room": room,
        "posture": "sitting",
    }


def activity_window(
    start: str,
    *,
    score: float,
    state: str = "active",
    zone: str = "living_room",
    room: str = "living_room",
    posture: str = "standing",
) -> dict[str, object]:
    window_start = datetime.fromisoformat(start)
    return {
        "window_start": window_start.isoformat(),
        "window_end": (window_start + timedelta(seconds=10)).isoformat(),
        "person_id": "elder_001",
        "room": room,
        "zone": zone,
        "active_score": score,
        "motion_state": state,
        "posture": posture,
        "valid_detection_ratio": 1.0,
        "data_quality": "valid",
        "center_path_norm": 0.0 if state == "low_motion" else 0.4,
        "pose_motion_norm": 0.0 if state == "low_motion" else 0.2,
        "zone_transition_score": 0.0,
        "posture_change_score": 0.0,
        "quality_flags": [],
    }


def normalized_frame(second: int, *, x: float = 0.20) -> dict[str, object]:
    timestamp = datetime.fromisoformat("2026-07-16T08:00:00+08:00") + timedelta(seconds=second)
    return {
        "timestamp": timestamp.isoformat(),
        "bbox": [x, 0.20, 0.10, 0.20],
        "bbox_confidence": 0.95,
        "keypoints": [
            {"name": name, "x": x + index * 0.01, "y": 0.30 + index * 0.01, "score": 0.95}
            for index, name in enumerate(CORE_KEYPOINTS)
        ],
        "keypoint_confidence": 0.95,
        "tracking_confidence": 0.95,
        "posture": "sitting",
    }


class MentalHealthDaytimeActivityTest(unittest.TestCase):
    def test_raw_records_are_aggregated_into_activity_windows(self) -> None:
        active_records = [frame_record(second, x=100.0 + second * 12.0) for second in range(10)]
        low_records = [
            frame_record(second, x=200.0, zone="sofa_area")
            for second in range(10, 20)
        ]

        windows = aggregate_activity_windows(active_records + low_records)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["motion_state"], "active")
        self.assertGreaterEqual(windows[0]["active_score"], 0.70)
        self.assertEqual(windows[0]["data_quality"], "valid")
        self.assertEqual(windows[1]["motion_state"], "low_motion")
        self.assertLessEqual(windows[1]["active_score"], 0.20)

    def test_daytime_activity_features_cover_v1_behavior_metrics(self) -> None:
        config = replace(
            DaytimeActivityConfig(),
            effective_activity_minutes=0.5,
            sedentary_min_minutes=0.5,
            bed_stay_min_minutes=0.5,
            outdoor_absence_min_minutes=1.0,
            meal_activity_min_minutes=0.5,
            min_valid_daytime_minutes=0.0,
        )
        windows = [
            activity_window(f"2026-07-16T07:00:{second:02d}+08:00", score=0.8, zone="kitchen_area", room="kitchen")
            for second in (0, 10, 20)
        ]
        windows.extend(
            activity_window(
                f"2026-07-16T08:00:{second:02d}+08:00",
                score=0.0,
                state="low_motion",
                zone="sofa_area",
                room="living_room",
                posture="sitting",
            )
            for second in (0, 10, 20)
        )
        windows.extend(
            activity_window(
                f"2026-07-16T09:00:{second:02d}+08:00",
                score=0.0,
                state="low_motion",
                zone="bed_area",
                room="bedroom",
                posture="lying",
            )
            for second in (0, 10, 20)
        )
        windows.append(activity_window("2026-07-16T10:00:00+08:00", score=0.7, zone="door_area", room="entry"))
        windows.append(activity_window("2026-07-16T10:02:00+08:00", score=0.6, zone="living_room", room="living_room"))

        [first_pass] = aggregate_daytime_activity_from_windows(
            windows,
            sleep_records=[
                {
                    "person_id": "elder_001",
                    "sleep_end_time": "2026-07-16T06:55:00+08:00",
                }
            ],
            config=config,
        )
        [result] = aggregate_daytime_activity_from_windows(
            windows,
            sleep_records=[
                {
                    "person_id": "elder_001",
                    "sleep_end_time": "2026-07-16T06:55:00+08:00",
                }
            ],
            history_daily_features=[
                {
                    "person_id": "elder_001",
                    "date": "2026-07-15",
                    "hourly_activity_vector": first_pass["hourly_activity_vector"],
                    "activity_peak_minute_of_day": first_pass["activity_peak_minute_of_day"],
                    "first_effective_activity_minute_of_day": first_pass["first_effective_activity_minute_of_day"],
                    "meal_window_activity_count": 2,
                }
            ],
            config=config,
        )

        self.assertEqual(result["data_quality"], "valid")
        self.assertAlmostEqual(result["daytime_active_minutes"], 0.8333)
        self.assertEqual(result["sedentary_bouts_count"], 1)
        self.assertAlmostEqual(result["sedentary_total_minutes"], 0.5)
        self.assertEqual(result["daytime_bed_bouts_count"], 1)
        self.assertAlmostEqual(result["daytime_bed_stay_minutes"], 0.5)
        self.assertEqual(result["room_transition_count"], 4)
        self.assertGreater(result["bedroom_stay_ratio"], 0.0)
        self.assertEqual(result["outdoor_event_count"], 1)
        self.assertAlmostEqual(result["wake_activation_delay_minutes"], 5.0)
        self.assertEqual(result["meal_window_activity_count"], 1)
        self.assertTrue(result["breakfast_related_activity"])
        self.assertEqual(result["routine_stability_score"], 1.0)
        self.assertEqual(result["activity_peak_shift_minutes"], 0.0)
        self.assertEqual(result["first_activity_shift_minutes"], 0.0)
        self.assertEqual(result["meal_routine_consistency"], 0.5)


class MentalHealthDaytimeActivityServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = ServiceSettings(api_token="api", model_path="missing.pt")
        self.client = TestClient(create_app(settings=settings))

    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer api"}

    def test_daytime_activity_endpoint_uses_roi_for_missing_zone(self) -> None:
        response = self.client.post(
            "/v1/mental-health/daytime-activity",
            headers=self.headers(),
            json={
                "person_id": "elder_001",
                "date": "2026-07-16",
                "frames": [normalized_frame(second) for second in range(10)],
                "roi_annotations": [
                    {
                        "roi_id": "roi_sofa_01",
                        "type": "sofa",
                        "room": "living_room",
                        "shape": {
                            "type": "polygon",
                            "points": [
                                {"x": 0.10, "y": 0.10},
                                {"x": 0.50, "y": 0.10},
                                {"x": 0.50, "y": 0.60},
                                {"x": 0.10, "y": 0.60},
                            ],
                            "bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.50, "y2": 0.60},
                        },
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["person_id"], "elder_001")
        self.assertEqual(data["windows"][0]["zone"], "sofa")
        self.assertEqual(data["windows"][0]["room"], "living_room")
        self.assertEqual(data["daily_features"][0]["person_id"], "elder_001")
        self.assertEqual(data["daily_features"][0]["date"], "2026-07-16")


if __name__ == "__main__":
    unittest.main()
