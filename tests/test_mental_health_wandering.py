from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta

from elderly_monitoring.modules.mental_health.feature_extraction.wandering import (
    WanderingDetectionConfig,
    aggregate_daily_wandering,
    detect_wandering_events,
)


def point(second: int, *, x: float, y: float, confidence: float = 0.95) -> dict[str, object]:
    timestamp = datetime.fromisoformat("2026-07-16T22:10:00+08:00") + timedelta(seconds=second)
    return {
        "timestamp": timestamp.isoformat(),
        "person_id": "elder_001",
        "track_id": "track_001",
        "camera_id": "living_room_cam",
        "x": x,
        "y": y,
        "det_confidence": confidence,
        "is_interpolated": False,
    }


def pacing_points() -> list[dict[str, object]]:
    records = []
    for second in range(150):
        x = 200.0 + 90.0 * math.sin(2 * math.pi * second / 30.0)
        records.append(point(second, x=x, y=240.0))
    return records


def lapping_points() -> list[dict[str, object]]:
    records = []
    for second in range(150):
        angle = 2 * math.pi * second / 40.0
        records.append(point(second, x=320.0 + 80.0 * math.cos(angle), y=240.0 + 80.0 * math.sin(angle)))
    return records


class MentalHealthWanderingTest(unittest.TestCase):
    def test_direct_movement_stays_normal(self) -> None:
        records = [point(second, x=100.0 + second * 2.0, y=240.0) for second in range(120)]

        events = detect_wandering_events(records, include_normal=True)

        self.assertTrue(events)
        self.assertTrue(all(event["decision"] == "normal_movement" for event in events))
        self.assertTrue(all(event["diagnosis"] is False for event in events))

    def test_pacing_is_recorded_as_wandering_event(self) -> None:
        events = detect_wandering_events(pacing_points())
        event = max(events, key=lambda item: item["wandering_score"])

        self.assertEqual(event["decision"], "record_as_wandering_event")
        self.assertEqual(event["wandering_type"], "pacing")
        self.assertGreaterEqual(event["turn_count"], 4)
        self.assertLessEqual(event["path_efficiency"], 0.35)
        self.assertTrue(event["is_night"])

    def test_lapping_is_identified_from_closed_path_shape(self) -> None:
        events = detect_wandering_events(lapping_points())
        event = max(events, key=lambda item: item["loop_score"])

        self.assertEqual(event["decision"], "record_as_wandering_event")
        self.assertEqual(event["wandering_type"], "lapping")
        self.assertGreaterEqual(event["loop_score"], 0.6)

    def test_low_quality_track_never_triggers_confirmed_event(self) -> None:
        records = [dict(item, det_confidence=0.20, is_interpolated=True) for item in pacing_points()]

        events = detect_wandering_events(records)

        self.assertTrue(events)
        self.assertTrue(all(event["decision"] == "record_as_low_confidence_candidate" for event in events))
        self.assertTrue(all("low_detection_confidence" in event["quality_flags"] for event in events))

    def test_daily_aggregation_counts_night_events_and_baseline_shift(self) -> None:
        history = [
            {"person_id": "elder_001", "date": "2026-07-13", "night_wandering_count": 0},
            {"person_id": "elder_001", "date": "2026-07-14", "night_wandering_count": 1},
            {"person_id": "elder_001", "date": "2026-07-15", "night_wandering_count": 1},
        ]
        config = replace(WanderingDetectionConfig(), step_seconds=120)
        events = detect_wandering_events(pacing_points(), config=config)

        [daily] = aggregate_daily_wandering(events, history_daily_features=history, config=config)

        self.assertEqual(daily["date"], "2026-07-16")
        self.assertGreaterEqual(daily["night_wandering_count"], 1)
        self.assertGreaterEqual(daily["pacing_count"], 1)
        self.assertEqual(daily["consecutive_nights_with_wandering"], 3)
        self.assertIsNotNone(daily["wandering_baseline_sigma"])
        self.assertFalse(daily["diagnosis"])

    def test_confirmed_roi_hits_add_explainable_risk_features(self) -> None:
        rois = [
            {
                "roi_id": "doorway_01",
                "type": "doorway",
                "roi_version_id": "roi_v3",
                "device_id": "living_room_cam",
                "roi_quality": "confirmed",
                "shape": {
                    "type": "polygon",
                    "points": [
                        {"x": 0.12, "y": 0.45},
                        {"x": 0.88, "y": 0.45},
                        {"x": 0.88, "y": 0.55},
                        {"x": 0.12, "y": 0.55},
                    ],
                    "bbox": {"x1": 0.12, "y1": 0.45, "x2": 0.88, "y2": 0.55},
                },
            }
        ]
        records = [
            point(second, x=200.0 + 90.0 * math.sin(2 * math.pi * second / 30.0), y=240.0)
            for second in range(150)
        ]

        events = detect_wandering_events(records, roi_annotations=rois)
        event = max(events, key=lambda item: item["wandering_score"])

        self.assertEqual(event["event_type"], "wandering")
        self.assertEqual(event["roi_version_id"], "roi_v3")
        self.assertIn("doorway_01", event["roi_hits"])
        self.assertGreaterEqual(event["doorway_hover_seconds"], 60)
        self.assertTrue(event["high_risk_roi_hit"])
        self.assertEqual(event["roi_quality"], "confirmed")

    def test_ignore_area_dominant_window_does_not_trigger_event(self) -> None:
        records = [
            dict(item, roi_id="ignore_01", roi_type="ignore_area", roi_quality="confirmed")
            for item in pacing_points()
        ]

        events = detect_wandering_events(records, include_normal=True)

        self.assertTrue(events)
        self.assertTrue(all(event["decision"] == "normal_movement" for event in events))
        self.assertTrue(any("ignore_area_dominant" in event["quality_flags"] for event in events))


if __name__ == "__main__":
    unittest.main()
