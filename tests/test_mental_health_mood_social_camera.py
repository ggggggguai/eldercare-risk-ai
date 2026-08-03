from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import unittest

from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_mood_social_camera_daily,
    aggregate_mood_social_camera_windows,
    extract_mood_social_camera_features,
)


PERSON_ID = "elder-camera-001"
CAMERA_A = "camera-a"
CAMERA_B = "camera-b"
SCENE_A = "fixed-living-room-v1"


def camera_window(
    start: str,
    *,
    score: float | None,
    camera_id: str = CAMERA_A,
    scene_version: str = SCENE_A,
    identity_confidence: float = 0.95,
    pose_validity: float = 0.95,
    tracking_confidence: float = 0.95,
    data_quality: str = "valid",
) -> dict[str, object]:
    window_start = datetime.fromisoformat(start)
    return {
        "window_start": window_start.isoformat(),
        "window_end": (window_start + timedelta(seconds=10)).isoformat(),
        "person_id": PERSON_ID,
        "camera_id": camera_id,
        "scene_version": scene_version,
        "active_score": score,
        "valid_detection_ratio": 1.0 if score is not None else 0.0,
        "identity_confidence": identity_confidence,
        "pose_validity": pose_validity,
        "tracking_confidence": tracking_confidence,
        "data_quality": data_quality,
    }


def consecutive_windows(
    start: str,
    count: int,
    *,
    score: float,
    skip_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    first = datetime.fromisoformat(start)
    skipped = skip_indices or set()
    return [
        camera_window(
            (first + timedelta(seconds=index * 10)).isoformat(),
            score=score,
        )
        for index in range(count)
        if index not in skipped
    ]


def raw_frame(observed_at: datetime, frame_id: int) -> dict[str, object]:
    return {
        "observed_at": observed_at.isoformat(),
        "frame_id": frame_id,
        "person_id": PERSON_ID,
        "camera_id": CAMERA_A,
        "scene_version": SCENE_A,
        "identity_confidence": 0.99,
        "bbox": [0.20, 0.20, 0.20, 0.40],
        "bbox_format": "xywh",
        "bbox_confidence": 0.99,
        "keypoints": [
            {"name": "left_shoulder", "x": 0.25, "y": 0.30, "score": 0.99},
            {"name": "right_shoulder", "x": 0.35, "y": 0.30, "score": 0.99},
            {"name": "left_hip", "x": 0.26, "y": 0.45, "score": 0.99},
            {"name": "right_hip", "x": 0.34, "y": 0.45, "score": 0.99},
            {"name": "left_knee", "x": 0.27, "y": 0.60, "score": 0.99},
            {"name": "right_knee", "x": 0.33, "y": 0.60, "score": 0.99},
            {"name": "left_ankle", "x": 0.28, "y": 0.78, "score": 0.99},
            {"name": "right_ankle", "x": 0.32, "y": 0.78, "score": 0.99},
        ],
        "keypoint_confidence": 0.99,
        "tracking_confidence": 0.99,
        "zone": "living_room",
        "room": "living_room",
        "posture": "sitting",
        "data_quality": "valid",
    }


def gait_record(
    *,
    camera_id: str,
    scene_version: str,
    gait_speed: float | None = None,
    sit_to_stand: float | None = None,
    turn_duration: float | None = None,
    postural_stability: float | None = None,
) -> dict[str, object]:
    return {
        "person_id": PERSON_ID,
        "date": "2026-07-28",
        "observed_at": "2026-07-28T12:00:00+08:00",
        "camera_id": camera_id,
        "scene_version": scene_version,
        "gait_speed_image_norm_per_sec": gait_speed,
        "sit_to_stand_duration_seconds": sit_to_stand,
        "turn_duration_seconds": turn_duration,
        "postural_stability": postural_stability,
    }


class MoodSocialCameraAggregationTest(unittest.TestCase):
    def activity(
        self,
        windows: list[dict[str, object]],
        *,
        gait: list[dict[str, object]] | tuple[()] = (),
    ) -> dict[str, object]:
        result = aggregate_mood_social_camera_daily(
            windows,
            camera_gait_records=gait,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["person_id"], PERSON_ID)
        self.assertEqual(result[0]["date"], "2026-07-28")
        self.assertIsInstance(result[0]["activity"], dict)
        return result[0]["activity"]

    def test_daytime_boundaries_use_half_open_shanghai_interval(self) -> None:
        activity = self.activity(
            [
                camera_window("2026-07-28T05:59:50+08:00", score=1.0),
                camera_window("2026-07-28T06:00:00+08:00", score=1.0),
                camera_window("2026-07-28T17:59:50+08:00", score=1.0),
                camera_window("2026-07-28T18:00:00+08:00", score=1.0),
            ]
        )

        self.assertAlmostEqual(activity["valid_daytime_detection_minutes"], 2 / 6, places=4)
        self.assertAlmostEqual(activity["daytime_active_minutes"], 2 / 6, places=4)
        self.assertAlmostEqual(activity["weighted_daytime_activity"], 2 / 6, places=4)
        self.assertAlmostEqual(activity["hourly_valid_detection_minutes"][6], 1 / 6, places=4)
        self.assertAlmostEqual(activity["hourly_valid_detection_minutes"][17], 1 / 6, places=4)
        self.assertEqual(activity["hourly_activity_intensity"][6], 1.0)
        self.assertEqual(activity["hourly_activity_intensity"][17], 1.0)
        self.assertEqual(activity["activity_peak_minute_of_day"], 360)
        for hour in (*range(0, 6), *range(18, 24)):
            self.assertEqual(activity["hourly_valid_detection_minutes"][hour], 0.0)
            self.assertIsNone(activity["hourly_activity_intensity"][hour])

    def test_preaggregated_windows_must_be_exactly_ten_seconds_and_epoch_aligned(self) -> None:
        valid = camera_window("2026-07-28T08:00:00+08:00", score=0.5)
        short = deepcopy(valid)
        short["window_end"] = "2026-07-28T08:00:09+08:00"
        cross_hour = camera_window("2026-07-28T07:59:55+08:00", score=0.5)

        for label, record in (("short", short), ("cross_hour", cross_hour)):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    aggregate_mood_social_camera_daily([record])

    def test_zero_coverage_has_null_semantics_and_empty_gait(self) -> None:
        invalid = camera_window(
            "2026-07-28T08:00:00+08:00",
            score=None,
            data_quality="offline",
        )
        activity = self.activity(
            [invalid],
            gait=[
                gait_record(
                    camera_id=CAMERA_A,
                    scene_version=SCENE_A,
                    gait_speed=0.5,
                )
            ],
        )

        self.assertEqual(activity["valid_daytime_detection_minutes"], 0.0)
        for field in (
            "daytime_active_minutes",
            "weighted_daytime_activity",
            "low_activity_minutes",
            "sedentary_bout_total_minutes",
            "longest_sedentary_bout_minutes",
            "activity_peak_minute_of_day",
        ):
            self.assertIsNone(activity[field], field)
        self.assertEqual(activity["hourly_valid_detection_minutes"], [0.0] * 24)
        self.assertEqual(activity["hourly_activity_intensity"], [None] * 24)
        self.assertEqual(activity["camera_gait_metrics"], [])

    def test_valid_stationary_observation_preserves_real_zero_values(self) -> None:
        activity = self.activity(
            [
                camera_window(
                    "2026-07-28T08:00:00+08:00",
                    score=0.0,
                )
            ]
        )

        self.assertAlmostEqual(
            activity["valid_daytime_detection_minutes"],
            1 / 6,
            places=4,
        )
        self.assertEqual(activity["daytime_active_minutes"], 0.0)
        self.assertEqual(activity["weighted_daytime_activity"], 0.0)
        self.assertAlmostEqual(activity["low_activity_minutes"], 1 / 6, places=4)
        self.assertEqual(activity["sedentary_bout_total_minutes"], 0.0)
        self.assertEqual(activity["longest_sedentary_bout_minutes"], 0.0)
        self.assertEqual(activity["hourly_activity_intensity"][8], 0.0)
        self.assertIsNone(activity["activity_peak_minute_of_day"])

    def test_sedentary_bout_uses_exact_thirty_minute_boundary(self) -> None:
        cases = (
            (179, 0.0),
            (180, 30.0),
        )
        for count, expected_bout in cases:
            with self.subTest(count=count):
                activity = self.activity(
                    consecutive_windows(
                        "2026-07-28T08:00:00+08:00",
                        count,
                        score=0.1,
                    )
                )
                self.assertAlmostEqual(activity["low_activity_minutes"], count / 6, places=4)
                self.assertAlmostEqual(
                    activity["sedentary_bout_total_minutes"],
                    expected_bout,
                    places=4,
                )
                self.assertAlmostEqual(
                    activity["longest_sedentary_bout_minutes"],
                    expected_bout,
                    places=4,
                )

    def test_missing_ten_second_slot_breaks_low_activity_continuity(self) -> None:
        activity = self.activity(
            consecutive_windows(
                "2026-07-28T08:00:00+08:00",
                180,
                score=0.1,
                skip_indices={90},
            )
        )

        self.assertAlmostEqual(activity["low_activity_minutes"], 179 / 6, places=4)
        self.assertEqual(activity["sedentary_bout_total_minutes"], 0.0)
        self.assertEqual(activity["longest_sedentary_bout_minutes"], 0.0)

    def test_hour_boundary_does_not_break_low_activity_bout(self) -> None:
        activity = self.activity(
            consecutive_windows(
                "2026-07-28T09:45:00+08:00",
                180,
                score=0.1,
            )
        )

        self.assertEqual(activity["sedentary_bout_total_minutes"], 30.0)
        self.assertEqual(activity["longest_sedentary_bout_minutes"], 30.0)
        self.assertEqual(activity["hourly_valid_detection_minutes"][9], 15.0)
        self.assertEqual(activity["hourly_valid_detection_minutes"][10], 15.0)

    def test_multicamera_quality_priority_is_stable_and_does_not_select_max_score(
        self,
    ) -> None:
        first = datetime.fromisoformat("2026-07-28T08:00:00+08:00")
        candidates: list[dict[str, object]] = []

        def pair(
            index: int,
            *,
            selected: dict[str, object],
            rejected: dict[str, object],
        ) -> None:
            start = (first + timedelta(seconds=index * 10)).isoformat()
            candidates.append(camera_window(start, **selected))
            candidates.append(camera_window(start, **rejected))

        pair(
            0,
            selected={
                "score": 0.1,
                "camera_id": CAMERA_A,
                "identity_confidence": 0.9,
                "pose_validity": 0.5,
                "tracking_confidence": 0.5,
            },
            rejected={
                "score": 0.9,
                "camera_id": CAMERA_B,
                "identity_confidence": 0.8,
                "pose_validity": 0.99,
                "tracking_confidence": 0.99,
            },
        )
        pair(
            1,
            selected={
                "score": 0.2,
                "camera_id": CAMERA_A,
                "identity_confidence": 0.9,
                "pose_validity": 0.9,
                "tracking_confidence": 0.1,
            },
            rejected={
                "score": 0.9,
                "camera_id": CAMERA_B,
                "identity_confidence": 0.9,
                "pose_validity": 0.8,
                "tracking_confidence": 0.99,
            },
        )
        pair(
            2,
            selected={
                "score": 0.3,
                "camera_id": CAMERA_A,
                "identity_confidence": 0.9,
                "pose_validity": 0.9,
                "tracking_confidence": 0.9,
            },
            rejected={
                "score": 0.9,
                "camera_id": CAMERA_B,
                "identity_confidence": 0.9,
                "pose_validity": 0.9,
                "tracking_confidence": 0.8,
            },
        )
        pair(
            3,
            selected={
                "score": 0.4,
                "camera_id": CAMERA_B,
                "identity_confidence": 0.8,
                "pose_validity": 0.8,
                "tracking_confidence": 0.8,
            },
            rejected={
                "score": None,
                "camera_id": CAMERA_A,
                "identity_confidence": 1.0,
                "pose_validity": 1.0,
                "tracking_confidence": 1.0,
                "data_quality": "offline",
            },
        )
        pair(
            4,
            selected={
                "score": 0.0,
                "camera_id": CAMERA_A,
                "identity_confidence": 0.9,
                "pose_validity": 0.9,
                "tracking_confidence": 0.9,
            },
            rejected={
                "score": 0.9,
                "camera_id": CAMERA_B,
                "identity_confidence": 0.9,
                "pose_validity": 0.9,
                "tracking_confidence": 0.9,
            },
        )

        forward = self.activity(candidates)
        reverse = self.activity(list(reversed(candidates)))

        self.assertEqual(forward, reverse)
        self.assertAlmostEqual(forward["valid_daytime_detection_minutes"], 5 / 6, places=4)
        self.assertAlmostEqual(forward["weighted_daytime_activity"], 1 / 6, places=4)
        self.assertAlmostEqual(forward["daytime_active_minutes"], 1 / 6, places=4)
        self.assertAlmostEqual(forward["low_activity_minutes"], 3 / 6, places=4)

    def test_hourly_values_reconstruct_daily_coverage_and_weighted_activity(self) -> None:
        activity = self.activity(
            [
                camera_window("2026-07-28T07:00:00+08:00", score=0.0),
                camera_window("2026-07-28T07:00:10+08:00", score=1.0),
                camera_window("2026-07-28T08:00:00+08:00", score=0.25),
            ]
        )

        self.assertAlmostEqual(activity["hourly_activity_intensity"][7], 0.5, places=6)
        self.assertAlmostEqual(activity["hourly_activity_intensity"][8], 0.25, places=6)
        reconstructed_valid = sum(activity["hourly_valid_detection_minutes"])
        reconstructed_weighted = sum(
            intensity * valid_minutes
            for intensity, valid_minutes in zip(
                activity["hourly_activity_intensity"],
                activity["hourly_valid_detection_minutes"],
            )
            if intensity is not None
        )
        self.assertAlmostEqual(
            reconstructed_valid,
            activity["valid_daytime_detection_minutes"],
            delta=0.1,
        )
        self.assertAlmostEqual(
            reconstructed_weighted,
            activity["weighted_daytime_activity"],
            delta=0.1,
        )

    def test_activity_peak_uses_minute_activity_mass_and_earliest_tie(self) -> None:
        activity = self.activity(
            [
                camera_window("2026-07-28T06:12:00+08:00", score=1.0),
                camera_window("2026-07-28T06:13:00+08:00", score=0.6),
                camera_window("2026-07-28T06:13:10+08:00", score=0.6),
                camera_window("2026-07-28T06:14:00+08:00", score=0.6),
                camera_window("2026-07-28T06:14:10+08:00", score=0.6),
            ]
        )
        all_zero = self.activity(
            [
                camera_window("2026-07-28T07:00:00+08:00", score=0.0),
                camera_window("2026-07-28T07:00:10+08:00", score=0.0),
            ]
        )

        self.assertEqual(activity["activity_peak_minute_of_day"], 373)
        self.assertIsNone(all_zero["activity_peak_minute_of_day"])

    def test_gait_is_preserved_for_nonselected_camera_and_isolated_by_scene(self) -> None:
        windows = [
            camera_window(
                "2026-07-28T12:00:00+08:00",
                score=0.5,
                camera_id=CAMERA_A,
                identity_confidence=0.99,
            ),
            camera_window(
                "2026-07-28T12:00:00+08:00",
                score=0.9,
                camera_id=CAMERA_B,
                scene_version="fixed-hall-v1",
                identity_confidence=0.90,
            ),
        ]
        gait = [
            gait_record(
                camera_id=CAMERA_A,
                scene_version="fixed-living-room-v1",
                gait_speed=0.2,
                sit_to_stand=2.0,
                turn_duration=3.0,
                postural_stability=0.8,
            ),
            gait_record(
                camera_id=CAMERA_A,
                scene_version="fixed-living-room-v1",
                gait_speed=0.4,
                sit_to_stand=4.0,
                turn_duration=5.0,
                postural_stability=0.6,
            ),
            gait_record(
                camera_id=CAMERA_A,
                scene_version="fixed-living-room-v2",
                gait_speed=1.2,
                sit_to_stand=6.0,
                turn_duration=7.0,
                postural_stability=0.4,
            ),
            gait_record(
                camera_id=CAMERA_A,
                scene_version="fixed-living-room-v2",
                gait_speed=1.4,
                sit_to_stand=8.0,
                turn_duration=9.0,
                postural_stability=0.2,
            ),
            gait_record(
                camera_id=CAMERA_B,
                scene_version="fixed-hall-v1",
                gait_speed=1.5,
            ),
        ]

        activity = self.activity(windows, gait=gait)
        metrics = activity["camera_gait_metrics"]
        pairs = [(item["camera_id"], item["scene_version"]) for item in metrics]
        self.assertEqual(
            pairs,
            [
                (CAMERA_A, "fixed-living-room-v1"),
                (CAMERA_A, "fixed-living-room-v2"),
                (CAMERA_B, "fixed-hall-v1"),
            ],
        )
        self.assertAlmostEqual(metrics[0]["gait_speed_image_norm_per_sec"], 0.3)
        self.assertAlmostEqual(metrics[0]["sit_to_stand_duration_seconds"], 3.0)
        self.assertAlmostEqual(metrics[0]["turn_duration_seconds"], 4.0)
        self.assertAlmostEqual(metrics[0]["postural_stability"], 0.7)
        self.assertAlmostEqual(metrics[1]["gait_speed_image_norm_per_sec"], 1.3)
        self.assertEqual(metrics[2]["gait_speed_image_norm_per_sec"], 1.5)

    def test_v3_activity_does_not_emit_legacy_or_forbidden_fields(self) -> None:
        [result] = extract_mood_social_camera_features(
            consecutive_windows(
                "2026-07-28T08:00:00+08:00",
                2,
                score=0.5,
            )
        )
        activity = result["activity"]

        for forbidden in (
            "daily_step_count",
            "sedentary_total_minutes",
            "sedentary_bouts_count",
            "hourly_activity_vector",
            "walking_speed_norm",
            "gait_speed_mps",
            "sit_to_stand_duration_norm",
            "turn_duration_norm",
        ):
            self.assertNotIn(forbidden, activity)


class MoodSocialCameraRawFrameTest(unittest.TestCase):
    def test_duplicate_frames_at_one_timestamp_cannot_forge_window_coverage(self) -> None:
        observed_at = datetime.fromisoformat("2026-07-28T08:00:00+08:00")
        records = [raw_frame(observed_at, frame_id) for frame_id in range(20)]

        windows = aggregate_mood_social_camera_windows(records)

        self.assertEqual(len(windows), 1)
        self.assertNotEqual(windows[0]["data_quality"], "valid")
        self.assertIsNone(windows[0]["active_score"])
        self.assertLess(windows[0]["valid_detection_ratio"], 0.60)

    def test_raw_frames_on_exact_ten_second_boundary_enter_new_slot(self) -> None:
        first = datetime.fromisoformat("2026-07-28T06:00:00+08:00")
        records = [
            raw_frame(first + timedelta(seconds=second), second)
            for second in range(20)
        ]

        windows = aggregate_mood_social_camera_windows(records)

        self.assertEqual(
            [item["window_start"] for item in windows],
            [
                "2026-07-28T06:00:00+08:00",
                "2026-07-28T06:00:10+08:00",
            ],
        )
        self.assertTrue(all(item["data_quality"] == "valid" for item in windows))

    def test_raw_active_score_uses_all_four_frozen_weights(self) -> None:
        first = datetime.fromisoformat("2026-07-28T08:00:00+08:00")
        records = []
        for second in range(10):
            record = raw_frame(first + timedelta(seconds=second), second)
            record["bbox"][0] += 0.01 * second
            for keypoint in record["keypoints"]:
                keypoint["x"] += 0.01 * second
            record["zone"] = "living_room" if second < 5 else "kitchen"
            record["room"] = record["zone"]
            record["posture"] = "sitting" if second < 5 else "standing"
            records.append(record)

        [window] = aggregate_mood_social_camera_windows(records)

        # center=.3, pose=.7, zone=1, posture=1:
        # .55*.3 + .30*.7 + .10*1 + .05*1 = .525
        self.assertEqual(window["data_quality"], "valid")
        self.assertAlmostEqual(window["active_score"], 0.525, places=4)


if __name__ == "__main__":
    unittest.main()
