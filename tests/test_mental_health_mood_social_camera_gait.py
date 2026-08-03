from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import unittest

from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_mood_social_camera_daily,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialActivity,
)


PERSON_A = "elder-camera-a"
PERSON_B = "elder-camera-b"
CAMERA_A = "camera-a"
CAMERA_B = "camera-b"
SCENE_A = "fixed-living-room-v1"
SCENE_B = "fixed-living-room-v2"
DAY_START = datetime.fromisoformat("2026-07-28T08:00:00+08:00")
GAIT_START = datetime.fromisoformat("2026-07-28T12:00:00+08:00")


def camera_window(
    offset_seconds: int = 0,
    *,
    person_id: str = PERSON_A,
    camera_id: str = CAMERA_A,
    scene_version: str = SCENE_A,
    active_score: float | None = 0.1,
    valid_detection_ratio: float = 1.0,
    data_quality: str = "valid",
) -> dict[str, object]:
    start = DAY_START + timedelta(seconds=offset_seconds)
    return {
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(seconds=10)).isoformat(),
        "person_id": person_id,
        "camera_id": camera_id,
        "scene_version": scene_version,
        "active_score": active_score,
        "valid_detection_ratio": valid_detection_ratio,
        "identity_confidence": 0.95,
        "pose_validity": 0.95,
        "tracking_confidence": 0.95,
        "data_quality": data_quality,
    }


def hip_gait_frame(
    offset_seconds: float,
    center_x: float,
    *,
    person_id: str = PERSON_A,
    camera_id: str = CAMERA_A,
    scene_version: str = SCENE_A,
    segment_id: str = "segment-a",
    track_id: str = "track-a",
) -> dict[str, object]:
    observed_at = GAIT_START + timedelta(seconds=offset_seconds)
    return {
        "person_id": person_id,
        "date": "2026-07-28",
        "observed_at": observed_at.isoformat(),
        "camera_id": camera_id,
        "scene_version": scene_version,
        "walking_segment_id": segment_id,
        "track_id": track_id,
        "data_quality": "valid",
        "keypoints": [
            {
                "name": "left_hip",
                "x": center_x - 0.01,
                "y": 0.45,
                "score": 0.99,
            },
            {
                "name": "right_hip",
                "x": center_x + 0.01,
                "y": 0.45,
                "score": 0.99,
            },
        ],
    }


def activity_from(
    windows: list[dict[str, object]],
    *,
    gait_records: list[dict[str, object]] | tuple[()] = (),
) -> dict[str, object]:
    results = aggregate_mood_social_camera_daily(
        windows,
        camera_gait_records=gait_records,
    )
    if len(results) != 1:
        raise AssertionError(f"expected one person-day, got {len(results)}")
    activity = results[0]["activity"]
    if not isinstance(activity, dict):
        raise AssertionError("activity output must be an object")
    return activity


class MoodSocialCameraGaitContractTest(unittest.TestCase):
    def assert_zero_coverage(self, activity: dict[str, object]) -> None:
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
        MoodSocialActivity.model_validate(activity)

    def test_camera_and_scene_are_required_for_windows_and_gait(self) -> None:
        for field in ("camera_id", "scene_version"):
            with self.subTest(source="window", field=field):
                invalid_window = camera_window()
                del invalid_window[field]
                with self.assertRaises(ValueError):
                    aggregate_mood_social_camera_daily([invalid_window])

            with self.subTest(source="gait", field=field):
                invalid_gait = hip_gait_frame(0, 0.20)
                del invalid_gait[field]
                with self.assertRaises(ValueError):
                    aggregate_mood_social_camera_daily(
                        [camera_window(active_score=0.5)],
                        camera_gait_records=[invalid_gait],
                    )

    def test_missing_score_or_subthreshold_detection_ratio_is_zero_coverage(
        self,
    ) -> None:
        cases = (
            camera_window(active_score=None, valid_detection_ratio=1.0),
            camera_window(active_score=0.1, valid_detection_ratio=0.5999),
        )
        for window in cases:
            with self.subTest(window=window):
                self.assert_zero_coverage(activity_from([window]))

    def test_invalid_window_breaks_one_hundred_eighty_low_activity_slots(
        self,
    ) -> None:
        windows = [
            camera_window(index * 10, active_score=0.1)
            for index in range(90)
        ]
        windows.append(
            camera_window(
                90 * 10,
                active_score=None,
                valid_detection_ratio=0.0,
                data_quality="offline",
            )
        )
        windows.extend(
            camera_window(index * 10, active_score=0.1)
            for index in range(91, 181)
        )

        activity = activity_from(windows)

        self.assertEqual(activity["valid_daytime_detection_minutes"], 30.0)
        self.assertEqual(activity["low_activity_minutes"], 30.0)
        self.assertEqual(activity["sedentary_bout_total_minutes"], 0.0)
        self.assertEqual(activity["longest_sedentary_bout_minutes"], 0.0)
        MoodSocialActivity.model_validate(activity)

    def test_different_people_in_the_same_slot_are_not_deduplicated(self) -> None:
        results = aggregate_mood_social_camera_daily(
            [
                camera_window(person_id=PERSON_A, active_score=0.5),
                camera_window(person_id=PERSON_B, active_score=0.5),
            ]
        )

        by_person = {str(result["person_id"]): result["activity"] for result in results}
        self.assertEqual(set(by_person), {PERSON_A, PERSON_B})
        for person_id, activity in by_person.items():
            with self.subTest(person_id=person_id):
                self.assertIsInstance(activity, dict)
                self.assertAlmostEqual(
                    activity["valid_daytime_detection_minutes"],
                    1 / 6,
                    places=4,
                )
                self.assertAlmostEqual(
                    activity["daytime_active_minutes"],
                    1 / 6,
                    places=4,
                )
                MoodSocialActivity.model_validate(activity)

    def test_raw_hip_frames_are_grouped_by_camera_scene_segment_and_track(
        self,
    ) -> None:
        gait_records = [
            # Scene A, segment A, track A: 0.1 and a later gap that must not pair.
            hip_gait_frame(0, 0.10),
            hip_gait_frame(1, 0.20),
            hip_gait_frame(3, 0.90),
            # Same scene and segment, different track: 0.2.
            hip_gait_frame(0, 0.60, track_id="track-b"),
            hip_gait_frame(1, 0.80, track_id="track-b"),
            # Same scene and track, different walking segment: 0.3.
            hip_gait_frame(0, 0.20, segment_id="segment-b"),
            hip_gait_frame(1, 0.50, segment_id="segment-b"),
            # Same camera, new scene: 0.5 and must remain a separate metric.
            hip_gait_frame(0, 0.10, scene_version=SCENE_B),
            hip_gait_frame(1, 0.60, scene_version=SCENE_B),
            # Different camera: 0.7 and must remain a separate metric.
            hip_gait_frame(0, 0.10, camera_id=CAMERA_B),
            hip_gait_frame(1, 0.80, camera_id=CAMERA_B),
        ]

        activity = activity_from(
            [camera_window(active_score=0.5)],
            gait_records=gait_records,
        )
        metrics = activity["camera_gait_metrics"]
        self.assertIsInstance(metrics, list)
        by_scene = {
            (metric["camera_id"], metric["scene_version"]): metric
            for metric in metrics
        }

        self.assertEqual(
            set(by_scene),
            {
                (CAMERA_A, SCENE_A),
                (CAMERA_A, SCENE_B),
                (CAMERA_B, SCENE_A),
            },
        )
        self.assertAlmostEqual(
            by_scene[(CAMERA_A, SCENE_A)][
                "gait_speed_image_norm_per_sec"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            by_scene[(CAMERA_A, SCENE_B)][
                "gait_speed_image_norm_per_sec"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            by_scene[(CAMERA_B, SCENE_A)][
                "gait_speed_image_norm_per_sec"
            ],
            0.7,
        )
        MoodSocialActivity.model_validate(activity)

    def test_metric_inputs_reject_mps_and_precomputed_personal_normalization(
        self,
    ) -> None:
        base = {
            "person_id": PERSON_A,
            "date": "2026-07-28",
            "observed_at": GAIT_START.isoformat(),
            "camera_id": CAMERA_A,
            "scene_version": SCENE_A,
            "gait_speed_image_norm_per_sec": 0.2,
        }
        for forbidden in ("gait_speed_mps", "walking_speed_norm"):
            with self.subTest(forbidden=forbidden):
                invalid = deepcopy(base)
                invalid[forbidden] = 0.3
                with self.assertRaises(ValueError):
                    aggregate_mood_social_camera_daily(
                        [camera_window(active_score=0.5)],
                        camera_gait_records=[invalid],
                    )

    def test_legacy_activity_fields_are_rejected_upstream(self) -> None:
        for forbidden in (
            "daily_step_count",
            "sedentary_total_minutes",
            "hourly_activity_vector",
        ):
            with self.subTest(forbidden=forbidden):
                invalid = camera_window(active_score=0.5)
                invalid[forbidden] = 1
                with self.assertRaises(ValueError):
                    aggregate_mood_social_camera_daily([invalid])

    def test_invalid_hip_frame_breaks_an_otherwise_adjacent_speed_pair(
        self,
    ) -> None:
        left = hip_gait_frame(0.0, 0.10)
        invalid = hip_gait_frame(0.4, 0.20)
        right = hip_gait_frame(0.8, 0.30)
        invalid["keypoints"] = [
            point
            for point in invalid["keypoints"]
            if point["name"] != "right_hip"
        ]

        activity = activity_from(
            [camera_window(active_score=0.5)],
            gait_records=[left, invalid, right],
        )

        self.assertEqual(activity["camera_gait_metrics"], [])
        MoodSocialActivity.model_validate(activity)

    def test_invalid_direct_metric_records_do_not_enter_daily_medians(self) -> None:
        valid = {
            "person_id": PERSON_A,
            "date": "2026-07-28",
            "observed_at": GAIT_START.isoformat(),
            "camera_id": CAMERA_A,
            "scene_version": SCENE_A,
            "gait_speed_image_norm_per_sec": 0.2,
            "sit_to_stand_duration_seconds": 3.0,
            "turn_duration_seconds": 2.0,
            "postural_stability": 0.8,
        }
        offline = deepcopy(valid)
        offline.update(
            {
                "gait_speed_image_norm_per_sec": 5.0,
                "sit_to_stand_duration_seconds": 20.0,
                "data_quality": "offline",
            }
        )
        rejected = deepcopy(valid)
        rejected.update(
            {
                "gait_speed_image_norm_per_sec": 7.0,
                "turn_duration_seconds": 30.0,
                "quality_rejected": True,
            }
        )

        activity = activity_from(
            [camera_window(active_score=0.5)],
            gait_records=[offline, valid, rejected],
        )
        [metric] = activity["camera_gait_metrics"]

        self.assertEqual(metric["gait_speed_image_norm_per_sec"], 0.2)
        self.assertEqual(metric["sit_to_stand_duration_seconds"], 3.0)
        self.assertEqual(metric["turn_duration_seconds"], 2.0)
        self.assertEqual(metric["postural_stability"], 0.8)

    def test_output_is_a_strict_mood_social_activity(self) -> None:
        activity = activity_from(
            [camera_window(active_score=0.5)],
            gait_records=[
                {
                    "person_id": PERSON_A,
                    "date": "2026-07-28",
                    "observed_at": GAIT_START.isoformat(),
                    "camera_id": CAMERA_A,
                    "scene_version": SCENE_A,
                    "gait_speed_image_norm_per_sec": 1.25,
                    "sit_to_stand_duration_seconds": 3.2,
                    "turn_duration_seconds": 2.4,
                    "postural_stability": 0.8,
                }
            ],
        )

        validated = MoodSocialActivity.model_validate(activity)
        self.assertEqual(validated.model_dump(mode="python"), activity)
        self.assertEqual(
            validated.camera_gait_metrics[0].gait_speed_image_norm_per_sec,
            1.25,
        )


if __name__ == "__main__":
    unittest.main()
