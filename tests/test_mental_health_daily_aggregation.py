from __future__ import annotations

from dataclasses import replace
import unittest

from elderly_monitoring.modules.mental_health.config import load_aggregation_config
from elderly_monitoring.modules.mental_health.daily_aggregation import aggregate_daily_behavior


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


def pose_record(
    *,
    person_id: str = "p01",
    device_id: str | None = "cam-a",
    frame_id: int = 1,
    observed_at: str | None = "2026-07-14T09:00:00+08:00",
    x: float = 0.20,
    quality: float = 0.90,
    scene_region: str = "living_room",
    keypoint_names: tuple[str, ...] = CORE_KEYPOINTS,
    **extra: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "person_id": person_id,
        "device_id": device_id,
        "frame_id": frame_id,
        "observed_at": observed_at,
        "timestamp_sec": float(frame_id - 1),
        "scene_region": scene_region,
        "keypoint_quality": quality,
        "keypoints": [
            {"name": name, "x": x, "y": 0.30, "score": quality}
            for name in keypoint_names
        ],
    }
    record.update(extra)
    return record


def by_person_and_date(outputs: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(item["person_id"]), str(item["date"])): item for item in outputs}


class MentalHealthDailyAggregationTest(unittest.TestCase):
    def test_elapsed_seconds_use_absolute_time_across_dst_spring_forward(self) -> None:
        config = replace(load_aggregation_config(), timezone="America/New_York")

        [daily] = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-03-08T01:59:59-05:00",
                    x=0.20,
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-03-08T03:00:01-04:00",
                    x=0.24,
                ),
            ],
            config=config,
        )

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["valid_observation_seconds"], 2.0)
        self.assertNotIn("large_observation_gap", daily["data_quality_flags"])

    def test_absolute_order_is_preserved_across_dst_fall_back(self) -> None:
        config = replace(load_aggregation_config(), timezone="America/New_York")

        [daily] = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-11-01T01:59:59-04:00",
                    x=0.20,
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-11-01T01:00:01-05:00",
                    x=0.24,
                ),
            ],
            config=config,
        )

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["start_time"], "2026-11-01T01:59:59-04:00")
        self.assertEqual(daily["end_time"], "2026-11-01T01:00:01-05:00")
        self.assertEqual(daily["nighttime_activity_ratio"], 1.0)

    def test_sorts_deduplicates_and_never_connects_different_people(self) -> None:
        p1_first = pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.20)
        p1_second = pose_record(frame_id=2, observed_at="2026-07-14T09:00:02+08:00", x=0.24)
        p1_third = pose_record(frame_id=3, observed_at="2026-07-14T09:00:04+08:00", x=0.28)
        p2_first = pose_record(
            person_id="p02", frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.40
        )
        p2_second = pose_record(
            person_id="p02", frame_id=2, observed_at="2026-07-14T09:00:02+08:00", x=0.40
        )

        outputs = aggregate_daily_behavior(
            [p1_third, p2_second, p1_second, p1_first, dict(p1_second), p2_first]
        )
        indexed = by_person_and_date(outputs)

        p1 = indexed[("p01", "2026-07-14")]
        self.assertEqual(p1["observation_seconds"], 4.0)
        self.assertEqual(p1["valid_observation_seconds"], 4.0)
        self.assertAlmostEqual(p1["activity_volume"], 0.08)
        self.assertEqual(p1["active_ratio"], 1.0)
        self.assertEqual(p1["observation_coverage"], 1.0)
        self.assertIn("duplicate_observation", p1["data_quality_flags"])

        p2 = indexed[("p02", "2026-07-14")]
        self.assertEqual(p2["observation_seconds"], 2.0)
        self.assertEqual(p2["activity_volume"], 0.0)
        self.assertEqual(p2["active_ratio"], 0.0)

    def test_microsecond_interval_does_not_round_duration_to_zero(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-07-14T09:00:00.000000+08:00",
                    x=0.20,
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:00.000040+08:00",
                    x=0.2001,
                ),
            ]
        )

        self.assertEqual(daily["observation_seconds"], 0.00004)
        self.assertEqual(daily["valid_observation_seconds"], 0.00004)
        self.assertEqual(daily["observation_coverage"], 1.0)
        self.assertEqual(daily["active_ratio"], 1.0)

    def test_cross_midnight_interval_is_split_into_natural_days_and_nighttime(self) -> None:
        outputs = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-07-14T23:59:58+08:00",
                    x=0.20,
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-15T00:00:02+08:00",
                    x=0.28,
                ),
            ]
        )
        indexed = by_person_and_date(outputs)

        for date in ("2026-07-14", "2026-07-15"):
            with self.subTest(date=date):
                daily = indexed[("p01", date)]
                self.assertEqual(daily["observation_seconds"], 2.0)
                self.assertEqual(daily["valid_observation_seconds"], 2.0)
                self.assertAlmostEqual(daily["activity_volume"], 0.04)
                self.assertEqual(daily["active_ratio"], 1.0)
                self.assertEqual(daily["nighttime_activity_ratio"], 1.0)

        self.assertEqual(
            indexed[("p01", "2026-07-14")]["end_time"],
            "2026-07-15T00:00:00+08:00",
        )
        self.assertEqual(
            indexed[("p01", "2026-07-15")]["start_time"],
            "2026-07-15T00:00:00+08:00",
        )

    def test_nighttime_ratio_uses_only_overlap_with_configured_night_window(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(frame_id=1, observed_at="2026-07-14T21:59:58+08:00", x=0.20),
                pose_record(frame_id=2, observed_at="2026-07-14T22:00:02+08:00", x=0.28),
            ]
        )

        self.assertEqual(daily["observation_seconds"], 4.0)
        self.assertEqual(daily["active_ratio"], 1.0)
        self.assertEqual(daily["nighttime_activity_ratio"], 1.0)

    def test_large_gap_is_not_counted_as_duration_activity_or_scene_transition(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-07-14T09:00:00+08:00",
                    x=0.20,
                    scene_region="living_room",
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:06+08:00",
                    x=0.40,
                    scene_region="kitchen",
                ),
            ]
        )

        self.assertEqual(daily["observation_seconds"], 0.0)
        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIsNone(daily["activity_volume"])
        self.assertIsNone(daily["active_ratio"])
        self.assertIsNone(daily["observation_coverage"])
        self.assertEqual(daily["scene_transition_count"], 0)
        self.assertIn("large_observation_gap", daily["data_quality_flags"])

    def test_low_quality_endpoint_counts_observation_but_not_valid_activity(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.20),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:02+08:00",
                    x=0.28,
                    quality=0.30,
                    quality_state="low_quality",
                ),
            ]
        )

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIsNone(daily["activity_volume"])
        self.assertIsNone(daily["active_ratio"])
        self.assertEqual(daily["observation_coverage"], 0.0)
        self.assertIn("insufficient_pose_quality", daily["data_quality_flags"])

    def test_timestamp_conflict_does_not_enter_valid_duration(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.20),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:02+08:00",
                    session_start_time="2026-07-14T08:59:00+08:00",
                    timestamp_sec=2.0,
                    x=0.28,
                ),
            ]
        )

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIn("timestamp_conflict", daily["data_quality_flags"])

    def test_insufficient_common_core_keypoints_rejects_interval(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.20),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:02+08:00",
                    x=0.28,
                    keypoint_names=CORE_KEYPOINTS[:3],
                ),
            ]
        )

        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIn("insufficient_common_core_keypoints", daily["data_quality_flags"])

    def test_scene_distribution_uses_valid_seconds_and_transitions_are_continuous(self) -> None:
        [daily] = aggregate_daily_behavior(
            [
                pose_record(
                    frame_id=1,
                    observed_at="2026-07-14T09:00:00+08:00",
                    scene_region="living_room",
                ),
                pose_record(
                    frame_id=2,
                    observed_at="2026-07-14T09:00:02+08:00",
                    scene_region="kitchen",
                ),
                pose_record(
                    frame_id=3,
                    observed_at="2026-07-14T09:00:04+08:00",
                    scene_region="kitchen",
                ),
            ]
        )

        self.assertEqual(
            daily["scene_region_distribution"],
            {"kitchen": 0.5, "living_room": 0.5},
        )
        self.assertEqual(daily["scene_transition_count"], 1)

    def test_overlapping_devices_choose_quality_then_stable_device_without_double_counting(self) -> None:
        records = [
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                x=0.20,
                scene_region="bedroom",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                x=0.28,
                scene_region="hall",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                x=0.20,
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                x=0.28,
                scene_region="kitchen",
            ),
        ]

        [forward] = aggregate_daily_behavior(records)
        [reverse] = aggregate_daily_behavior(list(reversed(records)))

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["observation_seconds"], 2.0)
        self.assertEqual(forward["valid_observation_seconds"], 2.0)
        self.assertEqual(forward["scene_region_distribution"], {"living_room": 1.0})
        self.assertEqual(forward["scene_transition_count"], 0)
        self.assertIn("overlapping_device_observations", forward["data_quality_flags"])

    def test_higher_quality_device_wins_overlapping_interval(self) -> None:
        records = [
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                quality=0.70,
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                quality=0.70,
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                quality=0.95,
                scene_region="bedroom",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                quality=0.95,
                scene_region="bedroom",
            ),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["scene_region_distribution"], {"bedroom": 1.0})

    def test_higher_quality_simultaneous_single_frame_is_resolved_before_intervals(self) -> None:
        records = [
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                quality=0.95,
                scene_region="bedroom",
                x=0.20,
            ),
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                quality=0.70,
                scene_region="living_room",
                x=0.20,
            ),
            pose_record(
                device_id="cam-b",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                quality=0.70,
                scene_region="living_room",
                x=0.28,
            ),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["observation_seconds"], 0.0)
        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIsNone(daily["activity_volume"])
        self.assertEqual(daily["scene_region_distribution"], {})
        self.assertIn("overlapping_device_observations", daily["data_quality_flags"])

    def test_partial_device_overlap_never_creates_scene_transition(self) -> None:
        records = [
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                quality=0.80,
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=2,
                observed_at="2026-07-14T09:00:04+08:00",
                quality=0.80,
                scene_region="kitchen",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:01+08:00",
                quality=0.95,
                scene_region="bedroom",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=2,
                observed_at="2026-07-14T09:00:03+08:00",
                quality=0.95,
                scene_region="bedroom",
            ),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["observation_seconds"], 4.0)
        self.assertEqual(daily["scene_transition_count"], 0)
        self.assertIn("overlapping_device_observations", daily["data_quality_flags"])

    def test_missing_device_id_is_rejected_when_named_device_records_are_present(self) -> None:
        records = [
            pose_record(device_id="cam-a", frame_id=1),
            pose_record(
                device_id=None,
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "device_id.*cross-device"):
            aggregate_daily_behavior(records)

    def test_simultaneous_single_observations_from_devices_are_flagged_without_duration(self) -> None:
        records = [
            pose_record(device_id="cam-a", frame_id=1),
            pose_record(device_id="cam-b", frame_id=1),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["observation_seconds"], 0.0)
        self.assertIn("overlapping_device_observations", daily["data_quality_flags"])

    def test_simultaneous_device_observation_at_transition_endpoint_suppresses_transition(self) -> None:
        records = [
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                scene_region="kitchen",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:02+08:00",
                scene_region="bedroom",
            ),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["scene_transition_count"], 0)
        self.assertIn("overlapping_device_observations", daily["data_quality_flags"])

    def test_single_foreign_device_point_inside_interval_is_overlap(self) -> None:
        records = [
            pose_record(
                device_id="cam-a",
                frame_id=1,
                observed_at="2026-07-14T09:00:00+08:00",
                scene_region="living_room",
            ),
            pose_record(
                device_id="cam-a",
                frame_id=2,
                observed_at="2026-07-14T09:00:02+08:00",
                scene_region="kitchen",
            ),
            pose_record(
                device_id="cam-b",
                frame_id=1,
                observed_at="2026-07-14T09:00:01+08:00",
                scene_region="bedroom",
            ),
        ]

        [daily] = aggregate_daily_behavior(records)

        self.assertEqual(daily["observation_seconds"], 2.0)
        self.assertEqual(daily["scene_transition_count"], 0)
        self.assertIn("overlapping_device_observations", daily["data_quality_flags"])

    def test_overlap_flag_is_limited_to_days_with_overlapping_observations(self) -> None:
        records = [
            pose_record(device_id="cam-a", frame_id=1, observed_at="2026-07-14T09:00:00+08:00"),
            pose_record(device_id="cam-a", frame_id=2, observed_at="2026-07-14T09:00:02+08:00"),
            pose_record(device_id="cam-b", frame_id=1, observed_at="2026-07-14T09:00:00+08:00"),
            pose_record(device_id="cam-b", frame_id=2, observed_at="2026-07-14T09:00:02+08:00"),
            pose_record(device_id="cam-a", frame_id=3, observed_at="2026-07-15T09:00:00+08:00"),
            pose_record(device_id="cam-a", frame_id=4, observed_at="2026-07-15T09:00:02+08:00"),
        ]

        outputs = by_person_and_date(aggregate_daily_behavior(records))

        self.assertIn(
            "overlapping_device_observations",
            outputs[("p01", "2026-07-14")]["data_quality_flags"],
        )
        self.assertNotIn(
            "overlapping_device_observations",
            outputs[("p01", "2026-07-15")]["data_quality_flags"],
        )

    def test_motion_proxy_uses_median_common_keypoint_displacement(self) -> None:
        first = pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00", x=0.20)
        second = pose_record(frame_id=2, observed_at="2026-07-14T09:00:02+08:00", x=0.24)
        second_keypoints = list(second["keypoints"])
        second_keypoints[0] = {**second_keypoints[0], "x": 0.90}
        second["keypoints"] = second_keypoints

        [daily] = aggregate_daily_behavior([first, second])

        self.assertAlmostEqual(daily["activity_volume"], 0.04)
        self.assertEqual(daily["active_ratio"], 1.0)

    def test_low_score_core_keypoint_is_not_counted_as_common_visible_point(self) -> None:
        first = pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00")
        second = pose_record(frame_id=2, observed_at="2026-07-14T09:00:02+08:00")
        second_keypoints = list(second["keypoints"])
        for index in range(5):
            second_keypoints[index] = {**second_keypoints[index], "score": 0.0}
        second["keypoints"] = second_keypoints

        [daily] = aggregate_daily_behavior([first, second])

        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIn("insufficient_common_core_keypoints", daily["data_quality_flags"])

    def test_core_keypoints_without_score_or_upstream_valid_flag_are_not_visible(self) -> None:
        first = pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00")
        second = pose_record(frame_id=2, observed_at="2026-07-14T09:00:02+08:00", x=0.30)
        for record in (first, second):
            record["keypoints"] = [
                {"name": point["name"], "x": point["x"], "y": point["y"]}
                for point in record["keypoints"]
            ]

        [daily] = aggregate_daily_behavior([first, second])

        self.assertEqual(daily["valid_observation_seconds"], 0.0)
        self.assertIsNone(daily["activity_volume"])
        self.assertIn("insufficient_common_core_keypoints", daily["data_quality_flags"])

    def test_conflicting_duplicate_payload_has_input_order_independent_result(self) -> None:
        first = pose_record(
            frame_id=1,
            observed_at="2026-07-14T09:00:00+08:00",
            scene_region="living_room",
        )
        conflicting = pose_record(
            frame_id=1,
            observed_at="2026-07-14T09:00:00+08:00",
            scene_region="bedroom",
        )
        end = pose_record(
            frame_id=2,
            observed_at="2026-07-14T09:00:02+08:00",
            scene_region="living_room",
        )

        [forward] = aggregate_daily_behavior([first, conflicting, end])
        [reverse] = aggregate_daily_behavior([conflicting, first, end])

        self.assertEqual(forward, reverse)

    def test_relative_only_records_are_never_guessed_into_a_date(self) -> None:
        records = [
            pose_record(
                frame_id=1,
                observed_at=None,
                timestamp_sec=0.0,
                session_start_time=None,
            ),
            pose_record(
                frame_id=2,
                observed_at=None,
                timestamp_sec=2.0,
                session_start_time=None,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "absolute event time"):
            aggregate_daily_behavior(records)

    def test_unassignable_time_record_is_skipped_without_polluting_other_days(self) -> None:
        records = [
            pose_record(frame_id=1, observed_at="2026-07-14T09:00:00+08:00"),
            pose_record(frame_id=2, observed_at="2026-07-14T09:00:02+08:00"),
            pose_record(
                device_id=None,
                frame_id=3,
                observed_at=None,
                session_start_time=None,
                timestamp_sec=0.0,
            ),
            pose_record(frame_id=4, observed_at="2026-07-15T09:00:00+08:00"),
            pose_record(frame_id=5, observed_at="2026-07-15T09:00:02+08:00"),
        ]

        outputs = by_person_and_date(aggregate_daily_behavior(records))

        for date in ("2026-07-14", "2026-07-15"):
            with self.subTest(date=date):
                daily = outputs[("p01", date)]
                self.assertEqual(daily["observation_seconds"], 2.0)
                self.assertNotIn("missing_absolute_time", daily["data_quality_flags"])


if __name__ == "__main__":
    unittest.main()
