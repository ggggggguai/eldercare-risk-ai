from __future__ import annotations

import math
import unittest

from elderly_monitoring.modules.mental_health.adapters import (
    MentalHealthDataError,
    adapt_behavior_record,
    adapt_sleep_record,
)
from elderly_monitoring.modules.mental_health.config import load_aggregation_config
from elderly_monitoring.modules.mental_health.config import aggregation_config_from_mapping


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


def behavior_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "person_id": "p01",
        "device_id": "cam-a",
        "frame_id": 1,
        "observed_at": "2026-07-14T09:00:00+08:00",
        "timestamp_sec": 0.0,
        "scene_region": "living_room",
        "keypoint_quality": 0.9,
        "keypoints": [
            {"name": name, "x": 0.2, "y": 0.3, "score": 0.9}
            for name in CORE_KEYPOINTS
        ],
    }
    record.update(overrides)
    return record


class MentalHealthAdapterTest(unittest.TestCase):
    def test_default_aggregation_config_is_loaded_from_yaml(self) -> None:
        config = load_aggregation_config()

        self.assertEqual(config.timezone, "Asia/Shanghai")
        self.assertEqual(config.max_gap_seconds, 5.0)
        self.assertEqual(config.timestamp_conflict_tolerance_seconds, 1.0)
        self.assertEqual(config.min_keypoint_quality, 0.45)
        self.assertEqual(config.min_common_core_keypoints, 4)
        self.assertEqual(config.active_motion_threshold, 0.02)
        self.assertEqual(config.night_start, "22:00")
        self.assertEqual(config.night_end, "06:00")

    def test_aggregation_config_rejects_invalid_timezone_and_thresholds(self) -> None:
        valid = {
            "timezone": "Asia/Shanghai",
            "max_gap_seconds": 5.0,
            "timestamp_conflict_tolerance_seconds": 1.0,
            "min_keypoint_quality": 0.45,
            "min_common_core_keypoints": 4,
            "active_motion_threshold": 0.02,
            "night_start": "22:00",
            "night_end": "06:00",
            "core_keypoints": list(CORE_KEYPOINTS),
        }
        for field, value in (
            ("timezone", "Mars/Olympus"),
            ("max_gap_seconds", 0),
            ("max_gap_seconds", "5.0"),
            ("max_gap_seconds", 10**400),
            ("min_keypoint_quality", 1.1),
            ("min_common_core_keypoints", 99),
            ("night_start", "22:0"),
        ):
            with self.subTest(field=field):
                invalid = {**valid, field: value}
                with self.assertRaisesRegex(ValueError, field):
                    aggregation_config_from_mapping(invalid)

        without_optional_core_list = dict(valid)
        without_optional_core_list.pop("core_keypoints")
        parsed = aggregation_config_from_mapping(without_optional_core_list)
        self.assertEqual(parsed.core_keypoints, CORE_KEYPOINTS)

    def test_observed_at_takes_precedence_and_is_converted_to_business_timezone(self) -> None:
        record = behavior_record(
            observed_at="2026-07-14T01:00:00Z",
            session_start_time="2026-07-14T00:59:59Z",
            timestamp_sec=1.0,
        )

        adapted = adapt_behavior_record(record, record_number=7)

        self.assertEqual(adapted.observed_at.isoformat(), "2026-07-14T09:00:00+08:00")
        self.assertTrue(adapted.usable_for_daily_aggregation)
        self.assertTrue(adapted.usable_for_valid_interval)
        self.assertNotIn("timestamp_conflict", adapted.data_quality_flags)

    def test_timestamp_conflict_keeps_observed_at_but_rejects_valid_duration(self) -> None:
        record = behavior_record(
            observed_at="2026-07-14T09:00:00+08:00",
            session_start_time="2026-07-14T08:59:00+08:00",
            timestamp_sec=0.0,
        )

        adapted = adapt_behavior_record(record, record_number=2)

        self.assertEqual(adapted.observed_at.isoformat(), "2026-07-14T09:00:00+08:00")
        self.assertTrue(adapted.usable_for_daily_aggregation)
        self.assertFalse(adapted.usable_for_valid_interval)
        self.assertIn("timestamp_conflict", adapted.data_quality_flags)

    def test_invalid_secondary_session_time_rejects_valid_duration(self) -> None:
        adapted = adapt_behavior_record(
            behavior_record(
                observed_at="2026-07-14T09:00:00+08:00",
                session_start_time="2026-07-14T09:00:00",
                timestamp_sec=0.0,
            )
        )

        self.assertEqual(adapted.observed_at.isoformat(), "2026-07-14T09:00:00+08:00")
        self.assertTrue(adapted.usable_for_daily_aggregation)
        self.assertFalse(adapted.usable_for_valid_interval)
        self.assertIn("invalid_session_time_source", adapted.data_quality_flags)

    def test_session_start_plus_relative_timestamp_produces_absolute_time(self) -> None:
        record = behavior_record(
            observed_at=None,
            session_start_time="2026-07-14T08:59:58+08:00",
            timestamp_sec=2.0,
        )

        adapted = adapt_behavior_record(record)

        self.assertEqual(adapted.observed_at.isoformat(), "2026-07-14T09:00:00+08:00")
        self.assertTrue(adapted.usable_for_daily_aggregation)

    def test_relative_timestamp_must_be_a_number_not_a_numeric_string(self) -> None:
        adapted = adapt_behavior_record(
            behavior_record(
                observed_at=None,
                session_start_time="2026-07-14T08:59:58+08:00",
                timestamp_sec="2.0",
            )
        )

        self.assertFalse(adapted.usable_for_daily_aggregation)
        self.assertIn("invalid_timestamp_sec", adapted.data_quality_flags)

    def test_overflowing_relative_timestamp_is_marked_unusable(self) -> None:
        adapted = adapt_behavior_record(
            behavior_record(
                observed_at=None,
                session_start_time="2026-07-14T08:59:58+08:00",
                timestamp_sec=1e20,
            )
        )

        self.assertIsNone(adapted.observed_at)
        self.assertFalse(adapted.usable_for_daily_aggregation)
        self.assertIn("invalid_timestamp_sec", adapted.data_quality_flags)

    def test_naive_or_relative_only_time_is_not_assigned_to_a_day(self) -> None:
        for record, expected_flag in (
            (behavior_record(observed_at="2026-07-14T09:00:00"), "timezone_missing"),
            (
                behavior_record(observed_at=None, session_start_time=None, timestamp_sec=2.0),
                "missing_absolute_time",
            ),
        ):
            with self.subTest(expected_flag=expected_flag):
                adapted = adapt_behavior_record(record)

                self.assertIsNone(adapted.observed_at)
                self.assertFalse(adapted.usable_for_daily_aggregation)
                self.assertIn(expected_flag, adapted.data_quality_flags)

    def test_track_id_cannot_substitute_for_stable_person_id(self) -> None:
        record = behavior_record(person_id=None, track_id=17)

        with self.assertRaisesRegex(
            MentalHealthDataError,
            r"behavior record 3.*person_id.*track_id",
        ):
            adapt_behavior_record(record, record_number=3)

    def test_non_normalized_or_non_finite_keypoints_are_rejected_from_valid_duration(self) -> None:
        for x_value in (120.0, math.nan):
            with self.subTest(x=x_value):
                keypoints = [
                    {"name": name, "x": x_value, "y": 0.3, "score": 0.9}
                    for name in CORE_KEYPOINTS
                ]
                adapted = adapt_behavior_record(behavior_record(keypoints=keypoints))

                self.assertTrue(adapted.usable_for_daily_aggregation)
                self.assertFalse(adapted.usable_for_valid_interval)
                self.assertIn("invalid_normalized_keypoints", adapted.data_quality_flags)

    def test_existing_pose_quality_flags_are_preserved_and_rejection_is_honored(self) -> None:
        adapted = adapt_behavior_record(
            behavior_record(
                quality_flags=["occluded_view", "pose_quality_rejected"],
            )
        )

        self.assertFalse(adapted.usable_for_valid_interval)
        self.assertIn("occluded_view", adapted.data_quality_flags)
        self.assertIn("pose_quality_rejected", adapted.data_quality_flags)

    def test_direct_observed_at_does_not_require_relative_timestamp(self) -> None:
        record = behavior_record(timestamp_sec=None)

        adapted = adapt_behavior_record(record)

        self.assertTrue(adapted.usable_for_daily_aggregation)
        self.assertTrue(adapted.usable_for_valid_interval)

    def test_sleep_adapter_preserves_missing_metrics_and_localizes_timestamp_date(self) -> None:
        adapted = adapt_sleep_record(
            {
                "person_id": "p01",
                "timestamp": "2026-07-14T16:30:00Z",
                "sleep_onset_latency": None,
                "night_awakenings": 2,
                "quality_score": 0.8,
                "quality_flags": ["device_partial"],
            },
            record_number=4,
        )

        self.assertEqual(adapted["date"], "2026-07-15")
        self.assertIsNone(adapted["sleep_onset_latency"])
        self.assertEqual(adapted["night_awakenings"], 2)
        self.assertIsNone(adapted["sleep_efficiency"])
        self.assertEqual(adapted["quality_score"], 0.8)
        self.assertIn("missing_sleep_onset_latency", adapted["quality_flags"])
        self.assertIn("missing_sleep_efficiency", adapted["quality_flags"])

    def test_sleep_adapter_rejects_invalid_units_ranges_and_types_with_field_context(self) -> None:
        invalid_values = (
            ("sleep_onset_latency", -1),
            ("sleep_onset_latency", 721),
            ("sleep_onset_latency", 10**400),
            ("sleep_onset_latency", "20"),
            ("night_awakenings", 1.5),
            ("night_awakenings", 101),
            ("sleep_efficiency", 85),
            ("sleep_efficiency", "0.85"),
            ("sleep_efficiency", math.nan),
            ("quality_score", 1.1),
            ("quality_score", "0.8"),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                record = {
                    "person_id": "p01",
                    "date": "2026-07-14",
                    "sleep_onset_latency": 20,
                    "night_awakenings": 2,
                    "sleep_efficiency": 0.85,
                    field: value,
                }

                with self.assertRaisesRegex(
                    MentalHealthDataError,
                    rf"sleep record 6.*{field}",
                ):
                    adapt_sleep_record(record, record_number=6)

    def test_sleep_adapter_requires_person_and_strict_date(self) -> None:
        for record, field in (
            ({"date": "2026-07-14"}, "person_id"),
            ({"person_id": "p01", "date": "2026-7-14"}, "date"),
            ({"person_id": "p01"}, "date"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(MentalHealthDataError, field):
                    adapt_sleep_record(record)

    def test_sleep_timestamp_localization_overflow_has_field_context(self) -> None:
        with self.assertRaisesRegex(
            MentalHealthDataError,
            r"sleep record 9.*timestamp",
        ):
            adapt_sleep_record(
                {
                    "person_id": "p01",
                    "timestamp": "0001-01-01T00:00:00+14:00",
                },
                record_number=9,
            )


if __name__ == "__main__":
    unittest.main()
