from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.sit_stand_event_labels import (
    build_sit_stand_review_queue,
    validate_sit_stand_event_labels,
)


def _event_label(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "fall-risk-sit-stand-event-label-v1",
        "label_id": "sitstand_000000000000000000000001",
        "event_id": "sitstand_event_0000000000000001",
        "asset_id": "asset_1",
        "video_id": "video_1",
        "subject_id": "subject_1",
        "source_group_id": "source_1",
        "sample_group_id": "sample_1",
        "split_group_id": "split_1",
        "interval_type": "event",
        "transition_type": "sit_to_stand",
        "onset_time": 1.0,
        "offset_time": 3.0,
        "boundary_precision": "exact",
        "outcome": "completed",
        "attempt_count": 1,
        "attempt_intervals": [{"onset_time": 1.0, "offset_time": 3.0}],
        "seat_off_proxy": None,
        "seat_contact_proxy": None,
        "support_contact_proxy": None,
        "stabilization_time_proxy": None,
        "scene_region": "living_room",
        "visibility": "full",
        "quality_flags": [],
        "review_status": "double_reviewed",
        "reviewed_by": ["reviewer_a", "reviewer_b"],
        "eligibility": "eligible",
        "hard_negative_type": None,
        "source_action_label_ids": ["actionv3_000000000000000000000001"],
    }
    row.update(overrides)
    return row


class SitStandEventLabelTest(unittest.TestCase):
    def test_event_and_explicit_background_are_valid_only_with_review_evidence(self) -> None:
        event = _event_label()
        background = _event_label(
            label_id="sitstand_000000000000000000000002",
            event_id="sitstand_background_0000000000000002",
            interval_type="explicit_background",
            transition_type=None,
            outcome=None,
            attempt_count=0,
            attempt_intervals=[],
            onset_time=4.0,
            offset_time=8.0,
            hard_negative_type="controlled_bend",
        )
        reviews = [
            {"label_id": event["label_id"], "reviewer_id": "reviewer_a"},
            {"label_id": background["label_id"], "reviewer_id": "reviewer_a"},
        ]

        report = validate_sit_stand_event_labels([event, background], reviews)

        self.assertTrue(report["valid"])
        self.assertEqual(report["counts"]["event"], 1)
        self.assertEqual(report["counts"]["explicit_background"], 1)
        self.assertEqual(report["explicit_background_duration_sec"], 4.0)

    def test_absence_of_an_event_label_cannot_be_inferred_as_background(self) -> None:
        event = _event_label()
        report = validate_sit_stand_event_labels(
            [event], [{"label_id": event["label_id"], "reviewer_id": "reviewer_a"}]
        )

        self.assertEqual(report["counts"]["explicit_background"], 0)
        self.assertEqual(report["explicit_background_duration_sec"], 0.0)
        self.assertFalse(report["gates"]["fp_hour_denominator_available"])

    def test_background_cannot_carry_transition_or_attempt_semantics(self) -> None:
        row = _event_label(interval_type="explicit_background")

        with self.assertRaisesRegex(ValueError, "explicit_background"):
            validate_sit_stand_event_labels(
                [row], [{"label_id": row["label_id"], "reviewer_id": "reviewer_a"}]
            )

    def test_real_label_file_requires_a_matching_human_review_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "review log"):
            validate_sit_stand_event_labels([_event_label()], [])

    def test_attempt_intervals_must_be_inside_the_event(self) -> None:
        row = _event_label(
            attempt_intervals=[{"onset_time": 0.5, "offset_time": 2.0}]
        )

        with self.assertRaisesRegex(ValueError, "attempt interval"):
            validate_sit_stand_event_labels(
                [row], [{"label_id": row["label_id"], "reviewer_id": "reviewer_a"}]
            )

    def test_review_queue_is_deterministic_and_excludes_test(self) -> None:
        actions = [
            {
                "label_id": "action_train",
                "asset_id": "asset_train",
                "video_id": "video_train",
                "action_id": "A04",
                "start_frame": 10,
                "end_frame_exclusive": 30,
                "start_time": 1.0,
                "end_time_exclusive": 3.0,
                "subject_id": "subject_train",
                "source_group_id": "source_train",
                "sample_group_id": "sample_train",
            },
            {
                "label_id": "action_test",
                "asset_id": "asset_test",
                "video_id": "video_test",
                "action_id": "A03",
                "start_frame": 20,
                "end_frame_exclusive": 40,
                "start_time": 2.0,
                "end_time_exclusive": 4.0,
                "subject_id": "subject_test",
                "source_group_id": "source_test",
                "sample_group_id": "sample_test",
            },
        ]
        assignments = [
            {
                "label_id": "action_train",
                "partition": "validation",
                "split_group_id": "split_train",
            },
            {
                "label_id": "action_test",
                "partition": "test",
                "split_group_id": "split_test",
            },
        ]
        manifest = [
            {
                "asset_id": "asset_train",
                "video_id": "video_train",
                "dataset": "fall_detection_2017",
                "fps": 10.0,
                "frame_count": 100,
                "duration_sec": 10.0,
                "scene_region": "home",
                "eligibility": True,
            },
            {
                "asset_id": "asset_test",
                "video_id": "video_test",
                "dataset": "ntu_rgbd",
                "fps": 10.0,
                "frame_count": 100,
                "duration_sec": 10.0,
                "scene_region": "lab",
                "eligibility": True,
            },
        ]

        first = build_sit_stand_review_queue(actions, assignments, manifest)
        second = build_sit_stand_review_queue(actions, assignments, manifest)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["partition"], "validation")
        self.assertFalse(first[0]["test_pose_read"])
        self.assertLess(first[0]["context_start_time"], 1.0)


if __name__ == "__main__":
    unittest.main()
