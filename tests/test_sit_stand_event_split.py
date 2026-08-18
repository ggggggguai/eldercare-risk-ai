from __future__ import annotations

import unittest

from elderly_monitoring.modules.fall_risk.sit_stand_event_split import (
    build_sit_stand_event_split,
)


HARD_NEGATIVES = (
    "controlled_bend",
    "controlled_squat",
    "fall_or_rapid_descent",
    "bed_transfer_or_lying",
)


def _fixture(group_count: int = 70) -> tuple[list[dict], list[dict], list[dict]]:
    labels: list[dict] = []
    assignments: list[dict] = []
    manifest: list[dict] = []
    label_index = 0
    for group_index in range(group_count):
        video_id = f"video_{group_index}"
        manifest.append({"video_id": video_id, "dataset": "fall_detection_2017"})
        shared = {
            "video_id": video_id,
            "subject_id": f"subject_{group_index}",
            "source_group_id": f"source_{group_index}",
            "sample_group_id": f"sample_{group_index}",
            "split_group_id": f"source_split_{group_index}",
            "eligibility": "eligible",
        }
        for direction in ("sit_to_stand", "stand_to_sit"):
            label_index += 1
            labels.append(
                {
                    **shared,
                    "label_id": f"label_{label_index}",
                    "event_id": f"event_{label_index}",
                    "interval_type": "event",
                    "transition_type": direction,
                    "onset_time": 1.0,
                    "offset_time": 2.0,
                    "source_action_label_ids": [f"action_{label_index}"],
                }
            )
            assignments.append(
                {
                    "label_id": f"action_{label_index}",
                    "partition": "train" if group_index % 2 else "validation",
                }
            )
        for hard_negative in HARD_NEGATIVES:
            label_index += 1
            labels.append(
                {
                    **shared,
                    "label_id": f"label_{label_index}",
                    "event_id": f"event_{label_index}",
                    "interval_type": "explicit_background",
                    "transition_type": None,
                    "hard_negative_type": hard_negative,
                    "onset_time": 2.0,
                    "offset_time": 3.0,
                    "source_action_label_ids": [f"action_{label_index}"],
                }
            )
            assignments.append(
                {
                    "label_id": f"action_{label_index}",
                    "partition": "train" if group_index % 2 else "validation",
                }
            )
    return labels, assignments, manifest


class SitStandEventSplitTest(unittest.TestCase):
    def test_build_is_deterministic_and_satisfies_development_gate(self) -> None:
        labels, source_assignments, manifest = _fixture()

        first = build_sit_stand_event_split(labels, source_assignments, manifest)
        second = build_sit_stand_event_split(labels, source_assignments, manifest)

        self.assertEqual(first, second)
        self.assertTrue(first["split"]["development_gate"]["passed"])
        self.assertTrue(first["split"]["development_rebalanced_from_source_split"])
        self.assertFalse(first["split"]["test_access"]["test_pose_read"])

    def test_source_train_and_validation_are_rebalanced_by_whole_group(self) -> None:
        labels, source_assignments, manifest = _fixture()
        source_assignments[0]["partition"] = "train"
        source_assignments[1]["partition"] = "validation"

        result = build_sit_stand_event_split(labels, source_assignments, manifest)
        by_label = {row["label_id"]: row for row in result["assignments"]}

        self.assertEqual(by_label["label_1"]["partition"], by_label["label_2"]["partition"])
        self.assertEqual(
            by_label["label_1"]["split_group_id"],
            by_label["label_2"]["split_group_id"],
        )

    def test_any_test_member_locks_the_entire_protection_group(self) -> None:
        labels, source_assignments, manifest = _fixture(71)
        test_video = "video_70"
        test_action_ids = {
            action_id
            for row in labels
            if row["video_id"] == test_video
            for action_id in row["source_action_label_ids"]
        }
        for assignment in source_assignments:
            if assignment["label_id"] == min(test_action_ids):
                assignment["partition"] = "test"

        result = build_sit_stand_event_split(labels, source_assignments, manifest)

        self.assertFalse(
            any(row["video_id"] == test_video for row in result["assignments"])
        )
        self.assertEqual(result["split"]["locked_test_label_count"], 6)
        self.assertEqual(result["split"]["locked_test_protection_group_count"], 1)

    def test_impossible_development_coverage_fails_closed(self) -> None:
        labels, source_assignments, manifest = _fixture(10)

        with self.assertRaisesRegex(ValueError, "cannot satisfy localization gate"):
            build_sit_stand_event_split(labels, source_assignments, manifest)

    def test_train_only_auxiliary_group_never_moves_to_validation(self) -> None:
        labels, source_assignments, manifest = _fixture()
        protected_video = "video_0"
        protected_ids = {
            row["label_id"] for row in labels if row["video_id"] == protected_video
        }
        for row in labels:
            if row["label_id"] in protected_ids:
                row["partition_policy"] = "train_only"
                row["eligibility"] = "auxiliary"

        result = build_sit_stand_event_split(labels, source_assignments, manifest)
        assigned = {
            row["partition"]
            for row in result["assignments"]
            if row["label_id"] in protected_ids
        }

        self.assertEqual(assigned, {"train"})
        self.assertEqual(result["split"]["train_only_label_count"], len(protected_ids))


if __name__ == "__main__":
    unittest.main()
