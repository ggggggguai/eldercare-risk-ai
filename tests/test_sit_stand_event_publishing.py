from __future__ import annotations

import hashlib
import json
import unittest

from elderly_monitoring.modules.fall_risk.sit_stand_event_publishing import (
    publish_sit_stand_event_labels,
)


def _action(label_id: str, action_id: str, *, tier: str = "primary") -> dict:
    action_type = {
        "A03": "controlled_sit_down",
        "A04": "normal_sit_to_stand",
        "A05": "controlled_squat",
        "C01": "failed_sit_to_stand",
    }[action_id]
    return {
        "schema_version": "fall-risk-action-label-v3",
        "label_id": label_id,
        "asset_id": f"asset_{label_id}",
        "video_id": f"video_{label_id}",
        "action_id": action_id,
        "action_type": action_type,
        "action_type_training_tier": tier,
        "annotator_id": "owner_annotation_batch",
        "boundary_precision": "exact",
        "start_time": 1.0,
        "end_time_exclusive": 3.0,
        "subject_id": f"subject_{label_id}",
        "source_group_id": f"source_{label_id}",
        "sample_group_id": f"sample_{label_id}",
        "quality_flags": [],
        "review_status": "single_annotated",
        "reviewer_ids": [],
        "target_status": "confirmed",
    }


def _decision(actions: list[dict]) -> dict:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in actions
    )
    return {
        "schema_version": "sit-stand-source-label-decision-v1",
        "decision_id": "sit_stand_source_labels_20260810",
        "approved_by_role": "project_owner",
        "approval_source": "interactive_user_confirmation_20260810",
        "source_action_labels_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_split_id": "split_fixture",
        "reuse_source_annotator_as_review_evidence": True,
    }


class SitStandEventPublishingTest(unittest.TestCase):
    def test_publishes_existing_human_boundaries_without_new_reviewer_input(self) -> None:
        actions = [
            _action("a03", "A03"),
            _action("a04", "A04"),
            _action("c01", "C01"),
            _action("a05", "A05"),
            _action("test_a04", "A04"),
        ]
        assignments = [
            {
                "label_id": row["label_id"],
                "partition": "test" if row["label_id"] == "test_a04" else "train",
                "split_group_id": f"split_{row['label_id']}",
            }
            for row in actions
        ]
        manifest = [
            {
                "video_id": row["video_id"],
                "asset_id": row["asset_id"],
                "dataset": "fixture",
                "scene_region": "home",
                "eligibility": True,
            }
            for row in actions
        ]

        first = publish_sit_stand_event_labels(
            actions, assignments, manifest, decision=_decision(actions)
        )
        second = publish_sit_stand_event_labels(
            actions, assignments, manifest, decision=_decision(actions)
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["labels"]), 4)
        by_source = {
            row["source_action_label_ids"][0]: row for row in first["labels"]
        }
        self.assertEqual(by_source["a03"]["transition_type"], "stand_to_sit")
        self.assertEqual(by_source["a04"]["transition_type"], "sit_to_stand")
        self.assertEqual(by_source["c01"]["outcome"], "failed")
        self.assertEqual(by_source["a05"]["interval_type"], "explicit_background")
        self.assertEqual(by_source["a05"]["hard_negative_type"], "controlled_squat")
        self.assertEqual(by_source["a04"]["onset_time"], 1.0)
        self.assertEqual(by_source["a04"]["offset_time"], 3.0)
        self.assertEqual(by_source["a04"]["reviewed_by"], ["owner_annotation_batch"])
        self.assertEqual(first["report"]["locked_test_source_label_count"], 1)
        self.assertFalse(first["report"]["test_truth_published"])

    def test_ignore_tier_stays_ignore_and_unlabelled_gaps_are_not_created(self) -> None:
        actions = [_action("ignored", "A04", tier="ignore")]
        result = publish_sit_stand_event_labels(
            actions,
            [{"label_id": "ignored", "partition": "validation", "split_group_id": "g"}],
            [
                {
                    "video_id": actions[0]["video_id"],
                    "asset_id": actions[0]["asset_id"],
                    "dataset": "fixture",
                    "scene_region": "home",
                    "eligibility": True,
                }
            ],
            decision=_decision(actions),
        )

        self.assertEqual(len(result["labels"]), 1)
        self.assertEqual(result["labels"][0]["interval_type"], "ignore")
        self.assertEqual(result["labels"][0]["onset_time"], 1.0)
        self.assertEqual(result["labels"][0]["offset_time"], 3.0)

    def test_missing_source_annotator_fails_instead_of_inventing_reviewer(self) -> None:
        actions = [_action("a04", "A04")]
        actions[0]["annotator_id"] = ""

        with self.assertRaisesRegex(ValueError, "annotator_id"):
            publish_sit_stand_event_labels(
                actions,
                [{"label_id": "a04", "partition": "train", "split_group_id": "g"}],
                [
                    {
                        "video_id": actions[0]["video_id"],
                        "asset_id": actions[0]["asset_id"],
                        "dataset": "fixture",
                        "scene_region": "home",
                        "eligibility": True,
                    }
                ],
                decision=_decision(actions),
            )


if __name__ == "__main__":
    unittest.main()
