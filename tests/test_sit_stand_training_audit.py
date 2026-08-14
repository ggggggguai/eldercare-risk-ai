from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.sit_stand_training_audit import (
    build_sit_stand_training_audit,
    write_sit_stand_training_audit,
)


class SitStandTrainingAuditTest(unittest.TestCase):
    def _write_json(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {
            "action_labels": root / "actions.jsonl",
            "manifest": root / "manifest.jsonl",
            "assignments": root / "assignments.jsonl",
            "split": root / "split.json",
            "validation": root / "validation.json",
            "training_config": root / "training.yaml",
            "evaluation_config": root / "evaluation.yaml",
            "historical_report": root / "historical.md",
        }
        self._write_jsonl(
            paths["action_labels"],
            [
                {
                    "label_id": "action_1",
                    "asset_id": "asset_1",
                    "video_id": "video_1",
                    "action_id": "A04",
                    "action_type_training_tier": "primary",
                    "boundary_precision": "exact",
                    "review_status": "single_annotated",
                    "subject_id": "subject_1",
                    "source_group_id": "source_1",
                    "sample_group_id": "sample_1",
                }
            ],
        )
        self._write_jsonl(
            paths["manifest"],
            [
                {
                    "asset_id": "asset_1",
                    "video_id": "video_1",
                    "dataset": "fixture",
                    "media_type": "video",
                    "duration_sec": 12.0,
                    "eligibility": True,
                }
            ],
        )
        self._write_jsonl(
            paths["assignments"],
            [
                {
                    "label_id": "action_1",
                    "partition": "validation",
                    "split_group_id": "split_1",
                }
            ],
        )
        hashes = {
            name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
            for name in ("action_labels", "manifest", "assignments")
        }
        self._write_json(
            paths["split"],
            {
                "split_id": "splitv3_fixture",
                "status": "provisional",
                "assignments_sha256": hashes["assignments"],
                "input_sha256": {
                    "action_labels": hashes["action_labels"],
                    "manifest": hashes["manifest"],
                },
                "leakage_issues": [],
            },
        )
        self._write_json(
            paths["validation"],
            {
                "valid": True,
                "training_ready": {"action_type": False},
                "input_sha256": {
                    "action_labels": hashes["action_labels"],
                    "manifest": hashes["manifest"],
                    "split_assignments": hashes["assignments"],
                    "split_report": hashlib.sha256(paths["split"].read_bytes()).hexdigest(),
                },
                "split": {"split_id": "splitv3_fixture", "valid": True},
            },
        )
        paths["training_config"].write_text(
            "schema_version: sit-stand-event-training-config-v1\ncurrent_status: provisional\n",
            encoding="utf-8",
        )
        paths["evaluation_config"].write_text(
            "protocol_version: sit-stand-event-eval-v1\nprotocol_status: development_provisional\n",
            encoding="utf-8",
        )
        paths["historical_report"].write_text(
            "dataset: missing/dataset.npz\ncheckpoint: missing/best_model.pt\n",
            encoding="utf-8",
        )
        return paths

    def test_audit_is_fail_closed_and_records_test_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_sit_stand_training_audit(**self._fixture(Path(tmp)))

        self.assertEqual(report["status"], "infrastructure_only")
        self.assertFalse(report["gates"]["event_localization"]["passed"])
        self.assertFalse(report["gates"]["phase_segmentation"]["passed"])
        self.assertFalse(report["gates"]["action_proxy"]["passed"])
        self.assertFalse(report["gates"]["functional_proxy"]["passed"])
        self.assertFalse(report["test_access"]["test_pose_read"])
        self.assertFalse(report["test_access"]["test_features_generated"])
        self.assertFalse(report["test_access"]["test_evaluated"])
        self.assertIn("evaluation_config", report["input_sha256"])
        self.assertEqual(report["facts"]["explicit_background_duration_sec"], 0.0)

    def test_one_event_and_one_background_do_not_unlock_development_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._fixture(root)
            paths["event_labels"] = root / "event_labels.jsonl"
            paths["review_log"] = root / "review_log.jsonl"
            common = {
                "schema_version": "fall-risk-sit-stand-event-label-v1",
                "asset_id": "asset_1",
                "video_id": "video_1",
                "subject_id": "subject_1",
                "source_group_id": "source_1",
                "sample_group_id": "sample_1",
                "split_group_id": "split_1",
                "boundary_precision": "exact",
                "seat_off_proxy": None,
                "seat_contact_proxy": None,
                "support_contact_proxy": None,
                "stabilization_time_proxy": None,
                "scene_region": "home",
                "visibility": "full",
                "quality_flags": [],
                "review_status": "double_reviewed",
                "reviewed_by": ["reviewer_a", "reviewer_b"],
                "eligibility": "eligible",
                "source_action_label_ids": ["action_1"],
            }
            rows = [
                {
                    **common,
                    "label_id": "sitstand_000000000000000000000001",
                    "event_id": "event_1",
                    "interval_type": "event",
                    "transition_type": "sit_to_stand",
                    "onset_time": 1.0,
                    "offset_time": 2.0,
                    "outcome": "completed",
                    "attempt_count": 1,
                    "attempt_intervals": [{"onset_time": 1.0, "offset_time": 2.0}],
                    "hard_negative_type": None,
                },
                {
                    **common,
                    "label_id": "sitstand_000000000000000000000002",
                    "event_id": "background_1",
                    "interval_type": "explicit_background",
                    "transition_type": None,
                    "onset_time": 3.0,
                    "offset_time": 4.0,
                    "outcome": None,
                    "attempt_count": 0,
                    "attempt_intervals": [],
                    "hard_negative_type": "controlled_bend",
                },
            ]
            self._write_jsonl(paths["event_labels"], rows)
            self._write_jsonl(
                paths["review_log"],
                [
                    {"label_id": row["label_id"], "reviewer_id": reviewer}
                    for row in rows
                    for reviewer in ("reviewer_a", "reviewer_b")
                ],
            )

            report = build_sit_stand_training_audit(**paths)

        self.assertEqual(report["status"], "infrastructure_only")
        self.assertFalse(report["gates"]["event_localization"]["passed"])
        self.assertFalse(
            report["authorization"]["continuous_model_training_allowed"]
        )

    def test_writer_creates_three_outputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_sit_stand_training_audit(**self._fixture(root))
            outputs = {
                "manifest_output": root / "out" / "manifest.json",
                "audit_output": root / "out" / "audit.md",
                "blockers_output": root / "out" / "blockers.md",
            }
            write_sit_stand_training_audit(report, **outputs)
            self.assertTrue(outputs["manifest_output"].is_file())
            self.assertIn("infrastructure_only", outputs["audit_output"].read_text())
            self.assertIn("event_localization", outputs["blockers_output"].read_text())
            with self.assertRaises(FileExistsError):
                write_sit_stand_training_audit(report, **outputs)


if __name__ == "__main__":
    unittest.main()
