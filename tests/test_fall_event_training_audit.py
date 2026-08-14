from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.fall_event_training_audit import (
    build_fall_event_training_audit,
    write_fall_event_training_audit,
)


class FallEventTrainingAuditTest(unittest.TestCase):
    def _write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {
            "manifest": root / "manifest.jsonl",
            "action_labels": root / "action_labels_v3.jsonl",
            "event_labels": root / "event_labels_v3.jsonl",
            "assignments": root / "assignments.jsonl",
            "split": root / "split.json",
            "formal": root / "formal.json",
            "validation": root / "validation.json",
            "evaluation": root / "evaluation.yaml",
            "training": root / "training.yaml",
        }
        self._write_jsonl(
            paths["manifest"],
            [
                {
                    "asset_id": "continuous_1",
                    "video_id": "continuous_1",
                    "media_type": "video",
                    "duration_sec": 1800.0,
                    "continuous_monitoring_eligible": True,
                },
                {
                    "asset_id": "clip_1",
                    "video_id": "clip_1",
                    "media_type": "video",
                    "duration_sec": 5.0,
                },
            ],
        )
        paths["action_labels"].write_text("action-labels\n", encoding="utf-8")
        paths["event_labels"].write_text("event-labels\n", encoding="utf-8")
        self._write_jsonl(
            paths["assignments"],
            [{"label_id": "label_1", "partition": "validation"}],
        )
        hashes = {
            name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
            for name in ("manifest", "action_labels", "event_labels", "assignments")
        }
        split = {
            "schema_version": "fall-risk-training-split-v3",
            "split_id": "splitv3_fixture",
            "assignment_count": 1,
            "assignments_sha256": hashes["assignments"],
            "input_sha256": {
                "manifest": hashes["manifest"],
                "action_labels": hashes["action_labels"],
                "event_labels": hashes["event_labels"],
            },
            "leakage_issues": [],
        }
        self._write_json(paths["split"], split)
        self._write_json(
            paths["formal"],
            {
                "formal_ready": False,
                "counts": {"errors": 0, "blockers": 2},
                "distributions": {
                    "issue_code": {"formal_uncertain": 2},
                },
            },
        )
        self._write_json(
            paths["validation"],
            {
                "valid": True,
                "issues": [],
                "warnings": [],
                "training_ready": {
                    "fall_event": True,
                    "near_fall_event": True,
                    "action_type": False,
                },
                "hard_negative_coverage": {
                    "fall_event": {"missing": [], "present": ["bend"]},
                },
                "input_sha256": {
                    "manifest": hashes["manifest"],
                    "action_labels": hashes["action_labels"],
                    "event_labels": hashes["event_labels"],
                    "split_assignments": hashes["assignments"],
                    "split_report": hashlib.sha256(
                        paths["split"].read_bytes()
                    ).hexdigest(),
                },
                "split": {
                    "split_id": "splitv3_fixture",
                    "valid": True,
                    "partition_supervision_counts": {
                        "fall_event": {
                            "train": {"primary_positive": 40},
                            "validation": {"primary_positive": 7},
                            "test": {"primary_positive": 999},
                        }
                    },
                },
            },
        )
        paths["evaluation"].write_text(
            "\n".join(
                [
                    "protocol_version: fall-event-eval-v1",
                    "protocol_status: development_provisional",
                    "task_type: fall_event",
                    "score_threshold: 0.5",
                    "iou_threshold: 0.5",
                    "onset_tolerance_sec: 0.25",
                    "event_merge_enabled: false",
                    "event_reset_gap_sec: 2.0",
                    "bootstrap_iterations: 50",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["training"].write_text(
            "schema_version: fall-event-training-config-v1\n"
            "current_status: provisional\n",
            encoding="utf-8",
        )
        return paths

    def test_audit_is_fail_closed_and_never_promotes_test_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            report = build_fall_event_training_audit(**paths)

        self.assertEqual(report["status"], "infrastructure_only")
        self.assertTrue(report["gates"]["hash_bindings"]["passed"])
        self.assertTrue(report["gates"]["split_leakage"]["passed"])
        self.assertFalse(report["gates"]["v2_formal"]["passed"])
        self.assertEqual(
            report["gates"]["test_release"]["status"], "custodian_required"
        )
        self.assertNotIn(
            "test", report["facts"]["fall_event_supervision_visible_to_development"]
        )
        self.assertEqual(report["facts"]["continuous_camera_hours"], 0.5)
        self.assertFalse(report["gates"]["evaluation_protocol"]["passed"])

    def test_hash_drift_blocks_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            paths["manifest"].write_text("{}\n", encoding="utf-8")
            report = build_fall_event_training_audit(**paths)

        self.assertFalse(report["gates"]["hash_bindings"]["passed"])
        self.assertIn("manifest", report["gates"]["hash_bindings"]["mismatches"])

    def test_label_record_count_cannot_substitute_for_independent_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
            validation["split"]["partition_supervision_counts"]["fall_event"][
                "validation"
            ]["primary_positive"] = 300
            self._write_json(paths["validation"], validation)
            report = build_fall_event_training_audit(**paths)

        gate = report["gates"]["validation_sample_scale"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "independent_group_audit_required")

    def test_writer_creates_all_outputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            report = build_fall_event_training_audit(**paths)
            outputs = {
                "manifest_output": root / "out" / "manifest.json",
                "audit_output": root / "out" / "audit.md",
                "blockers_output": root / "out" / "blockers.md",
            }
            write_fall_event_training_audit(report, **outputs)
            self.assertTrue(outputs["manifest_output"].is_file())
            self.assertIn(
                "infrastructure_only",
                outputs["audit_output"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "formal_uncertain",
                outputs["blockers_output"].read_text(encoding="utf-8"),
            )
            with self.assertRaises(FileExistsError):
                write_fall_event_training_audit(report, **outputs)

    def test_candidate_status_note_cannot_override_declared_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            candidate_report = root / "candidate.md"
            candidate_report.write_text(
                "Current split is `splitv3_fixture`.\n"
                "- source split: `splitv3_aaaaaaaaaaaaaaaaaaaaaaaa`\n",
                encoding="utf-8",
            )
            report = build_fall_event_training_audit(
                **paths, candidate_report=candidate_report
            )

        gate = report["gates"]["candidate_report_current_split"]
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["candidate_split_ids"], ["splitv3_aaaaaaaaaaaaaaaaaaaaaaaa"]
        )


if __name__ == "__main__":
    unittest.main()
