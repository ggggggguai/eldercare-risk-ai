from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train.train_gait_tabular_baselines import build_parser as build_tabular_parser
from scripts.train.train_gait_tcn import build_parser as build_tcn_parser

from elderly_monitoring.modules.fall_risk.gait_training_audit import (
    build_gait_training_audit,
    validate_gait_test_release,
    write_gait_training_audit,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GaitTrainingAuditTest(unittest.TestCase):
    def test_training_clis_have_no_boolean_test_unlock(self) -> None:
        for parser in (build_tabular_parser(), build_tcn_parser()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--evaluate-test"])

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
        pose_dir = root / "poses"
        pose_dir.mkdir()
        paths = {
            "manifest": root / "manifest.jsonl",
            "action_labels": root / "action_labels.jsonl",
            "action_schema": root / "action_schema.json",
            "assignments": root / "assignments.jsonl",
            "split": root / "split.json",
            "validation": root / "validation.json",
            "training": root / "training.yaml",
            "evaluation": root / "evaluation.yaml",
            "pose_dir": pose_dir,
        }
        labels: list[dict] = []
        assignments: list[dict] = []
        manifests: list[dict] = []
        cases = (
            ("train_normal", "A01", "normal_walk", "train"),
            ("train_positive", "B02", "dragging_walk", "train"),
            ("validation_normal", "A02", "normal_turn", "validation"),
            ("validation_positive", "B03", "shuffling_walk", "validation"),
            ("test_positive", "B04", "swaying_walk", "test"),
        )
        for index, (video_id, action_id, action_type, partition) in enumerate(cases):
            label_id = f"label_{index}"
            asset_id = f"asset_{index}"
            source_group_id = f"source_{index}"
            sample_group_id = f"sample_{index}"
            labels.append(
                {
                    "schema_version": "fall-risk-action-label-v3",
                    "label_id": label_id,
                    "asset_id": asset_id,
                    "video_id": video_id,
                    "action_id": action_id,
                    "action_type": action_type,
                    "start_frame": 0,
                    "end_frame_exclusive": 16,
                    "start_time": 0.0,
                    "end_time_exclusive": 4.0,
                    "training_tier": "primary",
                    "action_type_training_tier": "primary",
                    "source_group_id": source_group_id,
                    "sample_group_id": sample_group_id,
                }
            )
            assignments.append(
                {
                    "label_kind": "action",
                    "label_id": label_id,
                    "asset_id": asset_id,
                    "video_id": video_id,
                    "training_tier": "primary",
                    "source_group_id": source_group_id,
                    "sample_group_id": sample_group_id,
                    "split_group_id": source_group_id,
                    "partition": partition,
                }
            )
            manifests.append(
                {
                    "asset_id": asset_id,
                    "video_id": video_id,
                    "dataset": "fixture",
                    "media_type": "video",
                    "eligibility": True,
                    "path": f"unused/{video_id}.mp4",
                }
            )
            pose_path = pose_dir / f"{video_id}.jsonl"
            if partition == "test":
                pose_path.write_text("this must never be parsed\n", encoding="utf-8")
            else:
                pose_path.write_text(
                    json.dumps({"frame_id": 0, "track_id": 1}) + "\n",
                    encoding="utf-8",
                )

        self._write_jsonl(paths["manifest"], manifests)
        self._write_jsonl(paths["action_labels"], labels)
        self._write_jsonl(paths["assignments"], assignments)
        self._write_json(paths["action_schema"], {"schema_version": "fixture"})
        self._write_json(
            paths["split"],
            {
                "schema_version": "fall-risk-training-split-v3",
                "split_id": "splitv3_fixture",
                "assignments_sha256": _sha256(paths["assignments"]),
                "input_sha256": {
                    "manifest": _sha256(paths["manifest"]),
                    "action_labels": _sha256(paths["action_labels"]),
                    "action_schema": _sha256(paths["action_schema"]),
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
                    "manifest": _sha256(paths["manifest"]),
                    "action_labels": _sha256(paths["action_labels"]),
                    "action_schema": _sha256(paths["action_schema"]),
                    "split_assignments": _sha256(paths["assignments"]),
                    "split_report": _sha256(paths["split"]),
                },
                "split": {"split_id": "splitv3_fixture"},
            },
        )
        paths["training"].write_text(
            "schema_version: gait-hierarchical-training-config-v1\n"
            "current_status: development_only\n",
            encoding="utf-8",
        )
        paths["evaluation"].write_text(
            "protocol_version: gait-eval-v1\n"
            "protocol_status: development_provisional\n",
            encoding="utf-8",
        )
        return paths

    def test_audit_is_development_only_and_does_not_parse_test_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            paths = self._fixture(Path(tempdir))
            report = build_gait_training_audit(**paths)

        self.assertEqual(report["status"], "development_provisional")
        self.assertFalse(report["test_access"]["test_pose_read"])
        self.assertFalse(report["test_access"]["test_tensor_generated"])
        self.assertFalse(report["test_access"]["test_evaluated"])
        self.assertEqual(report["pose_cache"]["test_files_sealed"], 1)
        self.assertEqual(report["counts"]["action_id"]["B01"], {})
        self.assertFalse(report["gates"]["validation_scale"]["passed"])

    def test_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            paths = self._fixture(Path(tempdir))
            paths["manifest"].write_text("{}\n", encoding="utf-8")
            report = build_gait_training_audit(**paths)

        self.assertEqual(report["status"], "infrastructure_only")
        self.assertFalse(report["gates"]["hash_bindings"]["passed"])

    def test_writer_creates_all_outputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            report = build_gait_training_audit(**self._fixture(root))
            outputs = {
                "manifest_output": root / "out" / "manifest.json",
                "audit_output": root / "out" / "audit.md",
                "blockers_output": root / "out" / "blockers.md",
            }
            write_gait_training_audit(report, **outputs)
            self.assertTrue(outputs["manifest_output"].is_file())
            self.assertIn("development_provisional", outputs["audit_output"].read_text())
            self.assertIn("validation", outputs["blockers_output"].read_text())
            with self.assertRaises(FileExistsError):
                write_gait_training_audit(report, **outputs)

    def test_test_release_requires_complete_matching_frozen_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            paths = self._fixture(root)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            candidate = root / "candidate.json"
            self._write_json(
                candidate,
                {
                    "protocol_status": "development_provisional",
                    "source_split_id": "splitv3_fixture",
                    "checkpoint_sha256": _sha256(checkpoint),
                },
            )
            release = root / "release.json"
            split_data = json.loads(paths["split"].read_text())
            split_data["status"] = "frozen"
            self._write_json(paths["split"], split_data)
            paths["evaluation"].write_text(
                "protocol_version: gait-eval-v1\nprotocol_status: frozen\n",
                encoding="utf-8",
            )
            release_data = {
                "release_id": "release_fixture",
                "approved_by": "independent_custodian",
                "released_at": "2026-08-10T00:00:00Z",
                "split_id": "splitv3_fixture",
                "labels_sha256": _sha256(paths["action_labels"]),
                "manifest_sha256": _sha256(paths["manifest"]),
                "assignments_sha256": _sha256(paths["assignments"]),
                "evaluation_config_sha256": _sha256(paths["evaluation"]),
                "checkpoint_sha256": _sha256(checkpoint),
                "candidate_manifest_sha256": _sha256(candidate),
            }
            self._write_json(release, release_data)

            with self.assertRaisesRegex(ValueError, "candidate_frozen"):
                validate_gait_test_release(
                    release,
                    split_path=paths["split"],
                    labels_path=paths["action_labels"],
                    manifest_path=paths["manifest"],
                    assignments_path=paths["assignments"],
                    evaluation_config_path=paths["evaluation"],
                    checkpoint_path=checkpoint,
                    candidate_manifest_path=candidate,
                )

            candidate_data = json.loads(candidate.read_text())
            candidate_data["protocol_status"] = "candidate_frozen"
            self._write_json(candidate, candidate_data)
            release_data["candidate_manifest_sha256"] = _sha256(candidate)
            self._write_json(release, release_data)
            validated = validate_gait_test_release(
                release,
                split_path=paths["split"],
                labels_path=paths["action_labels"],
                manifest_path=paths["manifest"],
                assignments_path=paths["assignments"],
                evaluation_config_path=paths["evaluation"],
                checkpoint_path=checkpoint,
                candidate_manifest_path=candidate,
            )
            self.assertEqual(validated["release_id"], "release_fixture")

            paths["evaluation"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected YAML object|frozen evaluation protocol|evaluation_config_sha256"):
                validate_gait_test_release(
                    release,
                    split_path=paths["split"],
                    labels_path=paths["action_labels"],
                    manifest_path=paths["manifest"],
                    assignments_path=paths["assignments"],
                    evaluation_config_path=paths["evaluation"],
                    checkpoint_path=checkpoint,
                    candidate_manifest_path=candidate,
                )


if __name__ == "__main__":
    unittest.main()
