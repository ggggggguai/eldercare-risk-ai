from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.fall_event_data_governance import (
    govern_fall_event_training_data,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FallEventDataGovernanceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path | list[Path]]:
        action_rows = [
            self._action("normal", "normal_video", "A01", "primary"),
            self._action("recovered", "scf_video", "C04", "auxiliary"),
            self._action("ignored", "ignored_video", "A11", "primary"),
            # This malformed test row proves governance does not parse test truth.
            {"label_id": "actionv3_test", "video_id": "sealed_test"},
        ]
        event_rows = [
            self._event(
                "fall_primary",
                "fall_video",
                role="positive",
                tier="primary",
                source_type="v2_le2i_txt",
            ),
            self._event(
                "fall_aux",
                "ntu_video",
                role="positive",
                tier="auxiliary",
                source_type="v2_ntu_rgbd_manual_clip_label",
            ),
            self._event(
                "hard_negative",
                "negative_video",
                role="negative",
                tier="primary",
                hard_negative_type="controlled_sit_down",
            ),
            {"label_id": "eventv3_test", "video_id": "sealed_test"},
        ]
        manifest_rows = [
            self._manifest("normal_video", "fixture_normal"),
            self._manifest("scf_video", "self_collected_scf_mvp_v1"),
            self._manifest("ignored_video", "fixture_normal"),
            self._manifest("fall_video", "le2i_imvia"),
            self._manifest("ntu_video", "ntu_rgbd"),
            self._manifest("negative_video", "fixture_negative"),
            self._manifest("sealed_test", "sealed"),
        ]
        assignments = [
            self._assignment("actionv3_normal", "normal_video", "action", "train"),
            self._assignment("actionv3_recovered", "scf_video", "action", "train"),
            self._assignment("actionv3_ignored", "ignored_video", "action", "train"),
            self._assignment("actionv3_test", "sealed_test", "action", "test"),
            self._assignment("eventv3_fall_primary", "fall_video", "event", "train"),
            self._assignment("eventv3_fall_aux", "ntu_video", "event", "train"),
            self._assignment(
                "eventv3_hard_negative", "negative_video", "event", "validation"
            ),
            self._assignment("eventv3_test", "sealed_test", "event", "test"),
        ]
        paths = {
            "actions": root / "actions.jsonl",
            "events": root / "events.jsonl",
            "manifest": root / "manifest.jsonl",
            "assignments": root / "assignments.jsonl",
            "split": root / "split.json",
            "validation": root / "validation.json",
            "config": root / "governance.json",
            "pose_roots": [root / "pose_main", root / "pose_scf"],
        }
        _write_jsonl(paths["actions"], action_rows)
        _write_jsonl(paths["events"], event_rows)
        _write_jsonl(paths["manifest"], manifest_rows)
        _write_jsonl(paths["assignments"], assignments)
        paths["split"].write_text(
            json.dumps(
                {
                    "split_id": "fixture-split",
                    "assignments_sha256": _sha256(paths["assignments"]),
                    "input_sha256": {
                        "action_labels": _sha256(paths["actions"]),
                        "event_labels": _sha256(paths["events"]),
                        "manifest": _sha256(paths["manifest"]),
                    },
                    "leakage_issues": [],
                }
            ),
            encoding="utf-8",
        )
        paths["validation"].write_text(
            json.dumps(
                {
                    "valid": True,
                    "training_ready": {"fall_event": True},
                    "input_sha256": {
                        "action_labels": _sha256(paths["actions"]),
                        "event_labels": _sha256(paths["events"]),
                        "manifest": _sha256(paths["manifest"]),
                        "split_assignments": _sha256(paths["assignments"]),
                        "split_report": _sha256(paths["split"]),
                    },
                }
            ),
            encoding="utf-8",
        )
        config = {
            "schema_version": "fall-event-continuous-training-governance-v1",
            "governance_id": "fixture-governance",
            "input_sha256": {
                "action_labels": _sha256(paths["actions"]),
                "event_labels": _sha256(paths["events"]),
                "manifest": _sha256(paths["manifest"]),
                "split_assignments": _sha256(paths["assignments"]),
                "split_report": _sha256(paths["split"]),
                "validation_report": _sha256(paths["validation"]),
            },
            "action_negative_families": {
                "ordinary_background": ["A01"],
                "recovered_near_fall": ["C04"],
            },
            "onset_source_types": ["v2_le2i_txt"],
            "family_policy": {
                "primary_fall": {"presence_loss_weight": 1.0, "bucket_mass": 1.0},
                "auxiliary_fall": {"presence_loss_weight": 0.45, "bucket_mass": 0.65},
                "explicit_hard_negative": {
                    "presence_loss_weight": 1.0,
                    "bucket_mass": 1.0,
                },
                "recovered_near_fall": {
                    "primary_presence_loss_weight": 0.9,
                    "auxiliary_presence_loss_weight": 0.55,
                    "bucket_mass": 1.25,
                },
                "ordinary_background": {
                    "primary_presence_loss_weight": 0.45,
                    "auxiliary_presence_loss_weight": 0.25,
                    "bucket_mass": 0.5,
                },
            },
            "pose_policy": {
                "interpolated_coordinates": "mask_coordinates_and_derived_motion",
                "test_pose_access": "forbidden",
            },
            "sampling_policy": {
                "min_sampling_weight": 0.1,
                "max_sampling_weight": 20.0,
            },
        }
        paths["config"].write_text(json.dumps(config), encoding="utf-8")
        for video_id, pose_root in (
            ("normal_video", paths["pose_roots"][0]),
            ("fall_video", paths["pose_roots"][0]),
            ("ntu_video", paths["pose_roots"][0]),
            ("negative_video", paths["pose_roots"][0]),
            ("scf_video", paths["pose_roots"][1]),
        ):
            _write_jsonl(pose_root / f"{video_id}.jsonl", [{"frame_id": 0}])
        return paths

    def test_builds_layered_manifest_and_masks_proxy_onset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            result = govern_fall_event_training_data(
                action_labels=paths["actions"],
                event_labels=paths["events"],
                manifest=paths["manifest"],
                assignments=paths["assignments"],
                split=paths["split"],
                validation=paths["validation"],
                governance_config=paths["config"],
                pose_roots=paths["pose_roots"],
                output_dir=Path(tmpdir) / "output",
            )
            rows = [
                json.loads(line)
                for line in Path(result["manifest_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 5)
        by_label = {row["source_label_id"]: row for row in rows}
        self.assertEqual(
            by_label["eventv3_fall_primary"]["allowed_heads"],
            ["presence", "onset"],
        )
        self.assertEqual(by_label["eventv3_fall_primary"]["onset_loss_weight"], 1.0)
        self.assertEqual(by_label["eventv3_fall_aux"]["allowed_heads"], ["presence"])
        self.assertEqual(by_label["eventv3_fall_aux"]["onset_loss_weight"], 0.0)
        self.assertEqual(
            by_label["actionv3_recovered"]["supervision_family"],
            "recovered_near_fall",
        )
        self.assertTrue(by_label["actionv3_recovered"]["pose_path"].endswith("scf_video.jsonl"))
        self.assertEqual(audit["data_access"]["test_pose_read"], False)
        self.assertEqual(audit["counts"]["locked_test_labels"], 2)
        self.assertGreaterEqual(audit["sampling"]["training_weight_min"], 0.1)
        self.assertLessEqual(audit["sampling"]["training_weight_max"], 20.0)

    def test_is_deterministic_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            first = govern_fall_event_training_data(
                action_labels=paths["actions"],
                event_labels=paths["events"],
                manifest=paths["manifest"],
                assignments=paths["assignments"],
                split=paths["split"],
                validation=paths["validation"],
                governance_config=paths["config"],
                pose_roots=paths["pose_roots"],
                output_dir=Path(tmpdir) / "first",
            )
            second = govern_fall_event_training_data(
                action_labels=paths["actions"],
                event_labels=paths["events"],
                manifest=paths["manifest"],
                assignments=paths["assignments"],
                split=paths["split"],
                validation=paths["validation"],
                governance_config=paths["config"],
                pose_roots=paths["pose_roots"],
                output_dir=Path(tmpdir) / "second",
            )
            self.assertEqual(_sha256(Path(first["manifest_path"])), _sha256(Path(second["manifest_path"])))

            paths["actions"].write_text(
                paths["actions"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "governance action_labels SHA-256 mismatch"):
                govern_fall_event_training_data(
                    action_labels=paths["actions"],
                    event_labels=paths["events"],
                    manifest=paths["manifest"],
                    assignments=paths["assignments"],
                    split=paths["split"],
                    validation=paths["validation"],
                    governance_config=paths["config"],
                    pose_roots=paths["pose_roots"],
                    output_dir=Path(tmpdir) / "third",
                )

    @staticmethod
    def _action(suffix: str, video_id: str, action_id: str, tier: str) -> dict:
        return {
            "schema_version": "fall-risk-action-label-v3",
            "label_id": f"actionv3_{suffix}",
            "asset_id": f"asset_{video_id}",
            "video_id": video_id,
            "subject_id": f"subject_{video_id}",
            "source_group_id": f"source_{video_id}",
            "sample_group_id": f"sample_{video_id}",
            "action_id": action_id,
            "action_type": action_id.lower(),
            "training_tier": tier,
            "action_type_training_tier": tier,
            "boundary_precision": "approximate",
            "review_status": "single_annotated",
            "start_frame": 0,
            "end_frame_exclusive": 30,
            "start_time": 0.0,
            "end_time_exclusive": 2.0,
        }

    @staticmethod
    def _event(
        suffix: str,
        video_id: str,
        *,
        role: str,
        tier: str,
        source_type: str | None = None,
        hard_negative_type: str | None = None,
    ) -> dict:
        source_refs = [] if source_type is None else [{"source_type": source_type}]
        return {
            "schema_version": "fall-risk-event-label-v3",
            "label_id": f"eventv3_{suffix}",
            "asset_id": f"asset_{video_id}",
            "video_id": video_id,
            "subject_id": f"subject_{video_id}",
            "source_group_id": f"source_{video_id}",
            "sample_group_id": f"sample_{video_id}",
            "task_type": "fall_event",
            "label_role": role,
            "event_type": "fall" if role == "positive" else None,
            "training_tier": tier,
            "boundary_precision": "exact" if role == "positive" else "approximate",
            "hard_negative_type": hard_negative_type,
            "source_refs": source_refs,
            "start_frame": 5,
            "end_frame_exclusive": 25,
            "start_time": 0.25,
            "end_time_exclusive": 1.25,
        }

    @staticmethod
    def _manifest(video_id: str, dataset: str) -> dict:
        return {
            "asset_id": f"asset_{video_id}",
            "video_id": video_id,
            "dataset": dataset,
            "scene_region": "room",
            "eligibility": True,
            "media_type": "video",
        }

    @staticmethod
    def _assignment(label_id: str, video_id: str, kind: str, partition: str) -> dict:
        return {
            "label_id": label_id,
            "label_kind": kind,
            "task_type": "action" if kind == "action" else "fall_event",
            "asset_id": f"asset_{video_id}",
            "video_id": video_id,
            "subject_id": f"subject_{video_id}",
            "source_group_id": f"source_{video_id}",
            "sample_group_id": f"sample_{video_id}",
            "split_group_id": f"split_{video_id}",
            "partition": partition,
        }


if __name__ == "__main__":
    unittest.main()
