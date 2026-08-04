from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    FALL_HARD_NEGATIVES,
    migrate_v2_training_labels,
    validate_training_labels_v3,
    write_training_label_migration,
    write_training_split_v3,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION_SCHEMA = ROOT / "configs/data/fall_risk_action_label_schema_v3.json"
EVENT_SCHEMA = ROOT / "configs/data/fall_risk_event_label_schema_v3.json"
REVIEWED_DECISION = (
    ROOT / "configs/data/fall_risk_training_decision_20260804.json"
)


class FallRiskTrainingLabelsV3Test(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _manifest(self, root: Path, *, eligible: bool = True) -> tuple[dict, Path]:
        video_path = root / "video.avi"
        video_path.write_bytes(b"video")
        row = {
            "asset_id": "asset_video_1",
            "video_id": "video_1",
            "path": video_path.as_posix(),
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "fps_num": 25,
            "fps_den": 1,
            "frame_count": 200,
            "duration_sec": 8.0,
            "subject_id": "unknown",
            "source_group_id": "source_video_1",
            "original_event_id": "original_video_1",
            "eligibility": eligible,
            "exclusion_reasons": [] if eligible else ["media_probe_failed"],
        }
        return row, video_path

    def _source(self, root: Path) -> tuple[Path, str]:
        path = root / "annotations.xml"
        path.write_text("<annotations/>", encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _action(
        self,
        source_path: Path,
        source_sha256: str,
        *,
        label_id: str,
        action_id: str,
        action_name: str,
        start_frame: int,
        end_frame: int,
        quality: str = "clear",
        note: str = "",
    ) -> dict:
        event_types = {
            "A01": "normal_activity",
            "A02": "normal_activity",
            "A03": "normal_activity",
            "A04": "normal_activity",
            "A05": "normal_activity",
            "A06": "normal_activity",
            "A07": "normal_activity",
            "A08": "normal_activity",
            "A09": "normal_activity",
            "A10": "normal_activity",
            "A11": "normal_activity",
            "A12": "normal_activity",
            "C03": "near_fall",
            "C04": "wall_support",
            "D02": "fall",
            "D04": "long_static",
            "D05": "fall",
            "U01": "uncertain",
        }
        return {
            "label_id": label_id,
            "source_record_id": f"source:{label_id}",
            "source_annotation_path": source_path.as_posix(),
            "source_annotation_sha256": source_sha256,
            "source_export_id": "cvat_export_1",
            "asset_id": "asset_video_1",
            "video_id": "video_1",
            "file_path": "video.avi",
            "subject_id": "unknown",
            "scene": "home",
            "view": "fixed_camera",
            "action_id": action_id,
            "action_name": action_name,
            "event_type": event_types[action_id],
            "start_time": start_frame / 25,
            "end_time": end_frame / 25,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_index_base": 0,
            "labeler": "labeler_01",
            "quality": quality,
            "note": note,
            "source": "cvat",
            "cvat_task_id": "1",
            "cvat_track_id": int(label_id.rsplit("_", 1)[-1]),
            "bbox_start": [0.0, 0.0, 1.0, 1.0],
            "bbox_end": [0.0, 0.0, 1.0, 1.0],
        }

    def _official_fall(self, root: Path) -> dict:
        source = root / "video.txt"
        source.write_text("41\n80\n", encoding="utf-8")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        return {
            "label_id": "event_official_1",
            "source_record_id": "le2i:video_1:fall",
            "source_annotation_path": source.as_posix(),
            "source_annotation_sha256": source_hash,
            "asset_id": "asset_video_1",
            "video_id": "video_1",
            "event_type": "fall",
            "start_time": 1.6,
            "end_time": 3.16,
            "start_frame": 40,
            "end_frame": 79,
            "frame_index_base": 0,
            "severity": 4,
            "label_source": "le2i_txt",
            "note": "Official fall window.",
            "source_start_frame": 41,
            "source_end_frame": 80,
            "source_frame_index_base": 1,
        }

    def _mapped_fall(self, action: dict) -> dict:
        return {
            "label_id": "event_mapped_1",
            "source_record_id": f"mapped:{action['label_id']}",
            "source_annotation_path": action["source_annotation_path"],
            "source_annotation_sha256": action["source_annotation_sha256"],
            "source_export_id": action["source_export_id"],
            "asset_id": action["asset_id"],
            "video_id": action["video_id"],
            "event_type": "fall",
            "start_time": action["start_time"],
            "end_time": action["end_time"],
            "start_frame": action["start_frame"],
            "end_frame": action["end_frame"],
            "frame_index_base": 0,
            "severity": 4,
            "label_source": "cvat_action_mapping",
            "note": "Mapped fall.",
            "source_action_id": action["action_id"],
            "source_action_name": action["action_name"],
            "source_action_label_id": action["label_id"],
            "mapping_version": "fall-action-event-v2",
            "cvat_task_id": action["cvat_task_id"],
            "cvat_track_id": action["cvat_track_id"],
        }

    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "manifest": root / "manifest.jsonl",
            "v2_actions": root / "action_labels.jsonl",
            "v2_events": root / "event_labels.jsonl",
            "v3_actions": root / "action_labels_v3.jsonl",
            "v3_events": root / "event_labels_v3.jsonl",
            "migration_report": root / "migration.json",
            "split_assignments": root / "split" / "assignments.jsonl",
            "split_report": root / "split" / "split.json",
        }

    def _reviewed_decision(
        self,
        action_labels_path: Path,
        *,
        ntu_full_clip_fall_boundary: dict | None = None,
        near_fall_positive_actions: list[dict] | None = None,
        action_hard_negative_mappings: list[dict] | None = None,
        video_hard_negative_overrides: list[dict] | None = None,
        quality_hard_negative_mappings: list[dict] | None = None,
        event_hard_negative_mappings: list[dict] | None = None,
    ) -> dict:
        return {
            "schema_version": "fall-risk-reviewed-training-decision-v1",
            "decision_id": "fixture-reviewed-training-decision",
            "reviewed_at": "2026-08-04",
            "reviewer_id": "project_owner",
            "source_task_id": "019fcb65-e9d1-7400-b018-aba2d2fc9941",
            "action_labels_sha256": hashlib.sha256(
                action_labels_path.read_bytes()
            ).hexdigest(),
            "ntu_full_clip_fall_boundary": ntu_full_clip_fall_boundary
            or {"enabled": False},
            "near_fall_positive_actions": near_fall_positive_actions or [],
            "action_hard_negative_mappings": action_hard_negative_mappings or [],
            "video_hard_negative_overrides": video_hard_negative_overrides or [],
            "quality_hard_negative_mappings": quality_hard_negative_mappings or [],
            "event_hard_negative_mappings": event_hard_negative_mappings or [],
            "rationale": ["Fixture project-owner adjudication."],
        }

    def _write_migration_with_reviewed_decision(
        self,
        root: Path,
        *,
        manifests: list[dict],
        actions: list[dict],
        events: list[dict],
        decision_overrides: dict,
    ) -> tuple[dict, list[dict], list[dict]]:
        paths = self._paths(root)
        self._write_jsonl(paths["manifest"], manifests)
        self._write_jsonl(paths["v2_actions"], actions)
        self._write_jsonl(paths["v2_events"], events)
        decision_path = root / "reviewed-training-decision.json"
        decision_path.write_text(
            json.dumps(
                self._reviewed_decision(
                    paths["v2_actions"], **decision_overrides
                )
            ),
            encoding="utf-8",
        )
        report = write_training_label_migration(
            manifest_path=paths["manifest"],
            action_labels_v2_path=paths["v2_actions"],
            event_labels_v2_path=paths["v2_events"],
            action_labels_v3_path=paths["v3_actions"],
            event_labels_v3_path=paths["v3_events"],
            report_path=paths["migration_report"],
            reviewed_decision_paths=[decision_path],
        )
        migrated_actions = [
            json.loads(line)
            for line in paths["v3_actions"].read_text(encoding="utf-8").splitlines()
        ]
        migrated_events = [
            json.loads(line)
            for line in paths["v3_events"].read_text(encoding="utf-8").splitlines()
        ]
        return report, migrated_actions, migrated_events

    def test_migration_deduplicates_fall_links_actions_and_emits_ignore_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            actions = [
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_1",
                    action_id="D02",
                    action_name="lateral_fall",
                    start_frame=45,
                    end_frame=75,
                ),
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_2",
                    action_id="D04",
                    action_name="long_static_after_fall",
                    start_frame=80,
                    end_frame=120,
                ),
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_3",
                    action_id="C04",
                    action_name="rapid_support_contact",
                    start_frame=10,
                    end_frame=20,
                ),
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_4",
                    action_id="U01",
                    action_name="unable_to_judge",
                    start_frame=130,
                    end_frame=150,
                    quality="heavy_occlusion",
                    note="occluded",
                ),
            ]
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=actions,
                event_rows=[self._official_fall(root)],
            )

        self.assertEqual(len(result.action_labels), 4)
        self.assertEqual(len(result.event_labels), 3)
        fall = next(row for row in result.event_labels if row["label_role"] == "positive")
        self.assertEqual(fall["event_type"], "fall")
        self.assertEqual(fall["event_subtype"], "lateral")
        self.assertEqual(fall["training_tier"], "primary")
        self.assertEqual(fall["subtype_training_tier"], "auxiliary")
        self.assertEqual(fall["end_frame_exclusive"], 80)
        self.assertEqual(len(fall["source_refs"]), 2)
        action_by_id = {row["action_id"]: row for row in result.action_labels}
        self.assertEqual(action_by_id["D02"]["linked_event_id"], fall["label_id"])
        self.assertEqual(action_by_id["D04"]["linked_event_id"], fall["label_id"])
        self.assertIn(action_by_id["D04"]["label_id"], fall["linked_action_ids"])
        self.assertEqual(action_by_id["D04"]["action_type"], "post_fall_immobile")
        self.assertEqual(action_by_id["D04"]["training_tier"], "auxiliary")
        self.assertEqual(action_by_id["D04"]["action_type_training_tier"], "ignore")
        self.assertIsNone(action_by_id["C04"]["linked_event_id"])
        self.assertEqual(action_by_id["C04"]["training_tier"], "auxiliary")
        self.assertEqual(action_by_id["C04"]["action_type_training_tier"], "ignore")
        self.assertIsNone(action_by_id["U01"]["action_family"])
        self.assertEqual(action_by_id["U01"]["training_tier"], "ignore")
        self.assertEqual(action_by_id["U01"]["action_type_training_tier"], "ignore")
        ignore_tasks = {
            row["task_type"]
            for row in result.event_labels
            if row["label_role"] == "ignore"
        }
        self.assertEqual(ignore_tasks, {"fall_event", "near_fall_event"})
        self.assertEqual(result.report["deduplicated_official_action_falls"], 1)
        self.assertEqual(result.report["near_fall_positive_count"], 0)
        self.assertEqual(
            result.report["action_type_counts_by_tier"]["ignore"][
                "rapid_support_reaction"
            ],
            1,
        )

    def test_adjudicated_ntu_squat_becomes_a_manual_fall_hard_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            video_id = "ntu_rgbd_s017_p020_r001_a043_c001"
            manifest["video_id"] = video_id
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="A05",
                action_name="controlled_squat",
                start_frame=0,
                end_frame=57,
            )
            action["video_id"] = video_id
            decision_path = root / "ntu-decision.json"
            decision_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ntu-rgbd-a043-cvat-decision-v1",
                        "decision_id": "fixture-ntu-adjudication",
                        "reviewed_at": "2026-07-30",
                        "reviewer_id": "project_owner",
                        "source_action_code": "A043",
                        "decision": "accept_manual_cvat_labels",
                        "direct_filename_import": False,
                        "accepted_batch_id": "ntu_rgbd_a043_cvat_review",
                        "accepted_protocols": [
                            "segmented_nonfall_controlled_lie_down",
                            "segmented_normal_to_fall",
                            "segmented_normal_to_fall_to_sit_to_stand",
                            "segmented_normal_to_fall_with_trailing_outside",
                            "whole_clip_controlled_squat_hard_negative",
                            "whole_clip_fall_without_onset",
                            "whole_clip_uncertain",
                        ],
                        "v3_event_training_policy": "auxiliary_approximate",
                        "adjudications": [
                            {
                                "type": "accepted_hard_negative",
                                "source_names": [
                                    "S017C001P020R001A043_rgb.avi"
                                ],
                                "accepted_label": "A05_controlled_squat",
                                "training_use": "fall_hard_negative",
                                "evidence": "three_view_visual_review",
                                "reason": "Controlled lowering without loss of support.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths = self._paths(root)
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v2_actions"], [action])
            self._write_jsonl(paths["v2_events"], [])

            report = write_training_label_migration(
                manifest_path=paths["manifest"],
                action_labels_v2_path=paths["v2_actions"],
                event_labels_v2_path=paths["v2_events"],
                action_labels_v3_path=paths["v3_actions"],
                event_labels_v3_path=paths["v3_events"],
                report_path=paths["migration_report"],
                manual_negative_decision_paths=[decision_path],
            )

            events = [
                json.loads(line)
                for line in paths["v3_events"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(events), 1)
            negative = events[0]
            self.assertEqual(negative["task_type"], "fall_event")
            self.assertEqual(negative["label_role"], "negative")
            self.assertEqual(negative["hard_negative_type"], "squat_or_kneel")
            self.assertEqual(negative["training_tier"], "primary")
            self.assertEqual(negative["review_status"], "adjudicated")
            self.assertEqual(negative["reviewer_ids"], ["project_owner"])
            self.assertTrue(
                any(
                    source_ref["source_type"] == "manual_v3"
                    and source_ref["source_annotation_path"]
                    == decision_path.as_posix()
                    for source_ref in negative["source_refs"]
                )
            )
            self.assertEqual(report["manual_negative_count"], 1)
            self.assertEqual(report["unmatched_manual_negative_decisions"], [])

    def test_reviewed_decision_rejects_an_action_label_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="C03",
                action_name="stumble",
                start_frame=10,
                end_frame=20,
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v2_actions"], [action])
            self._write_jsonl(paths["v2_events"], [])
            decision = self._reviewed_decision(paths["v2_actions"])
            decision["action_labels_sha256"] = "0" * 64
            decision_path = root / "stale-decision.json"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "action-label hash mismatch"):
                write_training_label_migration(
                    manifest_path=paths["manifest"],
                    action_labels_v2_path=paths["v2_actions"],
                    event_labels_v2_path=paths["v2_events"],
                    action_labels_v3_path=paths["v3_actions"],
                    event_labels_v3_path=paths["v3_events"],
                    report_path=paths["migration_report"],
                    reviewed_decision_paths=[decision_path],
                )

    def test_reviewed_ntu_full_clip_fall_uses_first_and_last_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            manifest.update(
                video_id="ntu_rgbd_s001_p001_r001_a043_c001",
                dataset="ntu_rgbd",
            )
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=0,
                end_frame=198,
            )
            action.update(
                video_id=manifest["video_id"],
                source="ntu_rgbd_manual_clip_label",
                labeler="ntu_reviewer",
            )
            event = self._mapped_fall(action)
            event.update(start_frame=5, end_frame=150, start_time=0.2, end_time=6.0)

            report, _, migrated_events = self._write_migration_with_reviewed_decision(
                root,
                manifests=[manifest],
                actions=[action],
                events=[event],
                decision_overrides={
                    "ntu_full_clip_fall_boundary": {
                        "enabled": True,
                        "video_id_prefix": "ntu_rgbd_",
                        "action_ids": ["D01", "D02", "D03", "D05"],
                        "onset_frame": "first_frame",
                        "offset_frame": "last_frame",
                        "accepted_source_end_frame_gaps": [0, 1],
                        "boundary_precision": "exact",
                        "training_tier_policy": "preserve",
                    }
                },
            )

        [fall] = migrated_events
        self.assertEqual(fall["start_frame"], 0)
        self.assertEqual(fall["end_frame_exclusive"], 200)
        self.assertEqual(fall["onset_frame"], 0)
        self.assertEqual(fall["boundary_precision"], "exact")
        self.assertEqual(fall["review_status"], "adjudicated")
        self.assertEqual(fall["reviewer_ids"], ["project_owner"])
        self.assertEqual(
            report["reviewed_decision_matches"]["ntu_full_clip_fall_boundary"], 1
        )

    def test_reviewed_c03_becomes_a_linked_near_fall_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="C03",
                action_name="stumble",
                start_frame=10,
                end_frame=20,
            )

            report, migrated_actions, migrated_events = (
                self._write_migration_with_reviewed_decision(
                    root,
                    manifests=[manifest],
                    actions=[action],
                    events=[],
                    decision_overrides={
                        "near_fall_positive_actions": [
                            {
                                "action_id": "C03",
                                "event_subtype": "stumble_recovery",
                                "recovery_frame": "last_frame",
                                "training_tier_policy": "preserve",
                            }
                        ]
                    },
                )
            )

        [near_fall] = migrated_events
        [migrated_action] = migrated_actions
        self.assertEqual(near_fall["task_type"], "near_fall_event")
        self.assertEqual(near_fall["label_role"], "positive")
        self.assertEqual(near_fall["event_type"], "near_fall")
        self.assertEqual(near_fall["event_subtype"], "stumble_recovery")
        self.assertEqual(near_fall["onset_frame"], 10)
        self.assertEqual(near_fall["recovery_frame"], 20)
        self.assertIsNone(near_fall["peak_frame"])
        self.assertIsNone(near_fall["impact_frame"])
        self.assertEqual(near_fall["linked_action_ids"], [migrated_action["label_id"]])
        self.assertEqual(migrated_action["linked_event_id"], near_fall["label_id"])
        self.assertEqual(near_fall["review_status"], "adjudicated")
        self.assertIn("reviewed C03 labels are near-fall positives", near_fall["note"])
        self.assertNotIn("NTU full-clip fall", near_fall["note"])
        self.assertEqual(report["reviewed_decision_matches"]["near_fall_positive"], 1)
        self.assertEqual(report["unmatched_reviewed_decisions"], [])

    def test_reviewed_action_mappings_skip_ignored_and_do_not_duplicate_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            actions = [
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_1",
                    action_id="A03",
                    action_name="controlled_sit_down",
                    start_frame=10,
                    end_frame=20,
                ),
                self._action(
                    source_path,
                    source_hash,
                    label_id="action_2",
                    action_id="A03",
                    action_name="controlled_sit_down",
                    start_frame=30,
                    end_frame=40,
                    quality="heavy_occlusion",
                ),
            ]

            report, _, migrated_events = self._write_migration_with_reviewed_decision(
                root,
                manifests=[manifest],
                actions=actions,
                events=[],
                decision_overrides={
                    "action_hard_negative_mappings": [
                        {
                            "task_type": "fall_event",
                            "action_ids": ["A03"],
                            "hard_negative_type": "controlled_sit_down",
                        },
                        {
                            "task_type": "near_fall_event",
                            "action_ids": ["A03"],
                            "hard_negative_type": "fast_but_controlled_sit",
                        },
                    ]
                },
            )

        negatives = [row for row in migrated_events if row["label_role"] == "negative"]
        self.assertEqual(len(negatives), 2)
        self.assertEqual(
            {(row["task_type"], row["hard_negative_type"]) for row in negatives},
            {
                ("fall_event", "controlled_sit_down"),
                ("near_fall_event", "fast_but_controlled_sit"),
            },
        )
        self.assertTrue(
            all("canonical action A03" in row["note"] for row in negatives)
        )
        self.assertTrue(
            all("NTU full-clip fall" not in row["note"] for row in negatives)
        )
        self.assertEqual(report["reviewed_decision_matches"]["action_hard_negative"], 2)

    def test_reviewed_video_override_distinguishes_bed_entry_from_lie_down(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_manifest, _ = self._manifest(root)
            first_manifest.update(
                asset_id="asset_ur",
                video_id="ur_fall_adl_10_cam0",
                source_group_id="ur_fall",
                original_event_id="ur_fall_adl_10",
            )
            second_video = root / "video_2.avi"
            second_video.write_bytes(b"video-2")
            second_manifest = {
                **first_manifest,
                "asset_id": "asset_other",
                "video_id": "other_lie_down",
                "path": second_video.as_posix(),
                "sha256": hashlib.sha256(second_video.read_bytes()).hexdigest(),
                "source_group_id": "other_source",
                "original_event_id": "other_lie_down",
            }
            source_path, source_hash = self._source(root)
            actions = []
            for index, manifest in enumerate((first_manifest, second_manifest), start=1):
                action = self._action(
                    source_path,
                    source_hash,
                    label_id=f"action_{index}",
                    action_id="A07",
                    action_name="controlled_lie_down",
                    start_frame=10,
                    end_frame=20,
                )
                action.update(asset_id=manifest["asset_id"], video_id=manifest["video_id"])
                actions.append(action)

            _, _, migrated_events = self._write_migration_with_reviewed_decision(
                root,
                manifests=[first_manifest, second_manifest],
                actions=actions,
                events=[],
                decision_overrides={
                    "action_hard_negative_mappings": [
                        {
                            "task_type": "fall_event",
                            "action_ids": ["A07"],
                            "hard_negative_type": "controlled_lie_down",
                        }
                    ],
                    "video_hard_negative_overrides": [
                        {
                            "task_type": "fall_event",
                            "action_id": "A07",
                            "video_ids": ["ur_fall_adl_10_cam0"],
                            "hard_negative_type": "bed_entry_or_exit",
                        }
                    ],
                },
            )

        hard_negative_by_video = {
            row["video_id"]: row["hard_negative_type"] for row in migrated_events
        }
        self.assertEqual(
            hard_negative_by_video,
            {
                "ur_fall_adl_10_cam0": "bed_entry_or_exit",
                "other_lie_down": "controlled_lie_down",
            },
        )

    def test_reviewed_event_mapping_does_not_convert_ignored_falls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root, eligible=False)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=10,
                end_frame=20,
            )

            report, _, migrated_events = self._write_migration_with_reviewed_decision(
                root,
                manifests=[manifest],
                actions=[action],
                events=[self._mapped_fall(action)],
                decision_overrides={
                    "event_hard_negative_mappings": [
                        {
                            "source_task_type": "fall_event",
                            "source_label_role": "positive",
                            "task_type": "near_fall_event",
                            "hard_negative_type": "progressed_to_fall",
                        }
                    ]
                },
            )

        self.assertEqual(len(migrated_events), 1)
        self.assertEqual(migrated_events[0]["training_tier"], "ignore")
        self.assertEqual(migrated_events[0]["label_role"], "positive")
        self.assertEqual(
            report["unmatched_reviewed_decisions"],
            [
                "fixture-reviewed-training-decision:"
                "event_hard_negative:fall_event:near_fall_event"
            ],
        )

    def test_reviewed_partial_occlusion_is_primary_only_for_event_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="A01",
                action_name="normal_walk",
                start_frame=10,
                end_frame=20,
                quality="partial_occlusion",
            )

            _, migrated_actions, migrated_events = (
                self._write_migration_with_reviewed_decision(
                    root,
                    manifests=[manifest],
                    actions=[action],
                    events=[],
                    decision_overrides={
                        "quality_hard_negative_mappings": [
                            {
                                "task_type": "fall_event",
                                "quality_flag": "partial_occlusion",
                                "action_ids": ["A01"],
                                "hard_negative_type": "occlusion_or_camera_motion",
                                "training_tier_policy": "reviewed_primary",
                            }
                        ]
                    },
                )
            )

        self.assertEqual(migrated_actions[0]["training_tier"], "auxiliary")
        self.assertEqual(migrated_events[0]["training_tier"], "primary")
        self.assertEqual(
            migrated_events[0]["hard_negative_type"],
            "occlusion_or_camera_motion",
        )

    def test_mapped_action_fall_is_auxiliary_without_independent_event_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._mapped_fall(action)],
            )

        self.assertEqual(len(result.event_labels), 1)
        event = result.event_labels[0]
        self.assertEqual(event["event_subtype"], "lateral")
        self.assertEqual(event["training_tier"], "auxiliary")
        self.assertEqual(event["subtype_training_tier"], "auxiliary")

    def test_seated_fall_mapping_preserves_seated_event_subtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_5",
                action_id="D05",
                action_name="seated_fall",
                start_frame=20,
                end_frame=60,
            )

            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._mapped_fall(action)],
            )

        [event] = result.event_labels
        self.assertEqual(event["event_type"], "fall")
        self.assertEqual(event["event_subtype"], "seated")
        self.assertEqual(event["training_tier"], "auxiliary")

    def test_toaga_normal_walk_migrates_as_auxiliary_source_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            manifest.update(
                {
                    "dataset": "toaga",
                    "subset": "walking",
                    "subject_id": "toaga_oaw01",
                    "source_group_id": "toaga_oaw01",
                    "original_event_id": "toaga_oaw01_walking",
                }
            )
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_toaga_1",
                action_id="A01",
                action_name="normal_walk",
                start_frame=0,
                end_frame=199,
            )
            action.update(
                {
                    "source": "toaga_official_walking",
                    "source_export_id": "toaga_official_walking_1234567890abcdef12345678",
                    "source_record_id": "toaga_official_walking:video_1:full_video",
                    "labeler": "official_toaga_source",
                    "subject_id": "toaga_oaw01",
                    "note": "Source-derived full-video normal-walk label.",
                }
            )
            for field in ("cvat_task_id", "cvat_track_id", "bbox_start", "bbox_end"):
                action.pop(field)

            result = migrate_v2_training_labels(
                manifest_rows=[manifest], action_rows=[action], event_rows=[]
            )

        migrated = result.action_labels[0]
        self.assertEqual(migrated["training_tier"], "auxiliary")
        self.assertIsNone(migrated["track_id"])
        self.assertEqual(migrated["boundary_precision"], "unknown")
        self.assertEqual(migrated["review_status"], "source_verified")
        self.assertEqual(
            migrated["source_refs"][0]["source_type"],
            "v2_toaga_official_walking",
        )

    def test_ntu_rgbd_manual_clip_label_migrates_as_exact_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            manifest.update(
                {
                    "dataset": "ntu_rgbd",
                    "subset": "setup_s006",
                    "subject_id": "ntu_rgbd_p016",
                    "source_group_id": "ntu_rgbd_subject_p016",
                    "original_event_id": "ntu_rgbd_s006_p016_r001_a042",
                }
            )
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_ntu_1",
                action_id="C03",
                action_name="stumble_recovery",
                start_frame=0,
                end_frame=199,
            )
            action.update(
                {
                    "source": "ntu_rgbd_manual_clip_label",
                    "source_export_id": "ntu_rgbd_clip_label_1234567890abcdef12345678",
                    "source_record_id": "ntu_rgbd_clip_label:video_1:full_video",
                    "labeler": "project_owner_ntu_rgbd_boundary_review_20260725",
                    "subject_id": "ntu_rgbd_p016",
                    "note": "Project-mapped NTU RGB+D full-video action label.",
                }
            )
            for field in ("cvat_task_id", "cvat_track_id", "bbox_start", "bbox_end"):
                action.pop(field)

            result = migrate_v2_training_labels(
                manifest_rows=[manifest], action_rows=[action], event_rows=[]
            )
            paths = self._paths(root)
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], [])
            validation = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

        migrated = result.action_labels[0]
        self.assertEqual(migrated["training_tier"], "primary")
        self.assertIsNone(migrated["track_id"])
        self.assertEqual(migrated["boundary_precision"], "exact")
        self.assertEqual(migrated["review_status"], "single_annotated")
        self.assertEqual(
            migrated["source_refs"][0]["source_type"],
            "v2_ntu_rgbd_manual_clip_label",
        )
        self.assertTrue(validation["valid"], validation["issues"])

    def test_unresolved_post_fall_state_ignores_parent_and_action_type_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path, source_hash = self._source(root)
            manifests = []
            actions = []
            events = []
            for index in range(10):
                video_path = root / f"video_{index}.avi"
                video_path.write_bytes(f"video-{index}".encode())
                manifests.append(
                    {
                        "asset_id": f"asset_{index}",
                        "video_id": f"video_{index}",
                        "path": video_path.as_posix(),
                        "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                        "fps_num": 25,
                        "fps_den": 1,
                        "frame_count": 200,
                        "duration_sec": 8.0,
                        "subject_id": f"subject_{index}",
                        "source_group_id": f"source_{index % 3}",
                        "original_event_id": f"original_{index}",
                        "eligibility": True,
                        "exclusion_reasons": [],
                    }
                )
                post_fall = self._action(
                    source_path,
                    source_hash,
                    label_id=f"action_{100 + index}",
                    action_id="D04",
                    action_name="long_static_after_fall",
                    start_frame=80,
                    end_frame=120,
                )
                post_fall.update(
                    asset_id=f"asset_{index}",
                    video_id=f"video_{index}",
                    subject_id=f"subject_{index}",
                    cvat_track_id=100 + index,
                )
                actions.append(post_fall)
                if index == 9:
                    continue
                fall = self._action(
                    source_path,
                    source_hash,
                    label_id=f"action_{index}",
                    action_id="D02",
                    action_name="lateral_fall",
                    start_frame=40,
                    end_frame=79,
                )
                fall.update(
                    asset_id=f"asset_{index}",
                    video_id=f"video_{index}",
                    subject_id=f"subject_{index}",
                    cvat_track_id=index,
                )
                actions.append(fall)
                mapped = self._mapped_fall(fall)
                mapped["label_id"] = f"event_mapped_{index}"
                events.append(mapped)
            result = migrate_v2_training_labels(
                manifest_rows=manifests, action_rows=actions, event_rows=events
            )

        migrated = next(
            row
            for row in result.action_labels
            if row["video_id"] == "video_9" and row["action_id"] == "D04"
        )
        self.assertEqual(migrated["training_tier"], "ignore")
        self.assertEqual(migrated["action_type_training_tier"], "ignore")
        self.assertEqual(result.report["unresolved_post_fall_states"], 1)

    def test_non_video_manifest_rows_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[
                    manifest,
                    {
                        "asset_id": "table_1",
                        "video_id": None,
                        "media_type": "table",
                    },
                ],
                action_rows=[action],
                event_rows=[self._mapped_fall(action)],
            )

        self.assertEqual(len(result.action_labels), 1)
        self.assertEqual(len(result.event_labels), 1)

    def test_action_type_tier_uses_episode_and_source_group_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path, source_hash = self._source(root)
            manifests = []
            actions = []
            for index in range(30):
                video_path = root / f"video_{index}.avi"
                video_path.write_bytes(f"video-{index}".encode())
                manifests.append(
                    {
                        "asset_id": f"asset_{index}",
                        "video_id": f"video_{index}",
                        "path": video_path.as_posix(),
                        "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                        "fps_num": 25,
                        "fps_den": 1,
                        "frame_count": 200,
                        "duration_sec": 8.0,
                        "subject_id": f"subject_{index}",
                        "source_group_id": f"source_{index % 3}",
                        "original_event_id": f"original_{index}",
                        "eligibility": True,
                        "exclusion_reasons": [],
                    }
                )
                action = self._action(
                    source_path,
                    source_hash,
                    label_id=f"action_{index}",
                    action_id="A01",
                    action_name="normal_walk",
                    start_frame=10,
                    end_frame=20,
                )
                action.update(
                    asset_id=f"asset_{index}",
                    video_id=f"video_{index}",
                    subject_id=f"subject_{index}",
                    cvat_track_id=index,
                )
                actions.append(action)

            result = migrate_v2_training_labels(
                manifest_rows=manifests, action_rows=actions, event_rows=[]
            )
            self.assertEqual(
                {row["action_type_training_tier"] for row in result.action_labels},
                {"primary"},
            )
            paths = self._paths(root)
            self._write_jsonl(paths["manifest"], manifests)
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], [])
            write_training_split_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                assignments_path=paths["split_assignments"],
                report_path=paths["split_report"],
                seed="action-type-coverage",
            )
            validation = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
                split_assignments_path=paths["split_assignments"],
                split_report_path=paths["split_report"],
            )
            self.assertTrue(validation["training_ready"]["action_type"])
            self.assertEqual(
                set(validation["split"]["partition_action_type_counts"]["normal_walk"]),
                {"train", "validation", "test"},
            )

            auxiliary = migrate_v2_training_labels(
                manifest_rows=manifests[:10], action_rows=actions[:10], event_rows=[]
            )
            self.assertEqual(
                {row["action_type_training_tier"] for row in auxiliary.action_labels},
                {"auxiliary"},
            )

            ignored = migrate_v2_training_labels(
                manifest_rows=manifests[:9], action_rows=actions[:9], event_rows=[]
            )
            self.assertEqual(
                {row["action_type_training_tier"] for row in ignored.action_labels},
                {"ignore"},
            )

    def test_round_trip_output_passes_v3_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v2_actions"], [action])
            self._write_jsonl(paths["v2_events"], [self._official_fall(root)])
            write_training_label_migration(
                manifest_path=paths["manifest"],
                action_labels_v2_path=paths["v2_actions"],
                event_labels_v2_path=paths["v2_events"],
                action_labels_v3_path=paths["v3_actions"],
                event_labels_v3_path=paths["v3_events"],
                report_path=paths["migration_report"],
            )
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["counts"]["event_positive"], 1)
        self.assertEqual(report["counts"]["near_fall_positive"], 0)
        self.assertIn("near_fall_has_no_positive_labels", report["warnings"])
        self.assertIn("fall_missing_hard_negatives", report["warnings"])

    def test_validator_rejects_tampered_action_type_training_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._official_fall(root)],
            )
            changed_actions = [
                dict(row, action_type_training_tier="primary")
                for row in result.action_labels
            ]
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], changed_actions)
            self._write_jsonl(paths["v3_events"], result.event_labels)
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("action_type_training_tier_mismatch", codes)
        self.assertFalse(report["valid"])

    def test_validator_rejects_near_fall_without_recovery_and_double_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="C04",
                action_name="rapid_support_contact",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest], action_rows=[action], event_rows=[]
            )
            near_fall = {
                **result.action_labels[0],
                "label_id": "eventv3_" + "a" * 24,
                "physical_event_id": "physical_" + "b" * 24,
                "task_type": "near_fall_event",
                "label_role": "positive",
                "event_type": "near_fall",
                "event_subtype": "rapid_support_recovery",
                "event_outcome": "recovered_without_fall",
                "hard_negative_type": None,
                "onset_frame": 45,
                "peak_frame": None,
                "impact_frame": None,
                "recovery_frame": None,
                "linked_action_ids": [result.action_labels[0]["label_id"]],
                "contact_evidence": "proxy",
                "subtype_training_tier": "primary",
                "training_tier": "primary",
                "review_status": "single_annotated",
            }
            for key in ("action_id", "action_family", "action_type", "action_attributes", "linked_event_id"):
                near_fall.pop(key)
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], [near_fall])
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("near_fall_recovery_required", codes)
        self.assertIn("near_fall_double_review_required", codes)
        self.assertIn("linked_action_not_reciprocal", codes)

    def test_training_ready_requires_all_primary_hard_negative_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._official_fall(root)],
            )
            base = result.event_labels[0]
            manifests = []
            positives = []
            for index in range(3):
                video_path = root / f"video_{index}.avi"
                video_path.write_bytes(f"video-{index}".encode())
                current_manifest = {
                    **manifest,
                    "asset_id": f"asset_video_{index}",
                    "video_id": f"video_{index}",
                    "path": video_path.as_posix(),
                    "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                    "source_group_id": f"source_video_{index}",
                    "original_event_id": f"original_video_{index}",
                }
                manifests.append(current_manifest)
                positives.append(
                    {
                        **base,
                        "label_id": f"eventv3_{100 + index:024x}",
                        "physical_event_id": f"physical_{100 + index:024x}",
                        "asset_id": current_manifest["asset_id"],
                        "video_id": current_manifest["video_id"],
                        "content_sha256": current_manifest["sha256"],
                        "source_group_id": current_manifest["source_group_id"],
                        "sample_group_id": f"samplegrp_{100 + index:024x}",
                        "linked_action_ids": [],
                    }
                )
            negatives = []
            for index, hard_negative_type in enumerate(sorted(FALL_HARD_NEGATIVES)):
                positive = positives[index % len(positives)]
                negative = {
                    **positive,
                    "label_id": f"eventv3_{index:024x}",
                    "physical_event_id": None,
                    "task_type": "fall_event",
                    "label_role": "negative",
                    "event_type": None,
                    "event_subtype": None,
                    "event_outcome": None,
                    "hard_negative_type": hard_negative_type,
                    "onset_frame": None,
                    "peak_frame": None,
                    "impact_frame": None,
                    "recovery_frame": None,
                    "linked_action_ids": [],
                    "subtype_training_tier": "ignore",
                    "review_status": "single_reviewed",
                    "source_refs": [
                        {
                            **positive["source_refs"][0],
                            "source_type": "manual_v3",
                            "source_label_id": f"negative_{index}",
                        }
                    ],
                }
                negatives.append(negative)

            self._write_jsonl(paths["manifest"], manifests)
            self._write_jsonl(paths["v3_actions"], [])
            self._write_jsonl(paths["v3_events"], [*positives, *negatives])
            write_training_split_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                assignments_path=paths["split_assignments"],
                report_path=paths["split_report"],
                seed="test-split",
            )
            complete = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
                split_assignments_path=paths["split_assignments"],
                split_report_path=paths["split_report"],
            )
            self._write_jsonl(paths["v3_events"], [*positives, *negatives[:-1]])
            write_training_split_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                assignments_path=paths["split_assignments"],
                report_path=paths["split_report"],
                seed="test-split",
                overwrite=True,
            )
            incomplete = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
                split_assignments_path=paths["split_assignments"],
                split_report_path=paths["split_report"],
            )

        self.assertTrue(complete["valid"], complete["issues"])
        self.assertTrue(complete["training_ready"]["fall_event"])
        self.assertEqual(complete["hard_negative_coverage"]["fall_event"]["missing"], [])
        self.assertFalse(incomplete["training_ready"]["fall_event"])
        self.assertEqual(len(incomplete["hard_negative_coverage"]["fall_event"]["missing"]), 1)

    def test_validator_rejects_stale_split_after_label_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._official_fall(root)],
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], result.event_labels)
            write_training_split_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                assignments_path=paths["split_assignments"],
                report_path=paths["split_report"],
                seed="test-split",
            )
            changed_actions = [dict(row, note="changed after split") for row in result.action_labels]
            self._write_jsonl(paths["v3_actions"], changed_actions)
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
                split_assignments_path=paths["split_assignments"],
                split_report_path=paths["split_report"],
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("split_input_hash_mismatch", codes)
        self.assertFalse(report["valid"])
        self.assertFalse(report["training_ready"]["fall_event"])

    def test_validator_rejects_tampered_split_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._official_fall(root)],
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], result.event_labels)
            write_training_split_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                assignments_path=paths["split_assignments"],
                report_path=paths["split_report"],
                seed="test-split",
            )
            split_report = json.loads(paths["split_report"].read_text(encoding="utf-8"))
            split_report["split_id"] = "splitv3_" + "0" * 24
            paths["split_report"].write_text(
                json.dumps(split_report, sort_keys=True), encoding="utf-8"
            )
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
                split_assignments_path=paths["split_assignments"],
                split_report_path=paths["split_report"],
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("split_id_mismatch", codes)
        self.assertFalse(report["valid"])

    def test_validator_rejects_non_manual_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            manifest, _ = self._manifest(root)
            source_path, source_hash = self._source(root)
            action = self._action(
                source_path,
                source_hash,
                label_id="action_1",
                action_id="D02",
                action_name="lateral_fall",
                start_frame=45,
                end_frame=75,
            )
            result = migrate_v2_training_labels(
                manifest_rows=[manifest],
                action_rows=[action],
                event_rows=[self._official_fall(root)],
            )
            negative = {
                **result.event_labels[0],
                "label_id": "eventv3_" + "c" * 24,
                "physical_event_id": None,
                "label_role": "negative",
                "event_type": None,
                "event_subtype": None,
                "event_outcome": None,
                "hard_negative_type": "controlled_sit_down",
                "onset_frame": None,
                "linked_action_ids": [],
                "subtype_training_tier": "ignore",
                "review_status": "source_verified",
            }
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["v3_actions"], result.action_labels)
            self._write_jsonl(paths["v3_events"], [result.event_labels[0], negative])
            report = validate_training_labels_v3(
                manifest_path=paths["manifest"],
                action_labels_path=paths["v3_actions"],
                event_labels_path=paths["v3_events"],
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("negative_manual_confirmation_required", codes)
        self.assertIn("negative_review_required", codes)
        self.assertFalse(report["training_ready"]["fall_event"])

    def test_schema_files_are_closed_and_versioned(self) -> None:
        action_schema = json.loads(ACTION_SCHEMA.read_text(encoding="utf-8"))
        event_schema = json.loads(EVENT_SCHEMA.read_text(encoding="utf-8"))

        self.assertEqual(action_schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(event_schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(action_schema["additionalProperties"])
        self.assertFalse(event_schema["additionalProperties"])
        self.assertIn("training_tier", action_schema["required"])
        self.assertIn("action_type_training_tier", action_schema["required"])
        self.assertIn("label_role", event_schema["required"])

    def test_repository_reviewed_decision_is_bound_to_current_v2_actions(self) -> None:
        decision = json.loads(REVIEWED_DECISION.read_text(encoding="utf-8"))
        action_labels = ROOT / "data/annotations/fall_risk/action_labels.jsonl"

        self.assertEqual(
            decision["action_labels_sha256"],
            hashlib.sha256(action_labels.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            decision["source_task_id"],
            "019fcb65-e9d1-7400-b018-aba2d2fc9941",
        )


if __name__ == "__main__":
    unittest.main()
