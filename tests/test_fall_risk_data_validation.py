from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotation_validation import (
    DEFAULT_CONFIG,
    validate_fall_risk_data,
    write_validation_report,
)


class FallRiskDataValidationTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _manifest(self, video_path: Path, *, eligibility: bool = True) -> dict:
        return {
            "asset_id": "asset_video_1",
            "dataset": "le2i_imvia",
            "subset": "home_01",
            "path": video_path.as_posix(),
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "media_type": "video",
            "modality": "rgb_video",
            "video_id": "le2i_home_01_video_1",
            "fps_num": 25,
            "fps_den": 1,
            "fps": 25.0,
            "frame_count": 100,
            "duration_sec": 4.0,
            "width": 320,
            "height": 240,
            "subject_id": "unknown",
            "source_group_id": "le2i_home_01_video_1",
            "original_event_id": "le2i_home_01_video_1",
            "scene_region": "home",
            "view": "fixed_camera",
            "label_source": "official_annotation",
            "annotation_path": None,
            "source_uri": "https://example.test/le2i",
            "consent_id": None,
            "eligibility": eligibility,
            "exclusion_reasons": [] if eligibility else ["duplicate_content"],
        }

    def _action(self, video_path: Path, **updates: object) -> dict:
        source_path = video_path.parent / "annotations.xml"
        source_path.touch(exist_ok=True)
        row = {
            "label_id": "action_1",
            "source_record_id": "cvat:export:task:1:track:1",
            "source_annotation_path": source_path.as_posix(),
            "source_annotation_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_export_id": "cvat_export_1",
            "asset_id": "asset_video_1",
            "video_id": "le2i_home_01_video_1",
            "file_path": video_path.as_posix(),
            "subject_id": "unknown",
            "scene": "home",
            "view": "fixed_camera",
            "action_id": "D01",
            "action_name": "forward_fall",
            "event_type": "fall",
            "start_time": 0.4,
            "end_time": 0.8,
            "start_frame": 10,
            "end_frame": 20,
            "frame_index_base": 0,
            "labeler": "unknown",
            "quality": "clear",
            "note": "",
            "source": "cvat",
            "cvat_task_id": "1",
            "cvat_track_id": 1,
            "bbox_start": [1.0, 2.0, 3.0, 4.0],
            "bbox_end": [1.0, 2.0, 3.0, 4.0],
        }
        row.update(updates)
        return row

    def _event(self, video_path: Path, **updates: object) -> dict:
        action = self._action(video_path)
        row = {
            "label_id": self._stable_id("event", action["label_id"], "fall-action-event-v2"),
            "source_record_id": "mapped:action_1",
            "source_annotation_path": action["source_annotation_path"],
            "source_annotation_sha256": action["source_annotation_sha256"],
            "source_export_id": "cvat_export_1",
            "asset_id": "asset_video_1",
            "video_id": "le2i_home_01_video_1",
            "event_type": "fall",
            "start_time": 0.4,
            "end_time": 0.8,
            "start_frame": 10,
            "end_frame": 20,
            "frame_index_base": 0,
            "severity": 4,
            "label_source": "cvat_action_mapping",
            "note": "mapped from action",
            "source_action_id": "D01",
            "source_action_name": "forward_fall",
            "source_action_label_id": "action_1",
            "mapping_version": "fall-action-event-v2",
            "cvat_task_id": "1",
            "cvat_track_id": 1,
        }
        row.update(updates)
        return row

    def _stable_id(self, prefix: str, *parts: object) -> str:
        payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
        return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"

    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "manifest": root / "manifest.jsonl",
            "actions": root / "actions.jsonl",
            "events": root / "events.jsonl",
            "risk": root / "risk.jsonl",
            "profiles": root / "profiles.json",
        }

    def _write_fixture(
        self,
        root: Path,
        *,
        actions: list[dict] | None = None,
        events: list[dict] | None = None,
        manifest_eligibility: bool = True,
        profiles: dict | None = None,
    ) -> tuple[dict[str, Path], Path]:
        paths = self._paths(root)
        video = root / "video.avi"
        video.touch()
        self._write_jsonl(paths["manifest"], [self._manifest(video, eligibility=manifest_eligibility)])
        self._write_jsonl(paths["actions"], actions or [])
        self._write_jsonl(paths["events"], events or [])
        self._write_jsonl(paths["risk"], [])
        paths["profiles"].write_text(
            json.dumps(
                profiles
                or {"schema_version": "fall-risk-subject-profiles-v2", "subjects": []},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return paths, video

    def _validate(self, paths: dict[str, Path], *, mode: str = "formal") -> dict:
        return validate_fall_risk_data(
            manifest_path=paths["manifest"],
            action_labels_path=paths["actions"],
            event_labels_path=paths["events"],
            risk_labels_path=paths["risk"],
            subject_profiles_path=paths["profiles"],
            mode=mode,
        )

    def test_formal_accepts_structured_labels_without_review_or_license_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            self._write_jsonl(paths["actions"], [self._action(video)])
            self._write_jsonl(paths["events"], [self._event(video)])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(report["formal_ready"])
        self.assertEqual(
            set(report["input_sha256"]),
            {
                "manifest",
                "action_labels",
                "event_labels",
                "risk_labels",
                "subject_profiles",
                "validation_config",
            },
        )

    def test_formal_accepts_explicit_internal_authorization_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            manifest = self._manifest(video)
            manifest.update(
                {
                    "source_uri": "internal://authorization/internal_pre_vfallp_fixture",
                    "provenance_status": "internal_authorized_source_unverified",
                    "internal_authorization": {
                        "authorization_id": "internal_pre_vfallp_fixture",
                        "approval_reference": "test_internal_authorization",
                        "approved_at": "2026-07-22",
                        "evidence": {
                            "kind": "cvat_export_archive",
                            "name": "fixture.zip",
                            "sha256": "a" * 64,
                        },
                        "authorized_video_count": 1,
                    },
                }
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["actions"], [self._action(video)])
            self._write_jsonl(paths["events"], [self._event(video)])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(report["formal_ready"])

    def test_formal_accepts_project_collection_training_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            decision_path = root / "fall_tiktok_collection_decision.json"
            decision_path.write_text("{}\n", encoding="utf-8")
            manifest = self._manifest(video)
            manifest.update(
                {
                    "dataset": "fall_tiktok",
                    "source_group_id": "fall_tiktok_project_collection_pool",
                    "source_uri": "internal://collection/fall_tiktok_fixture",
                    "provenance_status": "project_collected_training_authorized",
                    "collection_status": "project_collected",
                    "training_use": "authorized",
                    "redistribution_use": "not_authorized_by_this_decision",
                    "consent_status": "not_recorded",
                    "subject_grouping_status": "unknown",
                    "collection_decision_id": "fixture-decision",
                    "collection_decision_path": decision_path.as_posix(),
                    "collection_decision_sha256": hashlib.sha256(
                        decision_path.read_bytes()
                    ).hexdigest(),
                    "collection_decided_at": "2026-07-28",
                    "collection_decided_by": "project_owner",
                }
            )
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["actions"], [self._action(video)])
            self._write_jsonl(paths["events"], [self._event(video)])

            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(report["formal_ready"])

    def test_formal_accepts_controlled_squat_as_normal_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            action = self._action(
                video,
                action_id="A05",
                action_name="controlled_squat",
                event_type="normal_activity",
            )
            event = self._event(
                video,
                event_type="normal_activity",
                severity=0,
                source_action_id="A05",
                source_action_name="controlled_squat",
            )
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(report["formal_ready"])

    def test_formal_accepts_controlled_bend_as_normal_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            action = self._action(
                video,
                action_id="A06",
                action_name="controlled_bend",
                event_type="normal_activity",
            )
            event = self._event(
                video,
                event_type="normal_activity",
                severity=0,
                source_action_id="A06",
                source_action_name="controlled_bend",
            )
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(report["formal_ready"])

    def test_formal_blocks_uncertain_labels_and_manifest_technical_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root, manifest_eligibility=False)
            action = self._action(
                video,
                action_id="U01",
                action_name="unable_to_judge",
                event_type="uncertain",
                note="occluded",
            )
            self._write_jsonl(paths["actions"], [action])
            report = self._validate(paths)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("formal_uncertain", codes)
        self.assertIn("formal_manifest_ineligible", codes)

    def test_source_hash_and_timeline_are_still_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            action = self._action(video, end_time=1.8, source_annotation_sha256="0" * 64)
            self._write_jsonl(paths["actions"], [action])
            report = self._validate(paths, mode="audit")

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("time_frame_mismatch", codes)
        self.assertIn("source_checksum_mismatch", codes)

    def test_toaga_official_walking_action_is_valid_without_cvat_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            toaga_action = self._action(
                video,
                action_id="A01",
                action_name="normal_walk",
                event_type="normal_activity",
                source="toaga_official_walking",
                source_export_id="toaga_official_walking_1234567890abcdef12345678",
                source_record_id="toaga_official_walking:video:full_video",
                labeler="official_toaga_source",
                note="Source-derived full-video normal-walk label.",
            )
            for field in ("cvat_task_id", "cvat_track_id", "bbox_start", "bbox_end"):
                toaga_action.pop(field)
            self._write_jsonl(paths["actions"], [toaga_action])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])

    def test_ntu_rgbd_clip_action_is_valid_without_cvat_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            ntu_action = self._action(
                video,
                action_id="C03",
                action_name="stumble_recovery",
                event_type="near_fall",
                source="ntu_rgbd_clip_label",
                source_export_id="ntu_rgbd_clip_label_1234567890abcdef12345678",
                source_record_id="ntu_rgbd_clip_label:video:full_video",
                labeler="project_ntu_rgbd_clip_map",
                note="Project-mapped NTU RGB+D full-video action label.",
            )
            for field in ("cvat_task_id", "cvat_track_id", "bbox_start", "bbox_end"):
                ntu_action.pop(field)
            self._write_jsonl(paths["actions"], [ntu_action])
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])

    def test_cvat_action_still_requires_cvat_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, video = self._write_fixture(root)
            cvat_action = self._action(video)
            cvat_action.pop("bbox_end")
            self._write_jsonl(paths["actions"], [cvat_action])
            report = self._validate(paths, mode="audit")

        self.assertIn("missing_fields", {issue["code"] for issue in report["issues"]})

    def test_profiles_require_consent(self) -> None:
        profile = {
            "schema_version": "fall-risk-subject-profiles-v2",
            "subjects": [
                {
                    "subject_id": "subject_01",
                    "profile_version": "v1",
                    "profile_source": "collection",
                    "consent_id": "consent_01",
                    "features": {"age_group": "70_79"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = self._write_fixture(Path(tmpdir), profiles=profile)
            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, _ = self._write_fixture(root)
            paths["actions"].write_text('{"label_id":"a","label_id":"b"}\n', encoding="utf-8")
            report = self._validate(paths, mode="audit")

        self.assertIn("invalid_jsonl", {issue["code"] for issue in report["issues"]})

    def test_validation_config_contains_only_schema_and_timeline_policy(self) -> None:
        self.assertEqual(
            set(DEFAULT_CONFIG),
            {"schema_version", "time_tolerance_sec"},
        )

    def test_validation_report_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"
            write_validation_report({"valid": True}, output)
            with self.assertRaises(FileExistsError):
                write_validation_report({"valid": False}, output)


if __name__ == "__main__":
    unittest.main()
