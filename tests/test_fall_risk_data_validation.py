import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from elderly_monitoring.modules.fall_risk.annotation_validation import (
    load_validation_config,
    validate_fall_risk_data,
    write_validation_report,
)


class FallRiskDataValidationTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
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
            "license_id": "CC-BY-NC-SA-3.0" if eligibility else "license_unknown",
            "source_uri": "https://example.test/le2i",
            "consent_id": None,
            "review_status": "pending",
            "eligibility": eligibility,
            "exclusion_reasons": [] if eligibility else ["license_unknown"],
        }

    def _action(self, video_path: Path, **updates: object) -> dict:
        source_path = video_path.parent / "annotations.xml"
        source_path.touch(exist_ok=True)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        row = {
            "label_id": "action_1",
            "source_record_id": "cvat:export:task:1:track:1",
            "source_annotation_path": source_path.as_posix(),
            "source_annotation_sha256": source_sha256,
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
            "review_status": "reviewed",
            "eligibility": True,
            "review_evidence_ids": ["review_1"],
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
        source_path = video_path.parent / "annotations.xml"
        source_path.touch(exist_ok=True)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        row = {
            "label_id": self._stable_id(
                "event", "action_1", "fall-action-event-v1"
            ),
            "source_record_id": "mapped:action_1",
            "source_annotation_path": source_path.as_posix(),
            "source_annotation_sha256": source_sha256,
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
            "review_status": "reviewed",
            "eligibility": True,
            "review_evidence_ids": ["review_2"],
            "note": "mapped from action",
            "source_action_id": "D01",
            "source_action_name": "forward_fall",
            "source_action_label_id": "action_1",
            "mapping_version": "fall-action-event-v1",
            "cvat_task_id": "1",
            "cvat_track_id": 1,
        }
        row.update(updates)
        return row

    def _record_sha256(self, row: dict) -> str:
        payload = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _stable_id(self, prefix: str, *parts: object) -> str:
        payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
        return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"

    def _review(self, row: dict, review_id: str) -> dict:
        label_id = row["label_id"]
        label_type = (
            "action"
            if label_id.startswith("action")
            else "risk"
            if label_id.startswith("risk")
            else "event"
        )
        return {
            "review_id": review_id,
            "label_id": label_id,
            "label_type": label_type,
            "reviewer_id": "reviewer_01",
            "decision": "approve",
            "reviewed_at": "2026-07-15T00:00:00Z",
            "reason_code": "verified_against_source",
            "note": "",
            "result_record_sha256": self._record_sha256(row),
        }

    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "manifest": root / "manifest.jsonl",
            "actions": root / "actions.jsonl",
            "events": root / "events.jsonl",
            "risk": root / "risk.jsonl",
            "profiles": root / "profiles.json",
            "reviews": root / "reviews.jsonl",
        }

    def _validate(self, paths: dict[str, Path], *, mode: str = "formal") -> dict:
        return validate_fall_risk_data(
            manifest_path=paths["manifest"],
            action_labels_path=paths["actions"],
            event_labels_path=paths["events"],
            risk_labels_path=paths["risk"],
            subject_profiles_path=paths["profiles"],
            review_log_path=paths["reviews"],
            mode=mode,
        )

    def test_empty_risk_review_and_profile_templates_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                json.dumps(
                    {"schema_version": "fall-risk-subject-profiles-v1", "subjects": []}
                ),
                encoding="utf-8",
            )

            report = self._validate(paths)

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["counts"]["risk_labels"], 0)

    def test_formal_mode_blocks_pending_uncertain_and_license_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video, eligibility=False)])
            self._write_jsonl(
                paths["actions"],
                [self._action(video, action_id="U01", action_name="unable_to_judge", event_type="uncertain", review_status="pending", note="occluded")],
            )
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("formal_manifest_ineligible", codes)
        self.assertIn("formal_review_status", codes)
        self.assertIn("formal_uncertain", codes)

    def test_high_risk_and_fall_require_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            action = self._action(
                video, review_evidence_ids=["review_1", "review_1b"]
            )
            event = self._event(
                video, review_evidence_ids=["review_2", "review_2b"]
            )
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            blocked = self._validate(paths)
            self._write_jsonl(
                paths["reviews"],
                [
                    self._review(action, "review_1"),
                    {
                        **self._review(action, "review_1b"),
                        "reviewer_id": "reviewer_02",
                    },
                    self._review(event, "review_2"),
                    {
                        **self._review(event, "review_2b"),
                        "reviewer_id": "reviewer_02",
                    },
                ],
            )
            accepted = self._validate(paths)

            tampered_reviews = [
                self._review(action, "review_1"),
                {
                    **self._review(action, "review_1b"),
                    "reviewer_id": "reviewer_02",
                },
                self._review(event, "review_2"),
                {
                    **self._review(event, "review_2b"),
                    "reviewer_id": "reviewer_02",
                },
            ]
            tampered_reviews[2]["result_record_sha256"] = "0" * 64
            self._write_jsonl(paths["reviews"], tampered_reviews)
            tampered = self._validate(paths)

        self.assertIn(
            "missing_review_evidence",
            {issue["code"] for issue in blocked["issues"]},
        )
        self.assertTrue(accepted["valid"], accepted["issues"])
        self.assertFalse(tampered["valid"])
        self.assertIn(
            "review_result_hash_mismatch",
            {issue["code"] for issue in tampered["issues"]},
        )

    def test_time_mismatch_and_manual_risk_score_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            action = self._action(video, end_time=1.8)
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(
                paths["risk"],
                [
                    {
                        "label_id": "risk_1",
                        "asset_id": "asset_video_1",
                        "video_id": "le2i_home_01_video_1",
                        "task_type": "functional_proxy",
                        "subject_id": "unknown",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "risk_level": 3,
                        "risk_score": 0.82,
                        "risk_factors": ["fall"],
                        "label_source": "manual_consensus",
                        "review_status": "final",
                        "eligibility": True,
                        "review_evidence_ids": [],
                    }
                ],
            )
            self._write_jsonl(paths["reviews"], [self._review(action, "review_1")])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths, mode="audit")

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("time_frame_mismatch", codes)
        self.assertIn("manual_risk_score_forbidden", codes)

    def test_mapped_event_must_preserve_action_provenance_and_severity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            action = self._action(
                video, review_evidence_ids=["review_1", "review_1b"]
            )
            event = self._event(
                video,
                severity=0,
                source_export_id="different_export",
                source_action_name="normal_walk",
                review_evidence_ids=["review_2", "review_2b"],
            )
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(
                paths["reviews"],
                [
                    self._review(action, "review_1"),
                    {
                        **self._review(action, "review_1b"),
                        "reviewer_id": "reviewer_02",
                    },
                    self._review(event, "review_2"),
                    {
                        **self._review(event, "review_2b"),
                        "reviewer_id": "reviewer_02",
                    },
                ],
            )
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("source_action_mismatch", codes)
        self.assertIn("mapped_event_severity_mismatch", codes)

    def test_unresolved_conflict_blocks_an_earlier_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            action = self._action(video)
            event = self._event(video)
            conflict = self._review(action, "review_3")
            conflict["decision"] = "conflict"
            conflict["reviewed_at"] = "2026-07-15T01:00:00Z"
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(
                paths["reviews"],
                [
                    self._review(action, "review_1"),
                    self._review(event, "review_2"),
                    conflict,
                ],
            )
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths)

        self.assertFalse(report["valid"])
        self.assertIn(
            "unresolved_review_decision",
            {issue["code"] for issue in report["issues"]},
        )

    def test_adjudication_must_bind_the_previous_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            action = self._action(
                video, review_evidence_ids=["review_3", "review_5"]
            )
            event = self._event(
                video, review_evidence_ids=["review_4", "review_6"]
            )
            approval = self._review(action, "review_1")
            approval["result_record_sha256"] = "1" * 64
            conflict = self._review(action, "review_2")
            conflict.update(
                {
                    "decision": "conflict",
                    "reviewer_id": "reviewer_02",
                    "reviewed_at": "2026-07-15T01:00:00Z",
                    "previous_record_sha256": "1" * 64,
                    "result_record_sha256": "2" * 64,
                    "supersedes_review_id": "review_1",
                }
            )
            adjudication = self._review(action, "review_3")
            adjudication.update(
                {
                    "decision": "adjudicate",
                    "reviewer_id": "reviewer_03",
                    "reviewed_at": "2026-07-15T02:00:00Z",
                    "previous_record_sha256": "0" * 64,
                    "supersedes_review_id": "review_2",
                }
            )
            event_approval = self._review(event, "review_4")
            second_action_approval = {
                **self._review(action, "review_5"),
                "reviewer_id": "reviewer_04",
            }
            second_event_approval = {
                **self._review(event, "review_6"),
                "reviewer_id": "reviewer_02",
            }
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [event])
            self._write_jsonl(paths["risk"], [])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )
            reviews = [
                approval,
                conflict,
                adjudication,
                event_approval,
                second_action_approval,
                second_event_approval,
            ]
            self._write_jsonl(paths["reviews"], reviews)
            bad_link = self._validate(paths)

            adjudication["previous_record_sha256"] = "2" * 64
            self._write_jsonl(paths["reviews"], reviews)
            accepted = self._validate(paths)

            adjudication["decision"] = "approve"
            self._write_jsonl(paths["reviews"], reviews)
            wrong_transition = self._validate(paths)

            adjudication["decision"] = "adjudicate"
            adjudication["reviewer_id"] = "reviewer_02"
            self._write_jsonl(paths["reviews"], reviews)
            same_person = self._validate(paths)

            adjudication["reviewer_id"] = "reviewer_03"
            adjudication["reviewed_at"] = "2026-07-15T00:30:00Z"
            self._write_jsonl(paths["reviews"], reviews)
            reverse_time = self._validate(paths)

        self.assertIn(
            "review_previous_hash_mismatch",
            {issue["code"] for issue in bad_link["issues"]},
        )
        self.assertTrue(accepted["valid"], accepted["issues"])
        self.assertIn(
            "invalid_review_transition",
            {issue["code"] for issue in wrong_transition["issues"]},
        )
        self.assertIn(
            "non_independent_adjudicator",
            {issue["code"] for issue in same_person["issues"]},
        )
        self.assertIn(
            "review_timestamp_not_increasing",
            {issue["code"] for issue in reverse_time["issues"]},
        )

    def test_two_review_ids_from_one_reviewer_are_not_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            action = self._action(
                video, review_evidence_ids=["review_1", "review_2"]
            )
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(
                paths["reviews"],
                [self._review(action, "review_1"), self._review(action, "review_2")],
            )
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths)

        self.assertFalse(report["valid"])
        self.assertIn(
            "insufficient_independent_reviewers",
            {issue["code"] for issue in report["issues"]},
        )

    def test_non_video_risk_label_passes_with_two_independent_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            table = root / "functional.csv"
            table.write_text("score\n1\n", encoding="utf-8")
            manifest = self._manifest(table)
            manifest.update(
                {
                    "asset_id": "asset_functional_1",
                    "media_type": "table",
                    "modality": "functional_table",
                }
            )
            manifest.pop("video_id")
            for field in (
                "fps_num",
                "fps_den",
                "fps",
                "frame_count",
                "duration_sec",
                "width",
                "height",
            ):
                manifest[field] = None
            risk = {
                "label_id": "risk_1",
                "asset_id": "asset_functional_1",
                "task_type": "functional_proxy",
                "subject_id": "subject_001",
                "start_time": 0.0,
                "end_time": 0.0,
                "risk_level": 2,
                "risk_factors": ["functional_proxy"],
                "label_source": "manual_consensus",
                "review_status": "final",
                "eligibility": True,
                "review_evidence_ids": ["review_1", "review_2"],
            }
            self._write_jsonl(paths["manifest"], [manifest])
            self._write_jsonl(paths["actions"], [])
            self._write_jsonl(paths["events"], [])
            risk["video_id"] = "wrong-video"
            self._write_jsonl(paths["risk"], [risk])
            self._write_jsonl(
                paths["reviews"],
                [
                    self._review(risk, "review_1"),
                    {
                        **self._review(risk, "review_2"),
                        "reviewer_id": "reviewer_02",
                    },
                ],
            )
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )
            mismatched_video = self._validate(paths)

            risk.pop("video_id")
            self._write_jsonl(paths["risk"], [risk])
            self._write_jsonl(
                paths["reviews"],
                [
                    self._review(risk, "review_1"),
                    {
                        **self._review(risk, "review_2"),
                        "reviewer_id": "reviewer_02",
                    },
                ],
            )
            report = self._validate(paths)

        self.assertIn(
            "manifest_video_mismatch",
            {issue["code"] for issue in mismatched_video["issues"]},
        )
        self.assertTrue(report["valid"], report["issues"])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            paths["actions"].write_text(
                '{"label_id":"one","label_id":"two"}\n', encoding="utf-8"
            )
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths, mode="audit")

        self.assertIn("invalid_jsonl", {issue["code"] for issue in report["issues"]})

    def test_formal_profile_requires_consent_and_rejects_contact_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-subject-profiles-v1",
                        "subjects": [
                            {
                                "subject_id": "subject_001",
                                "profile_version": "v1",
                                "profile_source": "manual_verified",
                                "review_status": "pending",
                                "consent_id": None,
                                "features": {
                                    "contact": "private@example.invalid"
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = self._validate(paths)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("potential_identity_data", codes)
        self.assertIn("formal_profile_review_status", codes)
        self.assertIn("formal_consent_missing", codes)

    def test_validation_cli_writes_report_and_refuses_overwrite(self) -> None:
        repo = Path(__file__).parents[1]
        script = repo / "scripts/annotation/validate_fall_risk_labels.py"
        config = repo / "configs/data/fall_risk_label_validation_v1.yaml"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )
            report_path = root / "report.json"
            command = [
                sys.executable,
                str(script),
                "--manifest",
                str(paths["manifest"]),
                "--action-labels",
                str(paths["actions"]),
                "--event-labels",
                str(paths["events"]),
                "--risk-labels",
                str(paths["risk"]),
                "--subject-profiles",
                str(paths["profiles"]),
                "--review-log",
                str(paths["reviews"]),
                "--config",
                str(config),
                "--mode",
                "formal",
                "--report-output",
                str(report_path),
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            first_report = report_path.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            second_report = report_path.read_bytes()

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(json.loads(first_report)["valid"])
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(second_report, first_report)

    def test_approval_without_result_hash_is_not_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._paths(root)
            video = root / "video.avi"
            video.touch()
            action = self._action(video)
            review = self._review(action, "review_1")
            del review["result_record_sha256"]
            self._write_jsonl(paths["manifest"], [self._manifest(video)])
            self._write_jsonl(paths["actions"], [action])
            self._write_jsonl(paths["events"], [])
            self._write_jsonl(paths["risk"], [])
            self._write_jsonl(paths["reviews"], [review])
            paths["profiles"].write_text(
                '{"schema_version":"fall-risk-subject-profiles-v1","subjects":[]}',
                encoding="utf-8",
            )

            report = self._validate(paths)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("missing_review_result_hash", codes)
        self.assertIn("missing_review_evidence", codes)

    def test_validation_report_no_overwrite_is_race_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"
            competing_payload = b'{"owner":"other-process"}\n'
            real_link = os.link

            def competing_link(source: str | os.PathLike, target: str | os.PathLike) -> None:
                Path(target).write_bytes(competing_payload)
                real_link(source, target)

            with mock.patch(
                "elderly_monitoring.modules.fall_risk.annotation_validation.os.link",
                side_effect=competing_link,
            ):
                with self.assertRaises(FileExistsError):
                    write_validation_report({"valid": True}, output)

            self.assertEqual(output.read_bytes(), competing_payload)

    def test_validation_config_cannot_weaken_v1_governance(self) -> None:
        repo = Path(__file__).parents[1]
        standard_path = repo / "configs/data/fall_risk_label_validation_v1.yaml"
        base = yaml.safe_load(standard_path.read_text(encoding="utf-8"))
        mutations = {
            "large tolerance": lambda value: value.update(time_tolerance_sec=1000),
            "one reviewer": lambda value: value.update(
                minimum_independent_reviewers=1
            ),
            "pending formal": lambda value: value["formal_review_statuses"].append(
                "pending"
            ),
            "remove pending blocker": lambda value: value[
                "ineligible_review_statuses"
            ].remove("pending"),
            "remove unknown license": lambda value: value[
                "unknown_license_values"
            ].remove("unknown"),
            "allow revise": lambda value: value[
                "review_approval_decisions"
            ].append("revise"),
            "remove high risk D": lambda value: value[
                "high_risk_action_prefixes"
            ].remove("D"),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "validation.yaml"
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = json.loads(json.dumps(base))
                    mutate(value)
                    path.write_text(
                        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        load_validation_config(path)


if __name__ == "__main__":
    unittest.main()
