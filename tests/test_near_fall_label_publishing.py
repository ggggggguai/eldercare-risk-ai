from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.near_fall_label_publishing import (
    publish_near_fall_manual_labels_v3,
)
from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    validate_training_labels_v3,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION_SCHEMA = ROOT / "configs/data/fall_risk_action_label_schema_v3.json"
EVENT_SCHEMA = ROOT / "configs/data/fall_risk_event_label_schema_v3.json"


class NearFallLabelPublishingTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path | dict]:
        root.mkdir(parents=True, exist_ok=True)
        source = root / "manual_review_export.json"
        source.write_text('{"reviewed":true}\n', encoding="utf-8")
        manifest_row = {
            "asset_id": "asset_video_1",
            "video_id": "video_1",
            "path": (root / "video.avi").as_posix(),
            "sha256": hashlib.sha256(b"video").hexdigest(),
            "fps_num": 25,
            "fps_den": 1,
            "frame_count": 200,
            "duration_sec": 8.0,
            "subject_id": "subject_1",
            "source_group_id": "source_subject_1",
            "original_event_id": "original_event_1",
            "eligibility": True,
            "exclusion_reasons": [],
        }
        Path(manifest_row["path"]).write_bytes(b"video")
        manifest = root / "manifest.jsonl"
        base = root / "event_labels_v3.jsonl"
        decisions = root / "near_fall_decisions.jsonl"
        actions = root / "action_labels_v3.jsonl"
        output = root / "event_labels_v3.near_fall_candidate.jsonl"
        report = root / "publication.json"
        self._write_jsonl(manifest, [manifest_row])
        self._write_jsonl(base, [])
        self._write_jsonl(actions, [])
        return {
            "source": source,
            "manifest_row": manifest_row,
            "manifest": manifest,
            "base": base,
            "decisions": decisions,
            "actions": actions,
            "output": output,
            "report": report,
        }

    def _positive(self, fixture: dict[str, Path | dict]) -> dict:
        source = fixture["source"]
        assert isinstance(source, Path)
        return {
            "schema_version": "near-fall-manual-decision-v1",
            "decision_id": "near_fall_positive_001",
            "source_type": "manual_near_fall_v1",
            "video_id": "video_1",
            "track_id": "person_1",
            "label_role": "positive",
            "start_frame": 10,
            "end_frame_exclusive": 41,
            "frame_index_base": 0,
            "target_status": "confirmed",
            "boundary_precision": "exact",
            "quality_flags": [],
            "annotator_id": "annotator_01",
            "reviewer_ids": ["reviewer_01", "reviewer_02"],
            "review_status": "double_reviewed",
            "note": "Independent reviewers confirmed recovery without a fall.",
            "source_annotation_path": source.as_posix(),
            "source_annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "physical_event_id": "physical_" + "1" * 24,
            "event_subtype": "stumble_recovery",
            "hard_negative_type": None,
            "onset_frame": 15,
            "peak_frame": 24,
            "impact_frame": None,
            "recovery_frame": 35,
            "linked_action_ids": [],
            "contact_evidence": "observed",
        }

    def _negative(
        self,
        fixture: dict[str, Path | dict],
        *,
        decision_id: str,
        hard_negative_type: str,
        start_frame: int,
    ) -> dict:
        source = fixture["source"]
        assert isinstance(source, Path)
        return {
            "schema_version": "near-fall-manual-decision-v1",
            "decision_id": decision_id,
            "source_type": "manual_near_fall_v1",
            "video_id": "video_1",
            "track_id": "person_1",
            "label_role": "negative",
            "start_frame": start_frame,
            "end_frame_exclusive": start_frame + 20,
            "frame_index_base": 0,
            "target_status": "confirmed",
            "boundary_precision": "exact",
            "quality_flags": [],
            "annotator_id": "annotator_01",
            "reviewer_ids": ["reviewer_01"],
            "review_status": "single_reviewed",
            "note": f"Manually confirmed {hard_negative_type} window.",
            "source_annotation_path": source.as_posix(),
            "source_annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "physical_event_id": None,
            "event_type": None,
            "event_subtype": None,
            "event_outcome": None,
            "hard_negative_type": hard_negative_type,
            "onset_frame": None,
            "peak_frame": None,
            "impact_frame": None,
            "recovery_frame": None,
            "linked_action_ids": [],
            "contact_evidence": "not_applicable",
        }

    def _publish(self, fixture: dict[str, Path | dict], rows: list[dict]) -> dict:
        decisions = fixture["decisions"]
        assert isinstance(decisions, Path)
        self._write_jsonl(decisions, rows)
        return publish_near_fall_manual_labels_v3(
            base_event_labels=fixture["base"],
            decisions=decisions,
            manifest=fixture["manifest"],
            output_event_labels=fixture["output"],
            report=fixture["report"],
            repo_root=ROOT,
        )

    def test_publishes_positive_and_explicit_manual_negatives_as_valid_v3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            rows = [
                self._positive(fixture),
                self._negative(
                    fixture,
                    decision_id="near_fall_progressed_001",
                    hard_negative_type="progressed_to_fall",
                    start_frame=50,
                ),
                self._negative(
                    fixture,
                    decision_id="near_fall_background_001",
                    hard_negative_type="background",
                    start_frame=80,
                ),
            ]
            result = self._publish(fixture, rows)
            output = fixture["output"]
            report = fixture["report"]
            actions = fixture["actions"]
            manifest = fixture["manifest"]
            assert all(isinstance(path, Path) for path in (output, report, actions, manifest))
            published = [json.loads(line) for line in output.read_text().splitlines()]
            validation = validate_training_labels_v3(
                manifest_path=manifest,
                action_labels_path=actions,
                event_labels_path=output,
                action_schema_path=ACTION_SCHEMA,
                event_schema_path=EVENT_SCHEMA,
            )

            self.assertEqual(result["added_row_count"], 3)
            self.assertTrue(validation["valid"], validation["issues"])
            self.assertEqual({row["label_role"] for row in published}, {"positive", "negative"})
            self.assertTrue(
                all(row["source_refs"][0]["source_type"] == "manual_v3" for row in published)
            )
            self.assertTrue(
                all(row["sample_group_id"].startswith("samplegrp_") for row in published)
            )
            self.assertEqual(
                {row["hard_negative_type"] for row in published if row["label_role"] == "negative"},
                {"progressed_to_fall", "background"},
            )
            self.assertFalse((output.with_suffix(output.suffix + ".part")).exists())
            self.assertFalse((report.with_suffix(report.suffix + ".part")).exists())

    def test_publication_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = self._fixture(root / "first")
            second = self._fixture(root / "second")
            first_rows = [self._positive(first)]
            second_rows = [self._positive(second)]
            self._publish(first, first_rows)
            self._publish(second, second_rows)
            first_output = first["output"]
            second_output = second["output"]
            assert isinstance(first_output, Path) and isinstance(second_output, Path)
            first_row = json.loads(first_output.read_text())
            second_row = json.loads(second_output.read_text())

            self.assertEqual(first_row["label_id"], second_row["label_id"])
            self.assertEqual(first_row["sample_group_id"], second_row["sample_group_id"])

    def test_rejects_positive_without_recovery_or_two_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            decision = self._positive(fixture)
            decision["recovery_frame"] = None
            decision["reviewer_ids"] = ["reviewer_01"]

            with self.assertRaisesRegex(ValueError, "two reviewers|recovery_frame"):
                self._publish(fixture, [decision])

    def test_rejects_non_manual_negative_uncertain_and_severe_quality(self) -> None:
        mutations = (
            ("source_type", "rule_output", "source_type"),
            ("target_status", "uncertain", "target_status"),
            ("quality_flags", ["heavy_occlusion"], "severe quality"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                decision = self._negative(
                    fixture,
                    decision_id="near_fall_negative_001",
                    hard_negative_type="normal_turn",
                    start_frame=50,
                )
                decision[field] = value

                with self.assertRaisesRegex(ValueError, message):
                    self._publish(fixture, [decision])

    def test_rejects_invalid_or_mismatched_source_hash_without_outputs(self) -> None:
        for source_hash, message in (("bad-hash", "SHA-256"), ("0" * 64, "mismatch")):
            with self.subTest(source_hash=source_hash), tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                decision = self._positive(fixture)
                decision["source_annotation_sha256"] = source_hash
                output = fixture["output"]
                report = fixture["report"]
                base = fixture["base"]
                assert isinstance(output, Path) and isinstance(report, Path) and isinstance(base, Path)
                before = base.read_bytes()

                with self.assertRaisesRegex(ValueError, message):
                    self._publish(fixture, [decision])

                self.assertEqual(base.read_bytes(), before)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())
                self.assertFalse(output.with_suffix(output.suffix + ".part").exists())

    def test_rejects_base_overwrite_and_duplicate_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            decision = self._positive(fixture)
            decisions = fixture["decisions"]
            base = fixture["base"]
            report = fixture["report"]
            assert isinstance(decisions, Path) and isinstance(base, Path) and isinstance(report, Path)
            self._write_jsonl(decisions, [decision])
            before = base.read_bytes()

            with self.assertRaisesRegex(ValueError, "separate candidate"):
                publish_near_fall_manual_labels_v3(
                    base_event_labels=base,
                    decisions=decisions,
                    manifest=fixture["manifest"],
                    output_event_labels=base,
                    report=report,
                    repo_root=decisions.parent,
                    overwrite=True,
                )
            self.assertEqual(base.read_bytes(), before)

            with self.assertRaisesRegex(ValueError, "duplicate"):
                self._publish(fixture, [decision, decision])

    def test_rejects_duplicate_physical_event_before_writing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._positive(fixture)
            second = {**first, "decision_id": "near_fall_positive_002"}
            output = fixture["output"]
            assert isinstance(output, Path)

            with self.assertRaisesRegex(ValueError, "physical_event_id"):
                self._publish(fixture, [first, second])

            self.assertFalse(output.exists())

    def test_rejects_invalid_boundary_contact_and_link_contracts(self) -> None:
        mutations = (
            ("boundary_precision", "frame-ish", "boundary_precision"),
            ("contact_evidence", "guessed", "contact_evidence"),
            ("linked_action_ids", ["actionv3_" + "1" * 24], "linked_action_ids"),
            ("peak_frame", 38, "peak_frame"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                decision = self._positive(fixture)
                decision[field] = value

                with self.assertRaisesRegex(ValueError, message):
                    self._publish(fixture, [decision])


if __name__ == "__main__":
    unittest.main()
