from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.label_publish import publish_v2_labels


class FallRiskLabelPublishTest(unittest.TestCase):
    def test_official_fall_window_replaces_overlapping_cvat_fall(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "generated" / "v2"
            cvat = source / "cvat_batch"
            official = source / "le2i_official"
            cvat.mkdir(parents=True)
            official.mkdir(parents=True)
            action = {
                "label_id": "action_1", "video_id": "video_1", "start_time": 0.0,
                "end_time": 3.0, "event_type": "fall", "action_id": "D01",
            }
            cvat_event = {
                "label_id": "event_cvat", "video_id": "video_1", "start_time": 0.5,
                "end_time": 2.5, "event_type": "fall", "label_source": "cvat_action_mapping",
            }
            official_event = {
                "label_id": "event_official", "video_id": "video_1", "start_time": 1.0,
                "end_time": 2.0, "event_type": "fall", "label_source": "le2i_txt",
            }
            (cvat / "action_labels.jsonl").write_text(json.dumps(action) + "\n", encoding="utf-8")
            (cvat / "event_labels.jsonl").write_text(json.dumps(cvat_event) + "\n", encoding="utf-8")
            (official / "event_labels.jsonl").write_text(json.dumps(official_event) + "\n", encoding="utf-8")
            action_output = root / "action_labels.jsonl"
            event_output = root / "event_labels.jsonl"
            report_output = root / "report.json"

            report = publish_v2_labels(
                source,
                action_output=action_output,
                event_output=event_output,
                report_output=report_output,
            )

            events = [json.loads(line) for line in event_output.read_text().splitlines()]
            self.assertEqual([row["label_id"] for row in events], ["event_official"])
            self.assertEqual(report["output_counts"]["excluded_overlapping_cvat_fall_events"], 1)

    def test_non_overlapping_and_non_fall_cvat_events_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "v2"
            batch = source / "cvat"
            batch.mkdir(parents=True)
            action = {"label_id": "action_1", "video_id": "video_1", "start_time": 0, "end_time": 1}
            events = [
                {"label_id": "event_1", "video_id": "video_1", "start_time": 0, "end_time": 1, "event_type": "near_fall", "label_source": "cvat_action_mapping"},
                {"label_id": "event_2", "video_id": "video_2", "start_time": 0, "end_time": 1, "event_type": "fall", "label_source": "cvat_action_mapping"},
            ]
            (batch / "action_labels.jsonl").write_text(json.dumps(action) + "\n", encoding="utf-8")
            (batch / "event_labels.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
            report = publish_v2_labels(
                source,
                action_output=root / "action.jsonl",
                event_output=root / "event.jsonl",
                report_output=root / "report.json",
            )
            self.assertEqual(report["output_counts"]["event_labels"], 2)

    def test_manually_reviewed_ntu_rgbd_batch_enters_root_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "v2"
            cvat = source / "cvat"
            ntu = source / "ntu_rgbd_clip_labels"
            cvat.mkdir(parents=True)
            ntu.mkdir(parents=True)
            cvat_action = {
                "label_id": "action_cvat",
                "video_id": "video_cvat",
                "start_time": 0,
                "end_time": 1,
            }
            ntu_action = {
                "label_id": "action_ntu",
                "video_id": "video_ntu",
                "start_time": 0,
                "end_time": 1,
            }
            cvat_event = {
                "label_id": "event_cvat",
                "video_id": "video_cvat",
                "start_time": 0,
                "end_time": 1,
                "event_type": "near_fall",
                "label_source": "cvat_action_mapping",
            }
            (cvat / "action_labels.jsonl").write_text(
                json.dumps(cvat_action) + "\n", encoding="utf-8"
            )
            (cvat / "event_labels.jsonl").write_text(
                json.dumps(cvat_event) + "\n", encoding="utf-8"
            )
            (ntu / "action_labels.jsonl").write_text(
                json.dumps(ntu_action) + "\n", encoding="utf-8"
            )

            report = publish_v2_labels(
                source,
                action_output=root / "action.jsonl",
                event_output=root / "event.jsonl",
                report_output=root / "report.json",
            )

            actions = [
                json.loads(line)
                for line in (root / "action.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["label_id"] for row in actions],
                ["action_cvat", "action_ntu"],
            )
            self.assertEqual(
                report["input_counts"]["excluded_source_isolated_action_labels"],
                0,
            )
            self.assertEqual(report["excluded_source_batches"], [])

    def test_reviewed_ntu_a043_cvat_batch_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "v2"
            accepted = source / "accepted"
            manual = source / "ntu_rgbd_a043_cvat_review"
            accepted.mkdir(parents=True)
            manual.mkdir(parents=True)
            (accepted / "action_labels.jsonl").write_text(
                json.dumps({"label_id": "accepted", "video_id": "video_1"}) + "\n",
                encoding="utf-8",
            )
            (accepted / "event_labels.jsonl").write_text(
                json.dumps(
                    {
                        "label_id": "event_accepted",
                        "video_id": "video_1",
                        "event_type": "fall",
                        "label_source": "cvat_action_mapping",
                        "start_time": 0,
                        "end_time": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (manual / "action_labels.jsonl").write_text(
                json.dumps({"label_id": "manual", "video_id": "video_2"}) + "\n",
                encoding="utf-8",
            )
            (manual / "event_labels.jsonl").write_text(
                json.dumps({"label_id": "event_manual", "video_id": "video_2"}) + "\n",
                encoding="utf-8",
            )

            report = publish_v2_labels(
                source,
                action_output=root / "action.jsonl",
                event_output=root / "event.jsonl",
                report_output=root / "report.json",
            )

            rows = [
                json.loads(line)
                for line in (root / "action.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["label_id"] for row in rows], ["accepted", "manual"]
            )
            self.assertEqual(report["excluded_source_batches"], [])
            self.assertEqual(
                report["input_counts"]["excluded_source_isolated_action_labels"],
                0,
            )
            self.assertEqual(
                report["input_counts"]["excluded_source_isolated_event_labels"],
                0,
            )

    def test_import_report_batch_id_mismatch_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "v2"
            accepted = source / "ntu_rgbd_a043_cvat_review"
            candidate = source / "ntu_rgbd_a043_cvat_review_candidate"
            accepted.mkdir(parents=True)
            candidate.mkdir(parents=True)
            for batch, suffix in ((accepted, "accepted"), (candidate, "candidate")):
                (batch / "action_labels.jsonl").write_text(
                    json.dumps(
                        {"label_id": f"action_{suffix}", "video_id": suffix}
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (batch / "event_labels.jsonl").write_text(
                    json.dumps(
                        {
                            "label_id": f"event_{suffix}",
                            "video_id": suffix,
                            "event_type": "fall",
                            "label_source": "cvat_action_mapping",
                            "start_time": 0,
                            "end_time": 1,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (batch / "import_report.json").write_text(
                    json.dumps(
                        {
                            "batch_id": "ntu_rgbd_a043_cvat_review",
                            "publication_status": "accepted_for_v2_publication",
                        }
                    ),
                    encoding="utf-8",
                )

            report = publish_v2_labels(
                source,
                action_output=root / "action.jsonl",
                event_output=root / "event.jsonl",
                report_output=root / "report.json",
            )

            actions = [
                json.loads(line)
                for line in (root / "action.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["label_id"] for row in actions], ["action_accepted"])
            self.assertEqual(
                report["excluded_source_batches"],
                [
                    {
                        "batch_id": "ntu_rgbd_a043_cvat_review_candidate",
                        "reason": "import_report_batch_id_mismatch",
                    }
                ],
            )
            self.assertEqual(
                report["input_counts"]["excluded_source_isolated_action_labels"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
