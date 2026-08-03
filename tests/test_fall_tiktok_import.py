from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

from elderly_monitoring.modules.fall_risk.annotations import convert_cvat_xml
from elderly_monitoring.modules.fall_risk.fall_tiktok import (
    load_fall_tiktok_collection_decision,
    prepare_fall_tiktok_cvat_export,
    write_prepared_fall_tiktok_export,
)


PROJECT_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <project>
      <owner><username>private-user</username><email>private@example.test</email></owner>
      <tasks>
        <task>
          <id>10</id><name>001.mp4</name><size>2</size>
          <start_frame>0</start_frame><stop_frame>1</stop_frame>
          <owner><username>task-owner</username></owner><source>001.mp4</source>
        </task>
        <task>
          <id>11</id><name>002.mp4</name><size>2</size>
          <start_frame>0</start_frame><stop_frame>1</stop_frame>
          <source>002.mp4</source>
        </task>
      </tasks>
    </project>
  </meta>
  <track id="1" label="A01_normal_walk" task_id="10">
    <box frame="0" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/>
    <box frame="1" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/>
  </track>
  <track id="2" label="D01_forward_fall" task_id="11">
    <box frame="2" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/>
    <box frame="3" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/>
  </track>
</annotations>
"""


class FallTiktokImportTest(unittest.TestCase):
    def test_loads_project_collection_training_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "decision.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-tiktok-collection-decision-v1",
                        "decision_id": "fixture-decision",
                        "decided_at": "2026-07-28",
                        "decided_by": "project_owner",
                        "dataset": "fall_tiktok",
                        "collection_status": "project_collected",
                        "training_use": "authorized",
                        "redistribution_use": "not_authorized_by_this_decision",
                        "consent_status": "not_recorded",
                        "subject_grouping_status": "unknown",
                        "source_group_id": "fall_tiktok_project_collection_pool",
                        "source_uri": "internal://collection/fall_tiktok_fixture",
                        "provenance_status": "project_collected_training_authorized",
                    }
                ),
                encoding="utf-8",
            )

            decision = load_fall_tiktok_collection_decision(path)

        self.assertEqual(decision.training_use, "authorized")
        self.assertEqual(
            decision.provenance_status,
            "project_collected_training_authorized",
        )

    def test_prepare_redacts_normalizes_and_converts_global_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "fall_tiktok.zip"
            source_map = root / "source_map.json"
            prepared = root / "fall_tiktok_redacted.zip"
            manifest = root / "manifest.jsonl"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("annotations.xml", PROJECT_XML)
            source_map.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-tiktok-source-map-v1",
                        "dataset": "fall_tiktok",
                        "entries": [
                            {
                                "sequence": 1,
                                "original_filename": "1.mp4",
                                "filename": "1.mp4",
                            },
                            {
                                "sequence": 2,
                                "original_filename": "4.mp4",
                                "filename": "2.mp4",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = prepare_fall_tiktok_cvat_export(source, source_map)
            repeat = prepare_fall_tiktok_cvat_export(source, source_map)
            self.assertEqual(result.payload, repeat.payload)
            self.assertEqual(result.task_count, 2)
            self.assertEqual(result.track_count, 2)
            self.assertEqual(result.removed_identity_elements, 2)
            write_prepared_fall_tiktok_export(result, prepared)

            with zipfile.ZipFile(BytesIO(result.payload)) as archive:
                prepared_root = ET.fromstring(archive.read("annotations.xml"))
            tasks = prepared_root.findall("./meta/project/tasks/task")
            self.assertEqual(
                [task.findtext("name") for task in tasks],
                [
                    "fall_risk__fall_tiktok__annotated_clips__fall_tiktok_clip_001",
                    "fall_risk__fall_tiktok__annotated_clips__fall_tiktok_clip_002",
                ],
            )
            self.assertEqual(
                [task.findtext("source") for task in tasks], ["1.mp4", "2.mp4"]
            )
            self.assertIsNone(prepared_root.find(".//owner"))
            self.assertIsNone(prepared_root.find(".//email"))

            rows = [
                self._manifest_row(root, 1, 2),
                self._manifest_row(root, 2, 2),
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            converted = convert_cvat_xml(
                prepared,
                manifest_path=manifest,
                fps=None,
                labeler="fall_tiktok_labeler",
            )

        self.assertEqual(len(converted.action_labels), 2)
        by_video = {row["video_id"]: row for row in converted.action_labels}
        self.assertEqual(by_video["fall_tiktok_clip_001"]["start_frame"], 0)
        self.assertEqual(by_video["fall_tiktok_clip_001"]["end_frame"], 1)
        self.assertEqual(by_video["fall_tiktok_clip_002"]["start_frame"], 0)
        self.assertEqual(by_video["fall_tiktok_clip_002"]["end_frame"], 1)
        self.assertEqual(by_video["fall_tiktok_clip_002"]["action_id"], "D01")
        self.assertFalse(converted.identity_metadata_present)

    @staticmethod
    def _manifest_row(root: Path, sequence: int, frame_count: int) -> dict:
        filename = f"{sequence}.mp4"
        return {
            "asset_id": f"asset_{sequence}",
            "dataset": "fall_tiktok",
            "subset": "annotated_clips",
            "path": (root / filename).as_posix(),
            "sha256": str(sequence) * 64,
            "media_type": "video",
            "modality": "rgb_video",
            "video_id": f"fall_tiktok_clip_{sequence:03d}",
            "fps_num": 30,
            "fps_den": 1,
            "fps": 30.0,
            "frame_count": frame_count,
            "duration_sec": frame_count / 30,
            "width": 10,
            "height": 10,
            "subject_id": "unknown",
            "source_group_id": "fall_tiktok_project_collection_pool",
            "original_event_id": f"fall_tiktok_sample_{sequence:03d}",
            "scene_region": "unknown",
            "view": "unknown",
            "label_source": "cvat_manual",
            "annotation_path": None,
            "source_uri": "internal://collection/fall_tiktok_fixture",
            "consent_id": None,
            "eligibility": True,
            "exclusion_reasons": [],
            "duplicate_group_id": None,
            "duplicate_of_asset_id": None,
        }


if __name__ == "__main__":
    unittest.main()
