import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from elderly_monitoring.modules.fall_risk.annotations import (
    ACTION_NAMES,
    convert_cvat_xml,
    import_pre_vfallp_cvat_labels,
    import_le2i_fall_labels,
    import_ntu_rgbd_clip_labels,
    import_toaga_normal_walk_labels,
    write_pre_vfallp_cvat_labels,
    write_fall_label_jsonl,
)


CVAT_PROJECT_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <project>
      <tasks>
        <task>
          <id>1</id>
          <name>fall_risk__le2i_imvia__home_01__le2i_home_01_video_1</name>
          <size>5</size>
          <start_frame>0</start_frame>
          <stop_frame>4</stop_frame>
          <source>video (1).avi</source>
        </task>
        <task>
          <id>2</id>
          <name>fall_risk__le2i_imvia__home_01__le2i_home_01_video_2</name>
          <size>4</size>
          <start_frame>0</start_frame>
          <stop_frame>3</stop_frame>
          <source>video (2).avi</source>
        </task>
      </tasks>
    </project>
  </meta>
  <track id="10" label="A04_normal_sit_to_stand" source="manual" task_id="1">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="1" ytl="2" xbr="3" ybr="4"/>
    <box frame="2" keyframe="1" outside="0" occluded="0" xtl="1" ytl="2" xbr="3" ybr="4"/>
    <box frame="3" keyframe="1" outside="1" occluded="0" xtl="1" ytl="2" xbr="3" ybr="4"/>
  </track>
  <track id="11" label="D02_lateral_fall" source="manual" task_id="1">
    <box frame="3" keyframe="1" outside="0" occluded="0" xtl="5" ytl="6" xbr="7" ybr="8"/>
    <box frame="4" keyframe="1" outside="0" occluded="0" xtl="5" ytl="6" xbr="7" ybr="8"/>
  </track>
  <track id="20" label="B05_unstable_turn" source="manual" task_id="2">
    <box frame="5" keyframe="1" outside="0" occluded="0" xtl="9" ytl="10" xbr="11" ybr="12"/>
    <box frame="6" keyframe="1" outside="1" occluded="0" xtl="9" ytl="10" xbr="11" ybr="12"/>
  </track>
  <track id="21" label="D04_long_static_after_fall" source="manual" task_id="2">
    <box frame="6" keyframe="1" outside="0" occluded="0" xtl="13" ytl="14" xbr="15" ybr="16">
      <attribute name="quality">partial_occlusion</attribute>
      <attribute name="note">躺倒后静止</attribute>
    </box>
    <box frame="8" keyframe="1" outside="0" occluded="0" xtl="13" ytl="14" xbr="15" ybr="16"/>
  </track>
</annotations>
"""


class FallRiskAnnotationConversionTest(unittest.TestCase):
    def _write_manifest(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _manifest_row(
        self,
        *,
        video_id: str,
        path: Path,
        annotation_path: Path | None = None,
        fps_num: int = 2,
        fps_den: int = 1,
        frame_count: int = 10,
        subset: str = "home_01",
        scene_region: str = "home",
    ) -> dict:
        return {
            "asset_id": f"asset_{video_id}",
            "dataset": "le2i_imvia",
            "subset": subset,
            "path": path.as_posix(),
            "sha256": "a" * 64,
            "media_type": "video",
            "modality": "rgb_video",
            "video_id": video_id,
            "fps_num": fps_num,
            "fps_den": fps_den,
            "fps": fps_num / fps_den,
            "frame_count": frame_count,
            "duration_sec": frame_count * fps_den / fps_num,
            "width": 320,
            "height": 240,
            "subject_id": "unknown",
            "source_group_id": f"group_{video_id}",
            "original_event_id": f"event_{video_id}",
            "scene_region": scene_region,
            "view": "fixed_camera",
            "label_source": "official_annotation",
            "annotation_path": annotation_path.as_posix() if annotation_path else None,
            "consent_id": None,
            "eligibility": True,
            "exclusion_reasons": [],
        }

    def test_convert_cvat_xml_outputs_action_and_event_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "annotations.xml"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")

            converted = convert_cvat_xml(
                xml_path,
                fps=2.0,
                file_root=Path("/dataset/FallDataset/Home_01/Videos"),
                labeler="labeler_01",
                default_scene="home",
            )

        self.assertEqual(len(converted.action_labels), 4)
        self.assertEqual(len(converted.event_labels), 4)

        first = converted.action_labels[0]
        self.assertEqual(first["video_id"], "le2i_home_01_video_1")
        self.assertEqual(first["file_path"], "/dataset/FallDataset/Home_01/Videos/video (1).avi")
        self.assertEqual(first["action_id"], "A04")
        self.assertEqual(first["action_name"], "normal_sit_to_stand")
        self.assertEqual(first["start_frame"], 0)
        self.assertEqual(first["end_frame"], 2)
        self.assertEqual(first["start_time"], 0.0)
        self.assertEqual(first["end_time"], 1.0)
        self.assertEqual(first["bbox_start"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(first["labeler"], "labeler_01")
        self.assertNotIn("eligibility", first)

        second_task_action = converted.action_labels[2]
        self.assertEqual(second_task_action["video_id"], "le2i_home_01_video_2")
        self.assertEqual(second_task_action["start_frame"], 0)
        self.assertEqual(second_task_action["end_frame"], 0)
        self.assertEqual(second_task_action["event_type"], "unstable_turning")

        long_static = converted.action_labels[3]
        self.assertEqual(long_static["quality"], "partial_occlusion")
        self.assertEqual(long_static["note"], "躺倒后静止")
        self.assertEqual(long_static["start_frame"], 1)
        self.assertEqual(long_static["end_frame"], 3)

        fall_event = converted.event_labels[1]
        self.assertEqual(fall_event["event_type"], "fall")
        self.assertEqual(fall_event["severity"], 4)
        self.assertEqual(fall_event["label_source"], "cvat_action_mapping")

        static_event = converted.event_labels[3]
        self.assertEqual(static_event["event_type"], "long_static")
        self.assertEqual(static_event["start_frame"], 1)
        self.assertEqual(static_event["end_frame"], 3)

    def test_write_fall_label_jsonl_persists_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "annotations.xml"
            action_path = Path(tmpdir) / "action_labels.jsonl"
            event_path = Path(tmpdir) / "event_labels.jsonl"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")

            counts = write_fall_label_jsonl(
                xml_path,
                action_output_path=action_path,
                event_output_path=event_path,
                fps=2.0,
            )

            action_lines = action_path.read_text(encoding="utf-8").splitlines()
            event_lines = event_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(counts, {"action_labels": 4, "event_labels": 4})
        self.assertEqual(len(action_lines), 4)
        self.assertEqual(len(event_lines), 4)
        self.assertEqual(json.loads(action_lines[0])["action_id"], "A04")
        self.assertEqual(json.loads(event_lines[-1])["event_type"], "long_static")

    def test_pre_vfallp_nested_cvat_import_redacts_and_maps_video_ids(self) -> None:
        source_xml = textwrap.dedent(
            """\
            <annotations>
              <meta>
                <task>
                  <id>19</id>
                  <name>1</name>
                  <size>3</size>
                  <start_frame>0</start_frame>
                  <stop_frame>2</stop_frame>
                  <source>FDFFM1.mp4</source>
                  <owner><username>annotator@example.test</username><email>annotator@example.test</email></owner>
                  <assignee><username>reviewer@example.test</username></assignee>
                </task>
              </meta>
              <track id="0" label="B01_slow_walk" source="manual">
                <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
                <box frame="1" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
              <track id="1" label="D01_forward_fall" source="manual">
                <box frame="2" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
            </annotations>
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "FDFFM1.mp4"
            video.write_bytes(b"video")
            outer_export = root / "pre_vfallp.zip"
            with tempfile.TemporaryDirectory() as inner_tmpdir:
                inner_export = Path(inner_tmpdir) / "task.zip"
                with zipfile.ZipFile(inner_export, "w") as archive:
                    archive.writestr("annotations.xml", source_xml)
                with zipfile.ZipFile(outer_export, "w") as archive:
                    archive.write(
                        inner_export,
                        "001__task_19__1__CVAT_for_video_1_1.zip",
                    )
                    archive.writestr(
                        "manifest.tsv",
                        "order\ttask_id\ttask_name\tproject_id\texport_file\n"
                        "1\t19\t1\t\t001__task_19__1__CVAT_for_video_1_1.zip\n",
                    )
            archive_sha256 = hashlib.sha256(outer_export.read_bytes()).hexdigest()
            video_id = "pre_vfallp_video_dizziness_fall_forward_fdffm1"
            manifest = root / "manifest.jsonl"
            self._write_manifest(
                manifest,
                [
                    {
                        **self._manifest_row(
                            video_id=video_id,
                            path=video,
                            fps_num=1,
                            fps_den=1,
                            frame_count=3,
                            subset="dizziness_fall_forward",
                            scene_region="unknown",
                        ),
                        "dataset": "pre_vfallp",
                        "source_group_id": "pre_vfallp_unresolved",
                        "source_uri": "internal://authorization/internal_pre_vfallp_fixture",
                        "eligibility": True,
                        "exclusion_reasons": [],
                        "provenance_status": "internal_authorized_source_unverified",
                        "internal_authorization": {
                            "authorization_id": "internal_pre_vfallp_fixture",
                            "approval_reference": "test_internal_authorization",
                            "approved_at": "2026-07-22",
                            "evidence": {
                                "kind": "cvat_export_archive",
                                "name": outer_export.name,
                                "sha256": archive_sha256,
                            },
                            "authorized_video_count": 1,
                        },
                    }
                ],
            )
            redacted_dir = root / "redacted"
            result = import_pre_vfallp_cvat_labels(
                outer_export,
                manifest_path=manifest,
                redacted_export_dir=redacted_dir,
                labeler="importer_01",
            )
            output_dir = root / "generated"
            report = write_pre_vfallp_cvat_labels(
                result,
                redacted_export_dir=redacted_dir,
                action_output_path=output_dir / "action_labels.jsonl",
                event_output_path=output_dir / "event_labels.jsonl",
                report_output_path=output_dir / "import_report.json",
            )

            redacted_export = redacted_dir / "001__task_19__1__CVAT_for_video_1_1.zip"
            with zipfile.ZipFile(redacted_export) as archive:
                redacted_xml = ET.fromstring(archive.read("annotations.xml"))

        self.assertEqual(len(result.action_labels), 2)
        self.assertEqual(len(result.event_labels), 2)
        self.assertEqual(report["imported_action_labels"], 2)
        self.assertEqual(report["imported_event_labels"], 2)
        self.assertEqual(
            result.action_labels[0]["source_annotation_path"],
            redacted_export.resolve().as_posix(),
        )
        self.assertEqual(
            result.action_labels[0]["video_id"], video_id
        )
        self.assertEqual(
            redacted_xml.findtext("./meta/task/name"),
            f"fall_risk__pre_vfallp__dizziness_fall_forward__{video_id}",
        )
        self.assertIsNone(redacted_xml.find(".//owner"))
        self.assertIsNone(redacted_xml.find(".//assignee"))
        self.assertNotIn("annotator@example.test", ET.tostring(redacted_xml, encoding="unicode"))

    def test_pre_vfallp_nested_import_accepts_directory_and_resized_source(self) -> None:
        source_xml = textwrap.dedent(
            """\
            <annotations>
              <meta>
                <task>
                  <id>21</id>
                  <name>fall_risk__legacy__{{CGFM1.mp4}}</name>
                  <size>2</size>
                  <start_frame>0</start_frame>
                  <stop_frame>1</stop_frame>
                  <source>CGFM1_resized.mp4</source>
                  <owner><username>annotator@example.test</username></owner>
                </task>
              </meta>
              <track id="0" label="B02_dragging_walk" source="manual">
                <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
                <box frame="1" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
              <track id="1" label="B03_shuffling_walk" source="manual">
                <box frame="1" keyframe="1" outside="1" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
            </annotations>
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "CGFM1.mp4"
            video.write_bytes(b"video")
            outer_export = root / "confusion_nph.zip"
            with tempfile.TemporaryDirectory() as inner_tmpdir:
                inner_export = Path(inner_tmpdir) / "task.zip"
                with zipfile.ZipFile(inner_export, "w") as archive:
                    archive.writestr("annotations.xml", source_xml)
                with zipfile.ZipFile(outer_export, "w") as archive:
                    archive.write(inner_export, "annotated/task.zip")
            archive_sha256 = hashlib.sha256(outer_export.read_bytes()).hexdigest()
            video_id = "pre_vfallp_video_confusion_nph_cgfm1"
            manifest = root / "manifest.jsonl"
            self._write_manifest(
                manifest,
                [
                    {
                        **self._manifest_row(
                            video_id=video_id,
                            path=video,
                            fps_num=1,
                            fps_den=1,
                            frame_count=2,
                            subset="confusion_nph",
                            scene_region="unknown",
                        ),
                        "dataset": "pre_vfallp",
                        "source_group_id": "pre_vfallp_unresolved",
                        "source_uri": "internal://authorization/internal_pre_vfallp_fixture",
                        "eligibility": True,
                        "exclusion_reasons": [],
                        "provenance_status": "internal_authorized_source_unverified",
                        "internal_authorization": {
                            "authorization_id": "internal_pre_vfallp_fixture",
                            "approval_reference": "test_internal_authorization",
                            "approved_at": "2026-07-22",
                            "evidence": {
                                "kind": "cvat_export_archive",
                                "name": outer_export.name,
                                "sha256": archive_sha256,
                            },
                            "authorized_video_count": 1,
                        },
                    }
                ],
            )

            result = import_pre_vfallp_cvat_labels(
                outer_export,
                manifest_path=manifest,
                redacted_export_dir=root / "redacted",
                labeler="importer_01",
            )

        self.assertEqual(len(result.redacted_exports), 1)
        self.assertEqual(result.redacted_exports[0].filename, "task.zip")
        self.assertEqual(result.action_labels[0]["video_id"], video_id)
        self.assertEqual(result.action_labels[0]["action_id"], "B02")
        self.assertEqual(
            result.report["ignored_tracks"],
            [
                {
                    "filename": "task.zip",
                    "label": "B03_shuffling_walk",
                    "reason": "outside_only",
                    "source_name": "CGFM1_resized.mp4",
                    "task_id": "21",
                    "track_id": "1",
                }
            ],
        )

    def test_pre_vfallp_project_export_is_split_into_redacted_task_exports(self) -> None:
        project_xml = textwrap.dedent(
            """\
            <annotations>
              <version>1.1</version>
              <meta>
                <project>
                  <id>7</id>
                  <name>weakness_fall_forward</name>
                  <owner><username>project-owner@example.test</username></owner>
                  <tasks>
                    <task>
                      <id>163</id>
                      <name>FWFFM1.mp4</name>
                      <size>2</size>
                      <start_frame>0</start_frame>
                      <stop_frame>1</stop_frame>
                      <source>FWFFM1.mp4</source>
                      <assignee><username>reviewer@example.test</username></assignee>
                    </task>
                    <task>
                      <id>164</id>
                      <name>FWFFM2.mp4</name>
                      <size>2</size>
                      <start_frame>0</start_frame>
                      <stop_frame>1</stop_frame>
                      <source>FWFFM2.mp4</source>
                    </task>
                  </tasks>
                </project>
              </meta>
              <track id="0" label="B01_slow_walk" source="manual" task_id="163">
                <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
              <track id="1" label="D01_forward_fall" source="manual" task_id="164">
                <box frame="3" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
            </annotations>
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            videos = [root / "FWFFM1.mp4", root / "FWFFM2.mp4"]
            for video in videos:
                video.write_bytes(video.name.encode("ascii"))
            outer_export = root / "weakness_fall_forward.zip"
            with zipfile.ZipFile(outer_export, "w") as archive:
                archive.writestr("annotations.xml", project_xml)
            archive_sha256 = hashlib.sha256(outer_export.read_bytes()).hexdigest()
            video_ids = [
                "pre_vfallp_video_weakness_fall_forward_fwffm1",
                "pre_vfallp_video_weakness_fall_forward_fwffm2",
            ]
            manifest = root / "manifest.jsonl"
            rows = []
            for video_id, video in zip(video_ids, videos, strict=True):
                rows.append(
                    {
                        **self._manifest_row(
                            video_id=video_id,
                            path=video,
                            fps_num=1,
                            fps_den=1,
                            frame_count=2,
                            subset="weakness_fall_forward",
                            scene_region="unknown",
                        ),
                        "dataset": "pre_vfallp",
                        "source_group_id": "pre_vfallp_unresolved",
                        "source_uri": "internal://authorization/internal_pre_vfallp_fixture",
                        "eligibility": True,
                        "exclusion_reasons": [],
                        "provenance_status": "internal_authorized_source_unverified",
                        "internal_authorization": {
                            "authorization_id": "internal_pre_vfallp_fixture",
                            "approval_reference": "test_internal_authorization",
                            "approved_at": "2026-07-22",
                            "evidence": {
                                "kind": "cvat_export_archive",
                                "name": outer_export.name,
                                "sha256": archive_sha256,
                            },
                            "authorized_video_count": 2,
                        },
                    }
                )
            self._write_manifest(manifest, rows)
            redacted_dir = root / "redacted"

            result = import_pre_vfallp_cvat_labels(
                outer_export,
                manifest_path=manifest,
                redacted_export_dir=redacted_dir,
                labeler="importer_01",
            )
            report = write_pre_vfallp_cvat_labels(
                result,
                redacted_export_dir=redacted_dir,
                action_output_path=root / "generated/action_labels.jsonl",
                event_output_path=root / "generated/event_labels.jsonl",
                report_output_path=root / "generated/import_report.json",
            )
            redacted_roots = []
            for export in result.redacted_exports:
                with zipfile.ZipFile(redacted_dir / export.filename) as archive:
                    redacted_roots.append(ET.fromstring(archive.read("annotations.xml")))

        self.assertEqual(len(result.redacted_exports), 2)
        self.assertEqual(len(result.action_labels), 2)
        self.assertEqual(len(result.event_labels), 2)
        self.assertEqual(report["imported_action_labels"], 2)
        self.assertEqual(
            {row["video_id"] for row in result.action_labels}, set(video_ids)
        )
        second_action = next(
            row for row in result.action_labels if row["video_id"] == video_ids[1]
        )
        self.assertEqual(second_action["start_frame"], 1)
        self.assertEqual(second_action["end_frame"], 1)
        self.assertTrue(
            all(len(root.findall("./meta/project/tasks/task")) == 1 for root in redacted_roots)
        )
        self.assertTrue(all(len(root.findall("./track")) == 1 for root in redacted_roots))
        self.assertTrue(all(root.find(".//owner") is None for root in redacted_roots))
        self.assertTrue(all(root.find(".//assignee") is None for root in redacted_roots))

    def test_single_task_export_without_task_id_uses_task_metadata(self) -> None:
        xml = textwrap.dedent(
            """\
            <annotations>
              <meta>
                <task>
                  <id>9</id>
                  <name>fall_risk__le2i_imvia__home_01__le2i_home_01_video_9</name>
                  <size>3</size>
                  <start_frame>0</start_frame>
                  <stop_frame>2</stop_frame>
                  <source>video (9).avi</source>
                </task>
              </meta>
              <track id="1" label="D01_forward_fall" source="manual">
                <box frame="1" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
                <box frame="2" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
            </annotations>
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "annotations.xml"
            xml_path.write_text(xml, encoding="utf-8")

            converted = convert_cvat_xml(xml_path, fps=1.0)

        self.assertEqual(converted.action_labels[0]["video_id"], "le2i_home_01_video_9")
        self.assertEqual(converted.action_labels[0]["start_frame"], 1)
        self.assertEqual(converted.action_labels[0]["end_frame"], 2)

    def test_le2i_file_name_task_maps_to_canonical_video_id(self) -> None:
        xml = textwrap.dedent(
            """\
            <annotations>
              <meta>
                <project>
                  <tasks>
                    <task>
                      <id>31</id>
                      <name>fall_risk__le2i_imvia__home_02__video (31).avi</name>
                      <size>2</size>
                      <start_frame>0</start_frame>
                      <stop_frame>1</stop_frame>
                      <source>video (31).avi</source>
                    </task>
                  </tasks>
                </project>
              </meta>
              <track id="1" label="B02_dragging_walk" source="manual" task_id="31">
                <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
                <box frame="1" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>
              </track>
            </annotations>
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "annotations.xml"
            xml_path.write_text(xml, encoding="utf-8")

            converted = convert_cvat_xml(xml_path, fps=1.0)

        action = converted.action_labels[0]
        event = converted.event_labels[0]
        self.assertEqual(action["video_id"], "le2i_home_02_video_31")
        self.assertEqual(action["event_type"], "gait_instability")
        self.assertEqual(event["event_type"], "gait_instability")
        self.assertEqual(event["severity"], 2)

    def test_manifest_supplies_per_video_timeline_path_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = root / "annotations.xml"
            manifest_path = root / "manifest.jsonl"
            video_1 = root / "video (1).avi"
            video_2 = root / "video (2).avi"
            xml_path.write_text(
                CVAT_PROJECT_XML.replace(
                    "<tasks>",
                    "<owner><username>private</username><email>private@example.invalid</email></owner><tasks>",
                    1,
                ),
                encoding="utf-8",
            )
            video_1.touch()
            video_2.touch()
            self._write_manifest(
                manifest_path,
                [
                    self._manifest_row(
                        video_id="le2i_home_01_video_1",
                        path=video_1,
                        fps_num=2,
                        frame_count=5,
                    ),
                    self._manifest_row(
                        video_id="le2i_home_01_video_2",
                        path=video_2,
                        fps_num=4,
                        frame_count=4,
                    ),
                ],
            )

            first = convert_cvat_xml(xml_path, manifest_path=manifest_path, fps=None)
            second = convert_cvat_xml(xml_path, manifest_path=manifest_path, fps=None)

        self.assertEqual(first.action_labels[0]["file_path"], video_1.as_posix())
        self.assertEqual(first.action_labels[0]["end_time"], 1.0)
        self.assertEqual(first.action_labels[3]["end_time"], 0.75)
        self.assertEqual(first.action_labels[0]["asset_id"], "asset_le2i_home_01_video_1")
        self.assertEqual(first.action_labels[0]["label_id"], second.action_labels[0]["label_id"])
        self.assertEqual(
            first.event_labels[0]["source_action_label_id"],
            first.action_labels[0]["label_id"],
        )
        self.assertEqual(first.event_labels[0]["mapping_version"], "fall-action-event-v2")
        self.assertTrue(first.identity_metadata_present)
        self.assertNotIn("owner", json.dumps(first.action_labels))
        self.assertNotIn("email", json.dumps(first.event_labels))

    def test_manifest_fps_conflict_and_frame_overflow_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = root / "annotations.xml"
            manifest_path = root / "manifest.jsonl"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")
            rows = [
                self._manifest_row(
                    video_id="le2i_home_01_video_1",
                    path=root / "video (1).avi",
                    fps_num=2,
                    frame_count=5,
                ),
                self._manifest_row(
                    video_id="le2i_home_01_video_2",
                    path=root / "video (2).avi",
                    fps_num=4,
                    frame_count=4,
                ),
            ]
            self._write_manifest(manifest_path, rows)

            with self.assertRaisesRegex(ValueError, "fps override conflicts"):
                convert_cvat_xml(xml_path, manifest_path=manifest_path, fps=3.0)

            rows[0]["frame_count"] = 4
            self._write_manifest(manifest_path, rows)
            with self.assertRaisesRegex(ValueError, "outside video bounds"):
                convert_cvat_xml(xml_path, manifest_path=manifest_path, fps=None)

    def test_unknown_action_code_and_name_are_rejected(self) -> None:
        bad_xml = CVAT_PROJECT_XML.replace("A04_normal_sit_to_stand", "A99_made_up", 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(bad_xml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown action label"):
                convert_cvat_xml(path, fps=2.0)

    def test_legacy_action_name_is_normalized_to_current_standard(self) -> None:
        legacy_xml = CVAT_PROJECT_XML.replace(
            "A04_normal_sit_to_stand", "A04_normal_stand", 1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(legacy_xml, encoding="utf-8")
            converted = convert_cvat_xml(path, fps=2.0)

        self.assertEqual(converted.action_labels[0]["action_id"], "A04")
        self.assertEqual(
            converted.action_labels[0]["action_name"], "normal_sit_to_stand"
        )

    def test_normal_squat_maps_to_zero_severity_normal_activity(self) -> None:
        squat_xml = CVAT_PROJECT_XML.replace(
            'label="A04_normal_sit_to_stand"', 'label="A05_controlled_squat"', 1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(squat_xml, encoding="utf-8")
            converted = convert_cvat_xml(path, fps=2.0)

        squat_action = converted.action_labels[0]
        squat_event = converted.event_labels[0]
        self.assertEqual(squat_action["action_id"], "A05")
        self.assertEqual(squat_action["action_name"], "controlled_squat")
        self.assertEqual(squat_action["event_type"], "normal_activity")
        self.assertEqual(squat_event["event_type"], "normal_activity")
        self.assertEqual(squat_event["severity"], 0)

    def test_normal_bend_maps_to_zero_severity_normal_activity(self) -> None:
        bend_xml = CVAT_PROJECT_XML.replace(
            'label="A04_normal_sit_to_stand"', 'label="A06_controlled_bend"', 1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(bend_xml, encoding="utf-8")
            converted = convert_cvat_xml(path, fps=2.0)

        bend_action = converted.action_labels[0]
        bend_event = converted.event_labels[0]
        self.assertEqual(bend_action["action_id"], "A06")
        self.assertEqual(bend_action["action_name"], "controlled_bend")
        self.assertEqual(bend_action["event_type"], "normal_activity")
        self.assertEqual(bend_event["event_type"], "normal_activity")
        self.assertEqual(bend_event["severity"], 0)

    def test_extended_action_standard_maps_added_labels(self) -> None:
        cases = (
            ("A07_controlled_lie_down", "controlled_lie_down", "normal_activity", 0),
            ("A11_assisted_sit_or_lowering", "assisted_sit_or_lowering", "normal_activity", 0),
            ("A12_normal_hop", "normal_hop", "normal_activity", 0),
            ("D05_seated_fall", "seated_fall", "fall", 4),
        )
        for label, action_name, event_type, severity in cases:
            with self.subTest(label=label):
                xml = CVAT_PROJECT_XML.replace(
                    'label="A04_normal_sit_to_stand"', f'label="{label}"', 1
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "annotations.xml"
                    path.write_text(xml, encoding="utf-8")
                    converted = convert_cvat_xml(path, fps=2.0)

                self.assertEqual(converted.action_labels[0]["action_name"], action_name)
                self.assertEqual(converted.action_labels[0]["event_type"], event_type)
                self.assertEqual(converted.event_labels[0]["event_type"], event_type)
                self.assertEqual(converted.event_labels[0]["severity"], severity)

    def test_cvat_label_config_matches_canonical_action_dictionary(self) -> None:
        config_path = (
            Path(__file__).parents[1] / "configs/data/fall_risk_cvat_labels_v2.json"
        )
        labels = json.loads(config_path.read_text(encoding="utf-8"))
        expected_names = [f"{action_id}_{name}" for action_id, name in ACTION_NAMES.items()]

        self.assertEqual([label["name"] for label in labels], expected_names)
        self.assertEqual(len(labels), len(ACTION_NAMES))
        for label in labels:
            self.assertEqual(label["type"], "rectangle")
            self.assertEqual(
                [attribute["name"] for attribute in label["attributes"]],
                ["quality", "target_subject", "note"],
            )

    def test_identity_like_labeler_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(CVAT_PROJECT_XML, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pseudonymous"):
                convert_cvat_xml(
                    path,
                    fps=2.0,
                    labeler="private@example.invalid",
                )

    def test_pair_write_rejects_overwrite_and_rolls_back_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = root / "annotations.xml"
            action_path = root / "action.jsonl"
            event_path = root / "event.jsonl"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")
            action_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_fall_label_jsonl(
                    xml_path,
                    action_output_path=action_path,
                    event_output_path=event_path,
                    fps=2.0,
                )
            self.assertEqual(action_path.read_text(encoding="utf-8"), "existing\n")
            self.assertFalse(event_path.exists())

            action_path.unlink()
            real_link = os.link
            calls = 0

            def fail_second_link(source: str | os.PathLike, target: str | os.PathLike) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated event replace failure")
                real_link(source, target)

            with mock.patch(
                "elderly_monitoring.modules.fall_risk.annotations.os.link",
                side_effect=fail_second_link,
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_fall_label_jsonl(
                        xml_path,
                        action_output_path=action_path,
                        event_output_path=event_path,
                        fps=2.0,
                    )
            self.assertFalse(action_path.exists())
            self.assertFalse(event_path.exists())

    def test_pair_write_preserves_competing_output_created_during_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = root / "annotations.xml"
            action_path = root / "action.jsonl"
            event_path = root / "event.jsonl"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")
            competing_payload = b'{"owner":"other-process"}\n'
            real_link = os.link
            calls = 0

            def race_on_second_link(
                source: str | os.PathLike, target: str | os.PathLike
            ) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    Path(target).write_bytes(competing_payload)
                real_link(source, target)

            with mock.patch(
                "elderly_monitoring.modules.fall_risk.annotations.os.link",
                side_effect=race_on_second_link,
            ):
                with self.assertRaises(FileExistsError):
                    write_fall_label_jsonl(
                        xml_path,
                        action_output_path=action_path,
                        event_output_path=event_path,
                        fps=2.0,
                    )

            self.assertFalse(action_path.exists())
            self.assertEqual(event_path.read_bytes(), competing_payload)

    def test_le2i_import_normalizes_one_based_windows_and_skips_bbox_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            home_txt = root / "video (31).txt"
            embedded_txt = root / "video (26).txt"
            coffee_txt = root / "video (50).txt"
            zero_txt = root / "video (61).txt"
            lecture_txt = root / "lecture.txt"
            home_txt.write_bytes(b"10\r\n20\r\n1,1,0,0,1,1\r\n")
            embedded_txt.write_bytes(
                b"1,1,0,0,1,1\r\n10\r\n20\r\n2,1,0,0,1,1\r\n"
            )
            coffee_txt.write_bytes(b"1,1,0,0,1,1\r\n2,1,0,0,1,1\r\n")
            zero_txt.write_bytes(b"0\r\n0\r\n1,1,0,0,1,1\r\n")
            lecture_txt.write_text("2\n4\n1,1,0,0,1,1\n", encoding="utf-8")
            rows = [
                self._manifest_row(
                    video_id="le2i_home_02_video_31",
                    path=root / "video (31).avi",
                    annotation_path=home_txt,
                    fps_num=2,
                    frame_count=20,
                    subset="home_02",
                ),
                self._manifest_row(
                    video_id="le2i_coffee_room_01_video_26",
                    path=root / "video (26).avi",
                    annotation_path=embedded_txt,
                    fps_num=25,
                    frame_count=20,
                    subset="coffee_room_01",
                    scene_region="coffee_room",
                ),
                self._manifest_row(
                    video_id="le2i_coffee_room_02_video_50",
                    path=root / "video (50).avi",
                    annotation_path=coffee_txt,
                    fps_num=25,
                    frame_count=2,
                    subset="coffee_room_02",
                    scene_region="coffee_room",
                ),
                self._manifest_row(
                    video_id="le2i_coffee_room_02_video_61",
                    path=root / "video (61).avi",
                    annotation_path=zero_txt,
                    fps_num=25,
                    frame_count=2,
                    subset="coffee_room_02",
                    scene_region="coffee_room",
                ),
                self._manifest_row(
                    video_id="le2i_lecture_room_video_1",
                    path=root / "lecture.avi",
                    annotation_path=lecture_txt,
                    fps_num=25,
                    frame_count=10,
                    subset="lecture_room",
                    scene_region="lecture_room",
                ),
            ]
            self._write_manifest(manifest_path, rows)

            imported = import_le2i_fall_labels(manifest_path)

        self.assertEqual(len(imported.event_labels), 2)
        event = next(
            row
            for row in imported.event_labels
            if row["video_id"] == "le2i_home_02_video_31"
        )
        self.assertEqual(event["video_id"], "le2i_home_02_video_31")
        self.assertEqual((event["start_frame"], event["end_frame"]), (9, 19))
        self.assertEqual((event["source_start_frame"], event["source_end_frame"]), (10, 20))
        self.assertEqual(event["source_frame_index_base"], 1)
        self.assertEqual(event["label_source"], "le2i_txt")
        self.assertNotIn("eligibility", event)
        self.assertIn(
            "le2i_coffee_room_01_video_26",
            {row["video_id"] for row in imported.event_labels},
        )
        self.assertEqual(imported.report["bbox_only_without_window"], 1)
        self.assertEqual(imported.report["explicit_no_fall_window"], 1)
        self.assertEqual(imported.report["excluded_unsupervised_subset"], 1)

    def test_toaga_import_generates_full_video_normal_walk_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            source_table = root / "Table_1.xlsx"
            source_table.write_bytes(b"official participant table")
            top_video = root / "OAW01-top.mp4"
            bottom_video = root / "OAW01-bottom.mp4"
            excluded_video = root / "OAW02-top.mp4"
            top_video.write_bytes(b"top")
            bottom_video.write_bytes(b"bottom")
            excluded_video.write_bytes(b"excluded")
            rows = [
                {
                    **self._manifest_row(
                        video_id="toaga_oaw01_top",
                        path=top_video,
                        annotation_path=source_table,
                        fps_num=30,
                        frame_count=90,
                        subset="walking",
                        scene_region="walking_lab",
                    ),
                    "dataset": "toaga",
                    "subject_id": "toaga_oaw01",
                    "source_group_id": "toaga_oaw01",
                    "original_event_id": "toaga_oaw01_walking",
                    "view": "top",
                },
                {
                    **self._manifest_row(
                        video_id="toaga_oaw01_bottom",
                        path=bottom_video,
                        annotation_path=source_table,
                        fps_num=25,
                        frame_count=75,
                        subset="walking",
                        scene_region="walking_lab",
                    ),
                    "dataset": "toaga",
                    "subject_id": "toaga_oaw01",
                    "source_group_id": "toaga_oaw01",
                    "original_event_id": "toaga_oaw01_walking",
                    "view": "bottom",
                },
                {
                    **self._manifest_row(
                        video_id="toaga_oaw02_top",
                        path=excluded_video,
                        annotation_path=source_table,
                        fps_num=30,
                        frame_count=90,
                        subset="walking",
                        scene_region="walking_lab",
                    ),
                    "dataset": "toaga",
                    "subject_id": "toaga_oaw02",
                    "source_group_id": "toaga_oaw02",
                    "original_event_id": "toaga_oaw02_walking",
                    "view": "top",
                    "eligibility": False,
                    "exclusion_reasons": ["duplicate_content"],
                },
            ]
            self._write_manifest(manifest_path, rows)

            first = import_toaga_normal_walk_labels(manifest_path)
            second = import_toaga_normal_walk_labels(manifest_path)

        self.assertEqual(first.report["manifest_toaga_walking_videos"], 3)
        self.assertEqual(first.report["imported_normal_walk_labels"], 2)
        self.assertEqual(first.report["excluded_ineligible"], 1)
        self.assertEqual(first.action_labels, second.action_labels)
        self.assertEqual(len(first.action_labels), 2)
        top = next(
            row for row in first.action_labels if row["video_id"] == "toaga_oaw01_top"
        )
        self.assertEqual((top["action_id"], top["action_name"]), ("A01", "normal_walk"))
        self.assertEqual(top["event_type"], "normal_activity")
        self.assertEqual((top["start_frame"], top["end_frame"]), (0, 89))
        self.assertEqual((top["start_time"], top["end_time"]), (0.0, 2.9667))
        self.assertEqual(top["source"], "toaga_official_walking")
        self.assertEqual(top["subject_id"], "toaga_oaw01")
        self.assertNotIn("cvat_task_id", top)
        self.assertNotIn("bbox_start", top)

    def test_ntu_rgbd_import_uses_manual_exact_review_and_excludes_a043(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            label_map = root / "ntu-map.json"
            label_map.write_text(
                json.dumps(
                    {
                        "schema_version": "ntu-rgbd-clip-label-map-v2",
                        "mapping_id": "ntu-rgbd-clip-label-map-v2",
                        "boundary_review": {
                            "decision_id": "ntu-rgbd-manual-boundary-review-20260725",
                            "reviewed_at": "2026-07-25",
                            "reviewer_id": "project_owner",
                            "boundary_precision": "exact",
                            "training_tier": "primary",
                        },
                        "mappings": [
                            {
                                "source_action_code": "A008",
                                "mode": "manual_exact",
                                "action_id": "A03",
                            },
                            {
                                "source_action_code": "A009",
                                "mode": "manual_exact",
                                "action_id": "A04",
                            },
                            {
                                "source_action_code": "A042",
                                "mode": "manual_exact",
                                "action_id": "C03",
                            },
                            {
                                "source_action_code": "A080",
                                "mode": "manual_exact",
                                "action_id": "A05",
                            },
                            {
                                "source_action_code": "A043",
                                "mode": "excluded",
                                "action_id": None,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            rows = []
            for source_code, video_id, frame_count in (
                ("A008", "ntu_rgbd_s001_p001_r001_a008_c001", 90),
                ("A009", "ntu_rgbd_s001_p001_r001_a009_c001", 91),
                ("A042", "ntu_rgbd_s001_p001_r001_a042_c001", 92),
                ("A080", "ntu_rgbd_s001_p001_r001_a080_c001", 93),
                ("A043", "ntu_rgbd_s001_p001_r001_a043_c001", 94),
            ):
                video = root / f"{video_id}.avi"
                video.write_bytes(video_id.encode("utf-8"))
                rows.append(
                    {
                        **self._manifest_row(
                            video_id=video_id,
                            path=video,
                            fps_num=30,
                            frame_count=frame_count,
                            subset="setup_s001",
                            scene_region="unknown",
                        ),
                        "dataset": "ntu_rgbd",
                        "subject_id": "ntu_rgbd_p001",
                        "source_group_id": "ntu_rgbd_subject_p001",
                        "original_event_id": video_id.rsplit("_c", 1)[0],
                        "view": "c001",
                        "source_action_code": source_code,
                    }
                )
            self._write_manifest(manifest_path, rows)

            first = import_ntu_rgbd_clip_labels(manifest_path, label_map)
            second = import_ntu_rgbd_clip_labels(manifest_path, label_map)

        self.assertEqual(first.action_labels, second.action_labels)
        self.assertEqual(first.report["manifest_ntu_rgbd_videos"], 5)
        self.assertEqual(first.report["imported_action_labels"], 4)
        self.assertEqual(first.report["excluded_videos"], 1)
        self.assertEqual(first.report["manual_exact_action_labels"], 4)
        self.assertEqual(
            first.report["boundary_review"]["decision_id"],
            "ntu-rgbd-manual-boundary-review-20260725",
        )
        self.assertEqual(first.report["imported_by_source_action_code"], {
            "A008": 1,
            "A009": 1,
            "A042": 1,
            "A080": 1,
        })
        self.assertEqual(
            {row["action_id"] for row in first.action_labels},
            {"A03", "A04", "A05", "C03"},
        )
        stumble = next(row for row in first.action_labels if row["action_id"] == "C03")
        self.assertEqual(stumble["action_name"], "stumble_recovery")
        self.assertEqual(stumble["event_type"], "near_fall")
        self.assertEqual((stumble["start_frame"], stumble["end_frame"]), (0, 91))
        self.assertEqual((stumble["start_time"], stumble["end_time"]), (0.0, 3.0333))
        self.assertEqual(stumble["source"], "ntu_rgbd_manual_clip_label")
        self.assertEqual(
            stumble["labeler"],
            "project_owner_ntu_rgbd_boundary_review_20260725",
        )
        self.assertIn("human-reviewed exact full-clip boundary", stumble["note"])
        self.assertNotIn("cvat_task_id", stumble)
        self.assertNotIn("bbox_start", stumble)

    def test_cli_defaults_to_v2_generated_outputs(self) -> None:
        script = Path(__file__).parents[1] / "scripts/annotation/convert_cvat_fall_labels.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = root / "annotations.xml"
            manifest_path = root / "manifest.jsonl"
            xml_path.write_text(CVAT_PROJECT_XML, encoding="utf-8")
            rows = [
                self._manifest_row(
                    video_id="le2i_home_01_video_1",
                    path=root / "video (1).avi",
                    fps_num=2,
                    frame_count=5,
                ),
                self._manifest_row(
                    video_id="le2i_home_01_video_2",
                    path=root / "video (2).avi",
                    fps_num=2,
                    frame_count=4,
                ),
            ]
            self._write_manifest(manifest_path, rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(xml_path),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = root / "data/annotations/fall_risk/generated/v2"
            self.assertTrue((generated / "action_labels.jsonl").exists())
            self.assertTrue((generated / "event_labels.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
