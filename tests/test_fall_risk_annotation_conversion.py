import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from elderly_monitoring.modules.fall_risk.annotations import (
    convert_cvat_xml,
    import_le2i_fall_labels,
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
  <track id="10" label="A04_normal_stand" source="manual" task_id="1">
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
            "license_id": "CC-BY-NC-SA-3.0",
            "consent_id": None,
            "review_status": "pending",
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
                review_status="reviewed",
                default_scene="home",
            )

        self.assertEqual(len(converted.action_labels), 4)
        self.assertEqual(len(converted.event_labels), 4)

        first = converted.action_labels[0]
        self.assertEqual(first["video_id"], "le2i_home_01_video_1")
        self.assertEqual(first["file_path"], "/dataset/FallDataset/Home_01/Videos/video (1).avi")
        self.assertEqual(first["action_id"], "A04")
        self.assertEqual(first["action_name"], "normal_stand")
        self.assertEqual(first["start_frame"], 0)
        self.assertEqual(first["end_frame"], 2)
        self.assertEqual(first["start_time"], 0.0)
        self.assertEqual(first["end_time"], 1.0)
        self.assertEqual(first["bbox_start"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(first["labeler"], "labeler_01")
        self.assertEqual(first["review_status"], "reviewed")

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
        self.assertEqual(first.event_labels[0]["mapping_version"], "fall-action-event-v1")
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
        bad_xml = CVAT_PROJECT_XML.replace("A04_normal_stand", "A99_made_up", 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.xml"
            path.write_text(bad_xml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown action label"):
                convert_cvat_xml(path, fps=2.0)

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
        self.assertEqual(event["review_status"], "auto_imported")
        self.assertIn(
            "le2i_coffee_room_01_video_26",
            {row["video_id"] for row in imported.event_labels},
        )
        self.assertEqual(imported.report["bbox_only_without_window"], 1)
        self.assertEqual(imported.report["explicit_no_fall_window"], 1)
        self.assertEqual(imported.report["excluded_unsupervised_subset"], 1)

    def test_cli_defaults_to_generated_and_cannot_promote_review_status(self) -> None:
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
            generated = root / "data/annotations/fall_risk/generated/v1"
            self.assertTrue((generated / "action_labels.jsonl").exists())
            self.assertTrue((generated / "event_labels.jsonl").exists())

            promoted = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(xml_path),
                    "--manifest",
                    str(manifest_path),
                    "--review-status",
                    "final",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(promoted.returncode, 0)
            self.assertIn("cannot promote", promoted.stderr.lower())


if __name__ == "__main__":
    unittest.main()
