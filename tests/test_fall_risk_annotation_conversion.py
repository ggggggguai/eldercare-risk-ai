import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import (
    convert_cvat_xml,
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
        self.assertEqual(fall_event["label_source"], "manual_reviewed")

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


if __name__ == "__main__":
    unittest.main()
