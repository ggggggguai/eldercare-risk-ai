from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

from elderly_monitoring.modules.fall_risk.annotations import ACTION_EVENT_MAP, ACTION_NAMES
from elderly_monitoring.modules.fall_risk.caucafall import (
    CAUCAFALL_LABEL_ALIASES,
    _normalize_inner_export,
)


def _task_xml(task_id: int, source: str, label: str) -> ET.Element:
    task = ET.Element("task")
    for name, value in (
        ("id", str(task_id)),
        ("name", source),
        ("size", "3"),
        ("start_frame", "0"),
        ("stop_frame", "2"),
        ("source", source),
    ):
        ET.SubElement(task, name).text = value
    return task


def _inner_payload() -> bytes:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    project = ET.SubElement(meta, "project")
    tasks = ET.SubElement(project, "tasks")
    for index in range(10):
        source = f"WalkS{index + 1}.avi"
        task = _task_xml(index + 1, source, "A01_normal_walk")
        tasks.append(task)
    ET.SubElement(meta, "owner").append(ET.Element("username"))
    ET.SubElement(meta, "email").text = "private@example.invalid"
    for index, label in enumerate(("A03_controlled_sit_down", "A08_routine_support_contact")):
        track = ET.SubElement(
            root,
            "track",
            {"id": str(index), "label": label, "task_id": str(index + 1)},
        )
        for frame in range(3):
            box = ET.SubElement(
                track,
                "box",
                {
                    "frame": str(frame),
                    "outside": "0",
                    "xtl": "1",
                    "ytl": "2",
                    "xbr": "10",
                    "ybr": "20",
                },
            )
            ET.SubElement(box, "attribute", {"name": "quality"}).text = "clear"
            ET.SubElement(box, "attribute", {"name": "target_subject"}).text = "unknown"
    buffer = tempfile.SpooledTemporaryFile()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("annotations.xml", ET.tostring(root, encoding="utf-8"))
    buffer.seek(0)
    return buffer.read()


def test_caucafall_aliases_have_v2_targets() -> None:
    for raw, canonical in CAUCAFALL_LABEL_ALIASES.items():
        action_id, action_name = canonical.split("_", 1)
        assert ACTION_NAMES[action_id] == action_name
        assert action_id in ACTION_EVENT_MAP


def test_caucafall_normalization_removes_identity_and_sets_subject() -> None:
    rows = [
        {
            "dataset": "caucafall",
            "path": f"data/external/caucafall/raw/Subject.{i}/WalkS{i}.avi",
            "video_id": f"caucafall_s{i:02d}_walk",
            "subset": "walk",
            "subject_id": f"caucafall_s{i:02d}",
            "frame_count": 3,
        }
        for i in range(1, 11)
    ]
    raw_counts = {}
    normalized_counts = {}
    aliases = {}
    removed = {}
    from collections import Counter

    payload, info = _normalize_inner_export(
        _inner_payload(),
        {Path(row["path"]).name: row for row in rows},
        Counter(raw_counts),
        Counter(normalized_counts),
        Counter(aliases),
        Counter(removed),
    )
    with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))
    assert info["task_count"] == 10
    assert root.find(".//owner") is None
    assert root.find(".//email") is None
    assert root.find(".//track").attrib["label"] == "A03_controlled_sit_down"
    assert root.find(".//track/box/attribute[@name='target_subject']").text == "caucafall_s01"
