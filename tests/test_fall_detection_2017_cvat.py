from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET

from elderly_monitoring.modules.fall_risk.fall_detection_2017_cvat import (
    _read_image_revision,
    _resample_track,
)


def _zip_xml(root: ET.Element) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("annotations.xml", ET.tostring(root, encoding="utf-8"))
    return buffer.getvalue()


def _box(frame: int, *, label: str = "A01_normal_walk") -> ET.Element:
    return ET.Element(
        "box",
        {
            "frame": str(frame),
            "outside": "0",
            "xtl": str(frame),
            "ytl": "0",
            "xbr": str(frame + 10),
            "ybr": "20",
        },
    )


def test_anomalous_frame_resampling_preserves_endpoints_and_target_timeline() -> None:
    track = ET.Element("track", {"label": "A01_normal_walk"})
    for frame in range(227):
        track.append(_box(frame))

    normalized = _resample_track(track, 227, 57)
    boxes = normalized.findall("box")

    assert [int(box.get("frame")) for box in boxes] == list(range(57))
    assert boxes[0].get("xtl") == "0"
    assert boxes[-1].get("xtl") == "226"


def test_image_revision_groups_contiguous_labels_into_tracks() -> None:
    root = ET.Element("annotations")
    project = ET.SubElement(ET.SubElement(root, "meta"), "project")
    tasks = ET.SubElement(project, "tasks")
    task = ET.SubElement(tasks, "task")
    ET.SubElement(task, "id").text = "552"
    ET.SubElement(task, "size").text = "4"
    ET.SubElement(task, "source").text = "20240922115152.mp4"
    for frame, label in enumerate(
        ("A01_normal_walk", "A01_normal_walk", "U01_unable_to_judge", "U01_unable_to_judge")
    ):
        image = ET.SubElement(root, "image", {"task_id": "552", "id": str(frame)})
        ET.SubElement(
            image,
            "box",
            {
                "label": label,
                "xtl": "0",
                "ytl": "0",
                "xbr": "10",
                "ybr": "20",
            },
        )

    result = _read_image_revision(_zip_xml(root), source_name="revision.zip")
    tracks = result["20240922115152.mp4"]

    assert [track.get("label") for track in tracks] == [
        "A01_normal_walk",
        "U01_unable_to_judge",
    ]
    assert [int(box.get("frame")) for box in tracks[0].findall("box")] == [0, 1]
    assert [int(box.get("frame")) for box in tracks[1].findall("box")] == [2, 3]
