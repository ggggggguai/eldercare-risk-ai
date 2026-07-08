from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import xml.etree.ElementTree as ET


ACTION_EVENT_MAP: dict[str, tuple[str, int]] = {
    "A01": ("normal_activity", 0),
    "A02": ("normal_activity", 0),
    "A03": ("normal_activity", 0),
    "A04": ("normal_activity", 0),
    "B01": ("gait_instability", 1),
    "B02": ("gait_instability", 2),
    "B03": ("gait_instability", 2),
    "B04": ("gait_instability", 2),
    "B05": ("unstable_turning", 2),
    "B06": ("sit_stand_difficulty", 2),
    "C01": ("sit_stand_difficulty", 3),
    "C02": ("wall_support", 3),
    "C03": ("near_fall", 3),
    "C04": ("wall_support", 3),
    "C05": ("rapid_body_drop", 3),
    "D01": ("fall", 4),
    "D02": ("fall", 4),
    "D03": ("fall", 4),
    "D04": ("long_static", 4),
    "U01": ("uncertain", 0),
}


@dataclass(frozen=True)
class CvatTaskInfo:
    task_id: str
    name: str
    size: int
    source: str
    start_frame: int = 0
    stop_frame: int = 0
    frame_offset: int = 0


@dataclass(frozen=True)
class ConvertedFallLabels:
    action_labels: list[dict[str, Any]]
    event_labels: list[dict[str, Any]]


def convert_cvat_xml(
    input_path: Path | str,
    *,
    fps: float = 24.0,
    file_root: Path | str | None = None,
    labeler: str = "unknown",
    review_status: str = "pending",
    default_subject_id: str = "unknown",
    default_scene: str = "home",
    default_view: str = "fixed_camera",
) -> ConvertedFallLabels:
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    with _open_cvat_xml(input_path) as xml_path:
        root = ET.parse(xml_path).getroot()
        tasks = _parse_tasks(root)
        action_labels = []
        event_labels = []

        for track in root.findall("track"):
            task = _task_for_track(track, tasks)
            action_id, action_name = _parse_action_label(track.attrib.get("label", ""))
            active_boxes = _active_boxes(track)
            if not active_boxes:
                continue

            start_frame = _normalize_frame(active_boxes[0], task)
            end_frame = _normalize_frame(active_boxes[-1], task)
            if end_frame < start_frame:
                raise ValueError(
                    f"track {track.attrib.get('id')} has end_frame before start_frame"
                )

            event_type, severity = ACTION_EVENT_MAP.get(action_id, ("uncertain", 0))
            attributes = _track_attributes(active_boxes)
            subject_id = attributes.get("target_subject") or default_subject_id
            quality = attributes.get("quality") or "clear"
            note = attributes.get("note") or ""
            video_id = _video_id_from_task_name(task.name)
            file_path = _file_path(task.source, file_root)
            label_source = _label_source(review_status)

            action_record = {
                "video_id": video_id,
                "file_path": file_path,
                "subject_id": subject_id,
                "scene": default_scene,
                "view": default_view,
                "action_id": action_id,
                "action_name": action_name,
                "event_type": event_type,
                "start_time": _frame_to_time(start_frame, fps),
                "end_time": _frame_to_time(end_frame, fps),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "labeler": labeler,
                "review_status": review_status,
                "quality": quality,
                "note": note,
                "source": "cvat",
                "cvat_task_id": task.task_id,
                "cvat_track_id": _maybe_int(track.attrib.get("id")),
                "bbox_start": _bbox(active_boxes[0]),
                "bbox_end": _bbox(active_boxes[-1]),
            }
            action_labels.append(action_record)

            event_labels.append(
                {
                    "video_id": video_id,
                    "event_type": event_type,
                    "start_time": action_record["start_time"],
                    "end_time": action_record["end_time"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "severity": severity,
                    "label_source": label_source,
                    "review_status": review_status,
                    "note": note or _event_note(action_id, action_name),
                    "source_action_id": action_id,
                    "source_action_name": action_name,
                    "cvat_task_id": task.task_id,
                    "cvat_track_id": action_record["cvat_track_id"],
                }
            )

    return ConvertedFallLabels(
        action_labels=sorted(action_labels, key=_record_sort_key),
        event_labels=sorted(event_labels, key=_record_sort_key),
    )


def write_fall_label_jsonl(
    input_path: Path | str,
    *,
    action_output_path: Path | str,
    event_output_path: Path | str,
    fps: float = 24.0,
    file_root: Path | str | None = None,
    labeler: str = "unknown",
    review_status: str = "pending",
    default_subject_id: str = "unknown",
    default_scene: str = "home",
    default_view: str = "fixed_camera",
) -> dict[str, int]:
    converted = convert_cvat_xml(
        input_path,
        fps=fps,
        file_root=file_root,
        labeler=labeler,
        review_status=review_status,
        default_subject_id=default_subject_id,
        default_scene=default_scene,
        default_view=default_view,
    )
    action_count = _write_jsonl(converted.action_labels, Path(action_output_path))
    event_count = _write_jsonl(converted.event_labels, Path(event_output_path))
    return {"action_labels": action_count, "event_labels": event_count}


def _open_cvat_xml(input_path: Path | str):
    path = Path(input_path)
    if path.suffix.lower() == ".zip":
        return _ZipXmlContext(path)
    return _PlainXmlContext(path)


class _PlainXmlContext:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _ZipXmlContext:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._tmpdir: TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmpdir = TemporaryDirectory()
        output_dir = Path(self._tmpdir.name)
        with zipfile.ZipFile(self.path) as archive:
            xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
            if not xml_names:
                raise ValueError(f"no XML file found in {self.path}")
            preferred = "annotations.xml" if "annotations.xml" in xml_names else xml_names[0]
            archive.extract(preferred, output_dir)
        return output_dir / preferred

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def _parse_tasks(root: ET.Element) -> dict[str, CvatTaskInfo]:
    project_tasks = root.findall("./meta/project/tasks/task")
    single_task = root.find("./meta/task")
    task_elements = project_tasks or ([single_task] if single_task is not None else [])

    tasks: dict[str, CvatTaskInfo] = {}
    cumulative_offset = 0
    for index, task in enumerate(task_elements):
        task_id = task.findtext("id") or str(index)
        size = int(task.findtext("size") or 0)
        start_frame = int(task.findtext("start_frame") or 0)
        stop_frame = int(task.findtext("stop_frame") or (size - 1 if size else 0))
        info = CvatTaskInfo(
            task_id=task_id,
            name=task.findtext("name") or f"task_{task_id}",
            size=size,
            source=task.findtext("source") or "",
            start_frame=start_frame,
            stop_frame=stop_frame,
            frame_offset=cumulative_offset,
        )
        tasks[task_id] = info
        cumulative_offset += size

    if not tasks:
        tasks["0"] = CvatTaskInfo(task_id="0", name="unknown_task", size=0, source="")
    return tasks


def _task_for_track(track: ET.Element, tasks: dict[str, CvatTaskInfo]) -> CvatTaskInfo:
    task_id = track.attrib.get("task_id")
    if task_id and task_id in tasks:
        return tasks[task_id]
    if len(tasks) == 1:
        return next(iter(tasks.values()))
    raise ValueError(f"track {track.attrib.get('id')} is missing a known task_id")


def _active_boxes(track: ET.Element) -> list[ET.Element]:
    boxes = [
        box
        for box in track.findall("box")
        if box.attrib.get("outside") != "1"
    ]
    return sorted(boxes, key=lambda box: int(box.attrib["frame"]))


def _normalize_frame(box: ET.Element, task: CvatTaskInfo) -> int:
    frame = int(box.attrib["frame"])
    if frame >= task.frame_offset and task.frame_offset:
        return frame - task.frame_offset
    return frame - task.start_frame


def _parse_action_label(label: str) -> tuple[str, str]:
    match = re.match(r"^(?P<action_id>[A-DU]\d{2})[_-](?P<name>.+)$", label)
    if match:
        return match.group("action_id"), match.group("name")
    raise ValueError(f"CVAT label must start with an action code like A01_: {label!r}")


def _track_attributes(active_boxes: list[ET.Element]) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for box in active_boxes:
        for attr in box.findall("attribute"):
            name = attr.attrib.get("name")
            if name and attr.text and name not in attributes:
                attributes[name] = attr.text
    return attributes


def _video_id_from_task_name(task_name: str) -> str:
    marker = "fall_risk__"
    if task_name.startswith(marker):
        parts = task_name.split("__")
        if len(parts) >= 4:
            return _canonical_video_id(parts[1], parts[2], parts[-1])
    return _sanitize_identifier(task_name)


def _canonical_video_id(dataset: str, subset: str, raw_video_id: str) -> str:
    if dataset == "le2i_imvia":
        match = re.fullmatch(
            r"video \((?P<index>\d+)\)\.(?:avi|mp4)",
            raw_video_id.strip(),
            re.IGNORECASE,
        )
        if match:
            subset_id = _sanitize_identifier(subset)
            return f"le2i_{subset_id}_video_{int(match.group('index'))}"
    return _sanitize_identifier(raw_video_id)


def _sanitize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def _file_path(source: str, file_root: Path | str | None) -> str:
    if not source:
        return ""
    if file_root is None:
        return source
    return str(Path(file_root) / source)


def _frame_to_time(frame: int, fps: float) -> float:
    return round(frame / fps, 4)


def _bbox(box: ET.Element) -> list[float]:
    return [
        round(float(box.attrib[key]), 2)
        for key in ("xtl", "ytl", "xbr", "ybr")
    ]


def _label_source(review_status: str) -> str:
    if review_status in {"reviewed", "final"}:
        return "manual_reviewed"
    return "manual"


def _event_note(action_id: str, action_name: str) -> str:
    if action_id.startswith("D"):
        return "由人工动作标签映射生成"
    return f"由人工动作标签 {action_id}_{action_name} 映射生成"


def _maybe_int(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _record_sort_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(record.get("video_id", "")),
        int(record.get("start_frame", 0)),
        int(record.get("end_frame", 0)),
    )


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(records)
