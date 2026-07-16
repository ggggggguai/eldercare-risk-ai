from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET


ACTION_NAMES: dict[str, str] = {
    "A01": "normal_walk",
    "A02": "normal_turn",
    "A03": "normal_sit",
    "A04": "normal_stand",
    "B01": "slow_walk",
    "B02": "dragging_walk",
    "B03": "shuffling_walk",
    "B04": "swaying_walk",
    "B05": "unstable_turn",
    "B06": "slow_sit_to_stand",
    "C01": "failed_sit_to_stand",
    "C02": "wall_support_walk",
    "C03": "stumble_recovery",
    "C04": "rapid_support_contact",
    "C05": "rapid_body_drop_recovery",
    "D01": "forward_fall",
    "D02": "lateral_fall",
    "D03": "backward_fall",
    "D04": "long_static_after_fall",
    "U01": "unable_to_judge",
}

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

ACTION_EVENT_MAPPING_VERSION = "fall-action-event-v1"
QUALITY_VALUES = {
    "clear",
    "partial_occlusion",
    "heavy_occlusion",
    "low_light",
    "off_screen",
    "multi_person_uncertain",
}
REVIEW_STATUS_VALUES = {"pending", "reviewed", "final"}
IDENTITY_METADATA_TAGS = {"owner", "assignee", "username", "email"}
_PSEUDONYM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


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
class VideoMetadata:
    asset_id: str
    video_id: str
    path: str
    fps_num: int
    fps_den: int
    frame_count: int
    duration_sec: float
    subject_id: str
    scene_region: str
    view: str
    source_group_id: str
    annotation_path: str | None
    manifest_record: Mapping[str, Any]

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


@dataclass(frozen=True)
class ConvertedFallLabels:
    action_labels: list[dict[str, Any]]
    event_labels: list[dict[str, Any]]
    identity_metadata_present: bool = False
    source_export_sha256: str = ""


@dataclass(frozen=True)
class Le2iImportResult:
    event_labels: list[dict[str, Any]]
    report: dict[str, int]


def read_le2i_fall_window(annotation_path: Path | str) -> tuple[int, int] | None:
    """Return the official one-based LE2I fall window, or None when absent."""
    path = Path(annotation_path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"LE2I annotation is not UTF-8: {path}") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty LE2I annotation: {path}")
    header_indices = [index for index, line in enumerate(lines) if _integer_line(line)]
    if not header_indices:
        _validate_le2i_bbox_rows(lines, path)
        return None
    if len(header_indices) != 2 or header_indices[1] != header_indices[0] + 1:
        raise ValueError(f"invalid LE2I fall window header: {path}")
    source_start = int(lines[header_indices[0]])
    source_end = int(lines[header_indices[1]])
    bbox_lines = [
        line for index, line in enumerate(lines) if index not in set(header_indices)
    ]
    if bbox_lines:
        _validate_le2i_bbox_rows(bbox_lines, path)
    if source_start == 0 and source_end == 0:
        return None
    if source_start < 1 or source_end < source_start:
        raise ValueError(f"invalid LE2I one-based fall window: {path}")
    return source_start, source_end


def convert_cvat_xml(
    input_path: Path | str,
    *,
    fps: float | None = 24.0,
    manifest_path: Path | str | None = None,
    file_root: Path | str | None = None,
    labeler: str = "unknown",
    review_status: str = "pending",
    default_subject_id: str = "unknown",
    default_scene: str = "home",
    default_view: str = "fixed_camera",
    time_tolerance_sec: float = 0.001,
) -> ConvertedFallLabels:
    """Convert CVAT video tracks into action and mapped-event candidates.

    The legacy single-FPS API remains available for development fixtures. A
    manifest-backed conversion always uses each video's exact rational FPS and
    treats a non-null ``fps`` argument as an explicit override that must agree.
    """
    if review_status not in REVIEW_STATUS_VALUES:
        raise ValueError(f"invalid review_status: {review_status!r}")
    _require_pseudonymous_identifier(labeler, "labeler", allow_unknown=True)
    _require_pseudonymous_identifier(
        default_subject_id, "default_subject_id", allow_unknown=True
    )
    if time_tolerance_sec < 0 or not math.isfinite(time_tolerance_sec):
        raise ValueError("time_tolerance_sec must be finite and non-negative")
    if manifest_path is None and (fps is None or fps <= 0 or not math.isfinite(fps)):
        raise ValueError("fps must be greater than 0 without a manifest")

    source_path = Path(input_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = _sha256_file(source_path)
    source_export_id = f"cvat_{source_sha256[:24]}"
    source_annotation_path = _portable_source_path(source_path)
    manifest = load_video_manifest(manifest_path) if manifest_path is not None else {}

    with _open_cvat_xml(source_path) as xml_path:
        root = ET.parse(xml_path).getroot()
        tasks = _parse_tasks(root)
        identity_metadata_present = any(
            element.tag.lower() in IDENTITY_METADATA_TAGS for element in root.iter()
        )
        action_labels: list[dict[str, Any]] = []
        event_labels: list[dict[str, Any]] = []
        seen_track_ids: set[str] = set()

        for track in root.findall("track"):
            track_id = track.attrib.get("id")
            if track_id is None or not track_id.strip():
                raise ValueError("CVAT track is missing id")
            if track_id in seen_track_ids:
                raise ValueError(f"duplicate CVAT track id: {track_id}")
            seen_track_ids.add(track_id)

            task = _task_for_track(track, tasks)
            action_id, action_name = _parse_action_label(track.attrib.get("label", ""))
            active_boxes = _normalized_active_boxes(track, task)
            if not active_boxes:
                raise ValueError(f"track {track_id} has no active boxes")
            start_frame, start_box = active_boxes[0]
            end_frame, end_box = active_boxes[-1]
            if end_frame < start_frame:
                raise ValueError(f"track {track_id} has end_frame before start_frame")

            video_id = _video_id_from_task_name(task.name)
            metadata = manifest.get(video_id)
            if manifest_path is not None and metadata is None:
                raise ValueError(f"video_id {video_id!r} is missing from manifest")

            if metadata is not None:
                if fps is not None and not math.isclose(
                    float(fps), metadata.fps, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"fps override conflicts with manifest for {video_id}: "
                        f"override={fps}, manifest={metadata.fps}"
                    )
                _validate_task_source(task, metadata)
                _validate_frame_bounds(video_id, start_frame, end_frame, metadata)
                timeline_num = metadata.fps_num
                timeline_den = metadata.fps_den
                file_path = metadata.path
                subject_default = metadata.subject_id
                scene = metadata.scene_region
                view = metadata.view
                asset_id = metadata.asset_id
            else:
                assert fps is not None
                timeline_num, timeline_den = _float_fps_ratio(float(fps))
                file_path = _file_path(task.source, file_root)
                subject_default = default_subject_id
                scene = default_scene
                view = default_view
                asset_id = video_id

            start_time = _frame_to_time(start_frame, timeline_num, timeline_den)
            end_time = _frame_to_time(end_frame, timeline_num, timeline_den)
            if metadata is not None and end_time > metadata.duration_sec + time_tolerance_sec:
                raise ValueError(
                    f"track {track_id} end_time exceeds duration for {video_id}"
                )

            event_type, severity = ACTION_EVENT_MAP[action_id]
            attributes = _track_attributes([box for _, box in active_boxes])
            subject_id = attributes.get("target_subject") or subject_default
            _require_pseudonymous_identifier(
                subject_id, "target_subject", allow_unknown=True
            )
            quality = attributes.get("quality") or "clear"
            if quality not in QUALITY_VALUES:
                raise ValueError(f"track {track_id} has invalid quality {quality!r}")
            note = attributes.get("note") or ""
            if _contains_contact_identifier(note):
                raise ValueError("note contains potential identity/contact data")
            if action_id == "U01" and not note.strip():
                raise ValueError(f"track {track_id} U01 requires a reason in note")

            source_record_id = (
                f"{source_export_id}:task:{task.task_id}:track:{track_id}"
            )
            action_label_id = _stable_id(
                "action", source_sha256, task.task_id, track_id, video_id
            )
            action_record = {
                "label_id": action_label_id,
                "source_record_id": source_record_id,
                "source_annotation_path": source_annotation_path,
                "source_annotation_sha256": source_sha256,
                "source_export_id": source_export_id,
                "asset_id": asset_id,
                "video_id": video_id,
                "file_path": file_path,
                "subject_id": subject_id,
                "scene": scene,
                "view": view,
                "action_id": action_id,
                "action_name": action_name,
                "event_type": event_type,
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_index_base": 0,
                "labeler": labeler,
                "review_status": review_status,
                "eligibility": False,
                "review_evidence_ids": [],
                "quality": quality,
                "note": note,
                "source": "cvat",
                "cvat_task_id": task.task_id,
                "cvat_track_id": _maybe_int(track_id),
                "bbox_start": _bbox(start_box),
                "bbox_end": _bbox(end_box),
            }
            action_labels.append(action_record)

            event_label_id = _stable_id(
                "event", action_label_id, ACTION_EVENT_MAPPING_VERSION
            )
            event_labels.append(
                {
                    "label_id": event_label_id,
                    "source_record_id": f"mapped:{action_label_id}",
                    "source_annotation_path": source_annotation_path,
                    "source_annotation_sha256": source_sha256,
                    "source_export_id": source_export_id,
                    "asset_id": asset_id,
                    "video_id": video_id,
                    "event_type": event_type,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_index_base": 0,
                    "severity": severity,
                    "label_source": "cvat_action_mapping",
                    "review_status": review_status,
                    "eligibility": False,
                    "review_evidence_ids": [],
                    "note": note or _event_note(action_id, action_name),
                    "source_action_id": action_id,
                    "source_action_name": action_name,
                    "source_action_label_id": action_label_id,
                    "mapping_version": ACTION_EVENT_MAPPING_VERSION,
                    "cvat_task_id": task.task_id,
                    "cvat_track_id": action_record["cvat_track_id"],
                }
            )

    return ConvertedFallLabels(
        action_labels=sorted(action_labels, key=_record_sort_key),
        event_labels=sorted(event_labels, key=_record_sort_key),
        identity_metadata_present=identity_metadata_present,
        source_export_sha256=source_sha256,
    )


def write_fall_label_jsonl(
    input_path: Path | str,
    *,
    action_output_path: Path | str,
    event_output_path: Path | str,
    fps: float | None = 24.0,
    manifest_path: Path | str | None = None,
    file_root: Path | str | None = None,
    labeler: str = "unknown",
    review_status: str = "pending",
    default_subject_id: str = "unknown",
    default_scene: str = "home",
    default_view: str = "fixed_camera",
    time_tolerance_sec: float = 0.001,
    overwrite: bool = False,
) -> dict[str, int]:
    converted = convert_cvat_xml(
        input_path,
        fps=fps,
        manifest_path=manifest_path,
        file_root=file_root,
        labeler=labeler,
        review_status=review_status,
        default_subject_id=default_subject_id,
        default_scene=default_scene,
        default_view=default_view,
        time_tolerance_sec=time_tolerance_sec,
    )
    return write_converted_fall_labels(
        converted,
        action_output_path=action_output_path,
        event_output_path=event_output_path,
        overwrite=overwrite,
    )


def write_converted_fall_labels(
    converted: ConvertedFallLabels,
    *,
    action_output_path: Path | str,
    event_output_path: Path | str,
    overwrite: bool = False,
) -> dict[str, int]:
    _atomic_write_jsonl_pair(
        converted.action_labels,
        Path(action_output_path),
        converted.event_labels,
        Path(event_output_path),
        overwrite=overwrite,
    )
    return {
        "action_labels": len(converted.action_labels),
        "event_labels": len(converted.event_labels),
    }


def load_video_manifest(path: Path | str) -> dict[str, VideoMetadata]:
    manifest_path = Path(path)
    records = _read_jsonl(manifest_path)
    videos: dict[str, VideoMetadata] = {}
    for index, row in enumerate(records, 1):
        video_id = row.get("video_id")
        if video_id is None:
            continue
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"manifest row {index} has invalid video_id")
        if video_id in videos:
            raise ValueError(f"duplicate manifest video_id: {video_id}")
        videos[video_id] = _video_metadata(row, index)
    return videos


def import_le2i_fall_labels(manifest_path: Path | str) -> Le2iImportResult:
    """Import only official LE2I TXT fall windows referenced by the manifest."""
    videos = load_video_manifest(manifest_path)
    events: list[dict[str, Any]] = []
    report = {
        "manifest_le2i_videos": 0,
        "imported_fall_windows": 0,
        "bbox_only_without_window": 0,
        "explicit_no_fall_window": 0,
        "excluded_unsupervised_subset": 0,
        "missing_annotation_path": 0,
    }
    for metadata in sorted(videos.values(), key=lambda item: item.video_id):
        row = metadata.manifest_record
        if row.get("dataset") != "le2i_imvia":
            continue
        report["manifest_le2i_videos"] += 1
        subset = _sanitize_identifier(str(row.get("subset", "")))
        if subset in {"lecture_room", "office"}:
            report["excluded_unsupervised_subset"] += 1
            continue
        if not metadata.annotation_path:
            report["missing_annotation_path"] += 1
            continue

        annotation_path = Path(metadata.annotation_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        raw = annotation_path.read_bytes()
        annotation_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"LE2I annotation is not UTF-8: {annotation_path}") from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"empty LE2I annotation: {annotation_path}")

        header_indices = [index for index, line in enumerate(lines) if _integer_line(line)]
        if not header_indices:
            _validate_le2i_bbox_rows(lines, annotation_path)
            report["bbox_only_without_window"] += 1
            continue
        if (
            len(header_indices) != 2
            or header_indices[1] != header_indices[0] + 1
        ):
            raise ValueError(f"invalid LE2I fall window header: {annotation_path}")

        source_start = int(lines[header_indices[0]])
        source_end = int(lines[header_indices[1]])
        bbox_lines = [
            line for index, line in enumerate(lines) if index not in set(header_indices)
        ]
        if bbox_lines:
            _validate_le2i_bbox_rows(bbox_lines, annotation_path)
        if source_start == 0 and source_end == 0:
            report["explicit_no_fall_window"] += 1
            continue
        if source_start < 1 or source_end < source_start:
            raise ValueError(f"invalid LE2I one-based fall window: {annotation_path}")
        start_frame = source_start - 1
        end_frame = source_end - 1
        _validate_frame_bounds(metadata.video_id, start_frame, end_frame, metadata)

        source_record_id = (
            f"le2i_txt:{metadata.video_id}:{annotation_sha256}:fall_window"
        )
        label_id = _stable_id(
            "event", "le2i_txt", metadata.video_id, annotation_sha256, source_start, source_end
        )
        events.append(
            {
                "label_id": label_id,
                "source_record_id": source_record_id,
                "source_annotation_path": metadata.annotation_path,
                "source_annotation_sha256": annotation_sha256,
                "asset_id": metadata.asset_id,
                "video_id": metadata.video_id,
                "event_type": "fall",
                "start_time": _frame_to_time(
                    start_frame, metadata.fps_num, metadata.fps_den
                ),
                "end_time": _frame_to_time(end_frame, metadata.fps_num, metadata.fps_den),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_index_base": 0,
                "source_start_frame": source_start,
                "source_end_frame": source_end,
                "source_frame_index_base": 1,
                "severity": 4,
                "label_source": "le2i_txt",
                "review_status": "auto_imported",
                "eligibility": False,
                "review_evidence_ids": [],
                "note": "Official LE2I TXT fall window.",
            }
        )
        report["imported_fall_windows"] += 1

    return Le2iImportResult(
        event_labels=sorted(events, key=_record_sort_key),
        report=report,
    )


def write_le2i_fall_labels(
    manifest_path: Path | str,
    *,
    event_output_path: Path | str,
    report_output_path: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    imported = import_le2i_fall_labels(manifest_path)
    event_path = Path(event_output_path)
    if report_output_path is None:
        _atomic_write_bytes(event_path, _jsonl_bytes(imported.event_labels), overwrite)
    else:
        _atomic_write_pair_bytes(
            event_path,
            _jsonl_bytes(imported.event_labels),
            Path(report_output_path),
            _json_bytes(imported.report),
            overwrite=overwrite,
        )
    return dict(imported.report)


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
        output_path = Path(self._tmpdir.name) / "annotations.xml"
        with zipfile.ZipFile(self.path) as archive:
            xml_names = [
                name for name in archive.namelist() if name.lower().endswith(".xml")
            ]
            if len(xml_names) != 1:
                raise ValueError(
                    f"CVAT ZIP must contain exactly one XML file; found {len(xml_names)}"
                )
            payload = archive.read(xml_names[0])
        output_path.write_bytes(payload)
        return output_path

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def _parse_tasks(root: ET.Element) -> dict[str, CvatTaskInfo]:
    project_tasks = root.findall("./meta/project/tasks/task")
    single_task = root.find("./meta/task")
    task_elements = project_tasks or ([single_task] if single_task is not None else [])
    if not task_elements:
        raise ValueError("CVAT export has no task metadata")

    tasks: dict[str, CvatTaskInfo] = {}
    cumulative_offset = 0
    for index, task in enumerate(task_elements):
        task_id = task.findtext("id") or str(index)
        if task_id in tasks:
            raise ValueError(f"duplicate CVAT task id: {task_id}")
        size = _required_non_negative_int(task.findtext("size"), "task size")
        if size <= 0:
            raise ValueError(f"CVAT task {task_id} has non-positive size")
        start_frame = _required_non_negative_int(
            task.findtext("start_frame") or "0", "task start_frame"
        )
        stop_default = start_frame + size - 1
        stop_frame = _required_non_negative_int(
            task.findtext("stop_frame") or str(stop_default), "task stop_frame"
        )
        if stop_frame < start_frame or stop_frame - start_frame + 1 != size:
            raise ValueError(f"CVAT task {task_id} has inconsistent frame range")
        name = task.findtext("name") or ""
        source = task.findtext("source") or ""
        if not name or not source:
            raise ValueError(f"CVAT task {task_id} is missing name or source")
        tasks[task_id] = CvatTaskInfo(
            task_id=task_id,
            name=name,
            size=size,
            source=source,
            start_frame=start_frame,
            stop_frame=stop_frame,
            frame_offset=cumulative_offset,
        )
        cumulative_offset += size
    return tasks


def _task_for_track(track: ET.Element, tasks: dict[str, CvatTaskInfo]) -> CvatTaskInfo:
    task_id = track.attrib.get("task_id")
    if task_id is not None:
        if task_id not in tasks:
            raise ValueError(
                f"track {track.attrib.get('id')} references unknown task_id {task_id}"
            )
        return tasks[task_id]
    if len(tasks) == 1:
        return next(iter(tasks.values()))
    raise ValueError(f"track {track.attrib.get('id')} is missing task_id")


def _normalized_active_boxes(
    track: ET.Element, task: CvatTaskInfo
) -> list[tuple[int, ET.Element]]:
    boxes = track.findall("box")
    if not boxes:
        return []
    try:
        raw_frames = [int(box.attrib["frame"]) for box in boxes]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"track {track.attrib.get('id')} has invalid box frame") from exc

    local_valid = all(task.start_frame <= frame <= task.stop_frame for frame in raw_frames)
    global_start = task.frame_offset + task.start_frame
    global_stop = task.frame_offset + task.stop_frame
    global_valid = all(global_start <= frame <= global_stop for frame in raw_frames)
    if task.frame_offset == 0 and local_valid:
        global_valid = False
    if local_valid and global_valid:
        raise ValueError(f"track {track.attrib.get('id')} frame coordinates are ambiguous")
    if not local_valid and not global_valid:
        raise ValueError(
            f"track {track.attrib.get('id')} frames do not fit task {task.task_id}"
        )

    normalized = []
    for raw_frame, box in zip(raw_frames, boxes):
        frame = raw_frame - task.frame_offset if global_valid else raw_frame
        if box.attrib.get("outside") != "1":
            normalized.append((frame, box))
    return sorted(normalized, key=lambda pair: pair[0])


def _parse_action_label(label: str) -> tuple[str, str]:
    match = re.fullmatch(r"(?P<action_id>[A-DU]\d{2})[_-](?P<name>[A-Za-z0-9_]+)", label)
    if not match:
        raise ValueError(f"invalid CVAT action label: {label!r}")
    action_id = match.group("action_id")
    action_name = match.group("name")
    expected = ACTION_NAMES.get(action_id)
    if expected is None or expected != action_name:
        raise ValueError(
            f"unknown action label {label!r}; expected a canonical action id/name pair"
        )
    return action_id, action_name


def _track_attributes(active_boxes: Iterable[ET.Element]) -> dict[str, str]:
    values: dict[str, set[str]] = {}
    for box in active_boxes:
        for attr in box.findall("attribute"):
            name = attr.attrib.get("name")
            text = (attr.text or "").strip()
            if name and text:
                values.setdefault(name, set()).add(text)
    conflicts = {name: choices for name, choices in values.items() if len(choices) > 1}
    if conflicts:
        raise ValueError(
            "CVAT track has conflicting attribute values: "
            + ", ".join(sorted(conflicts))
        )
    return {name: next(iter(choices)) for name, choices in values.items()}


def _video_id_from_task_name(task_name: str) -> str:
    marker = "fall_risk__"
    if not task_name.startswith(marker):
        raise ValueError(f"CVAT task name does not follow fall_risk convention: {task_name!r}")
    parts = task_name.split("__")
    if len(parts) < 4:
        raise ValueError(f"invalid fall_risk CVAT task name: {task_name!r}")
    return _canonical_video_id(parts[1], parts[2], parts[-1])


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
    return (Path(file_root) / source).as_posix()


def _frame_to_time(frame: int, fps_num: int, fps_den: int) -> float:
    return round(frame * fps_den / fps_num, 4)


def _bbox(box: ET.Element) -> list[float]:
    try:
        coordinates = [float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")]
    except (KeyError, ValueError) as exc:
        raise ValueError("CVAT box has invalid coordinates") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("CVAT box coordinates must be finite")
    if coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]:
        raise ValueError("CVAT box has reversed coordinates")
    return [round(value, 2) for value in coordinates]


def _event_note(action_id: str, action_name: str) -> str:
    if action_id.startswith("D"):
        return "Derived from a manually annotated fall action."
    return f"Derived from action {action_id}_{action_name}."


def _maybe_int(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _record_sort_key(record: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(record.get("video_id", "")),
        int(record.get("start_frame", 0)),
        int(record.get("end_frame", 0)),
        str(record.get("label_id", "")),
    )


def _video_metadata(row: Mapping[str, Any], index: int) -> VideoMetadata:
    required = (
        "asset_id",
        "video_id",
        "path",
        "fps_num",
        "fps_den",
        "frame_count",
        "duration_sec",
    )
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise ValueError(f"manifest row {index} is missing: {', '.join(missing)}")
    try:
        fps_num = int(row["fps_num"])
        fps_den = int(row["fps_den"])
        frame_count = int(row["frame_count"])
        duration_sec = float(row["duration_sec"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest row {index} has invalid media metadata") from exc
    if fps_num <= 0 or fps_den <= 0 or frame_count <= 0 or duration_sec <= 0:
        raise ValueError(f"manifest row {index} has non-positive media metadata")
    if not math.isfinite(duration_sec):
        raise ValueError(f"manifest row {index} has non-finite duration")
    return VideoMetadata(
        asset_id=str(row["asset_id"]),
        video_id=str(row["video_id"]),
        path=str(row["path"]),
        fps_num=fps_num,
        fps_den=fps_den,
        frame_count=frame_count,
        duration_sec=duration_sec,
        subject_id=str(row.get("subject_id") or "unknown"),
        scene_region=str(row.get("scene_region") or "unknown"),
        view=str(row.get("view") or "unknown"),
        source_group_id=str(row.get("source_group_id") or row["video_id"]),
        annotation_path=(
            str(row["annotation_path"]) if row.get("annotation_path") else None
        ),
        manifest_record=row,
    )


def _validate_task_source(task: CvatTaskInfo, metadata: VideoMetadata) -> None:
    if Path(task.source).name != Path(metadata.path).name:
        raise ValueError(
            f"CVAT source {task.source!r} does not match manifest path for "
            f"{metadata.video_id}"
        )


def _validate_frame_bounds(
    video_id: str, start_frame: int, end_frame: int, metadata: VideoMetadata
) -> None:
    if start_frame < 0 or end_frame < start_frame or end_frame >= metadata.frame_count:
        raise ValueError(
            f"frames {start_frame}-{end_frame} are outside video bounds for "
            f"{video_id} (frame_count={metadata.frame_count})"
        )


def _float_fps_ratio(fps: float) -> tuple[int, int]:
    from fractions import Fraction

    ratio = Fraction(str(fps)).limit_denominator(1_000_000)
    return ratio.numerator, ratio.denominator


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _portable_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _required_non_negative_int(value: str | None, field_name: str) -> int:
    try:
        result = int(value) if value is not None else -1
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}") from exc
    if result < 0:
        raise ValueError(f"invalid {field_name}")
    return result


def _integer_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", value.strip()))


def _validate_le2i_bbox_rows(lines: Iterable[str], path: Path) -> None:
    found = False
    for line in lines:
        if not line.strip():
            continue
        found = True
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"invalid LE2I bbox row in {path}")
        try:
            [int(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"invalid LE2I bbox row in {path}") from exc
    if not found:
        raise ValueError(f"LE2I annotation has no bbox rows: {path}")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    ).encode("utf-8")


def _json_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write_jsonl_pair(
    first_records: Iterable[Mapping[str, Any]],
    first_path: Path,
    second_records: Iterable[Mapping[str, Any]],
    second_path: Path,
    *,
    overwrite: bool,
) -> None:
    _atomic_write_pair_bytes(
        first_path,
        _jsonl_bytes(first_records),
        second_path,
        _jsonl_bytes(second_records),
        overwrite=overwrite,
    )


def _atomic_write_pair_bytes(
    first_path: Path,
    first_payload: bytes,
    second_path: Path,
    second_payload: bytes,
    *,
    overwrite: bool,
) -> None:
    if first_path.resolve() == second_path.resolve():
        raise ValueError("paired outputs must use different paths")
    for path in (first_path, second_path):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    temp_paths = [
        _write_temp_bytes(first_path.parent, first_payload),
        _write_temp_bytes(second_path.parent, second_payload),
    ]
    targets = [first_path, second_path]
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        if overwrite:
            for target in targets:
                if target.exists():
                    backup = _write_temp_bytes(target.parent, target.read_bytes())
                    backups[target] = backup
        for temporary, target in zip(temp_paths, targets):
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
            elif not overwrite:
                temporary = temp_paths[targets.index(target)]
                _unlink_if_same_file(target, temporary)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        for path in [*temp_paths, *backups.values()]:
            path.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temp_bytes(path.parent, payload)
    backup: Path | None = None
    try:
        if overwrite and path.exists():
            backup = _write_temp_bytes(path.parent, path.read_bytes())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        if backup is not None:
            backup.unlink(missing_ok=True)


def _write_temp_bytes(directory: Path, payload: bytes) -> Path:
    with NamedTemporaryFile(prefix=".fall-labels-", dir=directory, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _unlink_if_same_file(path: Path, expected_link: Path) -> None:
    try:
        if os.path.samefile(path, expected_link):
            path.unlink()
    except FileNotFoundError:
        return


def _require_pseudonymous_identifier(
    value: Any, field: str, *, allow_unknown: bool
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty pseudonymous identifier")
    if value.lower() == "unknown" and allow_unknown:
        return
    if (
        not _PSEUDONYM_PATTERN.fullmatch(value)
        or _EMAIL_PATTERN.search(value)
        or _PHONE_PATTERN.search(value)
    ):
        raise ValueError(f"{field} must be a pseudonymous identifier")


def _contains_contact_identifier(value: str) -> bool:
    return bool(_EMAIL_PATTERN.search(value) or _PHONE_PATTERN.search(value))
