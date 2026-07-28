from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET


ACTION_NAMES: dict[str, str] = {
    "A01": "normal_walk",
    "A02": "normal_turn",
    "A03": "controlled_sit_down",
    "A04": "normal_sit_to_stand",
    "A05": "controlled_squat",
    "A06": "controlled_bend",
    "A07": "controlled_lie_down",
    "A08": "routine_support_contact",
    "A09": "kneel_or_floor_activity",
    "A10": "normal_step_adjustment",
    "A11": "assisted_sit_or_lowering",
    "A12": "normal_hop",
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
    "D05": "seated_fall",
    "U01": "unable_to_judge",
}

LEGACY_ACTION_LABEL_ALIASES: dict[str, str] = {
    "A03_normal_sit": "A03_controlled_sit_down",
    "A04_normal_stand": "A04_normal_sit_to_stand",
    "A05_normal_squat": "A05_controlled_squat",
    "A06_normal_bend": "A06_controlled_bend",
    "A09_normal_kneel": "A09_kneel_or_floor_activity",
}

ACTION_EVENT_MAP: dict[str, tuple[str, int]] = {
    "A01": ("normal_activity", 0),
    "A02": ("normal_activity", 0),
    "A03": ("normal_activity", 0),
    "A04": ("normal_activity", 0),
    "A05": ("normal_activity", 0),
    "A06": ("normal_activity", 0),
    "A07": ("normal_activity", 0),
    "A08": ("wall_support", 3),
    "A09": ("normal_activity", 0),
    "A10": ("normal_activity", 0),
    "A11": ("normal_activity", 0),
    "A12": ("normal_activity", 0),
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
    "D05": ("fall", 4),
    "U01": ("uncertain", 0),
}

# Additive action labels keep mapping v2 so existing event IDs stay stable.
ACTION_EVENT_MAPPING_VERSION = "fall-action-event-v2"
QUALITY_VALUES = {
    "clear",
    "partial_occlusion",
    "heavy_occlusion",
    "low_light",
    "off_screen",
    "multi_person_uncertain",
}
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


@dataclass(frozen=True)
class ToagaImportResult:
    action_labels: list[dict[str, Any]]
    report: dict[str, int]


@dataclass(frozen=True)
class NtuRgbdImportResult:
    action_labels: list[dict[str, Any]]
    report: dict[str, Any]


@dataclass(frozen=True)
class PreVFallpRedactedExport:
    filename: str
    payload: bytes
    sha256: str
    task_id: str
    source_name: str
    video_id: str


@dataclass(frozen=True)
class PreVFallpCvatImportResult:
    action_labels: list[dict[str, Any]]
    event_labels: list[dict[str, Any]]
    redacted_exports: list[PreVFallpRedactedExport]
    report: dict[str, Any]


@dataclass(frozen=True)
class _NtuRgbdClipLabelMap:
    mapping_id: str
    mapping_sha256: str
    source_annotation_path: str
    boundary_review: Mapping[str, str]
    entries: Mapping[str, tuple[str, str | None]]


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


def import_pre_vfallp_cvat_labels(
    input_path: Path | str,
    *,
    manifest_path: Path | str,
    redacted_export_dir: Path | str,
    labeler: str = "cvat_pre_vfallp_import_20260722",
) -> PreVFallpCvatImportResult:
    """Prepare a user-authorized Pre_VFallp CVAT export for v2 import.

    The source archive is never copied. Nested task ZIPs and direct multi-task
    project exports are normalized into per-task ZIPs with identity metadata
    removed and manifest-resolvable task metadata, then converted through the
    normal CVAT converter. The caller writes artifacts only after preflight succeeds.
    """
    _require_pseudonymous_identifier(labeler, "labeler", allow_unknown=False)
    source_archive = Path(input_path)
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    source_archive_sha256 = _sha256_file(source_archive)
    destination_dir = Path(redacted_export_dir)
    manifest = load_video_manifest(manifest_path)
    by_source_name: dict[str, list[VideoMetadata]] = {}
    for metadata in manifest.values():
        row = metadata.manifest_record
        if row.get("dataset") != "pre_vfallp":
            continue
        by_source_name.setdefault(Path(metadata.path).name, []).append(metadata)

    redacted_exports: list[PreVFallpRedactedExport] = []
    authorization_ids: set[str] = set()
    authorized_counts: set[int] = set()
    identity_tags_removed: Counter[str] = Counter()
    ignored_tracks: list[dict[str, str]] = []
    seen_video_ids: set[str] = set()
    seen_export_filenames: set[str] = set()
    try:
        with zipfile.ZipFile(source_archive) as outer:
            invalid_member = outer.testzip()
            if invalid_member is not None:
                raise ValueError(f"invalid CVAT archive member: {invalid_member}")
            task_archives = _pre_vfallp_task_archives(outer)
            for filename, payload in task_archives:
                if filename in seen_export_filenames:
                    raise ValueError(
                        f"Pre_VFallp export has duplicate task ZIP basename: {filename}"
                    )
                seen_export_filenames.add(filename)
                (
                    redacted_export,
                    removed_tags,
                    authorization_id,
                    authorized_count,
                    ignored_for_export,
                ) = (
                    _prepare_pre_vfallp_cvat_task_export(
                        payload,
                        filename=filename,
                        manifest_by_source_name=by_source_name,
                        source_archive_name=source_archive.name,
                        source_archive_sha256=source_archive_sha256,
                    )
                )
                if redacted_export.video_id in seen_video_ids:
                    raise ValueError(
                        "nested CVAT export resolves multiple tasks to video_id: "
                        + redacted_export.video_id
                    )
                seen_video_ids.add(redacted_export.video_id)
                redacted_exports.append(redacted_export)
                identity_tags_removed.update(removed_tags)
                ignored_tracks.extend(ignored_for_export)
                authorization_ids.add(authorization_id)
                authorized_counts.add(authorized_count)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid Pre_VFallp CVAT ZIP: {source_archive}") from exc

    if len(authorization_ids) != 1 or len(authorized_counts) != 1:
        raise ValueError("Pre_VFallp export spans inconsistent internal authorizations")
    authorization_id = next(iter(authorization_ids))
    authorized_count = next(iter(authorized_counts))
    if len(redacted_exports) != authorized_count:
        raise ValueError(
            "Pre_VFallp export task count does not match internal authorization: "
            f"tasks={len(redacted_exports)} authorized_video_count={authorized_count}"
        )

    action_labels: list[dict[str, Any]] = []
    event_labels: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="pre_vfallp_cvat_import_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for export in redacted_exports:
            temporary_export = temporary_root / export.filename
            temporary_export.write_bytes(export.payload)
            converted = convert_cvat_xml(
                temporary_export,
                manifest_path=manifest_path,
                fps=None,
                labeler=labeler,
            )
            if converted.identity_metadata_present:
                raise ValueError("redacted Pre_VFallp CVAT export still has identity metadata")
            final_source_path = _output_source_annotation_path(
                destination_dir / export.filename
            )
            action_labels.extend(
                _with_source_annotation_path(converted.action_labels, final_source_path)
            )
            event_labels.extend(
                _with_source_annotation_path(converted.event_labels, final_source_path)
            )

    redacted_exports.sort(key=lambda item: item.filename)
    action_labels.sort(key=_record_sort_key)
    event_labels.sort(key=_record_sort_key)
    report: dict[str, Any] = {
        "schema_version": "pre-vfallp-cvat-import-v1",
        "authorization_id": authorization_id,
        "source_archive": {
            "filename": source_archive.name,
            "sha256": source_archive_sha256,
            "copied_into_repository": False,
        },
        "redacted_task_exports": [
            {
                "path": _output_source_annotation_path(destination_dir / export.filename),
                "sha256": export.sha256,
                "task_id": export.task_id,
                "source_name": export.source_name,
                "video_id": export.video_id,
            }
            for export in redacted_exports
        ],
        "identity_metadata_tags_removed": dict(sorted(identity_tags_removed.items())),
        "ignored_tracks": sorted(
            ignored_tracks,
            key=lambda row: (
                row["filename"],
                row["task_id"],
                row["track_id"],
            ),
        ),
        "imported_action_labels": len(action_labels),
        "imported_event_labels": len(event_labels),
        "action_id_counts": dict(
            sorted(Counter(row["action_id"] for row in action_labels).items())
        ),
        "event_type_counts": dict(
            sorted(Counter(row["event_type"] for row in event_labels).items())
        ),
        "source_group_ids": sorted(
            {
                str(manifest[export.video_id].source_group_id)
                for export in redacted_exports
            }
        ),
        "provenance_status": "internal_authorized_source_unverified",
    }
    return PreVFallpCvatImportResult(
        action_labels=action_labels,
        event_labels=event_labels,
        redacted_exports=redacted_exports,
        report=report,
    )


def write_pre_vfallp_cvat_labels(
    imported: PreVFallpCvatImportResult,
    *,
    redacted_export_dir: Path | str,
    action_output_path: Path | str,
    event_output_path: Path | str,
    report_output_path: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist redacted task ZIPs and their paired candidate JSONL atomically per type."""
    redacted_dir = Path(redacted_export_dir)
    action_path = Path(action_output_path)
    event_path = Path(event_output_path)
    report_path = Path(report_output_path)
    output_paths = (action_path, event_path, report_path)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("action, event, and import report outputs must be different paths")
    if not overwrite:
        existing = [path for path in (*output_paths, redacted_dir) if path.exists()]
        if existing:
            raise FileExistsError(existing[0])

    _write_redacted_pre_vfallp_exports(
        imported.redacted_exports, redacted_dir, overwrite=overwrite
    )
    write_converted_fall_labels(
        ConvertedFallLabels(
            action_labels=imported.action_labels,
            event_labels=imported.event_labels,
        ),
        action_output_path=action_path,
        event_output_path=event_path,
        overwrite=overwrite,
    )
    report = dict(imported.report)
    report["outputs"] = {
        "redacted_export_dir": _output_source_annotation_path(redacted_dir),
        "action_labels": _output_source_annotation_path(action_path),
        "event_labels": _output_source_annotation_path(event_path),
    }
    _atomic_write_bytes(report_path, _json_bytes(report), overwrite=overwrite)
    return report


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


def import_toaga_normal_walk_labels(manifest_path: Path | str) -> ToagaImportResult:
    """Import eligible TOAGA walking videos as source-derived A01 candidates."""
    videos = load_video_manifest(manifest_path)
    actions: list[dict[str, Any]] = []
    report = {
        "manifest_toaga_walking_videos": 0,
        "imported_normal_walk_labels": 0,
        "excluded_ineligible": 0,
        "missing_annotation_path": 0,
    }
    for metadata in sorted(videos.values(), key=lambda item: item.video_id):
        row = metadata.manifest_record
        if (
            row.get("dataset") != "toaga"
            or row.get("media_type") != "video"
            or row.get("subset") != "walking"
        ):
            continue
        report["manifest_toaga_walking_videos"] += 1
        if row.get("eligibility") is not True:
            report["excluded_ineligible"] += 1
            continue
        if not metadata.annotation_path:
            report["missing_annotation_path"] += 1
            raise ValueError(
                f"TOAGA walking video {metadata.video_id} is missing its source table"
            )
        annotation_path = Path(metadata.annotation_path)
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        annotation_sha256 = _sha256_file(annotation_path)
        source_export_id = f"toaga_official_walking_{annotation_sha256[:24]}"
        start_frame = 0
        end_frame = metadata.frame_count - 1
        source_record_id = (
            f"{source_export_id}:video:{metadata.video_id}:full_video"
        )
        action_label_id = _stable_id(
            "action",
            "toaga_official_walking",
            metadata.asset_id,
            metadata.video_id,
            str(row.get("sha256") or ""),
            annotation_sha256,
            start_frame,
            end_frame,
        )
        actions.append(
            {
                "label_id": action_label_id,
                "source_record_id": source_record_id,
                "source_annotation_path": metadata.annotation_path,
                "source_annotation_sha256": annotation_sha256,
                "source_export_id": source_export_id,
                "asset_id": metadata.asset_id,
                "video_id": metadata.video_id,
                "file_path": metadata.path,
                "subject_id": metadata.subject_id,
                "scene": metadata.scene_region,
                "view": metadata.view,
                "action_id": "A01",
                "action_name": ACTION_NAMES["A01"],
                "event_type": ACTION_EVENT_MAP["A01"][0],
                "start_time": _frame_to_time(
                    start_frame, metadata.fps_num, metadata.fps_den
                ),
                "end_time": _frame_to_time(
                    end_frame, metadata.fps_num, metadata.fps_den
                ),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_index_base": 0,
                "labeler": "official_toaga_source",
                "quality": "clear",
                "note": (
                    "Source-derived TOAGA full-video normal-walk label; "
                    "no frame-level CVAT review."
                ),
                "source": "toaga_official_walking",
            }
        )
        report["imported_normal_walk_labels"] += 1
    return ToagaImportResult(
        action_labels=sorted(actions, key=_record_sort_key), report=report
    )


def write_toaga_normal_walk_labels(
    manifest_path: Path | str,
    *,
    action_output_path: Path | str,
    report_output_path: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    imported = import_toaga_normal_walk_labels(manifest_path)
    action_path = Path(action_output_path)
    if report_output_path is None:
        _atomic_write_bytes(action_path, _jsonl_bytes(imported.action_labels), overwrite)
    else:
        _atomic_write_pair_bytes(
            action_path,
            _jsonl_bytes(imported.action_labels),
            Path(report_output_path),
            _json_bytes(imported.report),
            overwrite=overwrite,
        )
    return dict(imported.report)


def import_ntu_rgbd_clip_labels(
    manifest_path: Path | str,
    label_map_path: Path | str,
) -> NtuRgbdImportResult:
    """Import human-reviewed NTU RGB+D full-clip action labels.

    This intentionally emits action labels only. A reviewed C03 action clip does
    not independently establish a near-fall event label.
    """
    videos = load_video_manifest(manifest_path)
    label_map = _load_ntu_rgbd_clip_label_map(label_map_path)
    source_export_id = f"ntu_rgbd_clip_label_{label_map.mapping_sha256[:24]}"
    actions: list[dict[str, Any]] = []
    imported_by_code: dict[str, int] = {}
    excluded_by_code: dict[str, int] = {}
    report: dict[str, Any] = {
        "schema_version": "ntu-rgbd-clip-import-report-v2",
        "mapping_id": label_map.mapping_id,
        "mapping_path": label_map.source_annotation_path,
        "mapping_sha256": label_map.mapping_sha256,
        "boundary_review": dict(label_map.boundary_review),
        "manifest_ntu_rgbd_videos": 0,
        "imported_action_labels": 0,
        "manual_exact_action_labels": 0,
        "excluded_videos": 0,
        "excluded_ineligible": 0,
        "imported_by_source_action_code": imported_by_code,
        "excluded_by_source_action_code": excluded_by_code,
    }
    for metadata in sorted(videos.values(), key=lambda item: item.video_id):
        row = metadata.manifest_record
        if row.get("dataset") != "ntu_rgbd" or row.get("media_type") != "video":
            continue
        report["manifest_ntu_rgbd_videos"] += 1
        source_action_code = str(row.get("source_action_code") or "").upper()
        mapping = label_map.entries.get(source_action_code)
        if mapping is None:
            raise ValueError(
                f"NTU RGB+D video {metadata.video_id} has unmapped source action "
                f"code {source_action_code!r}"
            )
        mode, action_id = mapping
        if mode == "excluded":
            report["excluded_videos"] += 1
            excluded_by_code[source_action_code] = (
                excluded_by_code.get(source_action_code, 0) + 1
            )
            continue
        if row.get("eligibility") is not True:
            report["excluded_ineligible"] += 1
            continue
        if action_id is None:
            raise ValueError(
                f"NTU RGB+D manual mapping for {source_action_code} has no action_id"
            )
        start_frame = 0
        end_frame = metadata.frame_count - 1
        source_record_id = (
            f"{source_export_id}:video:{metadata.video_id}:full_video"
        )
        action_label_id = _stable_id(
            "action",
            "ntu_rgbd_manual_clip_label",
            metadata.asset_id,
            metadata.video_id,
            str(row.get("sha256") or ""),
            label_map.mapping_sha256,
            source_action_code,
            action_id,
            start_frame,
            end_frame,
        )
        actions.append(
            {
                "label_id": action_label_id,
                "source_record_id": source_record_id,
                "source_annotation_path": label_map.source_annotation_path,
                "source_annotation_sha256": label_map.mapping_sha256,
                "source_export_id": source_export_id,
                "asset_id": metadata.asset_id,
                "video_id": metadata.video_id,
                "file_path": metadata.path,
                "subject_id": metadata.subject_id,
                "scene": metadata.scene_region,
                "view": metadata.view,
                "action_id": action_id,
                "action_name": ACTION_NAMES[action_id],
                "event_type": ACTION_EVENT_MAP[action_id][0],
                "start_time": _frame_to_time(
                    start_frame, metadata.fps_num, metadata.fps_den
                ),
                "end_time": _frame_to_time(
                    end_frame, metadata.fps_num, metadata.fps_den
                ),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_index_base": 0,
                "labeler": "project_owner_ntu_rgbd_boundary_review_20260725",
                "quality": "clear",
                "note": (
                    "Project-owner human-reviewed exact full-clip boundary for "
                    f"NTU RGB+D {source_action_code}; decision "
                    f"{label_map.boundary_review['decision_id']}."
                ),
                "source": "ntu_rgbd_manual_clip_label",
            }
        )
        report["imported_action_labels"] += 1
        report["manual_exact_action_labels"] += 1
        imported_by_code[source_action_code] = (
            imported_by_code.get(source_action_code, 0) + 1
        )
    report["imported_by_source_action_code"] = dict(sorted(imported_by_code.items()))
    report["excluded_by_source_action_code"] = dict(
        sorted(excluded_by_code.items())
    )
    return NtuRgbdImportResult(
        action_labels=sorted(actions, key=_record_sort_key), report=report
    )


def write_ntu_rgbd_clip_labels(
    manifest_path: Path | str,
    label_map_path: Path | str,
    *,
    action_output_path: Path | str,
    report_output_path: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    imported = import_ntu_rgbd_clip_labels(manifest_path, label_map_path)
    action_path = Path(action_output_path)
    if report_output_path is None:
        _atomic_write_bytes(action_path, _jsonl_bytes(imported.action_labels), overwrite)
    else:
        _atomic_write_pair_bytes(
            action_path,
            _jsonl_bytes(imported.action_labels),
            Path(report_output_path),
            _json_bytes(imported.report),
            overwrite=overwrite,
        )
    return dict(imported.report)


def _load_ntu_rgbd_clip_label_map(
    label_map_path: Path | str,
) -> _NtuRgbdClipLabelMap:
    path = Path(label_map_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid NTU RGB+D label map: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("NTU RGB+D label map must be a JSON object")
    expected_top_level = {
        "schema_version",
        "mapping_id",
        "boundary_review",
        "mappings",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "NTU RGB+D v2 label map has an invalid top-level shape"
        )
    if payload.get("schema_version") != "ntu-rgbd-clip-label-map-v2":
        raise ValueError("unsupported NTU RGB+D label map schema_version")
    mapping_id = payload.get("mapping_id")
    if not isinstance(mapping_id, str) or not mapping_id.strip():
        raise ValueError("NTU RGB+D label map mapping_id must be a non-empty string")
    boundary_review = payload.get("boundary_review")
    expected_review = {
        "decision_id",
        "reviewed_at",
        "reviewer_id",
        "boundary_precision",
        "training_tier",
    }
    if not isinstance(boundary_review, dict) or set(boundary_review) != expected_review:
        raise ValueError("NTU RGB+D label map boundary_review has an invalid shape")
    if any(
        not isinstance(boundary_review.get(field), str)
        or not boundary_review[field].strip()
        for field in ("decision_id", "reviewer_id")
    ):
        raise ValueError("NTU RGB+D boundary review identifiers must be non-empty")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", boundary_review["reviewed_at"]) is None:
        raise ValueError("NTU RGB+D boundary review date must use YYYY-MM-DD")
    if boundary_review["boundary_precision"] != "exact":
        raise ValueError("NTU RGB+D boundary review must declare exact precision")
    if boundary_review["training_tier"] != "primary":
        raise ValueError("NTU RGB+D boundary review must declare primary tier")
    raw_entries = payload.get("mappings")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("NTU RGB+D label map mappings must be a non-empty list")

    entries: dict[str, tuple[str, str | None]] = {}
    for position, entry in enumerate(raw_entries, start=1):
        if not isinstance(entry, dict) or set(entry) != {
            "source_action_code",
            "mode",
            "action_id",
        }:
            raise ValueError(
                f"NTU RGB+D label map entry {position} has an invalid shape"
            )
        source_action_code = entry.get("source_action_code")
        mode = entry.get("mode")
        action_id = entry.get("action_id")
        if not isinstance(source_action_code, str) or re.fullmatch(
            r"A\d{3}", source_action_code
        ) is None:
            raise ValueError(
                f"NTU RGB+D label map entry {position} has an invalid source_action_code"
            )
        if source_action_code in entries:
            raise ValueError(
                f"duplicate NTU RGB+D source_action_code: {source_action_code}"
            )
        if mode == "manual_exact":
            if not isinstance(action_id, str) or action_id not in ACTION_NAMES:
                raise ValueError(
                    f"NTU RGB+D manual mapping {source_action_code} has invalid action_id"
                )
        elif mode == "excluded":
            if action_id is not None:
                raise ValueError(
                    f"NTU RGB+D excluded mapping {source_action_code} must use null action_id"
                )
        else:
            raise ValueError(
                f"NTU RGB+D label map entry {position} has invalid mode {mode!r}"
            )
        entries[source_action_code] = (mode, action_id)

    return _NtuRgbdClipLabelMap(
        mapping_id=mapping_id,
        mapping_sha256=hashlib.sha256(raw).hexdigest(),
        source_annotation_path=_portable_source_path(path),
        boundary_review={key: str(value) for key, value in boundary_review.items()},
        entries=entries,
    )


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


def _prepare_pre_vfallp_cvat_task_export(
    payload: bytes,
    *,
    filename: str,
    manifest_by_source_name: Mapping[str, list[VideoMetadata]],
    source_archive_name: str,
    source_archive_sha256: str,
) -> tuple[
    PreVFallpRedactedExport,
    Counter[str],
    str,
    int,
    list[dict[str, str]],
]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            invalid_member = archive.testzip()
            if invalid_member is not None:
                raise ValueError(
                    f"invalid nested CVAT task ZIP member {invalid_member!r} in {filename}"
                )
            xml_names = [
                name for name in archive.namelist() if name.lower().endswith(".xml")
            ]
            if len(xml_names) != 1:
                raise ValueError(
                    f"nested CVAT task ZIP {filename} must contain exactly one XML file"
                )
            root = ET.fromstring(archive.read(xml_names[0]))
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"invalid nested CVAT task ZIP: {filename}") from exc

    task_elements = _cvat_task_elements(root)
    if len(task_elements) != 1:
        raise ValueError(
            f"nested Pre_VFallp CVAT task ZIP {filename} must contain exactly one task"
        )
    tasks = _parse_tasks(root)
    if len(tasks) != 1:
        raise ValueError(
            f"nested Pre_VFallp CVAT task ZIP {filename} has ambiguous task metadata"
        )
    task = next(iter(tasks.values()))
    source_name = Path(task.source).name
    candidates_by_video_id: dict[str, VideoMetadata] = {}
    for candidate_name in _pre_vfallp_source_name_candidates(task):
        for candidate in manifest_by_source_name.get(candidate_name, []):
            candidates_by_video_id[candidate.video_id] = candidate
    candidates = list(candidates_by_video_id.values())
    if len(candidates) != 1:
        raise ValueError(
            f"CVAT source {task.source!r} in {filename} maps to {len(candidates)} "
            "Pre_VFallp manifest records"
        )
    metadata = candidates[0]
    if task.size != metadata.frame_count:
        raise ValueError(
            f"CVAT task {task.task_id} frame count {task.size} does not match "
            f"manifest frame_count {metadata.frame_count} for {metadata.video_id}"
        )
    authorization_id, authorized_count = _pre_vfallp_internal_authorization(
        metadata,
        source_archive_name=source_archive_name,
        source_archive_sha256=source_archive_sha256,
    )
    name_element = task_elements[0].find("name")
    if name_element is None:
        raise ValueError(f"nested CVAT task {filename} is missing a name element")
    source_element = task_elements[0].find("source")
    if source_element is None:
        raise ValueError(f"nested CVAT task {filename} is missing a source element")
    subset = _sanitize_identifier(str(metadata.manifest_record.get("subset", "")))
    if not subset:
        raise ValueError(f"Pre_VFallp manifest row {metadata.video_id} has no subset")
    name_element.text = f"fall_risk__pre_vfallp__{subset}__{metadata.video_id}"
    source_element.text = Path(metadata.path).name
    ignored_tracks = _remove_pre_vfallp_outside_only_tracks(
        root,
        task=task,
        filename=filename,
    )
    removed_tags = _remove_cvat_identity_metadata(root)
    redacted_payload = _deterministic_cvat_zip(root)
    return (
        PreVFallpRedactedExport(
            filename=filename,
            payload=redacted_payload,
            sha256=hashlib.sha256(redacted_payload).hexdigest(),
            task_id=task.task_id,
            source_name=source_name,
            video_id=metadata.video_id,
        ),
        removed_tags,
        authorization_id,
        authorized_count,
        ignored_tracks,
    )


def _remove_pre_vfallp_outside_only_tracks(
    root: ET.Element,
    *,
    task: CvatTaskInfo,
    filename: str,
) -> list[dict[str, str]]:
    ignored: list[dict[str, str]] = []
    for track in list(root.findall("track")):
        boxes = track.findall("box")
        if not boxes or any(box.attrib.get("outside") != "1" for box in boxes):
            continue
        root.remove(track)
        ignored.append(
            {
                "filename": filename,
                "task_id": task.task_id,
                "track_id": str(track.attrib.get("id") or "unknown"),
                "label": str(track.attrib.get("label") or "unknown"),
                "source_name": Path(task.source).name,
                "reason": "outside_only",
            }
        )
    return ignored


def _pre_vfallp_task_archives(
    archive: zipfile.ZipFile,
) -> list[tuple[str, bytes]]:
    inner_names = sorted(
        name for name in archive.namelist() if name.lower().endswith(".zip")
    )
    if inner_names:
        return [(Path(name).name, archive.read(name)) for name in inner_names]

    xml_names = sorted(
        name for name in archive.namelist() if name.lower().endswith(".xml")
    )
    if len(xml_names) != 1:
        raise ValueError(
            "Pre_VFallp export must contain task ZIPs or exactly one XML file; "
            f"found {len(xml_names)} XML files"
        )
    try:
        root = ET.fromstring(archive.read(xml_names[0]))
    except ET.ParseError as exc:
        raise ValueError("invalid direct Pre_VFallp CVAT XML export") from exc
    return _split_pre_vfallp_project_export(root)


def _split_pre_vfallp_project_export(
    root: ET.Element,
) -> list[tuple[str, bytes]]:
    task_elements = _cvat_task_elements(root)
    tasks = list(_parse_tasks(root).values())
    if len(task_elements) != len(tasks):
        raise ValueError("Pre_VFallp project task metadata is inconsistent")

    for track in root.findall("track"):
        task_id = track.attrib.get("task_id")
        if len(tasks) > 1 and task_id is None:
            raise ValueError(
                f"track {track.attrib.get('id')} is missing task_id in project export"
            )
        if task_id is not None and task_id not in {task.task_id for task in tasks}:
            raise ValueError(
                f"track {track.attrib.get('id')} references unknown task_id {task_id}"
            )

    exports: list[tuple[str, bytes]] = []
    for index, task in enumerate(tasks, 1):
        task_root = copy.deepcopy(root)
        task_container = task_root.find("./meta/project/tasks")
        if task_container is not None:
            copied_tasks = list(task_container.findall("task"))
            for copied_index, copied_task in enumerate(copied_tasks):
                if copied_index != index - 1:
                    task_container.remove(copied_task)

        for track in list(task_root.findall("track")):
            track_task_id = track.attrib.get("task_id")
            if track_task_id is not None and track_task_id != task.task_id:
                task_root.remove(track)
                continue
            _normalize_project_track_frames(track, task)

        source_stem = Path(_normalized_pre_vfallp_source_name(task.source)).stem
        safe_source_stem = _sanitize_identifier(source_stem) or "video"
        safe_task_id = _sanitize_identifier(task.task_id) or str(index)
        filename = (
            f"{index:03d}__task_{safe_task_id}__{safe_source_stem}"
            "__CVAT_for_video_1_1.zip"
        )
        exports.append((filename, _deterministic_cvat_zip(task_root)))
    return exports


def _normalize_project_track_frames(
    track: ET.Element,
    task: CvatTaskInfo,
) -> None:
    for element in track.iter():
        raw_value = element.attrib.get("frame")
        if raw_value is None:
            continue
        try:
            normalized_frame = int(raw_value) - task.frame_offset
        except ValueError as exc:
            raise ValueError(
                f"track {track.attrib.get('id')} has invalid frame {raw_value!r}"
            ) from exc
        if not task.start_frame <= normalized_frame <= task.stop_frame:
            raise ValueError(
                f"track {track.attrib.get('id')} frames do not fit task {task.task_id}"
            )
        element.attrib["frame"] = str(normalized_frame)


def _pre_vfallp_source_name_candidates(task: CvatTaskInfo) -> list[str]:
    candidates = [Path(task.source).name]
    normalized_source = _normalized_pre_vfallp_source_name(task.source)
    if normalized_source not in candidates:
        candidates.append(normalized_source)
    for embedded_name in re.findall(r"\{\{([^{}]+)\}\}", task.name):
        basename = Path(embedded_name).name
        if basename not in candidates:
            candidates.append(basename)
    task_name = Path(task.name).name
    if Path(task_name).suffix and task_name not in candidates:
        candidates.append(task_name)
    return candidates


def _normalized_pre_vfallp_source_name(source: str) -> str:
    path = Path(source)
    stem = path.stem
    if stem.lower().endswith("_resized"):
        stem = stem[: -len("_resized")]
    return f"{stem}{path.suffix}"


def _cvat_task_elements(root: ET.Element) -> list[ET.Element]:
    project_tasks = root.findall("./meta/project/tasks/task")
    single_task = root.find("./meta/task")
    if project_tasks:
        return project_tasks
    return [single_task] if single_task is not None else []


def _pre_vfallp_internal_authorization(
    metadata: VideoMetadata,
    *,
    source_archive_name: str,
    source_archive_sha256: str,
) -> tuple[str, int]:
    row = metadata.manifest_record
    if row.get("dataset") != "pre_vfallp" or row.get("eligibility") is not True:
        raise ValueError(
            f"Pre_VFallp video is not eligible for import: {metadata.video_id}"
        )
    authorization = row.get("internal_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError(
            f"Pre_VFallp video lacks an internal authorization: {metadata.video_id}"
        )
    authorization_id = authorization.get("authorization_id")
    authorized_count = authorization.get("authorized_video_count")
    if (
        not isinstance(authorization_id, str)
        or not authorization_id
        or not isinstance(authorized_count, int)
        or isinstance(authorized_count, bool)
        or authorized_count <= 0
    ):
        raise ValueError(
            f"Pre_VFallp video has an invalid internal authorization: {metadata.video_id}"
        )
    if row.get("source_uri") != f"internal://authorization/{authorization_id}":
        raise ValueError(
            f"Pre_VFallp video has inconsistent authorization URI: {metadata.video_id}"
        )
    evidence = authorization.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("kind") != "cvat_export_archive":
        raise ValueError(
            "Pre_VFallp CVAT import requires CVAT archive authorization evidence"
        )
    if evidence.get("name") != source_archive_name:
        raise ValueError("Pre_VFallp source archive filename does not match authorization")
    if evidence.get("sha256") != source_archive_sha256:
        raise ValueError("Pre_VFallp source archive checksum does not match authorization")
    return authorization_id, authorized_count


def _remove_cvat_identity_metadata(root: ET.Element) -> Counter[str]:
    removed: Counter[str] = Counter()

    def remove_from(parent: ET.Element) -> None:
        for child in list(parent):
            tag = _xml_local_name(child.tag)
            if tag in IDENTITY_METADATA_TAGS:
                for nested in child.iter():
                    nested_tag = _xml_local_name(nested.tag)
                    if nested_tag in IDENTITY_METADATA_TAGS:
                        removed[nested_tag] += 1
                parent.remove(child)
            else:
                remove_from(child)

    remove_from(root)
    return removed


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _deterministic_cvat_zip(root: ET.Element) -> bytes:
    xml_payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(
            info,
            xml_payload,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    return buffer.getvalue()


def _with_source_annotation_path(
    records: Iterable[Mapping[str, Any]], source_annotation_path: str
) -> list[dict[str, Any]]:
    return [
        {**record, "source_annotation_path": source_annotation_path}
        for record in records
    ]


def _output_source_annotation_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_redacted_pre_vfallp_exports(
    exports: Iterable[PreVFallpRedactedExport],
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"redacted export output is not a directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    backup_dir: Path | None = None
    committed = False
    try:
        filenames: set[str] = set()
        for export in exports:
            if Path(export.filename).name != export.filename:
                raise ValueError(f"invalid redacted export filename: {export.filename}")
            if export.filename in filenames:
                raise ValueError(f"duplicate redacted export filename: {export.filename}")
            filenames.add(export.filename)
            (staging_dir / export.filename).write_bytes(export.payload)
        if output_dir.exists():
            backup_dir = output_dir.parent / (
                f".{output_dir.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(output_dir, backup_dir)
        os.replace(staging_dir, output_dir)
        committed = True
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(backup_dir, output_dir)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if committed and backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)


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
    label = LEGACY_ACTION_LABEL_ALIASES.get(label, label)
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
