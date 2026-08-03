from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.annotations import (
    convert_cvat_xml,
    load_video_manifest,
    write_converted_fall_labels,
)


FALL_DETECTION_2017_BATCH_ID = "fall_detection_2017_manual"
_ANOMALOUS_SIZES = {
    "20240921140203.mp4": 57,
    "20240921140126.mp4": 57,
    "20240921135516.mp4": 58,
    "20240921144221.mp4": 58,
}
@dataclass(frozen=True)
class _Task:
    source_name: str
    task_id: str
    size: int
    width: int
    height: int
    frame_offset: int
    tracks: tuple[ET.Element, ...]
    tracks_are_local: bool = False


def import_fall_detection_2017_cvat_labels(
    base_archive: Path | str,
    *,
    revision_archive: Path | str | None,
    manifest_path: Path | str,
    output_dir: Path | str,
    labeler: str = "fall_detection_2017_manual_labeler",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge the two CVAT exports and create an isolated v2 JSONL batch.

    The exports are project-global CVAT tracks. The revision archive contains a
    replacement project for ADL_20240921, an image-style correction for task
    552, and a job-style correction for task 832. All three are normalized into
    one canonical project before using the regular CVAT-to-v2 converter.
    """
    base_path = Path(base_archive)
    revision_path = Path(revision_archive) if revision_archive else None
    manifest_file = Path(manifest_path)
    destination = Path(output_dir)
    for path in (base_path, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if revision_path is not None and not revision_path.is_file():
        raise FileNotFoundError(revision_path)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", labeler):
        raise ValueError("labeler must be a pseudonymous identifier")

    manifest = load_video_manifest(manifest_file)
    by_source, duplicate_sources = _manifest_by_source(manifest.values())
    base_entries = _expand_archive(base_path)
    revision_entries = _expand_archive(revision_path) if revision_path else []
    base_projects = {
        Path(name).name: _read_project(payload, Path(name).name)
        for name, payload in base_entries
    }
    revision_names = {Path(name).name: payload for name, payload in revision_entries}
    for required in ("ADL_20240921.zip", "ADL_20240922.zip", "Fall_20240919.zip"):
        if revision_path is not None and required not in revision_names:
            raise ValueError(f"revision archive is missing {required}")

    if "ADL_20240921.zip" in revision_names:
        base_projects["ADL_20240921.zip"] = _read_project(
            revision_names["ADL_20240921.zip"], "ADL_20240921.zip"
        )

    adl22_image_tracks: dict[str, tuple[ET.Element, ...]] = {}
    if "ADL_20240922.zip" in revision_names:
        adl22_image_tracks = _read_image_revision(
            revision_names["ADL_20240922.zip"], source_name="ADL_20240922.zip"
        )
    fall19_job_tracks: tuple[ET.Element, ...] = ()
    if "Fall_20240919.zip" in revision_names:
        fall19_job_tracks = _read_job_revision(
            revision_names["Fall_20240919.zip"], source_name="Fall_20240919.zip"
        )

    tasks: list[_Task] = []
    for archive_name in sorted(base_projects):
        project_tasks = base_projects[archive_name]
        for task in project_tasks:
            if archive_name == "ADL_20240922.zip" and task.source_name == "20240922115152.mp4":
                replacement = adl22_image_tracks.get(task.source_name)
                if replacement is None:
                    raise ValueError("ADL_20240922 revision has no task 552 image annotations")
                task = _Task(
                    **{**task.__dict__, "tracks": replacement, "tracks_are_local": True}
                )
            if archive_name == "Fall_20240919.zip" and task.source_name == "20240919161414.mp4":
                if not fall19_job_tracks:
                    raise ValueError("Fall_20240919 revision has no replacement track")
                task = _Task(
                    **{**task.__dict__, "tracks": fall19_job_tracks, "tracks_are_local": True}
                )
            tasks.append(task)

    by_source_task: dict[str, _Task] = {}
    duplicate_tasks: list[str] = []
    for task in tasks:
        if task.source_name in by_source_task:
            duplicate_tasks.append(task.source_name)
        else:
            by_source_task[task.source_name] = task
    if duplicate_tasks:
        raise ValueError(f"duplicate CVAT task sources: {sorted(set(duplicate_tasks))}")

    expected_sources = set(by_source)
    manifest_sources = expected_sources | {
        str(item["source_name"]) for item in duplicate_sources
    }
    observed_sources = set(by_source_task)
    missing_sources = sorted(expected_sources - observed_sources)
    extra_sources = sorted(observed_sources - manifest_sources)
    if extra_sources:
        raise ValueError(f"CVAT sources are absent from manifest: {extra_sources[:5]}")

    normalized_root, normalization = _normalize_tasks(
        by_source_task.values(), by_source
    )
    source_payload = _deterministic_zip(normalized_root)
    destination.mkdir(parents=True, exist_ok=True)
    source_output = destination / "source_annotations.zip"
    action_output = destination / "action_labels.jsonl"
    event_output = destination / "event_labels.jsonl"
    report_output = destination / "import_report.json"
    output_paths = (source_output, action_output, event_output, report_output)
    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    _atomic_write_bytes(source_output, source_payload)

    converted = convert_cvat_xml(
        source_output,
        manifest_path=manifest_file,
        fps=None,
        labeler=labeler,
    )
    if converted.identity_metadata_present:
        raise ValueError("normalized CVAT export still contains identity metadata")
    counts = write_converted_fall_labels(
        converted,
        action_output_path=action_output,
        event_output_path=event_output,
        overwrite=overwrite,
    )

    source_archives = [
        {
            "filename": Path(name).name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in [*base_entries, *revision_entries]
    ]
    report: dict[str, Any] = {
        "schema_version": "fall-detection-2017-cvat-import-v1",
        "batch_id": FALL_DETECTION_2017_BATCH_ID,
        "dataset": "fall_detection_2017",
        "source_archives": sorted(source_archives, key=lambda row: row["filename"]),
        "raw_task_count": len(tasks),
        "raw_track_count": sum(len(task.tracks) for task in tasks),
        "manifest_eligible_source_count": len(expected_sources),
        "annotated_source_count": len(observed_sources),
        "imported_source_count": normalization["normalized_task_count"],
        "missing_eligible_sources": missing_sources,
        "excluded_sources": duplicate_sources,
        "revision_overlay": {
            "replaced_project": "ADL_20240921.zip" if "ADL_20240921.zip" in revision_names else None,
            "image_task_source": "20240922115152.mp4" if adl22_image_tracks else None,
            "job_task_source": "20240919161414.mp4" if fall19_job_tracks else None,
        },
        "qc_warnings": (
            [
                {
                    "source_name": "20240922115152.mp4",
                    "label": "U01_unable_to_judge",
                    "note": "挥手",
                    "reason": (
                        "Supplement image export uses U01 with quality=clear; "
                        "preserved verbatim for review because the note describes "
                        "an out-of-taxonomy gesture rather than visual uncertainty."
                    ),
                }
            ]
            if adl22_image_tracks
            else []
        ),
        "normalization": normalization,
        "outputs": {
            "source_annotations": {
                "path": source_output.as_posix(),
                "sha256": hashlib.sha256(source_payload).hexdigest(),
            },
            "action_labels": {
                "path": action_output.as_posix(),
                "count": counts["action_labels"],
                "sha256": _sha256_file(action_output),
            },
            "event_labels": {
                "path": event_output.as_posix(),
                "count": counts["event_labels"],
                "sha256": _sha256_file(event_output),
            },
        },
        "provenance_status": "project_collected_manual_cvat_unverified",
        "training_policy": "candidate_requires_qc_review",
    }
    _atomic_write_json(report_output, report, overwrite=overwrite)
    return report


def _manifest_by_source(
    metadata_rows: Iterable[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible: dict[str, Any] = {}
    all_rows: dict[str, list[Any]] = {}
    for metadata in metadata_rows:
        row = metadata.manifest_record
        if row.get("dataset") != "fall_detection_2017":
            continue
        source_name = Path(metadata.path).name
        all_rows.setdefault(source_name, []).append(metadata)
        if row.get("eligibility") is True:
            if source_name in eligible:
                raise ValueError(f"duplicate eligible manifest source: {source_name}")
            eligible[source_name] = metadata
    if not eligible:
        raise ValueError("manifest has no eligible fall_detection_2017 rows")
    excluded = []
    for source_name, rows in sorted(all_rows.items()):
        if source_name not in eligible:
            excluded.append(
                {
                    "source_name": source_name,
                    "video_ids": sorted(str(row.video_id) for row in rows),
                    "reason": "manifest_eligibility_false",
                }
            )
    return eligible, excluded


def _expand_archive(path: Path | None) -> list[tuple[str, bytes]]:
    if path is None:
        return []
    raw = path.read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".zip")]
            if not members:
                return [(path.name, raw)]
            return [(name, archive.read(name)) for name in sorted(members)]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid archive: {path}") from exc


def _read_project(payload: bytes, archive_name: str) -> list[_Task]:
    root = _read_xml(payload, archive_name)
    task_elements = root.findall("./meta/project/tasks/task")
    if not task_elements:
        raise ValueError(f"{archive_name} is not a CVAT project export")
    tracks_by_task: dict[str, list[ET.Element]] = {}
    for track in root.findall("track"):
        task_id = track.get("task_id")
        if not task_id:
            raise ValueError(f"{archive_name} has a track without task_id")
        tracks_by_task.setdefault(task_id, []).append(copy.deepcopy(track))
    result: list[_Task] = []
    offset = 0
    for task_element in task_elements:
        task_id = task_element.findtext("id") or ""
        source_name = Path(task_element.findtext("source") or "").name
        size = _positive_int(task_element.findtext("size"), "task size")
        width = _positive_int(task_element.findtext("original_size/width"), "source width")
        height = _positive_int(task_element.findtext("original_size/height"), "source height")
        if not task_id or not source_name:
            raise ValueError(f"{archive_name} has a task with missing id/source")
        result.append(
            _Task(
                source_name=source_name,
                task_id=task_id,
                size=size,
                width=width,
                height=height,
                frame_offset=offset,
                tracks=tuple(tracks_by_task.get(task_id, ())),
            )
        )
        offset += size
    return result


def _read_image_revision(payload: bytes, *, source_name: str) -> dict[str, tuple[ET.Element, ...]]:
    root = _read_xml(payload, source_name)
    task_elements = root.findall("./meta/project/tasks/task")
    images_by_task: dict[str, list[ET.Element]] = {}
    for image in root.findall("image"):
        task_id = image.get("task_id")
        if task_id:
            images_by_task.setdefault(task_id, []).append(image)
    result: dict[str, tuple[ET.Element, ...]] = {}
    for task in task_elements:
        task_id = task.findtext("id") or ""
        images = images_by_task.get(task_id, [])
        if not images:
            continue
        expected_size = _positive_int(task.findtext("size"), "image task size")
        if len(images) != expected_size:
            raise ValueError(
                f"image revision task {task_id} has {len(images)} images, expected {expected_size}"
            )
        source = Path(task.findtext("source") or "").name
        tracks: list[ET.Element] = []
        current: ET.Element | None = None
        current_signature: tuple[Any, ...] | None = None
        for frame, image in enumerate(images):
            boxes = image.findall("box")
            if len(boxes) != 1:
                raise ValueError(f"image revision task {task_id} frame {frame} must have one box")
            box = copy.deepcopy(boxes[0])
            label = box.get("label") or ""
            signature = (
                label,
                tuple(sorted((a.get("name") or "", (a.text or "").strip()) for a in box.findall("attribute"))),
            )
            if current is None or signature != current_signature:
                current = ET.Element("track", {"label": label, "source": "manual"})
                tracks.append(current)
                current_signature = signature
            box.set("frame", str(frame))
            box.set("outside", "0")
            box.set("keyframe", "1" if len(current.findall("box")) == 0 else "0")
            current.append(box)
        result[source] = tuple(tracks)
    return result


def _read_job_revision(payload: bytes, *, source_name: str) -> tuple[ET.Element, ...]:
    root = _read_xml(payload, source_name)
    job = root.find("./meta/job")
    if job is None:
        raise ValueError(f"{source_name} is not a CVAT job export")
    tracks = tuple(copy.deepcopy(track) for track in root.findall("track"))
    if not tracks:
        raise ValueError(f"{source_name} contains no replacement track")
    for track in tracks:
        for box in track.findall("box"):
            frame = _positive_or_zero_int(box.get("frame"), "job box frame")
            box.set("frame", str(frame))
            box.set("outside", "0")
    return tracks


def _read_xml(payload: bytes, archive_name: str) -> ET.Element:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if len(xml_names) != 1:
                raise ValueError(f"{archive_name} must contain exactly one XML")
            return ET.fromstring(archive.read(xml_names[0]))
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"invalid CVAT archive: {archive_name}") from exc


def _normalize_tasks(
    tasks: Iterable[_Task],
    eligible_by_source: Mapping[str, Any],
) -> tuple[ET.Element, dict[str, Any]]:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    project = ET.SubElement(ET.SubElement(root, "meta"), "project")
    ET.SubElement(project, "id").text = FALL_DETECTION_2017_BATCH_ID
    ET.SubElement(project, "name").text = "fall_risk__fall_detection_2017__manual"
    tasks_element = ET.SubElement(project, "tasks")
    source_dimensions: Counter[str] = Counter()
    target_dimensions: Counter[str] = Counter()
    scale_factors: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    frame_mappings: list[dict[str, Any]] = []
    normalized_task_count = 0
    normalized_track_count = 0
    next_task_id = 1
    next_track_id = 0

    for raw_task in sorted(tasks, key=lambda task: task.source_name):
        metadata = eligible_by_source.get(raw_task.source_name)
        if metadata is None:
            continue
        if not raw_task.tracks:
            raise ValueError(f"eligible CVAT task has no tracks: {raw_task.source_name}")
        row = metadata.manifest_record
        target_size = int(metadata.frame_count)
        target_width = _positive_int(row.get("width"), "manifest width")
        target_height = _positive_int(row.get("height"), "manifest height")
        scale_x = target_width / raw_task.width
        scale_y = target_height / raw_task.height
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"non-isotropic scale for {raw_task.source_name}")
        source_dimensions[f"{raw_task.width}x{raw_task.height}"] += 1
        target_dimensions[f"{target_width}x{target_height}"] += 1
        scale_factors[_format_scale(scale_x)] += 1
        if raw_task.size != target_size:
            expected = _ANOMALOUS_SIZES.get(raw_task.source_name)
            if expected != target_size or raw_task.size not in (227, 230):
                raise ValueError(
                    f"frame count mismatch for {raw_task.source_name}: CVAT={raw_task.size}, manifest={target_size}"
                )
            frame_mappings.append(
                {
                    "source_name": raw_task.source_name,
                    "source_frame_count": raw_task.size,
                    "target_frame_count": target_size,
                    "method": "linear_endpoint_resample",
                }
            )

        task_id = str(next_task_id)
        next_task_id += 1
        task_element = ET.SubElement(tasks_element, "task")
        subset = _sanitize(str(row.get("subset") or "default"))
        ET.SubElement(task_element, "id").text = task_id
        ET.SubElement(task_element, "name").text = (
            f"fall_risk__fall_detection_2017__{subset}__{metadata.video_id}"
        )
        for tag, value in (
            ("size", str(target_size)),
            ("mode", "interpolation"),
            ("overlap", "0"),
            ("subset", subset),
            ("start_frame", "0"),
            ("stop_frame", str(target_size - 1)),
        ):
            ET.SubElement(task_element, tag).text = value
        original_size = ET.SubElement(task_element, "original_size")
        ET.SubElement(original_size, "width").text = str(target_width)
        ET.SubElement(original_size, "height").text = str(target_height)
        ET.SubElement(task_element, "source").text = raw_task.source_name

        local_tracks = []
        for raw_track in raw_task.tracks:
            track = _localize_track(raw_track, raw_task)
            normalized = _resample_track(track, raw_task.size, target_size)
            if not normalized.findall("box"):
                continue
            normalized.set("id", str(next_track_id))
            normalized.set("task_id", task_id)
            normalized.set("source", "manual")
            normalized.set("frame_coordinate_system", "project_global")
            for box in normalized.findall("box"):
                box.set("frame", str(int(box.get("frame") or 0) + _project_offset(tasks, raw_task, eligible_by_source)))
                _scale_box(box, scale_x, scale_y, target_width, target_height)
                _set_attribute(box, "target_subject", str(metadata.subject_id))
            action_counts[normalized.get("label") or ""] += 1
            local_tracks.append(normalized)
            next_track_id += 1
            normalized_track_count += 1
            root.append(normalized)
        if not local_tracks:
            raise ValueError(f"eligible CVAT task has no active tracks: {raw_task.source_name}")
        normalized_task_count += 1

    return root, {
        "normalized_task_count": normalized_task_count,
        "normalized_track_count": normalized_track_count,
        "action_id_counts_raw": dict(sorted(action_counts.items())),
        "source_dimensions": dict(sorted(source_dimensions.items())),
        "target_dimensions": dict(sorted(target_dimensions.items())),
        "scale_factors": dict(sorted(scale_factors.items())),
        "frame_mapping": {
            "method": "linear_endpoint_resample",
            "affected_source_count": len(frame_mappings),
            "affected_sources": frame_mappings,
        },
    }


def _project_offset(
    all_tasks: Iterable[_Task], raw_task: _Task, eligible_by_source: Mapping[str, Any]
) -> int:
    # The normalized XML is sorted by source name, so its project offsets are
    # based on eligible tasks only and use manifest frame counts.
    offset = 0
    for task in sorted(all_tasks, key=lambda item: item.source_name):
        if task.source_name not in eligible_by_source:
            continue
        if task.source_name == raw_task.source_name:
            return offset
        metadata = eligible_by_source[task.source_name]
        offset += int(metadata.frame_count)
    raise ValueError(f"task is absent from normalization order: {raw_task.source_name}")


def _localize_track(track: ET.Element, task: _Task) -> ET.Element:
    result = copy.deepcopy(track)
    boxes = result.findall("box")
    if not boxes:
        return result
    frames = [_positive_or_zero_int(box.get("frame"), "track box frame") for box in boxes]
    if task.tracks_are_local:
        if not all(0 <= frame < task.size for frame in frames):
            raise ValueError(f"local track frame outside task: {task.source_name}")
        return result
    global_valid = all(task.frame_offset <= frame < task.frame_offset + task.size for frame in frames)
    local_valid = all(0 <= frame < task.size for frame in frames)
    if global_valid and (task.frame_offset > 0 or not local_valid):
        for box in boxes:
            box.set("frame", str(int(box.get("frame") or 0) - task.frame_offset))
    elif not local_valid:
        raise ValueError(f"track frames do not fit task {task.source_name}")
    return result


def _resample_track(track: ET.Element, source_size: int, target_size: int) -> ET.Element:
    if source_size <= 0 or target_size <= 0:
        raise ValueError("frame counts must be positive")
    active = [
        (int(box.get("frame") or 0), box)
        for box in track.findall("box")
        if box.get("outside") != "1"
    ]
    active.sort(key=lambda item: item[0])
    if not active:
        return ET.Element("track", dict(track.attrib))
    if any(frame < 0 or frame >= source_size for frame, _ in active):
        raise ValueError("track box frame outside source task")
    if target_size == source_size:
        target_frames = [(frame, box) for frame, box in active]
    else:
        start = _map_frame(active[0][0], source_size, target_size)
        end = _map_frame(active[-1][0], source_size, target_size)
        target_frames = []
        for target_frame in range(start, end + 1):
            source_frame = _inverse_map_frame(target_frame, source_size, target_size)
            _, source_box = min(active, key=lambda item: (abs(item[0] - source_frame), item[0]))
            target_frames.append((target_frame, source_box))
    result = ET.Element("track", {key: value for key, value in track.attrib.items() if key not in {"id", "task_id"}})
    seen: set[int] = set()
    for frame, source_box in target_frames:
        if frame in seen:
            continue
        seen.add(frame)
        box = copy.deepcopy(source_box)
        box.set("frame", str(frame))
        box.set("outside", "0")
        result.append(box)
    return result


def _map_frame(frame: int, source_size: int, target_size: int) -> int:
    if source_size == 1 or target_size == 1:
        return 0
    return int(math.floor(frame * (target_size - 1) / (source_size - 1) + 0.5))


def _inverse_map_frame(frame: int, source_size: int, target_size: int) -> int:
    if target_size == 1 or source_size == 1:
        return 0
    return int(math.floor(frame * (source_size - 1) / (target_size - 1) + 0.5))


def _scale_box(box: ET.Element, scale_x: float, scale_y: float, width: int, height: int) -> None:
    for key, scale, upper in (
        ("xtl", scale_x, width),
        ("xbr", scale_x, width),
        ("ytl", scale_y, height),
        ("ybr", scale_y, height),
    ):
        try:
            value = float(box.attrib[key]) * scale
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid CVAT box coordinate {key}") from exc
        if not math.isfinite(value) or value < 0 or value > upper + 1e-6:
            raise ValueError(f"scaled CVAT box coordinate {key} is outside media bounds")
        box.set(key, f"{value:.2f}")


def _set_attribute(box: ET.Element, name: str, value: str) -> None:
    matches = [child for child in box.findall("attribute") if child.get("name") == name]
    if len(matches) > 1:
        raise ValueError(f"box has duplicate {name} attributes")
    target = matches[0] if matches else ET.SubElement(box, "attribute", {"name": name})
    target.text = value


def _deterministic_zip(root: ET.Element) -> bytes:
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, payload)
    return buffer.getvalue()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, content)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _positive_or_zero_int(value: Any, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "default"


def _format_scale(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
