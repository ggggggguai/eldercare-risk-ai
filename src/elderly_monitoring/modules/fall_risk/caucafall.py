from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from elderly_monitoring.modules.fall_risk.annotations import (
    ConvertedFallLabels,
    convert_cvat_xml,
    write_converted_fall_labels,
)


CAUCAFALL_LABEL_ALIASES: dict[str, str] = {
    "A01_normal_walk": "A01_normal_walk",
    "A02_normal_turn": "A02_normal_turn",
    "A03_controlled_sit_down": "A03_controlled_sit_down",
    "A03_normal_sit": "A03_controlled_sit_down",
    "A04_normal_sit_to_stand": "A04_normal_sit_to_stand",
    "A04_normal_stand": "A04_normal_sit_to_stand",
    "A05_controlled_squat": "A05_controlled_squat",
    "A05_normal_squat": "A05_controlled_squat",
    "A06_controlled_bend": "A06_controlled_bend",
    "A06_normal_bend": "A06_controlled_bend",
    "A07_controlled_lie_down": "A07_controlled_lie_down",
    "A08_routine_support_contact": "A08_routine_support_contact",
    "A09_kneel_or_floor_activity": "A09_kneel_or_floor_activity",
    "A09_normal_kneel": "A09_kneel_or_floor_activity",
    "A10_normal_step_adjustment": "A10_normal_step_adjustment",
    "A11_assisted_sit_or_lowering": "A11_assisted_sit_or_lowering",
    "A12_normal_hop": "A12_normal_hop",
    "C01_failed_sit_to_stand": "C01_failed_sit_to_stand",
    "D01_forward_fall": "D01_forward_fall",
    "D02_lateral_fall": "D02_lateral_fall",
    "D03_backward_fall": "D03_backward_fall",
    "D04_long_static_after_fall": "D04_long_static_after_fall",
    "D05_seated_fall": "D05_seated_fall",
    "U01_unable_to_judge": "U01_unable_to_judge",
}

_REMOVED_METADATA_TAGS = {"owner", "assignee", "username", "email", "url"}


@dataclass(frozen=True)
class CaucaFallImportResult:
    action_labels: list[dict[str, Any]]
    event_labels: list[dict[str, Any]]
    redacted_exports: list[dict[str, Any]]
    report: dict[str, Any]


def import_caucafall_cvat_labels(
    source_archive: Path | str,
    *,
    manifest_path: Path | str,
    redacted_export_dir: Path | str,
    labeler: str = "caucafall_cvat_manual_20260723",
    overwrite: bool = False,
) -> CaucaFallImportResult:
    source_path = Path(source_archive)
    manifest_file = Path(manifest_path)
    output_dir = Path(redacted_export_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    manifest_rows = _load_manifest(manifest_file)
    caucafall_rows = {
        Path(row["path"]).name: row
        for row in manifest_rows
        if row.get("dataset") == "caucafall"
    }
    if len(caucafall_rows) != 100:
        raise ValueError(
            "CaucaFall manifest must contain 100 videos before import; "
            f"found {len(caucafall_rows)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256(source_path)
    action_labels: list[dict[str, Any]] = []
    event_labels: list[dict[str, Any]] = []
    redacted_exports: list[dict[str, Any]] = []
    raw_label_counts: Counter[str] = Counter()
    normalized_label_counts: Counter[str] = Counter()
    alias_counts: Counter[str] = Counter()
    removed_metadata: Counter[str] = Counter()
    seen_sources: set[str] = set()

    try:
        with zipfile.ZipFile(source_path) as outer:
            invalid_member = outer.testzip()
            if invalid_member is not None:
                raise ValueError(f"invalid CaucaFall archive member: {invalid_member}")
            inner_names = sorted(
                name for name in outer.namelist() if name.lower().endswith(".zip")
            )
            if len(inner_names) != 10:
                raise ValueError(
                    "CaucaFall annotation archive must contain 10 inner ZIPs; "
                    f"found {len(inner_names)}"
                )
            for inner_name in inner_names:
                payload, info = _normalize_inner_export(
                    outer.read(inner_name),
                    caucafall_rows,
                    raw_label_counts,
                    normalized_label_counts,
                    alias_counts,
                    removed_metadata,
                )
                filename = Path(inner_name).name
                output_path = output_dir / filename
                if output_path.exists() and not overwrite:
                    raise FileExistsError(output_path)
                output_path.write_bytes(payload)
                converted = convert_cvat_xml(
                    output_path,
                    manifest_path=manifest_file,
                    fps=None,
                    labeler=labeler,
                )
                action_labels.extend(converted.action_labels)
                event_labels.extend(converted.event_labels)
                for source_name in info["source_names"]:
                    if source_name in seen_sources:
                        raise ValueError(f"duplicate CaucaFall video source: {source_name}")
                    seen_sources.add(source_name)
                redacted_exports.append(
                    {
                        "path": _relative_path(output_path),
                        "sha256": _sha256(output_path),
                        "filename": filename,
                        "source_names": info["source_names"],
                        "task_count": info["task_count"],
                        "track_count": info["track_count"],
                        "box_count": info["box_count"],
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid CaucaFall annotation archive: {source_path}") from exc

    if len(seen_sources) != len(caucafall_rows):
        missing = sorted(set(caucafall_rows) - seen_sources)
        raise ValueError(f"CaucaFall annotation archive is missing videos: {missing}")
    action_labels.sort(key=lambda row: (row["video_id"], row["start_frame"], row["label_id"]))
    event_labels.sort(key=lambda row: (row["video_id"], row["start_frame"], row["label_id"]))
    report = {
        "schema_version": "caucafall-cvat-import-v1",
        "source_archive": {
            "filename": source_path.name,
            "sha256": source_sha256,
            "copied_into_repository": False,
        },
        "redacted_exports": sorted(redacted_exports, key=lambda row: row["filename"]),
        "identity_metadata_tags_removed": dict(sorted(removed_metadata.items())),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "normalized_label_counts": dict(sorted(normalized_label_counts.items())),
        "alias_counts": dict(sorted(alias_counts.items())),
        "video_count": len(seen_sources),
        "track_count": len(action_labels),
        "event_count": len(event_labels),
        "provenance_status": "manual_cvat_source_verified",
    }
    return CaucaFallImportResult(
        action_labels=action_labels,
        event_labels=event_labels,
        redacted_exports=redacted_exports,
        report=report,
    )


def write_caucafall_cvat_labels(
    imported: CaucaFallImportResult,
    *,
    action_output_path: Path | str,
    event_output_path: Path | str,
    report_output_path: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    action_path = Path(action_output_path)
    event_path = Path(event_output_path)
    report_path = Path(report_output_path)
    outputs = (action_path, event_path, report_path)
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
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
        "action_labels": _relative_path(action_path),
        "event_labels": _relative_path(event_path),
        "report": _relative_path(report_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _normalize_inner_export(
    payload: bytes,
    manifest_rows: Mapping[str, Mapping[str, Any]],
    raw_label_counts: Counter[str],
    normalized_label_counts: Counter[str],
    alias_counts: Counter[str],
    removed_metadata: Counter[str],
) -> tuple[bytes, dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("nested CaucaFall ZIP has invalid compressed data")
            xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if len(xml_names) != 1:
                raise ValueError("nested CaucaFall ZIP must contain exactly one XML")
            root = ET.fromstring(archive.read(xml_names[0]))
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError("invalid nested CaucaFall CVAT ZIP") from exc

    tasks = root.findall("./meta/project/tasks/task")
    if not tasks:
        task = root.find("./meta/task")
        tasks = [task] if task is not None else []
    if len(tasks) != 10:
        raise ValueError(f"nested CaucaFall export must contain 10 tasks; found {len(tasks)}")

    source_names: list[str] = []
    track_count = 0
    box_count = 0
    for task in tasks:
        source_element = task.find("source")
        name_element = task.find("name")
        size_element = task.find("size")
        task_id = task.findtext("id") or ""
        source_name = Path(source_element.text or "").name if source_element is not None else ""
        if not source_name or source_name not in manifest_rows:
            raise ValueError(f"CaucaFall task {task_id} source is not in manifest: {source_name!r}")
        row = manifest_rows[source_name]
        expected_size = int(row["frame_count"])
        if size_element is None or int(size_element.text or "-1") != expected_size:
            raise ValueError(f"CaucaFall task {task_id} frame count does not match manifest")
        subset = str(row["subset"])
        video_id = str(row["video_id"])
        if name_element is None or source_element is None:
            raise ValueError(f"CaucaFall task {task_id} is missing name/source")
        name_element.text = f"fall_risk__caucafall__{subset}__{video_id}"
        source_element.text = source_name
        source_names.append(source_name)
        subject_id = str(row["subject_id"])
        for track in root.findall(".//track"):
            if track.attrib.get("task_id") != task_id:
                continue
            track_count += 1
            raw_label = str(track.attrib.get("label") or "")
            normalized_label = CAUCAFALL_LABEL_ALIASES.get(raw_label)
            if normalized_label is None:
                raise ValueError(f"unsupported CaucaFall CVAT label: {raw_label!r}")
            raw_label_counts[raw_label] += 1
            normalized_label_counts[normalized_label] += 1
            if normalized_label != raw_label:
                alias_counts[f"{raw_label}->{normalized_label}"] += 1
            track.attrib["label"] = normalized_label
            boxes = track.findall("box")
            box_count += len(boxes)
            note = ""
            if normalized_label != raw_label:
                note = f"Source CVAT label {raw_label} normalized to {normalized_label}."
            if normalized_label == "U01_unable_to_judge":
                note = "Manual U01 label; source export did not provide a reason."
            for box in boxes:
                _set_box_attribute(box, "target_subject", subject_id)
                if note:
                    _set_box_attribute(box, "note", note)

    removed_metadata.update(_remove_metadata(root))
    return _deterministic_zip(root), {
        "source_names": sorted(source_names),
        "task_count": len(tasks),
        "track_count": track_count,
        "box_count": box_count,
    }


def _set_box_attribute(box: ET.Element, name: str, value: str) -> None:
    matches = [
        child
        for child in box.findall("attribute")
        if child.attrib.get("name") == name
    ]
    if len(matches) > 1:
        raise ValueError(f"CVAT box has duplicate {name} attributes")
    target = matches[0] if matches else ET.SubElement(box, "attribute", {"name": name})
    target.text = value


def _remove_metadata(root: ET.Element) -> Counter[str]:
    removed: Counter[str] = Counter()

    def visit(parent: ET.Element) -> None:
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1].lower()
            if tag in _REMOVED_METADATA_TAGS:
                for nested in child.iter():
                    nested_tag = nested.tag.rsplit("}", 1)[-1].lower()
                    if nested_tag in _REMOVED_METADATA_TAGS:
                        removed[nested_tag] += 1
                parent.remove(child)
            else:
                visit(child)

    visit(root)
    return removed


def _deterministic_zip(root: ET.Element) -> bytes:
    xml_payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, xml_payload)
    return buffer.getvalue()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
