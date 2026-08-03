from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


SOURCE_MAP_SCHEMA = "fall-tiktok-source-map-v1"
COLLECTION_DECISION_SCHEMA = "fall-tiktok-collection-decision-v1"
IDENTITY_TAGS = {"owner", "assignee", "username", "email"}


@dataclass(frozen=True)
class FallTiktokPreparedExport:
    payload: bytes
    source_sha256: str
    prepared_sha256: str
    task_count: int
    track_count: int
    removed_identity_elements: int
    task_mappings: list[dict[str, Any]]


@dataclass(frozen=True)
class FallTiktokCollectionDecision:
    decision_id: str
    decided_at: str
    decided_by: str
    collection_status: str
    training_use: str
    redistribution_use: str
    consent_status: str
    subject_grouping_status: str
    source_group_id: str
    source_uri: str
    provenance_status: str


def prepare_fall_tiktok_cvat_export(
    input_path: Path | str,
    source_map_path: Path | str,
) -> FallTiktokPreparedExport:
    source_path = Path(input_path)
    mapping_path = Path(source_map_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    entries = load_fall_tiktok_source_map(mapping_path)

    try:
        with zipfile.ZipFile(source_path) as archive:
            root = ET.fromstring(archive.read("annotations.xml"))
    except (KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid fall_tiktok CVAT archive: {source_path}") from exc

    tasks = root.findall("./meta/project/tasks/task")
    if len(tasks) != len(entries):
        raise ValueError(
            "fall_tiktok task count does not match source map: "
            f"tasks={len(tasks)}, mappings={len(entries)}"
        )

    task_mappings: list[dict[str, Any]] = []
    for task, entry in zip(tasks, entries):
        sequence = entry["sequence"]
        expected_source_names = {f"{sequence:03d}.mp4", f"{sequence}.mp4"}
        task_name = task.findtext("name") or ""
        source_name = task.findtext("source") or ""
        if Path(task_name).name not in expected_source_names:
            raise ValueError(
                f"fall_tiktok task {sequence} has unexpected name {task_name!r}"
            )
        if Path(source_name).name not in expected_source_names:
            raise ValueError(
                f"fall_tiktok task {sequence} has unexpected source {source_name!r}"
            )

        video_id = f"fall_tiktok_clip_{sequence:03d}"
        canonical_task_name = (
            "fall_risk__fall_tiktok__annotated_clips__" + video_id
        )
        task.find("name").text = canonical_task_name
        task.find("source").text = entry["filename"]
        task_mappings.append(
            {
                "sequence": sequence,
                "cvat_task_id": task.findtext("id"),
                "original_task_name": task_name,
                "video_id": video_id,
                "source_filename": entry["filename"],
                "original_filename": entry["original_filename"],
            }
        )

    removed_identity_elements = _remove_identity_elements(root)
    payload = _deterministic_cvat_zip(root)
    return FallTiktokPreparedExport(
        payload=payload,
        source_sha256=_sha256_file(source_path),
        prepared_sha256=hashlib.sha256(payload).hexdigest(),
        task_count=len(tasks),
        track_count=len(root.findall("track")),
        removed_identity_elements=removed_identity_elements,
        task_mappings=task_mappings,
    )


def write_prepared_fall_tiktok_export(
    result: FallTiktokPreparedExport,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(result.payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_fall_tiktok_source_map(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SOURCE_MAP_SCHEMA:
        raise ValueError(f"unsupported fall_tiktok source map: {path}")
    if payload.get("dataset") != "fall_tiktok":
        raise ValueError(f"invalid fall_tiktok source map dataset: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"fall_tiktok source map entries must be non-empty: {path}")

    normalized: list[dict[str, Any]] = []
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"fall_tiktok source map entry {position} must be an object")
        sequence = entry.get("sequence")
        if sequence != position:
            raise ValueError(
                f"fall_tiktok source map sequence must be contiguous at {position}"
            )
        filename = entry.get("filename")
        original_filename = entry.get("original_filename")
        if filename != f"{sequence}.mp4":
            raise ValueError(
                f"fall_tiktok source map entry {position} has invalid filename"
            )
        if not isinstance(original_filename, str) or not original_filename.endswith(".mp4"):
            raise ValueError(
                f"fall_tiktok source map entry {position} has invalid original_filename"
            )
        normalized.append(
            {
                "sequence": sequence,
                "filename": filename,
                "original_filename": original_filename,
            }
        )
    if len({entry["filename"] for entry in normalized}) != len(normalized):
        raise ValueError("fall_tiktok source map has duplicate filenames")
    return normalized


def load_fall_tiktok_collection_decision(
    path: Path | str,
) -> FallTiktokCollectionDecision:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != COLLECTION_DECISION_SCHEMA
    ):
        raise ValueError(f"unsupported fall_tiktok collection decision: {path}")
    if payload.get("dataset") != "fall_tiktok":
        raise ValueError(f"invalid fall_tiktok collection decision dataset: {path}")

    expected = {
        "collection_status": "project_collected",
        "training_use": "authorized",
        "redistribution_use": "not_authorized_by_this_decision",
        "consent_status": "not_recorded",
        "subject_grouping_status": "unknown",
        "provenance_status": "project_collected_training_authorized",
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(
                f"fall_tiktok collection decision {field} must be {expected_value!r}"
            )

    for field in (
        "decision_id",
        "decided_at",
        "decided_by",
        "source_group_id",
        "source_uri",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"fall_tiktok collection decision {field} must be non-empty"
            )
    source_uri = str(payload["source_uri"])
    if not source_uri.startswith("internal://collection/"):
        raise ValueError(
            "fall_tiktok collection decision source_uri must use internal://collection/"
        )

    return FallTiktokCollectionDecision(
        decision_id=str(payload["decision_id"]),
        decided_at=str(payload["decided_at"]),
        decided_by=str(payload["decided_by"]),
        collection_status=str(payload["collection_status"]),
        training_use=str(payload["training_use"]),
        redistribution_use=str(payload["redistribution_use"]),
        consent_status=str(payload["consent_status"]),
        subject_grouping_status=str(payload["subject_grouping_status"]),
        source_group_id=str(payload["source_group_id"]),
        source_uri=source_uri,
        provenance_status=str(payload["provenance_status"]),
    )


def _remove_identity_elements(root: ET.Element) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag.lower() in IDENTITY_TAGS:
                parent.remove(child)
                removed += 1
    return removed


def _deterministic_cvat_zip(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    xml_payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    from io import BytesIO

    buffer = BytesIO()
    info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, xml_payload)
    return buffer.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
