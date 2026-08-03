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
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from elderly_monitoring.modules.fall_risk.annotations import (
    ConvertedFallLabels,
    VideoMetadata,
    convert_cvat_xml,
    load_video_manifest,
    write_converted_fall_labels,
)


NTU_RGBD_A043_BATCH_ID = "ntu_rgbd_a043_cvat_review"
DEFAULT_LABEL_CONFIG = Path("configs/data/fall_risk_cvat_labels_v2.json")
DEFAULT_DECISION_CONFIG = Path("configs/data/ntu_rgbd_a043_cvat_decision_v1.json")

_SOURCE_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})"
    r"R(?P<repetition>\d{3})A043_rgb\.(?:mp4|avi)"
)
_JOB_ARCHIVE_PATTERN = re.compile(
    r"(?P<stem>S\d{3}C\d{3}P\d{3}R\d{3}A043)(?:_rgb)?"
)
_ALLOWED_TRACK_LABELS = {
    "A01_normal_walk",
    "A04_normal_sit_to_stand",
    "A05_controlled_squat",
    "A07_controlled_lie_down",
    "D01_forward_fall",
    "D02_lateral_fall",
    "D03_backward_fall",
    "U01_unable_to_judge",
}
_FALL_TRACK_LABELS = {
    "D01_forward_fall",
    "D02_lateral_fall",
    "D03_backward_fall",
}
_IDENTITY_TAGS = {"owner", "assignee", "username", "email", "url"}
_PENDING_REVIEW_NOTE = (
    "NTU RGB+D A043 manual CVAT label accepted by the project owner."
)
_WHOLE_CLIP_NOTE = (
    "Whole-clip fall track; the manual annotation does not contain a fall onset."
)
_HARD_NEGATIVE_NOTE = (
    "Three-view review accepted this A043 source as a controlled-squat fall hard negative."
)


@dataclass(frozen=True)
class _RawTask:
    source_name: str
    size: int
    width: int
    height: int
    tracks: tuple[ET.Element, ...]
    original_label_names: tuple[str, ...]


def import_ntu_rgbd_a043_cvat_labels(
    source_paths: Sequence[Path | str],
    *,
    revision_paths: Sequence[Path | str] = (),
    manifest_path: Path | str,
    output_dir: Path | str,
    label_config_path: Path | str = DEFAULT_LABEL_CONFIG,
    decision_config_path: Path | str = DEFAULT_DECISION_CONFIG,
    labeler: str = "ntu_rgbd_a043_cvat_review_20260730",
    allow_incomplete: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize accepted external A043 CVAT exports into a v2 JSONL batch.

    The source exports contain annotations over 640x360 MP4 transcodes while the
    manifest refers to the original 1920x1080 AVI files. This importer verifies
    frame-count identity, scales boxes to the manifest dimensions, removes CVAT
    identity metadata, and keeps the result isolated from root-label publication.
    """
    manifest_file = Path(manifest_path)
    label_config_file = Path(label_config_path)
    decision_config_file = Path(decision_config_path)
    destination = Path(output_dir)
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    if not label_config_file.is_file():
        raise FileNotFoundError(label_config_file)
    if not decision_config_file.is_file():
        raise FileNotFoundError(decision_config_file)
    base_archives = _expand_archives(source_paths)
    revision_archives = (
        _expand_archives(revision_paths) if revision_paths else []
    )
    manifest = load_video_manifest(manifest_file)
    manifest_by_source = _a043_manifest_by_source(manifest.values())
    label_config = _load_label_config(label_config_file)
    decision = load_ntu_rgbd_a043_decision(decision_config_file)
    base_tasks, base_archive_report, removed_metadata = _read_tasks(
        base_archives,
        allow_empty_tasks=bool(revision_archives),
    )
    revision_tasks, revision_archive_report, revision_removed_metadata = _read_tasks(
        revision_archives,
        allow_empty_tasks=False,
    )
    removed_metadata.update(revision_removed_metadata)
    base_by_source = _index_unique_tasks(base_tasks, role="base")
    revision_by_source = _index_unique_tasks(revision_tasks, role="revision")
    unmatched_revisions = sorted(set(revision_by_source) - set(base_by_source))
    if unmatched_revisions:
        raise ValueError(
            "revision sources do not exist in base exports: "
            f"{unmatched_revisions}"
        )
    base_by_source.update(revision_by_source)
    raw_tasks = list(base_by_source.values())
    empty_sources = sorted(task.source_name for task in raw_tasks if not task.tracks)
    if empty_sources:
        raise ValueError(
            "NTU RGB+D A043 tasks have no tracks after revision overlay: "
            f"{empty_sources}"
        )
    if not raw_tasks:
        raise ValueError("NTU RGB+D A043 import contains no CVAT tasks")
    raw_tasks, adjudication = _apply_adjudications(
        raw_tasks,
        decision["adjudications"],
    )

    source_archives = [
        *({**row, "role": "base"} for row in base_archive_report),
        *({**row, "role": "revision"} for row in revision_archive_report),
    ]
    source_archives.sort(
        key=lambda item: (item["role"], item["source_group"], item["filename"])
    )
    seen_source_names = list(base_by_source)

    observed_setups = sorted(
        {_required_source_match(task.source_name).group("setup") for task in raw_tasks}
    )
    expected_source_names = {
        mp4_name
        for mp4_name, metadata in manifest_by_source.items()
        if str(metadata.manifest_record.get("subset"))
        in {f"setup_s{setup}" for setup in observed_setups}
    }
    actual_source_names = set(seen_source_names)
    missing_source_names = sorted(expected_source_names - actual_source_names)
    extra_source_names = sorted(actual_source_names - set(manifest_by_source))
    if extra_source_names:
        raise ValueError(
            f"A043 CVAT sources are absent from the NTU manifest: {extra_source_names}"
        )
    if missing_source_names and not allow_incomplete:
        raise ValueError(
            f"missing {len(missing_source_names)} expected A043 source(s): "
            f"{missing_source_names}"
        )

    normalized_root, normalization = _normalize_tasks(
        raw_tasks,
        manifest_by_source,
        label_config,
    )
    unexpected_protocols = sorted(
        set(normalization["protocol_counts"]) - set(decision["accepted_protocols"])
    )
    if unexpected_protocols:
        raise ValueError(
            f"NTU RGB+D A043 protocols lack acceptance: {unexpected_protocols}"
        )
    source_payload = _deterministic_zip(normalized_root)
    source_payload_sha256 = hashlib.sha256(source_payload).hexdigest()
    report: dict[str, Any] = {
        "schema_version": "ntu-rgbd-a043-cvat-import-v2",
        "batch_id": NTU_RGBD_A043_BATCH_ID,
        "source_action_code": "A043",
        "source_archives": source_archives,
        "observed_setups": observed_setups,
        "manifest_expected_source_count": len(expected_source_names),
        "source_video_count": len(actual_source_names),
        "revision_overlay": {
            "replaced_source_count": len(revision_by_source),
            "replaced_source_names": sorted(revision_by_source),
        },
        "adjudication": adjudication,
        "complete_against_manifest": not missing_source_names,
        "missing_source_names": missing_source_names,
        "identity_metadata_tags_removed": dict(sorted(removed_metadata.items())),
        "coordinate_transform": {
            "mode": "isotropic_scale_to_manifest_dimensions",
            "source_dimensions": normalization["source_dimensions"],
            "target_dimensions": normalization["target_dimensions"],
            "scale_factors": normalization["scale_factors"],
        },
        "protocol_counts": normalization["protocol_counts"],
        "raw_label_counts": normalization["raw_label_counts"],
        "original_label_schema_variants": _label_schema_report(raw_tasks),
        "cross_view_qc": normalization["cross_view_qc"],
        "manual_acceptance_decision": {
            "path": _portable_path(decision_config_file),
            "sha256": _sha256_file(decision_config_file),
            **decision,
        },
        "publication_status": "accepted_for_v2_publication",
        "known_limitations": [
            *(
                ["whole-clip fall tracks do not contain an annotated fall onset"]
                if normalization["protocol_counts"].get(
                    "whole_clip_fall_without_onset"
                )
                else []
            ),
            *(
                ["whole-clip uncertain tracks are excluded from model supervision"]
                if normalization["protocol_counts"].get("whole_clip_uncertain")
                else []
            ),
            *(
                ["reviewed A043 controlled-squat clips are fall hard negatives"]
                if normalization["protocol_counts"].get(
                    "whole_clip_controlled_squat_hard_negative"
                )
                else []
            ),
            *(
                ["source export set is incomplete against the selected setups"]
                if missing_source_names
                else []
            ),
        ],
        "redacted_source_sha256": source_payload_sha256,
        "original_exports_copied_into_repository": False,
    }
    return _write_batch(
        source_payload,
        report,
        manifest_path=manifest_file,
        output_dir=destination,
        labeler=labeler,
        overwrite=overwrite,
    )


def _expand_archives(source_paths: Sequence[Path | str]) -> list[tuple[str, Path]]:
    expanded: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for source in source_paths:
        path = Path(source)
        if path.is_dir():
            candidates = sorted(path.glob("*.zip"))
            source_group = path.name
        elif path.is_file() and path.suffix.lower() == ".zip":
            candidates = [path]
            source_group = path.stem
        else:
            raise FileNotFoundError(path)
        if not candidates:
            raise FileNotFoundError(f"no ZIP archives under {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                raise ValueError(f"duplicate source archive: {candidate}")
            seen.add(resolved)
            expanded.append((source_group, candidate))
    if not expanded:
        raise ValueError("at least one NTU RGB+D CVAT source path is required")
    return sorted(expanded, key=lambda item: (item[0], item[1].name))


def _a043_manifest_by_source(
    metadata_rows: Iterable[VideoMetadata],
) -> dict[str, VideoMetadata]:
    result: dict[str, VideoMetadata] = {}
    for metadata in metadata_rows:
        row = metadata.manifest_record
        if row.get("dataset") != "ntu_rgbd" or row.get("source_action_code") != "A043":
            continue
        if row.get("eligibility") is not True:
            continue
        avi_name = Path(metadata.path).name
        if not avi_name.lower().endswith("_rgb.avi"):
            raise ValueError(f"invalid NTU RGB+D A043 manifest path: {metadata.path}")
        mp4_name = f"{Path(avi_name).stem}.mp4"
        if mp4_name in result:
            raise ValueError(f"duplicate NTU RGB+D manifest source: {mp4_name}")
        result[mp4_name] = metadata
    if not result:
        raise ValueError("manifest has no eligible NTU RGB+D A043 videos")
    return result


def _load_label_config(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CVAT label config: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("CVAT label config must be a non-empty array")
    names = [item.get("name") for item in payload if isinstance(item, Mapping)]
    if len(names) != len(payload) or len(set(names)) != len(names):
        raise ValueError("CVAT label config has invalid or duplicate names")
    if not _ALLOWED_TRACK_LABELS.issubset(set(names)):
        raise ValueError("CVAT label config is missing required NTU A043 labels")
    return payload


def load_ntu_rgbd_a043_decision(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid NTU RGB+D A043 decision config: {path}") from exc
    expected = {
        "schema_version",
        "decision_id",
        "reviewed_at",
        "reviewer_id",
        "source_action_code",
        "decision",
        "direct_filename_import",
        "accepted_batch_id",
        "accepted_protocols",
        "v3_event_training_policy",
        "adjudications",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("NTU RGB+D A043 decision config has an invalid shape")
    required_values = {
        "schema_version": "ntu-rgbd-a043-cvat-decision-v1",
        "source_action_code": "A043",
        "decision": "accept_manual_cvat_labels",
        "direct_filename_import": False,
        "accepted_batch_id": NTU_RGBD_A043_BATCH_ID,
        "v3_event_training_policy": "auxiliary_approximate",
    }
    for field, expected_value in required_values.items():
        if payload.get(field) != expected_value:
            raise ValueError(
                f"NTU RGB+D A043 decision config has invalid {field}"
            )
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(payload.get("reviewed_at"))) is None:
        raise ValueError("NTU RGB+D A043 decision date must use YYYY-MM-DD")
    if not all(
        isinstance(payload.get(field), str) and payload[field].strip()
        for field in ("decision_id", "reviewer_id")
    ):
        raise ValueError("NTU RGB+D A043 decision identifiers must be non-empty")
    if set(payload.get("accepted_protocols") or []) != {
        "segmented_normal_to_fall",
        "segmented_normal_to_fall_with_trailing_outside",
        "segmented_normal_to_fall_to_sit_to_stand",
        "segmented_nonfall_controlled_lie_down",
        "whole_clip_controlled_squat_hard_negative",
        "whole_clip_fall_without_onset",
        "whole_clip_uncertain",
    }:
        raise ValueError("NTU RGB+D A043 decision has invalid accepted protocols")
    payload["adjudications"] = _validate_adjudications(payload.get("adjudications"))
    return payload


def _validate_adjudications(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("NTU RGB+D A043 adjudications must be an array")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"NTU RGB+D A043 adjudication {index} must be an object")
        decision_type = raw.get("type")
        common = {"type", "accepted_label", "evidence", "reason"}
        if decision_type == "label_override":
            expected = common | {"source_name", "expected_label"}
            if set(raw) != expected:
                raise ValueError(f"invalid label_override adjudication {index}")
            source_names = [_canonical_source_name(str(raw["source_name"]))]
            if raw.get("expected_label") not in _ALLOWED_TRACK_LABELS:
                raise ValueError(f"invalid expected label in adjudication {index}")
        elif decision_type == "accepted_hard_negative":
            expected = common | {"source_names", "training_use"}
            if set(raw) != expected:
                raise ValueError(f"invalid accepted_hard_negative adjudication {index}")
            names = raw.get("source_names")
            if not isinstance(names, list) or not names:
                raise ValueError(f"adjudication {index} requires source_names")
            source_names = [_canonical_source_name(str(name)) for name in names]
            if len(source_names) != len(set(source_names)):
                raise ValueError(f"adjudication {index} has duplicate source_names")
            if raw.get("accepted_label") != "A05_controlled_squat":
                raise ValueError(f"adjudication {index} has invalid hard-negative label")
            if raw.get("training_use") != "fall_hard_negative":
                raise ValueError(f"adjudication {index} has invalid training_use")
        else:
            raise ValueError(f"unsupported NTU RGB+D A043 adjudication type: {decision_type!r}")
        if raw.get("accepted_label") not in _ALLOWED_TRACK_LABELS:
            raise ValueError(f"invalid accepted label in adjudication {index}")
        if not all(
            isinstance(raw.get(field), str) and str(raw[field]).strip()
            for field in ("evidence", "reason")
        ):
            raise ValueError(f"adjudication {index} requires evidence and reason")
        normalized = dict(raw)
        if decision_type == "label_override":
            normalized["source_name"] = source_names[0]
        else:
            normalized["source_names"] = sorted(source_names)
        result.append(normalized)
    return result


def _apply_adjudications(
    tasks: Sequence[_RawTask],
    adjudications: Sequence[Mapping[str, Any]],
) -> tuple[list[_RawTask], dict[str, Any]]:
    by_source = {task.source_name: task for task in tasks}
    overrides_applied: list[str] = []
    overrides_confirmed: list[str] = []
    accepted_hard_negatives: set[str] = set()
    decisions_not_in_batch: set[str] = set()
    for decision in adjudications:
        if decision["type"] == "label_override":
            source_name = str(decision["source_name"])
            task = by_source.get(source_name)
            if task is None:
                decisions_not_in_batch.add(source_name)
                continue
            if len(task.tracks) != 1:
                raise ValueError(
                    f"label override requires one track for {source_name}"
                )
            track = task.tracks[0]
            actual = track.get("label") or ""
            expected = str(decision["expected_label"])
            accepted = str(decision["accepted_label"])
            if actual == expected:
                replacement = copy.deepcopy(track)
                replacement.set("label", accepted)
                by_source[source_name] = _RawTask(
                    source_name=task.source_name,
                    size=task.size,
                    width=task.width,
                    height=task.height,
                    tracks=(replacement,),
                    original_label_names=task.original_label_names,
                )
                overrides_applied.append(source_name)
            elif actual == accepted:
                overrides_confirmed.append(source_name)
            else:
                raise ValueError(
                    f"label override mismatch for {source_name}: "
                    f"expected {expected!r} or {accepted!r}, got {actual!r}"
                )
        else:
            accepted_label = str(decision["accepted_label"])
            for source_name in decision["source_names"]:
                source_name = str(source_name)
                task = by_source.get(source_name)
                if task is None:
                    decisions_not_in_batch.add(source_name)
                    continue
                labels = [track.get("label") or "" for track in task.tracks]
                if labels != [accepted_label]:
                    raise ValueError(
                        f"hard-negative adjudication mismatch for {source_name}: {labels}"
                    )
                accepted_hard_negatives.add(source_name)
    unexpected_hard_negatives = sorted(
        task.source_name
        for task in by_source.values()
        if any(
            track.get("label") == "A05_controlled_squat" for track in task.tracks
        )
        and task.source_name not in accepted_hard_negatives
    )
    if unexpected_hard_negatives:
        raise ValueError(
            "A05 controlled-squat tracks lack hard-negative adjudication: "
            f"{unexpected_hard_negatives}"
        )
    return list(by_source.values()), {
        "label_overrides_applied": sorted(overrides_applied),
        "label_overrides_already_present": sorted(overrides_confirmed),
        "accepted_hard_negative_sources": sorted(accepted_hard_negatives),
        "decisions_not_in_batch": sorted(decisions_not_in_batch),
    }


def _read_tasks(
    archives: Sequence[tuple[str, Path]],
    *,
    allow_empty_tasks: bool,
) -> tuple[list[_RawTask], list[dict[str, Any]], Counter[str]]:
    tasks: list[_RawTask] = []
    archive_report: list[dict[str, Any]] = []
    removed_metadata: Counter[str] = Counter()
    for source_group, archive_path in archives:
        raw = archive_path.read_bytes()
        archive_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                invalid_member = archive.testzip()
                if invalid_member is not None:
                    raise ValueError(
                        f"invalid compressed member {invalid_member!r} in {archive_path}"
                    )
                xml_names = [
                    name for name in archive.namelist() if name.lower().endswith(".xml")
                ]
                if len(xml_names) != 1:
                    raise ValueError(
                        f"CVAT archive must contain exactly one XML: {archive_path}"
                    )
                root = ET.fromstring(archive.read(xml_names[0]))
        except (zipfile.BadZipFile, ET.ParseError) as exc:
            raise ValueError(f"invalid CVAT archive: {archive_path}") from exc

        removed_metadata.update(_count_identity_metadata(root))
        task_elements = root.findall("./meta/project/tasks/task")
        is_project = bool(task_elements)
        is_job = False
        if not task_elements:
            single_task = root.find("./meta/task")
            task_elements = [single_task] if single_task is not None else []
        if not task_elements:
            job = root.find("./meta/job")
            if job is not None:
                task_elements = [job]
                is_job = True
        if not task_elements:
            raise ValueError(f"CVAT archive has no task metadata: {archive_path}")
        project_label_names = tuple(
            element.findtext("name") or ""
            for element in (
                root.findall("./meta/job/labels/label")
                if is_job
                else root.findall("./meta/project/labels/label")
            )
        )
        tracks_by_task: dict[str, list[ET.Element]] = defaultdict(list)
        if is_project:
            for track in root.findall("./track"):
                task_id = track.get("task_id")
                if not task_id:
                    raise ValueError(
                        f"project track is missing task_id in {archive_path}"
                    )
                tracks_by_task[task_id].append(track)
        cumulative_offset = 0
        source_names: list[str] = []
        for task_element in task_elements:
            task_id = task_element.findtext("id") or ""
            size = _positive_int(task_element.findtext("size"), "task size")
            start = _non_negative_int(
                task_element.findtext("start_frame") or "0", "task start_frame"
            )
            stop = _non_negative_int(
                task_element.findtext("stop_frame") or str(size - 1),
                "task stop_frame",
            )
            if start != 0 or stop != size - 1:
                raise ValueError(f"CVAT task {task_id} has unsupported frame range")
            source_name = (
                _job_source_name(archive_path)
                if is_job
                else _canonical_source_name(
                    Path(task_element.findtext("source") or "").name
                )
            )
            dimensions = (
                root.find("./meta/original_size")
                if is_job
                else task_element.find("original_size")
            )
            if dimensions is None:
                raise ValueError(f"CVAT task {task_id} is missing original_size")
            width = _positive_int(dimensions.findtext("width"), "source width")
            height = _positive_int(dimensions.findtext("height"), "source height")
            original_tracks = (
                tracks_by_task.get(task_id, []) if is_project else root.findall("./track")
            )
            if not original_tracks and not allow_empty_tasks:
                raise ValueError(f"CVAT task {task_id} has no tracks")
            tracks = tuple(
                _localize_track(track, size=size, frame_offset=cumulative_offset)
                for track in original_tracks
            )
            task_label_names = tuple(
                element.findtext("name") or ""
                for element in task_element.findall("./labels/label")
            )
            tasks.append(
                _RawTask(
                    source_name=source_name,
                    size=size,
                    width=width,
                    height=height,
                    tracks=tracks,
                    original_label_names=task_label_names or project_label_names,
                )
            )
            source_names.append(source_name)
            cumulative_offset += size
        archive_report.append(
            {
                "source_group": source_group,
                "filename": archive_path.name,
                "sha256": archive_sha256,
                "export_format": (
                    "project" if is_project else "job" if is_job else "task"
                ),
                "task_count": len(task_elements),
                "source_names": sorted(source_names),
            }
        )
    return tasks, sorted(
        archive_report, key=lambda item: (item["source_group"], item["filename"])
    ), removed_metadata


def _job_source_name(archive_path: Path) -> str:
    match = _JOB_ARCHIVE_PATTERN.fullmatch(archive_path.stem)
    if match is None:
        raise ValueError(
            "CVAT job archive filename must identify its A043 source: "
            f"{archive_path.name}"
        )
    return _canonical_source_name(f"{match.group('stem')}_rgb.avi")


def _index_unique_tasks(
    tasks: Sequence[_RawTask],
    *,
    role: str,
) -> dict[str, _RawTask]:
    result: dict[str, _RawTask] = {}
    duplicates: list[str] = []
    for task in tasks:
        if task.source_name in result:
            duplicates.append(task.source_name)
        else:
            result[task.source_name] = task
    if duplicates:
        raise ValueError(
            f"duplicate NTU RGB+D A043 {role} sources: {sorted(set(duplicates))}"
        )
    return result


def _localize_track(
    track: ET.Element,
    *,
    size: int,
    frame_offset: int,
) -> ET.Element:
    result = copy.deepcopy(track)
    boxes = result.findall("box")
    if not boxes:
        raise ValueError(f"CVAT track {track.get('id')} has no boxes")
    try:
        frames = [int(box.attrib["frame"]) for box in boxes]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CVAT track {track.get('id')} has an invalid frame") from exc
    local_valid = all(0 <= frame < size for frame in frames)
    global_valid = all(frame_offset <= frame < frame_offset + size for frame in frames)
    if frame_offset == 0 and local_valid:
        global_valid = False
    if local_valid == global_valid:
        raise ValueError(
            f"CVAT track {track.get('id')} frame coordinates are invalid or ambiguous"
        )
    if global_valid:
        for box in boxes:
            box.attrib["frame"] = str(int(box.attrib["frame"]) - frame_offset)
    return result


def _normalize_tasks(
    raw_tasks: Sequence[_RawTask],
    manifest_by_source: Mapping[str, VideoMetadata],
    label_config: Sequence[Mapping[str, Any]],
) -> tuple[ET.Element, dict[str, Any]]:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    project = ET.SubElement(ET.SubElement(root, "meta"), "project")
    ET.SubElement(project, "id").text = NTU_RGBD_A043_BATCH_ID
    ET.SubElement(project, "name").text = "fall_risk__ntu_rgbd__a043_cvat_review"
    tasks_element = ET.SubElement(project, "tasks")
    _append_label_config(ET.SubElement(project, "labels"), label_config)

    protocol_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter()
    source_dimensions: Counter[str] = Counter()
    target_dimensions: Counter[str] = Counter()
    scale_factors: Counter[str] = Counter()
    event_views: dict[
        str, list[tuple[str, str | None, int | None]]
    ] = defaultdict(list)
    next_track_id = 0
    cumulative_frame_offset = 0

    for task_index, raw_task in enumerate(
        sorted(raw_tasks, key=lambda item: item.source_name), 1
    ):
        metadata = manifest_by_source[raw_task.source_name]
        if raw_task.size != metadata.frame_count:
            raise ValueError(
                f"frame count mismatch for {raw_task.source_name}: "
                f"CVAT={raw_task.size}, manifest={metadata.frame_count}"
            )
        manifest_width = _positive_int(
            metadata.manifest_record.get("width"), "manifest width"
        )
        manifest_height = _positive_int(
            metadata.manifest_record.get("height"), "manifest height"
        )
        scale_x = manifest_width / raw_task.width
        scale_y = manifest_height / raw_task.height
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"non-isotropic media transform for {raw_task.source_name}: "
                f"scale_x={scale_x}, scale_y={scale_y}"
            )
        protocol, fall_label, fall_start = _classify_protocol(raw_task)
        protocol_counts[protocol] += 1
        source_dimensions[f"{raw_task.width}x{raw_task.height}"] += 1
        target_dimensions[f"{manifest_width}x{manifest_height}"] += 1
        scale_factors[_format_scale(scale_x)] += 1
        original_event_id = str(
            metadata.manifest_record.get("original_event_id")
            or metadata.video_id.rsplit("_c", 1)[0]
        )
        event_views[original_event_id].append((metadata.view, fall_label, fall_start))

        task_id = str(task_index)
        task_element = ET.SubElement(tasks_element, "task")
        for tag, value in (
            ("id", task_id),
            ("name", f"fall_risk__ntu_rgbd__{metadata.manifest_record.get('subset')}__{metadata.video_id}"),
            ("size", str(raw_task.size)),
            ("mode", "interpolation"),
            ("overlap", "0"),
            ("subset", "default"),
            ("start_frame", "0"),
            ("stop_frame", str(raw_task.size - 1)),
        ):
            ET.SubElement(task_element, tag).text = value
        original_size = ET.SubElement(task_element, "original_size")
        ET.SubElement(original_size, "width").text = str(manifest_width)
        ET.SubElement(original_size, "height").text = str(manifest_height)
        ET.SubElement(task_element, "source").text = Path(metadata.path).name

        for raw_track in raw_task.tracks:
            raw_label = raw_track.get("label") or ""
            if raw_label not in _ALLOWED_TRACK_LABELS:
                raise ValueError(
                    f"unsupported NTU RGB+D A043 track label: {raw_label!r}"
                )
            raw_label_counts[raw_label] += 1
            track = copy.deepcopy(raw_track)
            track.attrib["id"] = str(next_track_id)
            track.attrib["task_id"] = task_id
            track.attrib["source"] = "manual"
            track.attrib["frame_coordinate_system"] = "project_global"
            next_track_id += 1
            note = (
                _WHOLE_CLIP_NOTE
                if protocol == "whole_clip_fall_without_onset"
                else _required_uncertain_note(raw_track, raw_task.source_name)
                if protocol == "whole_clip_uncertain"
                else _HARD_NEGATIVE_NOTE
                if protocol == "whole_clip_controlled_squat_hard_negative"
                else _PENDING_REVIEW_NOTE
            )
            for box in track.findall("box"):
                _scale_box(box, scale_x, scale_y, manifest_width, manifest_height)
                box.attrib["frame"] = str(
                    int(box.attrib["frame"]) + cumulative_frame_offset
                )
                _set_box_attribute(box, "target_subject", metadata.subject_id)
                _set_box_attribute(box, "note", note)
            root.append(track)
        cumulative_frame_offset += raw_task.size

    cross_view = _cross_view_report(event_views)
    return root, {
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "source_dimensions": dict(sorted(source_dimensions.items())),
        "target_dimensions": dict(sorted(target_dimensions.items())),
        "scale_factors": dict(sorted(scale_factors.items())),
        "cross_view_qc": cross_view,
    }


def _classify_protocol(raw_task: _RawTask) -> tuple[str, str | None, int | None]:
    intervals: dict[str, tuple[int, int]] = {}
    coverage: Counter[int] = Counter()
    for track in raw_task.tracks:
        label = track.get("label") or ""
        if label not in _ALLOWED_TRACK_LABELS:
            raise ValueError(f"unsupported NTU RGB+D A043 track label: {label!r}")
        if label in intervals:
            raise ValueError(f"duplicate action track {label} for {raw_task.source_name}")
        active_frames = sorted(
            int(box.attrib["frame"])
            for box in track.findall("box")
            if box.get("outside") != "1"
        )
        if not active_frames:
            raise ValueError(f"empty action track {label} for {raw_task.source_name}")
        if active_frames != list(range(active_frames[0], active_frames[-1] + 1)):
            raise ValueError(f"non-contiguous action track for {raw_task.source_name}")
        intervals[label] = (active_frames[0], active_frames[-1])
        coverage.update(active_frames)
    ordered = sorted(
        ((start, end, label) for label, (start, end) in intervals.items()),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if ordered[0][0] != 0 or any(
        previous[1] + 1 != current[0]
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            f"unsupported action-boundary protocol for {raw_task.source_name}: {intervals}"
        )
    if any(count > 1 for count in coverage.values()):
        raise ValueError(f"overlapping action tracks for {raw_task.source_name}")
    final_end = ordered[-1][1]
    has_trailing_outside = any(
        box.get("outside") == "1"
        and int(box.attrib["frame"]) == raw_task.size - 1
        for track in raw_task.tracks
        for box in track.findall("box")
    )
    if final_end == raw_task.size - 1:
        trailing_outside = False
    elif final_end == raw_task.size - 2 and has_trailing_outside:
        trailing_outside = True
    else:
        raise ValueError(
            f"action tracks do not cover the clip for {raw_task.source_name}"
        )
    if len(coverage) != final_end + 1:
        raise ValueError(f"action tracks have gaps for {raw_task.source_name}")
    labels = tuple(item[2] for item in ordered)
    fall_labels = [label for label in labels if label in _FALL_TRACK_LABELS]
    if len(fall_labels) > 1:
        raise ValueError(f"expected at most one fall track for {raw_task.source_name}")
    fall_label = fall_labels[0] if fall_labels else None
    fall_start = intervals[fall_label][0] if fall_label else None
    if len(labels) == 1 and fall_label is not None and fall_start == 0:
        protocol = "whole_clip_fall_without_onset"
    elif labels == ("U01_unable_to_judge",):
        protocol = "whole_clip_uncertain"
    elif labels == ("A05_controlled_squat",):
        protocol = "whole_clip_controlled_squat_hard_negative"
    elif (
        labels == ("A01_normal_walk", fall_label)
        and fall_label is not None
    ):
        protocol = (
            "segmented_normal_to_fall_with_trailing_outside"
            if trailing_outside
            else "segmented_normal_to_fall"
        )
    elif labels == (
        "A01_normal_walk",
        fall_label,
        "A04_normal_sit_to_stand",
    ) and fall_label is not None:
        protocol = "segmented_normal_to_fall_to_sit_to_stand"
    elif labels == ("A01_normal_walk", "A07_controlled_lie_down"):
        protocol = "segmented_nonfall_controlled_lie_down"
    else:
        raise ValueError(
            f"unsupported action-boundary protocol for {raw_task.source_name}: {intervals}"
        )
    return protocol, fall_label, fall_start


def _required_uncertain_note(track: ET.Element, source_name: str) -> str:
    notes = {
        (attribute.text or "").strip()
        for box in track.findall("box")
        for attribute in box.findall("attribute")
        if attribute.get("name") == "note" and (attribute.text or "").strip()
    }
    if len(notes) != 1:
        raise ValueError(
            f"U01 track requires exactly one non-empty reason for {source_name}"
        )
    return next(iter(notes))


def _append_label_config(
    labels_element: ET.Element,
    label_config: Sequence[Mapping[str, Any]],
) -> None:
    for definition in label_config:
        label = ET.SubElement(labels_element, "label")
        ET.SubElement(label, "name").text = str(definition["name"])
        ET.SubElement(label, "type").text = str(definition.get("type") or "rectangle")
        attributes = ET.SubElement(label, "attributes")
        for raw_attribute in definition.get("attributes", []):
            attribute = ET.SubElement(attributes, "attribute")
            for tag in ("name", "mutable", "input_type", "default_value"):
                value = raw_attribute.get(tag, "")
                if isinstance(value, bool):
                    value = str(value)
                ET.SubElement(attribute, tag).text = str(value)
            values = raw_attribute.get("values", [])
            ET.SubElement(attribute, "values").text = "\n".join(
                str(value) for value in values
            )


def _scale_box(
    box: ET.Element,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> None:
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
        box.attrib[key] = f"{value:.2f}"


def _set_box_attribute(box: ET.Element, name: str, value: str) -> None:
    matches = [
        child for child in box.findall("attribute") if child.get("name") == name
    ]
    if len(matches) > 1:
        raise ValueError(f"CVAT box has duplicate {name} attributes")
    target = matches[0] if matches else ET.SubElement(box, "attribute", {"name": name})
    target.text = value


def _cross_view_report(
    event_views: Mapping[str, Sequence[tuple[str, str | None, int | None]]],
) -> dict[str, Any]:
    complete = [views for views in event_views.values() if len(views) == 3]
    complete_fall = [
        views for views in complete if all(label is not None for _, label, _ in views)
    ]
    complete_nonfall = [
        views for views in complete if all(label is None for _, label, _ in views)
    ]
    fall_coverage_conflicts = len(complete) - len(complete_fall) - len(complete_nonfall)
    direction_disagreement = sum(
        len({label for _, label, _ in views}) > 1 for views in complete_fall
    )
    boundary_deltas = [
        max(start for _, _, start in views if start is not None)
        - min(start for _, _, start in views if start is not None)
        for views in complete_fall
    ]
    return {
        "physical_event_groups": len(event_views),
        "complete_three_view_groups": len(complete),
        "incomplete_view_groups": len(event_views) - len(complete),
        "complete_fall_groups": len(complete_fall),
        "complete_nonfall_groups": len(complete_nonfall),
        "fall_coverage_conflict_groups": fall_coverage_conflicts,
        "direction_disagreement_groups": direction_disagreement,
        "boundary_delta_gt_2_frames": sum(delta > 2 for delta in boundary_deltas),
        "boundary_delta_gt_5_frames": sum(delta > 5 for delta in boundary_deltas),
        "maximum_boundary_delta_frames": max(boundary_deltas, default=0),
    }


def _label_schema_report(raw_tasks: Sequence[_RawTask]) -> list[dict[str, Any]]:
    variants: Counter[tuple[str, ...]] = Counter(
        task.original_label_names for task in raw_tasks
    )
    return [
        {
            "task_count": count,
            "label_count": len(names),
            "label_names": list(names),
        }
        for names, count in sorted(
            variants.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]


def _count_identity_metadata(root: ET.Element) -> Counter[str]:
    counts: Counter[str] = Counter()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in _IDENTITY_TAGS:
            counts[tag] += 1
    return counts


def _deterministic_zip(root: ET.Element) -> bytes:
    xml_payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, xml_payload)
    return buffer.getvalue()


def _write_batch(
    source_payload: bytes,
    report: dict[str, Any],
    *,
    manifest_path: Path,
    output_dir: Path,
    labeler: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    backup_dir: Path | None = None
    try:
        staged_source = staging_dir / "source_annotations.zip"
        staged_source.write_bytes(source_payload)
        converted = convert_cvat_xml(
            staged_source,
            manifest_path=manifest_path,
            fps=None,
            labeler=labeler,
        )
        final_source_path = _portable_path(output_dir / "source_annotations.zip")
        actions = _replace_source_path(converted.action_labels, final_source_path)
        events = _replace_source_path(converted.event_labels, final_source_path)
        write_converted_fall_labels(
            ConvertedFallLabels(action_labels=actions, event_labels=events),
            action_output_path=staging_dir / "action_labels.jsonl",
            event_output_path=staging_dir / "event_labels.jsonl",
        )
        final_report = dict(report)
        final_report["output_counts"] = {
            "action_labels": len(actions),
            "event_labels": len(events),
        }
        final_report["outputs"] = {
            "source_annotations": final_source_path,
            "action_labels": _portable_path(output_dir / "action_labels.jsonl"),
            "event_labels": _portable_path(output_dir / "event_labels.jsonl"),
            "import_report": _portable_path(output_dir / "import_report.json"),
        }
        (staging_dir / "import_report.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            backup_dir = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}"
            if backup_dir.exists():
                raise FileExistsError(backup_dir)
            os.replace(output_dir, backup_dir)
        os.replace(staging_dir, output_dir)
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
        return final_report
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _replace_source_path(
    rows: Sequence[Mapping[str, Any]], source_path: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        normalized["source_annotation_path"] = source_path
        result.append(normalized)
    return result


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_source_match(source_name: str) -> re.Match[str]:
    match = _SOURCE_PATTERN.fullmatch(source_name)
    if match is None:
        raise ValueError(f"invalid NTU RGB+D A043 CVAT source name: {source_name!r}")
    return match


def _canonical_source_name(source_name: str) -> str:
    _required_source_match(source_name)
    return f"{Path(source_name).stem}.mp4"


def _positive_int(value: object, field: str) -> int:
    result = _non_negative_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _non_negative_int(value: object, field: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _format_scale(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
