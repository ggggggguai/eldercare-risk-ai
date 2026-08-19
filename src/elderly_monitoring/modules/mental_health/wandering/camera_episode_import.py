"""Truth-separated boundary import for episode-first camera development."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)


CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION = "wandering-camera-episode-boundary-v1"
CAMERA_EPISODE_TRUTH_SCHEMA_VERSION = "wandering-camera-episode-truth-v1"
CAMERA_EPISODE_IMPORT_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-import-summary-v1"
)

BOUNDARY_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "source_video_id",
        "target_track_id",
        "start_sec",
        "end_sec_exclusive",
        "boundary_source",
        "boundary_status",
        "boundary_reason_codes",
        "cvat_track_id",
    }
)
TRUTH_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "source_video_id",
        "target_track_id",
        "cvat_track_id",
        "start_sec",
        "end_sec_exclusive",
        "observable_pattern",
        "purpose_context",
        "purpose_evidence",
        "evaluation_role",
        "script_type",
        "visibility_quality",
        "tracking_issue",
        "annotation_status",
        "note",
    }
)
BOUNDARY_SOURCES = frozenset({"simplified_jsonl", "cvat_xml", "whole_clip"})
BOUNDARY_STATUSES = frozenset({"ready", "boundary_uncertain"})
_CVAT_ATTRIBUTES = frozenset(
    {
        "observable_pattern",
        "purpose_context",
        "purpose_evidence",
        "evaluation_role",
        "script_type",
        "visibility_quality",
        "tracking_issue",
        "note",
    }
)
_PATTERNS = frozenset({"direct", "pacing", "lapping", "random", "unknown"})
_PURPOSES = frozenset({"purposeful", "nonpurposeful", "unknown"})
_PURPOSE_EVIDENCE = frozenset(
    {"scripted", "participant_report", "observed_context", "unknown"}
)
_ROLES = frozenset(
    {
        "wandering_like_positive",
        "purposeful_hard_negative",
        "ordinary_negative",
        "uncertain",
        "excluded",
    }
)
_VISIBILITY = frozenset({"good", "partial", "poor"})
_TRACKING_ISSUES = frozenset(
    {
        "not_checked",
        "none",
        "occlusion",
        "out_of_frame",
        "fragmented_track",
        "id_switch",
        "wrong_target",
        "multiple_issues",
    }
)


class CameraEpisodeImportError(ValueError):
    """CVAT or simplified boundary input is not a valid EP1A source."""


@dataclass(frozen=True)
class CvatEpisodeImport:
    boundaries: tuple[dict[str, Any], ...]
    truth_records: tuple[dict[str, Any], ...]


def validate_episode_boundary(
    value: Mapping[str, Any], media: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact truth-free episode boundary view."""

    if not isinstance(value, Mapping) or frozenset(value) != BOUNDARY_FIELDS:
        raise CameraEpisodeImportError("episode boundary fields must match the exact v1 set")
    row = dict(value)
    if row["schema_version"] != CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION:
        raise CameraEpisodeImportError("episode boundary schema_version is invalid")
    row["episode_id"] = _nonempty_text(row["episode_id"], "episode_id")
    row["source_video_id"] = _nonempty_text(
        row["source_video_id"], "source_video_id"
    )
    if row["source_video_id"] != media.get("source_video_id"):
        raise CameraEpisodeImportError("episode boundary source_video_id mismatch")
    row["target_track_id"] = _nonnegative_int(
        row["target_track_id"], "target_track_id"
    )
    row["start_sec"] = _finite_nonnegative(row["start_sec"], "start_sec")
    row["end_sec_exclusive"] = _finite_nonnegative(
        row["end_sec_exclusive"], "end_sec_exclusive"
    )
    duration = _finite_positive(media.get("duration_sec"), "media.duration_sec")
    if not row["start_sec"] < row["end_sec_exclusive"] <= duration + 1e-9:
        raise CameraEpisodeImportError("episode boundary interval is outside the video")
    row["end_sec_exclusive"] = min(row["end_sec_exclusive"], duration)
    if not isinstance(row["boundary_source"], str) or row[
        "boundary_source"
    ] not in BOUNDARY_SOURCES:
        raise CameraEpisodeImportError("episode boundary_source is invalid")
    if not isinstance(row["boundary_status"], str) or row[
        "boundary_status"
    ] not in BOUNDARY_STATUSES:
        raise CameraEpisodeImportError("episode boundary_status is invalid")
    reasons = row["boundary_reason_codes"]
    if not isinstance(reasons, list) or any(
        not isinstance(item, str) or not item for item in reasons
    ):
        raise CameraEpisodeImportError("boundary_reason_codes must be a string list")
    if (row["boundary_status"] == "ready") != (not reasons):
        raise CameraEpisodeImportError(
            "ready boundaries need no reasons and uncertain boundaries need reasons"
        )
    cvat_track_id = row["cvat_track_id"]
    if cvat_track_id is not None:
        row["cvat_track_id"] = _nonempty_text(cvat_track_id, "cvat_track_id")
    if row["boundary_source"] == "cvat_xml" and cvat_track_id is None:
        raise CameraEpisodeImportError("CVAT boundary requires cvat_track_id")
    if row["boundary_source"] != "cvat_xml" and cvat_track_id is not None:
        raise CameraEpisodeImportError("non-CVAT boundary must not claim cvat_track_id")
    return row


def load_episode_boundaries(
    path: str | Path, media: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load simplified or imported boundaries without accepting truth fields."""

    boundary_path = Path(path)
    try:
        lines = boundary_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodeImportError("cannot read episode boundary JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, CameraEpisodeImportError) as exc:
            raise CameraEpisodeImportError(
                f"invalid episode boundary JSON at line {line_number}"
            ) from exc
        rows.append(validate_episode_boundary(value, media))
    if not rows:
        raise CameraEpisodeImportError("episode boundary JSONL is empty")
    _validate_boundary_collection(rows)
    return sorted(
        rows,
        key=lambda row: (
            str(row["source_video_id"]),
            int(row["target_track_id"]),
            float(row["start_sec"]),
            str(row["episode_id"]),
        ),
    )


def validate_episode_truth(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one independent v1 truth row without consulting predictions."""

    if not isinstance(value, Mapping) or frozenset(value) != TRUTH_FIELDS:
        raise CameraEpisodeImportError("episode truth fields must match the exact v1 set")
    row = dict(value)
    if row["schema_version"] != CAMERA_EPISODE_TRUTH_SCHEMA_VERSION:
        raise CameraEpisodeImportError("episode truth schema_version is invalid")
    row["episode_id"] = _nonempty_text(row["episode_id"], "episode_id")
    row["source_video_id"] = _nonempty_text(
        row["source_video_id"], "source_video_id"
    )
    row["target_track_id"] = _nonnegative_int(
        row["target_track_id"], "target_track_id"
    )
    if row["cvat_track_id"] is not None:
        row["cvat_track_id"] = _nonempty_text(
            row["cvat_track_id"], "cvat_track_id"
        )
    row["start_sec"] = _finite_nonnegative(row["start_sec"], "start_sec")
    row["end_sec_exclusive"] = _finite_nonnegative(
        row["end_sec_exclusive"], "end_sec_exclusive"
    )
    if row["start_sec"] >= row["end_sec_exclusive"]:
        raise CameraEpisodeImportError("episode truth interval must be positive")
    _validate_truth_attributes(
        {name: row[name] for name in _CVAT_ATTRIBUTES}
    )
    if not isinstance(row["annotation_status"], str) or row[
        "annotation_status"
    ] not in {"accepted", "uncertain", "excluded"}:
        raise CameraEpisodeImportError("episode truth annotation_status is invalid")
    expected_status = {
        "uncertain": "uncertain",
        "excluded": "excluded",
    }.get(row["evaluation_role"], "accepted")
    if row["annotation_status"] != expected_status:
        raise CameraEpisodeImportError(
            "episode truth annotation_status/evaluation_role are inconsistent"
        )
    for field in ("script_type", "note"):
        if not isinstance(row[field], str) or any(
            char in row[field] for char in "\r\n\0"
        ):
            raise CameraEpisodeImportError(f"episode truth {field} is invalid")
    return row


def load_episode_truth(path: str | Path) -> list[dict[str, Any]]:
    """Load independent episode truth while preserving human-entered semantics."""

    truth_path = Path(path)
    try:
        lines = truth_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodeImportError("cannot read episode truth JSONL") from exc
    rows: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
            row = validate_episode_truth(value)
        except (json.JSONDecodeError, CameraEpisodeImportError) as exc:
            raise CameraEpisodeImportError(
                f"invalid episode truth JSON at line {line_number}"
            ) from exc
        if row["episode_id"] in episode_ids:
            raise CameraEpisodeImportError("duplicate episode_id in episode truth")
        episode_ids.add(row["episode_id"])
        rows.append(row)
    if not rows:
        raise CameraEpisodeImportError("episode truth JSONL is empty")
    return sorted(
        rows,
        key=lambda row: (
            str(row["source_video_id"]),
            int(row["target_track_id"]),
            float(row["start_sec"]),
            str(row["episode_id"]),
        ),
    )


def import_cvat_episode_xml(
    path: str | Path,
    media: Mapping[str, Any],
    *,
    target_track_id: int,
) -> CvatEpisodeImport:
    """Import CVAT for video 1.1 tracks into separate boundary/truth views."""

    track_id = _nonnegative_int(target_track_id, "target_track_id")
    fps = _finite_positive(media.get("nominal_fps"), "media.nominal_fps")
    duration = _finite_positive(media.get("duration_sec"), "media.duration_sec")
    source_video_id = _nonempty_text(
        media.get("source_video_id"), "media.source_video_id"
    )
    try:
        root = ET.parse(Path(path)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise CameraEpisodeImportError("cannot parse CVAT XML") from exc
    if root.tag != "annotations" or root.findtext("./version") != "1.1":
        raise CameraEpisodeImportError("CVAT XML must use annotations version 1.1")
    size_text = root.findtext("./meta/job/size")
    video_frame_count = None
    if size_text is not None:
        try:
            video_frame_count = int(size_text)
        except ValueError as exc:
            raise CameraEpisodeImportError("CVAT job size is invalid") from exc
        if video_frame_count <= 0:
            raise CameraEpisodeImportError("CVAT job size must be positive")

    boundaries: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    tracks = root.findall("./track")
    if not tracks:
        raise CameraEpisodeImportError("CVAT XML contains no episode tracks")
    for track in tracks:
        if track.get("label") != "wandering_episode":
            raise CameraEpisodeImportError("CVAT XML contains a non-episode track")
        cvat_track_id = _nonempty_text(track.get("id"), "cvat track id")
        boxes = list(track.findall("./box"))
        if not boxes:
            raise CameraEpisodeImportError("CVAT episode track contains no boxes")
        parsed_boxes = sorted(
            ((_frame(box), _outside(box), box) for box in boxes),
            key=lambda item: item[0],
        )
        if parsed_boxes[0][1]:
            raise CameraEpisodeImportError("CVAT episode must start with outside=0")
        visible = [item for item in parsed_boxes if not item[1]]
        if not visible:
            raise CameraEpisodeImportError("CVAT episode contains no visible frame")
        start_frame = visible[0][0]
        outside_frames = [frame for frame, outside, _box in parsed_boxes if outside]
        if outside_frames:
            end_frame_exclusive = min(outside_frames)
            if any(frame >= end_frame_exclusive for frame, _outside, _box in visible):
                raise CameraEpisodeImportError(
                    "CVAT visible frame appears at or after the outside boundary"
                )
        else:
            end_frame_exclusive = visible[-1][0] + 1
        if video_frame_count is not None and (
            start_frame >= video_frame_count
            or end_frame_exclusive > video_frame_count
        ):
            raise CameraEpisodeImportError("CVAT episode frame exceeds job size")
        start_sec = start_frame / fps
        end_sec = min(end_frame_exclusive / fps, duration)
        if not start_sec < end_sec:
            raise CameraEpisodeImportError("CVAT episode has an empty time interval")
        attribute_sets = [
            _box_attributes(box)
            for _frame_value, _outside_value, box in visible
            if box.findall("./attribute")
        ]
        if not attribute_sets:
            raise CameraEpisodeImportError("CVAT episode attributes are missing")
        attributes = attribute_sets[0]
        if any(value != attributes for value in attribute_sets[1:]):
            raise CameraEpisodeImportError(
                "CVAT episode attributes change inside one track"
            )
        _validate_truth_attributes(attributes)
        episode_id = f"{source_video_id}-cvat-{cvat_track_id}"
        boundary = validate_episode_boundary(
            {
                "schema_version": CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
                "episode_id": episode_id,
                "source_video_id": source_video_id,
                "target_track_id": track_id,
                "start_sec": start_sec,
                "end_sec_exclusive": end_sec,
                "boundary_source": "cvat_xml",
                "boundary_status": "ready",
                "boundary_reason_codes": [],
                "cvat_track_id": cvat_track_id,
            },
            media,
        )
        annotation_status = {
            "uncertain": "uncertain",
            "excluded": "excluded",
        }.get(attributes["evaluation_role"], "accepted")
        truth = validate_episode_truth({
            "schema_version": CAMERA_EPISODE_TRUTH_SCHEMA_VERSION,
            "episode_id": episode_id,
            "source_video_id": source_video_id,
            "target_track_id": track_id,
            "cvat_track_id": cvat_track_id,
            "start_sec": start_sec,
            "end_sec_exclusive": end_sec,
            **attributes,
            "annotation_status": annotation_status,
        })
        boundaries.append(boundary)
        truths.append(truth)
    _validate_boundary_collection(boundaries)
    ordering = sorted(
        range(len(boundaries)),
        key=lambda index: (
            float(boundaries[index]["start_sec"]),
            str(boundaries[index]["episode_id"]),
        ),
    )
    return CvatEpisodeImport(
        boundaries=tuple(boundaries[index] for index in ordering),
        truth_records=tuple(truths[index] for index in ordering),
    )


def build_cvat_episode_import_bundle(
    *,
    cvat_xml_path: str | Path,
    media: Mapping[str, Any],
    target_track_id: int,
    output_dir: str | Path,
) -> CvatEpisodeImport:
    """Write a fresh import directory while keeping truth out of boundary bytes."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"episode import output already exists: {output}")
    result = import_cvat_episode_xml(
        cvat_xml_path,
        media,
        target_track_id=target_track_id,
    )
    summary = {
        "schema_version": CAMERA_EPISODE_IMPORT_SUMMARY_SCHEMA_VERSION,
        "status": "oracle_boundary_import_ready",
        "source_video_id": media["source_video_id"],
        "target_track_id": target_track_id,
        "episode_count": len(result.boundaries),
        "boundary_truth_views_separated": True,
        "model_predictions_consumed": False,
    }
    _commit_new_directory(
        output,
        {
            "episode_boundaries.jsonl": canonical_jsonl_bytes(result.boundaries),
            "episode_truth.jsonl": canonical_jsonl_bytes(result.truth_records),
            "import_summary.json": canonical_json_bytes(summary),
        },
    )
    return result


def _validate_boundary_collection(rows: Sequence[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    intervals: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in ids:
            raise CameraEpisodeImportError("duplicate episode_id is forbidden")
        ids.add(episode_id)
        key = (str(row["source_video_id"]), int(row["target_track_id"]))
        interval = (float(row["start_sec"]), float(row["end_sec_exclusive"]))
        if any(
            interval[0] < prior[1] and interval[1] > prior[0]
            for prior in intervals.setdefault(key, [])
        ):
            raise CameraEpisodeImportError("overlapping episode boundaries are forbidden")
        intervals[key].append(interval)


def _validate_truth_attributes(value: Mapping[str, str]) -> None:
    if frozenset(value) != _CVAT_ATTRIBUTES:
        raise CameraEpisodeImportError("CVAT episode attributes must match the v1 set")
    if any(not isinstance(value[name], str) for name in _CVAT_ATTRIBUTES):
        raise CameraEpisodeImportError("CVAT episode attributes must be strings")
    if any(item == "not_set" for item in value.values()):
        raise CameraEpisodeImportError("CVAT episode has an unconfirmed not_set attribute")
    if value["observable_pattern"] not in _PATTERNS:
        raise CameraEpisodeImportError("CVAT observable_pattern is invalid")
    if value["purpose_context"] not in _PURPOSES:
        raise CameraEpisodeImportError("CVAT purpose_context is invalid")
    if value["purpose_evidence"] not in _PURPOSE_EVIDENCE:
        raise CameraEpisodeImportError("CVAT purpose_evidence is invalid")
    if (value["purpose_context"] == "unknown") != (
        value["purpose_evidence"] == "unknown"
    ):
        raise CameraEpisodeImportError("CVAT purpose context/evidence are inconsistent")
    if value["evaluation_role"] not in _ROLES:
        raise CameraEpisodeImportError("CVAT evaluation_role is invalid")
    if value["visibility_quality"] not in _VISIBILITY:
        raise CameraEpisodeImportError("CVAT visibility_quality is invalid")
    if value["tracking_issue"] not in _TRACKING_ISSUES:
        raise CameraEpisodeImportError("CVAT tracking_issue is invalid")
    if value["observable_pattern"] == "unknown" and value["evaluation_role"] not in {
        "uncertain",
        "excluded",
    }:
        raise CameraEpisodeImportError("unknown CVAT shape cannot be accepted")
    if value["evaluation_role"] == "purposeful_hard_negative" and (
        value["purpose_context"] != "purposeful"
        or value["observable_pattern"] == "direct"
    ):
        raise CameraEpisodeImportError("purposeful hard negative shape is inconsistent")


def _box_attributes(box: ET.Element) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in box.findall("./attribute"):
        name = item.get("name")
        if name is None or name in output:
            raise CameraEpisodeImportError("CVAT episode attribute name is invalid")
        output[name] = item.text or ""
    return output


def _frame(box: ET.Element) -> int:
    try:
        value = int(box.get("frame", ""))
    except ValueError as exc:
        raise CameraEpisodeImportError("CVAT frame is invalid") from exc
    if value < 0:
        raise CameraEpisodeImportError("CVAT frame must be non-negative")
    return value


def _outside(box: ET.Element) -> bool:
    value = box.get("outside", "0")
    if value not in {"0", "1"}:
        raise CameraEpisodeImportError("CVAT outside must be 0 or 1")
    return value == "1"


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"episode import output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
        raise CameraEpisodeImportError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CameraEpisodeImportError(f"{field} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraEpisodeImportError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise CameraEpisodeImportError(f"{field} must be finite and non-negative")
    return number


def _finite_positive(value: Any, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number <= 0.0:
        raise CameraEpisodeImportError(f"{field} must be positive")
    return number


def _reject_json_constant(value: str) -> None:
    raise CameraEpisodeImportError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "BOUNDARY_FIELDS",
    "CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION",
    "CAMERA_EPISODE_IMPORT_SUMMARY_SCHEMA_VERSION",
    "CAMERA_EPISODE_TRUTH_SCHEMA_VERSION",
    "CameraEpisodeImportError",
    "CvatEpisodeImport",
    "build_cvat_episode_import_bundle",
    "import_cvat_episode_xml",
    "load_episode_boundaries",
    "load_episode_truth",
    "validate_episode_boundary",
    "validate_episode_truth",
]
