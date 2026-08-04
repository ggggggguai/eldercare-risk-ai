from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    NEAR_FALL_HARD_NEGATIVES,
    _frame_time,
    _sample_group_id,
    _source_group,
    _stable_id,
    read_jsonl_strict,
)


EVENT_SCHEMA_VERSION = "fall-risk-event-label-v3"
MANUAL_DECISION_SCHEMA_VERSION = "near-fall-manual-decision-v1"
MANUAL_DECISION_SOURCE_TYPE = "manual_near_fall_v1"
_POSITIVE_REVIEW_STATUSES = {"double_reviewed", "adjudicated"}
_NEGATIVE_REVIEW_STATUSES = {"single_reviewed", "double_reviewed", "adjudicated"}
_SEVERE_FLAGS = {"heavy_occlusion", "off_screen", "multi_person_uncertain", "camera_cut"}
_EVENT_SUBTYPES = {
    "stumble_recovery",
    "rapid_support_recovery",
    "rapid_body_drop_recovery",
    "sudden_stop_recovery",
    "other_recovery",
}
_CONTACT_EVIDENCE = {"observed", "proxy", "unknown", "not_applicable"}
_BOUNDARY_PRECISIONS = {"exact", "approximate", "unknown"}
_QUALITY_FLAGS = {
    "partial_occlusion",
    "heavy_occlusion",
    "low_light",
    "motion_blur",
    "low_resolution",
    "off_screen",
    "multi_person_uncertain",
    "camera_cut",
}
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.:-]+$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def publish_near_fall_manual_labels_v3(
    *,
    base_event_labels: str | Path,
    decisions: str | Path,
    manifest: str | Path,
    output_event_labels: str | Path,
    report: str | Path,
    repo_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish human near-fall decisions as an auditable v3 candidate file.

    This function never infers a label from video, action names, directories or
    rule output. Every appended row must be an explicit human decision.
    """

    base_path = Path(base_event_labels)
    decision_path = Path(decisions)
    manifest_path = Path(manifest)
    output_path = Path(output_event_labels)
    report_path = Path(report)
    root = Path(repo_root).resolve()
    for path in (base_path, decision_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"near-fall publication input not found: {path}")
    if output_path.resolve() == base_path.resolve():
        raise ValueError(
            "near-fall publication output must be a separate candidate file; "
            "review it before replacing the v3 fact source"
        )
    input_paths = {path.resolve() for path in (base_path, decision_path, manifest_path)}
    if output_path.resolve() in input_paths or report_path.resolve() in input_paths:
        raise ValueError("near-fall publication outputs must not overwrite an input file")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("near-fall publication dataset and report paths must differ")
    if not overwrite and (output_path.exists() or report_path.exists()):
        existing = output_path if output_path.exists() else report_path
        raise FileExistsError(f"near-fall publication output already exists: {existing}")

    base_rows = read_jsonl_strict(base_path)
    decision_rows = read_jsonl_strict(decision_path)
    manifest_index = _index_manifest(read_jsonl_strict(manifest_path))
    existing_ids = {str(row.get("label_id")) for row in base_rows}
    existing_physical_events = {
        str(row["physical_event_id"])
        for row in base_rows
        if row.get("task_type") == "near_fall_event"
        and row.get("label_role") == "positive"
        and isinstance(row.get("physical_event_id"), str)
    }
    existing_decisions = {
        str(reference.get("source_label_id"))
        for row in base_rows
        for reference in row.get("source_refs", [])
        if isinstance(reference, Mapping)
        and reference.get("source_type") == "manual_v3"
    }
    additions: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    seen_labels: set[str] = set()
    seen_physical_events: set[str] = set()
    for decision in decision_rows:
        normalized = _build_event_row(
            decision,
            manifest_index=manifest_index,
            repo_root=root,
        )
        decision_id = str(normalized["_decision_id"])
        label_id = str(normalized["label_id"])
        if decision_id in existing_decisions or decision_id in seen_decisions:
            raise ValueError(f"duplicate near-fall manual decision: {decision_id}")
        if label_id in existing_ids or label_id in seen_labels:
            raise ValueError(f"near-fall v3 label_id collision: {label_id}")
        physical_event_id = normalized.get("physical_event_id")
        if isinstance(physical_event_id, str) and (
            physical_event_id in existing_physical_events
            or physical_event_id in seen_physical_events
        ):
            raise ValueError(f"duplicate near-fall physical_event_id: {physical_event_id}")
        seen_decisions.add(decision_id)
        seen_labels.add(label_id)
        if isinstance(physical_event_id, str):
            seen_physical_events.add(physical_event_id)
        normalized.pop("_decision_id", None)
        additions.append(normalized)

    combined = [*base_rows, *additions]
    combined.sort(key=_label_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(output_path, combined)
    publication = {
        "schema_version": "near-fall-manual-publication-v1",
        "status": "candidate_requires_v3_validation",
        "task_type": "near_fall_event",
        "base_event_labels_sha256": _sha256_file(base_path),
        "decisions_sha256": _sha256_file(decision_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "output_event_labels_sha256": _sha256_file(output_path),
        "base_row_count": len(base_rows),
        "added_row_count": len(additions),
        "output_row_count": len(combined),
        "added_counts_by_role": dict(
            sorted(Counter(str(row["label_role"]) for row in additions).items())
        ),
        "added_hard_negative_counts": dict(
            sorted(
                Counter(
                    str(row["hard_negative_type"])
                    for row in additions
                    if row["label_role"] == "negative"
                ).items()
            )
        ),
        "test_read": False,
        "test_evaluated": False,
        "next_step": (
            "review candidate rows, then run build_fall_training_split_v3.py and "
            "validate_fall_labels_v3.py against the approved v3 fact source"
        ),
    }
    _write_json_atomic(report_path, publication)
    return {
        "output_event_labels": output_path.as_posix(),
        "report": report_path.as_posix(),
        "added_row_count": len(additions),
        "status": publication["status"],
    }


def _build_event_row(
    decision: Mapping[str, Any],
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    _require_decision_shape(decision)
    if decision.get("schema_version") != MANUAL_DECISION_SCHEMA_VERSION:
        raise ValueError(
            "near-fall decision schema_version must be "
            f"{MANUAL_DECISION_SCHEMA_VERSION}"
        )
    decision_id = _required_string(decision, "decision_id")
    if not _ID_PATTERN.fullmatch(decision_id):
        raise ValueError(f"invalid near-fall decision_id: {decision_id}")
    video_id = _required_string(decision, "video_id")
    manifest = manifest_index.get(video_id)
    if manifest is None:
        raise ValueError(f"near-fall decision references missing video: {video_id}")
    if manifest.get("eligibility") is not True:
        raise ValueError(f"near-fall decision references ineligible video: {video_id}")
    start = _required_int(decision, "start_frame")
    end = _required_int(decision, "end_frame_exclusive")
    frame_count = _required_int(manifest, "frame_count")
    if not 0 <= start < end <= frame_count:
        raise ValueError(f"invalid near-fall half-open interval for {decision_id}")
    if decision.get("frame_index_base", 0) != 0:
        raise ValueError(f"near-fall decision must use frame_index_base=0: {decision_id}")
    boundary_precision = _required_string(decision, "boundary_precision")
    if boundary_precision not in _BOUNDARY_PRECISIONS:
        raise ValueError(f"invalid near-fall boundary_precision: {decision_id}")
    quality_flags = _string_list(decision, "quality_flags")
    if set(quality_flags) - _QUALITY_FLAGS:
        raise ValueError(f"invalid near-fall quality flag: {decision_id}")
    if set(quality_flags) & _SEVERE_FLAGS:
        raise ValueError(f"primary near-fall decision has severe quality flags: {decision_id}")
    label_role = decision.get("label_role")
    if label_role not in {"positive", "negative"}:
        raise ValueError(f"near-fall manual decision must be positive or negative: {decision_id}")
    if decision.get("target_status") != "confirmed":
        raise ValueError(f"near-fall decision target_status must be confirmed: {decision_id}")
    if decision.get("source_type") != MANUAL_DECISION_SOURCE_TYPE:
        raise ValueError(
            "near-fall decision source_type must be manual_near_fall_v1: "
            f"{decision_id}"
        )
    review_status = _required_string(decision, "review_status")
    reviewer_ids = _string_list(decision, "reviewer_ids")
    linked_action_ids = _string_list(decision, "linked_action_ids")
    if linked_action_ids:
        raise ValueError(
            "near-fall manual publication linked_action_ids must be empty until "
            f"a reciprocal action-label publisher exists: {decision_id}"
        )
    contact_evidence = _required_string(decision, "contact_evidence")
    if contact_evidence not in _CONTACT_EVIDENCE:
        raise ValueError(f"invalid near-fall contact_evidence: {decision_id}")
    if label_role == "positive":
        if review_status not in _POSITIVE_REVIEW_STATUSES:
            raise ValueError(f"near-fall positive requires double review: {decision_id}")
        if len(reviewer_ids) < 2:
            raise ValueError(f"near-fall positive needs two reviewers: {decision_id}")
        physical_event_id = _required_string(decision, "physical_event_id")
        if not re.fullmatch(r"^physical_[0-9a-f]{24}$", physical_event_id):
            raise ValueError(f"invalid physical_event_id: {decision_id}")
        onset = _required_int(decision, "onset_frame")
        recovery = _required_int(decision, "recovery_frame")
        if not start <= onset <= recovery < end:
            raise ValueError(f"invalid onset/recovery boundary: {decision_id}")
        peak = _optional_int(decision, "peak_frame")
        if peak is not None and not onset <= peak <= recovery:
            raise ValueError(f"near-fall peak_frame must be between onset and recovery: {decision_id}")
        if decision.get("impact_frame") is not None:
            raise ValueError(f"near-fall positive cannot have impact_frame: {decision_id}")
        if decision.get("event_subtype") not in _EVENT_SUBTYPES:
            raise ValueError(f"near-fall positive needs a recovery subtype: {decision_id}")
        if decision.get("hard_negative_type") is not None:
            raise ValueError(f"near-fall positive cannot have hard_negative_type: {decision_id}")
        event_type = "near_fall"
        event_outcome = "recovered_without_fall"
        subtype_training_tier = "primary"
    else:
        if review_status not in _NEGATIVE_REVIEW_STATUSES:
            raise ValueError(f"near-fall negative requires review: {decision_id}")
        required_reviewers = 2 if review_status == "double_reviewed" else 1
        if len(reviewer_ids) < required_reviewers:
            raise ValueError(f"near-fall negative requires reviewer_ids: {decision_id}")
        hard_negative_type = decision.get("hard_negative_type")
        if hard_negative_type not in {*NEAR_FALL_HARD_NEGATIVES, "background"}:
            raise ValueError(f"invalid near-fall hard negative type: {decision_id}")
        if any(decision.get(field) is not None for field in ("physical_event_id", "onset_frame", "peak_frame", "impact_frame", "recovery_frame")):
            raise ValueError(f"near-fall negative has positive-only fields: {decision_id}")
        if any(decision.get(field) is not None for field in ("event_type", "event_subtype", "event_outcome")):
            raise ValueError(f"near-fall negative has event fields: {decision_id}")
        event_type = None
        event_outcome = None
        subtype_training_tier = "ignore"
        physical_event_id = None
        onset = None
        peak = None
        recovery = None
    source_path = _resolve_source_path(
        _required_string(decision, "source_annotation_path"), repo_root
    )
    source_hash = _required_string(decision, "source_annotation_sha256")
    if not _HASH_PATTERN.fullmatch(source_hash):
        raise ValueError(f"invalid source annotation SHA-256: {decision_id}")
    if not source_path.is_file():
        raise FileNotFoundError(f"near-fall source annotation not found: {source_path}")
    if _sha256_file(source_path) != source_hash:
        raise ValueError(f"near-fall source annotation SHA-256 mismatch: {decision_id}")
    content_sha = _required_string(manifest, "sha256")
    if not _HASH_PATTERN.fullmatch(content_sha):
        raise ValueError(f"invalid manifest content SHA-256: {video_id}")
    asset_id = _required_string(manifest, "asset_id")
    label_id = _stable_id(
        "eventv3",
        EVENT_SCHEMA_VERSION,
        "near_fall_event",
        label_role,
        decision_id,
    )
    fps_num = _required_int(manifest, "fps_num")
    fps_den = _required_int(manifest, "fps_den")
    if fps_num <= 0 or fps_den <= 0:
        raise ValueError("manifest FPS must be positive")
    return {
        "_decision_id": decision_id,
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": label_id,
        "asset_id": asset_id,
        "video_id": video_id,
        "content_sha256": content_sha,
        "subject_id": _required_string(manifest, "subject_id"),
        "source_group_id": _source_group(manifest),
        "sample_group_id": _sample_group_id(manifest),
        "track_id": None if decision.get("track_id") is None else str(decision["track_id"]),
        "start_frame": start,
        "end_frame_exclusive": end,
        "frame_index_base": 0,
        "start_time": _frame_time(start, fps_num, fps_den),
        "end_time_exclusive": _frame_time(end, fps_num, fps_den),
        "target_status": "confirmed",
        "boundary_precision": boundary_precision,
        "quality_flags": quality_flags,
        "training_tier": "primary",
        "source_refs": [
            {
                "source_type": "manual_v3",
                "source_record_id": decision_id,
                "source_annotation_path": _display_source_path(source_path, repo_root),
                "source_annotation_sha256": source_hash,
                "source_label_id": decision_id,
            }
        ],
        "annotator_id": _required_string(decision, "annotator_id"),
        "reviewer_ids": reviewer_ids,
        "review_status": review_status,
        "note": _required_string(decision, "note"),
        "physical_event_id": physical_event_id,
        "task_type": "near_fall_event",
        "label_role": label_role,
        "event_type": event_type,
        "event_subtype": decision.get("event_subtype") if label_role == "positive" else None,
        "event_outcome": event_outcome,
        "hard_negative_type": decision.get("hard_negative_type") if label_role == "negative" else None,
        "onset_frame": onset,
        "peak_frame": peak,
        "impact_frame": None,
        "recovery_frame": recovery,
        "linked_action_ids": linked_action_ids,
        "contact_evidence": contact_evidence,
        "subtype_training_tier": subtype_training_tier,
    }


def _require_decision_shape(decision: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "decision_id",
        "video_id",
        "label_role",
        "start_frame",
        "end_frame_exclusive",
        "target_status",
        "boundary_precision",
        "quality_flags",
        "annotator_id",
        "reviewer_ids",
        "review_status",
        "note",
        "source_type",
        "source_annotation_path",
        "source_annotation_sha256",
        "linked_action_ids",
        "contact_evidence",
    }
    missing = sorted(key for key in required if key not in decision)
    if missing:
        raise ValueError(f"near-fall decision missing fields: {missing}")


def _index_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id", ""))
        if not video_id:
            continue
        if video_id in index:
            raise ValueError(f"duplicate manifest video_id: {video_id}")
        index[video_id] = dict(row)
    return index


def _resolve_source_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_source_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"near-fall decision field {field} must be a non-empty string")
    return value.strip()


def _required_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"near-fall decision field {field} must be an integer")
    return value


def _optional_int(row: Mapping[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"near-fall decision field {field} must be an integer or null")
    return value


def _string_list(row: Mapping[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"near-fall decision field {field} must be a string list")
    normalized = [item.strip() for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"near-fall decision field {field} contains duplicates")
    return normalized


def _label_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("video_id", "")), int(row.get("start_frame", 0)), str(row.get("label_id", "")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    try:
        with partial.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    try:
        partial.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
