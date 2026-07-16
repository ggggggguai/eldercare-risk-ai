from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

import yaml

from elderly_monitoring.modules.fall_risk.annotations import (
    ACTION_EVENT_MAP,
    ACTION_EVENT_MAPPING_VERSION,
    ACTION_NAMES,
    QUALITY_VALUES,
    read_le2i_fall_window,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "fall-risk-label-validation-v1",
    "time_tolerance_sec": 0.001,
    "formal_review_statuses": ["reviewed", "final"],
    "ineligible_review_statuses": [
        "pending",
        "uncertain",
        "missing",
        "license_unknown",
    ],
    "unknown_license_values": ["", "unknown", "license_unknown", "missing", "none"],
    "review_approval_decisions": ["approve", "adjudicate"],
    "minimum_independent_reviewers": 2,
    "high_risk_action_prefixes": ["C", "D"],
}

ACTION_REQUIRED = {
    "label_id",
    "source_record_id",
    "source_annotation_path",
    "source_annotation_sha256",
    "source_export_id",
    "asset_id",
    "video_id",
    "file_path",
    "subject_id",
    "scene",
    "view",
    "action_id",
    "action_name",
    "event_type",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "frame_index_base",
    "labeler",
    "review_status",
    "quality",
    "note",
    "source",
    "cvat_task_id",
    "cvat_track_id",
    "bbox_start",
    "bbox_end",
    "eligibility",
    "review_evidence_ids",
}
ACTION_ALLOWED = set(ACTION_REQUIRED)

EVENT_COMMON_REQUIRED = {
    "label_id",
    "source_record_id",
    "source_annotation_path",
    "source_annotation_sha256",
    "asset_id",
    "video_id",
    "event_type",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "frame_index_base",
    "severity",
    "label_source",
    "review_status",
    "note",
    "eligibility",
    "review_evidence_ids",
}
EVENT_MAPPED_REQUIRED = EVENT_COMMON_REQUIRED | {
    "source_export_id",
    "source_action_id",
    "source_action_name",
    "source_action_label_id",
    "mapping_version",
    "cvat_task_id",
    "cvat_track_id",
}
EVENT_LE2I_REQUIRED = EVENT_COMMON_REQUIRED | {
    "source_start_frame",
    "source_end_frame",
    "source_frame_index_base",
}
EVENT_ALLOWED = EVENT_MAPPED_REQUIRED | EVENT_LE2I_REQUIRED

RISK_REQUIRED = {
    "label_id",
    "asset_id",
    "task_type",
    "subject_id",
    "start_time",
    "end_time",
    "risk_level",
    "risk_factors",
    "label_source",
    "review_status",
    "eligibility",
    "review_evidence_ids",
}
RISK_ALLOWED = RISK_REQUIRED | {
    "video_id",
    "review_record_ids",
    "note",
    "risk_score",
}

REVIEW_REQUIRED = {
    "review_id",
    "label_id",
    "label_type",
    "reviewer_id",
    "decision",
    "reviewed_at",
    "reason_code",
    "note",
}
REVIEW_ALLOWED = REVIEW_REQUIRED | {
    "previous_record_sha256",
    "result_record_sha256",
    "supersedes_review_id",
}

EVENT_TYPES = {event_type for event_type, _ in ACTION_EVENT_MAP.values()} | {
    "sudden_stop_recovery",
    "recovery",
}
REVIEW_STATUSES = {
    "pending",
    "reviewed",
    "final",
    "auto_imported",
    "uncertain",
    "missing",
    "license_unknown",
    "rejected",
}
_PSEUDONYM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


@dataclass(frozen=True)
class ValidationConfig:
    time_tolerance_sec: float
    formal_review_statuses: frozenset[str]
    ineligible_review_statuses: frozenset[str]
    unknown_license_values: frozenset[str]
    review_approval_decisions: frozenset[str]
    minimum_independent_reviewers: int
    high_risk_action_prefixes: frozenset[str]
    schema_version: str = "fall-risk-label-validation-v1"


def load_validation_config(path: Path | str | None = None) -> ValidationConfig:
    config, _ = _load_validation_config_with_hash(path)
    return config


def _load_validation_config_with_hash(
    path: Path | str | None,
) -> tuple[ValidationConfig, str | None]:
    raw = dict(DEFAULT_CONFIG)
    config_sha256: str | None = None
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        payload = config_path.read_bytes()
        config_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            config_text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("validation config must be UTF-8") from exc
        try:
            loaded = yaml.safe_load(config_text)
        except yaml.YAMLError as exc:
            raise ValueError("validation config must be valid YAML") from exc
        if not isinstance(loaded, dict):
            raise ValueError("validation config must be a YAML mapping")
        unknown = set(loaded) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown validation config keys: {sorted(unknown)}")
        raw.update(loaded)
    tolerance = raw["time_tolerance_sec"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("time_tolerance_sec must be a number")
    tolerance = float(tolerance)
    if (
        tolerance < 0
        or not math.isfinite(tolerance)
        or tolerance > float(DEFAULT_CONFIG["time_tolerance_sec"])
    ):
        raise ValueError(
            "time_tolerance_sec must be finite, non-negative, and no greater "
            "than the v1 governance maximum"
        )
    minimum_reviewers = raw["minimum_independent_reviewers"]
    if not _exact_int(minimum_reviewers) or minimum_reviewers < 2:
        raise ValueError("minimum_independent_reviewers must be at least 2")
    if raw["schema_version"] != DEFAULT_CONFIG["schema_version"]:
        raise ValueError("unsupported validation config schema_version")
    formal_review_statuses = _config_string_set(raw, "formal_review_statuses")
    ineligible_review_statuses = _config_string_set(
        raw, "ineligible_review_statuses"
    )
    unknown_license_values = frozenset(
        value.lower()
        for value in _config_string_set(
            raw, "unknown_license_values", allow_empty=True
        )
    )
    review_approval_decisions = _config_string_set(
        raw, "review_approval_decisions"
    )
    high_risk_action_prefixes = _config_string_set(
        raw, "high_risk_action_prefixes"
    )
    default_formal = frozenset(DEFAULT_CONFIG["formal_review_statuses"])
    default_ineligible = frozenset(DEFAULT_CONFIG["ineligible_review_statuses"])
    default_unknown_licenses = frozenset(
        str(value).lower() for value in DEFAULT_CONFIG["unknown_license_values"]
    )
    default_approval = frozenset(DEFAULT_CONFIG["review_approval_decisions"])
    default_high_risk = frozenset(DEFAULT_CONFIG["high_risk_action_prefixes"])
    if not formal_review_statuses or not formal_review_statuses <= default_formal:
        raise ValueError("formal_review_statuses may only contain reviewed/final")
    if not ineligible_review_statuses >= default_ineligible:
        raise ValueError("ineligible_review_statuses cannot remove v1 blockers")
    if not unknown_license_values >= default_unknown_licenses:
        raise ValueError("unknown_license_values cannot remove v1 unknown values")
    if not review_approval_decisions or not review_approval_decisions <= default_approval:
        raise ValueError("review_approval_decisions may only contain approve/adjudicate")
    if not high_risk_action_prefixes >= default_high_risk:
        raise ValueError("high_risk_action_prefixes must include C and D")
    return (
        ValidationConfig(
            schema_version=str(raw["schema_version"]),
            time_tolerance_sec=tolerance,
            formal_review_statuses=formal_review_statuses,
            ineligible_review_statuses=ineligible_review_statuses,
            unknown_license_values=unknown_license_values,
            review_approval_decisions=review_approval_decisions,
            minimum_independent_reviewers=minimum_reviewers,
            high_risk_action_prefixes=high_risk_action_prefixes,
        ),
        config_sha256,
    )


def _config_string_set(
    raw: Mapping[str, Any], field: str, *, allow_empty: bool = False
) -> frozenset[str]:
    value = raw[field]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or (not item and not allow_empty) for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    return frozenset(value)


def validate_fall_risk_data(
    *,
    manifest_path: Path | str,
    action_labels_path: Path | str,
    event_labels_path: Path | str,
    risk_labels_path: Path | str,
    subject_profiles_path: Path | str,
    review_log_path: Path | str,
    mode: str = "audit",
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    if mode not in {"audit", "formal"}:
        raise ValueError("mode must be 'audit' or 'formal'")
    config, config_input_hash = _load_validation_config_with_hash(config_path)
    issues: list[dict[str, Any]] = []

    manifest_rows, manifest_input_hash = _load_jsonl_strict(
        Path(manifest_path), "manifest", issues
    )
    action_rows, action_input_hash = _load_jsonl_strict(
        Path(action_labels_path), "action", issues
    )
    event_rows, event_input_hash = _load_jsonl_strict(
        Path(event_labels_path), "event", issues
    )
    risk_rows, risk_input_hash = _load_jsonl_strict(
        Path(risk_labels_path), "risk", issues
    )
    review_rows, review_input_hash = _load_jsonl_strict(
        Path(review_log_path), "review", issues
    )
    profiles, profiles_input_hash = _load_json_object_strict(
        Path(subject_profiles_path), "profiles", issues
    )

    manifest = _manifest_index(manifest_rows, issues, config)
    review_evidence = _validate_reviews(
        review_rows,
        action_rows=action_rows,
        event_rows=event_rows,
        risk_rows=risk_rows,
        config=config,
        issues=issues,
    )
    source_hash_cache: dict[Path, str] = {}
    action_index = _validate_actions(
        action_rows,
        manifest,
        review_evidence,
        source_hash_cache,
        config,
        mode,
        issues,
    )
    _validate_events(
        event_rows,
        manifest,
        action_index,
        review_evidence,
        source_hash_cache,
        config,
        mode,
        issues,
    )
    _validate_risk_labels(
        risk_rows, manifest, review_evidence, config, mode, issues
    )
    _validate_profiles(profiles, config, mode, issues)

    counts = {
        "manifest_videos": sum(row.get("video_id") is not None for row in manifest_rows),
        "action_labels": len(action_rows),
        "event_labels": len(event_rows),
        "risk_labels": len(risk_rows),
        "review_records": len(review_rows),
        "subject_profiles": len(profiles.get("subjects", []))
        if isinstance(profiles, dict) and isinstance(profiles.get("subjects"), list)
        else 0,
        "errors": sum(issue["severity"] == "error" for issue in issues),
        "blockers": sum(issue["severity"] == "blocker" for issue in issues),
        "warnings": sum(issue["severity"] == "warning" for issue in issues),
    }
    invalid_severities = {"error"} | ({"blocker"} if mode == "formal" else set())
    valid = not any(issue["severity"] in invalid_severities for issue in issues)
    return {
        "schema_version": "fall-risk-label-validation-report-v1",
        "validation_config_version": config.schema_version,
        "mode": mode,
        "valid": valid,
        "input_sha256": {
            "manifest": manifest_input_hash,
            "action_labels": action_input_hash,
            "event_labels": event_input_hash,
            "risk_labels": risk_input_hash,
            "subject_profiles": profiles_input_hash,
            "review_log": review_input_hash,
            "validation_config": config_input_hash,
        },
        "counts": counts,
        "schema_valid": not any(issue["severity"] == "error" for issue in issues),
        "formal_ready": mode == "formal" and valid,
        "distributions": _distributions(
            manifest_rows,
            action_rows,
            event_rows,
            risk_rows,
            review_rows,
            issues,
        ),
        "issues": issues,
    }


def write_validation_report(
    report: Mapping[str, Any], output_path: Path | str, *, overwrite: bool = False
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with NamedTemporaryFile(prefix=".fall-validation-", dir=path.parent, delete=False) as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_jsonl_strict(
    path: Path, kind: str, issues: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        _issue(issues, "error", "missing_input", kind, message=f"missing {kind} file")
        return [], None
    rows: list[dict[str, Any]] = []
    payload = path.read_bytes()
    input_hash = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _issue(issues, "error", "invalid_jsonl", kind, message="UnicodeDecodeError")
        return rows, input_hash
    if not text:
        return rows, input_hash
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            _issue(
                issues,
                "error",
                "invalid_jsonl",
                kind,
                record_index=line_number,
                message="blank JSONL line",
            )
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            _issue(
                issues,
                "error",
                "invalid_jsonl",
                kind,
                record_index=line_number,
                message=type(exc).__name__,
            )
            continue
        if not isinstance(row, dict):
            _issue(
                issues,
                "error",
                "invalid_jsonl",
                kind,
                record_index=line_number,
                message="row is not an object",
            )
            continue
        rows.append(row)
    return rows, input_hash


def _load_json_object_strict(
    path: Path, kind: str, issues: list[dict[str, Any]]
) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        _issue(issues, "error", "missing_input", kind, message=f"missing {kind} file")
        return {}, None
    payload = path.read_bytes()
    input_hash = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _issue(
            issues,
            "error",
            "invalid_json",
            kind,
            message=type(exc).__name__,
        )
        return {}, input_hash
    if not isinstance(value, dict):
        _issue(issues, "error", "invalid_json", kind, message="root is not an object")
        return {}, input_hash
    return value, input_hash


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _manifest_index(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    config: ValidationConfig,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            _issue(
                issues,
                "error",
                "invalid_manifest_asset_id",
                "manifest",
                record_index=position,
            )
            continue
        if asset_id in index:
            _issue(
                issues,
                "error",
                "duplicate_manifest_asset_id",
                "manifest",
                label_id=asset_id,
            )
            continue
        video_id = row.get("video_id")
        required_fields = [
            "asset_id",
            "path",
            "sha256",
            "source_uri",
            "license_id",
            "eligibility",
            "exclusion_reasons",
        ]
        if video_id is not None:
            required_fields.extend(
                ["fps_num", "fps_den", "frame_count", "duration_sec"]
            )
            if not isinstance(video_id, str) or not video_id:
                _issue(
                    issues,
                    "error",
                    "invalid_manifest_video_id",
                    "manifest",
                    record_index=position,
                )
                continue
        missing = [field for field in required_fields if field not in row]
        if missing:
            _issue(
                issues,
                "error",
                "missing_fields",
                "manifest",
                video_id=str(video_id or ""),
                message=",".join(missing),
            )
        path = Path(str(row.get("path", "")))
        expected_sha256 = str(row.get("sha256", "")).lower()
        if not path.is_file():
            _issue(
                issues,
                "error",
                "missing_media_file",
                "manifest",
                label_id=asset_id,
            )
        elif not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            _issue(
                issues,
                "error",
                "invalid_content_hash",
                "manifest",
                label_id=asset_id,
            )
        elif _sha256_path(path) != expected_sha256:
            _issue(
                issues,
                "error",
                "manifest_content_hash_mismatch",
                "manifest",
                label_id=asset_id,
            )
        if row.get("eligibility") is True:
            source_uri = str(row.get("source_uri") or "").strip()
            license_id = str(row.get("license_id") or "").strip().lower()
            exclusion_reasons = row.get("exclusion_reasons")
            if (
                not re.match(r"^https?://", source_uri)
                or license_id in config.unknown_license_values
                or not isinstance(exclusion_reasons, list)
                or bool(exclusion_reasons)
            ):
                _issue(
                    issues,
                    "error",
                    "invalid_manifest_provenance",
                    "manifest",
                    label_id=asset_id,
                )
        index[asset_id] = row
        if video_id is not None:
            if video_id in index:
                _issue(
                    issues,
                    "error",
                    "duplicate_manifest_video_id",
                    "manifest",
                    video_id=video_id,
                )
            else:
                index[video_id] = row
    return index


def _validate_reviews(
    rows: list[dict[str, Any]],
    *,
    action_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    config: ValidationConfig,
    issues: list[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    label_records = _review_target_records(action_rows, event_rows, risk_rows)
    reviews: dict[str, dict[str, Any]] = {}
    valid_review_ids: set[str] = set()
    review_timestamps: dict[str, datetime] = {}
    for position, row in enumerate(rows, 1):
        if not _validate_shape(row, REVIEW_REQUIRED, REVIEW_ALLOWED, "review", position, issues):
            continue
        review_id = row["review_id"]
        if not isinstance(review_id, str) or not review_id:
            _issue(issues, "error", "invalid_id", "review", record_index=position)
            continue
        if review_id in reviews:
            _issue(
                issues,
                "error",
                "duplicate_label_id",
                "review",
                label_id=review_id,
            )
            continue
        reviews[review_id] = row
        row_valid = True
        label_type = row["label_type"]
        if not isinstance(label_type, str) or label_type not in {
            "action",
            "event",
            "risk",
        }:
            _issue(issues, "error", "invalid_enum", "review", label_id=review_id)
            row_valid = False
        decision = row["decision"]
        if not isinstance(decision, str) or decision not in {
            "approve",
            "revise",
            "reject",
            "conflict",
            "adjudicate",
        }:
            _issue(issues, "error", "invalid_enum", "review", label_id=review_id)
            row_valid = False
        if not isinstance(row["reviewer_id"], str) or not row["reviewer_id"]:
            _issue(issues, "error", "invalid_reviewer_id", "review", label_id=review_id)
            row_valid = False
        elif not _is_pseudonymous_identifier(row["reviewer_id"]):
            _issue(issues, "error", "unsafe_identifier", "review", label_id=review_id)
            row_valid = False
        timestamp = _parse_review_timestamp(row["reviewed_at"])
        if timestamp is None:
            _issue(issues, "error", "invalid_review_timestamp", "review", label_id=review_id)
            row_valid = False
        else:
            review_timestamps[review_id] = timestamp
        if not isinstance(row["label_id"], str) or not row["label_id"]:
            _issue(issues, "error", "invalid_review_label_id", "review", label_id=review_id)
            row_valid = False
        for field in ("reason_code", "note"):
            if not isinstance(row[field], str):
                _issue(
                    issues,
                    "error",
                    "invalid_review_text",
                    "review",
                    label_id=review_id,
                    message=field,
                )
                row_valid = False
            elif _contains_contact_identifier(row[field]):
                _issue(
                    issues,
                    "error",
                    "potential_identity_data",
                    "review",
                    label_id=review_id,
                    message=field,
                )
                row_valid = False
        for field in ("previous_record_sha256", "result_record_sha256"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
            ):
                _issue(
                    issues,
                    "error",
                    f"invalid_{field}",
                    "review",
                    label_id=review_id,
                )
                row_valid = False
        if (
            isinstance(decision, str)
            and decision in config.review_approval_decisions
            and "result_record_sha256" not in row
        ):
            _issue(
                issues,
                "error",
                "missing_review_result_hash",
                "review",
                label_id=review_id,
            )
            row_valid = False
        if decision == "adjudicate" and not isinstance(
            row.get("supersedes_review_id"), str
        ):
            _issue(
                issues,
                "error",
                "adjudication_without_conflict",
                "review",
                label_id=review_id,
            )
            row_valid = False
        if row_valid:
            valid_review_ids.add(review_id)

    superseded: set[str] = set()
    successor_by_review: dict[str, str] = {}
    for review_id in sorted(valid_review_ids):
        row = reviews[review_id]
        predecessor_id = row.get("supersedes_review_id")
        if predecessor_id is None:
            continue
        predecessor = reviews.get(predecessor_id) if isinstance(predecessor_id, str) else None
        if predecessor_id == review_id or predecessor is None:
            _issue(
                issues,
                "error",
                "invalid_supersedes_review_id",
                "review",
                label_id=review_id,
            )
            continue
        if predecessor_id not in valid_review_ids or any(
            row[field] != predecessor[field] for field in ("label_type", "label_id")
        ):
            _issue(
                issues,
                "error",
                "review_supersession_mismatch",
                "review",
                label_id=review_id,
            )
            continue
        predecessor_decision = predecessor["decision"]
        if predecessor_decision == "conflict" and row["decision"] != "adjudicate":
            _issue(
                issues,
                "error",
                "invalid_review_transition",
                "review",
                label_id=review_id,
                message="conflict must be followed by adjudicate",
            )
            continue
        if row["decision"] == "adjudicate":
            if predecessor_decision != "conflict":
                _issue(
                    issues,
                    "error",
                    "invalid_review_transition",
                    "review",
                    label_id=review_id,
                    message="adjudicate must directly supersede conflict",
                )
                continue
            prior_reviewers = _review_chain_reviewer_ids(predecessor_id, reviews)
            if row["reviewer_id"] in prior_reviewers:
                _issue(
                    issues,
                    "error",
                    "non_independent_adjudicator",
                    "review",
                    label_id=review_id,
                )
                continue
        predecessor_result = predecessor.get("result_record_sha256")
        if (
            not isinstance(predecessor_result, str)
            or row.get("previous_record_sha256") != predecessor_result
        ):
            _issue(
                issues,
                "error",
                "review_previous_hash_mismatch",
                "review",
                label_id=review_id,
            )
            continue
        if review_timestamps[review_id] <= review_timestamps[predecessor_id]:
            _issue(
                issues,
                "error",
                "review_timestamp_not_increasing",
                "review",
                label_id=review_id,
            )
            continue
        if predecessor_id in successor_by_review:
            _issue(
                issues,
                "error",
                "multiple_review_successors",
                "review",
                label_id=review_id,
            )
            continue
        successor_by_review[predecessor_id] = review_id
        superseded.add(predecessor_id)

    for review_id in successor_by_review:
        visited: set[str] = set()
        current = review_id
        while current in successor_by_review:
            if current in visited:
                _issue(
                    issues,
                    "error",
                    "review_supersession_cycle",
                    "review",
                    label_id=review_id,
                )
                valid_review_ids.difference_update(visited)
                break
            visited.add(current)
            current = successor_by_review[current]

    active_review_ids = valid_review_ids - superseded
    evidence: dict[tuple[str, str], set[str]] = {}
    reviewers_by_target: dict[tuple[str, str], set[str]] = {}
    for review_id in sorted(active_review_ids):
        row = reviews[review_id]
        if row["decision"] not in config.review_approval_decisions:
            _issue(
                issues,
                "blocker",
                "unresolved_review_decision",
                "review",
                label_id=review_id,
                message=str(row["decision"]),
            )
            continue
        key = (row["label_type"], row["label_id"])
        target = label_records.get(key)
        if target is None:
            _issue(
                issues,
                "error",
                "review_target_missing",
                "review",
                label_id=review_id,
            )
            continue
        expected = _canonical_record_sha256(target)
        if row.get("result_record_sha256") != expected:
            _issue(
                issues,
                "error",
                "review_result_hash_mismatch",
                "review",
                label_id=review_id,
            )
            continue
        evidence.setdefault(key, set()).add(review_id)
        reviewers_by_target.setdefault(key, set()).add(row["reviewer_id"])

    for key, target in label_records.items():
        if target.get("eligibility") is not True:
            continue
        reviewers = reviewers_by_target.get(key, set())
        if len(reviewers) < config.minimum_independent_reviewers:
            _issue(
                issues,
                "blocker",
                "insufficient_independent_reviewers",
                key[0],
                label_id=key[1],
                message=(
                    f"required={config.minimum_independent_reviewers},"
                    f"actual={len(reviewers)}"
                ),
            )
    return evidence


def _review_target_records(
    action_rows: Iterable[dict[str, Any]],
    event_rows: Iterable[dict[str, Any]],
    risk_rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for kind, rows in (
        ("action", action_rows),
        ("event", event_rows),
        ("risk", risk_rows),
    ):
        for row in rows:
            label_id = row.get("label_id")
            if isinstance(label_id, str) and label_id and (kind, label_id) not in records:
                records[(kind, label_id)] = row
    return records


def _review_chain_reviewer_ids(
    review_id: str, reviews: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    reviewers: set[str] = set()
    visited: set[str] = set()
    current: str | None = review_id
    while current is not None and current not in visited:
        visited.add(current)
        review = reviews.get(current)
        if review is None:
            break
        reviewer_id = review.get("reviewer_id")
        if isinstance(reviewer_id, str):
            reviewers.add(reviewer_id)
        predecessor = review.get("supersedes_review_id")
        current = predecessor if isinstance(predecessor, str) else None
    return reviewers


def _canonical_record_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_review_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})", value
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _stable_label_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _validate_actions(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
    review_evidence: Mapping[tuple[str, str], set[str]],
    source_hash_cache: dict[Path, str],
    config: ValidationConfig,
    mode: str,
    issues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        shape_valid = _validate_shape(
            row, ACTION_REQUIRED, ACTION_ALLOWED, "action", position, issues
        )
        if not shape_valid:
            if mode == "formal":
                _formal_partial_record(row, "action", position, config, issues)
            continue
        label_id = _register_label_id(row, index, "action", position, issues)
        if label_id is None:
            continue
        action_id = row["action_id"]
        expected_name = ACTION_NAMES.get(action_id)
        if expected_name is None or row["action_name"] != expected_name:
            _issue(issues, "error", "invalid_action_label", "action", label_id=label_id)
        expected_event = ACTION_EVENT_MAP.get(action_id)
        if expected_event is None or row["event_type"] != expected_event[0]:
            _issue(issues, "error", "invalid_action_event_mapping", "action", label_id=label_id)
        if row["quality"] not in QUALITY_VALUES:
            _issue(issues, "error", "invalid_enum", "action", label_id=label_id)
        if row["review_status"] not in REVIEW_STATUSES:
            _issue(issues, "error", "invalid_enum", "action", label_id=label_id)
        for field in ("subject_id", "labeler"):
            if not _is_pseudonymous_identifier(row[field], allow_unknown=True):
                _issue(
                    issues,
                    "error",
                    "unsafe_identifier",
                    "action",
                    label_id=label_id,
                    message=field,
                )
        if _contains_contact_identifier(row["note"]):
            _issue(
                issues,
                "error",
                "potential_identity_data",
                "action",
                label_id=label_id,
                message="note",
            )
        _validate_eligibility_and_reviews(
            row, "action", label_id, review_evidence, issues
        )
        if action_id == "U01" and not str(row["note"]).strip():
            _issue(issues, "error", "uncertain_reason_missing", "action", label_id=label_id)
        manifest_row = _validate_label_manifest_link(row, manifest, "action", label_id, issues)
        _validate_window(row, manifest_row, config, "action", label_id, issues)
        _validate_source_file(
            row, "action", label_id, source_hash_cache, issues
        )
        _validate_bbox(row.get("bbox_start"), "action", label_id, issues)
        _validate_bbox(row.get("bbox_end"), "action", label_id, issues)
        high_risk_action = str(action_id)[:1] in config.high_risk_action_prefixes
        if high_risk_action:
            _require_review(label_id, review_evidence, "action", issues)
        if mode == "formal":
            _formal_common(row, manifest_row, config, "action", label_id, issues)
            if row["event_type"] == "uncertain":
                _issue(issues, "blocker", "formal_uncertain", "action", label_id=label_id)
    return index


def _validate_events(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
    actions: Mapping[str, dict[str, Any]],
    review_evidence: Mapping[tuple[str, str], set[str]],
    source_hash_cache: dict[Path, str],
    config: ValidationConfig,
    mode: str,
    issues: list[dict[str, Any]],
) -> None:
    seen: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        source = row.get("label_source")
        required = (
            EVENT_MAPPED_REQUIRED
            if source == "cvat_action_mapping"
            else EVENT_LE2I_REQUIRED
            if source == "le2i_txt"
            else EVENT_COMMON_REQUIRED
        )
        allowed = (
            required
            if source in {"cvat_action_mapping", "le2i_txt"}
            else EVENT_COMMON_REQUIRED
        )
        if not _validate_shape(row, required, allowed, "event", position, issues):
            if mode == "formal":
                _formal_partial_record(row, "event", position, config, issues)
            continue
        label_id = _register_label_id(row, seen, "event", position, issues)
        if label_id is None:
            continue
        if row["event_type"] not in EVENT_TYPES:
            _issue(issues, "error", "invalid_enum", "event", label_id=label_id)
        if row["review_status"] not in REVIEW_STATUSES:
            _issue(issues, "error", "invalid_enum", "event", label_id=label_id)
        if _contains_contact_identifier(row["note"]):
            _issue(
                issues,
                "error",
                "potential_identity_data",
                "event",
                label_id=label_id,
                message="note",
            )
        _validate_eligibility_and_reviews(
            row, "event", label_id, review_evidence, issues
        )
        if not _exact_int(row["severity"]) or not 0 <= row["severity"] <= 4:
            _issue(issues, "error", "invalid_severity", "event", label_id=label_id)
        manifest_row = _validate_label_manifest_link(row, manifest, "event", label_id, issues)
        _validate_window(row, manifest_row, config, "event", label_id, issues)
        _validate_source_file(row, "event", label_id, source_hash_cache, issues)

        if source == "cvat_action_mapping":
            parent_id = row["source_action_label_id"]
            parent = actions.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None:
                _issue(issues, "error", "missing_source_action", "event", label_id=label_id)
            else:
                for field in (
                    "source_annotation_path",
                    "source_annotation_sha256",
                    "source_export_id",
                    "asset_id",
                    "video_id",
                    "start_time",
                    "end_time",
                    "start_frame",
                    "end_frame",
                    "frame_index_base",
                    "event_type",
                    "cvat_task_id",
                    "cvat_track_id",
                ):
                    if row[field] != parent[field]:
                        _issue(
                            issues,
                            "error",
                            "source_action_mismatch",
                            "event",
                            label_id=label_id,
                            message=field,
                        )
                if (
                    row["source_record_id"] != f"mapped:{parent_id}"
                    or row["source_action_id"] != parent["action_id"]
                    or row["source_action_name"] != parent["action_name"]
                ):
                    _issue(issues, "error", "source_action_mismatch", "event", label_id=label_id)
                expected_mapping = ACTION_EVENT_MAP.get(parent["action_id"])
                if expected_mapping is None or row["severity"] != expected_mapping[1]:
                    _issue(
                        issues,
                        "error",
                        "mapped_event_severity_mismatch",
                        "event",
                        label_id=label_id,
                    )
            if row["mapping_version"] != ACTION_EVENT_MAPPING_VERSION:
                _issue(issues, "error", "invalid_mapping_version", "event", label_id=label_id)
            if label_id != _stable_label_id(
                "event", parent_id, ACTION_EVENT_MAPPING_VERSION
            ):
                _issue(issues, "error", "invalid_stable_label_id", "event", label_id=label_id)
        elif source == "le2i_txt":
            if row["event_type"] != "fall" or row["source_frame_index_base"] != 1:
                _issue(issues, "error", "invalid_le2i_event", "event", label_id=label_id)
            if (
                _exact_int(row["source_start_frame"])
                and _exact_int(row["source_end_frame"])
                and (
                    row["source_start_frame"] - 1 != row["start_frame"]
                    or row["source_end_frame"] - 1 != row["end_frame"]
                )
            ):
                _issue(
                    issues,
                    "error",
                    "invalid_le2i_frame_normalization",
                    "event",
                    label_id=label_id,
                )
            try:
                official_window = read_le2i_fall_window(row["source_annotation_path"])
            except (OSError, ValueError):
                official_window = None
                _issue(
                    issues,
                    "error",
                    "invalid_le2i_source_window",
                    "event",
                    label_id=label_id,
                )
            if official_window != (
                row["source_start_frame"],
                row["source_end_frame"],
            ):
                _issue(
                    issues,
                    "error",
                    "le2i_source_window_mismatch",
                    "event",
                    label_id=label_id,
                )
            if label_id != _stable_label_id(
                "event",
                "le2i_txt",
                row["video_id"],
                row["source_annotation_sha256"],
                row["source_start_frame"],
                row["source_end_frame"],
            ):
                _issue(issues, "error", "invalid_stable_label_id", "event", label_id=label_id)
        else:
            _issue(issues, "error", "invalid_label_source", "event", label_id=label_id)

        if row["event_type"] == "fall":
            _require_review(label_id, review_evidence, "event", issues)
        if mode == "formal":
            _formal_common(row, manifest_row, config, "event", label_id, issues)
            if row["event_type"] == "uncertain":
                _issue(issues, "blocker", "formal_uncertain", "event", label_id=label_id)


def _validate_risk_labels(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
    review_evidence: Mapping[tuple[str, str], set[str]],
    config: ValidationConfig,
    mode: str,
    issues: list[dict[str, Any]],
) -> None:
    seen: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        if not _validate_shape(row, RISK_REQUIRED, RISK_ALLOWED, "risk", position, issues):
            if mode == "formal":
                _formal_partial_record(row, "risk", position, config, issues)
            continue
        label_id = _register_label_id(row, seen, "risk", position, issues)
        if label_id is None:
            continue
        if "risk_score" in row:
            _issue(
                issues,
                "error",
                "manual_risk_score_forbidden",
                "risk",
                label_id=label_id,
            )
        if not _exact_int(row["risk_level"]) or not 0 <= row["risk_level"] <= 4:
            _issue(issues, "error", "invalid_risk_level", "risk", label_id=label_id)
        if row["review_status"] not in REVIEW_STATUSES:
            _issue(issues, "error", "invalid_enum", "risk", label_id=label_id)
        if not _is_pseudonymous_identifier(row["subject_id"], allow_unknown=True):
            _issue(issues, "error", "unsafe_identifier", "risk", label_id=label_id)
        if _contains_contact_identifier(row.get("note", "")):
            _issue(
                issues,
                "error",
                "potential_identity_data",
                "risk",
                label_id=label_id,
                message="note",
            )
        _validate_eligibility_and_reviews(
            row, "risk", label_id, review_evidence, issues
        )
        if row["label_source"] not in {"manual_consensus", "clinical_proxy"}:
            _issue(issues, "error", "invalid_label_source", "risk", label_id=label_id)
        if row["task_type"] not in {"functional_proxy", "longitudinal_baseline"}:
            _issue(issues, "error", "invalid_task_type", "risk", label_id=label_id)
        factors = row["risk_factors"]
        if not isinstance(factors, list) or not all(
            isinstance(value, str) and value for value in factors
        ):
            _issue(issues, "error", "invalid_risk_factors", "risk", label_id=label_id)
        manifest_row = _validate_label_manifest_link(row, manifest, "risk", label_id, issues)
        if not _finite_number(row["start_time"]) or not _finite_number(row["end_time"]):
            _issue(issues, "error", "invalid_time", "risk", label_id=label_id)
        elif row["start_time"] < 0 or row["end_time"] < row["start_time"]:
            _issue(issues, "error", "invalid_time", "risk", label_id=label_id)
        _require_review(label_id, review_evidence, "risk", issues)
        if mode == "formal":
            _formal_common(row, manifest_row, config, "risk", label_id, issues)


def _validate_profiles(
    profiles: Mapping[str, Any],
    config: ValidationConfig,
    mode: str,
    issues: list[dict[str, Any]],
) -> None:
    if not profiles:
        return
    if set(profiles) != {"schema_version", "subjects"}:
        _issue(issues, "error", "invalid_profile_schema", "profiles")
        return
    if profiles.get("schema_version") != "fall-risk-subject-profiles-v1":
        _issue(issues, "error", "invalid_profile_schema", "profiles")
    subjects = profiles.get("subjects")
    if not isinstance(subjects, list):
        _issue(issues, "error", "invalid_profile_schema", "profiles")
        return
    seen: set[str] = set()
    allowed = {
        "subject_id",
        "profile_version",
        "profile_source",
        "review_status",
        "consent_id",
        "features",
        "note",
    }
    required = allowed - {"note"}
    for position, subject in enumerate(subjects, 1):
        if not isinstance(subject, dict) or not _validate_shape(
            subject, required, allowed, "profile", position, issues
        ):
            continue
        subject_id = subject["subject_id"]
        if not _is_pseudonymous_identifier(subject_id) or subject_id == "unknown":
            _issue(issues, "error", "invalid_subject_id", "profile", record_index=position)
        elif subject_id in seen:
            _issue(issues, "error", "duplicate_subject_id", "profile", record_index=position)
        else:
            seen.add(subject_id)
        if not _is_pseudonymous_identifier(subject["profile_version"]):
            _issue(issues, "error", "invalid_profile_version", "profile", record_index=position)
        if not _is_pseudonymous_identifier(subject["profile_source"]):
            _issue(issues, "error", "invalid_profile_source", "profile", record_index=position)
        if subject["review_status"] not in REVIEW_STATUSES:
            _issue(
                issues,
                "error",
                "invalid_profile_review_status",
                "profile",
                record_index=position,
            )
        consent_id = subject["consent_id"]
        if consent_id is not None and not _is_pseudonymous_identifier(consent_id):
            _issue(issues, "error", "invalid_consent_id", "profile", record_index=position)
        if mode == "formal":
            if subject["review_status"] not in config.formal_review_statuses:
                _issue(
                    issues,
                    "blocker",
                    "formal_profile_review_status",
                    "profile",
                    record_index=position,
                )
            if consent_id is None:
                _issue(
                    issues,
                    "blocker",
                    "formal_consent_missing",
                    "profile",
                    record_index=position,
                )
        if not isinstance(subject["features"], dict) or not _valid_profile_value(
            subject["features"]
        ):
            _issue(issues, "error", "invalid_profile_features", "profile", record_index=position)
        if _contains_contact_identifier(subject.get("note", "")) or _contains_contact_identifier(
            subject["features"]
        ):
            _issue(issues, "error", "potential_identity_data", "profile", record_index=position)


def _validate_shape(
    row: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
    kind: str,
    position: int,
    issues: list[dict[str, Any]],
) -> bool:
    missing = required - set(row)
    extra = set(row) - allowed
    if missing:
        _issue(
            issues,
            "error",
            "missing_fields",
            kind,
            record_index=position,
            message=",".join(sorted(missing)),
        )
    if extra:
        _issue(
            issues,
            "error",
            "unknown_fields",
            kind,
            record_index=position,
            message=",".join(sorted(extra)),
        )
    return not missing and not extra


def _register_label_id(
    row: dict[str, Any],
    index: dict[str, dict[str, Any]],
    kind: str,
    position: int,
    issues: list[dict[str, Any]],
) -> str | None:
    label_id = row.get("label_id")
    if not isinstance(label_id, str) or not label_id:
        _issue(issues, "error", "invalid_id", kind, record_index=position)
        return None
    if label_id in index:
        _issue(issues, "error", "duplicate_label_id", kind, label_id=label_id)
        return None
    index[label_id] = row
    return label_id


def _validate_label_manifest_link(
    row: Mapping[str, Any],
    manifest: Mapping[str, dict[str, Any]],
    kind: str,
    label_id: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    reference = row.get("asset_id") if kind == "risk" else row.get("video_id")
    manifest_row = manifest.get(reference) if isinstance(reference, str) else None
    if manifest_row is None:
        _issue(
            issues,
            "error",
            "missing_manifest_asset" if kind == "risk" else "missing_manifest_video",
            kind,
            label_id=label_id,
            video_id=str(row.get("video_id") or ""),
        )
        return None
    if row.get("asset_id") is not None and row.get("asset_id") != manifest_row.get("asset_id"):
        _issue(issues, "error", "manifest_asset_mismatch", kind, label_id=label_id)
    if (
        kind == "risk"
        and "video_id" in row
        and row.get("video_id") != manifest_row.get("video_id")
    ):
        _issue(issues, "error", "manifest_video_mismatch", kind, label_id=label_id)
    if kind == "action" and row.get("file_path") != manifest_row.get("path"):
        _issue(issues, "error", "manifest_path_mismatch", kind, label_id=label_id)
    media_path = Path(str(manifest_row.get("path", "")))
    if not media_path.is_file():
        _issue(issues, "error", "missing_media_file", kind, label_id=label_id)
    return manifest_row


def _validate_window(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, Any] | None,
    config: ValidationConfig,
    kind: str,
    label_id: str,
    issues: list[dict[str, Any]],
) -> None:
    if row.get("frame_index_base") != 0:
        _issue(issues, "error", "invalid_frame_index_base", kind, label_id=label_id)
    start_frame = row.get("start_frame")
    end_frame = row.get("end_frame")
    start_time = row.get("start_time")
    end_time = row.get("end_time")
    if not _exact_int(start_frame) or not _exact_int(end_frame):
        _issue(issues, "error", "invalid_frame", kind, label_id=label_id)
        return
    if start_frame < 0 or end_frame < start_frame:
        _issue(issues, "error", "invalid_frame", kind, label_id=label_id)
        return
    if not _finite_number(start_time) or not _finite_number(end_time):
        _issue(issues, "error", "invalid_time", kind, label_id=label_id)
        return
    if start_time < 0 or end_time < start_time:
        _issue(issues, "error", "invalid_time", kind, label_id=label_id)
    if manifest_row is None:
        return
    try:
        frame_count = int(manifest_row["frame_count"])
        fps_num = int(manifest_row["fps_num"])
        fps_den = int(manifest_row["fps_den"])
        duration = float(manifest_row["duration_sec"])
    except (KeyError, TypeError, ValueError):
        _issue(issues, "error", "invalid_manifest_timeline", kind, label_id=label_id)
        return
    if frame_count <= 0 or fps_num <= 0 or fps_den <= 0 or end_frame >= frame_count:
        _issue(issues, "error", "frame_out_of_bounds", kind, label_id=label_id)
        return
    expected_start = start_frame * fps_den / fps_num
    expected_end = end_frame * fps_den / fps_num
    if (
        abs(float(start_time) - expected_start) > config.time_tolerance_sec
        or abs(float(end_time) - expected_end) > config.time_tolerance_sec
    ):
        _issue(issues, "error", "time_frame_mismatch", kind, label_id=label_id)
    if float(end_time) > duration + config.time_tolerance_sec:
        _issue(issues, "error", "time_out_of_bounds", kind, label_id=label_id)


def _validate_source_file(
    row: Mapping[str, Any],
    kind: str,
    label_id: str,
    source_hash_cache: dict[Path, str],
    issues: list[dict[str, Any]],
) -> None:
    source_path = Path(str(row.get("source_annotation_path", "")))
    if not source_path.is_file():
        _issue(issues, "error", "missing_source_annotation", kind, label_id=label_id)
        return
    expected = row.get("source_annotation_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        _issue(issues, "error", "invalid_source_checksum", kind, label_id=label_id)
        return
    digest = source_hash_cache.get(source_path)
    if digest is None:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_hash_cache[source_path] = digest
    if digest != expected:
        _issue(issues, "error", "source_checksum_mismatch", kind, label_id=label_id)


def _validate_bbox(
    value: Any, kind: str, label_id: str, issues: list[dict[str, Any]]
) -> None:
    if not isinstance(value, list) or len(value) != 4 or not all(
        _finite_number(item) for item in value
    ):
        _issue(issues, "error", "invalid_bbox", kind, label_id=label_id)
        return
    if value[2] < value[0] or value[3] < value[1]:
        _issue(issues, "error", "invalid_bbox", kind, label_id=label_id)


def _formal_common(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, Any] | None,
    config: ValidationConfig,
    kind: str,
    label_id: str,
    issues: list[dict[str, Any]],
) -> None:
    status = str(row.get("review_status", "missing"))
    if status in config.ineligible_review_statuses or status not in config.formal_review_statuses:
        _issue(issues, "blocker", "formal_review_status", kind, label_id=label_id)
    if row.get("eligibility") is not True:
        _issue(issues, "blocker", "formal_label_ineligible", kind, label_id=label_id)
    if manifest_row is None:
        return
    license_id = str(manifest_row.get("license_id") or "").lower()
    if manifest_row.get("eligibility") is not True or license_id in config.unknown_license_values:
        _issue(
            issues,
            "blocker",
            "formal_manifest_ineligible",
            kind,
            label_id=label_id,
        )


def _formal_partial_record(
    row: Mapping[str, Any],
    kind: str,
    position: int,
    config: ValidationConfig,
    issues: list[dict[str, Any]],
) -> None:
    label_id = row.get("label_id") if isinstance(row.get("label_id"), str) else None
    status = str(row.get("review_status", "missing"))
    if status in config.ineligible_review_statuses or status not in config.formal_review_statuses:
        _issue(
            issues,
            "blocker",
            "formal_review_status",
            kind,
            record_index=position,
            label_id=label_id,
        )
    if row.get("event_type") == "uncertain" or row.get("action_id") == "U01":
        _issue(
            issues,
            "blocker",
            "formal_uncertain",
            kind,
            record_index=position,
            label_id=label_id,
        )
    high_risk_action = (
        kind == "action"
        and str(row.get("action_id", ""))[:1]
        in config.high_risk_action_prefixes
    )
    fall_event = kind == "event" and row.get("event_type") == "fall"
    if high_risk_action or fall_event:
        _issue(
            issues,
            "blocker",
            "missing_review_evidence",
            kind,
            record_index=position,
            label_id=label_id,
        )


def _require_review(
    label_id: str,
    review_evidence: Mapping[tuple[str, str], set[str]],
    kind: str,
    issues: list[dict[str, Any]],
) -> None:
    if not review_evidence.get((kind, label_id)):
        _issue(
            issues,
            "blocker",
            "missing_review_evidence",
            kind,
            label_id=label_id,
        )


def _validate_eligibility_and_reviews(
    row: Mapping[str, Any],
    kind: str,
    label_id: str,
    review_evidence: Mapping[tuple[str, str], set[str]],
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(row.get("eligibility"), bool):
        _issue(issues, "error", "invalid_eligibility", kind, label_id=label_id)
    declared = row.get("review_evidence_ids")
    if not isinstance(declared, list) or any(
        not isinstance(value, str) or not value for value in declared
    ):
        _issue(issues, "error", "invalid_review_references", kind, label_id=label_id)
        return
    if len(set(declared)) != len(declared):
        _issue(issues, "error", "duplicate_review_reference", kind, label_id=label_id)
    actual = review_evidence.get((kind, label_id), set())
    if set(declared) != actual:
        _issue(issues, "error", "review_reference_mismatch", kind, label_id=label_id)


def _sha256_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_pseudonymous_identifier(value: Any, *, allow_unknown: bool = False) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.lower() == "unknown":
        return allow_unknown
    return bool(_PSEUDONYM_PATTERN.fullmatch(value)) and not _contains_contact_identifier(
        value
    )


def _contains_contact_identifier(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_EMAIL_PATTERN.search(value) or _PHONE_PATTERN.search(value))
    if isinstance(value, Mapping):
        return any(
            _contains_contact_identifier(key) or _contains_contact_identifier(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_contact_identifier(item) for item in value)
    return False


def _valid_profile_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_valid_profile_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_pseudonymous_identifier(key) and _valid_profile_value(item)
            for key, item in value.items()
        )
    return False


def _distributions(
    manifest: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    known_subjects = sum(
        1 for row in manifest if row.get("subject_id") not in {None, "", "unknown"}
    )
    subject_sizes = Counter(
        str(row.get("subject_id"))
        for row in manifest
        if row.get("subject_id") not in {None, "", "unknown"}
    )
    source_group_sizes = Counter(
        str(row.get("source_group_id", "missing")) for row in manifest
    )
    original_event_sizes = Counter(
        str(row.get("original_event_id", "missing")) for row in manifest
    )
    review_decisions = Counter(str(row.get("decision", "missing")) for row in reviews)
    conflict_count = review_decisions.get("conflict", 0)
    adjudication_count = review_decisions.get("adjudicate", 0)
    return {
        "dataset": _counter(manifest, "dataset"),
        "scene": _counter(actions, "scene"),
        "scene_region": _counter(manifest, "scene_region"),
        "action_id": _counter(actions, "action_id"),
        "event_type": _counter(events, "event_type"),
        "label_source": _counter(events, "label_source"),
        "risk_task_type": _counter(risks, "task_type"),
        "quality": _counter(actions, "quality"),
        "review_decision": dict(sorted(review_decisions.items())),
        "review_label_type": _counter(reviews, "label_type"),
        "issue_code": _counter(issues, "code"),
        "review_status": dict(
            sorted(
                Counter(
                    str(row.get("review_status", "missing"))
                    for row in [*actions, *events, *risks]
                ).items()
            )
        ),
        "subject_coverage": {
            "known": known_subjects,
            "unknown": len(manifest) - known_subjects,
            "unique_known": len(subject_sizes),
        },
        "group_coverage": {
            "unique_source_groups": len(source_group_sizes),
            "unique_original_events": len(original_event_sizes),
        },
        "subject_group_size": _size_distribution(subject_sizes),
        "source_group_size": _size_distribution(source_group_sizes),
        "original_event_group_size": _size_distribution(original_event_sizes),
        "adjudication": {
            "conflict_records": conflict_count,
            "adjudication_records": adjudication_count,
            "adjudication_per_conflict": (
                adjudication_count / conflict_count if conflict_count else None
            ),
        },
    }


def _counter(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get(field, "missing")) for row in rows).items())
    )


def _size_distribution(group_sizes: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(Counter(str(size) for size in group_sizes.values()).items()))


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    record_type: str,
    *,
    record_index: int | None = None,
    label_id: str | None = None,
    video_id: str | None = None,
    message: str | None = None,
) -> None:
    issue: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "record_type": record_type,
    }
    if record_index is not None:
        issue["record_index"] = record_index
    if label_id:
        issue["label_id"] = label_id
    if video_id:
        issue["video_id"] = video_id
    if message:
        issue["message"] = message
    issues.append(issue)


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
