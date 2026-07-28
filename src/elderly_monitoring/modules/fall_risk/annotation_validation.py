from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
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
    "schema_version": "fall-risk-label-validation-v2",
    "time_tolerance_sec": 0.001,
}

ACTION_COMMON_REQUIRED = {
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
    "quality",
    "note",
    "source",
}
ACTION_CVAT_REQUIRED = ACTION_COMMON_REQUIRED | {
    "cvat_task_id",
    "cvat_track_id",
    "bbox_start",
    "bbox_end",
}
ACTION_TOAGA_REQUIRED = set(ACTION_COMMON_REQUIRED)
ACTION_NTU_RGBD_REQUIRED = set(ACTION_COMMON_REQUIRED)
ACTION_CVAT_ALLOWED = set(ACTION_CVAT_REQUIRED)
ACTION_TOAGA_ALLOWED = set(ACTION_TOAGA_REQUIRED)
ACTION_NTU_RGBD_ALLOWED = set(ACTION_NTU_RGBD_REQUIRED)

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
    "note",
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
}
RISK_ALLOWED = RISK_REQUIRED | {
    "video_id",
    "note",
    "risk_score",
}

EVENT_TYPES = {event_type for event_type, _ in ACTION_EVENT_MAP.values()} | {
    "sudden_stop_recovery",
    "recovery",
}
_PSEUDONYM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_INTERNAL_AUTHORIZATION_URI_PREFIX = "internal://authorization/"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INTERNAL_AUTHORIZATION_EVIDENCE_TYPES = {
    "cvat_export_archive",
    "local_media_inventory",
}


@dataclass(frozen=True)
class ValidationConfig:
    time_tolerance_sec: float
    schema_version: str = "fall-risk-label-validation-v2"


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
            "than the v2 contract maximum"
        )
    if raw["schema_version"] != DEFAULT_CONFIG["schema_version"]:
        raise ValueError("unsupported validation config schema_version")
    return (
        ValidationConfig(
            schema_version=str(raw["schema_version"]),
            time_tolerance_sec=tolerance,
        ),
        config_sha256,
    )


def validate_fall_risk_data(
    *,
    manifest_path: Path | str,
    action_labels_path: Path | str,
    event_labels_path: Path | str,
    risk_labels_path: Path | str,
    subject_profiles_path: Path | str,
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
    profiles, profiles_input_hash = _load_json_object_strict(
        Path(subject_profiles_path), "profiles", issues
    )

    manifest = _manifest_index(manifest_rows, issues, config)
    source_hash_cache: dict[Path, str] = {}
    action_index = _validate_actions(
        action_rows,
        manifest,
        source_hash_cache,
        config,
        mode,
        issues,
    )
    _validate_events(
        event_rows,
        manifest,
        action_index,
        source_hash_cache,
        config,
        mode,
        issues,
    )
    _validate_risk_labels(risk_rows, manifest, config, mode, issues)
    _validate_profiles(profiles, config, mode, issues)

    counts = {
        "manifest_videos": sum(row.get("video_id") is not None for row in manifest_rows),
        "action_labels": len(action_rows),
        "event_labels": len(event_rows),
        "risk_labels": len(risk_rows),
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
        "schema_version": "fall-risk-label-validation-report-v2",
        "validation_config_version": config.schema_version,
        "mode": mode,
        "valid": valid,
        "input_sha256": {
            "manifest": manifest_input_hash,
            "action_labels": action_input_hash,
            "event_labels": event_input_hash,
            "risk_labels": risk_input_hash,
            "subject_profiles": profiles_input_hash,
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
            exclusion_reasons = row.get("exclusion_reasons")
            has_public_provenance = bool(re.match(r"^https?://", source_uri))
            has_internal_authorization = _has_valid_internal_authorization_provenance(
                row, source_uri
            )
            if (
                not (has_public_provenance or has_internal_authorization)
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


def _has_valid_internal_authorization_provenance(
    row: Mapping[str, Any], source_uri: str
) -> bool:
    authorization = row.get("internal_authorization")
    if not isinstance(authorization, Mapping):
        return False
    required = (
        "authorization_id",
        "approval_reference",
        "approved_at",
        "evidence",
        "authorized_video_count",
    )
    if any(key not in authorization for key in required):
        return False
    authorization_id = authorization.get("authorization_id")
    approval_reference = authorization.get("approval_reference")
    approved_at = authorization.get("approved_at")
    evidence = authorization.get("evidence")
    authorized_video_count = authorization.get("authorized_video_count")
    if not isinstance(evidence, Mapping):
        return False
    evidence_kind = evidence.get("kind")
    evidence_name = evidence.get("name")
    evidence_sha256 = evidence.get("sha256")
    evidence_is_valid = (
        isinstance(evidence_kind, str)
        and evidence_kind in _INTERNAL_AUTHORIZATION_EVIDENCE_TYPES
        and isinstance(evidence_name, str)
        and bool(evidence_name)
        and isinstance(evidence_sha256, str)
        and bool(_SHA256_PATTERN.fullmatch(evidence_sha256))
    )
    if evidence_kind == "local_media_inventory":
        evidence_is_valid = evidence_is_valid and all(
            isinstance(evidence.get(field), str) and bool(evidence[field])
            for field in ("path", "inventory_id")
        )
    return (
        isinstance(authorization_id, str)
        and bool(authorization_id)
        and source_uri == f"{_INTERNAL_AUTHORIZATION_URI_PREFIX}{authorization_id}"
        and isinstance(approval_reference, str)
        and bool(approval_reference)
        and isinstance(approved_at, str)
        and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", approved_at))
        and evidence_is_valid
        and isinstance(authorized_video_count, int)
        and authorized_video_count > 0
        and row.get("provenance_status")
        == "internal_authorized_source_unverified"
    )


def _stable_label_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _validate_actions(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
    source_hash_cache: dict[Path, str],
    config: ValidationConfig,
    mode: str,
    issues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        source = row.get("source")
        required = (
            ACTION_CVAT_REQUIRED
            if source == "cvat"
            else ACTION_NTU_RGBD_REQUIRED
            if source in {"ntu_rgbd_clip_label", "ntu_rgbd_manual_clip_label"}
            else ACTION_TOAGA_REQUIRED
        )
        allowed = (
            ACTION_CVAT_ALLOWED
            if source == "cvat"
            else ACTION_NTU_RGBD_ALLOWED
            if source in {"ntu_rgbd_clip_label", "ntu_rgbd_manual_clip_label"}
            else ACTION_TOAGA_ALLOWED
        )
        shape_valid = _validate_shape(
            row, required, allowed, "action", position, issues
        )
        if not shape_valid:
            if mode == "formal":
                _formal_partial_record(row, "action", position, config, issues)
            continue
        label_id = _register_label_id(row, index, "action", position, issues)
        if label_id is None:
            continue
        if source not in {
            "cvat",
            "toaga_official_walking",
            "ntu_rgbd_clip_label",
            "ntu_rgbd_manual_clip_label",
        }:
            _issue(issues, "error", "invalid_action_source", "action", label_id=label_id)
        action_id = row["action_id"]
        expected_name = ACTION_NAMES.get(action_id)
        if expected_name is None or row["action_name"] != expected_name:
            _issue(issues, "error", "invalid_action_label", "action", label_id=label_id)
        expected_event = ACTION_EVENT_MAP.get(action_id)
        if expected_event is None or row["event_type"] != expected_event[0]:
            _issue(issues, "error", "invalid_action_event_mapping", "action", label_id=label_id)
        if row["quality"] not in QUALITY_VALUES:
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
        if action_id == "U01" and not str(row["note"]).strip():
            _issue(issues, "error", "uncertain_reason_missing", "action", label_id=label_id)
        manifest_row = _validate_label_manifest_link(row, manifest, "action", label_id, issues)
        _validate_window(row, manifest_row, config, "action", label_id, issues)
        _validate_source_file(
            row, "action", label_id, source_hash_cache, issues
        )
        if source == "cvat":
            _validate_bbox(row.get("bbox_start"), "action", label_id, issues)
            _validate_bbox(row.get("bbox_end"), "action", label_id, issues)
        if mode == "formal":
            _formal_common(row, manifest_row, config, "action", label_id, issues)
            if row["event_type"] == "uncertain":
                _issue(issues, "blocker", "formal_uncertain", "action", label_id=label_id)
    return index


def _validate_events(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
    actions: Mapping[str, dict[str, Any]],
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
        if _contains_contact_identifier(row["note"]):
            _issue(
                issues,
                "error",
                "potential_identity_data",
                "event",
                label_id=label_id,
                message="note",
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

        if mode == "formal":
            _formal_common(row, manifest_row, config, "event", label_id, issues)
            if row["event_type"] == "uncertain":
                _issue(issues, "blocker", "formal_uncertain", "event", label_id=label_id)


def _validate_risk_labels(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, dict[str, Any]],
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
    if profiles.get("schema_version") != "fall-risk-subject-profiles-v2":
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
        consent_id = subject["consent_id"]
        if consent_id is not None and not _is_pseudonymous_identifier(consent_id):
            _issue(issues, "error", "invalid_consent_id", "profile", record_index=position)
        if mode == "formal":
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
    if manifest_row is None:
        return
    if manifest_row.get("eligibility") is not True:
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
    if row.get("event_type") == "uncertain" or row.get("action_id") == "U01":
        _issue(
            issues,
            "blocker",
            "formal_uncertain",
            kind,
            record_index=position,
            label_id=label_id,
        )


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
    return {
        "dataset": _counter(manifest, "dataset"),
        "scene": _counter(actions, "scene"),
        "scene_region": _counter(manifest, "scene_region"),
        "action_id": _counter(actions, "action_id"),
        "event_type": _counter(events, "event_type"),
        "label_source": _counter(events, "label_source"),
        "risk_task_type": _counter(risks, "task_type"),
        "quality": _counter(actions, "quality"),
        "issue_code": _counter(issues, "code"),
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
