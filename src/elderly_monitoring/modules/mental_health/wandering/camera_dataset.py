"""Anonymous camera-development data contracts and readiness primitives.

This module validates caller-provided metadata, tracking JSONL, and human
annotations.  It deliberately does not decode media, run a detector/tracker,
create authorization receipts, or infer labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_camera_inputs,
    validate_media_sidecar,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
)


COLLECTION_CONFIG_SCHEMA_VERSION = "wandering-camera-collection-config-v1"
AUTHORIZATION_RECEIPT_SCHEMA_VERSION = "wandering-camera-authorization-receipt-v1"
COLLECTION_SCHEMA_VERSION = "wandering-camera-collection-v1"
ANNOTATION_SCHEMA_VERSION = "wandering-camera-episode-annotation-v1"
READINESS_SCHEMA_VERSION = "wandering-camera-readiness-v1"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_MARKERS = (
    "token=",
    "access_token",
    "authorization:",
    "bearer ",
    "password",
    "secret",
    "cookie:",
    "rtsp://",
    "http://",
    "https://",
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "approval_status",
        "active",
        "purpose",
        "dataset_role",
        "valid_from",
        "expires_at",
        "allowed_operations",
        "participant_ids",
        "session_ids",
        "camera_setup_ids",
        "source_group_ids",
        "governance",
    }
)
_GOVERNANCE_FIELDS = frozenset(
    {
        "consent_confirmed",
        "deidentified_storage",
        "access_control_confirmed",
        "retention_and_deletion_defined",
        "withdrawal_process_defined",
        "audio_policy",
    }
)
_COLLECTION_FIELDS = frozenset(
    {
        "schema_version",
        "collection_id",
        "dataset_role",
        "authorization_receipt_id",
        "participants",
        "camera_setups",
        "sessions",
        "sources",
        "tracklet_participant_bindings",
        "participant_present_intervals",
        "clock_alignments",
    }
)
_ANNOTATION_FIELDS = frozenset(
    {
        "annotation_id",
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
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
        "annotator_id",
        "reviewed_by",
    }
)


class CameraDatasetError(ValueError):
    """Camera collection, receipt, or annotation contract failed closed."""


def load_camera_collection_config(path: str | Path) -> dict[str, Any]:
    """Load the exact M0-CAM-RD collection configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraDatasetError("cannot read camera collection config") from exc
    if value != _expected_collection_config():
        raise CameraDatasetError("camera collection config fields have drifted")
    return value


def validate_authorization_receipt(
    value: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    operation: str,
    participant_ids: set[str] | None = None,
    session_ids: set[str] | None = None,
    camera_setup_ids: set[str] | None = None,
    source_group_ids: set[str] | None = None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Validate a C0 fact supplied by the responsible authorization process.

    The returned value is a repository-safe summary: it contains no allowed-ID
    lists, signatures, credentials, identity, media paths, or authorization
    headers.
    """

    if not isinstance(value, Mapping) or frozenset(value) != _RECEIPT_FIELDS:
        raise CameraDatasetError("authorization receipt fields must match the exact v1 set")
    if value.get("schema_version") != AUTHORIZATION_RECEIPT_SCHEMA_VERSION:
        raise CameraDatasetError("authorization receipt schema_version is invalid")
    expected = config.get("authorization", {})
    if value.get("approval_status") != expected.get("required_approval_status"):
        raise CameraDatasetError("authorization receipt is not approved")
    if value.get("active") is not expected.get("required_active"):
        raise CameraDatasetError("authorization receipt is not active")
    if value.get("purpose") != expected.get("required_purpose"):
        raise CameraDatasetError("authorization receipt purpose is not camera development")
    if value.get("dataset_role") != expected.get("required_dataset_role"):
        raise CameraDatasetError("authorization receipt is not development-only")
    receipt_id = _safe_token(value.get("receipt_id"), "receipt_id")
    allowed_operations = _unique_tokens(value.get("allowed_operations"), "allowed_operations")
    if operation not in expected.get("allowed_operations", []) or operation not in allowed_operations:
        raise CameraDatasetError("authorization receipt does not allow the requested operation")

    valid_from = _parse_timestamp(value.get("valid_from"), "valid_from")
    expires_at = _parse_timestamp(value.get("expires_at"), "expires_at")
    observed_now = _parse_timestamp(now, "now") if isinstance(now, str) else now
    if observed_now is None:
        observed_now = datetime.now().astimezone()
    if observed_now.tzinfo is None:
        raise CameraDatasetError("authorization time must include timezone")
    if valid_from > observed_now:
        raise CameraDatasetError("authorization receipt is not yet active")
    if observed_now >= expires_at:
        raise CameraDatasetError("authorization receipt has expired")
    if valid_from >= expires_at:
        raise CameraDatasetError("authorization receipt validity interval is invalid")

    allowed_scopes = {
        "participant_ids": set(_unique_tokens(value.get("participant_ids"), "participant_ids")),
        "session_ids": set(_unique_tokens(value.get("session_ids"), "session_ids")),
        "camera_setup_ids": set(
            _unique_tokens(value.get("camera_setup_ids"), "camera_setup_ids")
        ),
        "source_group_ids": set(_unique_tokens(value.get("source_group_ids"), "source_group_ids")),
    }
    requested = {
        "participant_ids": participant_ids,
        "session_ids": session_ids,
        "camera_setup_ids": camera_setup_ids,
        "source_group_ids": source_group_ids,
    }
    for name, identifiers in requested.items():
        if identifiers is not None and not set(identifiers).issubset(allowed_scopes[name]):
            raise CameraDatasetError(f"authorization receipt scope mismatch: {name}")

    governance = value.get("governance")
    if not isinstance(governance, Mapping) or frozenset(governance) != _GOVERNANCE_FIELDS:
        raise CameraDatasetError("authorization receipt governance fields are invalid")
    for field in _GOVERNANCE_FIELDS - {"audio_policy"}:
        if governance.get(field) is not True:
            raise CameraDatasetError(f"authorization governance requirement failed: {field}")
    if governance.get("audio_policy") != config.get("privacy", {}).get("audio_policy"):
        raise CameraDatasetError("authorization audio policy is invalid")

    canonical = canonical_json_bytes(dict(value))
    return {
        "schema_version": "wandering-camera-authorization-summary-v1",
        "receipt_id": receipt_id,
        "receipt_sha256": hashlib.sha256(canonical).hexdigest(),
        "approved": True,
        "active_at_validation": True,
        "purpose": "camera_development",
        "dataset_role": "development",
        "operation": operation,
        "scope_counts": {name: len(items) for name, items in sorted(allowed_scopes.items())},
        "valid_from": valid_from.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


def validate_collection_manifest(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate anonymous collection/session/group mappings and cross-references."""

    if not isinstance(value, Mapping) or frozenset(value) != _COLLECTION_FIELDS:
        raise CameraDatasetError("collection fields must match the exact v1 set")
    if value.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise CameraDatasetError("collection schema_version is invalid")
    output = dict(value)
    output["collection_id"] = _safe_token(value.get("collection_id"), "collection_id")
    if value.get("dataset_role") not in config.get("dataset_roles", []):
        raise CameraDatasetError("collection dataset_role is invalid")
    output["authorization_receipt_id"] = _safe_token(
        value.get("authorization_receipt_id"), "authorization_receipt_id"
    )

    participants = _mapping_list(value.get("participants"), "participants")
    participant_ids = _unique_mapping_ids(participants, "participant_id")
    for row in participants:
        _exact(row, {"participant_id"}, "participant")

    setups = _mapping_list(value.get("camera_setups"), "camera_setups")
    setup_ids = _unique_mapping_ids(setups, "camera_setup_id")
    adapter_setup_by_id: dict[str, str] = {}
    for row in setups:
        _exact(row, {"camera_setup_id", "setup_id"}, "camera_setup")
        adapter_setup_by_id[_safe_token(row["camera_setup_id"], "camera_setup_id")] = _safe_token(
            row["setup_id"], "setup_id"
        )

    sessions = _mapping_list(value.get("sessions"), "sessions")
    session_ids = _unique_mapping_ids(sessions, "session_id")
    session_by_id: dict[str, Mapping[str, Any]] = {}
    for row in sessions:
        _exact(
            row,
            {"session_id", "participant_id", "source_group_id", "camera_setup_ids"},
            "session",
        )
        session_id = _safe_token(row["session_id"], "session_id")
        participant_id = _safe_token(row["participant_id"], "participant_id")
        if participant_id not in participant_ids:
            raise CameraDatasetError("session participant cross-reference is invalid")
        session_setups = set(_unique_tokens(row["camera_setup_ids"], "camera_setup_ids"))
        if not session_setups or not session_setups.issubset(setup_ids):
            raise CameraDatasetError("session camera_setup cross-reference is invalid")
        _safe_token(row["source_group_id"], "source_group_id")
        session_by_id[session_id] = row

    sources = _mapping_list(value.get("sources"), "sources")
    source_ids = _unique_mapping_ids(sources, "source_video_id")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for row in sources:
        _exact(
            row,
            {
                "source_video_id",
                "session_id",
                "source_group_id",
                "camera_setup_id",
                "device_id",
                "setup_id",
                "stream_epoch",
                "tracking_ref",
                "media_sidecar_ref",
            },
            "source",
        )
        source_id = _safe_token(row["source_video_id"], "source_video_id")
        session_id = _safe_token(row["session_id"], "session_id")
        camera_setup_id = _safe_token(row["camera_setup_id"], "camera_setup_id")
        if session_id not in session_by_id or camera_setup_id not in setup_ids:
            raise CameraDatasetError("source session/camera_setup cross-reference is invalid")
        session = session_by_id[session_id]
        if row["source_group_id"] != session["source_group_id"]:
            raise CameraDatasetError("source_group isolation mismatch")
        if camera_setup_id not in session["camera_setup_ids"]:
            raise CameraDatasetError("source camera_setup is not part of its session")
        if row["setup_id"] != adapter_setup_by_id[camera_setup_id]:
            raise CameraDatasetError("source adapter setup_id cross-reference is invalid")
        _safe_token(row["device_id"], "device_id")
        _safe_token(row["stream_epoch"], "stream_epoch")
        _safe_relative_ref(row["tracking_ref"], "tracking_ref")
        _safe_relative_ref(row["media_sidecar_ref"], "media_sidecar_ref")
        source_by_id[source_id] = row

    clock_domains: dict[str, Mapping[str, Any]] = {}
    aligned_sources: set[str] = set()
    for row in _mapping_list(value.get("clock_alignments"), "clock_alignments"):
        _exact(
            row,
            {
                "clock_domain_id",
                "session_id",
                "source_group_id",
                "camera_setup_ids",
                "source_video_ids",
                "status",
            },
            "clock_alignment",
        )
        clock_domain_id = _safe_token(row["clock_domain_id"], "clock_domain_id")
        session_id = _safe_token(row["session_id"], "session_id")
        if session_id not in session_by_id or clock_domain_id in clock_domains:
            raise CameraDatasetError("duplicate or invalid clock alignment domain")
        if row["source_group_id"] != session_by_id[session_id]["source_group_id"]:
            raise CameraDatasetError("clock alignment source_group scope mismatch")
        if row["status"] not in {"aligned", "not_aligned", "single_camera"}:
            raise CameraDatasetError("clock alignment status is invalid")
        aligned_setups = set(_unique_tokens(row["camera_setup_ids"], "camera_setup_ids"))
        source_scope = set(_unique_tokens(row["source_video_ids"], "source_video_ids"))
        if not aligned_setups.issubset(set(session_by_id[session_id]["camera_setup_ids"])):
            raise CameraDatasetError("clock alignment camera_setup scope mismatch")
        if source_scope & aligned_sources:
            raise CameraDatasetError("source_video_id appears in multiple clock domains")
        for source_id in source_scope:
            source = source_by_id.get(source_id)
            if (
                source is None
                or source["session_id"] != session_id
                or source["camera_setup_id"] not in aligned_setups
            ):
                raise CameraDatasetError("clock alignment source scope mismatch")
        aligned_sources.update(source_scope)
        clock_domains[clock_domain_id] = row

    binding_keys: set[tuple[str, str, str, str, str, int]] = set()
    for row in _mapping_list(
        value.get("tracklet_participant_bindings"), "tracklet_participant_bindings"
    ):
        _exact(
            row,
            {
                "source_group_id",
                "source_video_id",
                "device_id",
                "setup_id",
                "stream_epoch",
                "track_id",
                "participant_id",
                "session_id",
                "camera_setup_id",
                "clock_domain_id",
            },
            "binding",
        )
        source_id = _safe_token(row["source_video_id"], "source_video_id")
        participant_id = _safe_token(row["participant_id"], "participant_id")
        session_id = _safe_token(row["session_id"], "session_id")
        camera_setup_id = _safe_token(row["camera_setup_id"], "camera_setup_id")
        clock_domain_id = _safe_token(row["clock_domain_id"], "clock_domain_id")
        track_id = _nonnegative_int(row["track_id"], "track_id")
        source = source_by_id.get(source_id)
        domain = clock_domains.get(clock_domain_id)
        if source is None or participant_id not in participant_ids or session_id not in session_by_id:
            raise CameraDatasetError("tracklet binding cross-reference is invalid")
        if participant_id != session_by_id[session_id]["participant_id"]:
            raise CameraDatasetError("tracklet binding participant/session mismatch")
        expected = {
            "source_group_id": source["source_group_id"],
            "device_id": source["device_id"],
            "setup_id": source["setup_id"],
            "stream_epoch": source["stream_epoch"],
            "session_id": source["session_id"],
            "camera_setup_id": source["camera_setup_id"],
        }
        if any(row[field] != expected[field] for field in expected):
            raise CameraDatasetError("tracklet binding source/session/setup scope mismatch")
        if (
            domain is None
            or domain["session_id"] != session_id
            or source_id not in domain["source_video_ids"]
            or camera_setup_id not in domain["camera_setup_ids"]
        ):
            raise CameraDatasetError("tracklet binding clock domain mismatch")
        key = (
            str(row["source_group_id"]),
            source_id,
            str(row["device_id"]),
            str(row["setup_id"]),
            str(row["stream_epoch"]),
            track_id,
        )
        if key in binding_keys:
            raise CameraDatasetError("duplicate tracklet binding")
        binding_keys.add(key)

    for row in _mapping_list(
        value.get("participant_present_intervals"), "participant_present_intervals"
    ):
        _exact(
            row,
            {
                "participant_id",
                "session_id",
                "clock_domain_id",
                "start_sec",
                "end_sec_exclusive",
            },
            "presence",
        )
        participant_id = _safe_token(row["participant_id"], "participant_id")
        session_id = _safe_token(row["session_id"], "session_id")
        clock_domain_id = _safe_token(row["clock_domain_id"], "clock_domain_id")
        if (
            session_id not in session_by_id
            or participant_id != session_by_id[session_id]["participant_id"]
            or clock_domain_id not in clock_domains
            or clock_domains[clock_domain_id]["session_id"] != session_id
        ):
            raise CameraDatasetError("participant presence cross-reference is invalid")
        _interval(row)

    # Reject sensitive strings even in values whose individual syntax passed.
    _reject_sensitive_tree(value)
    return output


def validate_episode_annotations(
    rows: Sequence[Mapping[str, Any]], collection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate human-supplied C2 episode annotations without inferring labels."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise CameraDatasetError("annotations must be a sequence")
    participant_ids = {row["participant_id"] for row in collection["participants"]}
    session_by_id = {row["session_id"]: row for row in collection["sessions"]}
    setup_ids = {row["camera_setup_id"] for row in collection["camera_setups"]}
    source_by_id = {row["source_video_id"]: row for row in collection["sources"]}
    binding_by_scope = {
        (
            row["source_group_id"],
            row["source_video_id"],
            row["device_id"],
            row["setup_id"],
            row["stream_epoch"],
            row["track_id"],
        ): row
        for row in collection["tracklet_participant_bindings"]
    }
    seen_ids: set[str] = set()
    intervals: dict[tuple[str, ...], list[tuple[float, float]]] = {}
    output: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or frozenset(raw) != _ANNOTATION_FIELDS:
            raise CameraDatasetError("annotation fields must match the exact section 10.3 set")
        row = dict(raw)
        annotation_id = _safe_token(row["annotation_id"], "annotation_id")
        if annotation_id in seen_ids:
            raise CameraDatasetError("duplicate or overlapping annotation is forbidden")
        seen_ids.add(annotation_id)
        for field in (
            "source_group_id",
            "source_video_id",
            "device_id",
            "setup_id",
            "stream_epoch",
            "participant_id",
            "session_id",
            "camera_setup_id",
            "clock_domain_id",
            "script_type",
            "visibility_quality",
            "tracking_issue",
            "annotator_id",
            "reviewed_by",
        ):
            _safe_token(row[field], field)
        source = source_by_id.get(row["source_video_id"])
        session = session_by_id.get(row["session_id"])
        track_id = _nonnegative_int(row["track_id"], "track_id")
        binding = binding_by_scope.get(
            (
                row["source_group_id"],
                row["source_video_id"],
                row["device_id"],
                row["setup_id"],
                row["stream_epoch"],
                track_id,
            )
        )
        if source is None or session is None:
            raise CameraDatasetError("annotation source/session cross-reference is invalid")
        if row["participant_id"] not in participant_ids or row["camera_setup_id"] not in setup_ids:
            raise CameraDatasetError("annotation participant/camera_setup cross-reference is invalid")
        if (
            source["session_id"] != row["session_id"]
            or source["camera_setup_id"] != row["camera_setup_id"]
            or session["participant_id"] != row["participant_id"]
            or source["source_group_id"] != row["source_group_id"]
            or source["device_id"] != row["device_id"]
            or source["setup_id"] != row["setup_id"]
            or source["stream_epoch"] != row["stream_epoch"]
            or binding is None
            or binding["participant_id"] != row["participant_id"]
            or binding["session_id"] != row["session_id"]
            or binding["camera_setup_id"] != row["camera_setup_id"]
            or binding["clock_domain_id"] != row["clock_domain_id"]
        ):
            raise CameraDatasetError("annotation group isolation cross-reference is invalid")
        start, end = _interval(row)
        key = (
            str(row["source_video_id"]),
            str(row["participant_id"]),
            str(row["stream_epoch"]),
            str(row["track_id"]),
        )
        if any(start < prior_end and end > prior_start for prior_start, prior_end in intervals.setdefault(key, [])):
            raise CameraDatasetError("overlapping annotations for one source/participant are forbidden")
        intervals[key].append((start, end))

        if row["observable_pattern"] not in {"direct", "pacing", "lapping", "random", "unknown"}:
            raise CameraDatasetError("observable_pattern is invalid")
        if row["purpose_context"] not in {"purposeful", "nonpurposeful", "unknown"}:
            raise CameraDatasetError("purpose_context is invalid")
        if row["purpose_evidence"] not in {
            "scripted",
            "participant_report",
            "observed_context",
            "unknown",
        }:
            raise CameraDatasetError("purpose_evidence is invalid")
        if (row["purpose_context"] == "unknown") != (row["purpose_evidence"] == "unknown"):
            raise CameraDatasetError("purpose context/evidence must remain explicit and consistent")
        if row["evaluation_role"] not in {
            "wandering_like_positive",
            "purposeful_hard_negative",
            "ordinary_negative",
            "uncertain",
            "excluded",
        }:
            raise CameraDatasetError("evaluation_role is invalid")
        if row["annotation_status"] not in {"accepted", "uncertain", "excluded"}:
            raise CameraDatasetError("annotation_status is invalid")
        expected_status = {
            "uncertain": "uncertain",
            "excluded": "excluded",
        }.get(str(row["evaluation_role"]), "accepted")
        if row["annotation_status"] != expected_status:
            raise CameraDatasetError("annotation status/evaluation role are inconsistent")
        if row["annotation_status"] == "accepted" and row["observable_pattern"] not in {
            "direct",
            "pacing",
            "lapping",
            "random",
        }:
            raise CameraDatasetError("accepted truth must have a scoreable shape")
        if row["observable_pattern"] == "unknown" and row["annotation_status"] == "accepted":
            raise CameraDatasetError("accepted truth must have a scoreable shape")
        if row["evaluation_role"] == "purposeful_hard_negative":
            if row["purpose_context"] != "purposeful" or row["observable_pattern"] == "direct":
                raise CameraDatasetError("purposeful hard negative must preserve wandering-like shape")
        output.append(row)
    return sorted(output, key=lambda row: (row["source_video_id"], row["start_sec"], row["annotation_id"]))


def build_camera_readiness(
    *,
    config: Mapping[str, Any],
    receipt_summary: Mapping[str, Any] | None = None,
    collection: Mapping[str, Any] | None = None,
    c1_source_count: int = 0,
    annotations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build milestone readiness without scientific performance claims."""

    c0 = bool(receipt_summary and receipt_summary.get("approved") is True)
    c1 = bool(c0 and collection is not None and c1_source_count > 0)
    c2 = bool(c1 and annotations)
    c3 = bool(
        c2
        and collection
        and collection.get("tracklet_participant_bindings")
        and collection.get("participant_present_intervals")
        and collection.get("clock_alignments")
    )
    milestones = {"C0": c0, "C1": c1, "C2": c2, "C3": c3}
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "readiness_status": "ready_for_labeled_development" if all(milestones.values()) else "not_ready",
        "evidence_scope": "synthetic_schema_contract_only" if not c0 else "authorization_metadata_validated",
        "authorized_camera_data_consumed": bool(c1),
        "milestones": milestones,
        "missing_milestones": [name for name in config["readiness_milestones"] if not milestones[name]],
        "m0cam_d_started": False,
        "sealed_camera_accessed": False,
    }


def prepare_camera_input_pair(
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
    *,
    camera_config: str | Path,
) -> dict[str, Any]:
    """Canonicalize a caller-supplied C1 pair; never opens the media itself."""

    output = Path(output_dir)
    if output.exists():
        raise CameraDatasetError(f"output already exists: {output}")
    try:
        loaded_config = load_camera_config(camera_config)
        adapter = load_camera_inputs(tracking_jsonl_path, media_sidecar_path, loaded_config)
    except (CameraInferenceError, CameraAdapterError, OSError, ValueError) as exc:
        raise CameraDatasetError("caller-supplied tracking/sidecar validation failed") from exc
    tracking_payload = canonical_jsonl_bytes(adapter.normalized_rows)
    sidecar_payload = canonical_json_bytes(dict(adapter.media_sidecar))
    receipt = {
        "schema_version": "wandering-camera-input-preparation-v1",
        "input_kind": "caller_supplied_tracking_and_sidecar",
        "media_opened": False,
        "detector_run": False,
        "tracker_run": False,
        "source_tracking_sha256": adapter.source_tracking_sha256,
        "normalized_tracking_sha256": hashlib.sha256(tracking_payload).hexdigest(),
        "media_sidecar_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
        "observation_count": len(adapter.observations),
    }
    _commit_new_directory(
        output,
        {
            "tracking.jsonl": tracking_payload,
            "media_sidecar.json": sidecar_payload,
            "preparation_summary.json": canonical_json_bytes(receipt),
        },
    )
    return receipt


def prepare_authorized_camera_session(
    *,
    receipt_path: str | Path,
    collection_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
    collection_config_path: str | Path,
    camera_config_path: str | Path,
) -> dict[str, Any]:
    """Receipt-gate a production C1 pair before reading protected tracking.

    The external C0 receipt is validated before collection, sidecar, or tracking
    inputs.  Collection scope is then checked before the sidecar and tracking
    pair is opened.  Synthetic authorization remains confined to the historical
    schema-only helper ``prepare_camera_input_pair`` and is not accepted here.
    """

    config = load_camera_collection_config(collection_config_path)
    receipt = _read_json_object(Path(receipt_path), "authorization receipt")
    summary = validate_authorization_receipt(
        receipt,
        config,
        operation="prepare_session",
    )
    collection = validate_collection_manifest(
        _read_json_object(Path(collection_path), "collection"), config
    )
    summary = validate_authorization_receipt(
        receipt,
        config,
        operation="prepare_session",
        **_collection_scope(collection),
    )
    _validate_collection_receipt_reference(collection, summary)

    try:
        camera_config = load_camera_config(camera_config_path)
        sidecar = validate_media_sidecar(
            _read_json_object(Path(media_sidecar_path), "media sidecar"), camera_config
        )
    except (CameraInferenceError, CameraAdapterError) as exc:
        raise CameraDatasetError("authorized media sidecar validation failed") from exc
    _bind_sidecar_to_collection(sidecar, collection)
    if sidecar["authorization_status"] not in {
        "authorized_camera_engineering_smoke",
        "authorized_camera_labeled_evaluation",
    }:
        raise CameraDatasetError("prepare_session production entry rejects synthetic media")

    result = prepare_camera_input_pair(
        tracking_jsonl_path,
        media_sidecar_path,
        output_dir,
        camera_config=camera_config_path,
    )
    return {
        **result,
        "authorization_summary": summary,
        "test_fixture_only": False,
    }


def write_authorized_camera_annotations(
    *,
    receipt_path: str | Path,
    collection_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    collection_config_path: str | Path,
) -> dict[str, Any]:
    """Receipt-gate C2 annotations before reading the protected JSONL."""

    config = load_camera_collection_config(collection_config_path)
    receipt = _read_json_object(Path(receipt_path), "authorization receipt")
    summary = validate_authorization_receipt(
        receipt,
        config,
        operation="validate_annotations",
    )
    collection = validate_collection_manifest(
        _read_json_object(Path(collection_path), "collection"), config
    )
    summary = validate_authorization_receipt(
        receipt,
        config,
        operation="validate_annotations",
        **_collection_scope(collection),
    )
    _validate_collection_receipt_reference(collection, summary)
    result = write_validated_annotations(
        _read_jsonl_objects(Path(annotations_path), "annotations"),
        collection,
        output_path,
    )
    return {
        **result,
        "authorization_summary": summary,
        "test_fixture_only": False,
    }


def write_validated_annotations(
    rows: Sequence[Mapping[str, Any]], collection: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    """Write only validated human-provided annotations to a fresh file."""

    destination = Path(output)
    if destination.exists():
        raise CameraDatasetError(f"output already exists: {destination}")
    validated = validate_episode_annotations(rows, collection)
    payload = canonical_jsonl_bytes(validated)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise CameraDatasetError("cannot write validated annotations") from exc
    return {
        "annotation_count": len(validated),
        "annotations_sha256": hashlib.sha256(payload).hexdigest(),
        "labels_inferred": False,
    }


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraDatasetError(f"cannot parse {role}") from exc
    if not isinstance(value, dict):
        raise CameraDatasetError(f"{role} must be a JSON object")
    return value


def _read_jsonl_objects(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraDatasetError(f"cannot parse {role} JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise CameraDatasetError(f"{role} JSONL rows must be objects")
    return rows


def _collection_scope(collection: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        "participant_ids": {
            str(row["participant_id"]) for row in collection["participants"]
        },
        "session_ids": {str(row["session_id"]) for row in collection["sessions"]},
        "camera_setup_ids": {
            str(row["camera_setup_id"]) for row in collection["camera_setups"]
        },
        "source_group_ids": {
            str(row["source_group_id"]) for row in collection["sessions"]
        },
    }


def _validate_collection_receipt_reference(
    collection: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    if collection["authorization_receipt_id"] != summary["receipt_id"]:
        raise CameraDatasetError("collection authorization receipt reference mismatch")
    if collection["dataset_role"] != "development":
        raise CameraDatasetError("sealed camera collection is forbidden in development")


def _bind_sidecar_to_collection(
    media: Mapping[str, Any], collection: Mapping[str, Any]
) -> None:
    matches = [
        row
        for row in collection["sources"]
        if row["source_video_id"] == media["source_video_id"]
    ]
    if len(matches) != 1:
        raise CameraDatasetError("media source_video_id is outside collection")
    source = matches[0]
    for field in (
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
    ):
        if source[field] != media[field]:
            raise CameraDatasetError(f"media sidecar collection mismatch: {field}")


def _expected_collection_config() -> dict[str, Any]:
    return {
        "schema_version": COLLECTION_CONFIG_SCHEMA_VERSION,
        "purpose": "m0cam_development_data_readiness",
        "dataset_roles": ["development", "sealed"],
        "authorization": {
            "receipt_schema_version": AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
            "required_approval_status": "approved",
            "required_active": True,
            "required_purpose": "camera_development",
            "required_dataset_role": "development",
            "allowed_operations": ["prepare_session", "validate_annotations", "run_development"],
        },
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "media_schema_version": "wandering-media-v1",
        "tracking_schema": "caller_supplied_track_observation_jsonl",
        "readiness_milestones": ["C0", "C1", "C2", "C3"],
        "privacy": {
            "anonymous_ids_only": True,
            "repository_receipt_summary_only": True,
            "forbid_direct_identity": True,
            "forbid_faces": True,
            "forbid_credentials": True,
            "forbid_live_urls": True,
            "forbid_local_absolute_media_paths": True,
            "audio_policy": "not_collected",
        },
    }


def _mapping_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise CameraDatasetError(f"{field} must be a list of objects")
    return value


def _unique_mapping_ids(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    identifiers = [_safe_token(row.get(field), field) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise CameraDatasetError(f"duplicate {field}")
    return set(identifiers)


def _unique_tokens(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CameraDatasetError(f"{field} must be a non-empty list")
    output = [_safe_token(item, field) for item in value]
    if len(output) != len(set(output)):
        raise CameraDatasetError(f"duplicate {field}")
    return output


def _safe_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise CameraDatasetError(f"{field} must be an anonymous safe token")
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        raise CameraDatasetError(f"{field} contains sensitive material")
    return value


def _safe_relative_ref(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CameraDatasetError(f"{field} must be a non-empty relative reference")
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        raise CameraDatasetError(f"{field} contains a URL or credential marker")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
        raise CameraDatasetError(f"{field} must not be an absolute or escaping media path")
    return value


def _reject_sensitive_tree(value: Any) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            raise CameraDatasetError("collection contains a URL, credential, or sensitive marker")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {
                "name",
                "face",
                "faces",
                "token",
                "authorization",
                "headers",
                "local_media_path",
            }:
                raise CameraDatasetError("collection contains a forbidden identity/credential field")
            _reject_sensitive_tree(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_sensitive_tree(item)


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise CameraDatasetError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CameraDatasetError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CameraDatasetError(f"{field} must include timezone")
    return parsed


def _exact(value: Mapping[str, Any], fields: set[str], role: str) -> None:
    if set(value) != fields:
        raise CameraDatasetError(f"{role} fields are invalid")


def _interval(value: Mapping[str, Any]) -> tuple[float, float]:
    start = _finite_nonnegative(value.get("start_sec"), "start_sec")
    end = _finite_nonnegative(value.get("end_sec_exclusive"), "end_sec_exclusive")
    if end <= start:
        raise CameraDatasetError("interval must satisfy start_sec < end_sec_exclusive")
    return start, end


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraDatasetError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise CameraDatasetError(f"{field} must be finite and nonnegative")
    return number


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CameraDatasetError(f"{field} must be a nonnegative integer")
    return value


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise CameraDatasetError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output.exists():
            raise CameraDatasetError(f"output already exists: {output}")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
