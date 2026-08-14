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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_camera_inputs,
    validate_media_sidecar,
)
from elderly_monitoring.modules.mental_health.wandering.camera_component import (
    validate_camera_components,
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


@dataclass(frozen=True)
class _VideoPreparationProvenance:
    """Exact private carrier for controller-observed video preparation facts."""

    controller: str
    controller_schema_version: str
    source_video_id: str
    source_sha256: str
    detector_backend: str
    detector_model: str
    detector_version: str
    tracker_backend: str
    tracker_config: str
    tracker_version: str
    confidence_threshold: float
    iou_threshold: float
    max_frames: None
    person_class_id: int
    person_id_prefix: str
    scene_region: str
    raw_observation_count: int
    video_width: int
    video_height: int
    nominal_fps: float
    duration_sec: float
    frame_count: int


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
    """Canonicalize only a synthetic/schema fixture pair; never open media."""

    output = Path(output_dir)
    source_sidecar_path = Path(media_sidecar_path)
    try:
        loaded_config = load_camera_config(camera_config)
        sidecar = validate_media_sidecar(
            _read_json_object(source_sidecar_path, "media sidecar"), loaded_config
        )
        validate_camera_components(sidecar)
        _validate_portable_media_metadata(sidecar)
        source_sidecar_payload = source_sidecar_path.read_bytes()
    except (CameraInferenceError, CameraAdapterError, OSError, ValueError) as exc:
        raise CameraDatasetError("caller-supplied tracking/sidecar validation failed") from exc
    if sidecar["authorization_status"] != "synthetic_fixture":
        raise CameraDatasetError("synthetic helper rejects authorized camera sidecars")
    if os.path.lexists(output):
        raise CameraDatasetError(f"output already exists: {output}")
    try:
        adapter = load_camera_inputs(tracking_jsonl_path, media_sidecar_path, loaded_config)
    except (CameraAdapterError, OSError, ValueError) as exc:
        raise CameraDatasetError("caller-supplied tracking/sidecar validation failed") from exc
    tracking_payload, sidecar_payload, pair_facts = _canonical_pair_payloads(
        adapter, source_sidecar_payload
    )
    summary = {
        "schema_version": "wandering-camera-input-preparation-v1",
        "input_kind": "synthetic_tracking_and_sidecar",
        "media_opened": False,
        "detector_run": False,
        "tracker_run": False,
        "camera_qc_run": False,
        "model_inference_run": False,
        "m0cam_d_started": False,
        "authorization_status": "synthetic_fixture",
        "evidence_scope": "synthetic_schema_contract_only",
        "authorized_camera_data_consumed": False,
        **pair_facts,
    }
    _commit_prepared_pair(
        output=output,
        tracking_payload=tracking_payload,
        sidecar_payload=sidecar_payload,
        summary=summary,
        camera_config=loaded_config,
    )
    return summary


def _canonical_pair_payloads(
    adapter: Any, source_sidecar_payload: bytes
) -> tuple[bytes, bytes, dict[str, Any]]:
    tracking_payload = canonical_jsonl_bytes(adapter.normalized_rows)
    normalized_tracking_sha256 = hashlib.sha256(tracking_payload).hexdigest()
    output_sidecar = dict(adapter.media_sidecar)
    output_sidecar["tracking_jsonl_sha256"] = normalized_tracking_sha256
    sidecar_payload = canonical_json_bytes(output_sidecar)
    output_sidecar_sha256 = hashlib.sha256(sidecar_payload).hexdigest()
    return tracking_payload, sidecar_payload, {
        "source_tracking_sha256": adapter.source_tracking_sha256,
        "normalized_tracking_sha256": normalized_tracking_sha256,
        "normalized_tracking_bytes": len(tracking_payload),
        "source_sidecar_sha256": hashlib.sha256(source_sidecar_payload).hexdigest(),
        "source_sidecar_bytes": len(source_sidecar_payload),
        "output_sidecar_sha256": output_sidecar_sha256,
        "output_sidecar_bytes": len(sidecar_payload),
        "media_sidecar_sha256": output_sidecar_sha256,
        "observation_count": len(adapter.observations),
    }


def _commit_prepared_pair(
    *,
    output: Path,
    tracking_payload: bytes,
    sidecar_payload: bytes,
    summary: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    summary_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    if summary_validator is not None:
        summary_validator(summary)
    summary_payload = canonical_json_bytes(dict(summary))

    def validate_staged_pair(staging: Path) -> None:
        try:
            load_camera_inputs(
                staging / "tracking.jsonl",
                staging / "media_sidecar.json",
                camera_config,
            )
            persisted_summary = _read_json_object(
                staging / "preparation_summary.json", "preparation summary"
            )
        except (CameraAdapterError, OSError, ValueError) as exc:
            raise CameraDatasetError("canonical prepared artifact round-trip failed") from exc
        if persisted_summary != dict(summary):
            raise CameraDatasetError("canonical preparation summary round-trip mismatch")
        if summary_validator is not None:
            summary_validator(persisted_summary)
        if (staging / "tracking.jsonl").read_bytes() != tracking_payload:
            raise CameraDatasetError("staged tracking bytes mismatch")
        if (staging / "media_sidecar.json").read_bytes() != sidecar_payload:
            raise CameraDatasetError("staged sidecar bytes mismatch")
        if (staging / "preparation_summary.json").read_bytes() != summary_payload:
            raise CameraDatasetError("staged preparation summary bytes mismatch")

    _commit_new_directory(
        output,
        {
            "tracking.jsonl": tracking_payload,
            "media_sidecar.json": sidecar_payload,
            "preparation_summary.json": summary_payload,
        },
        validate_staged_pair=validate_staged_pair,
    )


def _authorization_binding(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "receipt_id": summary["receipt_id"],
        "receipt_sha256": summary["receipt_sha256"],
        "approved": summary["approved"],
        "active_at_validation": summary["active_at_validation"],
        "purpose": summary["purpose"],
        "dataset_role": summary["dataset_role"],
        "operation": summary["operation"],
        "valid_from": summary["valid_from"],
        "expires_at": summary["expires_at"],
        "scope_counts": {
            "participants": summary["scope_counts"]["participant_ids"],
            "sessions": summary["scope_counts"]["session_ids"],
            "camera_setups": summary["scope_counts"]["camera_setup_ids"],
            "source_groups": summary["scope_counts"]["source_group_ids"],
        },
    }


def _selected_collection_source(
    collection: Mapping[str, Any], source_video_id: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in collection["sources"]
        if row["source_video_id"] == source_video_id
    ]
    if len(matches) != 1:
        raise CameraDatasetError("source binding must select exactly one collection source")
    return matches[0]


def _selected_unique_collection_source(collection: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = collection.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise CameraDatasetError(
            "pair preparation requires exactly one legal collection source"
        )
    source = sources[0]
    if not isinstance(source, Mapping):
        raise CameraDatasetError("collection source must be an object")
    return source


def _source_binding(
    source: Mapping[str, Any], source_sha256: str, *, basis: str
) -> dict[str, Any]:
    return {
        "source_video_id": source["source_video_id"],
        "session_id": source["session_id"],
        "source_group_id": source["source_group_id"],
        "camera_setup_id": source["camera_setup_id"],
        "device_id": source["device_id"],
        "setup_id": source["setup_id"],
        "stream_epoch": source["stream_epoch"],
        "source_sha256": source_sha256,
        "source_sha256_basis": basis,
    }


def _validate_video_provenance(
    provenance: _VideoPreparationProvenance,
    sidecar: Mapping[str, Any],
    adapter: Any,
    source: Mapping[str, Any],
) -> None:
    if type(provenance) is not _VideoPreparationProvenance:
        raise CameraDatasetError("video provenance must use the exact private schema")
    if provenance.controller != "receipt_first_shared_yolov8_bytetrack_c1_preparation":
        raise CameraDatasetError("video provenance controller mismatch")
    if provenance.controller_schema_version != "wandering-camera-video-controller-v1":
        raise CameraDatasetError("video provenance controller schema mismatch")
    if provenance.source_video_id != source["source_video_id"]:
        raise CameraDatasetError("video provenance source mismatch")
    if (
        not _SHA256.fullmatch(provenance.source_sha256)
        or provenance.source_sha256 != sidecar["source_sha256"]
    ):
        raise CameraDatasetError("video provenance source SHA-256 mismatch")
    expected_detector = {
        "backend": provenance.detector_backend,
        "model": provenance.detector_model,
        "version": provenance.detector_version,
    }
    expected_tracker = {
        "backend": provenance.tracker_backend,
        "config": provenance.tracker_config,
        "version": provenance.tracker_version,
    }
    if expected_detector != sidecar["detector"]:
        raise CameraDatasetError("video provenance detector mismatch")
    if expected_tracker != sidecar["tracker"]:
        raise CameraDatasetError("video provenance tracker mismatch")
    if expected_detector["backend"] != "ultralytics_yolo" or expected_detector["model"] != "yolov8n.pt":
        raise CameraDatasetError("video provenance detector is not the fixed controller")
    if expected_tracker["backend"] != "bytetrack" or expected_tracker["config"] != "bytetrack.yaml":
        raise CameraDatasetError("video provenance tracker is not the fixed controller")
    if provenance.detector_version != provenance.tracker_version:
        raise CameraDatasetError("video provenance component version mismatch")
    if (
        provenance.confidence_threshold != 0.25
        or provenance.iou_threshold != 0.5
        or provenance.max_frames is not None
        or provenance.person_class_id != 0
        or provenance.person_id_prefix != "anonymous_track"
        or provenance.scene_region != "camera_development"
    ):
        raise CameraDatasetError("video provenance tracking execution mismatch")
    if (
        isinstance(provenance.raw_observation_count, bool)
        or not isinstance(provenance.raw_observation_count, int)
        or provenance.raw_observation_count <= 0
        or provenance.raw_observation_count != len(adapter.observations)
    ):
        raise CameraDatasetError("video provenance observation count mismatch")
    if (
        provenance.video_width != sidecar["video_width"]
        or provenance.video_height != sidecar["video_height"]
        or not math.isclose(provenance.nominal_fps, sidecar["nominal_fps"], rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(provenance.duration_sec, sidecar["duration_sec"], rel_tol=0.0, abs_tol=1e-12)
        or provenance.frame_count <= 0
        or not math.isclose(
            provenance.frame_count / provenance.nominal_fps,
            provenance.duration_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise CameraDatasetError("video provenance metadata mismatch")


def _video_summary(provenance: _VideoPreparationProvenance) -> dict[str, Any]:
    return {
        "controller": provenance.controller,
        "controller_schema_version": provenance.controller_schema_version,
        "detector": {
            "backend": provenance.detector_backend,
            "model": provenance.detector_model,
            "version": provenance.detector_version,
        },
        "tracker": {
            "backend": provenance.tracker_backend,
            "config": provenance.tracker_config,
            "version": provenance.tracker_version,
        },
        "tracking_execution": {
            "confidence_threshold": provenance.confidence_threshold,
            "iou_threshold": provenance.iou_threshold,
            "max_frames": provenance.max_frames,
            "person_class_id": provenance.person_class_id,
            "person_id_prefix": provenance.person_id_prefix,
            "scene_region": provenance.scene_region,
        },
        "raw_observation_count": provenance.raw_observation_count,
        "video_metadata": {
            "width": provenance.video_width,
            "height": provenance.video_height,
            "fps": provenance.nominal_fps,
            "frame_count": provenance.frame_count,
            "duration_sec": provenance.duration_sec,
        },
        "authorization_status": "authorized_camera_engineering_smoke",
        "authorized_camera_data_consumed": True,
    }


def _validate_authorized_preparation_summary(
    summary: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    video_provenance: _VideoPreparationProvenance | None,
) -> None:
    video_direct = video_provenance is not None
    expected_basis = (
        "controller_observed_video_pre_and_post_tracking"
        if video_direct
        else "validated_sidecar_declaration"
    )
    expected_binding = {
        "source_video_id": source["source_video_id"],
        "session_id": source["session_id"],
        "source_group_id": source["source_group_id"],
        "camera_setup_id": source["camera_setup_id"],
        "device_id": source["device_id"],
        "setup_id": source["setup_id"],
        "stream_epoch": source["stream_epoch"],
        "source_sha256": sidecar["source_sha256"],
        "source_sha256_basis": expected_basis,
    }
    if summary.get("source_binding") != expected_binding:
        raise CameraDatasetError("preparation source binding is inconsistent")
    expected_kind = (
        "authorized_video_shared_tracking_and_sidecar"
        if video_direct
        else "authorized_tracking_and_sidecar"
    )
    if summary.get("input_kind") != expected_kind:
        raise CameraDatasetError("preparation input kind is inconsistent")
    for field in ("media_opened", "detector_run", "tracker_run"):
        if summary.get(field) is not video_direct:
            raise CameraDatasetError("preparation work flags are inconsistent")
    if summary.get("authorized_camera_data_consumed") is not True:
        raise CameraDatasetError("authorized preparation consumption flag is inconsistent")
    if video_direct:
        assert video_provenance is not None
        if video_provenance.source_sha256 != sidecar["source_sha256"]:
            raise CameraDatasetError("video source SHA-256 basis is inconsistent")


def _validate_portable_media_metadata(sidecar: Mapping[str, Any]) -> None:
    _validate_portable_media_ref(sidecar.get("media_ref"))
    capture_started_at = sidecar.get("capture_started_at")
    timezone = sidecar.get("timezone")
    if (capture_started_at is None) != (timezone is None):
        raise CameraDatasetError("capture_started_at and timezone must be both null or both provided")
    if capture_started_at is None:
        return
    _parse_timestamp(capture_started_at, "capture_started_at")
    _validate_timezone(timezone)


def _validate_portable_media_ref(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise CameraDatasetError("media_ref must be a safe relative POSIX reference")
    lowered = value.lower()
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if (
        "\\" in value
        or "://" in value
        or "@" in value
        or "?" in value
        or "#" in value
        or any(marker in lowered for marker in _SENSITIVE_MARKERS)
        or PurePosixPath(value).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise CameraDatasetError("media_ref must be a portable deidentified relative POSIX reference")
    return value


def _validate_timezone(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise CameraDatasetError("timezone must be UTC or an existing safe IANA timezone")
    lowered = value.lower()
    parts = value.split("/")
    if (
        "\\" in value
        or value.startswith("/")
        or "://" in value
        or "@" in value
        or "?" in value
        or "#" in value
        or PureWindowsPath(value).drive
        or any(part in ("", ".", "..") for part in parts)
        or any(marker in lowered for marker in _SENSITIVE_MARKERS)
    ):
        raise CameraDatasetError("timezone must be UTC or an existing safe IANA timezone")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CameraDatasetError("timezone must be UTC or an existing safe IANA timezone") from exc
    return value


def _active_checkout_root() -> Path:
    try:
        source_file = Path(__file__).resolve(strict=True)
    except OSError as exc:
        raise CameraDatasetError("cannot resolve active wandering source identity") from exc
    candidates = [
        parent
        for parent in source_file.parents
        if (parent / "pyproject.toml").is_file()
        and (parent / "src/elderly_monitoring").is_dir()
    ]
    if len(candidates) != 1:
        raise CameraDatasetError("active checkout root must be nearest and unique")
    return candidates[0]


def _require_active_checkout_identity(project_root: str | Path) -> Path:
    active_root = _active_checkout_root()
    try:
        asserted = Path(project_root)
        if not asserted.is_dir() or not os.path.samefile(asserted, active_root):
            raise CameraDatasetError("project_root is not the active checkout identity")
    except (OSError, TypeError) as exc:
        raise CameraDatasetError("project_root is not the active checkout identity") from exc
    return active_root


def _require_fixed_config_identity(
    path: str | Path, active_root: Path, relative_path: str, role: str
) -> Path:
    expected = active_root / relative_path
    try:
        asserted = Path(path)
        if not asserted.is_file() or not os.path.samefile(asserted, expected):
            raise CameraDatasetError(f"{role} is not the fixed active checkout config")
    except (OSError, TypeError) as exc:
        raise CameraDatasetError(f"{role} is not the fixed active checkout config") from exc
    return expected


def _require_external_file(path: str | Path, project_root: Path, role: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise CameraDatasetError(f"cannot resolve {role}") from exc
    _reject_project_path(resolved, project_root, role)
    if not resolved.is_file():
        raise CameraDatasetError(f"{role} must be a regular file")
    return resolved


def _require_external_destination(
    path: str | Path, project_root: Path, role: str
) -> Path:
    try:
        candidate = Path(path)
        probe = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if os.path.lexists(probe) and not probe.exists():
            raise CameraDatasetError(f"{role} path is a dangling filesystem entry")
        existing = probe
        while not existing.exists():
            parent = existing.parent
            if parent == existing:
                raise CameraDatasetError(f"cannot find existing ancestor for {role}")
            existing = parent
        _reject_project_path(existing.resolve(strict=True), project_root, role)
        resolved = Path(path).resolve(strict=False)
    except OSError as exc:
        raise CameraDatasetError(f"cannot resolve {role}") from exc
    _reject_project_path(resolved, project_root, role)
    return resolved


def _reject_project_path(path: Path, project_root: Path, role: str) -> None:
    try:
        root = project_root.resolve(strict=True)
        resolved = path.resolve(strict=False)
        for candidate in (resolved, *resolved.parents):
            if candidate.exists() and os.path.samefile(candidate, root):
                raise CameraDatasetError(
                    f"production {role} must resolve outside project root (active checkout)"
                )
    except CameraDatasetError:
        raise
    except OSError as exc:
        raise CameraDatasetError(f"cannot verify production {role} boundary") from exc


def prepare_authorized_camera_session(
    *,
    project_root: str | Path,
    receipt_path: str | Path,
    collection_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
    collection_config_path: str | Path,
    camera_config_path: str | Path,
) -> dict[str, Any]:
    """Receipt-gate an existing authorized C1 pair with fixed false provenance.

    The external C0 receipt is validated before collection, sidecar, or tracking
    inputs.  Collection scope is then checked before the sidecar and tracking
    pair is opened.  This public entry cannot claim media/detector/tracker work.
    """

    return _prepare_authorized_camera_session(
        project_root=project_root,
        receipt_path=receipt_path,
        collection_path=collection_path,
        tracking_jsonl_path=tracking_jsonl_path,
        media_sidecar_path=media_sidecar_path,
        output_dir=output_dir,
        collection_config_path=collection_config_path,
        camera_config_path=camera_config_path,
        video_provenance=None,
    )


def _prepare_authorized_camera_video_session(
    *,
    project_root: str | Path,
    receipt_path: str | Path,
    collection_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
    collection_config_path: str | Path,
    camera_config_path: str | Path,
    video_provenance: _VideoPreparationProvenance,
) -> dict[str, Any]:
    """Private video-controller entry with an exact, non-mapping provenance carrier."""

    return _prepare_authorized_camera_session(
        project_root=project_root,
        receipt_path=receipt_path,
        collection_path=collection_path,
        tracking_jsonl_path=tracking_jsonl_path,
        media_sidecar_path=media_sidecar_path,
        output_dir=output_dir,
        collection_config_path=collection_config_path,
        camera_config_path=camera_config_path,
        video_provenance=video_provenance,
    )


def _prepare_authorized_camera_session(
    *,
    project_root: str | Path,
    receipt_path: str | Path,
    collection_path: str | Path,
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    output_dir: str | Path,
    collection_config_path: str | Path,
    camera_config_path: str | Path,
    video_provenance: _VideoPreparationProvenance | None,
) -> dict[str, Any]:
    root = _require_active_checkout_identity(project_root)
    fixed_collection_config = _require_fixed_config_identity(
        collection_config_path,
        root,
        "configs/data/wandering_camera_collection_v1.yaml",
        "collection config",
    )
    fixed_camera_config = _require_fixed_config_identity(
        camera_config_path,
        root,
        "configs/modules/wandering_camera_v1.yaml",
        "camera config",
    )
    config = load_camera_collection_config(fixed_collection_config)
    try:
        camera_config = load_camera_config(fixed_camera_config)
    except (CameraInferenceError, OSError, ValueError) as exc:
        raise CameraDatasetError("fixed active camera config preflight failed") from exc

    receipt_file = _require_external_file(receipt_path, root, "authorization receipt")
    receipt = _read_json_object(receipt_file, "authorization receipt")
    authorization_summary = validate_authorization_receipt(
        receipt,
        config,
        operation="prepare_session",
    )
    collection_file = _require_external_file(collection_path, root, "collection")
    collection = validate_collection_manifest(
        _read_json_object(collection_file, "collection"), config
    )
    authorization_summary = validate_authorization_receipt(
        receipt,
        config,
        operation="prepare_session",
        **_collection_scope(collection),
    )
    _validate_collection_receipt_reference(collection, authorization_summary)
    source = _selected_unique_collection_source(collection)

    output = _require_external_destination(output_dir, root, "output")
    if os.path.lexists(output):
        raise CameraDatasetError(f"output already exists: {output}")

    sidecar_file = _require_external_file(media_sidecar_path, root, "media sidecar")
    try:
        sidecar = validate_media_sidecar(
            _read_json_object(sidecar_file, "media sidecar"), camera_config
        )
        validate_camera_components(sidecar)
        _validate_portable_media_metadata(sidecar)
        source_sidecar_payload = sidecar_file.read_bytes()
    except (CameraInferenceError, CameraAdapterError, OSError, ValueError) as exc:
        raise CameraDatasetError("authorized media sidecar validation failed") from exc
    _bind_sidecar_to_collection(sidecar, collection)
    if sidecar["authorization_status"] not in {
        "authorized_camera_engineering_smoke",
        "authorized_camera_labeled_evaluation",
    }:
        raise CameraDatasetError("prepare_session production entry rejects synthetic media")
    if sidecar["source_video_id"] != source["source_video_id"]:
        raise CameraDatasetError("media source does not match the unique collection source")
    tracking_file = _require_external_file(tracking_jsonl_path, root, "tracking JSONL")
    try:
        adapter = load_camera_inputs(tracking_file, sidecar_file, camera_config)
    except (CameraAdapterError, OSError, ValueError) as exc:
        raise CameraDatasetError("authorized tracking/sidecar validation failed") from exc

    if video_provenance is not None:
        _validate_video_provenance(video_provenance, sidecar, adapter, source)
    tracking_payload, sidecar_payload, pair_facts = _canonical_pair_payloads(
        adapter, source_sidecar_payload
    )
    summary: dict[str, Any] = {
        "schema_version": "wandering-camera-input-preparation-v1",
        "input_kind": (
            "authorized_video_shared_tracking_and_sidecar"
            if video_provenance is not None
            else "authorized_tracking_and_sidecar"
        ),
        "media_opened": video_provenance is not None,
        "detector_run": video_provenance is not None,
        "tracker_run": video_provenance is not None,
        "camera_qc_run": False,
        "model_inference_run": False,
        "m0cam_d_started": False,
        "authorization_status": sidecar["authorization_status"],
        "authorized_camera_data_consumed": True,
        **pair_facts,
        "authorization_binding": _authorization_binding(authorization_summary),
        "collection_binding": {
            "collection_id": collection["collection_id"],
            "collection_sha256": hashlib.sha256(
                canonical_json_bytes(collection)
            ).hexdigest(),
            "dataset_role": collection["dataset_role"],
            "authorization_receipt_id": collection["authorization_receipt_id"],
        },
        "source_binding": _source_binding(
            source,
            sidecar["source_sha256"],
            basis=(
                "controller_observed_video_pre_and_post_tracking"
                if video_provenance is not None
                else "validated_sidecar_declaration"
            ),
        ),
    }
    if video_provenance is not None:
        summary.update(_video_summary(video_provenance))
    _commit_prepared_pair(
        output=output,
        tracking_payload=tracking_payload,
        sidecar_payload=sidecar_payload,
        summary=summary,
        camera_config=camera_config,
        summary_validator=lambda candidate: _validate_authorized_preparation_summary(
            candidate,
            source=source,
            sidecar=sidecar,
            video_provenance=video_provenance,
        ),
    )
    return summary


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


def _commit_new_directory(
    output: Path,
    files: Mapping[str, bytes],
    *,
    validate_staged_pair: Callable[[Path], None] | None = None,
) -> None:
    if os.path.lexists(output):
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
        if validate_staged_pair is not None:
            validate_staged_pair(temporary)
        if os.path.lexists(output):
            raise CameraDatasetError(f"output already exists: {output}")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
