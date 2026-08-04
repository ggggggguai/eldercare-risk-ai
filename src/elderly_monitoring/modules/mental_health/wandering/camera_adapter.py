"""Strict bbox-only camera input adapter for wandering step 7.

The adapter deliberately consumes only the five trusted fields from the
existing fall-tracking JSONL.  Upstream ``person_id``, bbox center, and pixel
speed remain outside the wandering identity and trajectory contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MEDIA_SCHEMA_VERSION = "wandering-media-v1"
BBOX_TRACKLET_SCHEMA_VERSION = "wandering-bbox-tracklet-v1"
AUTHORIZATION_STATUSES = (
    "synthetic_fixture",
    "authorized_camera_engineering_smoke",
    "authorized_camera_labeled_evaluation",
)

_VALIDATION_SCOPE_BY_AUTHORIZATION_STATUS = {
    "synthetic_fixture": "synthetic_camera_contract",
    "authorized_camera_engineering_smoke": "authorized_camera_engineering_smoke",
    "authorized_camera_labeled_evaluation": "authorized_camera_labeled_evaluation",
}

_MEDIA_FIELDS = frozenset(
    {
        "schema_version",
        "source_video_id",
        "source_group_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "media_ref",
        "source_sha256",
        "tracking_jsonl_sha256",
        "video_width",
        "video_height",
        "nominal_fps",
        "duration_sec",
        "capture_started_at",
        "timezone",
        "coordinate_system",
        "detector",
        "tracker",
        "fixed_camera_assumed",
        "camera_motion_state",
        "authorization_status",
        "deidentification_status",
    }
)
_DETECTOR_FIELDS = frozenset({"backend", "model", "version"})
_TRACKER_FIELDS = frozenset({"backend", "config", "version"})
_CAMERA_MOTION_STATES = frozenset({"stable", "moved", "not_checked"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CREDENTIAL_MARKERS = ("token=", "access_token", "authorization:", "bearer ", "signature=")


class CameraAdapterError(ValueError):
    """The camera sidecar or tracking input violates the v1 contract."""


@dataclass(frozen=True)
class CameraObservation:
    scope_key: tuple[str, str, str, str, str, int]
    source_group_id: str
    source_video_id: str
    device_id: str
    setup_id: str
    stream_epoch: str
    track_id: int
    frame_id: int
    timestamp_sec: float
    bbox_xyxy_pixel: tuple[float, float, float, float]
    bbox_xyxy_norm: tuple[float, float, float, float]
    bbox_bottom_point: tuple[float, float]
    bbox_height: float
    track_confidence: float


@dataclass(frozen=True)
class WeightedBucket:
    scope_key: tuple[str, str, str, str, str, int]
    bucket_index: int
    point_time_sec: float
    bbox_bottom_point: tuple[float, float]
    bbox_height: float
    mean_detection_confidence: float
    raw_detection_count: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class CameraAdapterInput:
    media_sidecar: Mapping[str, Any]
    observations: tuple[CameraObservation, ...]
    normalized_rows: tuple[Mapping[str, Any], ...]
    source_tracking_sha256: str
    normalized_tracking_sha256: str


def validate_media_sidecar(
    value: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the exact ``wandering-media-v1`` mapping."""

    if not isinstance(value, Mapping) or frozenset(value) != _MEDIA_FIELDS:
        raise CameraAdapterError("wandering-media-v1 fields must match the exact v1 set")
    if value.get("schema_version") != MEDIA_SCHEMA_VERSION:
        raise CameraAdapterError(f"schema_version must be {MEDIA_SCHEMA_VERSION!r}")
    normalized = dict(value)
    for field in (
        "source_video_id",
        "source_group_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "deidentification_status",
    ):
        normalized[field] = _safe_token(value.get(field), field)
    authorization_status = _safe_token(value.get("authorization_status"), "authorization_status")
    validation_scope_for_authorization_status(authorization_status)
    normalized["authorization_status"] = authorization_status
    normalized["media_ref"] = _safe_media_ref(value.get("media_ref"))
    for field in ("source_sha256", "tracking_jsonl_sha256"):
        normalized[field] = _sha256(value.get(field), field)
    normalized["video_width"] = _positive_int(value.get("video_width"), "video_width")
    normalized["video_height"] = _positive_int(value.get("video_height"), "video_height")
    normalized["nominal_fps"] = _finite_positive(value.get("nominal_fps"), "nominal_fps")
    normalized["duration_sec"] = _finite_positive(value.get("duration_sec"), "duration_sec")
    minimum_fps = float(config.get("sampling", {}).get("minimum_video_fps", math.nan))
    if not math.isfinite(minimum_fps) or normalized["nominal_fps"] < minimum_fps:
        raise CameraAdapterError("nominal_fps is below the frozen minimum_video_fps")
    for field in ("capture_started_at", "timezone"):
        item = value.get(field)
        if item is not None:
            normalized[field] = _safe_text(item, field)
    if value.get("coordinate_system") != "pixel_xyxy_top_left":
        raise CameraAdapterError("coordinate_system must be pixel_xyxy_top_left")
    if not isinstance(value.get("fixed_camera_assumed"), bool) or value["fixed_camera_assumed"] is not True:
        raise CameraAdapterError("fixed_camera_assumed must be true for camera v1")
    if value.get("camera_motion_state") not in _CAMERA_MOTION_STATES:
        raise CameraAdapterError("camera_motion_state must be stable/moved/not_checked")
    normalized["detector"] = _component_mapping(value.get("detector"), _DETECTOR_FIELDS, "detector")
    normalized["tracker"] = _component_mapping(value.get("tracker"), _TRACKER_FIELDS, "tracker")
    return normalized


def validation_scope_for_authorization_status(value: str) -> str:
    """Map only frozen authorization states to their exact evidence scope."""

    try:
        return _VALIDATION_SCOPE_BY_AUTHORIZATION_STATUS[value]
    except (KeyError, TypeError) as exc:
        allowed = "/".join(AUTHORIZATION_STATUSES)
        raise CameraAdapterError(f"authorization_status must be one of {allowed}") from exc


def load_camera_inputs(
    tracking_jsonl_path: str | Path,
    media_sidecar_path: str | Path,
    config: Mapping[str, Any],
) -> CameraAdapterInput:
    """Hash, parse, and normalize one trusted tracking file plus sidecar."""

    tracking_path = Path(tracking_jsonl_path)
    sidecar_path = Path(media_sidecar_path)
    if not tracking_path.is_file() or not sidecar_path.is_file():
        raise CameraAdapterError("tracking JSONL and media sidecar must both exist")
    source_payload = tracking_path.read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    try:
        sidecar_raw = json.loads(
            sidecar_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CameraAdapterError) as exc:
        raise CameraAdapterError("cannot parse media sidecar as strict finite JSON") from exc
    media = validate_media_sidecar(sidecar_raw, config)
    if media["tracking_jsonl_sha256"] != source_sha256:
        raise CameraAdapterError("tracking JSONL SHA-256 does not match media sidecar")

    normalized_rows: list[dict[str, Any]] = []
    observations: list[CameraObservation] = []
    previous: dict[tuple[str, str, str, str, str, int], tuple[int, float]] = {}
    seen_frames: set[tuple[tuple[str, str, str, str, str, int], int]] = set()
    for line_number, raw_line in enumerate(source_payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeError, json.JSONDecodeError, CameraAdapterError) as exc:
            raise CameraAdapterError(f"invalid tracking JSON at line {line_number}") from exc
        if not isinstance(row, Mapping):
            raise CameraAdapterError(f"tracking line {line_number} must be a JSON object")
        normalized, observation = _normalize_tracking_row(row, media, line_number=line_number)
        prior = previous.get(observation.scope_key)
        if prior is not None:
            if observation.frame_id <= prior[0]:
                raise CameraAdapterError("frame_id must be strictly increasing within one track scope")
            if observation.timestamp_sec <= prior[1]:
                raise CameraAdapterError("timestamp_sec must be strictly increasing within one track scope")
        duplicate_key = (observation.scope_key, observation.frame_id)
        if duplicate_key in seen_frames:
            raise CameraAdapterError("duplicate (scope, track_id, frame_id) is forbidden")
        previous[observation.scope_key] = (observation.frame_id, observation.timestamp_sec)
        seen_frames.add(duplicate_key)
        normalized_rows.append(normalized)
        observations.append(observation)
    if not observations:
        raise CameraAdapterError("tracking JSONL contains no observations")
    if max(item.timestamp_sec for item in observations) > float(media["duration_sec"]) + 1e-9:
        raise CameraAdapterError("tracking timestamp exceeds sidecar duration_sec")

    paired = sorted(
        zip(observations, normalized_rows, strict=True),
        key=lambda item: (item[0].scope_key, item[0].timestamp_sec, item[0].frame_id),
    )
    sorted_observations = tuple(item[0] for item in paired)
    sorted_rows = tuple(item[1] for item in paired)
    normalized_payload = canonical_jsonl_bytes(sorted_rows)
    return CameraAdapterInput(
        media_sidecar=media,
        observations=sorted_observations,
        normalized_rows=sorted_rows,
        source_tracking_sha256=source_sha256,
        normalized_tracking_sha256=hashlib.sha256(normalized_payload).hexdigest(),
    )


def group_observations(
    observations: Iterable[CameraObservation],
) -> dict[tuple[str, str, str, str, str, int], tuple[CameraObservation, ...]]:
    """Group by the complete camera isolation key without cross-key merging."""

    grouped: dict[tuple[str, str, str, str, str, int], list[CameraObservation]] = {}
    for observation in observations:
        if not isinstance(observation, CameraObservation):
            raise CameraAdapterError("group_observations accepts CameraObservation values only")
        scope_key = (
            observation.source_group_id,
            observation.source_video_id,
            observation.device_id,
            observation.setup_id,
            observation.stream_epoch,
            observation.track_id,
        )
        grouped.setdefault(scope_key, []).append(observation)
    return {
        key: tuple(sorted(values, key=lambda item: (item.timestamp_sec, item.frame_id)))
        for key, values in sorted(grouped.items())
    }


def weighted_bucket_observations(
    observations: Sequence[CameraObservation],
    *,
    minimum_track_confidence: float,
    bucket_seconds: float,
) -> tuple[WeightedBucket, ...]:
    """Aggregate valid detections on the video-global fixed-time bucket grid."""

    threshold = _finite_nonnegative(minimum_track_confidence, "minimum_track_confidence")
    width = _finite_positive(bucket_seconds, "bucket_seconds")
    accepted = [item for item in observations if item.track_confidence >= threshold]
    by_bucket: dict[tuple[tuple[str, str, str, str, str, int], int], list[CameraObservation]] = {}
    for observation in accepted:
        bucket_index = int(math.floor(observation.timestamp_sec / width))
        by_bucket.setdefault((observation.scope_key, bucket_index), []).append(observation)
    output: list[WeightedBucket] = []
    for (scope_key, bucket_index), values in sorted(by_bucket.items()):
        ordered = sorted(values, key=lambda item: (item.timestamp_sec, item.frame_id))
        confidence = np.asarray([item.track_confidence for item in ordered], dtype=np.float64)
        total = float(confidence.sum())
        if total <= 0.0:
            continue
        weights = confidence / total
        points = np.asarray([item.bbox_bottom_point for item in ordered], dtype=np.float64)
        heights = np.asarray([item.bbox_height for item in ordered], dtype=np.float64)
        point = weights @ points
        height = float(weights @ heights)
        output.append(
            WeightedBucket(
                scope_key=scope_key,
                bucket_index=bucket_index,
                point_time_sec=(bucket_index + 0.5) * width,
                bbox_bottom_point=(float(point[0]), float(point[1])),
                bbox_height=height,
                mean_detection_confidence=float(weights @ confidence),
                raw_detection_count=len(ordered),
                frame_indices=tuple(item.frame_id for item in ordered),
            )
        )
    return tuple(output)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CameraAdapterError("camera artifact is not finite JSON") from exc
    return (text + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _normalize_tracking_row(
    row: Mapping[str, Any],
    media: Mapping[str, Any],
    *,
    line_number: int,
) -> tuple[dict[str, Any], CameraObservation]:
    required = ("frame_id", "track_id", "bbox", "track_confidence", "timestamp_sec")
    missing = [field for field in required if field not in row]
    if missing:
        raise CameraAdapterError(f"tracking line {line_number} missing fields: {missing}")
    frame_id = _nonnegative_int(row["frame_id"], "frame_id")
    track_id = _nonnegative_int(row["track_id"], "track_id")
    confidence = _finite_nonnegative(row["track_confidence"], "track_confidence")
    if confidence > 1.0:
        raise CameraAdapterError("track_confidence must be within [0,1]")
    timestamp = _finite_nonnegative(row["timestamp_sec"], "timestamp_sec")
    bbox_value = row["bbox"]
    if not isinstance(bbox_value, Sequence) or isinstance(bbox_value, (str, bytes)) or len(bbox_value) != 4:
        raise CameraAdapterError("bbox must be a four-number sequence")
    bbox = tuple(_finite_nonnegative(item, "bbox") for item in bbox_value)
    x1, y1, x2, y2 = bbox
    width = float(media["video_width"])
    height = float(media["video_height"])
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise CameraAdapterError("bbox must satisfy image-bounded pixel xyxy ordering")
    bbox_norm = (x1 / width, y1 / height, x2 / width, y2 / height)
    bottom = (((x1 + x2) / 2.0) / width, y2 / height)
    bbox_height = (y2 - y1) / height
    scope = (
        str(media["source_group_id"]),
        str(media["source_video_id"]),
        str(media["device_id"]),
        str(media["setup_id"]),
        str(media["stream_epoch"]),
        track_id,
    )
    normalized = {
        "bbox": [float(item) for item in bbox],
        "frame_id": frame_id,
        "timestamp_sec": timestamp,
        "track_confidence": confidence,
        "track_id": track_id,
    }
    return normalized, CameraObservation(
        scope_key=scope,
        source_group_id=scope[0],
        source_video_id=scope[1],
        device_id=scope[2],
        setup_id=scope[3],
        stream_epoch=scope[4],
        track_id=track_id,
        frame_id=frame_id,
        timestamp_sec=timestamp,
        bbox_xyxy_pixel=bbox,
        bbox_xyxy_norm=bbox_norm,
        bbox_bottom_point=bottom,
        bbox_height=bbox_height,
        track_confidence=confidence,
    )


def _component_mapping(value: Any, fields: frozenset[str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise CameraAdapterError(f"{name} fields must match the exact v1 set")
    output = {field: _safe_text(value[field], f"{name}.{field}") for field in sorted(fields)}
    path_field = "model" if name == "detector" else "config"
    path_value = output[path_field]
    if (
        "\\" in path_value
        or "/" in path_value
        or PureWindowsPath(path_value).drive
        or PureWindowsPath(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or "://" in path_value
    ):
        raise CameraAdapterError(f"{name}.{path_field} must be a deidentified file name, not a path")
    return output


def _safe_media_ref(value: Any) -> str:
    text = _safe_text(value, "media_ref")
    lowered = text.lower()
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS) or "://" in text:
        raise CameraAdapterError("media_ref must not contain URLs or credentials")
    if "\\" in text or PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute():
        raise CameraAdapterError("media_ref must be a relative POSIX path or deidentified reference")
    parts = PurePosixPath(text).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise CameraAdapterError("media_ref must be a portable relative POSIX reference")
    return text


def _safe_token(value: Any, field: str) -> str:
    text = _safe_text(value, field)
    if not _SAFE_TOKEN.fullmatch(text):
        raise CameraAdapterError(f"{field} contains unsupported characters")
    return text


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(char in value for char in "\r\n\0"):
        raise CameraAdapterError(f"{field} must be a safe non-empty string")
    lowered = value.lower()
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise CameraAdapterError(f"{field} must not contain credentials")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CameraAdapterError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CameraAdapterError(f"{field} must be a non-negative integer")
    return value


def _finite_positive(value: Any, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number <= 0.0:
        raise CameraAdapterError(f"{field} must be positive")
    return number


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraAdapterError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise CameraAdapterError(f"{field} must be finite and non-negative")
    return number


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CameraAdapterError(f"{field} must be lowercase SHA-256")
    return value


def _reject_json_constant(value: str) -> None:
    raise CameraAdapterError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "AUTHORIZATION_STATUSES",
    "BBOX_TRACKLET_SCHEMA_VERSION",
    "MEDIA_SCHEMA_VERSION",
    "CameraAdapterError",
    "CameraAdapterInput",
    "CameraObservation",
    "WeightedBucket",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "group_observations",
    "load_camera_inputs",
    "validate_media_sidecar",
    "validation_scope_for_authorization_status",
    "weighted_bucket_observations",
]
