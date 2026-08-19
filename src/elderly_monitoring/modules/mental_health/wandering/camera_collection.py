"""Receipt-first C1 tracking preparation for anonymous development video.

This controller only prepares a canonical tracking JSONL plus
``wandering-media-v1`` sidecar.  It does not run camera QC, episode building,
RF/TCN inference, primary inference, labeling, or M0-CAM-D evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_component import (
    validate_camera_component,
)
from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    CameraDatasetError,
    _VideoPreparationProvenance,
    _prepare_authorized_camera_video_session,
    _require_active_checkout_identity,
    _require_external_destination,
    _require_external_file,
    _require_fixed_config_identity,
    _validate_portable_media_ref,
    _validate_timezone,
    load_camera_collection_config,
    validate_authorization_receipt,
    validate_collection_manifest,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
)


DETECTOR_MODEL = "yolov8n.pt"
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.5
TRACKER_CONFIG = "bytetrack.yaml"
SCENE_REGION = "camera_development"
PERSON_ID_PREFIX = "anonymous_track"
PERSON_CLASS_ID = 0
MAX_FRAMES = None
AUTHORIZATION_STATUS = "authorized_camera_engineering_smoke"

_CAMERA_MOTION_STATES = frozenset({"stable", "moved", "not_checked"})
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
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


class CameraCollectionPreparationError(CameraDatasetError):
    """Receipt-first C1 preparation failed closed."""


@dataclass(frozen=True)
class _VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class _TrackingRuntime:
    run: Callable[..., int]
    detector_backend: str
    detector_version: str
    tracker_backend: str
    tracker_version: str


def build_authorized_camera_tracking_pair(
    *,
    project_root: str | Path,
    receipt_path: str | Path,
    collection_path: str | Path,
    source_video_id: str,
    input_video_path: str | Path,
    media_ref: str,
    camera_motion_state: str,
    deidentification_status: str,
    capture_started_at: str | None,
    timezone: str | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build one C1 pair after receipt, collection, source, and output gates.

    Detector and tracker behavior is deliberately fixed by this controller.
    No authorization state, evidence scope, model, threshold, frame limit,
    loader, or test implementation can be supplied by a caller.
    """

    try:
        root = _require_active_checkout_identity(project_root)
        collection_config_path = _require_fixed_config_identity(
            root / "configs/data/wandering_camera_collection_v1.yaml",
            root,
            "configs/data/wandering_camera_collection_v1.yaml",
            "collection config",
        )
        camera_config_path = _require_fixed_config_identity(
            root / "configs/modules/wandering_camera_v1.yaml",
            root,
            "configs/modules/wandering_camera_v1.yaml",
            "camera config",
        )
        collection_config = load_camera_collection_config(collection_config_path)
        load_camera_config(camera_config_path)
    except (CameraDatasetError, CameraInferenceError, OSError, ValueError) as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc

    try:
        receipt_file = _require_external_file(receipt_path, root, "authorization receipt")
    except CameraDatasetError as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc

    # C0 is the first external input.  Nothing about the collection, video,
    # runtime, output, or temporary workspace is touched before this succeeds.
    try:
        receipt = _read_json_object(receipt_file, "authorization receipt")
        receipt_summary = validate_authorization_receipt(
            receipt,
            collection_config,
            operation="prepare_session",
        )
    except CameraDatasetError as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc

    try:
        collection_file = _require_external_file(collection_path, root, "collection")
        collection = validate_collection_manifest(
            _read_json_object(collection_file, "collection"),
            collection_config,
        )
        receipt_summary = validate_authorization_receipt(
            receipt,
            collection_config,
            operation="prepare_session",
            **_collection_scope(collection),
        )
        _validate_collection_receipt_reference(collection, receipt_summary)
        source = _select_source(collection, source_video_id)
    except CameraDatasetError as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc

    try:
        output = _require_external_destination(output_dir, root, "output")
    except CameraDatasetError as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc
    if os.path.lexists(output):
        raise CameraCollectionPreparationError(f"output already exists: {output}")

    _validate_operator_fields(
        media_ref=media_ref,
        camera_motion_state=camera_motion_state,
        deidentification_status=deidentification_status,
        capture_started_at=capture_started_at,
        timezone=timezone,
    )

    # PORTABLE is deliberately imported and inspected only after the owner
    # receipt/collection/source/output/operator gates.  Passing a basename to
    # Ultralytics would re-enable its implicit weight download behavior.
    detector_identity = _resolve_portable_detector(root)

    # Protected media and optional vision imports remain unreachable until all
    # receipt/collection/source/output/operator and detector-asset checks pass.
    try:
        video_candidate = _require_external_file(input_video_path, root, "input video")
        video = _resolve_video(video_candidate)
        source_sha256 = _sha256_file(video)
        metadata = _probe_video(video)
        runtime = _load_tracking_runtime()
        detector, tracker = _validate_tracking_runtime_components(runtime)
    except (CameraDatasetError, CameraAdapterError, OSError, ValueError, ImportError) as exc:
        raise CameraCollectionPreparationError("cannot prepare authorized input video") from exc

    with tempfile.TemporaryDirectory(prefix="wandering-camera-c1-raw-") as raw_dir:
        raw_root = Path(raw_dir)
        raw_tracking = raw_root / "tracking.jsonl"
        detector_model_path = _verify_portable_detector(root, detector_identity)
        with _portable_network_guard():
            observation_count = runtime.run(
                video_path=video,
                output_path=raw_tracking,
                model_name=str(detector_model_path),
                scene_region=SCENE_REGION,
                person_id_prefix=PERSON_ID_PREFIX,
                confidence_threshold=DETECTOR_CONFIDENCE,
                iou_threshold=DETECTOR_IOU,
                tracker_config=TRACKER_CONFIG,
                max_frames=MAX_FRAMES,
            )
        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
            or observation_count <= 0
            or not raw_tracking.is_file()
        ):
            raise CameraCollectionPreparationError("shared tracker produced no C1 observations")
        try:
            post_tracking_source_sha256 = _sha256_file(video)
        except OSError as exc:
            raise CameraCollectionPreparationError(
                "cannot verify authorized input video after tracking"
            ) from exc
        if post_tracking_source_sha256 != source_sha256:
            raise CameraCollectionPreparationError(
                "authorized input video changed during tracking"
            )

        raw_sidecar = raw_root / "media_sidecar.json"
        sidecar = {
            "schema_version": "wandering-media-v1",
            "source_video_id": source["source_video_id"],
            "source_group_id": source["source_group_id"],
            "device_id": source["device_id"],
            "setup_id": source["setup_id"],
            "stream_epoch": source["stream_epoch"],
            "media_ref": media_ref,
            "source_sha256": source_sha256,
            "tracking_jsonl_sha256": _sha256_file(raw_tracking),
            "video_width": metadata.width,
            "video_height": metadata.height,
            "nominal_fps": metadata.fps,
            "duration_sec": metadata.duration_sec,
            "capture_started_at": capture_started_at,
            "timezone": timezone,
            "coordinate_system": "pixel_xyxy_top_left",
            "detector": {
                **detector,
            },
            "tracker": {
                **tracker,
            },
            "fixed_camera_assumed": True,
            "camera_motion_state": camera_motion_state,
            "authorization_status": AUTHORIZATION_STATUS,
            "deidentification_status": deidentification_status,
        }
        raw_sidecar.write_bytes(canonical_json_bytes(sidecar))

        try:
            result = _prepare_authorized_camera_video_session(
                project_root=root,
                receipt_path=receipt_file,
                collection_path=collection_file,
                tracking_jsonl_path=raw_tracking,
                media_sidecar_path=raw_sidecar,
                output_dir=output,
                collection_config_path=collection_config_path,
                camera_config_path=camera_config_path,
                video_provenance=_VideoPreparationProvenance(
                    controller="receipt_first_shared_yolov8_bytetrack_c1_preparation",
                    controller_schema_version="wandering-camera-video-controller-v1",
                    source_video_id=source["source_video_id"],
                    source_sha256=source_sha256,
                    detector_backend=detector["backend"],
                    detector_model=DETECTOR_MODEL,
                    detector_version=detector["version"],
                    tracker_backend=tracker["backend"],
                    tracker_config=TRACKER_CONFIG,
                    tracker_version=tracker["version"],
                    confidence_threshold=DETECTOR_CONFIDENCE,
                    iou_threshold=DETECTOR_IOU,
                    max_frames=MAX_FRAMES,
                    person_class_id=PERSON_CLASS_ID,
                    person_id_prefix=PERSON_ID_PREFIX,
                    scene_region=SCENE_REGION,
                    raw_observation_count=observation_count,
                    video_width=metadata.width,
                    video_height=metadata.height,
                    nominal_fps=metadata.fps,
                    duration_sec=metadata.duration_sec,
                    frame_count=metadata.frame_count,
                ),
            )
        except CameraDatasetError as exc:
            raise CameraCollectionPreparationError(str(exc)) from exc

    return result


def _resolve_portable_detector(root: Path) -> Any:
    try:
        from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
            PortableAssetBlockedError,
            PortableContractError,
            resolve_verified_detector_asset,
        )
    except ImportError as exc:
        raise CameraCollectionPreparationError(
            "portable detector asset is unavailable"
        ) from exc
    try:
        return resolve_verified_detector_asset(root)
    except (PortableAssetBlockedError, PortableContractError, OSError, ValueError) as exc:
        raise CameraCollectionPreparationError(
            "portable detector asset is unavailable"
        ) from exc


def _verify_portable_detector(root: Path, detector_identity: Any) -> Path:
    try:
        from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
            PortableAssetBlockedError,
            PortableContractError,
            verify_detector_asset_again,
        )
    except ImportError as exc:
        raise CameraCollectionPreparationError(
            "portable detector asset changed before tracking"
        ) from exc
    try:
        return verify_detector_asset_again(root, detector_identity)
    except (PortableAssetBlockedError, PortableContractError, OSError, ValueError) as exc:
        raise CameraCollectionPreparationError(
            "portable detector asset changed before tracking"
        ) from exc


@contextmanager
def _portable_network_guard():
    try:
        from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
            NetworkAccessDeniedError,
            PortableAssetBlockedError,
            deny_network_access,
        )
    except ImportError as exc:
        raise CameraCollectionPreparationError(
            "portable detector network guard is unavailable"
        ) from exc
    try:
        with deny_network_access():
            yield
    except (NetworkAccessDeniedError, PortableAssetBlockedError) as exc:
        raise CameraCollectionPreparationError(
            "portable detector network guard is unavailable"
        ) from exc


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraCollectionPreparationError(f"cannot parse {role}") from exc
    if not isinstance(value, dict):
        raise CameraCollectionPreparationError(f"{role} must be a JSON object")
    return value


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
    collection: Mapping[str, Any], receipt_summary: Mapping[str, Any]
) -> None:
    if collection["authorization_receipt_id"] != receipt_summary["receipt_id"]:
        raise CameraCollectionPreparationError(
            "collection authorization receipt reference mismatch"
        )
    if collection["dataset_role"] != "development":
        raise CameraCollectionPreparationError(
            "sealed camera collection is forbidden in development"
        )


def _select_source(
    collection: Mapping[str, Any], source_video_id: str
) -> Mapping[str, Any]:
    matches = [
        row for row in collection["sources"] if row["source_video_id"] == source_video_id
    ]
    if len(matches) != 1:
        raise CameraCollectionPreparationError(
            "source_video_id must select exactly one legal collection source"
        )
    return matches[0]


def _validate_operator_fields(
    *,
    media_ref: str,
    camera_motion_state: str,
    deidentification_status: str,
    capture_started_at: str | None,
    timezone: str | None,
) -> None:
    if camera_motion_state not in _CAMERA_MOTION_STATES:
        raise CameraCollectionPreparationError(
            "camera_motion_state must be supplied as stable/moved/not_checked"
        )
    if not isinstance(deidentification_status, str) or not _SAFE_TOKEN.fullmatch(
        deidentification_status
    ):
        raise CameraCollectionPreparationError(
            "deidentification_status must be an operator-supplied anonymous token"
        )
    if any(marker in deidentification_status.lower() for marker in _SENSITIVE_MARKERS):
        raise CameraCollectionPreparationError(
            "deidentification_status must not contain sensitive material"
        )
    try:
        _validate_portable_media_ref(media_ref)
    except CameraDatasetError as exc:
        raise CameraCollectionPreparationError(str(exc)) from exc
    if (capture_started_at is None) != (timezone is None):
        raise CameraCollectionPreparationError(
            "capture_started_at and timezone must be both null or both provided"
        )
    if capture_started_at is not None:
        try:
            parsed = datetime.fromisoformat(capture_started_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise CameraCollectionPreparationError(
                "capture_started_at must be an ISO-8601 timestamp"
            ) from exc
        if parsed.tzinfo is None:
            raise CameraCollectionPreparationError(
                "capture_started_at must include a timezone offset"
            )
        try:
            _validate_timezone(timezone)
        except CameraDatasetError as exc:
            raise CameraCollectionPreparationError(str(exc)) from exc


def _resolve_video(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("input video must be a non-empty regular file")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> _VideoMetadata:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("C1 video preparation requires the project vision dependencies") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("cannot open authorized input video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    metadata = _VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)
    if (
        metadata.width <= 0
        or metadata.height <= 0
        or metadata.frame_count <= 0
        or not math.isfinite(metadata.fps)
        or metadata.fps <= 0.0
    ):
        raise ValueError("input video metadata must provide positive width/height/fps/frame count")
    return metadata


def _load_tracking_runtime() -> _TrackingRuntime:
    try:
        import ultralytics
        from elderly_monitoring.modules.fall_risk.tracking import run_yolov8_bytetrack
    except ImportError as exc:
        raise ImportError("C1 tracking requires the project vision dependencies") from exc
    return _TrackingRuntime(
        run=run_yolov8_bytetrack,
        detector_backend="ultralytics_yolo",
        detector_version=str(ultralytics.__version__),
        tracker_backend="bytetrack",
        tracker_version=str(ultralytics.__version__),
    )


def _validate_tracking_runtime_components(
    runtime: _TrackingRuntime,
) -> tuple[dict[str, str], dict[str, str]]:
    detector = validate_camera_component(
        {
            "backend": runtime.detector_backend,
            "model": DETECTOR_MODEL,
            "version": runtime.detector_version,
        },
        name="detector",
    )
    tracker = validate_camera_component(
        {
            "backend": runtime.tracker_backend,
            "config": TRACKER_CONFIG,
            "version": runtime.tracker_version,
        },
        name="tracker",
    )
    if detector["backend"] != "ultralytics_yolo" or tracker["backend"] != "bytetrack":
        raise CameraAdapterError("tracking runtime component backend mismatch")
    if detector["version"] != tracker["version"]:
        raise CameraAdapterError("tracking runtime component version mismatch")
    return detector, tracker


__all__ = [
    "CameraCollectionPreparationError",
    "build_authorized_camera_tracking_pair",
]
