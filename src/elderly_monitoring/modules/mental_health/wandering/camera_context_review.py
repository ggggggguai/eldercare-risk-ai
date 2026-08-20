"""W5D-03A three-frame purpose/context review core."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import cv2
import httpx
import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    _validate_episode_result,
)


CAMERA_CONTEXT_REVIEW_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-context-review-config-v1"
)
CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION = "wandering-handoff-context-review-v2"
CONTEXT_LABELS = frozenset(
    {"phone_call", "searching", "cleaning", "exercise", "social", "other", "unknown"}
)
PROVIDER_MODES = frozenset({"disabled", "fake", "openai-compatible"})
FRAME_POSITIONS = ("start", "middle", "end")
_DIGEST_LENGTH = 64


class CameraContextReviewError(ValueError):
    """Context input, identity, provider contract, or output is invalid."""


@dataclass(frozen=True)
class ContextFrame:
    position: str
    requested_timestamp_sec: float
    selected_timestamp_sec: float | None
    frame_index: int | None
    track_id: str | int | None
    bbox_xyxy: tuple[float, float, float, float] | None
    status: str
    error_code: str | None
    scene_ref: Mapping[str, Any]
    person_crop_ref: Mapping[str, Any]
    quality_flags: tuple[str, ...]
    scene_bytes: bytes | None
    person_crop_bytes: bytes | None

    @classmethod
    def unavailable(
        cls,
        *,
        position: str,
        requested_timestamp_sec: float,
        error_code: str,
        selected_timestamp_sec: float | None = None,
        frame_index: int | None = None,
        track_id: str | int | None = None,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
        scene_ref: Mapping[str, Any] | None = None,
        scene_bytes: bytes | None = None,
        quality_flags: Sequence[str] = (),
    ) -> ContextFrame:
        return cls(
            position=position,
            requested_timestamp_sec=float(requested_timestamp_sec),
            selected_timestamp_sec=selected_timestamp_sec,
            frame_index=frame_index,
            track_id=track_id,
            bbox_xyxy=bbox_xyxy,
            status="unavailable",
            error_code=error_code,
            scene_ref=scene_ref or _null_ref("context_scene_frame", f"{position}:scene"),
            person_crop_ref=_null_ref("context_person_crop", f"{position}:crop"),
            quality_flags=tuple(_unique_strings(quality_flags)),
            scene_bytes=scene_bytes,
            person_crop_bytes=None,
        )

    def as_reference(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "requested_timestamp_sec": self.requested_timestamp_sec,
            "selected_timestamp_sec": self.selected_timestamp_sec,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "status": self.status,
            "error_code": self.error_code,
            "scene_ref": dict(self.scene_ref),
            "person_crop_ref": dict(self.person_crop_ref),
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True)
class ContextReviewRequest:
    episode_id: str
    source_video_id: str
    frames: tuple[ContextFrame, ...]
    tracking_qc_summary: Mapping[str, Any]


@dataclass(frozen=True)
class ContextProviderResult:
    status: str
    context_label: str
    confidence: float | None
    rationale: str | None
    error_code: str | None
    provider: str
    provider_model: str | None


class ContextProvider(Protocol):
    provider_id: str
    model_id: str | None
    mode: str

    def review(self, request: ContextReviewRequest) -> ContextProviderResult:
        """Review one eligible episode without mutating it."""


@dataclass(frozen=True)
class CameraContextReviewBuildResult:
    output_dir: Path
    input_episode_count: int
    eligible_episode_count: int
    context_review_count: int
    ready_count: int
    unavailable_count: int
    status: str


class DisabledContextProvider:
    mode = "disabled"
    provider_id = "provider-disabled"
    model_id = None

    def review(self, request: ContextReviewRequest) -> ContextProviderResult:
        del request
        return _unavailable_provider_result(
            provider=self.provider_id,
            provider_model=self.model_id,
            error_code="provider_disabled",
        )


class DeterministicFakeContextProvider:
    mode = "fake"
    provider_id = "deterministic-fake"
    model_id = "wandering-context-fake-v1"
    _LABELS = ("phone_call", "searching", "cleaning", "exercise", "social", "other")

    def review(self, request: ContextReviewRequest) -> ContextProviderResult:
        digest = hashlib.sha256(
            f"{request.source_video_id}:{request.episode_id}".encode("utf-8")
        ).digest()
        label = self._LABELS[digest[0] % len(self._LABELS)]
        return ContextProviderResult(
            status="ready",
            context_label=label,
            confidence=0.5,
            rationale="Deterministic fake response for offline contract verification.",
            error_code=None,
            provider=self.provider_id,
            provider_model=self.model_id,
        )


class OpenAICompatibleContextProvider:
    """Small OpenAI-compatible multimodal adapter with strict response parsing."""

    mode = "openai-compatible"

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        model: str | None,
        provider_id: str = "openai-compatible",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.strip() if isinstance(endpoint, str) else None
        self.api_key = api_key.strip() if isinstance(api_key, str) else None
        self.model_id = model.strip() if isinstance(model, str) else None
        self.provider_id = provider_id
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    def review(self, request: ContextReviewRequest) -> ContextProviderResult:
        if not self.endpoint:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_endpoint_missing",
            )
        if not self.api_key:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_api_key_missing",
            )
        if not self.model_id:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=None,
                error_code="provider_model_missing",
            )
        try:
            payload = self._request_payload(request)
        except CameraContextReviewError:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_request_invalid",
            )
        try:
            if self._client is None:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        headers=self._headers(),
                        json=payload,
                    )
            else:
                response = self._client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
        except httpx.TimeoutException:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_timeout",
            )
        except httpx.HTTPStatusError:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_http_error",
            )
        except httpx.RequestError:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_request_error",
            )
        except Exception:
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_error",
            )
        try:
            parsed = _parse_openai_compatible_response(response.json())
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return _unavailable_provider_result(
                provider=self.provider_id,
                provider_model=self.model_id,
                error_code="provider_invalid_response",
            )
        return ContextProviderResult(
            status="ready" if parsed["context_label"] != "unknown" else "uncertain",
            context_label=parsed["context_label"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            error_code=None,
            provider=self.provider_id,
            provider_model=self.model_id,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_payload(self, request: ContextReviewRequest) -> dict[str, Any]:
        if len(request.frames) != 3 or any(frame.status != "ready" for frame in request.frames):
            raise CameraContextReviewError("provider request requires three ready frames")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Classify only observable purpose/context. Return one JSON object with "
                    "context_label, confidence, rationale. Allowed labels: "
                    "phone_call, searching, cleaning, exercise, social, other, unknown. "
                    "Do not infer or alter trajectory shape or episode boundaries. "
                    f"episode_id={request.episode_id}; source_video_id={request.source_video_id}."
                ),
            }
        ]
        for frame in request.frames:
            if frame.scene_bytes is None or frame.person_crop_bytes is None:
                raise CameraContextReviewError("provider frame bytes are unavailable")
            content.append({"type": "text", "text": f"{frame.position} full scene"})
            content.append(_image_part(frame.scene_bytes))
            content.append({"type": "text", "text": f"{frame.position} person crop"})
            content.append(_image_part(frame.person_crop_bytes))
        return {
            "model": self.model_id,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a visual context reviewer. Describe observable purpose only; "
                        "never change shape, boundary, truth, QC, or model decisions."
                    ),
                },
                {"role": "user", "content": content},
            ],
        }


def load_camera_context_review_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the W5D-03A production configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraContextReviewError("cannot read camera context review config") from exc
    if not isinstance(value, Mapping):
        raise CameraContextReviewError("camera context review config must be a mapping")
    config = dict(value)
    required = {
        "schema_version",
        "context_review_id",
        "inputs",
        "home_input_status",
        "home_annotation_status",
        "home_smoke_status",
        "validation_scope",
        "trigger",
        "frames",
        "provider",
        "outputs",
    }
    if set(config) != required:
        raise CameraContextReviewError("camera context review config fields drifted")
    if config["schema_version"] != CAMERA_CONTEXT_REVIEW_CONFIG_SCHEMA_VERSION:
        raise CameraContextReviewError("camera context review config schema drifted")
    _nonempty(config["context_review_id"], "context_review_id")
    inputs = _mapping(config["inputs"], "inputs")
    if set(inputs) != {"episode_results", "development_index", "context_schema"}:
        raise CameraContextReviewError("camera context input fields drifted")
    for name in ("episode_results", "context_schema"):
        descriptor = _mapping(inputs[name], name)
        if set(descriptor) != {"path", "sha256", "schema_version"}:
            raise CameraContextReviewError(f"{name} descriptor fields drifted")
        _descriptor_syntax(descriptor, name)
    development = _mapping(inputs["development_index"], "development_index")
    if set(development) != {"path", "sha256", "schema_version", "expected_batch_counts"}:
        raise CameraContextReviewError("development index descriptor fields drifted")
    _descriptor_syntax(development, "development_index")
    if development["schema_version"] != "wandering-camera-development-index-v1":
        raise CameraContextReviewError("development index schema identity drifted")
    expected_counts = _mapping(development["expected_batch_counts"], "expected_batch_counts")
    if expected_counts != {"B01": 36, "B02": 12}:
        raise CameraContextReviewError("development batch counts drifted")
    if inputs["episode_results"]["schema_version"] != "wandering-handoff-episode-result-v1":
        raise CameraContextReviewError("episode result schema identity drifted")
    if inputs["context_schema"]["schema_version"] != CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION:
        raise CameraContextReviewError("context result schema identity drifted")
    if config["home_input_status"] not in {"awaiting_input", "available"}:
        raise CameraContextReviewError("home input status is invalid")
    if config["home_annotation_status"] not in {"not_provided", "unlabeled", "available"}:
        raise CameraContextReviewError("home annotation status is invalid")
    if config["home_smoke_status"] not in {
        "not_run_input_unavailable",
        "ready",
        "uncertain",
        "error",
    }:
        raise CameraContextReviewError("home smoke status is invalid")
    if config["validation_scope"] not in {
        "b01_b02_development",
        "deterministic_fixture",
        "home_truth_free_smoke",
        "home_labeled_development",
    }:
        raise CameraContextReviewError("validation scope is invalid")
    trigger = _mapping(config["trigger"], "trigger")
    if set(trigger) != {"policy_id", "binary_labels", "episode_statuses", "explicit_episode_ids"}:
        raise CameraContextReviewError("trigger fields drifted")
    _nonempty(trigger["policy_id"], "trigger.policy_id")
    if trigger["binary_labels"] != ["wandering_like"] or trigger["episode_statuses"] != ["uncertain"]:
        raise CameraContextReviewError("trigger policy drifted")
    if not isinstance(trigger["explicit_episode_ids"], list):
        raise CameraContextReviewError("explicit episode IDs must be a list")
    frames = _mapping(config["frames"], "frames")
    if set(frames) != {"positions", "jpeg_quality", "bbox_max_gap_seconds", "crop_padding_ratio"}:
        raise CameraContextReviewError("frame selection fields drifted")
    if frames["positions"] != list(FRAME_POSITIONS):
        raise CameraContextReviewError("frame positions drifted")
    if not isinstance(frames["jpeg_quality"], int) or not 1 <= frames["jpeg_quality"] <= 100:
        raise CameraContextReviewError("JPEG quality is invalid")
    if float(frames["bbox_max_gap_seconds"]) <= 0:
        raise CameraContextReviewError("bbox max gap must be positive")
    if not 0 <= float(frames["crop_padding_ratio"]) <= 1:
        raise CameraContextReviewError("crop padding ratio is invalid")
    provider = _mapping(config["provider"], "provider")
    if set(provider) != {"default_mode", "allowed_modes", "disabled", "fake", "openai_compatible"}:
        raise CameraContextReviewError("provider config fields drifted")
    if provider["default_mode"] not in PROVIDER_MODES or set(provider["allowed_modes"]) != PROVIDER_MODES:
        raise CameraContextReviewError("provider modes drifted")
    outputs = _mapping(config["outputs"], "outputs")
    if set(outputs) != {
        "required_files",
        "frame_artifact_dir",
        "fresh_output_directory_must_not_exist",
        "algorithm_event_emitted",
    }:
        raise CameraContextReviewError("output config fields drifted")
    if outputs["required_files"] != [
        "context_reviews.jsonl",
        "run_summary.json",
        "handoff_manifest.partial.json",
        "README.md",
        "VERIFICATION.md",
    ]:
        raise CameraContextReviewError("required context output files drifted")
    if outputs["fresh_output_directory_must_not_exist"] is not True:
        raise CameraContextReviewError("fresh output policy drifted")
    if outputs["algorithm_event_emitted"] is not False:
        raise CameraContextReviewError("AlgorithmEvent boundary drifted")
    return config


def select_eligible_episode_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    binary_labels: Iterable[str] = ("wandering_like",),
    episode_statuses: Iterable[str] = ("uncertain",),
    explicit_episode_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select context triggers without modifying the input episode rows."""

    binary_set = set(binary_labels)
    status_set = set(episode_statuses)
    explicit_set = set(explicit_episode_ids)
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    seen: set[str] = set()
    for original in rows:
        row = dict(original)
        episode_id = _nonempty(row.get("episode_id"), "episode_id")
        if episode_id in seen:
            raise CameraContextReviewError("episode result IDs are duplicated")
        seen.add(episode_id)
        binary = row.get("binary")
        binary_label = binary.get("predicted_label") if isinstance(binary, Mapping) else None
        eligible = (
            binary_label in binary_set
            or row.get("status") in status_set
            or episode_id in explicit_set
        )
        if eligible:
            selected.append(row)
        else:
            skipped["not_context_eligible"] += 1
    selected.sort(key=lambda row: (str(row.get("source_video_id")), float(row.get("start_sec", 0)), str(row["episode_id"])))
    return selected, dict(skipped)


def extract_episode_frames(
    *,
    episode: Mapping[str, Any],
    video_path: str | Path,
    tracking_rows: Sequence[Mapping[str, Any]],
    stage_output_root: str | Path,
    final_output_root: str | Path,
    jpeg_quality: int,
    bbox_max_gap_seconds: float,
    crop_padding_ratio: float,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
) -> tuple[ContextFrame, ContextFrame, ContextFrame]:
    """Extract start/middle/end scene frames and person crops for one episode."""

    start = float(episode["start_sec"])
    end = float(episode["end_sec_exclusive"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise CameraContextReviewError("episode interval is invalid")
    capture = capture_factory(str(video_path))
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        return _unavailable_frame_triplet(start, end, "video_open_failed")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 1.0
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        duration = frame_count / fps if frame_count > 0 else end
        last_time = max(0.0, duration - 1.0 / fps)
        requested = (
            start,
            (start + end) / 2.0,
            max(start, end - 1.0 / fps),
        )
        target_track = _dominant_track_id(tracking_rows, start=start, end=end)
        episode_token = hashlib.sha256(str(episode["episode_id"]).encode("utf-8")).hexdigest()[:20]
        frames: list[ContextFrame] = []
        for position, timestamp in zip(FRAME_POSITIONS, requested, strict=True):
            flags: list[str] = []
            selected = min(max(timestamp, 0.0), last_time)
            if abs(selected - timestamp) > 1e-6:
                flags.append("frame_timestamp_clamped")
            capture.set(cv2.CAP_PROP_POS_MSEC, selected * 1000.0)
            ok, image = capture.read()
            if not ok or image is None or not isinstance(image, np.ndarray) or image.size == 0:
                frames.append(
                    ContextFrame.unavailable(
                        position=position,
                        requested_timestamp_sec=timestamp,
                        selected_timestamp_sec=selected,
                        error_code="frame_decode_failed",
                        quality_flags=flags,
                    )
                )
                continue
            raw_index = float(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1.0
            frame_index = max(0, int(round(raw_index))) if math.isfinite(raw_index) else max(0, int(round(selected * fps)))
            decoded_raw = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            decoded = decoded_raw if math.isfinite(decoded_raw) and decoded_raw >= 0 else frame_index / fps
            if abs(decoded - selected) > max(0.1, 1.5 / fps):
                flags.append("frame_seek_approximate")
            scene_rel = Path("frame_artifacts") / episode_token / f"{position}-scene.jpg"
            scene_bytes = _encode_jpeg(image, quality=jpeg_quality)
            scene_ref = _write_frame_artifact(
                stage_root=Path(stage_output_root),
                final_root=Path(final_output_root),
                relative_path=scene_rel,
                payload=scene_bytes,
                ref_type="context_scene_frame",
                ref_id=f"{episode['episode_id']}:{position}:scene:frame-{frame_index}:t-{decoded:.6f}",
            )
            tracking = _nearest_tracking_row(
                tracking_rows,
                timestamp=decoded,
                track_id=target_track,
                max_gap_seconds=float(bbox_max_gap_seconds),
            )
            if tracking is None:
                frames.append(
                    ContextFrame.unavailable(
                        position=position,
                        requested_timestamp_sec=timestamp,
                        selected_timestamp_sec=decoded,
                        frame_index=frame_index,
                        track_id=target_track,
                        error_code="bbox_unavailable",
                        scene_ref=scene_ref,
                        scene_bytes=scene_bytes,
                        quality_flags=[*flags, "person_crop_unavailable"],
                    )
                )
                continue
            bbox = _normalized_bbox(tracking.get("bbox"), image.shape, padding_ratio=crop_padding_ratio)
            if bbox is None:
                frames.append(
                    ContextFrame.unavailable(
                        position=position,
                        requested_timestamp_sec=timestamp,
                        selected_timestamp_sec=decoded,
                        frame_index=frame_index,
                        track_id=tracking.get("track_id"),
                        error_code="bbox_invalid",
                        scene_ref=scene_ref,
                        scene_bytes=scene_bytes,
                        quality_flags=[*flags, "person_crop_unavailable"],
                    )
                )
                continue
            x1, y1, x2, y2 = bbox
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                frames.append(
                    ContextFrame.unavailable(
                        position=position,
                        requested_timestamp_sec=timestamp,
                        selected_timestamp_sec=decoded,
                        frame_index=frame_index,
                        track_id=tracking.get("track_id"),
                        bbox_xyxy=bbox,
                        error_code="person_crop_empty",
                        scene_ref=scene_ref,
                        scene_bytes=scene_bytes,
                        quality_flags=[*flags, "person_crop_unavailable"],
                    )
                )
                continue
            tracking_time = float(tracking.get("timestamp_sec", decoded))
            if abs(tracking_time - decoded) > 1.5 / fps:
                flags.append("bbox_timestamp_offset")
            crop_rel = Path("frame_artifacts") / episode_token / f"{position}-crop.jpg"
            crop_bytes = _encode_jpeg(crop, quality=jpeg_quality)
            crop_ref = _write_frame_artifact(
                stage_root=Path(stage_output_root),
                final_root=Path(final_output_root),
                relative_path=crop_rel,
                payload=crop_bytes,
                ref_type="context_person_crop",
                ref_id=f"{episode['episode_id']}:{position}:crop:frame-{frame_index}:t-{decoded:.6f}",
            )
            frames.append(
                ContextFrame(
                    position=position,
                    requested_timestamp_sec=timestamp,
                    selected_timestamp_sec=decoded,
                    frame_index=frame_index,
                    track_id=tracking.get("track_id"),
                    bbox_xyxy=bbox,
                    status="ready",
                    error_code=None,
                    scene_ref=scene_ref,
                    person_crop_ref=crop_ref,
                    quality_flags=tuple(_unique_strings(flags)),
                    scene_bytes=scene_bytes,
                    person_crop_bytes=crop_bytes,
                )
            )
        return tuple(frames)  # type: ignore[return-value]
    finally:
        capture.release()


def _build_context_review_row(
    *,
    episode: Mapping[str, Any],
    frames: Sequence[ContextFrame],
    provider: ContextProvider,
    config_id: str,
    config_sha256: str,
    policy_id: str,
    policy_sha256: str,
    source_refs: Sequence[Mapping[str, Any]],
    explicit_episode_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one context row while preserving the upstream episode snapshot."""

    if len(frames) != 3 or tuple(frame.position for frame in frames) != FRAME_POSITIONS:
        raise CameraContextReviewError("context review requires ordered start/middle/end frames")
    reasons = _trigger_reasons(episode, explicit_episode_ids=explicit_episode_ids)
    if not reasons:
        raise CameraContextReviewError("context row requested for an ineligible episode")
    unavailable_frames = [frame for frame in frames if frame.status != "ready"]
    if unavailable_frames:
        result = _unavailable_provider_result(
            provider=provider.provider_id,
            provider_model=provider.model_id,
            error_code=unavailable_frames[0].error_code or "frame_unavailable",
        )
    else:
        result = provider.review(
            ContextReviewRequest(
                episode_id=str(episode["episode_id"]),
                source_video_id=str(episode["source_video_id"]),
                frames=tuple(frames),
                tracking_qc_summary={
                    "episode_status": episode.get("status"),
                    "qc_status": episode.get("qc_status"),
                    "quality_flags": list(episode.get("quality_flags", [])),
                },
            )
        )
    episode_bytes = canonical_json_bytes(episode)
    review_digest = hashlib.sha256(
        f"{config_id}:{episode['episode_id']}".encode("utf-8")
    ).hexdigest()
    quality_flags = [
        "context_only_does_not_modify_episode",
        "development_only",
        f"provider_mode_{provider.mode.replace('-', '_')}",
    ]
    quality_flags.extend(flag for frame in frames for flag in frame.quality_flags)
    if result.status == "unavailable":
        quality_flags.append("context_provider_or_frame_unavailable")
    binary = episode.get("binary")
    binary_label = binary.get("predicted_label") if isinstance(binary, Mapping) else None
    return {
        "schema_version": CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
        "module": "mental_health",
        "record_id": f"w5d03a-{review_digest}",
        "context_review_id": f"context-review-{review_digest}",
        "episode_id": str(episode["episode_id"]),
        "episode_record_id": str(episode["record_id"]),
        "person_id": str(episode["person_id"]),
        "session_id": str(episode["session_id"]),
        "source_video_id": str(episode["source_video_id"]),
        "start_time": episode.get("start_time"),
        "end_time": episode.get("end_time"),
        "start_sec": float(episode["start_sec"]),
        "end_sec_exclusive": float(episode["end_sec_exclusive"]),
        "status": result.status,
        "context_label": result.context_label,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "error_code": result.error_code,
        "provider": result.provider,
        "provider_model": result.provider_model,
        "trigger": {
            "reasons": reasons,
            "binary_label": binary_label,
            "episode_status": str(episode["status"]),
        },
        "frame_refs": [frame.as_reference() for frame in frames],
        "quality_flags": _unique_strings(quality_flags),
        "episode_identity": dict(_mapping(episode["identity"], "episode.identity")),
        "identity": {
            "model_id": result.provider_model,
            "model_sha256": None,
            "config_id": config_id,
            "config_sha256": config_sha256,
            "policy_id": policy_id,
            "policy_sha256": policy_sha256,
        },
        "episode_snapshot_sha256": hashlib.sha256(episode_bytes).hexdigest(),
        "source_refs": [dict(value) for value in _deduplicate_refs(source_refs)],
    }


def build_camera_context_review_bundle(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    provider_mode: str | None = None,
    source_video_ids: Sequence[str] | None = None,
    episode_results_path: str | Path | None = None,
    provider: ContextProvider | None = None,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    now: Callable[[], datetime] | None = None,
) -> CameraContextReviewBuildResult:
    """Build a fresh W5D-03A context bundle from immutable W5D-02 results."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera context review output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=True)
    config = load_camera_context_review_config(config_file)
    config_sha256 = _sha256_file(config_file)
    input_files = {
        name: _verify_descriptor(root, descriptor, name)
        for name, descriptor in config["inputs"].items()
        if name != "development_index"
    }
    input_files["development_index"] = _verify_descriptor(
        root, config["inputs"]["development_index"], "development_index"
    )
    if episode_results_path is not None:
        override_descriptor = dict(config["inputs"]["episode_results"])
        override_descriptor["path"] = Path(episode_results_path).resolve().as_posix()
        input_files["episode_results"] = _verify_descriptor(
            root, override_descriptor, "episode_results override"
        )
    schema_payload = json.loads(input_files["context_schema"].read_text(encoding="utf-8"))
    if schema_payload.get("$id") != CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION:
        raise CameraContextReviewError("context schema file identity drifted")
    episodes = _load_jsonl(input_files["episode_results"], "episode results")
    for episode in episodes:
        _validate_episode_result(episode)
    index_rows = _load_development_index(
        input_files["development_index"],
        expected_counts=config["inputs"]["development_index"]["expected_batch_counts"],
    )
    index_by_video = {str(row["source_video_id"]): row for row in index_rows}
    missing_index = sorted({str(row["source_video_id"]) for row in episodes} - set(index_by_video))
    if missing_index:
        raise CameraContextReviewError("episode results reference videos outside the development index")
    requested_videos = set(source_video_ids or ())
    if requested_videos:
        unknown = sorted(requested_videos - set(index_by_video))
        if unknown:
            raise CameraContextReviewError(f"unknown source video IDs: {unknown}")
        episodes = [row for row in episodes if str(row["source_video_id"]) in requested_videos]
        if not episodes:
            raise CameraContextReviewError("source video filter selected no episode results")
    episode_input_snapshot = canonical_jsonl_bytes(episodes)
    trigger = config["trigger"]
    eligible, skipped = select_eligible_episode_results(
        episodes,
        binary_labels=trigger["binary_labels"],
        episode_statuses=trigger["episode_statuses"],
        explicit_episode_ids=trigger["explicit_episode_ids"],
    )
    if not eligible:
        raise CameraContextReviewError("context review input has no eligible episodes")
    mode = provider_mode or str(config["provider"]["default_mode"])
    if mode not in PROVIDER_MODES or mode not in config["provider"]["allowed_modes"]:
        raise CameraContextReviewError("provider mode is not allowed")
    provider_instance = provider or _provider_from_config(config["provider"], mode=mode)
    if provider is not None:
        mode = provider.mode
    started_at = (now or _utc_now)()
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if run_id is None:
        run_id = "w5d03a-" + hashlib.sha256(
            f"{config_sha256}:{mode}:{','.join(sorted(requested_videos))}".encode("utf-8")
        ).hexdigest()[:16]
    _nonempty(run_id, "run_id")
    policy_sha256 = hashlib.sha256(canonical_json_bytes(trigger)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    final_root = output.resolve(strict=False)
    tracking_cache: dict[str, list[dict[str, Any]]] = {}
    context_rows: list[dict[str, Any]] = []
    try:
        for episode in eligible:
            source_video_id = str(episode["source_video_id"])
            index_row = index_by_video[source_video_id]
            artifacts = index_row["artifacts"]
            if source_video_id not in tracking_cache:
                tracking_cache[source_video_id] = _load_jsonl(
                    Path(artifacts["tracking"]["path"]), "tracking"
                )
            try:
                frames = extract_episode_frames(
                    episode=episode,
                    video_path=artifacts["video"]["path"],
                    tracking_rows=tracking_cache[source_video_id],
                    stage_output_root=stage,
                    final_output_root=final_root,
                    jpeg_quality=int(config["frames"]["jpeg_quality"]),
                    bbox_max_gap_seconds=float(config["frames"]["bbox_max_gap_seconds"]),
                    crop_padding_ratio=float(config["frames"]["crop_padding_ratio"]),
                    capture_factory=capture_factory,
                )
            except Exception:
                frames = _unavailable_frame_triplet(
                    float(episode["start_sec"]),
                    float(episode["end_sec_exclusive"]),
                    "frame_extraction_failed",
                )
            source_refs = _context_source_refs(
                episode=episode,
                index_row=index_row,
                episode_results_path=input_files["episode_results"],
                episode_results_sha=str(config["inputs"]["episode_results"]["sha256"]),
                development_index_path=input_files["development_index"],
                development_index_sha=str(config["inputs"]["development_index"]["sha256"]),
            )
            row = _build_context_review_row(
                episode=episode,
                frames=frames,
                provider=provider_instance,
                config_id=str(config["context_review_id"]),
                config_sha256=config_sha256,
                policy_id=str(trigger["policy_id"]),
                policy_sha256=policy_sha256,
                source_refs=source_refs,
                explicit_episode_ids=trigger["explicit_episode_ids"],
            )
            _validate_context_review(row)
            context_rows.append(row)
        if canonical_jsonl_bytes(episodes) != episode_input_snapshot:
            raise CameraContextReviewError("context producer mutated upstream episode results")
        if len(context_rows) != len(eligible):
            raise CameraContextReviewError("eligible episode and context row counts differ")
        if len({row["episode_id"] for row in context_rows}) != len(context_rows):
            raise CameraContextReviewError("context rows are not one-to-one with episodes")
        context_rows.sort(
            key=lambda row: (
                str(row["source_video_id"]),
                float(row["start_sec"]),
                str(row["episode_id"]),
            )
        )
        finished_at = (now or _utc_now)()
        if finished_at.tzinfo is None:
            finished_at = finished_at.replace(tzinfo=timezone.utc)
        status = _context_run_status(context_rows, provider_mode=mode)
        summary = _build_run_summary(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            episodes=episodes,
            eligible=eligible,
            context_rows=context_rows,
            skipped=skipped,
            provider_mode=mode,
            provider=provider_instance,
            config=config,
            config_sha256=config_sha256,
            policy_sha256=policy_sha256,
            input_files=input_files,
        )
        context_payload = canonical_jsonl_bytes(context_rows)
        summary_payload = canonical_json_bytes(summary)
        readme_payload = _readme_text(config, summary, provider_mode=mode).encode("utf-8")
        verification_payload = _verification_text(
            config=config,
            config_sha256=config_sha256,
            policy_sha256=policy_sha256,
            summary=summary,
            context_rows=context_rows,
            skipped=skipped,
            provider_mode=mode,
        ).encode("utf-8")
        payloads = {
            "context_reviews.jsonl": context_payload,
            "run_summary.json": summary_payload,
            "README.md": readme_payload,
            "VERIFICATION.md": verification_payload,
        }
        for name, payload in payloads.items():
            (stage / name).write_bytes(payload)
        manifest = _build_partial_manifest(
            run_id=run_id,
            status=status,
            config=config,
            config_sha256=config_sha256,
            policy_sha256=policy_sha256,
            provider_mode=mode,
            context_rows=context_rows,
            stage=stage,
            final_root=final_root,
            payloads=payloads,
            input_files=input_files,
        )
        manifest_payload = canonical_json_bytes(manifest)
        (stage / "handoff_manifest.partial.json").write_bytes(manifest_payload)
        actual_files = {path.name for path in stage.iterdir() if path.is_file()}
        if actual_files != set(config["outputs"]["required_files"]):
            raise CameraContextReviewError("top-level context output files drifted")
        if output.exists():
            raise FileExistsError(f"camera context review output already exists: {output}")
        os.replace(stage, output)
        counts = Counter(str(row["status"]) for row in context_rows)
        return CameraContextReviewBuildResult(
            output_dir=output,
            input_episode_count=len(episodes),
            eligible_episode_count=len(eligible),
            context_review_count=len(context_rows),
            ready_count=counts.get("ready", 0),
            unavailable_count=counts.get("unavailable", 0),
            status=status,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_context_review(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "module",
        "record_id",
        "context_review_id",
        "episode_id",
        "episode_record_id",
        "person_id",
        "session_id",
        "source_video_id",
        "start_time",
        "end_time",
        "start_sec",
        "end_sec_exclusive",
        "status",
        "context_label",
        "confidence",
        "rationale",
        "error_code",
        "provider",
        "provider_model",
        "trigger",
        "frame_refs",
        "quality_flags",
        "episode_identity",
        "identity",
        "episode_snapshot_sha256",
        "source_refs",
    }
    if set(value) != expected:
        raise CameraContextReviewError("context review fields drifted")
    if value["schema_version"] != CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION or value["module"] != "mental_health":
        raise CameraContextReviewError("context review schema or module drifted")
    for name in (
        "record_id",
        "context_review_id",
        "episode_id",
        "episode_record_id",
        "person_id",
        "session_id",
        "source_video_id",
        "provider",
    ):
        _nonempty(value[name], name)
    start = float(value["start_sec"])
    end = float(value["end_sec_exclusive"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise CameraContextReviewError("context interval is invalid")
    if value["status"] not in {"ready", "uncertain", "unavailable", "error"}:
        raise CameraContextReviewError("context status is invalid")
    if value["context_label"] not in CONTEXT_LABELS:
        raise CameraContextReviewError("context label is invalid")
    confidence = value["confidence"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise CameraContextReviewError("context confidence is invalid")
    if value["status"] == "unavailable":
        if value["context_label"] != "unknown" or confidence is not None or value["rationale"] is not None:
            raise CameraContextReviewError("unavailable context must not contain a fabricated result")
        _nonempty(value["error_code"], "error_code")
    trigger = _mapping(value["trigger"], "trigger")
    if set(trigger) != {"reasons", "binary_label", "episode_status"}:
        raise CameraContextReviewError("context trigger fields drifted")
    if not isinstance(trigger["reasons"], list) or not trigger["reasons"]:
        raise CameraContextReviewError("context trigger reasons are missing")
    frames = value["frame_refs"]
    if not isinstance(frames, list) or len(frames) != 3:
        raise CameraContextReviewError("context must contain three frame references")
    if tuple(frame.get("position") for frame in frames) != FRAME_POSITIONS:
        raise CameraContextReviewError("context frame positions drifted")
    for frame in frames:
        _validate_frame_ref(frame)
    _validate_identity(value["episode_identity"], "episode_identity")
    _validate_identity(value["identity"], "identity")
    _digest(value["episode_snapshot_sha256"], "episode_snapshot_sha256")
    if not isinstance(value["quality_flags"], list) or len(value["quality_flags"]) != len(set(value["quality_flags"])):
        raise CameraContextReviewError("context quality flags are invalid")
    refs = value["source_refs"]
    if not isinstance(refs, list) or not refs:
        raise CameraContextReviewError("context source refs are missing")
    for ref in refs:
        _validate_source_ref(ref)


def _validate_frame_ref(value: Mapping[str, Any]) -> None:
    expected = {
        "position",
        "requested_timestamp_sec",
        "selected_timestamp_sec",
        "frame_index",
        "track_id",
        "bbox_xyxy",
        "status",
        "error_code",
        "scene_ref",
        "person_crop_ref",
        "quality_flags",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CameraContextReviewError("context frame reference fields drifted")
    if value["position"] not in FRAME_POSITIONS or value["status"] not in {"ready", "unavailable"}:
        raise CameraContextReviewError("context frame position or status is invalid")
    timestamp = value["requested_timestamp_sec"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or timestamp < 0:
        raise CameraContextReviewError("requested frame timestamp is invalid")
    selected = value["selected_timestamp_sec"]
    if selected is not None and (
        isinstance(selected, bool) or not isinstance(selected, (int, float)) or selected < 0
    ):
        raise CameraContextReviewError("selected frame timestamp is invalid")
    bbox = value["bbox_xyxy"]
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
        raise CameraContextReviewError("context bbox is invalid")
    _validate_source_ref(value["scene_ref"])
    _validate_source_ref(value["person_crop_ref"])
    if value["status"] == "ready":
        if value["error_code"] is not None or selected is None or value["frame_index"] is None or bbox is None:
            raise CameraContextReviewError("ready context frame is incomplete")
        if value["scene_ref"]["sha256"] is None or value["person_crop_ref"]["sha256"] is None:
            raise CameraContextReviewError("ready context frame artifacts are missing")
    elif not isinstance(value["error_code"], str) or not value["error_code"]:
        raise CameraContextReviewError("unavailable context frame error is missing")


def _provider_from_config(config: Mapping[str, Any], *, mode: str) -> ContextProvider:
    if mode == "disabled":
        return DisabledContextProvider()
    if mode == "fake":
        return DeterministicFakeContextProvider()
    value = config["openai_compatible"]
    endpoint = os.environ.get(str(value["endpoint_env"]))
    api_key = os.environ.get(str(value["api_key_env"]))
    model = os.environ.get(str(value["model_env"])) or str(value["default_model"])
    return OpenAICompatibleContextProvider(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        provider_id=str(value["provider_id"]),
        timeout_seconds=float(value["timeout_seconds"]),
    )


def _trigger_reasons(
    episode: Mapping[str, Any], *, explicit_episode_ids: Iterable[str] = ()
) -> list[str]:
    reasons: list[str] = []
    binary = episode.get("binary")
    if isinstance(binary, Mapping) and binary.get("predicted_label") == "wandering_like":
        reasons.append("binary_wandering_like")
    if episode.get("status") == "uncertain":
        reasons.append("episode_status_uncertain")
    if str(episode.get("episode_id")) in set(explicit_episode_ids):
        reasons.append("explicit_review")
    return _unique_strings(reasons)


def _parse_openai_compatible_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("provider payload is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("provider choices are invalid")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("provider message is invalid")
    content = message.get("content")
    if isinstance(content, list):
        parts = [part.get("text") for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
        if len(parts) != 1:
            raise ValueError("provider content parts are invalid")
        content = parts[0]
    if not isinstance(content, str):
        raise ValueError("provider content is invalid")
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"context_label", "confidence", "rationale"}:
        raise ValueError("provider result fields drifted")
    label = parsed["context_label"]
    confidence = parsed["confidence"]
    rationale = parsed["rationale"]
    if label not in CONTEXT_LABELS:
        raise ValueError("provider label is invalid")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("provider confidence is invalid")
    if rationale is not None and not isinstance(rationale, str):
        raise ValueError("provider rationale is invalid")
    return {
        "context_label": str(label),
        "confidence": float(confidence),
        "rationale": rationale,
    }


def _image_part(payload: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(payload).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _unavailable_provider_result(
    *, provider: str, provider_model: str | None, error_code: str
) -> ContextProviderResult:
    return ContextProviderResult(
        status="unavailable",
        context_label="unknown",
        confidence=None,
        rationale=None,
        error_code=error_code,
        provider=provider,
        provider_model=provider_model,
    )


def _unavailable_frame_triplet(
    start: float, end: float, error_code: str
) -> tuple[ContextFrame, ContextFrame, ContextFrame]:
    midpoint = (start + end) / 2.0
    timestamps = (start, midpoint, max(start, end - 1e-6))
    return tuple(
        ContextFrame.unavailable(
            position=position,
            requested_timestamp_sec=timestamp,
            error_code=error_code,
        )
        for position, timestamp in zip(FRAME_POSITIONS, timestamps, strict=True)
    )  # type: ignore[return-value]


def _dominant_track_id(
    rows: Sequence[Mapping[str, Any]], *, start: float, end: float
) -> str | int | None:
    counts: Counter[Any] = Counter()
    confidences: defaultdict[Any, float] = defaultdict(float)
    for row in rows:
        try:
            timestamp = float(row["timestamp_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        track_id = row.get("track_id")
        if track_id is None or not start <= timestamp < end:
            continue
        counts[track_id] += 1
        confidences[track_id] += float(row.get("track_confidence") or 0.0)
    if not counts:
        return None
    return sorted(counts, key=lambda value: (-counts[value], -confidences[value], str(value)))[0]


def _nearest_tracking_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestamp: float,
    track_id: str | int | None,
    max_gap_seconds: float,
) -> Mapping[str, Any] | None:
    candidates: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in rows:
        if track_id is not None and row.get("track_id") != track_id:
            continue
        try:
            offset = abs(float(row["timestamp_sec"]) - timestamp)
        except (KeyError, TypeError, ValueError):
            continue
        if offset <= max_gap_seconds and isinstance(row.get("bbox"), list):
            candidates.append((offset, -float(row.get("track_confidence") or 0.0), row))
    return min(candidates, key=lambda value: (value[0], value[1]))[2] if candidates else None


def _normalized_bbox(
    value: Any, shape: Sequence[int], *, padding_ratio: float
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    height, width = int(shape[0]), int(shape[1])
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    x1 = max(0.0, math.floor(x1 - pad_x))
    y1 = max(0.0, math.floor(y1 - pad_y))
    x2 = min(float(width), math.ceil(x2 + pad_x))
    y2 = min(float(height), math.ceil(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _encode_jpeg(image: np.ndarray, *, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise CameraContextReviewError("JPEG encoding failed")
    return encoded.tobytes()


def _write_frame_artifact(
    *,
    stage_root: Path,
    final_root: Path,
    relative_path: Path,
    payload: bytes,
    ref_type: str,
    ref_id: str,
) -> dict[str, Any]:
    stage_path = stage_root / relative_path
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_bytes(payload)
    final_path = (final_root / relative_path).resolve(strict=False)
    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "artifact_path": final_path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _null_ref(ref_type: str, ref_id: str) -> dict[str, Any]:
    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "artifact_path": None,
        "sha256": None,
    }


def _load_development_index(
    path: Path, *, expected_counts: Mapping[str, int]
) -> list[dict[str, Any]]:
    rows = _load_jsonl(path, "development index")
    counts = Counter(str(row.get("batch_id")) for row in rows)
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise CameraContextReviewError("development index batch counts drifted")
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if row.get("schema_version") != "wandering-camera-development-index-v1":
            raise CameraContextReviewError("development index schema drifted")
        source_video_id = _nonempty(row.get("source_video_id"), "source_video_id")
        if source_video_id in seen:
            raise CameraContextReviewError("development source video ID is duplicated")
        seen.add(source_video_id)
        artifacts = _mapping(row.get("artifacts"), "development artifacts")
        if set(artifacts) != {"video", "tracking", "media_sidecar", "truth", "cvat_xml"}:
            raise CameraContextReviewError("development artifact set drifted")
        resolved_artifacts: dict[str, Any] = {}
        for name, descriptor in artifacts.items():
            path_value = Path(_nonempty(descriptor.get("path"), f"{name}.path")).resolve(strict=True)
            expected_sha = _digest(descriptor.get("sha256"), f"{name}.sha256")
            if _sha256_file(path_value) != expected_sha:
                raise CameraContextReviewError(f"development {name} hash drifted")
            resolved_artifacts[name] = {**dict(descriptor), "path": path_value}
        row["artifacts"] = resolved_artifacts
        resolved.append(row)
    return resolved


def _context_source_refs(
    *,
    episode: Mapping[str, Any],
    index_row: Mapping[str, Any],
    episode_results_path: Path,
    episode_results_sha: str,
    development_index_path: Path,
    development_index_sha: str,
) -> list[dict[str, Any]]:
    refs: list[Mapping[str, Any]] = [
        {
            "ref_type": "episode_results",
            "ref_id": "w5d02-episode-results",
            "artifact_path": episode_results_path.as_posix(),
            "sha256": episode_results_sha,
        },
        {
            "ref_type": "development_index",
            "ref_id": "w5d00-development-index",
            "artifact_path": development_index_path.as_posix(),
            "sha256": development_index_sha,
        },
    ]
    refs.extend(episode.get("source_refs", []))
    for name in ("video", "tracking", "media_sidecar"):
        descriptor = index_row["artifacts"][name]
        refs.append(
            {
                "ref_type": name,
                "ref_id": str(index_row["source_video_id"]),
                "artifact_path": Path(descriptor["path"]).as_posix(),
                "sha256": str(descriptor["sha256"]),
            }
        )
    return [dict(ref) for ref in _deduplicate_refs(refs)]


def _build_run_summary(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    episodes: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    skipped: Mapping[str, int],
    provider_mode: str,
    provider: ContextProvider,
    config: Mapping[str, Any],
    config_sha256: str,
    policy_sha256: str,
    input_files: Mapping[str, Path],
) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in context_rows)
    errors = Counter(
        str(row["error_code"])
        for row in context_rows
        if row.get("error_code") is not None
    )
    ready_frames = sum(
        frame["status"] == "ready" for row in context_rows for frame in row["frame_refs"]
    )
    unavailable_frames = 3 * len(context_rows) - ready_frames
    provider_invocations = sum(
        all(frame["status"] == "ready" for frame in row["frame_refs"])
        and provider_mode != "disabled"
        for row in context_rows
    )
    return {
        "schema_version": "wandering-handoff-run-summary-v1",
        "module": "mental_health",
        "run_id": run_id,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "person_id": _single_or_none(row["person_id"] for row in episodes),
        "session_id": None,
        "source_video_id": None,
        "status": status,
        "input_counts": {
            "episode_results": len(episodes),
            "eligible_episodes": len(eligible),
            "source_videos": len({row["source_video_id"] for row in episodes}),
        },
        "output_counts": {
            "context_reviews": len(context_rows),
            "ready": counts.get("ready", 0),
            "uncertain": counts.get("uncertain", 0),
            "unavailable": counts.get("unavailable", 0),
            "error": counts.get("error", 0),
            "ready_frames": ready_frames,
            "unavailable_frames": unavailable_frames,
            "provider_invocations": provider_invocations,
            "provider_invocation_skipped": len(context_rows) - provider_invocations,
            "noneligible_episode_skipped": int(skipped.get("not_context_eligible", 0)),
        },
        "failure_counts": dict(sorted(errors.items())),
        "degraded_components": (
            [f"context_provider_{provider_mode.replace('-', '_')}"]
            if provider_mode != "openai-compatible" or errors
            else []
        ),
        "quality_flags": _unique_strings(
            [
                "context_only_does_not_modify_episode",
                "development_only",
                f"home_input_status_{config['home_input_status']}",
                f"home_annotation_status_{config['home_annotation_status']}",
                f"home_smoke_status_{config['home_smoke_status']}",
                f"validation_scope_{config['validation_scope']}",
            ]
        ),
        "identity": {
            "model_id": provider.model_id,
            "model_sha256": None,
            "config_id": config["context_review_id"],
            "config_sha256": config_sha256,
            "policy_id": config["trigger"]["policy_id"],
            "policy_sha256": policy_sha256,
        },
        "source_refs": [
            {
                "ref_type": "episode_results",
                "ref_id": "w5d02-episode-results",
                "artifact_path": input_files["episode_results"].as_posix(),
                "sha256": config["inputs"]["episode_results"]["sha256"],
            },
            {
                "ref_type": "development_index",
                "ref_id": "w5d00-development-index",
                "artifact_path": input_files["development_index"].as_posix(),
                "sha256": config["inputs"]["development_index"]["sha256"],
            },
        ],
    }


def _build_partial_manifest(
    *,
    run_id: str,
    status: str,
    config: Mapping[str, Any],
    config_sha256: str,
    policy_sha256: str,
    provider_mode: str,
    context_rows: Sequence[Mapping[str, Any]],
    stage: Path,
    final_root: Path,
    payloads: Mapping[str, bytes],
    input_files: Mapping[str, Path],
) -> dict[str, Any]:
    descriptors = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "record_count": len(context_rows) if name == "context_reviews.jsonl" else 1 if name == "run_summary.json" else None,
            "schema_version": CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION if name == "context_reviews.jsonl" else "wandering-handoff-run-summary-v1" if name == "run_summary.json" else None,
        }
        for name, payload in payloads.items()
    }
    frame_files = sorted((stage / "frame_artifacts").rglob("*.jpg"))
    return {
        "schema_version": "wandering-camera-context-core-handoff-manifest-partial-v1",
        "module": "mental_health",
        "stage": "W5D-03A",
        "handoff_id": run_id,
        "status": status,
        "home_input_status": config["home_input_status"],
        "home_annotation_status": config["home_annotation_status"],
        "home_smoke_status": config["home_smoke_status"],
        "validation_scope": config["validation_scope"],
        "provider_mode": provider_mode,
        "algorithm_event_emitted": False,
        "episode_results_passthrough": {
            "artifact_path": input_files["episode_results"].as_posix(),
            "sha256": config["inputs"]["episode_results"]["sha256"],
            "mutated": False,
        },
        "context_review_schema": {
            "schema_version": CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
            "artifact_path": input_files["context_schema"].as_posix(),
            "sha256": config["inputs"]["context_schema"]["sha256"],
        },
        "identity": {
            "config_id": config["context_review_id"],
            "config_sha256": config_sha256,
            "policy_id": config["trigger"]["policy_id"],
            "policy_sha256": policy_sha256,
        },
        "artifacts": descriptors,
        "frame_artifacts": {
            "artifact_root": (final_root / "frame_artifacts").as_posix(),
            "file_count": len(frame_files),
            "scene_count": sum(path.name.endswith("-scene.jpg") for path in frame_files),
            "person_crop_count": sum(path.name.endswith("-crop.jpg") for path in frame_files),
        },
        "pending_files": [
            "daily_reports.jsonl",
            "baseline_profiles.jsonl",
            "baseline_deviations.jsonl",
            "handoff_manifest.json",
        ],
        "quality_flags": [
            "development_only",
            "context_only_does_not_modify_episode",
            "home_smoke_pending",
        ],
    }


def _context_run_status(
    rows: Sequence[Mapping[str, Any]], *, provider_mode: str
) -> str:
    if any(row["status"] == "error" for row in rows):
        return "error"
    if any(row["status"] in {"uncertain", "unavailable"} for row in rows):
        return "uncertain"
    if provider_mode in {"fake", "disabled"}:
        return "uncertain"
    return "ready"


def _readme_text(
    config: Mapping[str, Any], summary: Mapping[str, Any], *, provider_mode: str
) -> str:
    counts = summary["output_counts"]
    if provider_mode == "fake":
        provider_statement = (
            "Deterministic fake output verifies the offline ready-result contract; "
            "it is not a context prediction or accuracy result."
        )
    elif provider_mode == "disabled":
        provider_statement = (
            "Disabled-provider output verifies unknown/unavailable degradation "
            "without blocking the upstream or downstream pipeline."
        )
    else:
        provider_statement = (
            "OpenAI-compatible output records only the parsed provider result; "
            "raw provider responses are not included in the handoff."
        )
    return "\n".join(
        [
            "# W5D-03A three-frame context core",
            "",
            f"status: `{summary['status']}`",
            f"validation scope: `{config['validation_scope']}`",
            f"provider mode: `{provider_mode}`",
            f"eligible/context rows: `{summary['input_counts']['eligible_episodes']}/{counts['context_reviews']}`",
            f"ready/unavailable/error: `{counts['ready']}/{counts['unavailable']}/{counts['error']}`",
            f"ready/unavailable frames: `{counts['ready_frames']}/{counts['unavailable_frames']}`",
            f"home input: `{config['home_input_status']}`",
            f"home smoke: `{config['home_smoke_status']}`",
            "",
            "Each eligible W5D-02 episode has exactly one context row. The context layer preserves shape, boundary, QC, truth, and threshold.",
            provider_statement,
            "This is B01+B02 development evidence, not home-scene validation. AlgorithmEvent is not emitted.",
            "",
        ]
    )


def _verification_text(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    policy_sha256: str,
    summary: Mapping[str, Any],
    context_rows: Sequence[Mapping[str, Any]],
    skipped: Mapping[str, int],
    provider_mode: str,
) -> str:
    payload = {
        "stage": "W5D-03A",
        "status": summary["status"],
        "validation_scope": config["validation_scope"],
        "home_input_status": config["home_input_status"],
        "home_annotation_status": config["home_annotation_status"],
        "home_smoke_status": config["home_smoke_status"],
        "provider_mode": provider_mode,
        "config_sha256": config_sha256,
        "policy_sha256": policy_sha256,
        "input_counts": summary["input_counts"],
        "output_counts": summary["output_counts"],
        "failure_counts": summary["failure_counts"],
        "skipped_counts": dict(skipped),
        "one_trigger_one_row": summary["input_counts"]["eligible_episodes"] == len(context_rows),
        "unique_episode_ids": len({row["episode_id"] for row in context_rows}) == len(context_rows),
        "three_frame_refs_per_row": all(len(row["frame_refs"]) == 3 for row in context_rows),
        "episode_results_mutated": False,
        "algorithm_event_emitted": False,
        "claims": {
            "context_accuracy_reported": False,
            "home_validated": False,
            "cross_scene_generalization_claimed": False,
        },
    }
    return "# W5D-03A verification\n\n```json\n" + json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n```\n"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraContextReviewError(f"cannot read {role}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise CameraContextReviewError(f"{role} must contain JSON objects")
    return rows


def _verify_descriptor(root: Path, descriptor: Mapping[str, Any], role: str) -> Path:
    path = Path(_nonempty(descriptor.get("path"), f"{role}.path"))
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    expected = _digest(descriptor.get("sha256"), f"{role}.sha256")
    if _sha256_file(path) != expected:
        raise CameraContextReviewError(f"{role} hash drifted")
    return path


def _descriptor_syntax(value: Mapping[str, Any], role: str) -> None:
    _nonempty(value.get("path"), f"{role}.path")
    _digest(value.get("sha256"), f"{role}.sha256")
    _nonempty(value.get("schema_version"), f"{role}.schema_version")


def _validate_identity(value: Any, role: str) -> None:
    identity = _mapping(value, role)
    expected = {
        "model_id",
        "model_sha256",
        "config_id",
        "config_sha256",
        "policy_id",
        "policy_sha256",
    }
    if set(identity) != expected:
        raise CameraContextReviewError(f"{role} fields drifted")
    _nonempty(identity["config_id"], f"{role}.config_id")
    _digest(identity["config_sha256"], f"{role}.config_sha256")
    _nonempty(identity["policy_id"], f"{role}.policy_id")
    if identity["model_sha256"] is not None:
        _digest(identity["model_sha256"], f"{role}.model_sha256")
    if identity["policy_sha256"] is not None:
        _digest(identity["policy_sha256"], f"{role}.policy_sha256")


def _validate_source_ref(value: Any) -> None:
    ref = _mapping(value, "source_ref")
    if set(ref) != {"ref_type", "ref_id", "artifact_path", "sha256"}:
        raise CameraContextReviewError("source ref fields drifted")
    _nonempty(ref["ref_type"], "source_ref.ref_type")
    _nonempty(ref["ref_id"], "source_ref.ref_id")
    if ref["artifact_path"] is not None and not isinstance(ref["artifact_path"], str):
        raise CameraContextReviewError("source ref artifact path is invalid")
    if ref["sha256"] is not None:
        _digest(ref["sha256"], "source_ref.sha256")


def _deduplicate_refs(refs: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[Mapping[str, Any]] = []
    for ref in refs:
        key = (ref.get("ref_type"), ref.get("ref_id"), ref.get("artifact_path"), ref.get("sha256"))
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraContextReviewError(f"{role} must be a mapping")
    return value


def _nonempty(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraContextReviewError(f"{role} must be a non-empty string")
    return value.strip()


def _digest(value: Any, role: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise CameraContextReviewError(f"{role} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CameraContextReviewError(f"{role} must be a SHA-256 digest") from exc
    if value.lower() != value:
        raise CameraContextReviewError(f"{role} must use lowercase hex")
    return value


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_or_none(values: Iterable[Any]) -> str | None:
    unique = {str(value) for value in values if value is not None}
    return next(iter(unique)) if len(unique) == 1 else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
