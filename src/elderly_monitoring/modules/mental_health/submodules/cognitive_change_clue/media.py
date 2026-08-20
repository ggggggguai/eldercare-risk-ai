"""Bounded request-scoped media loading and synchronization checks."""

from __future__ import annotations

import base64
import binascii
import json
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from elderly_monitoring.modules.asr.audio_decode import (
    AudioDecodeError,
    DecodedAudio,
    decode_audio_bytes,
)
from elderly_monitoring.modules.asr.settings import ASRSettings
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    FACE_SAMPLE_COUNT,
    sample_frame_positions,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    AudioMediaInput,
    CognitiveErrorItem,
    FaceMediaInput,
)


AUDIO_MAX_BYTES = 50 * 1024 * 1024
VIDEO_MAX_BYTES = 100 * 1024 * 1024
MIN_DURATION_MS = 3_000
MAX_DURATION_MS = 60_000
ALIGNMENT_TOLERANCE_MS = 200


@dataclass(frozen=True)
class MediaReadSettings:
    connect_timeout_seconds: float = 5.0
    total_timeout_seconds: float = 15.0
    max_redirects: int = 2
    ffprobe_path: str = "ffprobe"
    probe_timeout_seconds: float = 15.0


@dataclass(frozen=True)
class DecodedAudioMedia:
    decoded: DecodedAudio
    payload: bytes


@dataclass(frozen=True)
class DecodedFaceMedia:
    duration_ms: int
    frame_paths: tuple[Path, ...]
    actual_fps: float


class _LimitedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, max_redirects: int) -> None:
        super().__init__()
        self.max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirects = int(getattr(req, "_cognitive_redirects", 0))
        if redirects >= self.max_redirects:
            raise URLError("redirect limit exceeded")
        parsed = urlparse(newurl)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise URLError("redirect target must use HTTP or HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            setattr(redirected, "_cognitive_redirects", redirects + 1)
        return redirected


def read_media_source(
    *,
    source_type: str,
    source: str,
    max_bytes: int,
    request_id: str,
    field: str,
    settings: MediaReadSettings,
) -> bytes:
    if source_type == "base64":
        if len(source) > ((max_bytes + 2) // 3) * 4 + 8:
            raise _media_error("MEDIA_TOO_LARGE", request_id, field, "source exceeds byte limit")
        try:
            payload = base64.b64decode(source, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _media_error("MEDIA_INVALID", request_id, field, "invalid base64") from exc
        if len(payload) > max_bytes:
            raise _media_error("MEDIA_TOO_LARGE", request_id, field, "source exceeds byte limit")
        if not payload:
            raise _media_error("MEDIA_INVALID", request_id, field, "source is empty")
        return payload

    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _media_error("MEDIA_INVALID", request_id, field, "URL must use HTTP or HTTPS")
    request = Request(source, headers={"User-Agent": "eldercare-cognitive/3.3"})
    opener = build_opener(_LimitedRedirectHandler(settings.max_redirects))
    payload = bytearray()
    started = time.monotonic()
    try:
        with opener.open(request, timeout=settings.connect_timeout_seconds) as response:
            declared_size = response.headers.get("Content-Length")
            if declared_size:
                try:
                    declared_bytes = int(declared_size)
                except ValueError:
                    declared_bytes = 0
                if declared_bytes > max_bytes:
                    raise _media_error(
                        "MEDIA_TOO_LARGE", request_id, field, "source exceeds byte limit"
                    )
            while True:
                if time.monotonic() - started > settings.total_timeout_seconds:
                    raise TimeoutError("media read timeout")
                chunk = response.read(min(1024 * 1024, max_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise _media_error(
                        "MEDIA_TOO_LARGE", request_id, field, "source exceeds byte limit"
                    )
    except CognitiveAPIError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise _media_error(
            "MEDIA_UNREACHABLE", request_id, field, "URL could not be read within the limit"
        ) from exc
    if not payload:
        raise _media_error("MEDIA_INVALID", request_id, field, "source is empty")
    return bytes(payload)


def decode_audio_media(
    media: AudioMediaInput,
    *,
    request_id: str,
    read_settings: MediaReadSettings | None = None,
) -> DecodedAudioMedia:
    settings = read_settings or MediaReadSettings()
    payload = read_media_source(
        source_type=media.source_type,
        source=media.source,
        max_bytes=AUDIO_MAX_BYTES,
        request_id=request_id,
        field="audio_input.source",
        settings=settings,
    )
    decoder_settings = ASRSettings(
        min_audio_seconds=3.0,
        max_audio_seconds=60.0,
        max_source_bytes=AUDIO_MAX_BYTES,
    )
    try:
        decoded = decode_audio_bytes(
            payload,
            decoder_settings,
            declared_format=media.format,
            declared_sample_rate=media.sample_rate,
            declared_channels=media.channels,
        )
    except AudioDecodeError as exc:
        message = str(exc)
        if "duration must be between" in message:
            code = "MEDIA_TOO_LONG"
        elif "declared audio" in message or "unsupported audio format" in message:
            code = "MEDIA_INVALID"
        else:
            code = "MEDIA_DECODE_FAILED"
        raise _media_error(code, request_id, "audio_input", message) from exc
    return DecodedAudioMedia(decoded=decoded, payload=payload)


@contextmanager
def decode_face_media(
    media: FaceMediaInput,
    *,
    request_id: str,
    read_settings: MediaReadSettings | None = None,
) -> Iterator[DecodedFaceMedia]:
    settings = read_settings or MediaReadSettings()
    payload = read_media_source(
        source_type=media.source_type,
        source=media.source,
        max_bytes=VIDEO_MAX_BYTES,
        request_id=request_id,
        field="face_input.source",
        settings=settings,
    )
    with tempfile.TemporaryDirectory(prefix="cognitive-face-") as directory:
        root = Path(directory)
        video_path = root / "input.mp4"
        video_path.write_bytes(payload)
        duration_ms, probe_fps = _probe_video(
            video_path,
            request_id=request_id,
            settings=settings,
        )
        _validate_duration(duration_ms, request_id, "face_input")
        frame_paths, decoded_fps = _extract_sampled_frames(video_path, root, request_id)
        yield DecodedFaceMedia(
            duration_ms=duration_ms,
            frame_paths=frame_paths,
            actual_fps=probe_fps or decoded_fps,
        )


def validate_media_alignment(
    *,
    capture_duration_ms: int,
    audio_duration_ms: int | None,
    face_duration_ms: int | None,
    request_id: str,
) -> None:
    durations = {
        "audio_input": audio_duration_ms,
        "face_input": face_duration_ms,
    }
    for field, duration in durations.items():
        if duration is None:
            continue
        _validate_duration(duration, request_id, field)
        if abs(duration - capture_duration_ms) > ALIGNMENT_TOLERANCE_MS:
            raise _media_error(
                "MEDIA_MISALIGNED",
                request_id,
                field,
                "decoded duration differs from capture_window by more than 200 ms",
            )
    if (
        audio_duration_ms is not None
        and face_duration_ms is not None
        and abs(audio_duration_ms - face_duration_ms) > ALIGNMENT_TOLERANCE_MS
    ):
        raise _media_error(
            "MEDIA_MISALIGNED",
            request_id,
            "audio_input,face_input",
            "decoded audio/video duration difference exceeds 200 ms",
        )


def _probe_video(
    path: Path,
    *,
    request_id: str,
    settings: MediaReadSettings,
) -> tuple[int, float]:
    command = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration:stream=codec_type,duration,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=settings.probe_timeout_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "video probe failed"
        ) from exc
    if completed.returncode != 0:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "video probe failed"
        )
    try:
        probe = json.loads(completed.stdout)
        format_info = probe.get("format") or {}
        format_names = {
            value.strip().lower()
            for value in str(format_info.get("format_name") or "").split(",")
        }
        streams = [
            stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"
        ]
    except (TypeError, ValueError, AttributeError) as exc:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "invalid video probe result"
        ) from exc
    if not format_names.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
        raise _media_error(
            "MEDIA_INVALID", request_id, "face_input.format", "decoded container is not MP4"
        )
    if not streams:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "MP4 has no video stream"
        )
    raw_duration = streams[0].get("duration") or format_info.get("duration")
    try:
        duration_ms = int(round(float(raw_duration) * 1000.0))
    except (TypeError, ValueError) as exc:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "video duration is unavailable"
        ) from exc
    fps = _parse_frame_rate(streams[0].get("avg_frame_rate"))
    return duration_ms, fps


def _extract_sampled_frames(
    video_path: Path,
    root: Path,
    request_id: str,
) -> tuple[tuple[Path, ...], float]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "video decoder could not open MP4"
        )
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[tuple[int, object]] = []
        if frame_count > 0:
            for position in sample_frame_positions(frame_count, FACE_SAMPLE_COUNT):
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    frames.append((position, frame))
        else:
            position = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is not None and frame.size:
                    frames.append((position, frame))
                position += 1
            if frames:
                positions = set(sample_frame_positions(len(frames), FACE_SAMPLE_COUNT))
                frames = [item for index, item in enumerate(frames) if index in positions]
    finally:
        capture.release()
    if not frames:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "video contains no decodable frames"
        )
    paths: list[Path] = []
    for index, (_, frame) in enumerate(frames):
        target = root / f"frame_{index:03d}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            target.write_bytes(encoded.tobytes())
            paths.append(target)
    if not paths:
        raise _media_error(
            "MEDIA_DECODE_FAILED", request_id, "face_input", "sampled frames could not be encoded"
        )
    return tuple(paths), fps


def _parse_frame_rate(value: object) -> float:
    text = str(value or "")
    try:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return 0.0 if denominator_value == 0 else float(numerator) / denominator_value
    except (ValueError, TypeError):
        return 0.0


def _validate_duration(duration_ms: int, request_id: str, field: str) -> None:
    if not MIN_DURATION_MS <= int(duration_ms) <= MAX_DURATION_MS:
        raise _media_error(
            "MEDIA_TOO_LONG",
            request_id,
            field,
            "decoded duration must be between 3000 and 60000 ms",
        )


def _media_error(
    code: str,
    request_id: str,
    field: str,
    reason: str,
) -> CognitiveAPIError:
    return CognitiveAPIError(
        code=code,  # type: ignore[arg-type]
        request_id=request_id,
        errors=(CognitiveErrorItem(field=field, reason=reason),),
    )


__all__ = [
    "ALIGNMENT_TOLERANCE_MS",
    "AUDIO_MAX_BYTES",
    "VIDEO_MAX_BYTES",
    "DecodedAudioMedia",
    "DecodedFaceMedia",
    "MediaReadSettings",
    "decode_audio_media",
    "decode_face_media",
    "read_media_source",
    "validate_media_alignment",
]
