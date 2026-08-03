"""Decode V3.3 audio inputs into bounded 16 kHz mono float32 waveforms."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from elderly_monitoring.modules.asr.schemas import AudioInput
from elderly_monitoring.modules.asr.settings import ASRSettings


class AudioDecodeError(ValueError):
    """Raised when an input violates the V3.3 media contract or cannot be decoded."""


@dataclass(frozen=True)
class DecodedAudio:
    samples: Any
    sample_rate: int
    duration_ms: int
    source_format: str
    original_sample_rate: int
    original_channels: int


class _LimitedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, max_redirects: int) -> None:
        super().__init__()
        self.max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirects = int(getattr(req, "_asr_redirects", 0))
        if redirects >= self.max_redirects:
            raise AudioDecodeError("audio URL exceeded the redirect limit")
        if urlparse(newurl).scheme not in {"http", "https"}:
            raise AudioDecodeError("audio URL redirect must use http or https")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            setattr(redirected, "_asr_redirects", redirects + 1)
        return redirected


def decode_audio_input(audio_input: AudioInput, settings: ASRSettings) -> DecodedAudio:
    if audio_input.source_type == "base64":
        if len(audio_input.source) > ((settings.max_source_bytes + 2) // 3) * 4 + 8:
            raise AudioDecodeError("audio source exceeds the configured byte limit")
        try:
            payload = base64.b64decode(audio_input.source, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AudioDecodeError("audio base64 is invalid") from exc
    else:
        payload = _download(audio_input.source, settings)
    return _decode_payload(
        payload,
        settings,
        declared_format=audio_input.format,
        declared_sample_rate=audio_input.sample_rate,
        declared_channels=audio_input.channels,
    )


def decode_audio_path(path: str | Path, settings: ASRSettings) -> DecodedAudio:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise AudioDecodeError(f"audio file is not available: {audio_path}")
    if audio_path.stat().st_size > settings.max_source_bytes:
        raise AudioDecodeError("audio source exceeds the configured byte limit")
    expected = {".wav": "wav", ".mp3": "mp3", ".m4a": "m4a"}.get(
        audio_path.suffix.lower()
    )
    return _decode_payload(
        audio_path.read_bytes(),
        settings,
        declared_format=expected,
    )


def decode_audio_bytes(payload: bytes, settings: ASRSettings) -> DecodedAudio:
    return _decode_payload(payload, settings)


def _download(source: str, settings: ASRSettings) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AudioDecodeError("audio URL must use http or https")
    request = Request(source, headers={"User-Agent": "eldercare-risk-asr/1.0"})
    opener = build_opener(_LimitedRedirectHandler(settings.url_max_redirects))
    started = time.monotonic()
    payload = bytearray()
    try:
        with opener.open(request, timeout=settings.url_connect_timeout_seconds) as response:
            status_code = int(getattr(response, "status", 200))
            if not 200 <= status_code < 300:
                raise AudioDecodeError(f"audio URL returned HTTP {status_code}")
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > settings.max_source_bytes:
                raise AudioDecodeError("audio source exceeds the configured byte limit")
            while True:
                if time.monotonic() - started > settings.url_total_timeout_seconds:
                    raise AudioDecodeError("audio URL exceeded the total read timeout")
                chunk = response.read(min(1024 * 1024, settings.max_source_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > settings.max_source_bytes:
                    raise AudioDecodeError("audio source exceeds the configured byte limit")
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError("audio URL could not be read") from exc
    return bytes(payload)


def _decode_payload(
    payload: bytes,
    settings: ASRSettings,
    *,
    declared_format: str | None = None,
    declared_sample_rate: int | None = None,
    declared_channels: int | None = None,
) -> DecodedAudio:
    if not payload:
        raise AudioDecodeError("audio source is empty")
    if len(payload) > settings.max_source_bytes:
        raise AudioDecodeError("audio source exceeds the configured byte limit")
    try:
        return _decode_with_soundfile(
            payload,
            settings,
            declared_format=declared_format,
            declared_sample_rate=declared_sample_rate,
            declared_channels=declared_channels,
        )
    except AudioDecodeError as soundfile_error:
        try:
            return _decode_with_ffmpeg(
                payload,
                settings,
                declared_format=declared_format,
                declared_sample_rate=declared_sample_rate,
                declared_channels=declared_channels,
            )
        except AudioDecodeError as ffmpeg_error:
            if "declared audio" in str(soundfile_error):
                raise soundfile_error
            raise ffmpeg_error from soundfile_error


def _decode_with_soundfile(
    payload: bytes,
    settings: ASRSettings,
    *,
    declared_format: str | None,
    declared_sample_rate: int | None,
    declared_channels: int | None,
) -> DecodedAudio:
    try:
        import numpy as np
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(payload)) as audio_file:
            original_sample_rate = int(audio_file.samplerate)
            original_channels = int(audio_file.channels)
            source_format = _soundfile_format(audio_file.format)
            samples = audio_file.read(dtype="float32", always_2d=True)
    except ModuleNotFoundError:
        return _decode_pcm_wav(
            payload,
            settings,
            declared_format=declared_format,
            declared_sample_rate=declared_sample_rate,
            declared_channels=declared_channels,
        )
    except Exception as exc:
        raise AudioDecodeError("SoundFile could not decode the audio container") from exc
    _validate_metadata(
        source_format,
        original_sample_rate,
        original_channels,
        declared_format,
        declared_sample_rate,
        declared_channels,
    )
    if samples.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    samples = np.mean(samples, axis=1, dtype=np.float32)
    duration_seconds = float(len(samples)) / float(original_sample_rate)
    _validate_duration(duration_seconds, settings)
    if original_sample_rate != settings.target_sample_rate:
        try:
            import torch
            import torchaudio.functional as audio_functional

            samples = audio_functional.resample(
                torch.from_numpy(samples),
                orig_freq=original_sample_rate,
                new_freq=settings.target_sample_rate,
            ).numpy()
        except Exception as exc:
            raise AudioDecodeError("audio resampling failed") from exc
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    return DecodedAudio(
        samples=samples,
        sample_rate=settings.target_sample_rate,
        duration_ms=int(round(duration_seconds * 1000)),
        source_format=source_format,
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
    )


def _decode_with_ffmpeg(
    payload: bytes,
    settings: ASRSettings,
    *,
    declared_format: str | None,
    declared_sample_rate: int | None,
    declared_channels: int | None,
) -> DecodedAudio:
    suffix = f".{declared_format}" if declared_format else ".audio"
    with tempfile.TemporaryDirectory(prefix="eldercare-asr-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(payload)
        metadata = _probe_audio(input_path, settings)
        source_format = _ffprobe_format(metadata["format_name"], metadata["codec_name"])
        original_sample_rate = int(metadata["sample_rate"])
        original_channels = int(metadata["channels"])
        _validate_metadata(
            source_format,
            original_sample_rate,
            original_channels,
            declared_format,
            declared_sample_rate,
            declared_channels,
        )
        command = [
            settings.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(settings.target_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        completed = _run_media_command(command, None, settings.media_decode_timeout_seconds)
    if completed.returncode != 0 or not completed.stdout:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"FFmpeg could not decode audio: {message or 'empty output'}")
    try:
        import numpy as np

        samples = np.frombuffer(completed.stdout, dtype="<f4").copy()
    except Exception as exc:
        raise AudioDecodeError("FFmpeg output could not be parsed") from exc
    duration_seconds = float(len(samples)) / float(settings.target_sample_rate)
    _validate_duration(duration_seconds, settings)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    return DecodedAudio(
        samples=samples,
        sample_rate=settings.target_sample_rate,
        duration_ms=int(round(duration_seconds * 1000)),
        source_format=source_format,
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
    )


def _probe_audio(input_path: Path, settings: ASRSettings) -> dict[str, Any]:
    command = [
        settings.ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=format_name:stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(input_path),
    ]
    completed = _run_media_command(command, None, settings.media_decode_timeout_seconds)
    if completed.returncode != 0:
        raise AudioDecodeError("FFprobe could not inspect the audio container")
    try:
        raw = json.loads(completed.stdout.decode("utf-8"))
        stream = raw["streams"][0]
        return {
            "format_name": raw["format"]["format_name"],
            "codec_name": stream.get("codec_name", ""),
            "sample_rate": stream["sample_rate"],
            "channels": stream["channels"],
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioDecodeError("FFprobe returned incomplete audio metadata") from exc


def _run_media_command(
    command: list[str],
    payload: bytes | None,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[bytes]:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        return subprocess.run(
            command,
            input=payload,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            creationflags=creation_flags,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(f"media decoder is unavailable or timed out: {command[0]}") from exc


def _decode_pcm_wav(
    payload: bytes,
    settings: ASRSettings,
    *,
    declared_format: str | None,
    declared_sample_rate: int | None,
    declared_channels: int | None,
) -> DecodedAudio:
    try:
        import numpy as np

        with wave.open(io.BytesIO(payload), "rb") as wav:
            original_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            original_sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            pcm = wav.readframes(frame_count)
    except Exception as exc:
        raise AudioDecodeError("PCM WAV decoder could not decode the audio") from exc
    _validate_metadata(
        "wav",
        original_sample_rate,
        original_channels,
        declared_format,
        declared_sample_rate,
        declared_channels,
    )
    if not pcm:
        raise AudioDecodeError("decoded audio is empty")
    duration_seconds = frame_count / float(original_sample_rate)
    _validate_duration(duration_seconds, settings)
    try:
        samples = _pcm_bytes_to_float32(pcm, sample_width, np)
        samples = samples.reshape(-1, original_channels).mean(axis=1, dtype=np.float32)
        if original_sample_rate != settings.target_sample_rate:
            target_length = int(
                round(len(samples) * settings.target_sample_rate / original_sample_rate)
            )
            old_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
            samples = np.interp(new_positions, old_positions, samples).astype(np.float32)
    except Exception as exc:
        raise AudioDecodeError("PCM WAV conversion failed") from exc
    samples = np.clip(samples, -1.0, 1.0)
    return DecodedAudio(
        samples=samples,
        sample_rate=settings.target_sample_rate,
        duration_ms=int(round(duration_seconds * 1000)),
        source_format="wav",
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
    )


def _validate_metadata(
    actual_format: str,
    actual_sample_rate: int,
    actual_channels: int,
    declared_format: str | None,
    declared_sample_rate: int | None,
    declared_channels: int | None,
) -> None:
    if actual_format not in {"wav", "mp3", "m4a", "aac"}:
        raise AudioDecodeError(f"unsupported audio format: {actual_format}")
    if declared_format is not None and actual_format != declared_format:
        raise AudioDecodeError(
            f"declared audio format {declared_format} does not match decoded {actual_format}"
        )
    if not 8000 <= actual_sample_rate <= 48000:
        raise AudioDecodeError("decoded audio sample rate is outside 8000-48000 Hz")
    if actual_channels not in {1, 2}:
        raise AudioDecodeError("decoded audio must contain one or two channels")
    if declared_sample_rate is not None and declared_sample_rate != actual_sample_rate:
        raise AudioDecodeError("declared audio sample rate does not match decoded media")
    if declared_channels is not None and declared_channels != actual_channels:
        raise AudioDecodeError("declared audio channels do not match decoded media")


def _pcm_bytes_to_float32(pcm: bytes, sample_width: int, np: Any) -> Any:
    if sample_width == 1:
        return (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | raw[:, 1].astype(np.int32) << 8
            | raw[:, 2].astype(np.int32) << 16
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioDecodeError(f"unsupported PCM WAV sample width: {sample_width}")


def _validate_duration(duration_seconds: float, settings: ASRSettings) -> None:
    if not settings.min_audio_seconds <= duration_seconds <= settings.max_audio_seconds:
        raise AudioDecodeError(
            "audio duration must be between "
            f"{settings.min_audio_seconds:g} and {settings.max_audio_seconds:g} seconds"
        )


def _soundfile_format(format_name: str) -> str:
    normalized = format_name.upper()
    if normalized in {"WAV", "WAVEX", "RF64"}:
        return "wav"
    if normalized in {"MPEG", "MP3"}:
        return "mp3"
    return normalized.lower()


def _ffprobe_format(format_name: str, codec_name: str) -> str:
    names = {item.strip().lower() for item in format_name.split(",")}
    if "wav" in names:
        return "wav"
    if "mp3" in names:
        return "mp3"
    if "aac" in names:
        return "aac"
    if names.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
        return "m4a" if codec_name.lower() in {"aac", "alac"} else "mp4"
    return next(iter(names), "unknown")
