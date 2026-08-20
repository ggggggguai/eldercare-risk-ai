"""Runtime settings for the local Paraformer ASR package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ASRSettings:
    model_root: Path = PROJECT_ROOT / "models" / "asr" / "asr-paraformer-zh-v1.0"
    device: str = "auto"
    target_sample_rate: int = 16000
    min_audio_seconds: float = 3.0
    max_audio_seconds: float = 60.0
    max_source_bytes: int = 50 * 1024 * 1024
    url_connect_timeout_seconds: float = 5.0
    url_total_timeout_seconds: float = 15.0
    url_max_redirects: int = 2
    media_decode_timeout_seconds: float = 15.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    native_assets_manifest: Path = PROJECT_ROOT / "configs" / "runtime" / "asr-native-assets.json"
    batch_size_seconds: int = 60
    merge_length_seconds: int = 15

    def __post_init__(self) -> None:
        if not isinstance(self.model_root, Path):
            object.__setattr__(self, "model_root", Path(self.model_root))
        if not isinstance(self.native_assets_manifest, Path):
            object.__setattr__(self, "native_assets_manifest", Path(self.native_assets_manifest))
        if self.device not in {"auto", "cpu", "cuda:0"}:
            raise ValueError("ASR device must be auto, cpu, or cuda:0")
        if (
            self.target_sample_rate <= 0
            or self.min_audio_seconds <= 0
            or self.max_audio_seconds < self.min_audio_seconds
        ):
            raise ValueError("ASR sample rate and duration limits must be positive")
        if (
            self.max_source_bytes <= 0
            or self.url_connect_timeout_seconds <= 0
            or self.url_total_timeout_seconds < self.url_connect_timeout_seconds
            or self.url_max_redirects < 0
            or self.media_decode_timeout_seconds <= 0
        ):
            raise ValueError("ASR source limits must be positive")

    @classmethod
    def load(cls, environ: Mapping[str, str] | None = None) -> "ASRSettings":
        env = os.environ if environ is None else environ
        values = {}
        converters = {
            "ASR_MODEL_ROOT": ("model_root", Path),
            "ASR_DEVICE": ("device", str),
            "ASR_TARGET_SAMPLE_RATE": ("target_sample_rate", int),
            "ASR_MIN_AUDIO_SECONDS": ("min_audio_seconds", float),
            "ASR_MAX_AUDIO_SECONDS": ("max_audio_seconds", float),
            "ASR_MAX_SOURCE_BYTES": ("max_source_bytes", int),
            "ASR_URL_CONNECT_TIMEOUT_SECONDS": ("url_connect_timeout_seconds", float),
            "ASR_URL_TOTAL_TIMEOUT_SECONDS": ("url_total_timeout_seconds", float),
            "ASR_URL_MAX_REDIRECTS": ("url_max_redirects", int),
            "ASR_MEDIA_DECODE_TIMEOUT_SECONDS": ("media_decode_timeout_seconds", float),
            "ASR_FFMPEG_PATH": ("ffmpeg_path", str),
            "ASR_FFPROBE_PATH": ("ffprobe_path", str),
            "ASR_NATIVE_ASSETS_MANIFEST": ("native_assets_manifest", Path),
            "ASR_BATCH_SIZE_SECONDS": ("batch_size_seconds", int),
            "ASR_MERGE_LENGTH_SECONDS": ("merge_length_seconds", int),
        }
        for env_name, (field_name, converter) in converters.items():
            if env_name in env:
                values[field_name] = converter(env[env_name])
        return cls(**values)
