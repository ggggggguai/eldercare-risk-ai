from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path = Path("models/yolov8n-pose.pt")
    gait_model_path: Path | None = None
    gait_model_device: str = "auto"
    gait_model_window_frames: int = 16
    sit_stand_runtime_mode: str = "rule_baseline"
    sit_stand_model_path: Path | None = None
    sit_stand_model_device: str = "cpu"
    sit_stand_model_batch_size: int = 128
    fall_event_runtime_mode: str = "shadow"
    fall_event_shadow_checkpoint_paths: tuple[Path, ...] = ()
    fall_event_shadow_device: str = "cpu"
    fall_event_shadow_threshold: float = 0.5
    api_token: str = "change-me"
    callback_token: str = "change-me"
    baseline_history_path: Path | None = None
    max_inference_fps: float = 10.0
    pose_inference_size: int = 640
    pose_window_sec: float = 10.0
    analysis_interval_sec: float = 0.5
    fusion_interval_sec: float = 2.0
    primary_lost_timeout_sec: float = 2.0
    event_cooldown_sec: float = 30.0
    callback_timeout_sec: float = 5.0
    callback_retry_delays_sec: tuple[float, ...] = (0.5, 1.0, 2.0)
    outbox_capacity: int = 32
    outbox_drain_timeout_sec: float = 3.0
    session_stop_timeout_sec: float = 5.0
    frame_queue_capacity: int = 2
    stream_reader_backend: str = "ffmpeg"
    ffmpeg_scale_width: int = 640
    stream_open_timeout_ms: int = 5000
    stream_read_timeout_ms: int = 5000
    reconnect_attempts: int = 3
    reconnect_delay_sec: float = 1.0
    reconnect_stable_after_sec: float = 30.0
    reconnect_stable_after_frames: int = 120
    scene_risk_scores: Mapping[str, float] = field(default_factory=dict)
    branch_quality: Mapping[str, Any] = field(default_factory=dict)
    fall_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.baseline_history_path is not None and not isinstance(self.baseline_history_path, Path):
            object.__setattr__(self, "baseline_history_path", Path(self.baseline_history_path))
        if self.gait_model_path is not None and not isinstance(self.gait_model_path, Path):
            object.__setattr__(self, "gait_model_path", Path(self.gait_model_path))
        if self.sit_stand_model_path is not None and not isinstance(self.sit_stand_model_path, Path):
            object.__setattr__(self, "sit_stand_model_path", Path(self.sit_stand_model_path))
        object.__setattr__(
            self,
            "fall_event_shadow_checkpoint_paths",
            tuple(Path(value) for value in self.fall_event_shadow_checkpoint_paths),
        )
        if self.gait_model_window_frames < 2:
            raise ValueError("gait_model_window_frames must be at least 2")
        if self.sit_stand_runtime_mode not in {"rule_baseline", "experimental_tcn"}:
            raise ValueError(
                "sit_stand_runtime_mode must be rule_baseline or experimental_tcn"
            )
        if (
            self.sit_stand_runtime_mode == "experimental_tcn"
            and self.sit_stand_model_path is None
        ):
            raise ValueError("sit_stand_model_path is required for experimental_tcn")
        if self.sit_stand_model_batch_size < 1:
            raise ValueError("sit_stand_model_batch_size must be positive")
        if self.fall_event_runtime_mode not in {"shadow", "experimental_tcn"}:
            raise ValueError(
                "fall_event_runtime_mode must be shadow or experimental_tcn"
            )
        if (
            self.fall_event_runtime_mode == "experimental_tcn"
            and not self.fall_event_shadow_checkpoint_paths
        ):
            raise ValueError(
                "fall_event_shadow_checkpoint_paths are required for experimental_tcn"
            )
        if self.frame_queue_capacity < 1:
            raise ValueError("frame_queue_capacity must be at least 1")
        if self.pose_inference_size < 32:
            raise ValueError("pose_inference_size must be at least 32")
        if self.outbox_capacity < 1:
            raise ValueError("outbox_capacity must be at least 1")
        if self.outbox_drain_timeout_sec < 0:
            raise ValueError("outbox_drain_timeout_sec must be non-negative")
        if self.session_stop_timeout_sec <= 0:
            raise ValueError("session_stop_timeout_sec must be positive")
        if self.stream_reader_backend not in {"opencv", "ffmpeg"}:
            raise ValueError("stream_reader_backend must be opencv or ffmpeg")
        if self.ffmpeg_scale_width < 2:
            raise ValueError("ffmpeg_scale_width must be at least 2")
        if self.reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        if self.reconnect_delay_sec < 0:
            raise ValueError("reconnect_delay_sec must be non-negative")
        if self.reconnect_stable_after_sec <= 0:
            raise ValueError("reconnect_stable_after_sec must be positive")
        if self.reconnect_stable_after_frames < 1:
            raise ValueError("reconnect_stable_after_frames must be at least 1")
        if not 0.0 < self.fall_event_shadow_threshold < 1.0:
            raise ValueError("fall_event_shadow_threshold must be within (0, 1)")
        if self.fall_event_shadow_device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("fall_event_shadow_device must be cpu, cuda, mps or auto")

    @classmethod
    def load(cls, path: Path | None = None, environ: Mapping[str, str] | None = None) -> "ServiceSettings":
        env = os.environ if environ is None else environ
        config_path = path or Path(env.get("FALL_RISK_SERVICE_CONFIG", "configs/modules/fall_risk_service.yaml"))
        raw: dict[str, Any] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("fall risk service config must be a mapping")
            raw.update(loaded)

        overrides: dict[str, tuple[str, Any]] = {
            "MODEL_PATH": ("model_path", Path),
            "GAIT_MODEL_PATH": ("gait_model_path", Path),
            "GAIT_MODEL_DEVICE": ("gait_model_device", str),
            "GAIT_MODEL_WINDOW_FRAMES": ("gait_model_window_frames", int),
            "SIT_STAND_RUNTIME_MODE": ("sit_stand_runtime_mode", str),
            "SIT_STAND_MODEL_PATH": ("sit_stand_model_path", Path),
            "SIT_STAND_MODEL_DEVICE": ("sit_stand_model_device", str),
            "SIT_STAND_MODEL_BATCH_SIZE": ("sit_stand_model_batch_size", int),
            "FALL_EVENT_RUNTIME_MODE": ("fall_event_runtime_mode", str),
            "ALGORITHM_API_TOKEN": ("api_token", str),
            "CALLBACK_TOKEN": ("callback_token", str),
            "BASELINE_HISTORY_PATH": ("baseline_history_path", Path),
            "MAX_INFERENCE_FPS": ("max_inference_fps", float),
            "POSE_INFERENCE_SIZE": ("pose_inference_size", int),
            "POSE_WINDOW_SEC": ("pose_window_sec", float),
            "ANALYSIS_INTERVAL_SEC": ("analysis_interval_sec", float),
            "FUSION_INTERVAL_SEC": ("fusion_interval_sec", float),
            "PRIMARY_LOST_TIMEOUT_SEC": ("primary_lost_timeout_sec", float),
            "EVENT_COOLDOWN_SEC": ("event_cooldown_sec", float),
            "CALLBACK_TIMEOUT_SEC": ("callback_timeout_sec", float),
            "OUTBOX_CAPACITY": ("outbox_capacity", int),
            "OUTBOX_DRAIN_TIMEOUT_SEC": ("outbox_drain_timeout_sec", float),
            "SESSION_STOP_TIMEOUT_SEC": ("session_stop_timeout_sec", float),
            "FRAME_QUEUE_CAPACITY": ("frame_queue_capacity", int),
            "STREAM_READER_BACKEND": ("stream_reader_backend", str),
            "FFMPEG_SCALE_WIDTH": ("ffmpeg_scale_width", int),
            "RECONNECT_ATTEMPTS": ("reconnect_attempts", int),
            "RECONNECT_DELAY_SEC": ("reconnect_delay_sec", float),
            "RECONNECT_STABLE_AFTER_SEC": ("reconnect_stable_after_sec", float),
            "RECONNECT_STABLE_AFTER_FRAMES": ("reconnect_stable_after_frames", int),
        }
        for env_name, (field_name, converter) in overrides.items():
            if env_name in env:
                value = env[env_name]
                if field_name in {"gait_model_path", "baseline_history_path"} and not value.strip():
                    raw[field_name] = None
                else:
                    raw[field_name] = converter(value)
        if "model_path" in raw:
            raw["model_path"] = Path(raw["model_path"])
        if raw.get("baseline_history_path"):
            raw["baseline_history_path"] = Path(raw["baseline_history_path"])
        if raw.get("gait_model_path"):
            raw["gait_model_path"] = Path(raw["gait_model_path"])
        if raw.get("sit_stand_model_path"):
            raw["sit_stand_model_path"] = Path(raw["sit_stand_model_path"])
        if raw.get("fall_event_shadow_checkpoint_paths"):
            raw["fall_event_shadow_checkpoint_paths"] = tuple(
                Path(value) for value in raw["fall_event_shadow_checkpoint_paths"]
            )
        if "callback_retry_delays_sec" in raw:
            raw["callback_retry_delays_sec"] = tuple(float(value) for value in raw["callback_retry_delays_sec"])
        return cls(**raw)
