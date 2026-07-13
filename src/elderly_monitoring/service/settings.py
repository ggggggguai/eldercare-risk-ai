from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path = Path("yolov8n-pose.pt")
    api_token: str = "change-me"
    callback_token: str = "change-me"
    baseline_history_path: Path | None = None
    max_inference_fps: float = 8.0
    pose_window_sec: float = 10.0
    analysis_interval_sec: float = 0.5
    fusion_interval_sec: float = 2.0
    primary_lost_timeout_sec: float = 2.0
    event_cooldown_sec: float = 30.0
    callback_timeout_sec: float = 5.0
    callback_retry_delays_sec: tuple[float, ...] = (0.5, 1.0, 2.0)
    stream_open_timeout_ms: int = 5000
    stream_read_timeout_ms: int = 5000
    reconnect_attempts: int = 3
    reconnect_delay_sec: float = 1.0
    scene_risk_scores: Mapping[str, float] = field(default_factory=dict)
    fall_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.baseline_history_path is not None and not isinstance(self.baseline_history_path, Path):
            object.__setattr__(self, "baseline_history_path", Path(self.baseline_history_path))

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
            "ALGORITHM_API_TOKEN": ("api_token", str),
            "CALLBACK_TOKEN": ("callback_token", str),
            "BASELINE_HISTORY_PATH": ("baseline_history_path", Path),
            "MAX_INFERENCE_FPS": ("max_inference_fps", float),
            "POSE_WINDOW_SEC": ("pose_window_sec", float),
            "ANALYSIS_INTERVAL_SEC": ("analysis_interval_sec", float),
            "FUSION_INTERVAL_SEC": ("fusion_interval_sec", float),
            "PRIMARY_LOST_TIMEOUT_SEC": ("primary_lost_timeout_sec", float),
            "EVENT_COOLDOWN_SEC": ("event_cooldown_sec", float),
            "RECONNECT_ATTEMPTS": ("reconnect_attempts", int),
            "RECONNECT_DELAY_SEC": ("reconnect_delay_sec", float),
        }
        for env_name, (field_name, converter) in overrides.items():
            if env_name in env:
                raw[field_name] = converter(env[env_name])
        if "model_path" in raw:
            raw["model_path"] = Path(raw["model_path"])
        if raw.get("baseline_history_path"):
            raw["baseline_history_path"] = Path(raw["baseline_history_path"])
        if "callback_retry_delays_sec" in raw:
            raw["callback_retry_delays_sec"] = tuple(float(value) for value in raw["callback_retry_delays_sec"])
        return cls(**raw)
