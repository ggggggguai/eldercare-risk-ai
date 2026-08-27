from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class EnvironmentSettings:
    mode: str = "disabled"
    max_age_sec: float = 3.0
    max_transport_age_sec: float = 2.0
    max_future_skew_sec: float = 0.5
    store_capacity_per_device: int = 256
    shadow_log_path: Path | None = None
    roi_window_frames: int = 3
    roi_min_usable_frames: int = 2
    roi_min_hits: int = 2
    roi_max_span_sec: float = 1.0
    camera_bindings: Mapping[str, Any] = field(default_factory=dict)
    assist_weight: float | None = None
    assist_min_behavior_anchor: float | None = None
    assist_policy_version: str | None = None
    assist_acceptance_report: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "shadow", "assist"}:
            raise ValueError("environment.mode must be disabled, shadow or assist")
        if self.max_age_sec <= 0 or self.max_transport_age_sec < 0 or self.max_future_skew_sec < 0:
            raise ValueError("environment age limits are invalid")
        if self.store_capacity_per_device < 1:
            raise ValueError("environment.store_capacity_per_device must be positive")
        if self.roi_window_frames < 1 or self.roi_min_usable_frames < 1 or self.roi_min_hits < 1:
            raise ValueError("environment ROI frame limits must be positive")
        if self.roi_min_usable_frames > self.roi_window_frames or self.roi_min_hits > self.roi_window_frames:
            raise ValueError("environment ROI limits cannot exceed roi_window_frames")
        if self.roi_max_span_sec <= 0:
            raise ValueError("environment.roi_max_span_sec must be positive")
        if not isinstance(self.camera_bindings, Mapping):
            raise ValueError("environment.camera_bindings must be a mapping")
        if self.shadow_log_path is not None and not isinstance(self.shadow_log_path, Path):
            object.__setattr__(self, "shadow_log_path", Path(self.shadow_log_path))
        if self.mode in {"shadow", "assist"} and self.shadow_log_path is None:
            raise ValueError("environment.shadow_log_path is required in shadow/assist mode")
        if self.mode == "assist":
            if self.assist_weight is None or not 0.0 <= float(self.assist_weight) <= 1.0:
                raise ValueError("environment.assist.weight is required in assist mode")
            if self.assist_min_behavior_anchor is None or not 0.0 <= float(self.assist_min_behavior_anchor) <= 1.0:
                raise ValueError("environment.assist.min_behavior_anchor is required in assist mode")
            if not self.assist_policy_version or not self.assist_acceptance_report:
                raise ValueError("environment assist policy_version and acceptance_report are required")
            if not self.camera_bindings:
                raise ValueError("environment camera_bindings are required in assist mode")
            for camera_id, binding in self.camera_bindings.items():
                if not isinstance(binding, Mapping) or not binding.get("environment_device_id"):
                    raise ValueError(f"environment binding {camera_id!r} is incomplete")
                calibration = binding.get("calibration")
                if not isinstance(calibration, Mapping):
                    raise ValueError(f"environment calibration is required for {camera_id!r}")
                severe = calibration.get("severe_low_lux")
                normal = calibration.get("normal_lux")
                if severe is None or normal is None or float(severe) >= float(normal) or not calibration.get("version"):
                    raise ValueError(f"environment lux calibration is incomplete for {camera_id!r}")
                for probe_id, roi in (binding.get("water_probe_rois") or {}).items():
                    if not isinstance(roi, Mapping) or not roi.get("version") or not roi.get("polygon"):
                        raise ValueError(f"environment ROI {probe_id!r} is incomplete for {camera_id!r}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "EnvironmentSettings":
        values = dict(raw or {})
        assist = values.pop("assist", {})
        if not isinstance(assist, Mapping):
            raise ValueError("environment.assist must be a mapping")
        values.update({
            "assist_weight": assist.get("weight", values.get("assist_weight")),
            "assist_min_behavior_anchor": assist.get(
                "min_behavior_anchor", values.get("assist_min_behavior_anchor")
            ),
            "assist_policy_version": assist.get(
                "policy_version", values.get("assist_policy_version")
            ),
            "assist_acceptance_report": assist.get(
                "acceptance_report", values.get("assist_acceptance_report")
            ),
        })
        if values.get("shadow_log_path"):
            values["shadow_log_path"] = Path(values["shadow_log_path"])
        allowed = {field.name for field in dataclass_fields(EnvironmentSettings)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


def dataclass_fields(cls: Any) -> tuple[Any, ...]:
    """Small local wrapper keeps the import list compact."""
    from dataclasses import fields

    return fields(cls)


@dataclass(frozen=True)
class ServiceSettings:
    release_id: str = "fall-risk-v0.1"
    model_path: Path = Path("models/yolov8n-pose.pt")
    demo_stream_url: str | None = None
    gait_model_path: Path | None = None
    gait_model_device: str = "auto"
    gait_model_window_frames: int = 16
    sit_stand_runtime_mode: str = "rule_baseline"
    sit_stand_model_path: Path | None = None
    sit_stand_model_device: str = "cpu"
    sit_stand_model_batch_size: int = 128
    near_fall_runtime_mode: str = "rule_baseline"
    near_fall_model_path: Path | None = None
    near_fall_score_threshold: float = 0.7
    near_fall_alert_cooldown_sec: float = 30.0
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
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)

    def __post_init__(self) -> None:
        normalized_release_id = str(self.release_id).strip()
        if not normalized_release_id:
            raise ValueError("release_id must not be empty")
        object.__setattr__(self, "release_id", normalized_release_id)
        if not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.demo_stream_url is not None:
            stream_url = self.demo_stream_url.strip()
            scheme, separator, remainder = stream_url.partition("://")
            if not separator or scheme.lower() not in {"rtsp", "rtmp", "http", "https"} or not remainder:
                raise ValueError("demo_stream_url must be a supported stream URL")
            object.__setattr__(self, "demo_stream_url", stream_url)
        if self.baseline_history_path is not None and not isinstance(self.baseline_history_path, Path):
            object.__setattr__(self, "baseline_history_path", Path(self.baseline_history_path))
        if self.gait_model_path is not None and not isinstance(self.gait_model_path, Path):
            object.__setattr__(self, "gait_model_path", Path(self.gait_model_path))
        if self.sit_stand_model_path is not None and not isinstance(self.sit_stand_model_path, Path):
            object.__setattr__(self, "sit_stand_model_path", Path(self.sit_stand_model_path))
        if self.near_fall_model_path is not None and not isinstance(
            self.near_fall_model_path, Path
        ):
            object.__setattr__(
                self, "near_fall_model_path", Path(self.near_fall_model_path)
            )
        object.__setattr__(
            self,
            "fall_event_shadow_checkpoint_paths",
            tuple(Path(value) for value in self.fall_event_shadow_checkpoint_paths),
        )
        if not isinstance(self.environment, EnvironmentSettings):
            object.__setattr__(self, "environment", EnvironmentSettings.from_mapping(self.environment))
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
        if self.near_fall_runtime_mode not in {"rule_baseline", "tabular_rescorer"}:
            raise ValueError(
                "near_fall_runtime_mode must be rule_baseline or tabular_rescorer"
            )
        if (
            self.near_fall_runtime_mode == "tabular_rescorer"
            and self.near_fall_model_path is None
        ):
            raise ValueError(
                "near_fall_model_path is required for tabular_rescorer"
            )
        if not 0.0 < self.near_fall_score_threshold < 1.0:
            raise ValueError("near_fall_score_threshold must be within (0, 1)")
        if self.near_fall_alert_cooldown_sec <= 0:
            raise ValueError("near_fall_alert_cooldown_sec must be positive")
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
        config_path = path or Path(
            env.get(
                "FALL_RISK_SERVICE_CONFIG",
                "configs/modules/fall_risk_service_v2.yaml",
            )
        )
        raw: dict[str, Any] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("fall risk service config must be a mapping")
            raw.update(loaded)
        environment_raw = raw.pop("environment", None)
        if environment_raw is not None:
            raw["environment"] = EnvironmentSettings.from_mapping(environment_raw)

        overrides: dict[str, tuple[str, Any]] = {
            "FALL_RISK_RELEASE_ID": ("release_id", str),
            "MODEL_PATH": ("model_path", Path),
            "DEMO_FALL_RISK_STREAM_URL": ("demo_stream_url", str),
            "GAIT_MODEL_PATH": ("gait_model_path", Path),
            "GAIT_MODEL_DEVICE": ("gait_model_device", str),
            "GAIT_MODEL_WINDOW_FRAMES": ("gait_model_window_frames", int),
            "SIT_STAND_RUNTIME_MODE": ("sit_stand_runtime_mode", str),
            "SIT_STAND_MODEL_PATH": ("sit_stand_model_path", Path),
            "SIT_STAND_MODEL_DEVICE": ("sit_stand_model_device", str),
            "SIT_STAND_MODEL_BATCH_SIZE": ("sit_stand_model_batch_size", int),
            "NEAR_FALL_RUNTIME_MODE": ("near_fall_runtime_mode", str),
            "NEAR_FALL_MODEL_PATH": ("near_fall_model_path", Path),
            "NEAR_FALL_SCORE_THRESHOLD": ("near_fall_score_threshold", float),
            "NEAR_FALL_ALERT_COOLDOWN_SEC": (
                "near_fall_alert_cooldown_sec",
                float,
            ),
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
        if raw.get("near_fall_model_path"):
            raw["near_fall_model_path"] = Path(raw["near_fall_model_path"])
        if raw.get("fall_event_shadow_checkpoint_paths"):
            raw["fall_event_shadow_checkpoint_paths"] = tuple(
                Path(value) for value in raw["fall_event_shadow_checkpoint_paths"]
            )
        if "ENVIRONMENT_MODE" in env or "ENVIRONMENT_SHADOW_LOG_PATH" in env:
            current = raw.get("environment")
            current_values = {
                field.name: getattr(current, field.name)
                for field in dataclass_fields(EnvironmentSettings)
            } if isinstance(current, EnvironmentSettings) else dict(current or {})
            if "ENVIRONMENT_MODE" in env:
                current_values["mode"] = env["ENVIRONMENT_MODE"]
            if "ENVIRONMENT_SHADOW_LOG_PATH" in env:
                current_values["shadow_log_path"] = env["ENVIRONMENT_SHADOW_LOG_PATH"] or None
            raw["environment"] = EnvironmentSettings.from_mapping(current_values)
        if "callback_retry_delays_sec" in raw:
            raw["callback_retry_delays_sec"] = tuple(float(value) for value in raw["callback_retry_delays_sec"])
        return cls(**raw)
