from __future__ import annotations

from typing import Any, Mapping
import json
from pathlib import Path

from elderly_monitoring.common.schemas import AlgorithmEvent
from elderly_monitoring.modules.fall_risk.pipeline import FallRiskPipeline
from elderly_monitoring.runtime.event_policy import EventPolicy
from elderly_monitoring.runtime.feature_assembly import FeatureAssembler, FeatureAssemblyConfig
from elderly_monitoring.runtime.streaming_pose import StreamingPoseTracker
from elderly_monitoring.service.callback import CallbackSender


class RealtimeFallRiskEngine:
    def __init__(self, *, assembler: Any, fusion_interval_sec: float = 2.0, pipeline: FallRiskPipeline | None = None) -> None:
        self.assembler = assembler
        self.fusion_interval_sec = fusion_interval_sec
        self.pipeline = pipeline or FallRiskPipeline()
        self._last_fusion: float | None = None

    def process_pose(self, record: Mapping[str, Any], *, monotonic_sec: float) -> AlgorithmEvent | None:
        snapshot = self.assembler.add_pose(record, monotonic_sec=monotonic_sec)
        if snapshot is None or not snapshot.usable:
            return None
        if self._last_fusion is not None and monotonic_sec - self._last_fusion < self.fusion_interval_sec and not snapshot.urgent:
            return None
        self._last_fusion = monotonic_sec
        return self.pipeline.predict_from_features(snapshot.features)

    def reset_window(self) -> None:
        self._last_fusion = None
        self.assembler.reset()


class FallRiskSessionEngine:
    """Compose one pose tracker, in-memory feature analysis and callback policy."""

    def __init__(self, *, session: Any, model_path: str, callback_token: str = "", scene_risk_scores: Mapping[str, float] | None = None, baseline_history_path: str | Path | None = None, **kwargs: Any) -> None:
        self.session = session
        self.tracker = StreamingPoseTracker(model_name=model_path, person_id=session.person_id, scene_region=session.scene_region)
        history = _load_jsonl(Path(baseline_history_path)) if baseline_history_path else []
        self.assembler = FeatureAssembler(
            person_id=session.person_id,
            device_id=session.device_id,
            scene_region=session.scene_region,
            scene_risk_scores=scene_risk_scores,
            config=FeatureAssemblyConfig(window_sec=float(kwargs.get("pose_window_sec", 10.0)), analysis_interval_sec=float(kwargs.get("analysis_interval_sec", 0.5))),
            baseline_history=history,
        )
        self.engine = RealtimeFallRiskEngine(assembler=self.assembler, fusion_interval_sec=float(kwargs.get("fusion_interval_sec", 2.0)))
        self.policy = EventPolicy(cooldown_sec=float(kwargs.get("event_cooldown_sec", 30.0)))
        self.callback = CallbackSender(token=callback_token, timeout=float(kwargs.get("callback_timeout_sec", 5.0)), retry_delays=kwargs.get("callback_retry_delays_sec", (0.5, 1.0, 2.0)))
        self.frame_id = 0
        self.primary_pose_count = 0
        self.max_inference_fps = float(kwargs.get("max_inference_fps", 8.0))
        self._last_inference_at: float | None = None

    def process_frame(self, frame: Any, *, timestamp_sec: float = 0.0) -> None:
        if self._last_inference_at is not None and self.max_inference_fps > 0 and timestamp_sec - self._last_inference_at < 1.0 / self.max_inference_fps:
            return
        self._last_inference_at = timestamp_sec
        self.frame_id += 1
        height, width = frame.shape[:2]
        result = self.tracker.process_frame(frame, frame_id=self.frame_id, timestamp_sec=timestamp_sec, frame_size=(float(width), float(height)))
        if result.window_reset:
            self.engine.reset_window()
        if result.primary_pose is None:
            return
        self.primary_pose_count += 1
        event = self.engine.process_pose(result.primary_pose.to_dict(), monotonic_sec=timestamp_sec)
        if event is not None and self.policy.should_send(event, monotonic_sec=timestamp_sec):
            self.callback.send(self.session.callback_url, event, session_id=self.session.session_id)

    def close(self) -> None:
        self.tracker.close()
        self.callback.close()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
