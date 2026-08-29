from __future__ import annotations

from collections import deque
import threading
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from elderly_monitoring.service.stream_reader import StreamReader
from elderly_monitoring.service.frame_buffer import FramePacket, LatestFrameBuffer


class SessionStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class MonitoringSession:
    session_id: str
    request_id: str
    stream_url: str
    device_id: str
    person_id: str
    scene_region: str
    callback_url: str
    status: SessionStatus = SessionStatus.STARTING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_frame_at: datetime | None = None
    last_error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    reader: Any | None = field(default=None, repr=False)
    engine: Any | None = field(default=None, repr=False)
    stream_epoch: int = 0
    frame_diagnostics: dict[str, Any] = field(default_factory=dict)
    epoch_history: list[dict[str, Any]] = field(default_factory=list)
    runtime_diagnostics: dict[str, Any] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] | None = None
    latest_frame: Any | None = field(default=None, repr=False)
    active_frame_buffer: LatestFrameBuffer | None = field(default=None, repr=False)
    risk_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=32), repr=False
    )
    event_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=12), repr=False
    )
    last_risk_signature: tuple[Any, ...] | None = field(default=None, repr=False)
    last_timeline_signature: tuple[Any, ...] | None = field(default=None, repr=False)
    next_epoch_reason: str = field(default="session_started", repr=False)
    stream_url_revision: int = field(default=0, repr=False)


class SessionManager:
    def __init__(
        self,
        *,
        reader_factory: Callable[..., Any] = StreamReader,
        engine_factory: Callable[..., Any] | None = None,
        model_path: str = "yolov8n-pose.pt",
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 1.0,
        frame_queue_capacity: int = 2,
        stop_budget_sec: float = 5.0,
        reconnect_stable_after_sec: float = 30.0,
        reconnect_stable_after_frames: int = 120,
        **runtime_kwargs: Any,
    ) -> None:
        self.reader_factory = reader_factory
        self.engine_factory = engine_factory or _default_engine_factory
        self.model_path = model_path
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self.frame_queue_capacity = frame_queue_capacity
        self.stop_budget_sec = float(stop_budget_sec)
        self.reconnect_stable_after_sec = float(reconnect_stable_after_sec)
        self.reconnect_stable_after_frames = int(reconnect_stable_after_frames)
        self.runtime_kwargs = runtime_kwargs
        self.sessions: dict[str, MonitoringSession] = {}
        self._lock = threading.RLock()

    def start(self, **kwargs: Any) -> MonitoringSession:
        with self._lock:
            existing = next((session for session in self.sessions.values() if session.request_id == kwargs["request_id"]), None)
            if existing:
                return existing
            if any(session.status not in {SessionStatus.STOPPED, SessionStatus.FAILED} for session in self.sessions.values()):
                raise ValueError("another session is active")
            session = MonitoringSession(session_id=str(uuid.uuid4()), **kwargs)
            self.sessions[session.session_id] = session
            session.thread = threading.Thread(target=self._run, args=(session,), daemon=True, name=f"fall-session-{session.session_id[:8]}")
            session.thread.start()
            return session

    def get(self, session_id: str) -> MonitoringSession | None:
        session = self.sessions.get(session_id)
        if session is not None:
            self._refresh_runtime_diagnostics(session)
        return session

    def update_url(self, session_id: str, stream_url: str) -> MonitoringSession | None:
        session = self.get(session_id)
        if session is None:
            return None
        with self._lock:
            session.stream_url = stream_url
            session.stream_url_revision += 1
            session.next_epoch_reason = "stream_url_updated"
            if session.reader is not None:
                session.reader.release()
            session.reader = None
        return session

    def update_baseline_period(
        self, session_id: str, period: dict[str, Any] | None
    ) -> MonitoringSession | None:
        session = self.get(session_id)
        if session is None:
            return None
        engine = session.engine
        if engine is None:
            raise ValueError("session engine is not ready")
        updater = getattr(engine, "update_baseline_period", None)
        if updater is None:
            raise ValueError("session engine does not support baseline period updates")
        result = updater(period)
        if period is None:
            session.baseline_comparison = None
        else:
            provenance = period.get("data_provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            baseline_quality = (
                result.get("baseline_quality", {})
                if isinstance(result, Mapping)
                else {}
            )
            session.baseline_comparison = {
                "period_id": str(period.get("period_id") or ""),
                "data_type": str(provenance.get("data_type") or "unspecified"),
                "display_label": str(provenance.get("display_label") or ""),
                "disclosure": str(provenance.get("disclosure") or ""),
                "baseline_deviation_score": (
                    result.get("baseline_deviation_score")
                    if isinstance(result, Mapping)
                    else None
                ),
                "baseline_state": (
                    result.get("baseline_state")
                    if isinstance(result, Mapping)
                    else None
                ),
                "baseline_confidence": (
                    result.get("baseline_confidence")
                    if isinstance(result, Mapping)
                    else None
                ),
                "deviation_factors": list(
                    result.get("deviation_factors", [])
                    if isinstance(result, Mapping)
                    else []
                ),
                "history_day_count": baseline_quality.get("history_day_count"),
                "history_record_count": baseline_quality.get(
                    "history_record_count"
                ),
            }
        self._refresh_runtime_diagnostics(session)
        return session

    def stop(self, session_id: str) -> MonitoringSession | None:
        session = self.get(session_id)
        if session is None:
            return None
        if session.status == SessionStatus.STOPPED:
            return session
        session.status = SessionStatus.STOPPING
        session.stop_event.set()
        stop_started = time.monotonic()
        if session.reader is not None:
            session.reader.release()
        if session.thread and session.thread is not threading.current_thread():
            session.thread.join(timeout=self.stop_budget_sec)
        thread_alive = bool(session.thread and session.thread.is_alive())
        session.runtime_diagnostics["stop"] = {
            "duration_sec": round(time.monotonic() - stop_started, 6),
            "thread_alive_after_budget": thread_alive,
            "budget_sec": self.stop_budget_sec,
        }
        if not thread_alive and session.status != SessionStatus.STOPPED:
            session.status = SessionStatus.STOPPED
        return session

    def _run(self, session: MonitoringSession) -> None:
        try:
            session.engine = self.engine_factory(session=session, model_path=self.model_path, **self.runtime_kwargs)
            attempts = 0
            while not session.stop_event.is_set():
                with self._lock:
                    stream_url = session.stream_url
                    url_revision = session.stream_url_revision
                reader_kwargs = {
                    key: value for key, value in self.runtime_kwargs.items()
                    if key in {"open_timeout_ms", "read_timeout_ms"}
                }
                reader = self.reader_factory(stream_url, **reader_kwargs)
                with self._lock:
                    if (
                        session.stop_event.is_set()
                        or session.stream_url_revision != url_revision
                    ):
                        reader.release()
                        attempts = 0
                        continue
                    session.reader = reader
                try:
                    reader.open()
                    with self._lock:
                        stopped = session.stop_event.is_set()
                        stale_url = session.stream_url_revision != url_revision
                        if (stopped or stale_url) and session.reader is reader:
                            session.reader = None
                        if not stopped and not stale_url:
                            session.stream_epoch += 1
                            epoch = session.stream_epoch
                            epoch_reason = session.next_epoch_reason
                            session.next_epoch_reason = "stream_reconnected"
                            session.risk_history.clear()
                            session.event_history.clear()
                            session.last_risk_signature = None
                            session.last_timeline_signature = None
                    if stopped:
                        reader.release()
                        break
                    if stale_url:
                        reader.release()
                        attempts = 0
                        continue
                    epoch_started = time.monotonic()
                    if hasattr(session.engine, "begin_stream_epoch"):
                        session.engine.begin_stream_epoch(
                            epoch,
                            reason=epoch_reason,
                            started_monotonic_sec=epoch_started,
                        )
                    session.status = SessionStatus.RUNNING
                    buffer = LatestFrameBuffer(capacity=self.frame_queue_capacity)
                    session.active_frame_buffer = buffer
                    producer = threading.Thread(
                        target=self._capture_frames,
                        args=(session, reader, buffer, epoch),
                        daemon=True,
                        name=f"fall-capture-{session.session_id[:8]}-{epoch}",
                    )
                    producer.start()
                    while not session.stop_event.is_set():
                        packet = buffer.get(timeout=0.1)
                        if packet is None:
                            if buffer.snapshot()["closed"]:
                                break
                            continue
                        if packet.stream_epoch != session.stream_epoch:
                            continue
                        session.last_frame_at = datetime.now(timezone.utc)
                        if hasattr(session.engine, "process_frame"):
                            session.engine.process_frame(
                                packet.frame,
                                source_pts_sec=packet.source_pts_sec,
                                received_monotonic_sec=packet.received_monotonic_sec,
                                stream_epoch=packet.stream_epoch,
                            )
                    buffer.discard_pending(reason="epoch_ended")
                    buffer.close(reason="epoch_ended")
                    reader.release()
                    producer.join(timeout=1.0)
                    diagnostics = buffer.snapshot()
                    self._record_epoch_diagnostics(session, epoch, diagnostics)
                    if session.active_frame_buffer is buffer:
                        session.active_frame_buffer = None
                    reader.release()
                    with self._lock:
                        if session.reader is reader:
                            session.reader = None
                    if session.stop_event.is_set():
                        break
                    session.status = SessionStatus.RECONNECTING
                    epoch_duration_sec = time.monotonic() - epoch_started
                    stable_epoch = (
                        epoch_duration_sec >= self.reconnect_stable_after_sec
                        or int(diagnostics.get("put_count", 0))
                        >= self.reconnect_stable_after_frames
                    )
                    if stable_epoch or session.stream_url_revision != url_revision:
                        attempts = 0
                    attempts += 1
                    if attempts > self.reconnect_attempts:
                        raise RuntimeError("stream reconnect attempts exhausted")
                    if session.stop_event.wait(self.reconnect_delay_sec):
                        break
                except Exception:
                    reader.release()
                    active_buffer = session.active_frame_buffer
                    if active_buffer is not None:
                        active_buffer.close(reason="stream_error")
                        self._record_epoch_diagnostics(
                            session,
                            session.stream_epoch,
                            active_buffer.snapshot(),
                        )
                        if session.active_frame_buffer is active_buffer:
                            session.active_frame_buffer = None
                    with self._lock:
                        if session.reader is reader:
                            session.reader = None
                        stale_url = session.stream_url_revision != url_revision
                    if session.stop_event.is_set():
                        break
                    if stale_url:
                        attempts = 0
                        continue
                    attempts += 1
                    if attempts > self.reconnect_attempts:
                        raise
                    session.status = SessionStatus.RECONNECTING
                    if session.stop_event.wait(self.reconnect_delay_sec):
                        break
        except Exception as exc:
            session.last_error = str(exc)
            session.status = SessionStatus.FAILED
        finally:
            if session.reader is not None:
                session.reader.release()
                session.reader = None
            if session.engine is not None and hasattr(session.engine, "close"):
                self._refresh_runtime_diagnostics(session)
                session.engine.close()
                self._refresh_runtime_diagnostics(session)
            session.engine = None
            if session.stop_event.is_set() or session.status != SessionStatus.FAILED:
                session.status = SessionStatus.STOPPED

    @staticmethod
    def _refresh_runtime_diagnostics(session: MonitoringSession) -> None:
        engine = session.engine
        if engine is None:
            return
        sampling = getattr(engine, "sampling_diagnostics", None)
        if isinstance(sampling, dict):
            if sampling.get("frame_count", 0) or "sampling" not in session.runtime_diagnostics:
                session.runtime_diagnostics["sampling"] = sampling
        engine_runtime = getattr(engine, "runtime_diagnostics", None)
        if isinstance(engine_runtime, dict):
            session.runtime_diagnostics.update(engine_runtime)
            SessionManager._record_live_history(session, engine_runtime)
        if session.baseline_comparison is not None:
            session.runtime_diagnostics["baseline_comparison"] = dict(
                session.baseline_comparison
            )
        else:
            session.runtime_diagnostics.pop("baseline_comparison", None)
        session.runtime_diagnostics["risk_history"] = list(session.risk_history)
        session.runtime_diagnostics["event_history"] = list(session.event_history)
        frame_queue = dict(session.frame_diagnostics)
        active_buffer = session.active_frame_buffer
        if active_buffer is not None:
            active = active_buffer.snapshot()
            for field in ("put_count", "get_count", "dropped_oldest"):
                frame_queue[field] = int(frame_queue.get(field, 0)) + int(
                    active.get(field, 0)
                )
            frame_queue.update({
                "capacity": active.get("capacity"),
                "depth": active.get("depth"),
                "closed": active.get("closed"),
                "close_reason": active.get("close_reason"),
                "last_dropped_source_pts_sec": active.get(
                    "last_dropped_source_pts_sec"
                ),
            })
        if frame_queue:
            session.runtime_diagnostics["frame_queue"] = frame_queue
        session.runtime_diagnostics.update({
            "processed_frames": int(getattr(engine, "frame_id", 0)),
            "primary_pose_count": int(getattr(engine, "primary_pose_count", 0)),
            "analysis_count": int(
                getattr(getattr(engine, "assembler", None), "analysis_count", 0)
            ),
            "time_boundary_count": int(getattr(engine, "time_boundary_count", 0)),
            "last_time_boundary_reason": getattr(engine, "last_time_boundary_reason", None),
        })

    @staticmethod
    def _record_live_history(
        session: MonitoringSession,
        runtime_diagnostics: dict[str, Any],
    ) -> None:
        event = runtime_diagnostics.get("latest_event")
        if not isinstance(event, dict):
            return
        score = event.get("risk_score")
        if not isinstance(score, (int, float)):
            return
        risk_signature = (
            session.stream_epoch,
            event.get("timestamp"),
            event.get("risk_level"),
            float(score),
        )
        if risk_signature != session.last_risk_signature:
            session.last_risk_signature = risk_signature
            session.risk_history.append(round(float(score), 6))
        level = event.get("risk_level")
        if not isinstance(level, int) or level <= 0:
            session.last_timeline_signature = None
            return
        trigger = str(event.get("trigger_event") or "risk_event")
        timeline_signature = (session.stream_epoch, level, trigger)
        if timeline_signature == session.last_timeline_signature:
            return
        session.last_timeline_signature = timeline_signature
        source_time = (
            (event.get("evidence_window") or {}).get("end_time")
            if isinstance(event.get("evidence_window"), dict)
            else None
        )
        if not isinstance(source_time, (int, float)):
            last_frame = runtime_diagnostics.get("last_frame")
            source_time = (
                last_frame.get("source_pts_sec", 0.0)
                if isinstance(last_frame, dict)
                else 0.0
            )
        elapsed = (
            max(0, int(source_time))
            if isinstance(source_time, (int, float))
            else 0
        )
        session.event_history.append({
            "event_id": event.get("event_id"),
            "timestamp": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
            "trigger_event": trigger,
            "risk_level": level,
            "risk_score": round(float(score), 6),
            "confidence": event.get("confidence"),
            "evidence_window": (
                dict(event.get("evidence_window"))
                if isinstance(event.get("evidence_window"), dict)
                else None
            ),
            "risk_factors": list(event.get("risk_factors") or []),
            "recommended_action": event.get("recommended_action"),
        })

    @staticmethod
    def _capture_frames(
        session: MonitoringSession,
        reader: Any,
        buffer: LatestFrameBuffer,
        stream_epoch: int,
    ) -> None:
        try:
            while not session.stop_event.is_set() and stream_epoch == session.stream_epoch:
                frame = reader.read()
                received = time.monotonic()
                if frame is None:
                    break
                session.latest_frame = frame
                source_pts = getattr(reader, "source_pts_sec", None)
                buffer.put(FramePacket(
                    frame=frame,
                    stream_epoch=stream_epoch,
                    source_pts_sec=(float(source_pts) if source_pts is not None else None),
                    received_monotonic_sec=received,
                ))
        finally:
            buffer.close(reason=("session_stopped" if session.stop_event.is_set() else "stream_ended"))

    @staticmethod
    def _record_epoch_diagnostics(
        session: MonitoringSession,
        stream_epoch: int,
        diagnostics: dict[str, Any],
    ) -> None:
        entry = {"stream_epoch": stream_epoch, **diagnostics}
        session.epoch_history.append(entry)
        aggregate = dict(session.frame_diagnostics)
        for field in ("put_count", "get_count", "dropped_oldest"):
            aggregate[field] = int(aggregate.get(field, 0)) + int(diagnostics.get(field, 0))
        aggregate.update({
            "last_stream_epoch": stream_epoch,
            "last_close_reason": diagnostics.get("close_reason"),
            "last_dropped_source_pts_sec": diagnostics.get("last_dropped_source_pts_sec"),
        })
        session.frame_diagnostics = aggregate


def _default_engine_factory(*, session: MonitoringSession, model_path: str, **kwargs: Any) -> Any:
    from elderly_monitoring.runtime.realtime_fall_risk import FallRiskSessionEngine

    return FallRiskSessionEngine(session=session, model_path=model_path, **kwargs)
