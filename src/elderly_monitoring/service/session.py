from __future__ import annotations

import threading
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from elderly_monitoring.service.stream_reader import StreamReader


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


class SessionManager:
    def __init__(
        self,
        *,
        reader_factory: Callable[..., Any] = StreamReader,
        engine_factory: Callable[..., Any] | None = None,
        model_path: str = "yolov8n-pose.pt",
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 1.0,
        **runtime_kwargs: Any,
    ) -> None:
        self.reader_factory = reader_factory
        self.engine_factory = engine_factory or _default_engine_factory
        self.model_path = model_path
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
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
        return self.sessions.get(session_id)

    def update_url(self, session_id: str, stream_url: str) -> MonitoringSession | None:
        session = self.get(session_id)
        if session is None:
            return None
        with self._lock:
            session.stream_url = stream_url
            if session.reader is not None:
                session.reader.release()
            session.reader = None
        return session

    def stop(self, session_id: str) -> MonitoringSession | None:
        session = self.get(session_id)
        if session is None:
            return None
        if session.status == SessionStatus.STOPPED:
            return session
        session.status = SessionStatus.STOPPING
        session.stop_event.set()
        if session.reader is not None:
            session.reader.release()
        if session.thread and session.thread is not threading.current_thread():
            session.thread.join(timeout=5.0)
        if session.status != SessionStatus.STOPPED:
            session.status = SessionStatus.STOPPED
        return session

    def _run(self, session: MonitoringSession) -> None:
        try:
            monotonic_start = time.monotonic()
            session.engine = self.engine_factory(session=session, model_path=self.model_path, **self.runtime_kwargs)
            attempts = 0
            while not session.stop_event.is_set():
                reader_kwargs = {
                    key: value for key, value in self.runtime_kwargs.items()
                    if key in {"open_timeout_ms", "read_timeout_ms"}
                }
                reader = self.reader_factory(session.stream_url, **reader_kwargs)
                session.reader = reader
                try:
                    reader.open()
                    session.status = SessionStatus.RUNNING
                    attempts = 0
                    while not session.stop_event.is_set():
                        frame = reader.read()
                        if frame is None:
                            break
                        session.last_frame_at = datetime.now(timezone.utc)
                        if hasattr(session.engine, "process_frame"):
                            session.engine.process_frame(frame, timestamp_sec=time.monotonic() - monotonic_start)
                    reader.release()
                    session.reader = None
                    if session.stop_event.is_set():
                        break
                    session.status = SessionStatus.RECONNECTING
                    attempts += 1
                    if attempts > self.reconnect_attempts:
                        raise RuntimeError("stream reconnect attempts exhausted")
                    if session.stop_event.wait(self.reconnect_delay_sec):
                        break
                except Exception:
                    reader.release()
                    session.reader = None
                    if session.stop_event.is_set():
                        break
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
                session.engine.close()
            session.engine = None
            if session.stop_event.is_set() or session.status != SessionStatus.FAILED:
                session.status = SessionStatus.STOPPED


def _default_engine_factory(*, session: MonitoringSession, model_path: str, **kwargs: Any) -> Any:
    from elderly_monitoring.runtime.realtime_fall_risk import FallRiskSessionEngine

    return FallRiskSessionEngine(session=session, model_path=model_path, **kwargs)
