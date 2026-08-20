from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any
import uuid

from elderly_monitoring.common.schemas import AlgorithmEvent


@dataclass(frozen=True)
class RiskEventVersion:
    event_id: str
    episode_id: str
    session_id: str
    stream_epoch: int
    lifecycle_state: str
    version_kind: str
    generated_at: str
    payload_json: str

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def risk_level(self) -> int:
        return int(self.payload["risk_level"])

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle_state in {"recovered", "unresolved"}


@dataclass
class EventPolicy:
    cooldown_sec: float = 30.0

    def __post_init__(self) -> None:
        self._last: tuple[int, str, tuple[str, ...], float] | None = None

    def should_send(self, event: AlgorithmEvent, *, monotonic_sec: float) -> bool:
        if event.risk_level <= 0:
            return False
        signature = (event.risk_level, event.trigger_event, tuple(sorted(event.risk_factors)))
        if self._last is None:
            self._last = (*signature, monotonic_sec)
            return True
        last_level, _, _, last_time = self._last
        if event.risk_level > last_level:
            self._last = (*signature, monotonic_sec)
            return True
        if event.risk_level < last_level:
            return False
        if signature == self._last[:3] and monotonic_sec - last_time < self.cooldown_sec:
            return False
        self._last = (*signature, monotonic_sec)
        return True

    def reset(self) -> None:
        self._last = None

    def create_version(
        self,
        event: AlgorithmEvent,
        *,
        monotonic_sec: float,
        episode_id: str,
        stream_epoch: int,
        lifecycle_state: str,
        version_kind: str,
        session_id: str = "",
        force: bool = False,
    ) -> RiskEventVersion | None:
        if not force and not self.should_send(event, monotonic_sec=monotonic_sec):
            return None
        if force and event.risk_level > 0:
            signature = (
                event.risk_level,
                event.trigger_event,
                tuple(sorted(event.risk_factors)),
            )
            self._last = (*signature, monotonic_sec)
        event_id = str(uuid.uuid4())
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = event.to_dict()
        payload.update({
            "event_id": event_id,
            "episode_id": str(episode_id),
            "session_id": str(session_id),
            "stream_epoch": int(stream_epoch),
            "lifecycle_state": str(lifecycle_state),
            "version_kind": str(version_kind),
            "first_generated_time": generated_at,
            "schema_version": "1.0",
        })
        return RiskEventVersion(
            event_id=event_id,
            episode_id=str(episode_id),
            session_id=str(session_id),
            stream_epoch=int(stream_epoch),
            lifecycle_state=str(lifecycle_state),
            version_kind=str(version_kind),
            generated_at=generated_at,
            payload_json=json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
