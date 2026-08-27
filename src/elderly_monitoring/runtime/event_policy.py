from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Mapping
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
    cooldown_by_trigger: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        self.cooldown_by_trigger = {
            str(trigger): float(value)
            for trigger, value in self.cooldown_by_trigger.items()
        }
        if any(value < 0 for value in self.cooldown_by_trigger.values()):
            raise ValueError("trigger-specific cooldowns must be non-negative")
        self._last: tuple[int, str, tuple[str, ...], float] | None = None
        self._last_trigger_send: dict[str, float] = {}

    def should_send(self, event: AlgorithmEvent, *, monotonic_sec: float) -> bool:
        if event.risk_level <= 0:
            return False
        if self._trigger_cooldown_active(event, monotonic_sec=monotonic_sec):
            return False
        signature = (event.risk_level, event.trigger_event, tuple(sorted(event.risk_factors)))
        if self._last is None:
            self._last = (*signature, monotonic_sec)
            self._record_trigger_send(event, monotonic_sec=monotonic_sec)
            return True
        last_level, _, _, last_time = self._last
        if event.risk_level > last_level:
            self._last = (*signature, monotonic_sec)
            self._record_trigger_send(event, monotonic_sec=monotonic_sec)
            return True
        if event.risk_level < last_level:
            return False
        cooldown = self.cooldown_by_trigger.get(
            event.trigger_event,
            self.cooldown_sec,
        )
        if signature == self._last[:3] and monotonic_sec - last_time < cooldown:
            return False
        self._last = (*signature, monotonic_sec)
        self._record_trigger_send(event, monotonic_sec=monotonic_sec)
        return True

    def reset(self, *, preserve_trigger_cooldowns: bool = False) -> None:
        self._last = None
        if not preserve_trigger_cooldowns:
            self._last_trigger_send.clear()

    def _trigger_cooldown_active(
        self, event: AlgorithmEvent, *, monotonic_sec: float
    ) -> bool:
        cooldown = self.cooldown_by_trigger.get(event.trigger_event)
        last_send = self._last_trigger_send.get(event.trigger_event)
        return bool(
            cooldown is not None
            and last_send is not None
            and monotonic_sec - last_send < cooldown
        )

    def _record_trigger_send(
        self, event: AlgorithmEvent, *, monotonic_sec: float
    ) -> None:
        if event.trigger_event in self.cooldown_by_trigger:
            self._last_trigger_send[event.trigger_event] = monotonic_sec

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
            if self._trigger_cooldown_active(event, monotonic_sec=monotonic_sec):
                return None
            signature = (
                event.risk_level,
                event.trigger_event,
                tuple(sorted(event.risk_factors)),
            )
            self._last = (*signature, monotonic_sec)
            self._record_trigger_send(event, monotonic_sec=monotonic_sec)
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
