from __future__ import annotations

from dataclasses import dataclass

from elderly_monitoring.common.schemas import AlgorithmEvent


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
