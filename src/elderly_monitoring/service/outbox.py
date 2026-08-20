from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Any, Iterable

from elderly_monitoring.runtime.event_policy import RiskEventVersion
from elderly_monitoring.service.callback import DeliveryAttempt


@dataclass
class _OutboxRecord:
    callback_url: str
    version: RiskEventVersion
    status: str = "pending"
    attempt_count: int = 0
    retry_count: int = 0
    last_error: str | None = None
    last_status_code: int | None = None
    next_attempt_monotonic_sec: float = 0.0
    in_flight: bool = False

    def diagnostic(self) -> dict[str, Any]:
        payload = self.version.payload
        evidence = payload.get("evidence_window")
        source_event_time = None
        if isinstance(evidence, dict):
            source_event_time = evidence.get("end_time")
        return {
            "event_id": self.version.event_id,
            "episode_id": self.version.episode_id,
            "session_id": self.version.session_id,
            "stream_epoch": self.version.stream_epoch,
            "lifecycle_state": self.version.lifecycle_state,
            "version_kind": self.version.version_kind,
            "first_generated_time": self.version.generated_at,
            "source_event_time": source_event_time,
            "evidence_window": evidence,
            "risk_level": self.version.risk_level,
            "retry_count": self.retry_count,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
            "delivery_status": self.status,
            "in_flight": self.in_flight,
        }


class CallbackOutbox:
    """Process-local bounded callback queue with one retry scheduler thread."""

    def __init__(
        self,
        *,
        sender: Any,
        capacity: int = 32,
        retry_delays: Iterable[float] = (0.5, 1.0, 2.0),
        monotonic: Any = time.monotonic,
        autostart: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("outbox capacity must be at least 1")
        delays = tuple(float(value) for value in retry_delays)
        if not delays:
            raise ValueError("outbox retry_delays must include at least one attempt")
        if any(value < 0 for value in delays):
            raise ValueError("outbox retry delays must be non-negative")
        self.sender = sender
        self.capacity = int(capacity)
        self.retry_delays = delays
        self._monotonic = monotonic
        self._condition = threading.Condition(threading.RLock())
        self._active: OrderedDict[str, _OutboxRecord] = OrderedDict()
        self._history: OrderedDict[str, _OutboxRecord] = OrderedDict()
        self._accepting = True
        self._halt = False
        self._worker: threading.Thread | None = None
        self._totals = {
            "delivered": 0,
            "failed": 0,
            "rejected": 0,
            "retry_count": 0,
            "final_failures": 0,
        }
        self._queue_full_decisions = {
            "rejected_regular": 0,
            "rejected_protected": 0,
            "evicted_regular": 0,
        }
        self._last_error: str | None = None
        self._last_queue_full_decision: dict[str, Any] | None = None
        self._stop_result: dict[str, Any] | None = None
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._run,
                daemon=True,
                name="fall-callback-outbox",
            )
            self._worker.start()

    def enqueue(self, callback_url: str, version: RiskEventVersion) -> bool:
        with self._condition:
            if not self._accepting or version.event_id in self._active:
                self._totals["rejected"] += 1
                self._last_error = "outbox_not_accepting_or_duplicate"
                return False
            if len(self._active) >= self.capacity and not self._make_room(version):
                return False
            self._history.pop(version.event_id, None)
            self._active[version.event_id] = _OutboxRecord(
                callback_url=str(callback_url),
                version=version,
                next_attempt_monotonic_sec=self._monotonic(),
            )
            self._condition.notify_all()
            return True

    def replay(self, event_id: str) -> bool:
        with self._condition:
            record = self._history.get(event_id)
            if record is None:
                return False
            callback_url = record.callback_url
            version = record.version
        return self.enqueue(callback_url, version)

    def wait_for_terminal(self, event_id: str, *, timeout: float) -> bool:
        deadline = self._monotonic() + max(0.0, float(timeout))
        with self._condition:
            while event_id in self._active:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return event_id in self._history

    def stop(self, *, drain_budget_sec: float) -> dict[str, Any]:
        budget = max(0.0, float(drain_budget_sec))
        deadline = self._monotonic() + budget
        with self._condition:
            if self._stop_result is not None:
                return dict(self._stop_result)
            self._accepting = False
            while self._active:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            self._halt = True
            self._condition.notify_all()
            unfinished = len(self._active)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=max(0.0, deadline - self._monotonic()))
        else:
            self.sender.close()
        result = {
            "budget_sec": budget,
            "delivered": self._totals["delivered"],
            "failed": self._totals["failed"],
            "unfinished": unfinished,
            "rejected": self._totals["rejected"],
            "worker_alive_after_budget": bool(worker and worker.is_alive()),
        }
        with self._condition:
            self._stop_result = dict(result)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            pending = sum(record.status == "pending" for record in self._active.values())
            retrying = sum(record.status == "retrying" for record in self._active.values())
            return {
                "depth": len(self._active),
                "capacity": self.capacity,
                "accepting": self._accepting,
                "status_counts": {
                    "pending": pending,
                    "retrying": retrying,
                    "delivered": self._totals["delivered"],
                    "failed": self._totals["failed"],
                },
                "retry_count": self._totals["retry_count"],
                "final_failures": self._totals["final_failures"],
                "rejected": self._totals["rejected"],
                "queue_full_decisions": dict(self._queue_full_decisions),
                "last_queue_full_decision": (
                    dict(self._last_queue_full_decision)
                    if self._last_queue_full_decision is not None
                    else None
                ),
                "last_error": self._last_error,
                "active_items": [record.diagnostic() for record in self._active.values()],
                "recent_terminal_items": [
                    record.diagnostic() for record in self._history.values()
                ],
                "stop_drain_result": (
                    dict(self._stop_result) if self._stop_result is not None else None
                ),
            }

    def _make_room(self, incoming: RiskEventVersion) -> bool:
        protected_incoming = incoming.is_terminal or incoming.risk_level >= 4
        candidate_id = next(
            (
                event_id
                for event_id, record in self._active.items()
                if not record.in_flight
                and not record.version.is_terminal
                and record.version.risk_level < 4
            ),
            None,
        )
        if protected_incoming and candidate_id is not None:
            candidate = self._active.pop(candidate_id)
            candidate.status = "failed"
            candidate.last_error = "queue_evicted_for_protected_event"
            self._last_error = candidate.last_error
            self._totals["failed"] += 1
            self._totals["final_failures"] += 1
            self._queue_full_decisions["evicted_regular"] += 1
            self._last_queue_full_decision = {
                "decision": "evicted_regular",
                "evicted_event_id": candidate.version.event_id,
                "incoming_event_id": incoming.event_id,
            }
            self._remember(candidate)
            return True

        decision = "rejected_protected" if protected_incoming else "rejected_regular"
        self._queue_full_decisions[decision] += 1
        self._totals["rejected"] += 1
        self._last_error = f"queue_full_{decision}"
        self._last_queue_full_decision = {
            "decision": decision,
            "incoming_event_id": incoming.event_id,
            "incoming_lifecycle_state": incoming.lifecycle_state,
            "incoming_risk_level": incoming.risk_level,
        }
        return False

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    record = self._next_ready_record()
                    while record is None:
                        if self._halt:
                            return
                        wait_for = self._next_wait_duration()
                        self._condition.wait(wait_for)
                        record = self._next_ready_record()
                    record.in_flight = True
                try:
                    result: DeliveryAttempt = self.sender.send_once(
                        record.callback_url,
                        record.version.payload,
                    )
                except Exception as exc:
                    result = DeliveryAttempt(
                        success=False,
                        error=f"{type(exc).__name__}: callback sender failed",
                    )
                with self._condition:
                    record.in_flight = False
                    record.attempt_count += 1
                    record.last_status_code = result.status_code
                    if result.success:
                        record.status = "delivered"
                        record.last_error = None
                        self._totals["delivered"] += 1
                        self._complete(record)
                    else:
                        record.last_error = result.error or "callback_attempt_failed"
                        self._last_error = record.last_error
                        if record.attempt_count >= len(self.retry_delays):
                            record.status = "failed"
                            self._totals["failed"] += 1
                            self._totals["final_failures"] += 1
                            self._complete(record)
                        else:
                            record.status = "retrying"
                            record.retry_count += 1
                            self._totals["retry_count"] += 1
                            record.next_attempt_monotonic_sec = (
                                self._monotonic()
                                + self.retry_delays[record.attempt_count - 1]
                            )
                    self._condition.notify_all()
        finally:
            self.sender.close()

    def _next_ready_record(self) -> _OutboxRecord | None:
        if self._halt:
            return None
        now = self._monotonic()
        ready = [
            record
            for record in self._active.values()
            if not record.in_flight and record.next_attempt_monotonic_sec <= now
        ]
        if not ready:
            return None
        return min(ready, key=lambda record: record.next_attempt_monotonic_sec)

    def _next_wait_duration(self) -> float | None:
        waiting = [
            record.next_attempt_monotonic_sec
            for record in self._active.values()
            if not record.in_flight
        ]
        if not waiting:
            return None
        return max(0.0, min(waiting) - self._monotonic())

    def _complete(self, record: _OutboxRecord) -> None:
        self._active.pop(record.version.event_id, None)
        self._remember(record)

    def _remember(self, record: _OutboxRecord) -> None:
        self._history[record.version.event_id] = record
        self._history.move_to_end(record.version.event_id)
        while len(self._history) > self.capacity:
            self._history.popitem(last=False)
