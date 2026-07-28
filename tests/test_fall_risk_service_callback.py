import unittest
import threading
import time

import httpx

from elderly_monitoring.common.schemas import AlgorithmEvent
from elderly_monitoring.service.callback import CallbackSender
from elderly_monitoring.runtime.event_policy import EventPolicy
from elderly_monitoring.service.outbox import CallbackOutbox, DeliveryAttempt


def _event():
    return AlgorithmEvent(
        module="fall_risk", person_id="elder-1", timestamp="2026-07-12T12:00:00+08:00",
        risk_level=3, risk_score=0.8, confidence=0.9, trigger_event="near_fall",
        risk_factors=["near_fall_event"], recommended_action="notify_guardian", model_version="test",
    )


class CallbackSenderTest(unittest.TestCase):
    def test_2xx_sends_once_with_service_fields_and_auth(self) -> None:
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(204)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        sender = CallbackSender(token="secret", client=client, retry_delays=(0.0, 0.0, 0.0))
        self.assertTrue(sender.send("https://backend.example/events", _event(), session_id="session-1"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer secret")
        payload = __import__("json").loads(requests[0].content)
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertTrue(payload["event_id"])

    def test_errors_retry_three_times_without_raising(self) -> None:
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(503)

        sender = CallbackSender(
            token="secret", client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry_delays=(0.0, 0.0, 0.0),
        )
        self.assertFalse(sender.send("https://backend.example/events", _event(), session_id="session-1"))
        self.assertEqual(len(calls), 3)

    def test_send_once_reports_non_2xx_without_retrying(self) -> None:
        calls = []
        sender = CallbackSender(
            token="secret",
            client=httpx.Client(transport=httpx.MockTransport(
                lambda request: calls.append(request) or httpx.Response(504)
            )),
        )
        result = sender.send_once("https://backend.example/events", {"event_id": "event-1"})
        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 504)
        self.assertEqual(len(calls), 1)

    def test_send_once_reports_connection_and_timeout_failures(self) -> None:
        for error in (
            httpx.ConnectError("cannot connect"),
            httpx.ReadTimeout("timed out"),
        ):
            with self.subTest(error=type(error).__name__):
                def handler(request):
                    raise error

                sender = CallbackSender(
                    token="secret",
                    client=httpx.Client(transport=httpx.MockTransport(handler)),
                )
                result = sender.send_once(
                    "https://backend.example/events", {"event_id": "event-1"}
                )
                self.assertFalse(result.success)
                self.assertIn(type(error).__name__, result.error)


class _ControlledSender:
    def __init__(self, results=None, gate=None):
        self.results = list(results or [DeliveryAttempt(success=True, status_code=204)])
        self.gate = gate
        self.calls = []
        self.called = threading.Event()
        self.closed = False

    def send_once(self, callback_url, payload):
        self.calls.append((callback_url, dict(payload)))
        self.called.set()
        if self.gate is not None:
            self.gate.wait(timeout=1.0)
        if self.results:
            return self.results.pop(0)
        return DeliveryAttempt(success=True, status_code=204)

    def close(self):
        self.closed = True


class _RaisingSender(_ControlledSender):
    def send_once(self, callback_url, payload):
        self.calls.append((callback_url, dict(payload)))
        raise RuntimeError("unexpected sender failure")


def _version(level=3, *, episode_id="episode-1", epoch=1, kind="initial", state="suspected"):
    return EventPolicy(cooldown_sec=0.0).create_version(
        _event().__class__(**{**_event().__dict__, "risk_level": level}),
        monotonic_sec=0.0,
        episode_id=episode_id,
        stream_epoch=epoch,
        lifecycle_state=state,
        version_kind=kind,
        force=True,
    )


class CallbackOutboxTest(unittest.TestCase):
    def test_retry_and_explicit_replay_keep_event_id_and_payload(self) -> None:
        sender = _ControlledSender([
            DeliveryAttempt(success=False, error="connection_failed"),
            DeliveryAttempt(success=True, status_code=204),
            DeliveryAttempt(success=True, status_code=204),
        ])
        outbox = CallbackOutbox(sender=sender, capacity=2, retry_delays=(0.0, 0.0))
        version = _version()
        self.assertTrue(outbox.enqueue("https://backend.example/events", version))
        self.assertTrue(outbox.wait_for_terminal(version.event_id, timeout=1.0))
        first_payloads = [payload for _, payload in sender.calls]
        self.assertEqual(len(first_payloads), 2)
        self.assertEqual({payload["event_id"] for payload in first_payloads}, {version.event_id})
        self.assertEqual(first_payloads[0], first_payloads[1])

        self.assertTrue(outbox.replay(version.event_id))
        self.assertTrue(outbox.wait_for_terminal(version.event_id, timeout=1.0))
        self.assertEqual(sender.calls[-1][1], first_payloads[0])
        outbox.stop(drain_budget_sec=0.2)

    def test_slow_sender_does_not_block_enqueue(self) -> None:
        gate = threading.Event()
        sender = _ControlledSender(gate=gate)
        outbox = CallbackOutbox(sender=sender, capacity=2, retry_delays=(0.0,))
        started = time.monotonic()
        self.assertTrue(outbox.enqueue("https://backend.example/events", _version()))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.05)
        self.assertTrue(sender.called.wait(timeout=0.2))
        gate.set()
        outbox.stop(drain_budget_sec=0.5)

    def test_capacity_policy_preserves_emergency_and_terminal_versions(self) -> None:
        sender = _ControlledSender()
        outbox = CallbackOutbox(
            sender=sender, capacity=2, retry_delays=(0.0,), autostart=False
        )
        regular = _version(level=1, episode_id="regular")
        emergency = _version(level=4, episode_id="emergency")
        terminal = _version(
            level=0, episode_id="terminal", kind="recovered", state="recovered"
        )
        self.assertTrue(outbox.enqueue("https://backend.example/events", regular))
        self.assertTrue(outbox.enqueue("https://backend.example/events", emergency))
        self.assertTrue(outbox.enqueue("https://backend.example/events", terminal))
        snapshot = outbox.snapshot()
        self.assertEqual(snapshot["depth"], 2)
        self.assertEqual(snapshot["queue_full_decisions"]["evicted_regular"], 1)
        self.assertEqual(snapshot["status_counts"]["failed"], 1)
        active_ids = {item["event_id"] for item in snapshot["active_items"]}
        self.assertEqual(active_ids, {emergency.event_id, terminal.event_id})
        outbox.stop(drain_budget_sec=0.0)

    def test_full_protected_queue_records_terminal_rejection(self) -> None:
        outbox = CallbackOutbox(
            sender=_ControlledSender(), capacity=2,
            retry_delays=(0.0,), autostart=False,
        )
        emergency = _version(level=4, episode_id="emergency")
        terminal_one = _version(
            level=0, episode_id="terminal-1", kind="recovered", state="recovered"
        )
        terminal_two = _version(
            level=0, episode_id="terminal-2", kind="unresolved", state="unresolved"
        )
        self.assertTrue(outbox.enqueue("https://backend.example/events", emergency))
        self.assertTrue(outbox.enqueue("https://backend.example/events", terminal_one))
        self.assertFalse(outbox.enqueue("https://backend.example/events", terminal_two))

        snapshot = outbox.snapshot()
        self.assertEqual(snapshot["depth"], 2)
        self.assertEqual(snapshot["rejected"], 1)
        self.assertEqual(snapshot["queue_full_decisions"]["rejected_protected"], 1)
        self.assertEqual(
            snapshot["last_queue_full_decision"]["incoming_event_id"],
            terminal_two.event_id,
        )
        outbox.stop(drain_budget_sec=0.0)

    def test_stop_returns_within_budget_and_reports_unfinished(self) -> None:
        gate = threading.Event()
        sender = _ControlledSender(gate=gate)
        outbox = CallbackOutbox(sender=sender, capacity=1, retry_delays=(0.0,))
        version = _version()
        outbox.enqueue("https://backend.example/events", version)
        self.assertTrue(sender.called.wait(timeout=0.2))
        started = time.monotonic()
        result = outbox.stop(drain_budget_sec=0.02)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.15)
        self.assertEqual(result["unfinished"], 1)
        self.assertEqual(result["rejected"], 0)
        self.assertTrue(result["worker_alive_after_budget"])
        gate.set()

    def test_unexpected_sender_error_becomes_final_failure(self) -> None:
        sender = _RaisingSender()
        outbox = CallbackOutbox(sender=sender, capacity=1, retry_delays=(0.0,))
        version = _version()
        self.assertTrue(outbox.enqueue("https://backend.example/events", version))
        self.assertTrue(outbox.wait_for_terminal(version.event_id, timeout=1.0))
        snapshot = outbox.snapshot()
        self.assertEqual(snapshot["final_failures"], 1)
        self.assertIn("RuntimeError", snapshot["last_error"])
        outbox.stop(drain_budget_sec=0.2)


if __name__ == "__main__":
    unittest.main()
