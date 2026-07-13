import unittest

import httpx

from elderly_monitoring.common.schemas import AlgorithmEvent
from elderly_monitoring.service.callback import CallbackSender


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


if __name__ == "__main__":
    unittest.main()
