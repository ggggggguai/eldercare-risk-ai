import unittest

from elderly_monitoring.common.schemas import AlgorithmEvent
from elderly_monitoring.runtime.event_policy import EventPolicy


def _event(level: int, trigger: str = "near_fall", factors=None):
    return AlgorithmEvent(
        module="fall_risk", person_id="elder-1", timestamp="2026-07-12T12:00:00+08:00",
        risk_level=level, risk_score=0.8, confidence=0.9, trigger_event=trigger,
        risk_factors=factors or ["near_fall_event"], recommended_action="notify_guardian",
        model_version="test",
    )


class EventPolicyTest(unittest.TestCase):
    def test_normal_is_suppressed(self) -> None:
        self.assertFalse(EventPolicy().should_send(_event(0), monotonic_sec=0.0))

    def test_first_event_upgrade_and_cooldown(self) -> None:
        policy = EventPolicy(cooldown_sec=30.0)
        self.assertTrue(policy.should_send(_event(1), monotonic_sec=0.0))
        self.assertFalse(policy.should_send(_event(1), monotonic_sec=5.0))
        self.assertTrue(policy.should_send(_event(2), monotonic_sec=6.0))
        self.assertFalse(policy.should_send(_event(1), monotonic_sec=7.0))
        self.assertTrue(policy.should_send(_event(2), monotonic_sec=37.0))


if __name__ == "__main__":
    unittest.main()
