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

    def test_trigger_specific_cooldown_does_not_change_other_events(self) -> None:
        near_policy = EventPolicy(
            cooldown_sec=30.0,
            cooldown_by_trigger={"near_fall": 15.0},
        )
        self.assertTrue(near_policy.should_send(_event(3), monotonic_sec=0.0))
        self.assertFalse(near_policy.should_send(_event(3), monotonic_sec=14.9))
        self.assertTrue(near_policy.should_send(_event(3), monotonic_sec=15.0))

        fall_policy = EventPolicy(
            cooldown_sec=30.0,
            cooldown_by_trigger={"near_fall": 15.0},
        )
        fall = _event(4, trigger="fall_or_long_static", factors=["suspected_fall_event"])
        self.assertTrue(fall_policy.should_send(fall, monotonic_sec=0.0))
        self.assertFalse(fall_policy.should_send(fall, monotonic_sec=15.0))
        self.assertTrue(fall_policy.should_send(fall, monotonic_sec=30.0))

    def test_near_fall_cooldown_survives_episode_reset_and_forced_version(self) -> None:
        policy = EventPolicy(
            cooldown_sec=30.0,
            cooldown_by_trigger={"near_fall": 15.0},
        )
        first = policy.create_version(
            _event(3),
            monotonic_sec=0.0,
            episode_id="episode-1",
            stream_epoch=1,
            lifecycle_state="suspected",
            version_kind="initial",
            force=True,
        )
        policy.reset(preserve_trigger_cooldowns=True)
        suppressed = policy.create_version(
            _event(3),
            monotonic_sec=10.0,
            episode_id="episode-2",
            stream_epoch=1,
            lifecycle_state="suspected",
            version_kind="initial",
            force=True,
        )
        retained = policy.create_version(
            _event(3),
            monotonic_sec=15.0,
            episode_id="episode-2",
            stream_epoch=1,
            lifecycle_state="suspected",
            version_kind="initial",
            force=True,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(suppressed)
        self.assertIsNotNone(retained)

    def test_episode_versions_share_episode_and_use_distinct_event_ids(self) -> None:
        policy = EventPolicy(cooldown_sec=30.0)
        first = policy.create_version(
            _event(3), monotonic_sec=0.0, episode_id="episode-1",
            stream_epoch=7, lifecycle_state="suspected", version_kind="initial",
            force=True,
        )
        upgrade = policy.create_version(
            _event(4), monotonic_sec=1.0, episode_id="episode-1",
            stream_epoch=7, lifecycle_state="confirmed_static",
            version_kind="confirmed_static", force=True,
        )
        recovered = policy.create_version(
            _event(0, trigger="episode_recovered"), monotonic_sec=4.0,
            episode_id="episode-1", stream_epoch=7,
            lifecycle_state="recovered", version_kind="recovered", force=True,
        )
        unresolved = policy.create_version(
            _event(0, trigger="episode_unresolved"), monotonic_sec=5.0,
            episode_id="episode-1", stream_epoch=7,
            lifecycle_state="unresolved", version_kind="unresolved", force=True,
        )

        versions = (first, upgrade, recovered, unresolved)
        self.assertTrue(all(version is not None for version in versions))
        self.assertEqual({version.episode_id for version in versions}, {"episode-1"})
        self.assertEqual(len({version.event_id for version in versions}), 4)
        self.assertEqual(recovered.payload["risk_level"], 0)
        self.assertEqual(unresolved.payload["lifecycle_state"], "unresolved")


if __name__ == "__main__":
    unittest.main()
