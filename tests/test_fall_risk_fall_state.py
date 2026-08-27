import unittest

from elderly_monitoring.runtime.fall_state import (
    CrossEpochEpisodeError,
    EpisodeState,
    FallEpisodeStateMachine,
    FallStateConfig,
    FallStateDetector,
    InvalidEpisodeTransition,
)


def _pose(timestamp, hip_y, angle, center_y=None, quality=0.9, motion=0.0, geometry_valid=1.0):
    return {
        "timestamp_sec": timestamp,
        "hip_center_y": hip_y,
        "trunk_angle_deg": angle,
        "bbox_center_y": hip_y if center_y is None else center_y,
        "core_keypoint_quality": quality,
        "motion_score": motion,
        "trunk_geometry_valid": geometry_valid,
    }


class FallStateDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FallStateDetector(
            FallStateConfig(
                observation_window_sec=1.0,
                min_quality=0.6,
                hip_drop_threshold=0.18,
                center_drop_threshold=0.15,
                horizontal_angle_threshold=60.0,
                static_duration_sec=2.0,
                static_motion_threshold=0.02,
            )
        )

    def test_normal_or_tilt_only_does_not_trigger(self) -> None:
        self.detector.update(_pose(0.0, 0.4, 10))
        normal = self.detector.update(_pose(0.8, 0.45, 15))
        tilted = self.detector.update(_pose(1.0, 0.45, 75))
        self.assertEqual(normal.fall_event_score, 0.0)
        self.assertEqual(tilted.fall_event_score, 0.0)

    def test_drop_horizontal_and_quality_trigger_then_static(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        fall = self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.assertGreaterEqual(fall.fall_event_score, 0.8)
        self.detector.update(_pose(1.7, 0.62, 75, motion=0.01))
        static = self.detector.update(_pose(2.8, 0.62, 75, motion=0.01))
        self.assertGreaterEqual(static.long_static_score, 0.8)

    def test_low_quality_and_reset_do_not_reuse_old_event(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10))
        low = self.detector.update(_pose(0.7, 0.65, 80, quality=0.2))
        self.assertEqual(low.fall_event_score, 0.0)
        self.detector.reset()
        after_reset = self.detector.update(_pose(3.0, 0.65, 80, motion=0.0))
        self.assertEqual(after_reset.long_static_score, 0.0)

    def test_upright_motion_requires_confirmation_before_episode_recovery(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        fall = self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.assertTrue(fall.suspected_fall)

        recovery_candidate = self.detector.update(
            _pose(1.2, 0.45, 20, center_y=0.45, motion=0.08)
        )

        self.assertTrue(recovery_candidate.suspected_fall)
        self.assertGreaterEqual(recovery_candidate.fall_event_score, 0.8)

        recovered_signal = self.detector.update(
            _pose(4.3, 0.45, 20, center_y=0.45, motion=0.08)
        )

        self.assertFalse(recovered_signal.suspected_fall)
        self.assertEqual(recovered_signal.fall_event_score, 0.0)
        self.assertEqual(recovered_signal.long_static_score, 0.0)

    def test_low_quality_observation_does_not_advance_static_duration(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.detector.update(_pose(1.0, 0.62, 75, motion=0.01))

        unavailable = self.detector.update(
            _pose(9.0, 0.62, 75, quality=0.1, motion=0.0)
        )

        self.assertEqual(unavailable.long_static_score, 0.0)

    def test_small_post_fall_motion_does_not_reset_static_timer(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.detector.update(_pose(1.0, 0.62, 75, motion=0.03))

        static = self.detector.update(_pose(3.1, 0.62, 75, motion=0.03))

        self.assertGreaterEqual(static.static_duration_sec, 2.0)
        self.assertGreaterEqual(static.long_static_score, 0.8)

    def test_missing_trunk_geometry_cannot_trigger_recovery(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))

        still_suspected = self.detector.update(
            _pose(1.2, 0.45, 20, center_y=0.45, motion=0.08, geometry_valid=0.0)
        )

        self.assertTrue(still_suspected.suspected_fall)
        self.assertGreaterEqual(still_suspected.fall_event_score, 0.8)

    def test_hold_preserves_fall_signal_during_invalid_window(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.detector.update(_pose(1.0, 0.62, 75, motion=0.01))

        held = self.detector.hold(3.2)

        self.assertTrue(held.suspected_fall)
        self.assertEqual(held.long_static_score, 0.0)

    def test_invalid_hold_does_not_advance_static_duration(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.detector.update(_pose(1.0, 0.62, 75, motion=0.01))

        held = self.detector.hold(99.0)

        self.assertLess(held.static_duration_sec, 0.1)
        self.assertEqual(held.long_static_score, 0.0)

    def test_missing_trunk_geometry_cannot_trigger_fall(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))

        result = self.detector.update(
            _pose(0.7, 0.62, 75, center_y=0.60, geometry_valid=0.0)
        )

        self.assertFalse(result.suspected_fall)
        self.assertEqual(result.fall_event_score, 0.0)

    def test_out_of_order_timestamp_cannot_create_new_fall(self) -> None:
        self.detector.update(_pose(2.0, 0.35, 10, center_y=0.35))

        result = self.detector.update(
            _pose(1.0, 0.65, 80, center_y=0.60)
        )

        self.assertFalse(result.suspected_fall)
        self.assertEqual(result.fall_event_score, 0.0)


class FallEpisodeStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = FallEpisodeStateMachine(
            recovery_confirmation_sec=2.0,
            episode_ttl_sec=5.0,
        )
        self.machine.begin_epoch(3, source_time_sec=0.0, monotonic_sec=10.0)

    def test_full_episode_lifecycle_and_repeated_observation(self) -> None:
        suspected = self.machine.observe(
            risk_level=4,
            static_condition=False,
            observation_status="valid",
            source_time_sec=1.0,
            monotonic_sec=11.0,
            trigger_condition="fall_score_threshold",
        )
        self.assertEqual(suspected.to_state, EpisodeState.SUSPECTED)
        episode_id = suspected.episode_id
        self.assertIsNone(self.machine.observe(
            risk_level=4,
            static_condition=False,
            observation_status="valid",
            source_time_sec=1.5,
            monotonic_sec=11.5,
            trigger_condition="fall_score_threshold",
        ))

        confirmed = self.machine.observe(
            risk_level=4,
            static_condition=True,
            observation_status="valid",
            source_time_sec=3.0,
            monotonic_sec=13.0,
            trigger_condition="static_duration_threshold",
        )
        self.assertEqual(confirmed.to_state, EpisodeState.CONFIRMED_STATIC)
        self.assertEqual(confirmed.episode_id, episode_id)
        self.assertIsNone(self.machine.observe(
            risk_level=0,
            static_condition=False,
            observation_status="valid",
            source_time_sec=4.0,
            monotonic_sec=14.0,
            trigger_condition="normal_observation",
        ))
        recovered = self.machine.observe(
            risk_level=0,
            static_condition=False,
            observation_status="valid",
            source_time_sec=6.1,
            monotonic_sec=16.1,
            trigger_condition="normal_observation",
        )
        self.assertEqual(recovered.to_state, EpisodeState.RECOVERED)
        self.assertEqual(recovered.episode_id, episode_id)
        self.assertEqual(recovered.stream_epoch, 3)
        self.assertEqual(recovered.last_observed_source_time_sec, 6.1)
        self.assertEqual(recovered.ttl_sec, 5.0)
        self.assertEqual(recovered.trigger_condition, "recovery_confirmation_elapsed")
        self.assertIsNone(recovered.cleanup_reason)

    def test_unavailable_and_inference_error_do_not_create_edges(self) -> None:
        for status in ("unavailable", "inference_error"):
            with self.subTest(status=status):
                self.assertIsNone(self.machine.observe(
                    risk_level=4,
                    static_condition=True,
                    observation_status=status,
                    source_time_sec=1.0,
                    monotonic_sec=11.0,
                    trigger_condition="must_not_be_used",
                ))
                self.assertEqual(self.machine.state, EpisodeState.IDLE)

    def test_ttl_and_technical_cleanup_produce_unresolved_edges(self) -> None:
        first = self.machine.observe(
            risk_level=3,
            static_condition=False,
            observation_status="valid",
            source_time_sec=1.0,
            monotonic_sec=11.0,
            trigger_condition="risk_threshold",
        )
        self.assertIsNone(self.machine.observe(
            risk_level=0,
            static_condition=False,
            observation_status="unavailable",
            source_time_sec=9.0,
            monotonic_sec=12.0,
            trigger_condition="quality_failure",
        ))
        expired = self.machine.expire(monotonic_sec=16.1)
        self.assertEqual(expired.to_state, EpisodeState.UNRESOLVED)
        self.assertEqual(expired.episode_id, first.episode_id)
        self.assertEqual(expired.last_observed_source_time_sec, 1.0)
        self.assertEqual(expired.cleanup_reason, "observation_ttl_expired")

        self.machine.begin_epoch(4, source_time_sec=0.0, monotonic_sec=20.0)
        second = self.machine.observe(
            risk_level=2,
            static_condition=False,
            observation_status="valid",
            source_time_sec=2.0,
            monotonic_sec=22.0,
            trigger_condition="risk_threshold",
        )
        cleared = self.machine.clear(
            reason="stream_reconnected",
            source_time_sec=2.0,
            monotonic_sec=23.0,
        )
        self.assertEqual(cleared.to_state, EpisodeState.UNRESOLVED)
        self.assertEqual(cleared.episode_id, second.episode_id)
        self.assertEqual(cleared.cleanup_reason, "stream_reconnected")

    def test_illegal_and_cross_epoch_transitions_are_rejected(self) -> None:
        with self.assertRaises(InvalidEpisodeTransition):
            self.machine.transition_for_test(
                EpisodeState.CONFIRMED_STATIC,
                source_time_sec=1.0,
                monotonic_sec=11.0,
                trigger_condition="illegal_direct_confirmation",
            )
        with self.assertRaises(CrossEpochEpisodeError):
            self.machine.observe(
                risk_level=4,
                static_condition=False,
                observation_status="valid",
                source_time_sec=1.0,
                monotonic_sec=11.0,
                trigger_condition="wrong_epoch",
                stream_epoch=99,
            )

    def test_snapshot_exposes_bounded_transition_history(self) -> None:
        self.machine.observe(
            risk_level=1,
            static_condition=False,
            observation_status="valid",
            source_time_sec=1.0,
            monotonic_sec=11.0,
            trigger_condition="risk_threshold",
        )
        snapshot = self.machine.snapshot()
        self.assertEqual(snapshot["state"], "suspected")
        self.assertEqual(len(snapshot["transitions"]), 1)


if __name__ == "__main__":
    unittest.main()
