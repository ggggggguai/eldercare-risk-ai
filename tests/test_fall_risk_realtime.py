import unittest
from types import SimpleNamespace
import threading
import time

from elderly_monitoring.common.schemas import AlgorithmEvent
from elderly_monitoring.runtime.event_policy import EventPolicy
from elderly_monitoring.runtime.feature_assembly import FeatureSnapshot
from elderly_monitoring.runtime.fall_state import FallEpisodeStateMachine
from elderly_monitoring.runtime.realtime_fall_risk import (
    FallRiskSessionEngine,
    RealtimeFallRiskEngine,
)
from elderly_monitoring.service.outbox import CallbackOutbox, DeliveryAttempt


class _Assembler:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.reset_called = False

    def add_pose(self, record, monotonic_sec):
        return self.snapshot

    def reset(self, **kwargs):
        self.reset_called = True

    def update_baseline_period(self, record):
        self.updated_baseline_period = record
        return {"baseline_state": "stable"} if record is not None else None


class _SequenceAssembler(_Assembler):
    def __init__(self, snapshots):
        super().__init__(None)
        self.snapshots = iter(snapshots)

    def add_pose(self, record, monotonic_sec):
        return next(self.snapshots)


class RealtimeEngineTest(unittest.TestCase):
    def test_completed_baseline_period_is_forwarded_to_assembler(self) -> None:
        assembler = _Assembler(FeatureSnapshot(features={}, quality_flags=[], usable=False))
        engine = RealtimeFallRiskEngine(assembler=assembler)

        result = engine.update_baseline_period({"period_id": "2026-07-12"})

        self.assertEqual(result["baseline_state"], "stable")
        self.assertEqual(assembler.updated_baseline_period["period_id"], "2026-07-12")

    def test_low_quality_does_not_fuse(self) -> None:
        assembler = _Assembler(FeatureSnapshot(features={}, quality_flags=["insufficient_pose_quality"], usable=False))
        engine = RealtimeFallRiskEngine(assembler=assembler, fusion_interval_sec=0.0)
        self.assertIsNone(engine.process_pose({}, monotonic_sec=1.0))

    def test_valid_features_generate_algorithm_event(self) -> None:
        features = {
            "person_id": "elder-1", "device_id": "cam-1", "scene_region": "home",
            "timestamp": "2026-07-12T12:00:00+08:00", "keypoint_quality": 0.9,
            "feature_coverage": 1.0, "gait_risk_score": 0.0, "sit_stand_risk_score": 0.0,
            "near_fall_event_score": 0.9, "baseline_deviation_score": 0.0,
            "scene_risk_score": 0.0, "activity_rhythm_score": 0.0,
            "fall_event_score": 0.0, "long_static_score": 0.0,
        }
        assembler = _Assembler(FeatureSnapshot(features=features, quality_flags=[], usable=True, urgent=True))
        engine = RealtimeFallRiskEngine(assembler=assembler, fusion_interval_sec=100.0)
        event = engine.process_pose({}, monotonic_sec=1.0)
        self.assertEqual(event.risk_level, 3)

    def test_window_reset_is_forwarded(self) -> None:
        assembler = _Assembler(None)
        engine = RealtimeFallRiskEngine(assembler=assembler)
        engine.reset_window()
        self.assertTrue(assembler.reset_called)
        self.assertIsNone(engine.last_fusion_duration_ms)

    def test_unavailable_snapshot_does_not_enter_fusion(self) -> None:
        assembler = _Assembler(
            FeatureSnapshot(
                features={"near_fall_event_score": None},
                quality_flags=["near_fall:insufficient_frames"],
                usable=False,
                branch_diagnostics={
                    "near_fall": {"status": "unavailable"}
                },
            )
        )
        engine = RealtimeFallRiskEngine(assembler=assembler, fusion_interval_sec=0.0)

        self.assertIsNone(engine.process_pose({}, monotonic_sec=1.0))
        self.assertEqual(
            engine.last_snapshot.branch_diagnostics["near_fall"]["status"],
            "unavailable",
        )

    def test_analysis_throttle_keeps_last_completed_snapshot(self) -> None:
        snapshot = FeatureSnapshot(
            features={},
            quality_flags=[],
            usable=False,
            branch_diagnostics={"gait": {"status": "valid", "score": 0.2}},
        )
        engine = RealtimeFallRiskEngine(
            assembler=_SequenceAssembler([snapshot, None]),
        )

        engine.process_pose({}, monotonic_sec=1.0)
        engine.process_pose({}, monotonic_sec=1.1)

        self.assertIs(engine.last_snapshot, snapshot)

    def test_unusable_completed_window_clears_current_event_without_erasing_history(self) -> None:
        valid = FeatureSnapshot(
            features={
                "person_id": "elder-1",
                "timestamp": "2026-08-27T12:00:00+08:00",
                "fall_event_score": 0.9,
                "long_static_score": 0.0,
            },
            quality_flags=[],
            usable=True,
            urgent=True,
        )
        unavailable = FeatureSnapshot(
            features={},
            quality_flags=["fall_state:insufficient_branch_keypoint_coverage"],
            usable=False,
        )
        engine = RealtimeFallRiskEngine(
            assembler=_SequenceAssembler([valid, unavailable]),
            fusion_interval_sec=0.0,
        )

        first = engine.process_pose({}, monotonic_sec=1.0)
        self.assertEqual(first.risk_level, 4)
        self.assertIs(engine.current_event, first)

        self.assertIsNone(engine.process_pose({}, monotonic_sec=2.0))
        self.assertIsNone(engine.current_event)


class _BlockingSender:
    def __init__(self):
        self.gate = threading.Event()
        self.called = threading.Event()
        self.payloads = []

    def send_once(self, callback_url, payload):
        self.payloads.append(dict(payload))
        self.called.set()
        self.gate.wait(timeout=1.0)
        return DeliveryAttempt(success=True, status_code=204)

    def close(self):
        pass


class _Tracker:
    primary_track_id = "track-1"
    target_diagnostics = {"state": "bound"}

    def process_frame(self, *args, **kwargs):
        pose = SimpleNamespace(to_dict=lambda: {})
        return SimpleNamespace(
            window_reset=False,
            target_reason="bound_track_continues",
            target_state="bound",
            target_changed=False,
            poses=[pose],
            primary_pose=pose,
            stage_timings_ms={},
        )

    def reset(self):
        pass

    def close(self):
        pass


class _Sampling:
    def observe(self, **kwargs):
        return None

    def reset(self, **kwargs):
        pass

    def snapshot(self):
        return {}


class _Frame:
    shape = (8, 8, 3)


def _risk_event(level=3):
    return AlgorithmEvent(
        module="fall_risk",
        person_id="elder-1",
        timestamp="2026-07-24T12:00:00+08:00",
        risk_level=level,
        risk_score=0.8 if level else 0.0,
        confidence=0.9,
        trigger_event="near_fall" if level else "normal",
        risk_factors=["near_fall_event"] if level else [],
        recommended_action="notify_guardian" if level else "record_only",
        model_version="test",
    )


def _session_engine(sender):
    instance = FallRiskSessionEngine.__new__(FallRiskSessionEngine)
    instance.session = SimpleNamespace(
        session_id="session-1",
        callback_url="https://backend.example/events",
    )
    instance.tracker = _Tracker()
    snapshot = FeatureSnapshot(
        features={"long_static_score": 0.0},
        quality_flags=[],
        usable=True,
        branch_diagnostics={"fall_state": {"status": "valid"}},
    )
    instance.engine = SimpleNamespace(
        last_snapshot=snapshot,
        last_fusion_duration_ms=0.1,
        process_pose=lambda record, monotonic_sec: _risk_event(),
        reset_window=lambda **kwargs: None,
    )
    instance.policy = EventPolicy(cooldown_sec=30.0)
    instance.outbox = CallbackOutbox(
        sender=sender, capacity=4, retry_delays=(0.0,)
    )
    instance.callback = sender
    instance.episode = FallEpisodeStateMachine(
        recovery_confirmation_sec=1.0, episode_ttl_sec=10.0
    )
    instance.episode.begin_epoch(1, source_time_sec=0.0, monotonic_sec=0.0)
    instance.outbox_drain_timeout_sec = 0.2
    instance._last_episode_event = None
    instance._last_source_time_sec = 0.0
    instance.frame_id = 0
    instance.primary_pose_count = 0
    instance.max_inference_fps = 0.0
    instance._last_inference_at = None
    instance.stream_epoch = 1
    instance._epoch_started_monotonic_sec = 0.0
    instance.sampling_monitor = _Sampling()
    instance.time_boundary_count = 0
    instance.last_time_boundary_reason = None
    instance.epoch_reset_history = []
    instance.last_frame_diagnostics = {}
    instance.close_diagnostics = None
    instance.assembler = SimpleNamespace(
        records=[], reset=lambda **kwargs: None, analysis_count=0
    )
    return instance


class FallRiskSessionEngineAsyncCallbackTest(unittest.TestCase):
    def test_process_frame_only_enqueues_when_http_attempt_is_blocked(self) -> None:
        sender = _BlockingSender()
        engine = _session_engine(sender)

        started = time.monotonic()
        engine.process_frame(
            _Frame(), source_pts_sec=1.0,
            received_monotonic_sec=1.0, stream_epoch=1,
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)
        self.assertTrue(sender.called.wait(timeout=0.2))
        self.assertEqual(engine.episode.state.value, "suspected")
        self.assertEqual(engine.outbox.snapshot()["capacity"], 4)
        sender.gate.set()
        engine.outbox.stop(drain_budget_sec=0.5)

    def test_epoch_switch_keeps_old_epoch_terminal_version_in_outbox(self) -> None:
        sender = _BlockingSender()
        engine = _session_engine(sender)
        engine.process_frame(
            _Frame(), source_pts_sec=1.0,
            received_monotonic_sec=1.0, stream_epoch=1,
        )
        self.assertTrue(sender.called.wait(timeout=0.2))

        engine.begin_stream_epoch(
            2, reason="stream_reconnected", started_monotonic_sec=2.0
        )
        snapshot = engine.outbox.snapshot()
        self.assertEqual(engine.episode.stream_epoch, 2)
        self.assertEqual(engine.episode.state.value, "idle")
        self.assertTrue(any(
            item["stream_epoch"] == 1
            and item["lifecycle_state"] == "unresolved"
            for item in snapshot["active_items"]
        ))
        sender.gate.set()
        engine.outbox.stop(drain_budget_sec=0.5)


if __name__ == "__main__":
    unittest.main()
