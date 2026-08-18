import unittest
from datetime import datetime, timezone

from elderly_monitoring.modules.fall_risk.environment import (
    low_light_score,
    water_exposure_score,
)
from elderly_monitoring.modules.fall_risk.pipeline import FallRiskPipeline
from elderly_monitoring.runtime.environment_store import (
    EnvironmentConflictError,
    EnvironmentStore,
)


def _payload(sequence=1, *, lux=18.0, water="dry"):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "device_id": "env-node-01",
        "boot_id": "boot-1",
        "boot_started_at": now,
        "sequence": sequence,
        "observed_at": now,
        "clock_status": "synced",
        "illumination_lux": lux,
        "illumination_status": "ok",
        "water_probes": {
            "bathroom_door": {"state": water, "sensor_status": "ok"}
        },
        "sensor_status": "ok",
    }


def _pose(timestamp, x=0.2, y=0.8):
    names = ["left_ankle", "right_ankle"]
    return {
        "timestamp_sec": timestamp,
        "keypoints": [
            {"name": name, "x": x, "y": y, "valid": True}
            for name in names
        ],
    }


class EnvironmentStoreTest(unittest.TestCase):
    def test_duplicate_and_conflict_are_distinguished(self):
        store = EnvironmentStore(capacity_per_device=2)
        payload = _payload()
        record, status = store.ingest(payload, received_monotonic_sec=10.0)
        self.assertEqual(status, "accepted")
        duplicate, status = store.ingest(payload, received_monotonic_sec=11.0)
        self.assertEqual(status, "duplicate")
        self.assertEqual(record.sequence, duplicate.sequence)
        changed = _payload()
        changed["illumination_lux"] = 99
        with self.assertRaises(EnvironmentConflictError):
            store.ingest(changed, received_monotonic_sec=11.0)

    def test_snapshot_is_causal_and_bounded(self):
        store = EnvironmentStore()
        store.ingest(_payload(), received_monotonic_sec=10.0)
        self.assertEqual(
            store.snapshot_for_frame("env-node-01", frame_received_monotonic_sec=9.9).reason,
            "causal_snapshot_missing",
        )
        snapshot = store.snapshot_for_frame("env-node-01", frame_received_monotonic_sec=10.5)
        self.assertTrue(snapshot.valid)
        self.assertEqual(snapshot.record.sequence, 1)


class EnvironmentFeatureTest(unittest.TestCase):
    def test_low_light_is_calibrated_and_clamped(self):
        self.assertEqual(
            low_light_score(10, normal_lux=100, severe_low_lux=0), 0.9
        )
        self.assertEqual(
            low_light_score(200, normal_lux=100, severe_low_lux=0), 0.0
        )
        self.assertIsNone(
            low_light_score(10, normal_lux=None, severe_low_lux=0)
        )

    def test_water_requires_repeated_roi_hits(self):
        result = water_exposure_score(
            [_pose(0.0), _pose(0.4), _pose(0.8)],
            {"bathroom_door": {"state": "water", "sensor_status": "ok"}},
            {"bathroom_door": {"version": "v1", "polygon": [[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]}},
        )
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.score, 1.0)

    def test_assist_cannot_create_level_four(self):
        sample = {
            "person_id": "p1",
            "gait_risk_score": 0.6,
            "fusion_mask": {"gait_risk_score": True},
            "environment_mask": {"environment_interaction_score": True},
            "environment_interaction_score": 1.0,
            "behavior_anchor": 0.6,
            "light_interaction_score": 0.6,
        }
        base = FallRiskPipeline().predict_from_features(sample)
        assisted = FallRiskPipeline(
            environment_mode="assist",
            environment_weight=0.2,
            environment_min_behavior_anchor=0.5,
            environment_policy_version="policy-v1",
        ).predict_from_features(sample)
        self.assertEqual(base.risk_score, 0.6)
        self.assertEqual(assisted.risk_score, 0.8)
        self.assertEqual(assisted.risk_level, 3)
        self.assertNotEqual(assisted.model_version, base.model_version)
        self.assertEqual(assisted.confidence, base.confidence)


if __name__ == "__main__":
    unittest.main()
