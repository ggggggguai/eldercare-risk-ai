import unittest

from elderly_monitoring.modules.fall_risk.pipeline import FallRiskPipeline
from elderly_monitoring.runtime.feature_assembly import FeatureAssembler, FeatureAssemblyConfig
from elderly_monitoring.runtime.fall_state import FallStateConfig


def _record(frame_id: int, timestamp: float, quality: float = 0.9):
    names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    return {
        "frame_id": frame_id,
        "person_id": "elder-1",
        "track_id": 1,
        "scene_region": "bathroom",
        "timestamp_sec": timestamp,
        "bbox": [0.2, 0.1, 0.8, 0.9],
        "keypoint_quality": quality,
        "keypoints": [
            {"name": name, "x": 0.4 + (index % 2) * 0.1, "y": 0.2 + index * 0.08, "score": quality}
            for index, name in enumerate(names)
        ],
    }


def _baseline_period(day: int, *, activity_volume: float = 100.0):
    metrics = (
        "mean_gait_speed",
        "mean_sit_stand_duration",
        "near_fall_rate_per_hour",
        "nighttime_activity_rate_per_hour",
        "activity_volume",
    )
    return {
        "record_type": "fall_baseline_period_features",
        "schema_version": "fall-baseline-period-features-v1",
        "person_id": "elder-1",
        "device_id": "device-1",
        "camera_profile_id": "camera-home-1",
        "period_id": f"2026-06-{day:02d}",
        "period_start": f"2026-06-{day:02d}T00:00:00+08:00",
        "period_end": f"2026-06-{day:02d}T23:59:59+08:00",
        "timezone": "Asia/Shanghai",
        "completed": True,
        "aggregation_version": "fall-baseline-period-aggregation-v1",
        "upstream_versions": {"activity": "activity-test-v1"},
        "input_summary": {"source_record_count": 1},
        "valid_monitoring_hours": 2.0,
        "gait_stability_features": {"mean_center_speed_norm_per_sec": 0.42},
        "duration": 2.4,
        "near_fall_event_count": 0,
        "nighttime_activity_count": 1,
        "activity_volume": activity_volume,
        "metric_quality": {
            metric: {
                "available": True,
                "observation_count": 1,
                "quality": 0.9,
                "coverage": 0.9,
                "exposure_hours": 2.0,
                "missing_reason": None,
            }
            for metric in metrics
        },
    }


class FeatureAssemblyTest(unittest.TestCase):
    def test_completed_baseline_period_enters_main_fusion_path(self) -> None:
        history = [_baseline_period(day) for day in range(1, 11)]
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            baseline_history=history,
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )

        result = assembler.update_baseline_period(
            _baseline_period(11, activity_volume=20.0)
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["baseline_state"], "stable")
        self.assertIsNotNone(result["baseline_deviation_score"])

        snapshot = assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)

        self.assertEqual(snapshot.branch_diagnostics["baseline"]["status"], "valid")
        self.assertEqual(
            snapshot.features["baseline_deviation_score"],
            result["baseline_deviation_score"],
        )
        self.assertTrue(snapshot.features["fusion_mask"]["baseline_deviation_score"])
        self.assertFalse(snapshot.features["fusion_mask"]["activity_rhythm_score"])

    def test_pose_window_cannot_be_scored_as_completed_baseline_period(self) -> None:
        history = [{
            "record_type": "fall_baseline_period_features",
            "schema_version": "fall-baseline-period-features-v1",
            "person_id": "elder-1",
            "device_id": "device-1",
            "camera_profile_id": "camera-home-1",
            "period_id": f"2026-06-{day:02d}",
            "period_start": f"2026-06-{day:02d}T00:00:00+08:00",
            "period_end": f"2026-06-{day:02d}T23:59:59+08:00",
            "timezone": "Asia/Shanghai",
            "completed": True,
            "aggregation_version": "fall-baseline-period-aggregation-v1",
            "upstream_versions": {"activity": "activity-test-v1"},
            "input_summary": {"source_record_count": 1},
            "valid_monitoring_hours": 2.0,
            "activity_volume": 100.0,
            "baseline_quality": 0.9,
            "metric_quality": {
                "activity_volume": {
                    "available": True,
                    "observation_count": 1,
                    "quality": 0.9,
                    "coverage": 0.9,
                    "exposure_hours": 2.0,
                    "missing_reason": None,
                }
            },
        } for day in range(1, 8)]
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            baseline_history=history,
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )

        snapshot = assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)

        self.assertIsNone(snapshot.features["baseline_deviation_score"])
        self.assertFalse(snapshot.features["fusion_mask"]["baseline_deviation_score"])
        self.assertEqual(snapshot.branch_diagnostics["baseline"]["status"], "unavailable")
        self.assertIn(
            "completed_baseline_period_unavailable",
            snapshot.branch_diagnostics["baseline"]["reasons"],
        )

    def test_window_prunes_old_records_and_respects_interval(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="bathroom",
            scene_risk_scores={"bathroom": 0.7},
            config=FeatureAssemblyConfig(window_sec=10.0, analysis_interval_sec=0.5),
        )
        self.assertIsNotNone(assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0))
        self.assertIsNone(assembler.add_pose(_record(2, 0.2), monotonic_sec=0.2))
        snapshot = assembler.add_pose(_record(3, 0.6), monotonic_sec=0.6)
        self.assertIsNotNone(snapshot)
        assembler.add_pose(_record(4, 11.0), monotonic_sec=11.0)
        self.assertTrue(all(record["timestamp_sec"] >= 1.0 for record in assembler.records))

    def test_outputs_scene_and_insufficient_baseline_mark(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="bathroom",
            scene_risk_scores={"bathroom": 0.7},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )
        snapshot = assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)
        self.assertEqual(snapshot.features["scene_risk_score"], 0.7)
        self.assertIsNone(snapshot.features["baseline_deviation_score"])
        self.assertLess(snapshot.features["feature_coverage"], 1.0)
        self.assertIn("insufficient_baseline_history", snapshot.quality_flags)
        self.assertEqual(
            snapshot.branch_diagnostics["baseline"]["status"], "unavailable"
        )

    def test_short_window_uses_unavailable_instead_of_zero_risk(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )

        snapshot = assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)

        self.assertFalse(snapshot.usable)
        self.assertIsNone(snapshot.features["gait_risk_score"])
        self.assertIsNone(snapshot.features["near_fall_event_score"])
        self.assertEqual(snapshot.branch_diagnostics["gait"]["status"], "unavailable")
        self.assertIn(
            "insufficient_frames",
            snapshot.branch_diagnostics["gait"]["reasons"],
        )
        self.assertFalse(snapshot.features["fusion_mask"]["gait_risk_score"])

    def test_scene_risk_is_not_fused_without_behavioral_or_baseline_signal(self) -> None:
        from elderly_monitoring.modules.fall_risk.pipeline import FallRiskPipeline

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="bathroom",
            scene_risk_scores={"bathroom": 0.7},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )
        assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)
        snapshot = assembler.add_pose(_record(2, 0.1), monotonic_sec=0.1)

        self.assertTrue(snapshot.usable)
        self.assertEqual(snapshot.features["scene_risk_score"], 0.7)
        self.assertFalse(snapshot.features["fusion_mask"]["scene_risk_score"])
        self.assertEqual(
            FallRiskPipeline().predict_from_features(snapshot.features).risk_level,
            0,
        )

    def test_branch_exception_is_isolated_as_inference_error(self) -> None:
        class BrokenPredictor:
            model_version = "broken-gait-test"

            def predict_records(self, records):
                raise AssertionError("unexpected model failure")

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            gait_predictor=BrokenPredictor(),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.2),
                monotonic_sec=frame_id * 0.2,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(
            snapshot.branch_diagnostics["gait"]["status"], "inference_error"
        )
        self.assertIsNone(snapshot.features["gait_risk_score"])
        self.assertFalse(snapshot.features["fusion_mask"]["gait_risk_score"])
        self.assertTrue(snapshot.usable)

    def test_branch_diagnostics_include_window_quality_timing_and_versions(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.2),
                monotonic_sec=frame_id * 0.2,
            )

        gait = snapshot.branch_diagnostics["gait"]
        self.assertEqual(gait["status"], "valid")
        self.assertEqual(gait["input_frame_count"], 12)
        self.assertGreaterEqual(gait["effective_fps"], 4.0)
        self.assertEqual(gait["window_start_sec"], 0.0)
        self.assertEqual(gait["window_end_sec"], 2.2)
        self.assertIn("model_version", gait)
        self.assertIn("duration_ms", gait)
        self.assertIn("pose_quality", snapshot.stage_timings_ms)

    def test_unchanged_window_reuses_cached_snapshot(self) -> None:
        class Predictor:
            model_version = "cache-test"

            def __init__(self):
                self.calls = 0

            def predict_records(self, records):
                self.calls += 1
                return 0.4

        predictor = Predictor()
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            gait_predictor=predictor,
        )
        for frame_id in range(12):
            assembler.add_pose(
                _record(frame_id, frame_id * 0.2),
                monotonic_sec=frame_id * 0.2,
            )
        calls_before = predictor.calls

        first = assembler._assemble()
        second = assembler._assemble()

        self.assertIs(first, second)
        self.assertEqual(predictor.calls, calls_before)

    def test_reset_clears_window(self) -> None:
        assembler = FeatureAssembler(person_id="elder-1", scene_region="home", scene_risk_scores={})
        assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)
        assembler.reset()
        self.assertEqual(list(assembler.records), [])

    def test_configured_gait_predictor_is_used_by_runtime_assembly(self) -> None:
        class Predictor:
            model_version = "gait-tcn-test"

            def predict_records(self, records):
                return 0.73

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            gait_predictor=Predictor(),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["gait_risk_score"], 0.73)
        self.assertEqual(snapshot.features["gait_score_source"], "tcn")

    def test_configured_sit_stand_predictor_is_used_by_runtime_assembly(self) -> None:
        class Predictor:
            model_version = "sit-stand-tcn-test"

            def predict_records(self, records):
                return [{
                    "person_id": "elder-1",
                    "track_id": 1,
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "transition_type": "sit_to_stand",
                    "sit_stand_risk_score": 0.73,
                    "score_source": "tcn",
                    "model_version": self.model_version,
                    "risk_factors": ["experimental_sit_stand_tcn_event"],
                }]

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            sit_stand_predictor=Predictor(),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["sit_stand_risk_score"], 0.73)
        self.assertEqual(
            snapshot.branch_diagnostics["sit_stand"]["score_source"], "tcn"
        )
        self.assertEqual(
            snapshot.branch_diagnostics["sit_stand"]["model_version"],
            "sit-stand-tcn-test",
        )

    def test_configured_near_fall_predictor_is_used_by_runtime_assembly(self) -> None:
        class Predictor:
            model_version = "near-fall-tabular-test"

            def predict_records(self, records):
                return [{
                    "person_id": "elder-1",
                    "track_id": 1,
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "event_type": "balance_recovery",
                    "near_fall_event_score": 0.82,
                    "score_source": "tabular_rescorer",
                    "model_version": self.model_version,
                    "risk_factors": ["lateral_instability"],
                }]

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            near_fall_predictor=Predictor(),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["near_fall_event_score"], 0.82)
        self.assertEqual(
            snapshot.branch_diagnostics["near_fall"]["score_source"],
            "tabular_rescorer",
        )
        self.assertEqual(
            snapshot.branch_diagnostics["near_fall"]["model_version"],
            "near-fall-tabular-test",
        )

    def test_continuous_fall_tcn_shadow_is_diagnostic_only(self) -> None:
        class ShadowPredictor:
            model_version = "fall-event-continuous-tcn-test-shadow"

            def predict_records(self, records):
                return {
                    "fall_event_tcn_shadow_score": 0.83,
                    "fall_event_tcn_shadow_detected": True,
                    "fall_event_tcn_shadow_onset_frame": 18,
                    "fall_event_tcn_shadow_model_version": self.model_version,
                    "fall_event_tcn_shadow_status": "provisional_shadow",
                }

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            fall_event_predictor=ShadowPredictor(),
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["fall_event_tcn_shadow_score"], 0.83)
        self.assertTrue(snapshot.features["fall_event_tcn_shadow_detected"])
        self.assertEqual(
            snapshot.branch_diagnostics["fall_event_tcn_shadow"]["status"],
            "valid",
        )
        self.assertNotIn("fall_event_tcn_shadow_score", snapshot.features["fusion_mask"])

    def test_continuous_fall_tcn_can_drive_fall_event_with_rule_fallback(self) -> None:
        class RuntimePredictor:
            model_version = "fall-event-continuous-tcn-test-runtime"

            def predict_records(self, records):
                return {
                    "fall_event_tcn_score": 0.83,
                    "fall_event_tcn_detected": True,
                    "fall_event_tcn_onset_frame": 18,
                    "fall_event_tcn_model_version": self.model_version,
                    "fall_event_tcn_status": "valid",
                }

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            fall_state_config=FallStateConfig(
                hip_drop_threshold=0.0,
                center_drop_threshold=0.0,
                horizontal_angle_threshold=0.0,
            ),
            fall_event_predictor=RuntimePredictor(),
            fall_event_runtime_mode="experimental_tcn",
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["fall_event_tcn_score"], 0.83)
        self.assertEqual(snapshot.features["fall_event_score"], 0.9)
        self.assertEqual(
            snapshot.features["fall_event_score_source"], "continuous_tcn"
        )
        self.assertTrue(snapshot.features["fusion_mask"]["fall_event_score"])
        self.assertTrue(snapshot.urgent)
        event = FallRiskPipeline().predict_from_features(snapshot.features)
        self.assertEqual(event.risk_level, 4)
        self.assertEqual(event.metadata["fall_event_score_source"], "continuous_tcn")
        self.assertEqual(event.metadata["fall_event_tcn_score"], 0.83)

    def test_continuous_fall_tcn_does_not_promote_stable_sitting_without_kinematic_corroboration(self) -> None:
        class SittingFalsePositivePredictor:
            model_version = "fall-event-continuous-tcn-test-runtime"

            def predict_records(self, records):
                return {
                    "fall_event_tcn_score": 0.73,
                    "fall_event_tcn_detected": True,
                    "fall_event_tcn_onset_frame": 8,
                    "fall_event_tcn_model_version": self.model_version,
                    "fall_event_tcn_status": "valid",
                }

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            fall_event_predictor=SittingFalsePositivePredictor(),
            fall_event_runtime_mode="experimental_tcn",
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["fall_event_tcn_score"], 0.73)
        self.assertTrue(snapshot.features["fall_event_tcn_detected"])
        self.assertEqual(snapshot.features["fall_event_score"], 0.0)
        self.assertEqual(
            snapshot.features["fall_event_score_source"],
            "continuous_tcn_uncorroborated_rule_fallback",
        )
        self.assertEqual(
            snapshot.branch_diagnostics["fall_event_tcn"]["activation_status"],
            "suppressed",
        )
        self.assertEqual(
            snapshot.branch_diagnostics["fall_event_tcn"]["activation_reason"],
            "no_kinematic_fall_corroboration",
        )
        event = FallRiskPipeline().predict_from_features(snapshot.features)
        self.assertNotEqual(event.risk_level, 4)

    def test_continuous_fall_tcn_unavailable_uses_rule_fallback(self) -> None:
        class UnavailableRuntimePredictor:
            model_version = "fall-event-continuous-tcn-test-runtime"

            def predict_records(self, records):
                return {
                    "fall_event_tcn_score": None,
                    "fall_event_tcn_detected": False,
                    "fall_event_tcn_reason": "insufficient_observed_frames",
                    "fall_event_tcn_model_version": self.model_version,
                    "fall_event_tcn_status": "unavailable",
                }

        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="home",
            scene_risk_scores={},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
            fall_event_predictor=UnavailableRuntimePredictor(),
            fall_event_runtime_mode="experimental_tcn",
        )
        snapshot = None
        for frame_id in range(12):
            snapshot = assembler.add_pose(
                _record(frame_id, frame_id * 0.1),
                monotonic_sec=frame_id * 0.1,
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.features["fall_event_score"], 0.0)
        self.assertEqual(
            snapshot.features["fall_event_score_source"],
            "fall_state_rule_fallback",
        )
        self.assertEqual(
            snapshot.branch_diagnostics["fall_event_tcn"]["status"],
            "unavailable",
        )
        self.assertTrue(snapshot.features["fusion_mask"]["fall_event_score"])


if __name__ == "__main__":
    unittest.main()
