import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.baseline import (
    BaselineModelConfig,
    PersonalBaselineTracker,
    build_personal_baselines,
    run_baseline_jsonl,
    score_baseline_deviation,
    load_baseline_config,
)


def make_daily_record(
    day: int,
    *,
    person_id: str = "elder_001",
    track_id: int = 1,
    gait_speed: float = 0.42,
    center_speed_cv: float = 0.18,
    hip_lateral_sway: float = 0.010,
    sit_duration: float = 2.4,
    failed_attempts: int = 0,
    stabilization_time: float = 0.5,
    near_fall_count: int = 0,
    nighttime_count: int = 1,
    activity_volume: float = 100.0,
    scene_region: str = "living_room",
    quality: float = 0.9,
    camera_profile_id: str = "camera-home-1",
    monitoring_hours: float = 2.0,
) -> dict[str, object]:
    return {
        "record_type": "fall_baseline_period_features",
        "schema_version": "fall-baseline-period-features-v1",
        "person_id": person_id,
        "device_id": "device-home-1",
        "camera_profile_id": camera_profile_id,
        "period_id": f"2026-06-{day:02d}",
        "period_start": f"2026-06-{day:02d}T00:00:00+08:00",
        "period_end": f"2026-06-{day:02d}T23:59:59+08:00",
        "timezone": "Asia/Shanghai",
        "completed": True,
        "aggregation_version": "fall-baseline-period-aggregation-v1",
        "upstream_versions": {"gait": "gait-rule-test-v1"},
        "input_summary": {"source_record_count": 1},
        "valid_monitoring_hours": monitoring_hours,
        "track_id": track_id,
        "timestamp": f"2026-06-{day:02d}T10:00:00+08:00",
        "start_time": f"2026-06-{day:02d}T10:00:00+08:00",
        "end_time": f"2026-06-{day:02d}T10:30:00+08:00",
        "scene_region": scene_region,
        "gait_risk_score": 0.1,
        "gait_stability_features": {
            "mean_center_speed_norm_per_sec": gait_speed,
            "center_speed_cv": center_speed_cv,
            "hip_lateral_sway": hip_lateral_sway,
        },
        "sit_stand_risk_score": 0.1,
        "duration": sit_duration,
        "failed_attempts": failed_attempts,
        "stabilization_time": stabilization_time,
        "near_fall_event_count": near_fall_count,
        "near_fall_event_score": 0.0 if near_fall_count == 0 else 0.5,
        "nighttime_activity_count": nighttime_count,
        "activity_volume": activity_volume,
        "metric_quality": {
            metric: {
                "available": True,
                "observation_count": 1,
                "quality": quality,
                "coverage": quality,
                "exposure_hours": monitoring_hours,
                "missing_reason": None,
            }
            for metric in (
                "mean_gait_speed",
                "mean_sit_stand_duration",
                "near_fall_rate_per_hour",
                "nighttime_activity_rate_per_hour",
                "activity_volume",
            )
        },
        "quality_coverage": {
            "usable_frame_ratio": quality,
            "mean_core_keypoint_quality": quality,
            "gait_keypoint_coverage": quality,
            "sit_stand_keypoint_coverage": quality,
            "core_keypoint_coverage": quality,
            "insufficient_gait_quality": quality < 0.5,
            "insufficient_sit_stand_quality": quality < 0.5,
            "insufficient_near_fall_quality": quality < 0.5,
        },
    }


def history_records(*, person_id: str = "elder_001", track_id: int = 1) -> list[dict[str, object]]:
    return [
        make_daily_record(
            day,
            person_id=person_id,
            track_id=track_id,
            gait_speed=0.42 + ((day % 2) * 0.01),
            sit_duration=2.4 + ((day % 2) * 0.1),
            activity_volume=100.0 + ((day % 3) * 2.0),
        )
        for day in range(1, 8)
    ]


def config() -> BaselineModelConfig:
    return BaselineModelConfig(min_history_days=3, stable_history_days=7, min_history_records=3)


class FallRiskBaselineTest(unittest.TestCase):
    def test_stable_current_observation_has_low_deviation_score(self) -> None:
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, gait_speed=0.425, sit_duration=2.45, activity_volume=101.0)],
            baselines,
            config=config(),
        )

        self.assertLess(result["baseline_deviation_score"], 0.25)
        self.assertEqual(result["deviation_factors"], [])
        self.assertIn("baseline_features", result)
        self.assertIn("baseline_reference", result)
        self.assertNotIn("risk_level", result)
        self.assertNotIn("recommended_action", result)
        self.assertNotIn("emergency_alert", result)
        self.assertEqual(result["baseline_state"], "stable")
        self.assertTrue(result["available_metric_mask"]["mean_gait_speed"])
        self.assertEqual(result["history_cutoff"], "2026-06-08T00:00:00+08:00")

    def test_gait_speed_drop_from_personal_baseline_is_reported(self) -> None:
        baselines = build_personal_baselines(history_records(), config=config())
        stable = score_baseline_deviation([make_daily_record(8)], baselines, config=config())[0]

        [result] = score_baseline_deviation(
            [make_daily_record(8, gait_speed=0.25)],
            baselines,
            config=config(),
        )

        self.assertGreater(result["baseline_deviation_score"], stable["baseline_deviation_score"])
        self.assertIn("gait_speed_drop_from_baseline", result["deviation_factors"])

    def test_sit_stand_duration_increase_from_baseline_is_reported(self) -> None:
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, sit_duration=5.5, stabilization_time=1.8)],
            baselines,
            config=config(),
        )

        self.assertGreater(result["baseline_deviation_score"], 0.25)
        self.assertIn("sit_stand_duration_increase_from_baseline", result["deviation_factors"])

    def test_near_fall_frequency_increase_is_reported(self) -> None:
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, near_fall_count=3)],
            baselines,
            config=config(),
        )

        self.assertIn("near_fall_frequency_increase", result["deviation_factors"])
        self.assertGreater(result["baseline_deviation_score"], 0.20)

    def test_activity_and_scene_pattern_changes_are_reported(self) -> None:
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation(
            [
                make_daily_record(
                    8,
                    nighttime_count=5,
                    activity_volume=55.0,
                    scene_region="bathroom_door",
                )
            ],
            baselines,
            config=config(),
        )

        self.assertIn("nighttime_activity_increase", result["deviation_factors"])
        self.assertIn("activity_volume_drop", result["deviation_factors"])
        self.assertIn("scene_region_pattern_shift", result["deviation_factors"])

    def test_insufficient_history_does_not_create_false_high_deviation(self) -> None:
        baselines = build_personal_baselines([make_daily_record(1)], config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(2, gait_speed=0.10, sit_duration=7.0, near_fall_count=4)],
            baselines,
            config=config(),
        )

        self.assertIsNone(result["baseline_deviation_score"])
        self.assertEqual(result["baseline_state"], "cold")
        self.assertIn("insufficient_baseline_history", result["deviation_factors"])
        self.assertTrue(result["baseline_quality"]["insufficient_baseline_history"])

    def test_low_quality_data_is_downgraded_instead_of_high_risk(self) -> None:
        poor_history = [make_daily_record(day, quality=0.35) for day in range(1, 8)]
        baselines = build_personal_baselines(poor_history, config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, gait_speed=0.10, sit_duration=7.0, quality=0.35)],
            baselines,
            config=config(),
        )

        self.assertLessEqual(result["baseline_deviation_score"], 0.25)
        self.assertIn("reduced_baseline_quality", result["deviation_factors"])

    def test_future_and_current_periods_are_excluded_from_reference(self) -> None:
        history = history_records() + [
            make_daily_record(8, gait_speed=0.05),
            make_daily_record(9, gait_speed=0.05),
        ]
        baselines = build_personal_baselines(reversed(history), config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, gait_speed=0.42)], baselines, config=config()
        )

        self.assertEqual(result["baseline_reference"]["history_period_count"], 7)
        self.assertLess(
            result["baseline_reference"]["window_end"],
            result["baseline_features"]["period_start"],
        )

    def test_duplicate_period_is_deduplicated_and_conflict_is_rejected(self) -> None:
        duplicate = make_daily_record(1)
        baselines = build_personal_baselines(
            [duplicate, dict(duplicate), *history_records()[1:]], config=config()
        )
        [result] = score_baseline_deviation([make_daily_record(8)], baselines, config=config())
        self.assertEqual(result["baseline_reference"]["history_period_count"], 7)

        conflict = make_daily_record(1, gait_speed=0.1)
        with self.assertRaisesRegex(ValueError, "conflicting baseline period"):
            build_personal_baselines([duplicate, conflict], config=config())

    def test_timezone_boundary_uses_explicit_period_bounds(self) -> None:
        history = history_records()
        current = make_daily_record(8)
        current["period_start"] = "2026-06-07T16:00:00+00:00"
        current["period_end"] = "2026-06-08T15:59:59+00:00"
        baselines = build_personal_baselines(history, config=config())

        [result] = score_baseline_deviation([current], baselines, config=config())

        self.assertEqual(result["baseline_reference"]["history_period_count"], 7)
        self.assertEqual(result["history_cutoff"], "2026-06-07T16:00:00+00:00")

    def test_unknown_and_camera_mismatch_history_are_unavailable(self) -> None:
        unknown = [make_daily_record(day, person_id="unknown") for day in range(1, 8)]
        other_camera = [
            make_daily_record(day, camera_profile_id="camera-home-2") for day in range(1, 8)
        ]
        baselines = build_personal_baselines(unknown + other_camera, config=config())

        [result] = score_baseline_deviation(
            [make_daily_record(8, camera_profile_id="camera-home-1")],
            baselines,
            config=config(),
        )

        self.assertIsNone(result["baseline_deviation_score"])
        self.assertEqual(result["baseline_state"], "none")

    def test_pose_frames_are_not_accepted_as_period_features(self) -> None:
        pose_frame = {
            "person_id": "elder_001",
            "track_id": 1,
            "timestamp_sec": 1.0,
            "keypoints": [],
        }
        self.assertEqual(build_personal_baselines([pose_frame], config=config()), {})

    def test_missing_required_traceability_fields_fail_closed(self) -> None:
        period = make_daily_record(1)
        period.pop("input_summary")
        self.assertEqual(build_personal_baselines([period], config=config()), {})

        period = make_daily_record(1)
        period.pop("metric_quality")
        self.assertEqual(build_personal_baselines([period], config=config()), {})

    def test_missing_gait_keeps_other_metric_available(self) -> None:
        current = make_daily_record(8, sit_duration=5.5)
        current["gait_stability_features"] = {}
        current["metric_quality"]["mean_gait_speed"] = {
            "available": False,
            "observation_count": 0,
            "quality": 0.0,
            "coverage": 0.0,
            "exposure_hours": 2.0,
            "missing_reason": "no_valid_gait_window",
        }
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation([current], baselines, config=config())

        self.assertFalse(result["available_metric_mask"]["mean_gait_speed"])
        self.assertIsNone(result["metric_deviation_scores"]["mean_gait_speed"])
        self.assertTrue(result["available_metric_mask"]["mean_sit_stand_duration"])
        self.assertIn("sit_stand_duration_increase_from_baseline", result["deviation_factors"])

        missing_history = history_records()
        missing_history[0] = current
        missing_baseline = build_personal_baselines(missing_history, config=config())
        stats = missing_baseline["elder_001"]["camera_references"]["camera-home-1"][
            "metric_references"
        ]["mean_gait_speed"]
        self.assertEqual(stats["unavailable_period_count"], 1)
        self.assertEqual(stats["missing_reason_counts"], {"no_valid_gait_window": 1})

    def test_metric_below_quality_gate_is_unavailable(self) -> None:
        current = make_daily_record(8, gait_speed=0.1)
        current["metric_quality"]["mean_gait_speed"]["quality"] = 0.2
        baselines = build_personal_baselines(history_records(), config=config())

        [result] = score_baseline_deviation([current], baselines, config=config())

        self.assertFalse(result["available_metric_mask"]["mean_gait_speed"])
        self.assertIsNone(result["metric_deviation_scores"]["mean_gait_speed"])
        self.assertNotIn("gait_speed_drop_from_baseline", result["deviation_factors"])

    def test_event_counts_are_normalized_by_monitoring_exposure(self) -> None:
        history = [
            make_daily_record(day, near_fall_count=2, monitoring_hours=2.0)
            for day in range(1, 8)
        ]
        baselines = build_personal_baselines(history, config=config())

        [same_rate] = score_baseline_deviation(
            [make_daily_record(8, near_fall_count=4, monitoring_hours=4.0)],
            baselines,
            config=config(),
        )

        self.assertNotIn("near_fall_frequency_increase", same_rate["deviation_factors"])

    def test_robust_reference_resists_single_extreme_value(self) -> None:
        history = history_records()
        history[-1] = make_daily_record(7, gait_speed=4.2)
        baselines = build_personal_baselines(history, config=config())
        [result] = score_baseline_deviation([make_daily_record(8)], baselines, config=config())
        stats = result["baseline_reference"]["metric_references"]["mean_gait_speed"]

        self.assertLess(stats["median"], 0.5)
        self.assertLess(stats["winsorized_mean"], 1.0)
        self.assertLess(result["baseline_deviation_score"], 0.25)

    def test_tracker_freezes_slow_reference_during_drift_and_recovers(self) -> None:
        tracker = PersonalBaselineTracker(config=config())
        states = [tracker.update(record)["baseline_state"] for record in history_records()]
        self.assertEqual(states[:3], ["none", "cold", "cold"])
        self.assertEqual(states[3:7], ["initial", "initial", "initial", "initial"])

        first_bad = tracker.update(make_daily_record(8, gait_speed=0.15))
        slow_before = first_bad["baseline_reference"]["metric_references"]["mean_gait_speed"]["median"]
        second_bad = tracker.update(make_daily_record(9, gait_speed=0.14))
        third_bad = tracker.update(make_daily_record(10, gait_speed=0.13))
        slow_after = third_bad["baseline_reference"]["metric_references"]["mean_gait_speed"]["median"]

        self.assertEqual(second_bad["baseline_state"], "drift_suspected")
        self.assertEqual(third_bad["baseline_state"], "drift_suspected")
        self.assertEqual(slow_before, slow_after)
        self.assertTrue(third_bad["slow_reference_frozen"])

        recovery_1 = tracker.update(make_daily_record(11, gait_speed=0.42))
        recovery_2 = tracker.update(make_daily_record(12, gait_speed=0.42))
        stable = tracker.update(make_daily_record(13, gait_speed=0.42))
        self.assertEqual(recovery_1["baseline_state"], "recovery")
        self.assertEqual(recovery_2["baseline_state"], "recovery")
        self.assertEqual(stable["baseline_state"], "stable")
        self.assertFalse(stable["slow_reference_frozen"])

    def test_gradual_drift_accumulates_cusum_without_updating_slow_reference(self) -> None:
        tracker = PersonalBaselineTracker(config=config())
        for record in history_records():
            tracker.update(record)

        outputs = [
            tracker.update(make_daily_record(day, gait_speed=speed))
            for day, speed in zip(range(8, 13), (0.38, 0.36, 0.34, 0.32, 0.30))
        ]

        self.assertTrue(any(item["fast_state"]["cusum"] for item in outputs))
        self.assertEqual(outputs[-1]["baseline_state"], "drift_suspected")
        self.assertTrue(outputs[-1]["slow_reference_frozen"])

    def test_versioned_config_loads_provisional_thresholds(self) -> None:
        loaded = load_baseline_config()
        self.assertEqual(loaded.config_version, "fall-personal-baseline-config-v1")
        self.assertEqual(loaded.aggregation_period, "day")
        self.assertEqual(loaded.initial_fusion_weight, 0.25)

    def test_multiple_person_ids_are_modelled_independently(self) -> None:
        p1_history = history_records(person_id="elder_001", track_id=1)
        p2_history = [
            make_daily_record(day, person_id="elder_002", track_id=9, gait_speed=0.25)
            for day in range(1, 8)
        ]
        baselines = build_personal_baselines(p1_history + p2_history, config=config())

        results = score_baseline_deviation(
            [
                make_daily_record(8, person_id="elder_001", track_id=2, gait_speed=0.25),
                make_daily_record(8, person_id="elder_002", track_id=10, gait_speed=0.25),
            ],
            baselines,
            config=config(),
        )
        by_person = {result["person_id"]: result for result in results}

        self.assertIn("gait_speed_drop_from_baseline", by_person["elder_001"]["deviation_factors"])
        self.assertNotIn("gait_speed_drop_from_baseline", by_person["elder_002"]["deviation_factors"])
        self.assertLess(by_person["elder_002"]["baseline_deviation_score"], 0.25)

    def test_run_baseline_jsonl_reads_and_writes_deviation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "history.jsonl"
            current_path = Path(tmpdir) / "current.jsonl"
            output_path = Path(tmpdir) / "baseline.jsonl"
            baseline_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in history_records()),
                encoding="utf-8",
            )
            current_path.write_text(
                json.dumps(make_daily_record(8, gait_speed=0.25), ensure_ascii=False),
                encoding="utf-8",
            )

            count = run_baseline_jsonl(
                baseline_input_path=baseline_path,
                current_input_path=current_path,
                output_path=output_path,
                config=config(),
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("baseline_deviation_score", payload)
        self.assertIn("baseline_features", payload)
        self.assertIn("baseline_reference", payload)
        self.assertIn("deviation_factors", payload)
        self.assertEqual(payload["model_version"], "fall-personal-baseline-robust-v1")
        self.assertNotIn("risk_level", payload)
        self.assertNotIn("recommended_action", payload)
        self.assertNotIn("emergency_alert", payload)


if __name__ == "__main__":
    unittest.main()
