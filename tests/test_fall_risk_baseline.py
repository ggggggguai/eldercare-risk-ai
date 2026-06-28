import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.baseline import (
    BaselineModelConfig,
    build_personal_baselines,
    run_baseline_jsonl,
    score_baseline_deviation,
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
) -> dict[str, object]:
    return {
        "person_id": person_id,
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

        self.assertLessEqual(result["baseline_deviation_score"], 0.20)
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
        self.assertEqual(payload["model_version"], "fall-baseline-rule-v0.1")
        self.assertNotIn("risk_level", payload)
        self.assertNotIn("recommended_action", payload)
        self.assertNotIn("emergency_alert", payload)


if __name__ == "__main__":
    unittest.main()
