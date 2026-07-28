import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.gait import (
    GaitAnalysisConfig,
    extract_gait_windows,
    run_gait_jsonl,
)


GAIT_POINTS = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def make_gait_record(
    frame_id: int,
    *,
    center_x: float,
    center_y: float = 0.55,
    person_id: str = "elder_001",
    track_id: int = 1,
    left_swing: float = 0.02,
    right_swing: float = 0.02,
    usable_for_gait: bool = True,
    valid: bool = True,
    source: str = "observed",
    core_quality: float = 0.9,
    trunk_offset_x: float = 0.0,
    shoulder_width: float = 0.08,
) -> dict[str, object]:
    phase = math.sin(frame_id * math.pi / 2.0)
    left_ankle_x = center_x - 0.05 + (left_swing * phase)
    right_ankle_x = center_x + 0.05 - (right_swing * phase)
    point_values = {
        "left_shoulder": (
            center_x + trunk_offset_x - (shoulder_width / 2.0),
            center_y - 0.25,
        ),
        "right_shoulder": (
            center_x + trunk_offset_x + (shoulder_width / 2.0),
            center_y - 0.25,
        ),
        "left_hip": (center_x - 0.035, center_y),
        "right_hip": (center_x + 0.035, center_y),
        "left_knee": (center_x - 0.04, center_y + 0.17),
        "right_knee": (center_x + 0.04, center_y + 0.17),
        "left_ankle": (left_ankle_x, center_y + 0.34),
        "right_ankle": (right_ankle_x, center_y + 0.34),
    }
    return {
        "frame_id": frame_id,
        "timestamp_sec": round(frame_id * 0.1, 4),
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": "home",
        "core_keypoint_quality": core_quality,
        "keypoints": [
            {
                "name": name,
                "x": round(x, 4),
                "y": round(y, 4),
                "x_smooth": round(x, 4),
                "y_smooth": round(y, 4),
                "score": core_quality,
                "valid": valid,
                "source": source,
                "is_jump_outlier": False,
            }
            for name, (x, y) in point_values.items()
        ],
        "window_quality": {
            "mean_core_keypoint_quality": core_quality,
            "low_quality_frame_ratio": 0.0 if usable_for_gait else 1.0,
            "interpolated_point_ratio": 0.0,
            "jump_outlier_count": 0,
            "usable_for_gait": usable_for_gait,
            "usable_for_sit_stand": usable_for_gait,
            "usable_for_near_fall": usable_for_gait,
        },
    }


def stable_sequence() -> list[dict[str, object]]:
    return [make_gait_record(frame_id, center_x=0.30 + (0.01 * frame_id)) for frame_id in range(10)]


class FallRiskGaitTest(unittest.TestCase):
    def test_stable_gait_window_has_low_risk_and_feature_details(self) -> None:
        [window] = extract_gait_windows(stable_sequence(), config=GaitAnalysisConfig(window_frames=10))

        self.assertLess(window["gait_risk_score"], 0.25)
        self.assertFalse(window["quality_coverage"]["insufficient_gait_quality"])
        self.assertIn("center_speed_cv", window["gait_stability_features"])
        self.assertIn("ankle_motion_asymmetry", window["gait_stability_features"])
        self.assertEqual(
            {
                "gait_speed_norm_per_sec",
                "cadence_steps_per_min_proxy",
                "stride_length_norm_proxy",
                "left_right_asymmetry",
                "center_lateral_sway",
                "trunk_tilt_mean_deg",
                "path_deviation",
                "turn_instability_proxy",
            }
            - set(window["gait_stability_features"]),
            set(),
        )
        self.assertEqual(window["score_source"], "rule_fallback")
        self.assertEqual(window["fallback_reason"], "model_unavailable")
        self.assertEqual(window["fallback_score"], window["gait_risk_score"])
        self.assertIsNone(
            window["gait_stability_features"]["cadence_steps_per_min_proxy"]
        )
        self.assertNotIn("cadence_reduced", window["risk_factors"])
        self.assertEqual(window["risk_factors"], [])

    def test_tcn_score_is_primary_while_rules_remain_explanations(self) -> None:
        class Predictor:
            model_version = "gait-tcn-test"

            def predict_records(self, records):
                self.record_count = len(records)
                return 0.72

        predictor = Predictor()

        [window] = extract_gait_windows(
            stable_sequence(),
            config=GaitAnalysisConfig(window_frames=10),
            model_predictor=predictor,
        )

        self.assertEqual(window["gait_risk_score"], 0.72)
        self.assertEqual(window["model_score"], 0.72)
        self.assertLess(window["fallback_score"], 0.25)
        self.assertEqual(window["score_source"], "tcn")
        self.assertIsNone(window["fallback_reason"])
        self.assertEqual(window["model_version"], "gait-tcn-test")
        self.assertEqual(predictor.record_count, 10)

    def test_tcn_failure_uses_rule_fallback_and_reports_reason(self) -> None:
        class FailingPredictor:
            model_version = "gait-tcn-broken"

            def predict_records(self, records):
                raise RuntimeError("checkpoint is incompatible")

        [window] = extract_gait_windows(
            stable_sequence(),
            config=GaitAnalysisConfig(window_frames=10),
            model_predictor=FailingPredictor(),
        )

        self.assertEqual(window["score_source"], "rule_fallback")
        self.assertEqual(window["fallback_reason"], "model_inference_failed")
        self.assertIsNone(window["model_score"])
        self.assertEqual(window["gait_risk_score"], window["fallback_score"])

    def test_variable_center_speed_increases_risk(self) -> None:
        centers = [0.30, 0.31, 0.31, 0.36, 0.361, 0.42, 0.421, 0.48, 0.481, 0.54]
        records = [make_gait_record(frame_id, center_x=center) for frame_id, center in enumerate(centers)]

        [stable] = extract_gait_windows(stable_sequence(), config=GaitAnalysisConfig(window_frames=10))
        [unstable] = extract_gait_windows(records, config=GaitAnalysisConfig(window_frames=10))

        self.assertGreater(unstable["gait_risk_score"], stable["gait_risk_score"])
        self.assertIn("center_speed_instability", unstable["risk_factors"])

    def test_lower_limb_asymmetry_is_reported(self) -> None:
        records = [
            make_gait_record(frame_id, center_x=0.30 + (0.01 * frame_id), left_swing=0.035, right_swing=0.0)
            for frame_id in range(10)
        ]

        [window] = extract_gait_windows(records, config=GaitAnalysisConfig(window_frames=10))

        self.assertGreaterEqual(window["gait_stability_features"]["ankle_motion_asymmetry"], 0.5)
        self.assertIn("lower_limb_asymmetry", window["risk_factors"])

    def test_pause_or_hesitation_is_reported(self) -> None:
        centers = [0.30, 0.31, 0.32, 0.32, 0.32, 0.32, 0.33, 0.34, 0.35, 0.36]
        records = [make_gait_record(frame_id, center_x=center) for frame_id, center in enumerate(centers)]

        [window] = extract_gait_windows(records, config=GaitAnalysisConfig(window_frames=10))

        self.assertGreaterEqual(window["gait_stability_features"]["pause_frame_ratio"], 0.3)
        self.assertIn("pause_or_hesitation", window["risk_factors"])

    def test_longer_window_produces_bounded_cadence_proxy(self) -> None:
        records = [
            make_gait_record(frame_id, center_x=0.30 + (0.005 * frame_id))
            for frame_id in range(30)
        ]

        [window] = extract_gait_windows(
            records,
            config=GaitAnalysisConfig(window_frames=30),
        )

        cadence = window["gait_stability_features"][
            "cadence_steps_per_min_proxy"
        ]
        self.assertIsNotNone(cadence)
        self.assertGreaterEqual(cadence, 0.0)
        self.assertLessEqual(cadence, 240.0)

    def test_low_quality_window_is_marked_insufficient_without_false_high_risk(self) -> None:
        records = [
            make_gait_record(frame_id, center_x=0.30, usable_for_gait=False, valid=False, core_quality=0.2)
            for frame_id in range(8)
        ]

        [window] = extract_gait_windows(records, config=GaitAnalysisConfig(window_frames=8))

        self.assertEqual(window["gait_risk_score"], 0.0)
        self.assertTrue(window["quality_coverage"]["insufficient_gait_quality"])
        self.assertIn("insufficient_gait_quality", window["risk_factors"])
        self.assertEqual(window["score_source"], "unavailable")
        self.assertEqual(window["fallback_reason"], "insufficient_gait_quality")

    def test_multiple_tracks_are_analyzed_independently(self) -> None:
        records = stable_sequence() + [
            make_gait_record(
                frame_id,
                center_x=0.70 - (0.005 * frame_id),
                person_id="elder_002",
                track_id=2,
                left_swing=0.0,
                right_swing=0.03,
            )
            for frame_id in range(10)
        ]

        windows = extract_gait_windows(records, config=GaitAnalysisConfig(window_frames=10))
        keys = {(window["person_id"], window["track_id"]) for window in windows}

        self.assertEqual(keys, {("elder_001", 1), ("elder_002", 2)})

    def test_run_gait_jsonl_reads_and_writes_gait_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "poses_cleaned.jsonl"
            output_path = Path(tmpdir) / "gait.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in stable_sequence()),
                encoding="utf-8",
            )

            count = run_gait_jsonl(
                input_path=input_path,
                output_path=output_path,
                config=GaitAnalysisConfig(window_frames=10),
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("gait_risk_score", payload)
        self.assertIn("gait_stability_features", payload)
        self.assertIn("quality_coverage", payload)

    def test_cli_exposes_tcn_primary_branch_options(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/collect/run_fall_gait.py", "--help"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--tcn-checkpoint", completed.stdout)
        self.assertIn("--tcn-window-frames", completed.stdout)


if __name__ == "__main__":
    unittest.main()
