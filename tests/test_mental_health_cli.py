from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from elderly_monitoring.inference import run_features


CORE_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def pose_record(
    observed_at: str | None,
    *,
    person_id: str = "p01",
    frame_id: int = 1,
    x: float = 0.20,
) -> dict[str, object]:
    return {
        "person_id": person_id,
        "device_id": "cam-a",
        "frame_id": frame_id,
        "observed_at": observed_at,
        "timestamp_sec": float(frame_id - 1),
        "scene_region": "home",
        "keypoint_quality": 0.9,
        "keypoints": [
            {"name": name, "x": x, "y": 0.30, "score": 0.9}
            for name in CORE_KEYPOINTS
        ],
    }


def history_records(person_id: str = "p01") -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for day in range(1, 8):
        records.extend(
            [
                pose_record(
                    f"2026-07-{day:02d}T09:00:00+08:00",
                    person_id=person_id,
                    frame_id=day * 10,
                    x=0.20,
                ),
                pose_record(
                    f"2026-07-{day:02d}T09:00:02+08:00",
                    person_id=person_id,
                    frame_id=day * 10 + 1,
                    x=0.28,
                ),
            ]
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class MentalHealthCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.history_path = self.root / "history.jsonl"
        self.current_path = self.root / "current.jsonl"
        write_jsonl(self.history_path, history_records())
        write_jsonl(
            self.current_path,
            [
                pose_record("2026-07-08T09:00:00+08:00", frame_id=80, x=0.20),
                pose_record("2026-07-08T09:00:02+08:00", frame_id=81, x=0.21),
            ],
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "elderly_monitoring.inference.run_features",
                *arguments,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def daily_arguments(self) -> tuple[str, ...]:
        return (
            "--module",
            "mental_health",
            "--history-behavior",
            str(self.history_path),
            "--current-behavior",
            str(self.current_path),
        )

    def test_daily_cli_is_stable_sorted_and_only_outputs_mental_health_events(self) -> None:
        write_jsonl(
            self.history_path,
            history_records("p02") + history_records("p01"),
        )
        write_jsonl(
            self.current_path,
            [
                pose_record(
                    "2026-07-08T09:00:00+08:00",
                    person_id="p02",
                    frame_id=80,
                    x=0.20,
                ),
                pose_record(
                    "2026-07-08T09:00:00+08:00",
                    person_id="p01",
                    frame_id=80,
                    x=0.20,
                ),
                pose_record(
                    "2026-07-08T09:00:02+08:00",
                    person_id="p02",
                    frame_id=81,
                    x=0.21,
                ),
                pose_record(
                    "2026-07-08T09:00:02+08:00",
                    person_id="p01",
                    frame_id=81,
                    x=0.21,
                ),
            ],
        )
        first = self.run_cli(*self.daily_arguments())
        second = self.run_cli(*self.daily_arguments())

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        rows = [json.loads(line) for line in first.stdout.splitlines()]
        self.assertEqual(
            [(row["date"], row["person_id"]) for row in rows],
            sorted((row["date"], row["person_id"]) for row in rows),
        )
        self.assertEqual(
            set(rows[0]),
            {"person_id", "date", "daily_features", "baseline_features", "event"},
        )
        self.assertEqual(rows[0]["event"]["module"], "mental_health")
        self.assertEqual(rows[0]["event"]["timestamp"], "2026-07-08T09:00:02+08:00")

    def test_sleep_json_is_merged_without_replacing_missing_values_with_zero(self) -> None:
        sleep_path = self.root / "sleep.json"
        sleep_path.write_text(
            json.dumps(
                [
                    {
                        "person_id": "p01",
                        "date": f"2026-07-{day:02d}",
                        "sleep_onset_latency": 20.0 if day < 8 else 60.0,
                        "night_awakenings": 2 if day < 8 else 6,
                        "sleep_efficiency": 0.85 if day < 8 else None,
                        "quality_score": 0.9,
                    }
                    for day in range(1, 9)
                ]
            ),
            encoding="utf-8",
        )

        result = self.run_cli(
            *self.daily_arguments(),
            "--sleep",
            str(sleep_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout)
        self.assertEqual(row["daily_features"]["sleep_onset_latency"], 60.0)
        self.assertIsNone(row["daily_features"]["sleep_efficiency"])
        self.assertEqual(row["baseline_features"]["sleep_disturbance_score"], 1.0)

    def test_successful_output_path_contains_complete_jsonl_and_stdout_is_empty(self) -> None:
        output_path = self.root / "outputs" / "mental-health.jsonl"

        result = self.run_cli(
            *self.daily_arguments(),
            "--output",
            str(output_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        row = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(row["event"]["module"], "mental_health")

    def test_daily_cli_does_not_construct_or_call_fall_pipeline(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(
            run_features,
            "FallRiskPipeline",
            side_effect=AssertionError("fall pipeline must remain independent"),
        ), mock.patch("sys.stdout", stdout):
            exit_code = run_features.main(list(self.daily_arguments()))

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["event"]["module"], "mental_health")

    def test_mental_health_run_does_not_change_fall_event_fields(self) -> None:
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "features" / "fall_risk_sample.json"
        sample = run_features.load_sample(sample_path)
        before = run_features.FallRiskPipeline().predict_from_features(sample).to_dict()

        with mock.patch("sys.stdout", io.StringIO()):
            exit_code = run_features.main(list(self.daily_arguments()))
        after = run_features.FallRiskPipeline().predict_from_features(sample).to_dict()

        self.assertEqual(exit_code, 0)
        self.assertEqual(before, after)

    def test_self_report_can_trigger_independent_level_four_manual_review(self) -> None:
        self_report_path = self.root / "self-report.json"
        self_report_path.write_text(
            json.dumps(
                {
                    "person_id": "p01",
                    "date": "2026-07-08",
                    "manual_emergency_flag": True,
                }
            ),
            encoding="utf-8",
        )

        result = self.run_cli(
            *self.daily_arguments(),
            "--self-report",
            str(self_report_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        event = json.loads(result.stdout)["event"]
        self.assertEqual(event["module"], "mental_health")
        self.assertEqual(event["risk_level"], 4)
        self.assertEqual(event["recommended_action"], "manual_review")

    def test_evaluation_time_is_only_fallback_for_insufficient_data_without_observation(self) -> None:
        write_jsonl(
            self.current_path,
            [pose_record(None, frame_id=80)],
        )

        result = self.run_cli(
            *self.daily_arguments(),
            "--evaluation-time",
            "2026-07-08T18:30:00+08:00",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout)
        self.assertEqual(row["daily_features"]["end_time"], None)
        self.assertEqual(row["event"]["trigger_event"], "insufficient_data")
        self.assertEqual(row["event"]["timestamp"], "2026-07-08T18:30:00+08:00")

    def test_missing_event_and_evaluation_time_fails_without_output(self) -> None:
        write_jsonl(self.current_path, [pose_record(None, frame_id=80)])
        output_path = self.root / "result.jsonl"

        result = self.run_cli(
            *self.daily_arguments(),
            "--output",
            str(output_path),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evaluation_time", result.stderr)
        self.assertFalse(output_path.exists())

    def test_bad_record_reports_file_line_and_field_without_partial_output(self) -> None:
        self.current_path.write_text(
            json.dumps(pose_record("2026-07-08T09:00:00+08:00"))
            + "\n"
            + json.dumps({"frame_id": 2})
            + "\n",
            encoding="utf-8",
        )
        output_path = self.root / "result.jsonl"

        result = self.run_cli(
            *self.daily_arguments(),
            "--output",
            str(output_path),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.current_path), result.stderr)
        self.assertIn("line 2", result.stderr)
        self.assertIn("person_id", result.stderr)
        self.assertFalse(output_path.exists())

    def test_existing_single_feature_fall_cli_remains_compatible(self) -> None:
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "features" / "fall_risk_sample.json"

        result = self.run_cli(
            "--module",
            "fall_risk",
            "--input",
            str(sample_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["module"], "fall_risk")


if __name__ == "__main__":
    unittest.main()
