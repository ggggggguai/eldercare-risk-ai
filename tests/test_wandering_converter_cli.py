from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    load_trajectory_jsonl,
)


ROOT = Path(__file__).parents[1]
SMARTCARE_SCRIPT = ROOT / "scripts" / "wandering" / "convert_smartcare.py"
WANDERING_PATTERNS_SCRIPT = (
    ROOT / "scripts" / "wandering" / "convert_wandering_patterns.py"
)


def _double_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = json.dumps(rows, separators=(",", ":"))
    path.write_text(json.dumps(inner), encoding="utf-8")


class WanderingConverterCliTest(unittest.TestCase):
    def test_smartcare_cli_builds_strict_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "smartcare"
            normal = [
                {"date": "normal", "x": value, "y": value, "stress": False}
                for value in (0, 1, 2, 3)
            ]
            wandering = [
                {"date": "wandering", "x": value, "y": value, "stress": True}
                for value in (10, 11, 10, 11)
            ]
            _double_json(source / "raw" / "dataset.json", normal + wandering)
            _double_json(
                source / "raw" / "dataset-validacao.json", normal + wandering
            )
            output = root / "converted"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SMARTCARE_SCRIPT),
                    "--source-root",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(load_trajectory_jsonl(output / "train_pool.jsonl")), 2)
            self.assertEqual(
                len(load_trajectory_jsonl(output / "official_validation.jsonl")), 2
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["sample_count_written"], 4)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("unshare"),
        "Linux unshare isolation is required",
    )
    def test_wandering_patterns_cli_uses_isolation_and_builds_strict_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "wandering_patterns"
            primary = source / "raw" / "patterns_dataset.pkl"
            duplicate = (
                source
                / "source_repository"
                / "model_data"
                / "patterns_dataset.pkl"
            )
            primary.parent.mkdir(parents=True)
            duplicate.parent.mkdir(parents=True)
            rows: list[dict[str, object]] = []
            for index, label in enumerate(
                ("direct", "pacing", "lapping", "random")
            ):
                x = np.asarray([0.0, 1.0, 2.0, 3.0]) + index
                y = np.asarray([0.0, 0.5, 0.0, -0.5])
                coords = np.stack((x, y), axis=1)
                coords_object = np.empty(len(coords), dtype=object)
                coords_object[:] = [point.tolist() for point in coords]
                rows.append(
                    {
                        "CartesianX": x,
                        "CartesianY": y,
                        "Coords": coords_object,
                        "Slope": np.zeros(4),
                        "Path_Efficiency": np.ones(4),
                        "Coords_Slope": np.column_stack((coords, np.zeros(4))),
                        "pattern": label,
                    }
                )
            frame = pd.DataFrame(rows)
            frame.columns = pd.Index(list(frame.columns), dtype=object)
            frame["pattern"] = frame["pattern"].astype(object)
            frame.to_pickle(primary, protocol=5)
            duplicate.write_bytes(primary.read_bytes())
            output = root / "converted"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(WANDERING_PATTERNS_SCRIPT),
                    "--source-root",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            samples = load_trajectory_jsonl(output / "samples.jsonl")
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                {sample.pattern_label.value for sample in samples},
                {"direct", "pacing", "lapping", "random"},
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["group_fields_found"], [])


if __name__ == "__main__":
    unittest.main()
