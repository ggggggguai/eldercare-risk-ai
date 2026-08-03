from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "wandering"
    / "_extract_wandering_patterns_pickle.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(*, include_group: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, label in enumerate(("direct", "pacing", "lapping", "random")):
        x = np.asarray([0.0, 1.0, 2.0, 3.0]) + index
        y = np.asarray([0.0, 0.5, 0.0, -0.5])
        coords = np.stack((x, y), axis=1)
        coords_object = np.empty(len(coords), dtype=object)
        coords_object[:] = [point.tolist() for point in coords]
        row: dict[str, object] = {
            "CartesianX": x,
            "CartesianY": y,
            "Coords": coords_object,
            "Slope": np.zeros(4),
            "Path_Efficiency": np.ones(4),
            "Coords_Slope": np.column_stack((coords, np.zeros(4))),
            "pattern": label,
        }
        if include_group:
            row["participant_id"] = f"p{index % 2}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.columns = pd.Index(list(frame.columns), dtype=object)
    frame["pattern"] = frame["pattern"].astype(object)
    if include_group:
        frame["participant_id"] = frame["participant_id"].astype(object)
    return frame


class WanderingPatternsPickleExtractorTest(unittest.TestCase):
    def _run(self, pickle_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(pickle_path),
                "--output-jsonl",
                str(output_dir / "rows.jsonl"),
                "--output-metadata",
                str(output_dir / "metadata.json"),
                "--expected-sha256",
                _sha256(pickle_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "WANDERING_PICKLE_ISOLATED": "1"},
        )

    def test_extracts_only_inert_coordinates_labels_and_proven_group_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pickle_path = root / "patterns_dataset.pkl"
            _frame(include_group=True).to_pickle(pickle_path, protocol=5)
            output_dir = root / "extract"
            output_dir.mkdir()

            completed = self._run(pickle_path, output_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["record_count"], 4)
            self.assertEqual(metadata["source_sha256"], _sha256(pickle_path))
            self.assertEqual(metadata["group_fields_found"], ["participant_id"])
            self.assertTrue(metadata["pickle_globals_loaded"])
            rows = [
                json.loads(line)
                for line in (output_dir / "rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["record_index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                {row["pattern"] for row in rows},
                {"direct", "pacing", "lapping", "random"},
            )
            self.assertEqual(rows[0]["group_values"], {"participant_id": "p0"})
            self.assertEqual(rows[0]["points"][1], [1.0, 0.5])

    def test_forbidden_pickle_global_is_rejected_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "should-not-exist"

            class Exploit:
                def __reduce__(self) -> tuple[object, tuple[str]]:
                    return os.system, (f"touch {marker}",)

            pickle_path = root / "malicious.pkl"
            with pickle_path.open("wb") as handle:
                pickle.dump(Exploit(), handle, protocol=5)
            output_dir = root / "extract"
            output_dir.mkdir()

            completed = self._run(pickle_path, output_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("forbidden pickle global", completed.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((output_dir / "rows.jsonl").exists())
            self.assertFalse((output_dir / "metadata.json").exists())

    def test_hash_mismatch_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pickle_path = root / "patterns_dataset.pkl"
            _frame().to_pickle(pickle_path, protocol=5)
            output_dir = root / "extract"
            output_dir.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(pickle_path),
                    "--output-jsonl",
                    str(output_dir / "rows.jsonl"),
                    "--output-metadata",
                    str(output_dir / "metadata.json"),
                    "--expected-sha256",
                    "0" * 64,
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "WANDERING_PICKLE_ISOLATED": "1"},
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SHA-256 mismatch", completed.stderr)
            self.assertFalse((output_dir / "rows.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
