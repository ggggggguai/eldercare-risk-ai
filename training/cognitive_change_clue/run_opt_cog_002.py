"""Execute the complete OPT-COG-002 experiment sequence without GPU overlap."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .train_v34 import REPORT_ROOT, V34TrainingError
except ImportError:
    from train_v34 import REPORT_ROOT, V34TrainingError  # type: ignore[no-redef]


DEFAULT_RUNS = {
    "V34-A0": "COG-20260805-001",
    "V34-A1": "COG-20260805-002",
    "V34-A2": "COG-20260806-001",
    "V34-A3-Audio": "COG-20260806-002",
    "V34-A3-Text": "COG-20260806-003",
    "V34-Face": "COG-20260806-004",
    "V34-A3": "COG-20260806-005",
}


def _execute(command: list[str]) -> None:
    print(json.dumps({"status": "starting", "command": command}, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise V34TrainingError(
            f"OPT-COG-002 subprocess failed rc={completed.returncode}: {' '.join(command)}"
        )


def run(args: argparse.Namespace) -> dict:
    runs = {key: value for key, value in DEFAULT_RUNS.items()}
    neural = ("V34-A1", "V34-A2", "V34-A3-Audio", "V34-A3-Text", "V34-Face")
    for candidate_id in neural:
        _execute(
            [
                sys.executable,
                "-m",
                "training.cognitive_change_clue.run_v34_resilient",
                "--candidate",
                candidate_id,
                "--run-id",
                runs[candidate_id],
                "--device",
                args.device,
            ]
        )
    _execute(
        [
            sys.executable,
            "-m",
            "training.cognitive_change_clue.v34_late_fusion",
            "--run-id",
            runs["V34-A3"],
            "--audio-run",
            runs["V34-A3-Audio"],
            "--text-run",
            runs["V34-A3-Text"],
            "--device",
            args.device,
        ]
    )
    selection = REPORT_ROOT / "OPT-COG-002_selection_report.json"
    _execute(
        [
            sys.executable,
            "-m",
            "training.cognitive_change_clue.select_v34_backbone",
            "--v34_a0-run",
            runs["V34-A0"],
            "--v34_a1-run",
            runs["V34-A1"],
            "--v34_a2-run",
            runs["V34-A2"],
            "--v34_a3-run",
            runs["V34-A3"],
            "--face-run",
            runs["V34-Face"],
            "--output",
            str(selection),
        ]
    )
    audit_path = REPORT_ROOT / "OPT-COG-002_run_audit.json"
    _execute(
        [
            sys.executable,
            "-m",
            "training.cognitive_change_clue.audit_opt_cog_002",
            "--selection",
            str(selection),
            "--output",
            str(audit_path),
        ]
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {"task_id": "OPT-COG-002", "status": audit["status"], "runs": runs, "audit": str(audit_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
