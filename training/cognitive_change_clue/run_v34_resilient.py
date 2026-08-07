"""Run a V3.4 neural candidate and recover only documented AMP overflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .train_v34 import CANDIDATES, CONFIG_PATH, REPORT_ROOT, V34TrainingError
except ImportError:
    from train_v34 import CANDIDATES, CONFIG_PATH, REPORT_ROOT, V34TrainingError  # type: ignore[no-redef]


def _recoverable_failure(run_dir: Path, candidate_id: str) -> dict | None:
    diagnostics = sorted(
        run_dir.glob(f"fold_*/{candidate_id}/failure_diagnostic.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not diagnostics:
        return None
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    if not str(payload.get("error", "")).startswith("gradient is non-finite:"):
        return None
    checkpoint = payload.get("latest_checkpoint")
    if not checkpoint:
        return None
    return {"diagnostic": str(diagnostics[0]), "error": payload["error"]}


def run(args: argparse.Namespace) -> int:
    if args.candidate not in CANDIDATES or CANDIDATES[args.candidate].get("kind", "neural") != "neural":
        raise V34TrainingError("resilient runner only accepts neural V3.4 candidates")
    command = [
        sys.executable,
        "-m",
        "training.cognitive_change_clue.train_v34",
        "--config",
        str(args.config),
        "--candidate",
        args.candidate,
        "--run-id",
        args.run_id,
        "--folds",
        args.folds,
        "--device",
        args.device,
    ]
    run_dir = REPORT_ROOT / "runs" / args.run_id
    for attempt in range(args.max_amp_retries + 1):
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return 0
        recovery = _recoverable_failure(run_dir, args.candidate)
        if recovery is None or attempt >= args.max_amp_retries:
            return completed.returncode
        print(
            json.dumps(
                {
                    "status": "retrying_recorded_amp_overflow",
                    "attempt": attempt + 1,
                    **recovery,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-amp-retries", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
