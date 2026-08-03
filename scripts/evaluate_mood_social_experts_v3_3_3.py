"""Run the read-only V3.3.3 five-expert evaluation audit (EVAL-001)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from elderly_monitoring.modules.mental_health.mood_social.evaluation import (
    MoodSocialEvaluationError,
    run_expert_evaluation,
)


DEFAULT_CONFIG = Path("configs/evaluation/mood_social_expert_audit_v3_3_3.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    config = (
        args.config.resolve()
        if args.config is not None
        else repository_root / DEFAULT_CONFIG
    )
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    try:
        result = run_expert_evaluation(
            repository_root,
            config,
            report_dir=args.report_dir,
            overwrite=args.overwrite,
            command=command,
        )
    except (MoodSocialEvaluationError, OSError, ValueError, RuntimeError) as exc:
        print(f"EVAL-001 failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
