from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_logistic import (
    evaluate_sit_stand_rule,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the E0 sit-stand rule on provisional validation clips."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_sit_stand_rule(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
