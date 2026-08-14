from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.fall_event_tcn import (
    evaluate_fall_event_candidate_tcn,
    evaluate_fall_event_candidate_ensemble,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a fall-event candidate TCN on validation only."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint path; repeat for mean-probability ensemble evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if len(args.checkpoint) == 1:
            if args.threshold is not None:
                raise ValueError(
                    "--threshold is only supported for ensemble evaluation"
                )
            report = evaluate_fall_event_candidate_tcn(
                args.data,
                args.checkpoint[0],
                args.output,
                metadata_path=args.metadata,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            report = evaluate_fall_event_candidate_ensemble(
                args.data,
                args.checkpoint,
                args.output,
                metadata_path=args.metadata,
                device=args.device,
                batch_size=args.batch_size,
                threshold=(0.5 if args.threshold is None else args.threshold),
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
