from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_training import (
    GaitWindowPreparationConfig,
    prepare_gait_window_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build gait-instability [T,V,C] windows from pose-quality JSONL."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels.jsonl"),
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("data/processed/fall_risk/gait_pose_quality"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/gait_window_v1"),
    )
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--stride-frames", type=int, default=32)
    parser.add_argument("--min-observed-frames", type=int, default=12)
    parser.add_argument("--allow-insufficient-gait-quality", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-search-iterations", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_gait_window_dataset(
            args.labels,
            args.pose_dir,
            args.output_dir,
            config=GaitWindowPreparationConfig(
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                min_observed_frames=args.min_observed_frames,
                require_usable_gait=not args.allow_insufficient_gait_quality,
                seed=args.seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                split_search_iterations=args.split_search_iterations,
            ),
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
