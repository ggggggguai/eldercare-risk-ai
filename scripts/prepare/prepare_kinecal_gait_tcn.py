from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.kinecal_gait import (
    KinecalGaitPreparationConfig,
    prepare_kinecal_gait_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare KINECAL 3 m walk skeletons as leakage-safe gait TCN tensors."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external/kinecal/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/kinecal_gait_tcn"),
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.10)
    parser.add_argument("--window-frames", type=int, default=128)
    parser.add_argument("--stride-frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_kinecal_gait_dataset(
            args.input_dir,
            args.output_dir,
            config=KinecalGaitPreparationConfig(
                target_fps=args.target_fps,
                max_gap_sec=args.max_gap_sec,
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                seed=args.seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
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
