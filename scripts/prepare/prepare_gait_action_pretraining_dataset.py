from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_action_pretraining import (
    ActionPretrainingPreparationConfig,
    prepare_action_pretraining_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build non-test semantic action windows for TCN encoder pretraining."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/split.json"),
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/action_pretraining_v1"),
    )
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--min-observed-frames", type=int, default=10)
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-windows-per-segment", type=int, default=2)
    parser.add_argument("--auxiliary-weight", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_action_pretraining_dataset(
            args.labels,
            args.pose_dir,
            args.output_dir,
            manifest_path=args.manifest,
            assignments_path=args.assignments,
            split_report_path=args.split_report,
            config=ActionPretrainingPreparationConfig(
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                min_observed_frames=args.min_observed_frames,
                target_fps=args.target_fps,
                max_gap_sec=args.max_gap_sec,
                max_windows_per_segment=args.max_windows_per_segment,
                auxiliary_weight=args.auxiliary_weight,
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
