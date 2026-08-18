from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_training import (
    GAIT_TARGET_PROFILES,
    GaitWindowPreparationConfig,
    prepare_gait_window_dataset,
)


def _parse_action_ids(value: str) -> tuple[str, ...]:
    action_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(action_ids)) != len(action_ids):
        raise argparse.ArgumentTypeError("excluded action IDs must be unique")
    return action_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build gait-instability [T,V,C] windows from pose-quality JSONL."
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
        "--additional-pose-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional cleaned-pose directory; repeat for multiple fallback roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/gait_window_v1"),
    )
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--min-observed-frames", type=int, default=10)
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-windows-per-segment", type=int, default=4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.35)
    parser.add_argument(
        "--expand-real-context",
        action="store_true",
        help="Fill short windows with real same-track frames and emit label-span masks.",
    )
    parser.add_argument("--primary-min-labeled-observations", type=int, default=10)
    parser.add_argument("--weak-min-labeled-observations", type=int, default=5)
    parser.add_argument("--representation-min-labeled-observations", type=int, default=2)
    parser.add_argument("--weak-context-weight", type=float, default=0.35)
    parser.add_argument(
        "--target-profile",
        choices=GAIT_TARGET_PROFILES,
        default="gait_instability_b01_b04",
        help="Explicit gait target definition; observable profile separates B01 as a proxy.",
    )
    parser.add_argument(
        "--exclude-action-ids",
        type=_parse_action_ids,
        default=(),
        help="Comma-separated action IDs to exclude for a sensitivity dataset.",
    )
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
            manifest_path=args.manifest,
            assignments_path=args.assignments,
            split_report_path=args.split_report,
            additional_pose_dirs=args.additional_pose_dir,
            config=GaitWindowPreparationConfig(
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                min_observed_frames=args.min_observed_frames,
                require_usable_gait=not args.allow_insufficient_gait_quality,
                target_fps=args.target_fps,
                max_gap_sec=args.max_gap_sec,
                max_windows_per_segment=args.max_windows_per_segment,
                auxiliary_weight=args.auxiliary_weight,
                context_expansion=args.expand_real_context,
                primary_min_labeled_observations=(
                    args.primary_min_labeled_observations
                ),
                weak_min_labeled_observations=args.weak_min_labeled_observations,
                representation_min_labeled_observations=(
                    args.representation_min_labeled_observations
                ),
                weak_context_weight=args.weak_context_weight,
                target_profile=args.target_profile,
                excluded_action_ids=args.exclude_action_ids,
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
