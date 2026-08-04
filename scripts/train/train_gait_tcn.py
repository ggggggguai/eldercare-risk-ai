from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_tcn import (
    GaitTCNTrainingConfig,
    train_gait_tcn,
)


def _parse_dilations(value: str) -> tuple[int, ...]:
    try:
        dilations = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dilations must be comma-separated integers") from exc
    if not dilations or any(item < 1 for item in dilations):
        raise argparse.ArgumentTypeError("dilations must contain positive integers")
    return dilations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight TCN on prepared [T,V,C] gait windows."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/fall_risk/gait_window_v1/dataset.npz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/fall_risk/gait_window_v1/tcn"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dilations", type=_parse_dilations, default=(1, 2, 4, 8))
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality-as-feature",
        action="store_true",
        help="Legacy ablation: expose continuous pose quality to the classifier.",
    )
    parser.add_argument(
        "--no-hierarchical-walking-gate",
        action="store_true",
        help="Disable the walking-gated conditional gait head.",
    )
    parser.add_argument("--walking-gate-loss-weight", type=float, default=0.25)
    parser.add_argument("--temporal-shift-frames", type=int, default=1)
    parser.add_argument("--keypoint-dropout-probability", type=float, default=0.05)
    parser.add_argument("--coordinate-jitter-std", type=float, default=0.005)
    parser.add_argument(
        "--no-pose-augmentation",
        action="store_true",
        help="Disable temporal shift, keypoint dropout, and coordinate jitter.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Transfer only the shared encoder from an action-pretraining checkpoint.",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Train the new binary head alone for this many initial epochs.",
    )
    parser.add_argument(
        "--partition-scheme",
        choices=("frozen", "fold_a", "fold_b"),
        default="frozen",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--no-mirror-augmentation", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the locked test partition after the candidate is frozen.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_gait_tcn(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            config=GaitTCNTrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_channels=args.hidden_channels,
                kernel_size=args.kernel_size,
                dilations=args.dilations,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                seed=args.seed,
                use_quality_as_feature=args.quality_as_feature,
                hierarchical_walking_gate=not args.no_hierarchical_walking_gate,
                walking_gate_loss_weight=args.walking_gate_loss_weight,
                temporal_shift_frames=(
                    0 if args.no_pose_augmentation else args.temporal_shift_frames
                ),
                keypoint_dropout_probability=(
                    0.0
                    if args.no_pose_augmentation
                    else args.keypoint_dropout_probability
                ),
                coordinate_jitter_std=(
                    0.0 if args.no_pose_augmentation else args.coordinate_jitter_std
                ),
                partition_scheme=args.partition_scheme,
                device=args.device,
                augment_mirror=not args.no_mirror_augmentation,
                evaluate_test=True if args.evaluate_test else None,
                pretrained_checkpoint=(
                    args.pretrained_checkpoint.as_posix()
                    if args.pretrained_checkpoint is not None
                    else None
                ),
                freeze_encoder_epochs=args.freeze_encoder_epochs,
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
