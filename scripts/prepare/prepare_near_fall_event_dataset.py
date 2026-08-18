from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.near_fall_training import (
    NearFallDatasetConfig,
    prepare_near_fall_event_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train/validation near-fall recovery-confirmation dataset. "
            "The command fails closed while the v3 near-fall data gate is blocked."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels_v3.jsonl"),
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
        "--split",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/split.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("reports/fall_risk/training-labels-v3-validation.json"),
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
        help="Additional cleaned pose directory; may be supplied more than once.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/near_fall_event_v2.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/near_fall_event_v2"),
    )
    return parser


def _load_dataset_config(path: Path) -> NearFallDatasetConfig:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("near-fall training config must be a mapping")
    if payload.get("schema_version") not in {
        "near-fall-event-training-config-v1",
        "near-fall-event-training-config-v2",
    }:
        raise ValueError("unsupported near-fall training config schema")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("near-fall training config is missing dataset")
    return NearFallDatasetConfig(**dataset)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_near_fall_event_dataset(
            labels=args.labels,
            manifest=args.manifest,
            assignments=args.assignments,
            split=args.split,
            validation=args.validation,
            pose_dir=args.pose_dir,
            additional_pose_dirs=args.additional_pose_dir,
            output_dir=args.output_dir,
            config=_load_dataset_config(args.config),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
