from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.fall_event_data_governance import (
    govern_fall_event_training_data,
)


DEFAULT_POSE_ROOTS = (
    Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"),
    Path("reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-bound train/validation continuous-fall supervision manifest; "
            "test labels and pose remain sealed."
        )
    )
    parser.add_argument(
        "--action-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--event-labels",
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
        "--governance-config",
        type=Path,
        default=Path("configs/data/fall_event_continuous_training_governance_v1.json"),
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        action="append",
        dest="pose_roots",
        help="Development pose root; repeat for multiple roots. Defaults include root cache and SCF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/fall_risk/fall_event_continuous_governance_v1"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = govern_fall_event_training_data(
            action_labels=args.action_labels,
            event_labels=args.event_labels,
            manifest=args.manifest,
            assignments=args.assignments,
            split=args.split,
            validation=args.validation,
            governance_config=args.governance_config,
            pose_roots=args.pose_roots or DEFAULT_POSE_ROOTS,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
