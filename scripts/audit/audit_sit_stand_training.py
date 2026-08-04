from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_training import (
    audit_sit_stand_training_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit sit-stand labels, protection groups and training gates."
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
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_sit_stand_training_inputs(
            labels=args.labels,
            manifest=args.manifest,
            assignments=args.assignments,
            split=args.split,
            validation=args.validation,
            pose_dir=args.pose_dir,
        )
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(
                    f"sit-stand audit output already exists: {args.output}"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
