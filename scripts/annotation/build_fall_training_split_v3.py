#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    write_training_split_v3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shared leakage-safe split for fall-risk training labels v3."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
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
        "--assignments-output",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/split.json"),
    )
    parser.add_argument(
        "--seed",
        default="fall-risk-training-labels-v3-20260721",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_training_split_v3(
        manifest_path=args.manifest,
        action_labels_path=args.action_labels,
        event_labels_path=args.event_labels,
        assignments_path=args.assignments_output,
        report_path=args.report_output,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
