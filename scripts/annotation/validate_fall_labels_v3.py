#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    write_training_label_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model-training fall-risk action/event labels v3."
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
        "--action-schema",
        type=Path,
        default=Path("configs/data/fall_risk_action_label_schema_v3.json"),
    )
    parser.add_argument(
        "--event-schema",
        type=Path,
        default=Path("configs/data/fall_risk_event_label_schema_v3.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fall_risk/training-labels-v3-validation.json"),
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/split.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_training_label_validation_report(
        report_path=args.report_output,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
        action_labels_path=args.action_labels,
        event_labels_path=args.event_labels,
        action_schema_path=args.action_schema,
        event_schema_path=args.event_schema,
        split_assignments_path=args.split_assignments,
        split_report_path=args.split_report,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
