#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.training_labels_v3 import (
    write_training_label_migration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate v2 fall-risk labels into model-training v3 labels."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--action-labels-v2",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels.jsonl"),
    )
    parser.add_argument(
        "--event-labels-v2",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels.jsonl"),
    )
    parser.add_argument(
        "--action-labels-v3",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--event-labels-v3",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fall_risk/training-labels-v3-migration.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_training_label_migration(
        manifest_path=args.manifest,
        action_labels_v2_path=args.action_labels_v2,
        event_labels_v2_path=args.event_labels_v2,
        action_labels_v3_path=args.action_labels_v3,
        event_labels_v3_path=args.event_labels_v3,
        report_path=args.report_output,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
