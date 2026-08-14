from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_training_audit import (
    build_sit_stand_training_audit,
    write_sit_stand_training_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit sit-stand labels, protection groups and training gates."
    )
    parser.add_argument(
        "--action-labels",
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
        "--training-config",
        type=Path,
        default=Path("configs/training/sit_stand_event_v1.yaml"),
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation/sit_stand_event_v1.provisional.yaml"),
    )
    parser.add_argument(
        "--historical-report",
        type=Path,
        default=Path("reports/fall_risk/sit_stand_event_v1/README.md"),
    )
    parser.add_argument(
        "--event-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/sit_stand_event_labels_v1.jsonl"),
    )
    parser.add_argument(
        "--review-log",
        type=Path,
        default=Path("data/annotations/fall_risk/sit_stand_event_review_log_v1.jsonl"),
    )
    parser.add_argument(
        "--event-assignments",
        type=Path,
        default=Path("data/splits/fall_risk/sit_stand_event_v1/assignments.jsonl"),
    )
    parser.add_argument(
        "--event-split",
        type=Path,
        default=Path("data/splits/fall_risk/sit_stand_event_v1/split.json"),
    )
    parser.add_argument(
        "--write-p0-outputs",
        action="store_true",
        help="Write the fixed P0 manifest and Markdown reports; refuses overwrite.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("reports/reproducibility/sit_stand_training_manifest.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/fall_risk/sit_stand_training_audit.md"),
    )
    parser.add_argument(
        "--blockers-output",
        type=Path,
        default=Path("reports/fall_risk/sit_stand_training_blockers.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_sit_stand_training_audit(
            action_labels=args.action_labels,
            manifest=args.manifest,
            assignments=args.assignments,
            split=args.split,
            validation=args.validation,
            training_config=args.training_config,
            evaluation_config=args.evaluation_config,
            historical_report=args.historical_report,
            event_labels=args.event_labels,
            review_log=args.review_log,
            event_assignments=args.event_assignments,
            event_split=args.event_split,
        )
        if args.write_p0_outputs:
            write_sit_stand_training_audit(
                report,
                manifest_output=args.manifest_output,
                audit_output=args.audit_output,
                blockers_output=args.blockers_output,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
