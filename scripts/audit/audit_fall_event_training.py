from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.fall_event_training_audit import (
    build_fall_event_training_audit,
    write_fall_event_training_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed P0 audit for fall-event model training."
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
        "--formal",
        type=Path,
        default=Path("reports/fall_risk/label_validation_formal_v2.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("reports/fall_risk/training-labels-v3-validation.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("configs/evaluation/fall_event_v1.provisional.yaml"),
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("configs/training/fall_event_v1.yaml"),
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=Path("reports/fall_risk/fall_event_proxy_v2_v3split/README.md"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "reports/reproducibility/fall_event_training_manifest.json"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/fall_risk/fall_event_training_audit.md"),
    )
    parser.add_argument(
        "--blockers-output",
        type=Path,
        default=Path("reports/fall_risk/fall_event_blockers.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_fall_event_training_audit(
            manifest=args.manifest,
            action_labels=args.action_labels,
            event_labels=args.event_labels,
            assignments=args.assignments,
            split=args.split,
            formal=args.formal,
            validation=args.validation,
            evaluation=args.evaluation,
            training=args.training,
            candidate_report=args.candidate_report,
        )
        write_fall_event_training_audit(
            report,
            manifest_output=args.manifest_output,
            audit_output=args.audit_output,
            blockers_output=args.blockers_output,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_id": report["manifest_id"],
                "p0_passed": report["p0_passed"],
                "status": report["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
