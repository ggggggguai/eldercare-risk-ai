from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotation_validation import (
    validate_fall_risk_data,
    write_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate fall-risk labels and formal-evaluation eligibility."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--action-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels.jsonl"),
    )
    parser.add_argument(
        "--event-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels.jsonl"),
    )
    parser.add_argument(
        "--risk-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/risk_labels.jsonl"),
    )
    parser.add_argument(
        "--subject-profiles",
        type=Path,
        default=Path("data/annotations/fall_risk/subject_profiles.json"),
    )
    parser.add_argument(
        "--review-log",
        type=Path,
        default=Path("data/annotations/fall_risk/annotation_review_log.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/fall_risk_label_validation_v1.yaml"),
    )
    parser.add_argument("--mode", choices=("audit", "formal"), default="audit")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fall_risk/annotation_validation.json"),
    )
    args = parser.parse_args()
    try:
        report = validate_fall_risk_data(
            manifest_path=args.manifest,
            action_labels_path=args.action_labels,
            event_labels_path=args.event_labels,
            risk_labels_path=args.risk_labels,
            subject_profiles_path=args.subject_profiles,
            review_log_path=args.review_log,
            mode=args.mode,
            config_path=args.config,
        )
        write_validation_report(report, args.report_output, overwrite=False)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    counts = report["counts"]
    print(
        f"valid={str(report['valid']).lower()} errors={counts['errors']} "
        f"blockers={counts['blockers']} warnings={counts['warnings']}"
    )
    print(f"report_output={args.report_output}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
