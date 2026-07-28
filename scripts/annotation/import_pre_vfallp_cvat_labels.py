#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import (
    import_pre_vfallp_cvat_labels,
    write_pre_vfallp_cvat_labels,
)


DEFAULT_BATCH_ID = "pre_vfallp_dizziness_fall_forward_side"
DEFAULT_REDACTED_DIR = Path(
    "data/annotations/fall_risk/cvat_exports/raw"
) / DEFAULT_BATCH_ID
DEFAULT_OUTPUT_DIR = Path("data/annotations/fall_risk/generated/v2") / DEFAULT_BATCH_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Redact and import a user-authorized nested Pre_VFallp CVAT export "
            "as traceable v2 candidate JSONL."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--redacted-export-dir", type=Path, default=DEFAULT_REDACTED_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--action-output", type=Path, default=None)
    parser.add_argument("--event-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--labeler", default="cvat_pre_vfallp_import_20260722")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    action_output = args.action_output or args.output_dir / "action_labels.jsonl"
    event_output = args.event_output or args.output_dir / "event_labels.jsonl"
    report_output = args.report_output or args.output_dir / "import_report.json"
    try:
        imported = import_pre_vfallp_cvat_labels(
            args.input,
            manifest_path=args.manifest,
            redacted_export_dir=args.redacted_export_dir,
            labeler=args.labeler,
        )
        report = write_pre_vfallp_cvat_labels(
            imported,
            redacted_export_dir=args.redacted_export_dir,
            action_output_path=action_output,
            event_output_path=event_output,
            report_output_path=report_output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"authorization_id={report['authorization_id']}")
    print(f"redacted_task_exports={len(report['redacted_task_exports'])}")
    print(f"imported_action_labels={report['imported_action_labels']}")
    print(f"imported_event_labels={report['imported_event_labels']}")
    print(f"redacted_export_dir={args.redacted_export_dir}")
    print(f"action_output={action_output}")
    print(f"event_output={event_output}")
    print(f"report_output={report_output}")


if __name__ == "__main__":
    main()
