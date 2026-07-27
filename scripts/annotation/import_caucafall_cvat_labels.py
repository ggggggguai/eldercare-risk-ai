from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.caucafall import (
    import_caucafall_cvat_labels,
    write_caucafall_cvat_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redact and import the nested manual CaucaFall CVAT export."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--redacted-export-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/cvat_exports/raw/caucafall_manual"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/caucafall_manual"),
    )
    parser.add_argument(
        "--labeler",
        default="caucafall_cvat_manual_20260723",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    imported = import_caucafall_cvat_labels(
        args.input,
        manifest_path=args.manifest,
        redacted_export_dir=args.redacted_export_dir,
        labeler=args.labeler,
        overwrite=args.overwrite,
    )
    report = write_caucafall_cvat_labels(
        imported,
        action_output_path=args.output_dir / "action_labels.jsonl",
        event_output_path=args.output_dir / "event_labels.jsonl",
        report_output_path=args.output_dir / "import_report.json",
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
