#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import (
    write_toaga_normal_walk_labels,
)


DEFAULT_OUTPUT_DIR = Path(
    "data/annotations/fall_risk/generated/v2/toaga_official_walking"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import eligible TOAGA walking videos as source-derived "
            "A01/normal_walk action candidates."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--action-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "action_labels.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "import_report.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = write_toaga_normal_walk_labels(
            args.manifest,
            action_output_path=args.action_output,
            report_output_path=args.report_output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"imported_normal_walk_labels={report['imported_normal_walk_labels']}")
    print(f"excluded_ineligible={report['excluded_ineligible']}")
    print(f"action_output={args.action_output}")
    print(f"report_output={args.report_output}")


if __name__ == "__main__":
    main()
