from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import write_le2i_fall_labels


DEFAULT_OUTPUT_DIR = Path("data/annotations/fall_risk/generated/v1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import official LE2I TXT fall windows as traceable event candidates."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--event-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "le2i_official_event_labels.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "le2i_official_import_report.json",
    )
    args = parser.parse_args()
    try:
        report = write_le2i_fall_labels(
            args.manifest,
            event_output_path=args.event_output,
            report_output_path=args.report_output,
            overwrite=False,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"imported_fall_windows={report['imported_fall_windows']}")
    print(f"bbox_only_without_window={report['bbox_only_without_window']}")
    print(f"explicit_no_fall_window={report['explicit_no_fall_window']}")
    print(f"excluded_unsupervised_subset={report['excluded_unsupervised_subset']}")
    print(f"event_output={args.event_output}")
    print(f"report_output={args.report_output}")


if __name__ == "__main__":
    main()
