from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.fall_detection_2017_cvat import (
    import_fall_detection_2017_cvat_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the fall_detection_2017 CVAT exports into a v2 JSONL batch."
    )
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--revision-archive", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/annotations/fall_risk/generated/v2/fall_detection_2017_manual"
        ),
    )
    parser.add_argument("--labeler", default="fall_detection_2017_manual_labeler")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = import_fall_detection_2017_cvat_labels(
            args.base_archive,
            revision_archive=args.revision_archive,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            labeler=args.labeler,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["outputs"], ensure_ascii=False, sort_keys=True))
    print(f"report_output={args.output_dir / 'import_report.json'}")


if __name__ == "__main__":
    main()
