#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import write_ntu_rgbd_clip_labels
from elderly_monitoring.modules.fall_risk.data_manifest import (
    write_ntu_rgbd_clip_manifest,
)


DEFAULT_OUTPUT_DIR = Path(
    "data/annotations/fall_risk/generated/v2/ntu_rgbd_clip_labels"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an external NTU RGB+D RGB manifest and import only project-mapped "
            "single-action clip labels."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("configs/data/ntu_rgbd_clip_label_map_v2.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/manifests/ntu_rgbd_clip_manifest.jsonl"),
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
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        outputs = (args.manifest_output, args.action_output, args.report_output)
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError("manifest, action, and report outputs must be different paths")
        if not args.label_map.is_file():
            raise FileNotFoundError(args.label_map)
        if not args.overwrite:
            existing = next((path for path in outputs if path.exists()), None)
            if existing is not None:
                raise FileExistsError(existing)
        manifest = write_ntu_rgbd_clip_manifest(
            args.source_root,
            args.manifest_output,
            overwrite=args.overwrite,
            ffprobe_bin=args.ffprobe_bin,
            workers=args.workers,
        )
        report = write_ntu_rgbd_clip_labels(
            args.manifest_output,
            args.label_map,
            action_output_path=args.action_output,
            report_output_path=args.report_output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"manifest_videos={manifest.summary['video_count']}")
    print(f"manifest_sha256={manifest.manifest_sha256}")
    print(f"imported_action_labels={report['imported_action_labels']}")
    print(f"excluded_videos={report['excluded_videos']}")
    print(f"manifest_output={args.manifest_output}")
    print(f"action_output={args.action_output}")
    print(f"report_output={args.report_output}")


if __name__ == "__main__":
    main()
