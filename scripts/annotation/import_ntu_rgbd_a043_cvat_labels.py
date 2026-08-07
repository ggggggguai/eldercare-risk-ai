#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.ntu_rgbd_cvat import (
    NTU_RGBD_A043_BATCH_ID,
    import_ntu_rgbd_a043_cvat_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize accepted external NTU RGB+D A043 CVAT exports into a "
            "v2 JSONL batch."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="CVAT ZIP or directory containing task ZIPs; repeat for each source.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ntu_rgbd_clip_manifest.jsonl"),
    )
    parser.add_argument(
        "--revision",
        type=Path,
        action="append",
        default=[],
        help=(
            "CVAT ZIP or directory with corrected tasks that replace matching "
            "--source tasks; repeat for each revision source."
        ),
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=Path("configs/data/fall_risk_cvat_labels_v2.json"),
    )
    parser.add_argument(
        "--decision-config",
        type=Path,
        default=Path("configs/data/ntu_rgbd_a043_cvat_decision_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2")
        / NTU_RGBD_A043_BATCH_ID,
    )
    parser.add_argument(
        "--labeler",
        default="ntu_rgbd_a043_cvat_review_20260730",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = import_ntu_rgbd_a043_cvat_labels(
            args.source,
            revision_paths=args.revision,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            label_config_path=args.label_config,
            decision_config_path=args.decision_config,
            labeler=args.labeler,
            allow_incomplete=args.allow_incomplete,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
