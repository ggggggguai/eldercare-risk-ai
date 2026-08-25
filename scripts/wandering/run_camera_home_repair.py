from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_home_repair import (
    build_camera_home_repair_batch,
)


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the home-development proposal and shape stages from cached "
            "tracking with the confidence-0.70 repair profile."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-video-id",
        action="append",
        default=None,
        help="Repeat to select videos; omit to process every cached tracking bundle.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_camera_home_repair_batch(
        project_root=args.project_root,
        tracking_root=args.tracking_root,
        output_dir=args.output_dir,
        source_video_ids=args.source_video_id,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "video_count": result.video_count,
                "proposal_count": result.proposal_count,
                "ready_count": result.ready_count,
                "unavailable_count": result.unavailable_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
