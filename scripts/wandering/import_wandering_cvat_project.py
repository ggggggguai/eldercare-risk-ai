from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_cvat_project import (
    build_cvat_project_import_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a multi-task CVAT wandering project, normalize cumulative task "
            "frame offsets, and derive auditable single-subject machine Track IDs."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--cvat-xml", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--tracking-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--clock-domain-id", required=True)
    parser.add_argument("--minimum-temporal-coverage", type=float, default=0.8)
    parser.add_argument("--geometry-iou-threshold", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_cvat_project_import_bundle(
        project_root=args.project_root,
        cvat_xml_path=args.cvat_xml,
        video_root=args.video_root,
        tracking_output_root=args.tracking_output_root,
        output_dir=args.output_dir,
        participant_id=args.participant_id,
        session_id=args.session_id,
        clock_domain_id=args.clock_domain_id,
        minimum_temporal_coverage=args.minimum_temporal_coverage,
        geometry_iou_threshold=args.geometry_iou_threshold,
    )
    print(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "task_count": result.task_count,
                "episode_count": result.episode_count,
                "aligned_episode_count": result.aligned_episode_count,
                "low_coverage_episode_count": result.low_coverage_episode_count,
                "geometry_mismatch_episode_count": (
                    result.geometry_mismatch_episode_count
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
