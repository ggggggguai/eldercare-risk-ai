from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_handoff import (
    CameraHandoffError,
    build_camera_handoff_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_handoff_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble verified W5D-02 episode, W5D-03A context, and W5D-04 "
            "daily/baseline artifacts into the exact eight-file W5D-05 handoff."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--episode-bundle", type=Path, default=None)
    parser.add_argument("--context-bundle", type=Path, default=None)
    parser.add_argument("--daily-bundle", type=Path, default=None)
    parser.add_argument(
        "--allow-unpinned-stage-overrides",
        action="store_true",
        help=(
            "Accept explicitly supplied fresh stage directories after full stage "
            "validation instead of requiring the production manifest hashes."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_handoff_bundle(
            project_root=ROOT,
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            episode_bundle_dir=args.episode_bundle,
            context_bundle_dir=args.context_bundle,
            daily_bundle_dir=args.daily_bundle,
            allow_unpinned_stage_overrides=args.allow_unpinned_stage_overrides,
        )
    except (CameraHandoffError, FileExistsError) as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "message": str(exc),
                    "successful_output_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "episode_result_count": result.episode_result_count,
                "context_review_count": result.context_review_count,
                "daily_report_count": result.daily_report_count,
                "baseline_profile_count": result.baseline_profile_count,
                "baseline_deviation_count": result.baseline_deviation_count,
                "status": result.status,
                "delivery_status": result.delivery_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
