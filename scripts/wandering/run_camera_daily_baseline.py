from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_daily_baseline import (
    WanderingCameraDailyBaselineError,
    build_camera_daily_baseline_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_daily_baseline_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build W5D-04 explicit person/session/day reports and strictly-prior "
            "3/7/14-day rolling baseline profiles/deviations."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--episode-results",
        type=Path,
        default=None,
        help="Use an explicitly supplied validated W5D-02 episode JSONL.",
    )
    parser.add_argument(
        "--context-reviews",
        type=Path,
        default=None,
        help="Use an explicitly supplied validated W5D-03A context JSONL.",
    )
    parser.add_argument(
        "--replay-days",
        type=int,
        default=None,
        help=(
            "Build a deterministic multi-day state-machine replay. This does not "
            "represent real longitudinal observation."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_daily_baseline_bundle(
            project_root=ROOT,
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            replay_days=args.replay_days,
            episode_results_path=args.episode_results,
            context_reviews_path=args.context_reviews,
        )
    except (WanderingCameraDailyBaselineError, FileExistsError) as exc:
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
                "daily_report_count": result.daily_report_count,
                "baseline_profile_count": result.baseline_profile_count,
                "baseline_deviation_count": result.baseline_deviation_count,
                "person_count": result.person_count,
                "local_date_count": result.local_date_count,
                "status": result.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
