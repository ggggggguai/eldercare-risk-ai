from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    CameraContextReviewError,
    build_camera_context_review_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_baseline import (
    WanderingCameraDailyBaselineError,
    build_camera_daily_baseline_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    CameraEpisodePipelineError,
    build_camera_episode_pipeline,
)
from elderly_monitoring.modules.mental_health.wandering.camera_handoff import (
    CameraHandoffError,
    build_camera_handoff_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
EPISODE_CONFIG = ROOT / "configs/modules/wandering_camera_episode_pipeline_v1.yaml"
CONTEXT_CONFIG = ROOT / "configs/modules/wandering_camera_context_review_v1.yaml"
DAILY_CONFIG = ROOT / "configs/modules/wandering_camera_daily_baseline_v1.yaml"
HANDOFF_CONFIG = ROOT / "configs/modules/wandering_camera_handoff_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete registered B01+B02 cached-tracking flow: automatic "
            "episode/shape, context, daily/baseline, and W5D-05 handoff."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--provider-mode",
        choices=("disabled", "fake", "openai-compatible"),
        default="fake",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "message": f"E2E output root already exists: {output_root}",
                    "successful_output_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    episode_dir = output_root / "episode"
    context_dir = output_root / "context"
    daily_dir = output_root / "daily"
    handoff_dir = output_root / "handoff"
    try:
        episode = build_camera_episode_pipeline(
            project_root=ROOT,
            config_path=EPISODE_CONFIG,
            output_dir=episode_dir,
            run_id=f"{args.run_id}-episode",
        )
        context = build_camera_context_review_bundle(
            project_root=ROOT,
            config_path=CONTEXT_CONFIG,
            output_dir=context_dir,
            run_id=f"{args.run_id}-context",
            provider_mode=args.provider_mode,
            episode_results_path=episode_dir / "episode_results.jsonl",
        )
        daily = build_camera_daily_baseline_bundle(
            project_root=ROOT,
            config_path=DAILY_CONFIG,
            output_dir=daily_dir,
            run_id=f"{args.run_id}-daily",
            episode_results_path=episode_dir / "episode_results.jsonl",
            context_reviews_path=context_dir / "context_reviews.jsonl",
        )
        handoff = build_camera_handoff_bundle(
            project_root=ROOT,
            config_path=HANDOFF_CONFIG,
            output_dir=handoff_dir,
            run_id=args.run_id,
            episode_bundle_dir=episode_dir,
            context_bundle_dir=context_dir,
            daily_bundle_dir=daily_dir,
            allow_unpinned_stage_overrides=True,
        )
    except (
        CameraEpisodePipelineError,
        CameraContextReviewError,
        WanderingCameraDailyBaselineError,
        CameraHandoffError,
        FileExistsError,
    ) as exc:
        if output_root.exists():
            shutil.rmtree(output_root)
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
                "output_root": output_root.as_posix(),
                "handoff_dir": handoff.output_dir.as_posix(),
                "episode_result_count": episode.result_count,
                "context_review_count": context.context_review_count,
                "daily_report_count": daily.daily_report_count,
                "baseline_profile_count": daily.baseline_profile_count,
                "baseline_deviation_count": daily.baseline_deviation_count,
                "status": handoff.status,
                "delivery_status": handoff.delivery_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
