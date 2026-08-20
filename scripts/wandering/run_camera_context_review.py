from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    CameraContextReviewError,
    build_camera_context_review_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_context_review_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run W5D-03A eligible episode selection, start/middle/end scene and "
            "person-crop extraction, optional multimodal context review, and "
            "failure-safe partial handoff generation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--episode-results",
        type=Path,
        default=None,
        help=(
            "Use a fresh W5D-02 episode_results.jsonl with the pinned schema and "
            "content hash instead of the configured canonical path."
        ),
    )
    parser.add_argument(
        "--provider-mode",
        choices=("disabled", "fake", "openai-compatible"),
        default=None,
        help=(
            "Override the configured provider mode. openai-compatible reads "
            "endpoint, API key, and model from the environment names in the config."
        ),
    )
    parser.add_argument(
        "--source-video-id",
        action="append",
        default=None,
        help="Restrict an engineering run to one or more registered source videos.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_context_review_bundle(
            project_root=ROOT,
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            provider_mode=args.provider_mode,
            source_video_ids=args.source_video_id,
            episode_results_path=args.episode_results,
        )
    except (CameraContextReviewError, FileExistsError) as exc:
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
                "input_episode_count": result.input_episode_count,
                "eligible_episode_count": result.eligible_episode_count,
                "context_review_count": result.context_review_count,
                "ready_count": result.ready_count,
                "unavailable_count": result.unavailable_count,
                "status": result.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
