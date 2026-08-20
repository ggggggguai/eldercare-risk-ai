from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_review import (
    CameraEpisodeProposalReviewError,
    build_camera_episode_proposal_review_pack,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/modules/wandering_camera_episode_proposal_review_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a truth-free human review pack for S0 proposals and S2A shape results."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_episode_proposal_review_pack(
            project_root=ROOT,
            config_path=args.config,
            batch_index_path=args.batch_index,
            output_dir=args.output_dir,
        )
    except (CameraEpisodeProposalReviewError, FileExistsError) as exc:
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
                "output_dir": str(result.output_dir),
                "source_video_count": result.source_video_count,
                "proposal_count": result.proposal_count,
                "trajectory_plot_count": result.trajectory_plot_count,
                "model_forward_invocation_count": (
                    result.model_forward_invocation_count
                ),
                "model_invocation_skipped_count": (
                    result.model_invocation_skipped_count
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
