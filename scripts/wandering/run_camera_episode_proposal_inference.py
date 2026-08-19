from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    CameraEpisodeProposalInferenceError,
    build_camera_episode_proposal_inference_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/modules/wandering_camera_episode_proposal_shape_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed shape inference for original S0 camera episode proposals."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_episode_proposal_inference_bundle(
            project_root=ROOT,
            config_path=args.config,
            batch_index_path=args.batch_index,
            output_dir=args.output_dir,
        )
    except (CameraEpisodeProposalInferenceError, FileExistsError) as exc:
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
                "proposal_count": result.proposal_count,
                "ready_count": result.ready_count,
                "unavailable_count": result.unavailable_count,
                "boundary_uncertain_count": result.boundary_uncertain_count,
                "inference_error_count": result.inference_error_count,
                "model_forward_invocation_count": (
                    result.model_forward_invocation_count
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
