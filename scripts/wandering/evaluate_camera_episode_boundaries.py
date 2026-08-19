from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation import (
    CameraEpisodeBoundaryEvaluationError,
    build_camera_episode_boundary_evaluation_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/modules/wandering_camera_episode_boundary_eval_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate S0 camera boundary proposals against an independent "
            "boundary-only view without running shape inference."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_episode_boundary_evaluation_bundle(
            project_root=args.project_root,
            config_path=args.config,
            batch_index_path=args.batch_index,
            output_dir=args.output_dir,
        )
    except CameraEpisodeBoundaryEvaluationError as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "failure_type": "identity_or_input_error",
                    "error_code": exc.code,
                    "message": str(exc),
                    "successful_evaluation_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except FileExistsError as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "failure_type": "output_error",
                    "error_code": "output_exists",
                    "message": str(exc),
                    "successful_evaluation_written": False,
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
                "ready_truth_count": result.ready_truth_count,
                "uncertain_truth_count": result.uncertain_truth_count,
                "proposal_count": result.proposal_count,
                "match_count": result.match_count,
                "failure_count": result.failure_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
