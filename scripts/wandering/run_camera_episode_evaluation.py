from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_evaluation import (
    CameraEpisodeEvaluationError,
    build_camera_episode_evaluation_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_episode_eval_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple truth-separated EP1A oracle-boundary prediction bundles."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_episode_evaluation_bundle(
            config_path=args.config,
            batch_index_path=args.batch_index,
            output_dir=args.output_dir,
        )
    except CameraEpisodeEvaluationError as exc:
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
                "truth_count": result.truth_count,
                "shape_eligible_count": result.shape_eligible_count,
                "ready_count": result.ready_count,
                "failure_count": result.failure_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
