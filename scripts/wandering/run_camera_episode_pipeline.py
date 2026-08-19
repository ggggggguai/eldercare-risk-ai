from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    CameraEpisodePipelineError,
    build_camera_episode_pipeline,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_episode_pipeline_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the W5D-02 tracking/sidecar -> proposal -> QC -> 80-point -> "
            "shape pipeline and write a fresh daily-adapter handoff."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_camera_episode_pipeline(
            project_root=ROOT,
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except (CameraEpisodePipelineError, FileExistsError) as exc:
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
                "video_count": result.video_count,
                "proposal_count": result.proposal_count,
                "result_count": result.result_count,
                "status": result.status,
                "prediction_coverage": result.prediction_coverage,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
