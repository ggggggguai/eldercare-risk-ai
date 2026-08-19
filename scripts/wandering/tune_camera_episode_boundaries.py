from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_development import (
    CameraEpisodeBoundaryDevelopmentError,
    build_camera_episode_segmenter_search,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_segmenter_search_v2.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and deterministically replay the W5D-01 B01+B02 recall-first "
            "camera episode segmenter grid."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = build_camera_episode_segmenter_search(
            project_root=args.project_root,
            config_path=args.config,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
        )
    except (CameraEpisodeBoundaryDevelopmentError, FileExistsError, OSError) as exc:
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
                "status": "selected",
                "output_dir": result.output_dir.as_posix(),
                "selected_candidate_id": result.selected_candidate_id,
                "candidate_count": result.candidate_count,
                "selected_gate_satisfied": result.selected_gate_satisfied,
                "manifest_sha256": result.manifest_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
