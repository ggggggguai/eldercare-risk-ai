from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    build_camera_episode_boundary_proposal_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs/modules/wandering_camera_episode_boundary_proposal_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate truth-free, manual-review camera episode boundary proposals."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tracking-jsonl", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_camera_episode_boundary_proposal_bundle(
        project_root=args.project_root,
        proposal_config_path=args.config,
        tracking_jsonl_path=args.tracking_jsonl,
        media_sidecar_path=args.media_sidecar,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "proposal_count": result.proposal_count,
                "locomotion_proposal_count": result.locomotion_proposal_count,
                "proposed_count": result.proposed_count,
                "uncertain_count": result.uncertain_count,
                "rejected_by_qc_count": result.rejected_by_qc_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
