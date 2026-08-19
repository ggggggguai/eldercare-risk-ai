from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_episode_inference import (
    build_camera_episode_inference_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE_CONFIG = ROOT / "configs/modules/wandering_camera_episode_v1.yaml"
DEFAULT_CANDIDATE_MANIFEST = ROOT / (
    "reports/mental_health/wandering_performance/"
    "m0r_score_entry_hardening_v1/artifacts/"
    "topowander_m0r_candidate_v3/candidate_manifest.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run truth-separated oracle-boundary episode shape inference with the "
            "fixed TopoWander candidate."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--episode-config", type=Path, default=DEFAULT_EPISODE_CONFIG)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--tracking-jsonl", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episode-boundaries", type=Path)
    mode.add_argument("--whole-clip", action="store_true")
    parser.add_argument("--episode-id")
    parser.add_argument("--target-track-id", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.whole_clip and (args.episode_id is None or args.target_track_id is None):
        raise SystemExit("--whole-clip requires --episode-id and --target-track-id")
    if not args.whole_clip and (args.episode_id is not None or args.target_track_id is not None):
        raise SystemExit(
            "boundary JSONL already carries target_track_id; do not pass whole-clip selectors"
        )
    result = build_camera_episode_inference_bundle(
        project_root=args.project_root,
        episode_config_path=args.episode_config,
        tracking_jsonl_path=args.tracking_jsonl,
        media_sidecar_path=args.media_sidecar,
        candidate_manifest_path=args.candidate_manifest,
        expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
        output_dir=args.output_dir,
        episode_boundaries_path=args.episode_boundaries,
        whole_clip_episode_id=args.episode_id,
        target_track_id=args.target_track_id,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "episode_count": result.episode_count,
                "ready_count": result.ready_count,
                "unavailable_count": result.unavailable_count,
                "boundary_uncertain_count": result.boundary_uncertain_count,
                "inference_error_count": result.inference_error_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

