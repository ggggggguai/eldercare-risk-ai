from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    build_primary_camera_inference_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manifest-bound TopoWander-MPT primary candidate on trusted "
            "offline camera tracking windows."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tracking-jsonl", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--episode-merge-gap-seconds",
        type=float,
        required=True,
        help="Explicit development-only merge gap; this is not a frozen policy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_primary_camera_inference_bundle(
        camera_config_path=args.config,
        project_root=args.project_root,
        tracking_jsonl_path=args.tracking_jsonl,
        media_sidecar_path=args.media_sidecar,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        episode_merge_gap_seconds=args.episode_merge_gap_seconds,
        output_dir=args.output,
    )
    print(
        f"Built {result.window_count} primary camera windows at {result.output_dir} "
        f"(ready={result.ready_window_count}, unavailable={result.unavailable_window_count}, "
        f"inference_error={result.inference_error_count}, "
        f"episode_candidates={result.episode_candidate_count}, "
        f"manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
