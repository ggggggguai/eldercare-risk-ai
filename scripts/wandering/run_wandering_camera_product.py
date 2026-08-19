from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_product import (
    build_wandering_camera_product,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic session-level wandering evidence from an existing "
            "tracking JSONL and media sidecar."
        )
    )
    parser.add_argument("--tracking-jsonl", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument(
        "--episode-merge-gap-seconds",
        type=float,
        required=True,
        help="Explicit development-only episode gap; it is not a frozen policy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    result = build_wandering_camera_product(
        project_root=project_root,
        tracking_jsonl_path=args.tracking_jsonl,
        media_sidecar_path=args.media_sidecar,
        episode_merge_gap_seconds=args.episode_merge_gap_seconds,
        output_dir=args.output,
    )
    print(
        f"Built synthetic session evidence at {result.output_dir} "
        f"(session_status={result.session_status}, degraded={str(result.degraded).lower()}, "
        f"windows={result.window_count}, episodes={result.episode_candidate_count}, "
        f"manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
