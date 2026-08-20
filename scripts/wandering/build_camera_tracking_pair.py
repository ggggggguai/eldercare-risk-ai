#!/usr/bin/env python
"""Prepare one receipt-gated wandering C1 tracking pair from video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_collection import (
    build_authorized_camera_tracking_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one anonymous development-video tracking JSONL and media sidecar "
            "with the fixed shared YOLOv8/ByteTrack controller."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--media-ref", required=True)
    parser.add_argument(
        "--camera-motion-state",
        choices=("stable", "moved", "not_checked"),
        required=True,
    )
    parser.add_argument("--deidentification-status", required=True)
    parser.add_argument("--capture-started-at")
    parser.add_argument("--timezone")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_authorized_camera_tracking_pair(
        project_root=args.project_root,
        receipt_path=args.receipt,
        collection_path=args.collection,
        source_video_id=args.source_video_id,
        input_video_path=args.input_video,
        media_ref=args.media_ref,
        camera_motion_state=args.camera_motion_state,
        deidentification_status=args.deidentification_status,
        capture_started_at=args.capture_started_at,
        timezone=args.timezone,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
