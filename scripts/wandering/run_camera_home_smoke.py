from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_home_smoke import (
    build_camera_home_smoke_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one W5D-03B raw home MP4 through fresh tracking, automatic "
            "episode/shape, and three-frame context smoke."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--source-group-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--stream-epoch", required=True)
    parser.add_argument("--media-ref", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--clock-domain-id", required=True)
    parser.add_argument("--timezone")
    parser.add_argument(
        "--home-annotation-status",
        choices=("not_provided", "unlabeled", "available"),
        default="not_provided",
    )
    parser.add_argument("--capture-started-at")
    parser.add_argument("--provider-mode", choices=("fake", "disabled"), default="fake")
    parser.add_argument("--detector-model", default="yolov8n.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--tracker-config", default="bytetrack.yaml")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_camera_home_smoke_bundle(
        project_root=args.project_root,
        input_video_path=args.input_video,
        output_dir=args.output_dir,
        run_id=args.run_id,
        source_video_id=args.source_video_id,
        source_group_id=args.source_group_id,
        device_id=args.device_id,
        setup_id=args.setup_id,
        stream_epoch=args.stream_epoch,
        media_ref=args.media_ref,
        participant_id=args.participant_id,
        session_id=args.session_id,
        clock_domain_id=args.clock_domain_id,
        timezone_name=args.timezone,
        home_annotation_status=args.home_annotation_status,
        capture_started_at=args.capture_started_at,
        provider_mode=args.provider_mode,
        detector_model=args.detector_model,
        detector_confidence=args.detector_confidence,
        detector_iou=args.detector_iou,
        tracker_config=args.tracker_config,
    )
    print(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "proposal_count": result.proposal_count,
                "episode_result_count": result.episode_result_count,
                "context_review_count": result.context_review_count,
                "home_smoke_status": result.home_smoke_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
