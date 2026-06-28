from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.tracking import run_yolov8_bytetrack


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 person detection with ByteTrack.")
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL track path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument("--scene-region", default="unknown", help="Scene label for all track observations.")
    parser.add_argument("--person-prefix", default="elder", help="Prefix used to build stable person_id values.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    args = parser.parse_args()

    count = run_yolov8_bytetrack(
        video_path=args.input,
        output_path=args.output,
        model_name=args.model,
        scene_region=args.scene_region,
        person_id_prefix=args.person_prefix,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        tracker_config=args.tracker,
        max_frames=args.max_frames,
    )
    print(f"Wrote {count} track observations to {args.output}")


if __name__ == "__main__":
    main()
