from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.pose import run_yolov8_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="提取跌倒风险视频中的人体姿态关键点。")
    parser.add_argument("--input", type=Path, required=True, help="输入视频路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 姿态路径。")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose 模型路径或名称。")
    parser.add_argument("--scene-region", default="unknown", help="写入每条姿态记录的场景区域标签。")
    parser.add_argument("--person-prefix", default="elder", help="生成 person_id 使用的前缀。")
    parser.add_argument("--confidence", type=float, default=0.25, help="姿态模型检测置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值。")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics 跟踪器配置。")
    parser.add_argument("--max-frames", type=int, default=None, help="可选帧数上限，用于烟测。")
    parser.add_argument(
        "--absolute-coordinates",
        action="store_true",
        help="输出像素坐标；默认输出 0-1 归一化坐标。",
    )
    args = parser.parse_args()

    count = run_yolov8_pose(
        video_path=args.input,
        output_path=args.output,
        model_name=args.model,
        scene_region=args.scene_region,
        person_id_prefix=args.person_prefix,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        tracker_config=args.tracker,
        max_frames=args.max_frames,
        normalize_coordinates=not args.absolute_coordinates,
    )
    print(f"已写入 {count} 条姿态关键点记录：{args.output}")


if __name__ == "__main__":
    main()
