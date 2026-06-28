from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.near_fall import NearFallDetectionConfig, run_near_fall_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从稳定姿态关键点 JSONL 提取近跌倒候选事件和局部分数。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入 cleaned/smoothed pose JSONL 路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出 near-fall event JSONL 路径。")
    parser.add_argument("--window-sec", type=float, default=1.5, help="近跌倒检测时间窗秒数。")
    parser.add_argument("--step-sec", type=float, default=0.5, help="近跌倒检测滑窗步长秒数。")
    parser.add_argument(
        "--window-frames",
        type=int,
        default=None,
        help="可选固定帧数窗口，设置后优先于秒级窗口。",
    )
    parser.add_argument(
        "--step-frames",
        type=int,
        default=None,
        help="可选固定帧数步长，需配合 --window-frames 使用。",
    )
    parser.add_argument("--min-event-frames", type=int, default=5, help="生成候选近跌倒窗口所需的最少帧数。")
    parser.add_argument("--merge-gap-sec", type=float, default=0.5, help="相邻同类候选事件合并的最大间隔秒数。")
    parser.add_argument("--min-output-score", type=float, default=0.25, help="输出候选事件的最低局部分数。")
    parser.add_argument(
        "--min-usable-frame-ratio",
        type=float,
        default=0.60,
        help="窗口内可用于近跌倒分析的最小帧比例。",
    )
    parser.add_argument(
        "--min-core-keypoint-coverage",
        type=float,
        default=0.70,
        help="肩、髋、膝、踝关键点最小覆盖率。",
    )
    parser.add_argument(
        "--lateral-velocity-threshold",
        type=float,
        default=0.22,
        help="归一化横向速度 proxy 风险阈值。",
    )
    parser.add_argument(
        "--lateral-acceleration-threshold",
        type=float,
        default=0.90,
        help="归一化横向加速度 proxy 风险阈值。",
    )
    parser.add_argument(
        "--path-deviation-threshold",
        type=float,
        default=0.055,
        help="身体中心路径偏移 proxy 风险阈值。",
    )
    parser.add_argument("--hip-drop-threshold", type=float, default=0.10, help="髋部快速下沉 proxy 风险阈值。")
    parser.add_argument(
        "--trunk-angle-delta-threshold",
        type=float,
        default=22.0,
        help="躯干角变化 proxy 风险阈值。",
    )
    args = parser.parse_args()

    config = NearFallDetectionConfig(
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        window_frames=args.window_frames,
        step_frames=args.step_frames,
        min_event_frames=args.min_event_frames,
        merge_gap_sec=args.merge_gap_sec,
        min_output_score=args.min_output_score,
        min_usable_frame_ratio=args.min_usable_frame_ratio,
        min_core_keypoint_coverage=args.min_core_keypoint_coverage,
        lateral_velocity_threshold=args.lateral_velocity_threshold,
        lateral_acceleration_threshold=args.lateral_acceleration_threshold,
        path_deviation_threshold=args.path_deviation_threshold,
        hip_drop_threshold=args.hip_drop_threshold,
        trunk_angle_delta_threshold_deg=args.trunk_angle_delta_threshold,
    )
    count = run_near_fall_jsonl(input_path=args.input, output_path=args.output, config=config)
    print(f"已写入 {count} 条近跌倒候选事件记录：{args.output}")


if __name__ == "__main__":
    main()
