from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.sit_stand import SitStandAnalysisConfig, run_sit_stand_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="从稳定姿态关键点 JSONL 提取坐站转换能力特征和局部风险分。")
    parser.add_argument("--input", type=Path, required=True, help="输入 cleaned/smoothed pose JSONL 路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出 sit-stand event JSONL 路径。")
    parser.add_argument("--min-event-frames", type=int, default=5, help="生成候选坐站事件所需的最少帧数。")
    parser.add_argument("--min-usable-frame-ratio", type=float, default=0.60, help="事件内可用于坐站分析的最小帧比例。")
    parser.add_argument("--min-sit-stand-keypoint-coverage", type=float, default=0.70, help="肩、髋、膝、踝关键点最小覆盖率。")
    parser.add_argument("--movement-start-delta", type=float, default=0.02, help="触发候选坐站运动的髋部垂直变化阈值。")
    parser.add_argument("--min-vertical-displacement", type=float, default=0.10, help="确认坐站转换的最小髋部垂直位移。")
    parser.add_argument("--post-stand-window-sec", type=float, default=2.0, help="起身完成后用于评估摇晃和站稳的观察窗口秒数。")
    parser.add_argument("--normal-duration-sec", type=float, default=3.0, help="坐站耗时开始计入风险的秒数。")
    parser.add_argument("--high-duration-sec", type=float, default=6.0, help="坐站耗时高风险参考秒数。")
    parser.add_argument("--trunk-forward-angle-threshold", type=float, default=30.0, help="躯干前倾角风险阈值。")
    parser.add_argument("--post-stand-sway-threshold", type=float, default=0.18, help="起身后摇晃风险阈值。")
    parser.add_argument("--normal-stabilization-sec", type=float, default=0.8, help="站稳时间开始计入风险的秒数。")
    parser.add_argument("--high-stabilization-sec", type=float, default=2.0, help="站稳时间高风险参考秒数。")
    args = parser.parse_args()

    config = SitStandAnalysisConfig(
        min_event_frames=args.min_event_frames,
        min_usable_frame_ratio=args.min_usable_frame_ratio,
        min_sit_stand_keypoint_coverage=args.min_sit_stand_keypoint_coverage,
        movement_start_delta=args.movement_start_delta,
        min_vertical_displacement=args.min_vertical_displacement,
        post_stand_window_sec=args.post_stand_window_sec,
        normal_duration_sec=args.normal_duration_sec,
        high_duration_sec=args.high_duration_sec,
        trunk_forward_angle_threshold_deg=args.trunk_forward_angle_threshold,
        post_stand_sway_threshold=args.post_stand_sway_threshold,
        normal_stabilization_sec=args.normal_stabilization_sec,
        high_stabilization_sec=args.high_stabilization_sec,
    )
    count = run_sit_stand_jsonl(input_path=args.input, output_path=args.output, config=config)
    print(f"已写入 {count} 条坐站转换事件记录：{args.output}")


if __name__ == "__main__":
    main()
