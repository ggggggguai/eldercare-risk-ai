from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.gait import GaitAnalysisConfig, run_gait_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="从稳定姿态关键点 JSONL 提取步态稳定性特征和规则风险分。")
    parser.add_argument("--input", type=Path, required=True, help="输入 cleaned/smoothed pose JSONL 路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出 gait feature JSONL 路径。")
    parser.add_argument("--window-sec", type=float, default=2.0, help="步态分析时间窗秒数。")
    parser.add_argument("--window-frames", type=int, default=None, help="可选固定帧数窗口，设置后优先于秒级窗口。")
    parser.add_argument("--min-window-frames", type=int, default=5, help="生成一个步态窗口所需的最少帧数。")
    parser.add_argument("--min-usable-frame-ratio", type=float, default=0.60, help="窗口内可用于步态分析的最小帧比例。")
    parser.add_argument("--min-gait-keypoint-coverage", type=float, default=0.70, help="髋、膝、踝关键点最小覆盖率。")
    parser.add_argument("--pause-speed-threshold", type=float, default=0.03, help="归一化中心速度停顿阈值。")
    parser.add_argument("--center-speed-cv-threshold", type=float, default=0.60, help="中心速度变异风险阈值。")
    parser.add_argument("--ankle-asymmetry-threshold", type=float, default=0.45, help="左右踝运动不对称风险阈值。")
    parser.add_argument("--hip-sway-threshold", type=float, default=0.035, help="髋部相对路径摆动风险阈值。")
    parser.add_argument("--pause-ratio-threshold", type=float, default=0.25, help="停顿帧比例风险阈值。")
    parser.add_argument("--shuffling-motion-threshold", type=float, default=0.018, help="疑似拖步/小碎步踝部运动阈值。")
    args = parser.parse_args()

    config = GaitAnalysisConfig(
        window_sec=args.window_sec,
        window_frames=args.window_frames,
        min_window_frames=args.min_window_frames,
        min_usable_frame_ratio=args.min_usable_frame_ratio,
        min_gait_keypoint_coverage=args.min_gait_keypoint_coverage,
        pause_speed_threshold_norm_per_sec=args.pause_speed_threshold,
        center_speed_cv_risk_threshold=args.center_speed_cv_threshold,
        ankle_asymmetry_risk_threshold=args.ankle_asymmetry_threshold,
        hip_sway_risk_threshold=args.hip_sway_threshold,
        pause_ratio_risk_threshold=args.pause_ratio_threshold,
        shuffling_motion_threshold=args.shuffling_motion_threshold,
    )
    count = run_gait_jsonl(input_path=args.input, output_path=args.output, config=config)
    print(f"已写入 {count} 条步态稳定性窗口记录：{args.output}")


if __name__ == "__main__":
    main()
