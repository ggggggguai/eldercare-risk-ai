from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.pose_quality import PoseQualityConfig, run_pose_quality_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗并平滑跌倒风险姿态关键点 JSONL。")
    parser.add_argument("--input", type=Path, required=True, help="输入原始姿态 JSONL 路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出稳定姿态 JSONL 路径。")
    parser.add_argument("--min-keypoint-score", type=float, default=0.30, help="关键点有效置信度阈值。")
    parser.add_argument("--low-quality-threshold", type=float, default=0.45, help="单帧核心关键点低质量阈值。")
    parser.add_argument("--low-quality-run-frames", type=int, default=3, help="连续低质量帧标记阈值。")
    parser.add_argument("--max-interp-gap-frames", type=int, default=2, help="允许线性插值的最大缺失帧数。")
    parser.add_argument("--alpha", type=float, default=0.40, help="指数平滑系数。")
    parser.add_argument("--jump-threshold-norm", type=float, default=0.18, help="归一化坐标异常跳变阈值。")
    parser.add_argument("--window-sec", type=float, default=1.0, help="质量摘要时间窗秒数。")
    parser.add_argument("--window-frames", type=int, default=None, help="可选固定帧数窗口，设置后优先于秒级窗口。")
    args = parser.parse_args()

    config = PoseQualityConfig(
        min_keypoint_score=args.min_keypoint_score,
        low_quality_threshold=args.low_quality_threshold,
        low_quality_run_frames=args.low_quality_run_frames,
        max_interp_gap_frames=args.max_interp_gap_frames,
        alpha=args.alpha,
        jump_threshold_norm=args.jump_threshold_norm,
        window_sec=args.window_sec,
        window_frames=args.window_frames,
    )
    count = run_pose_quality_jsonl(input_path=args.input, output_path=args.output, config=config)
    print(f"已写入 {count} 条稳定姿态关键点记录：{args.output}")


if __name__ == "__main__":
    main()
