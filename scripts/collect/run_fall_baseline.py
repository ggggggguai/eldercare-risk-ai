from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.baseline import BaselineModelConfig, run_baseline_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从历史和当前结构化 JSONL 计算个体化行为基线偏离特征。"
    )
    parser.add_argument("--baseline-input", type=Path, required=True, help="历史基线窗口 JSONL 路径。")
    parser.add_argument("--current-input", type=Path, required=True, help="当前观测窗口 JSONL 路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出 baseline deviation JSONL 路径。")
    parser.add_argument("--min-history-days", type=int, default=3, help="初始基线所需的最少历史天数。")
    parser.add_argument("--stable-history-days", type=int, default=7, help="稳定基线参考历史天数。")
    parser.add_argument("--max-history-days", type=int, default=14, help="滚动统计最多使用的历史天数。")
    parser.add_argument("--min-history-records", type=int, default=10, help="建立个人基线所需的最少历史记录数。")
    parser.add_argument(
        "--aggregation-period",
        choices=("day", "hour"),
        default="day",
        help="个体基线聚合粒度，默认按天。",
    )
    parser.add_argument("--min-quality-score", type=float, default=0.60, help="基线质量降级阈值。")
    args = parser.parse_args()

    config = BaselineModelConfig(
        min_history_days=args.min_history_days,
        stable_history_days=args.stable_history_days,
        max_history_days=args.max_history_days,
        min_history_records=args.min_history_records,
        aggregation_period=args.aggregation_period,
        min_quality_score=args.min_quality_score,
    )
    count = run_baseline_jsonl(
        baseline_input_path=args.baseline_input,
        current_input_path=args.current_input,
        output_path=args.output,
        config=config,
    )
    print(f"已写入 {count} 条个体化行为基线偏离记录：{args.output}")


if __name__ == "__main__":
    main()
