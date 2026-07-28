from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_tabular import (
    GaitTabularTrainingConfig,
    SUPPORTED_GAIT_BASELINES,
    train_gait_tabular_baselines,
)


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(models) - set(SUPPORTED_GAIT_BASELINES))
    if not models or unknown:
        raise argparse.ArgumentTypeError(
            "models must be a comma-separated subset of rule,lightgbm,ebm"
        )
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train rule, LightGBM and EBM gait-window baselines."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/fall_risk/gait_window_v1/dataset.npz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/fall_risk/gait_window_v1/tabular"),
    )
    parser.add_argument("--models", type=_parse_models, default=SUPPORTED_GAIT_BASELINES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lightgbm-estimators", type=int, default=200)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--ebm-max-rounds", type=int, default=500)
    parser.add_argument("--ebm-outer-bags", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_gait_tabular_baselines(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            config=GaitTabularTrainingConfig(
                models=args.models,
                seed=args.seed,
                lightgbm_estimators=args.lightgbm_estimators,
                lightgbm_learning_rate=args.lightgbm_learning_rate,
                ebm_max_rounds=args.ebm_max_rounds,
                ebm_outer_bags=args.ebm_outer_bags,
            ),
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
