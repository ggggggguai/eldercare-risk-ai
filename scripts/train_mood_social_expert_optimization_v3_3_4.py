"""Run OPT-EXPERT-001 strict nested-OOF expert optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.mood_social.experts.optimization import (
    ExpertOptimizationError,
    load_expert_optimization_config,
    optimize_experts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mood_social_expert_optimization_v3_3_4.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = load_expert_optimization_config(path, repository_root=root)
        result = optimize_experts(config, overwrite=args.overwrite, command=sys.argv)
    except (ExpertOptimizationError, OSError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
