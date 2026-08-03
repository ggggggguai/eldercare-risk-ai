"""Train TREND-001 PersonalTrendExpert V3.3.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.mood_social.personal_trend import (
    PersonalTrendError,
    train_personal_trend_expert,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mood_social_personal_trend_v3_3_3.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        result = train_personal_trend_expert(
            root,
            config,
            overwrite=args.overwrite,
            command=sys.argv,
        )
    except PersonalTrendError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "pass", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
