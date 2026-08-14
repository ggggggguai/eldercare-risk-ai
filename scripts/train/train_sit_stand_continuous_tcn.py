from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.sit_stand_continuous_tcn import (
    ContinuousSitStandTCNConfig,
    train_continuous_sit_stand_tcn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the provisional causal multi-head sit-stand TCN."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/sit_stand_event_v1.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-provisional", action="store_true")
    return parser


def _load_config(path: Path, profile: str) -> ContinuousSitStandTCNConfig:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sit-stand training config must be a mapping")
    profiles = payload.get("continuous_tcn_profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"sit-stand config is missing continuous profile {profile}")
    values = dict(profiles[profile])
    values["dilations"] = tuple(values["dilations"])
    return ContinuousSitStandTCNConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_continuous_sit_stand_tcn(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            config=_load_config(args.config, args.profile),
            allow_provisional=args.allow_provisional,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
