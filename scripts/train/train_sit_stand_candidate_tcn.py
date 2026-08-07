from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.sit_stand_tcn import (
    SitStandCandidateTCNConfig,
    train_sit_stand_candidate_tcn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the provisional sit-stand candidate clip TCN."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/fall_risk/sit_stand_event_v1/dataset.npz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/sit_stand_event_v1.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--allow-provisional", action="store_true")
    return parser


def _load_training_config(
    path: Path, profile: str
) -> SitStandCandidateTCNConfig:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sit-stand training config must be a mapping")
    if payload.get("schema_version") != "sit-stand-event-training-config-v1":
        raise ValueError("unsupported sit-stand training config schema")
    profiles = payload.get("candidate_tcn_profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"sit-stand config is missing candidate TCN profile {profile}")
    values = dict(profiles[profile])
    values["dilations"] = tuple(values["dilations"])
    return SitStandCandidateTCNConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_sit_stand_candidate_tcn(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            config=_load_training_config(args.config, args.profile),
            allow_provisional=args.allow_provisional,
            resume_checkpoint=args.resume_checkpoint,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
