from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.fall_event_tcn import (
    FallEventTCNConfig,
    train_fall_event_candidate_tcn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the provisional fall-event candidate-clip TCN."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/fall_risk/fall_event_proxy_v1/dataset.npz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/fall_event_v1.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subtype-loss-weight", type=float, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
    )
    parser.add_argument("--allow-provisional", action="store_true")
    return parser


def _load_training_config(path: Path, profile: str) -> FallEventTCNConfig:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fall-event training config must be a mapping")
    if payload.get("schema_version") != "fall-event-training-config-v1":
        raise ValueError("unsupported fall-event training config schema")
    profiles = payload.get("training_profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"fall-event training config is missing profile {profile}")
    return FallEventTCNConfig(**profiles[profile])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        training_config = _load_training_config(args.config, args.profile)
        if args.seed is not None:
            training_config = replace(training_config, seed=args.seed)
        if args.device is not None:
            training_config = replace(training_config, device=args.device)
        if args.subtype_loss_weight is not None:
            training_config = replace(
                training_config,
                subtype_loss_weight=args.subtype_loss_weight,
            )
        summary = train_fall_event_candidate_tcn(
            args.data,
            args.output_dir,
            metadata_path=args.metadata,
            config=training_config,
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
