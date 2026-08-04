from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.near_fall_tcn import (
    NearFallTCNConfig,
    train_near_fall_tcn,
)
from elderly_monitoring.modules.fall_risk.near_fall_training import (
    create_synthetic_near_fall_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the recovery-confirmation near-fall TCN on train/validation only. "
            "Use --synthetic-smoke only for infrastructure verification."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/fall_risk/near_fall_event_v1/dataset.npz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/near_fall_event_v1.yaml"),
    )
    parser.add_argument(
        "--profile", choices=("synthetic_smoke", "pilot"), default="pilot"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Create an explicitly synthetic dataset under the output directory and train it.",
    )
    parser.add_argument("--synthetic-seed", type=int, default=42)
    return parser


def _load_training_config(path: Path, profile: str) -> NearFallTCNConfig:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("near-fall training config must be a mapping")
    if payload.get("schema_version") != "near-fall-event-training-config-v1":
        raise ValueError("unsupported near-fall training config schema")
    profiles = payload.get("tcn_profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"near-fall config is missing TCN profile {profile}")
    values = dict(profiles[profile])
    values["dilations"] = tuple(values["dilations"])
    return NearFallTCNConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = "synthetic_smoke" if args.synthetic_smoke else args.profile
        data_path = args.data
        metadata_path = args.metadata
        run_output = args.output_dir
        if args.synthetic_smoke:
            synthetic = create_synthetic_near_fall_dataset(
                args.output_dir / "synthetic_dataset",
                seed=args.synthetic_seed,
            )
            data_path = Path(synthetic["dataset_path"])
            metadata_path = Path(synthetic["metadata_path"])
            run_output = args.output_dir / "training"
        summary = train_near_fall_tcn(
            data_path,
            run_output,
            metadata_path=metadata_path,
            config=_load_training_config(args.config, profile),
            allow_synthetic=args.synthetic_smoke,
            resume_checkpoint=args.resume_checkpoint,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
