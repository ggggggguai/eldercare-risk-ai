from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_training_audit import (
    validate_gait_test_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release-gated gait test evaluator; no test input is opened before custodian validation."
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/splits/fall_risk/training_labels_v3/split.json"))
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fall_risk_video_manifest.jsonl"))
    parser.add_argument("--assignments", type=Path, default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation/gait_v1.provisional.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--test-dataset",
        type=Path,
        required=True,
        help="Independently sealed test-only dataset released by the custodian.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        release = validate_gait_test_release(
            args.release,
            split_path=args.split,
            labels_path=args.labels,
            manifest_path=args.manifest,
            assignments_path=args.assignments,
            evaluation_config_path=args.evaluation_config,
            checkpoint_path=args.checkpoint,
            candidate_manifest_path=args.candidate_manifest,
        )
        if not args.test_dataset.is_file():
            raise FileNotFoundError(f"sealed test dataset is missing: {args.test_dataset}")
        raise NotImplementedError(
            "test scoring requires the separately released test-only tensor evaluator; "
            "this command intentionally stops before opening test features"
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"release_id": release["release_id"], "test_evaluated": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
