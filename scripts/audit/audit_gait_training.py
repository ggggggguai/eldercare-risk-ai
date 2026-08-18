from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.gait_training_audit import (
    build_gait_training_audit,
    write_gait_training_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed gait model training audit without opening test pose."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fall_risk_video_manifest.jsonl"))
    parser.add_argument("--action-labels", type=Path, default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"))
    parser.add_argument("--action-schema", type=Path, default=Path("configs/data/fall_risk_action_label_schema_v3.json"))
    parser.add_argument("--assignments", type=Path, default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/fall_risk/training_labels_v3/split.json"))
    parser.add_argument("--validation", type=Path, default=Path("reports/fall_risk/training-labels-v3-validation.json"))
    parser.add_argument("--training", type=Path, default=Path("configs/training/gait_hierarchical_v1.yaml"))
    parser.add_argument("--evaluation", type=Path, default=Path("configs/evaluation/gait_v1.provisional.yaml"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"))
    parser.add_argument(
        "--additional-pose-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional cleaned-pose directory; repeat for fallback roots.",
    )
    parser.add_argument("--dataset-metadata", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=Path("reports/reproducibility/gait_training_manifest.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("reports/fall_risk/gait_training_audit.md"))
    parser.add_argument("--blockers-output", type=Path, default=Path("reports/fall_risk/gait_training_blockers.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_gait_training_audit(
            manifest=args.manifest,
            action_labels=args.action_labels,
            action_schema=args.action_schema,
            assignments=args.assignments,
            split=args.split,
            validation=args.validation,
            training=args.training,
            evaluation=args.evaluation,
            pose_dir=args.pose_dir,
            additional_pose_dirs=tuple(args.additional_pose_dir),
            dataset_metadata=args.dataset_metadata,
            historical_reports=(
                Path("reports/fall_risk/gait-action-pretraining-20260803.md"),
                Path("reports/fall_risk/gait_window_v5_effect_first/development-20260803/README.md"),
            ),
        )
        write_gait_training_audit(
            report,
            manifest_output=args.manifest_output,
            audit_output=args.audit_output,
            blockers_output=args.blockers_output,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest_id": report["manifest_id"], "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
