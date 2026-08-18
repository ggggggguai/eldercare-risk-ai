from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.near_fall_action_auxiliary import (
    DEFAULT_AUXILIARY_LOSS_WEIGHT,
    build_action_auxiliary_near_fall_dataset,
)
from elderly_monitoring.modules.fall_risk.near_fall_training import NearFallDatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Append low-weight A01/A04 action auxiliary negatives to a near-fall dataset.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--action-labels", type=Path, default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"))
    parser.add_argument("--assignments", type=Path, default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fall_risk_video_manifest.jsonl"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=DEFAULT_AUXILIARY_LOSS_WEIGHT)
    args = parser.parse_args()
    result = build_action_auxiliary_near_fall_dataset(
        base_dataset_path=args.base_dir / "dataset.npz",
        base_metadata_path=args.base_dir / "metadata.json",
        base_samples_path=args.base_dir / "samples.jsonl",
        action_labels_path=args.action_labels,
        assignments_path=args.assignments,
        manifest_path=args.manifest,
        pose_dir=args.pose_dir,
        output_dir=args.output_dir,
        config=NearFallDatasetConfig(fallback_window_secs=(2.0,)),
        auxiliary_loss_weight=args.auxiliary_loss_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

