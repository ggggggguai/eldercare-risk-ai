from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.near_fall_self_collected import (
    build_augmented_near_fall_dataset,
)
from elderly_monitoring.modules.fall_risk.near_fall_training import (
    NearFallDatasetConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an E1/E2/E3 SCF near-fall augmentation dataset.")
    parser.add_argument("--experiment", choices=("E1", "E2", "E3"), required=True)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            "data/processed/fall_risk/near_fall_event_v2/"
            "splitv3-c7fd01cf-input992017e2-base"
        ),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/self_collected_scf_mvp_v1_candidate"),
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_augmented_near_fall_dataset(
        base_dataset_path=args.base_dir / "dataset.npz",
        base_metadata_path=args.base_dir / "metadata.json",
        base_samples_path=args.base_dir / "samples.jsonl",
        candidates_path=args.candidate_dir / "near_fall_candidates.jsonl",
        manifest_path=args.candidate_dir / "manifest.jsonl",
        pose_dir=args.pose_dir,
        output_dir=args.output_dir,
        experiment=args.experiment,
        config=NearFallDatasetConfig(fallback_window_secs=(2.0,)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
