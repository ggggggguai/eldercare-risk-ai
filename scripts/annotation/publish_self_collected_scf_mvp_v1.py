#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.scf_publication import (
    publish_scf_mvp_v1_training_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the reviewed SCF_MVP_V1 auxiliary-train subset as v2 labels."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/self_collected_scf_mvp_v1_candidate"),
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path("configs/data/self_collected_scf_mvp_v1_decision.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/SCF_MVP_V1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = publish_scf_mvp_v1_training_batch(
        candidate_dir=args.candidate_dir,
        decision_path=args.decision,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
