#!/usr/bin/env python
"""Independent authorized camera-development entry for fixed TopoWander-MPT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_development import (
    run_authorized_camera_development,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the receipt-gated M0-CAM development path.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--mode", choices=("engineering_smoke", "labeled_evaluation"), required=True
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--evaluation-policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_authorized_camera_development(
        project_root=args.project_root,
        config_path=args.config,
        receipt_path=args.receipt,
        collection_path=args.collection,
        tracking_path=args.tracking,
        media_sidecar_path=args.media_sidecar,
        candidate_manifest_path=args.candidate_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        mode=args.mode,
        annotations_path=args.annotations,
        evaluation_policy_path=args.evaluation_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
