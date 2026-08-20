#!/usr/bin/env python
"""Validate/canonicalize caller-provided wandering tracking and sidecar files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    prepare_authorized_camera_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize an existing tracking JSONL + wandering-media-v1 pair; no media decode."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    result = prepare_authorized_camera_session(
        project_root=root,
        receipt_path=args.receipt,
        collection_path=args.collection,
        tracking_jsonl_path=args.tracking,
        media_sidecar_path=args.media_sidecar,
        output_dir=args.output_dir,
        collection_config_path=root / "configs/data/wandering_camera_collection_v1.yaml",
        camera_config_path=root / "configs/modules/wandering_camera_v1.yaml",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
