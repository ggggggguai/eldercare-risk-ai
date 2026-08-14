#!/usr/bin/env python
"""Validate and canonicalize caller-provided human camera annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    write_authorized_camera_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate caller-provided human episode annotations; labels are never inferred."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    result = write_authorized_camera_annotations(
        receipt_path=args.receipt,
        collection_path=args.collection,
        annotations_path=args.annotations,
        output_path=args.output,
        collection_config_path=root / "configs/data/wandering_camera_collection_v1.yaml",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
