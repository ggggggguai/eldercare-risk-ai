#!/usr/bin/env python
"""Verify externally restored psychological runtime assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.service.mental_health_assets import (
    MentalHealthAssetError,
    load_asset_manifest,
    verify_asset_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("deploy/mental_health/asset_manifest.json"),
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            {
                "status": "passed",
                "asset_count": len(load_asset_manifest(args.manifest)["assets"]),
                "scope": "manifest_only",
            }
            if args.manifest_only
            else verify_asset_manifest(
                project_root=args.project_root,
                manifest_path=args.manifest,
            )
        )
    except MentalHealthAssetError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
