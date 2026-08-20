#!/usr/bin/env python
"""Verify the ASR external asset bundle and optional native runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.asr.deployment_assets import (
    ASRDeploymentAssetError,
    load_asr_asset_manifest,
    verify_asr_assets,
)
from elderly_monitoring.modules.asr.runtime_integrity import (
    ASRNativeRuntimeError,
    verify_native_assets,
)
from elderly_monitoring.modules.asr.settings import ASRSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("deploy/asr/asset_manifest.json"),
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--verify-native", action="store_true")
    args = parser.parse_args()
    try:
        if args.manifest_only:
            payload = load_asr_asset_manifest(args.manifest)
            result = {
                "status": "passed",
                "scope": "manifest_only",
                "asset_count": len(payload["assets"]),
            }
        else:
            result = verify_asr_assets(
                project_root=args.project_root,
                manifest_path=args.manifest,
            )
        if args.verify_native:
            verify_native_assets(ASRSettings.load())
            result["native_runtime"] = "passed"
    except (ASRDeploymentAssetError, ASRNativeRuntimeError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
