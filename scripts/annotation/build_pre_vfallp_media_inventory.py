#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.data_manifest import (
    write_pre_vfallp_media_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a content-addressed Pre_VFallp local media inventory for "
            "an internal authorization record."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-id", required=True)
    parser.add_argument("--subset", action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = write_pre_vfallp_media_inventory(
        args.repo_root,
        args.output,
        inventory_id=args.inventory_id,
        subsets=args.subset,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
