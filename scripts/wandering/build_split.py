#!/usr/bin/env python3
"""Build the hash-frozen wandering split-v1 artifacts into a new directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.splits import (
    SplitDataError,
    build_wandering_split_from_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/wandering_split_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/mental_health/wandering/v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_wandering_split_from_files(
            config_path=args.config,
            project_root=args.project_root,
            output_dir=args.output,
        )
    except (FileExistsError, OSError, SplitDataError) as exc:
        print(f"Wandering split build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "split_sha256": result.split.split_sha256,
                "report_payload_sha256": result.report["report_payload_sha256"],
                "partition_counts": result.report["partition_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
