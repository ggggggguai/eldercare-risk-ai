#!/usr/bin/env python3
"""Build the frozen wandering step-4 preprocessing bundle in a new directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    TrajectoryDatasetError,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    PreprocessingDataError,
    build_wandering_preprocessing_from_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/wandering_preprocessing_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wandering/preprocessing/v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_wandering_preprocessing_from_files(
            config_path=args.config,
            project_root=args.project_root,
            output_dir=args.output,
        )
    except (FileExistsError, OSError, PreprocessingDataError, TrajectoryDatasetError) as exc:
        print(f"Wandering preprocessing build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "split_sha256": result.report["split_sha256"],
                "counts": result.report["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
