#!/usr/bin/env python3
"""Create seeded per-class contact sheets from a strict trajectory JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    TrajectoryDatasetError,
    load_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.visualization import (
    WanderingVisualizationError,
    write_visual_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--input-role", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        samples = load_trajectory_jsonl(args.input)
        bundle = write_visual_review(
            samples,
            source_name=args.source_name,
            input_role=args.input_role,
            input_sha256=_sha256_file(args.input),
            output_dir=args.output,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    except (TrajectoryDatasetError, WanderingVisualizationError, OSError, ValueError) as exc:
        print(f"Wandering visualization failed: {exc}", file=sys.stderr)
        return 2
    selection = json.loads(bundle.selection_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_dir": str(bundle.output_dir),
                "review_status": selection["review_status"],
                "class_counts": {
                    label: len(sample_ids)
                    for label, sample_ids in selection["classes"].items()
                },
                "plots": selection["plots"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
