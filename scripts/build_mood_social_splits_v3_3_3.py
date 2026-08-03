"""Build the frozen DATA-007 mood-social nested participant split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.mental_health.mood_social.splits import (
    DEFAULT_SPLIT_RELATIVE,
    build_and_write_split_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    output_path = (
        args.output_path
        if args.output_path is not None
        else repository_root / DEFAULT_SPLIT_RELATIVE
    )
    manifest, digest = build_and_write_split_manifest(
        repository_root=repository_root,
        output_path=output_path,
        overwrite=args.overwrite,
    )
    summary = {
        "dataset_count": manifest["totals"]["dataset_count"],
        "inner_fold_count": manifest["protocol"]["inner_fold_count"],
        "outer_fold_count": manifest["protocol"]["outer_fold_count"],
        "participant_count": manifest["totals"]["participant_count"],
        "positive_row_count": manifest["totals"]["positive_row_count"],
        "row_count": manifest["totals"]["row_count"],
        "split_id": manifest["split_id"],
        "split_manifest_sha256": digest,
        "status": "pass",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
