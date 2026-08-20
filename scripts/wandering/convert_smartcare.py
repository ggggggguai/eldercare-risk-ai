#!/usr/bin/env python3
"""Convert the two inert SmartCare JSON sources into a strict bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.converters import (
    WanderingConversionError,
    convert_smartcare,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--main-relative-path", default="raw/dataset.json")
    parser.add_argument(
        "--validation-relative-path", default="raw/dataset-validacao.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = convert_smartcare(
            args.source_root,
            args.output,
            main_relative_path=args.main_relative_path,
            validation_relative_path=args.validation_relative_path,
        )
    except (WanderingConversionError, OSError, ValueError) as exc:
        print(f"SmartCare conversion failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(bundle.output_dir),
                "sample_count_read": bundle.report.sample_count_read,
                "sample_count_written": bundle.report.sample_count_written,
                "sample_count_rejected": bundle.report.sample_count_rejected,
                "class_counts": dict(bundle.report.class_counts),
                "group_fields_found": list(bundle.report.group_fields_found),
                "manifest_sha256": bundle.manifest.sha256,
                "report_sha256": bundle.report.sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
