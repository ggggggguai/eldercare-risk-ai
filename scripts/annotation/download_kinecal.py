from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.fall_risk.kinecal import (
    KINECAL_BASE_URL,
    download_kinecal,
    verify_kinecal_download,
)


DEFAULT_OUTPUT_DIR = Path("data/external/kinecal/raw")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the KINECAL v1.0.3 risk-group skeleton subset for gait TCN experiments."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=KINECAL_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep available files and return success even if recordings are missing.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the existing local manifest, file sizes and SHA-256 hashes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if args.verify_only:
        verification = verify_kinecal_download(args.output_dir)
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if verification["errors"] else 0

    try:
        summary = download_kinecal(
            args.output_dir,
            base_url=args.base_url,
            workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
            strict=not args.allow_missing,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
