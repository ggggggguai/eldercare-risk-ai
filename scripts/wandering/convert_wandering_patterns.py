#!/usr/bin/env python3
"""Isolate the WanderingPatterns pickle, then build the strict JSONL bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.converters import (
    WanderingConversionError,
    convert_wandering_patterns_safe_extract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--primary-relative-path", default="raw/patterns_dataset.pkl"
    )
    parser.add_argument(
        "--duplicate-relative-path",
        default="source_repository/model_data/patterns_dataset.pkl",
    )
    parser.add_argument("--isolation-timeout-seconds", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = _convert(args)
    except (
        WanderingConversionError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"WanderingPatterns conversion failed: {exc}", file=sys.stderr)
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


def _convert(args: argparse.Namespace):
    if not sys.platform.startswith("linux"):
        raise WanderingConversionError(
            "WanderingPatterns pickle extraction requires Linux/WSL unshare isolation"
        )
    unshare = shutil.which("unshare")
    if unshare is None:
        raise WanderingConversionError("unshare is required for pickle isolation")
    if (
        isinstance(args.isolation_timeout_seconds, bool)
        or args.isolation_timeout_seconds <= 0
    ):
        raise WanderingConversionError("isolation timeout must be positive")

    source_root = args.source_root.resolve(strict=True)
    primary = (source_root / args.primary_relative_path).resolve(strict=True)
    duplicate = (source_root / args.duplicate_relative_path).resolve(strict=True)
    if not primary.is_file() or not duplicate.is_file():
        raise WanderingConversionError("primary and duplicate pickle files are required")
    primary_hash = _sha256_file(primary)
    duplicate_hash = _sha256_file(duplicate)
    if primary_hash != duplicate_hash:
        raise WanderingConversionError(
            "WanderingPatterns primary and duplicate pickle SHA-256 values differ"
        )

    script_dir = Path(__file__).resolve().parent
    wrapper = script_dir / "_run_isolated_pickle_extract.sh"
    extractor = script_dir / "_extract_wandering_patterns_pickle.py"
    python_env = Path(sys.prefix).resolve(strict=True)
    if not wrapper.is_file() or not extractor.is_file():
        raise WanderingConversionError("pickle isolation helper files are missing")

    with tempfile.TemporaryDirectory(prefix="wandering-pickle-extract-") as temp_dir:
        extract_dir = Path(temp_dir).resolve(strict=True)
        command = [
            unshare,
            "--user",
            "--map-root-user",
            "--net",
            "--mount",
            "--pid",
            "--fork",
            "--mount-proc",
            "/bin/bash",
            str(wrapper),
            str(primary),
            str(extract_dir),
            str(extractor),
            str(python_env),
            primary_hash,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.isolation_timeout_seconds,
            cwd="/",
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no detail"
            raise WanderingConversionError(
                f"isolated pickle extraction exited {completed.returncode}: {detail}"
            )
        return convert_wandering_patterns_safe_extract(
            source_root,
            args.output,
            extracted_jsonl=extract_dir / "rows.jsonl",
            extraction_metadata=extract_dir / "metadata.json",
            primary_relative_path=args.primary_relative_path,
            duplicate_relative_path=args.duplicate_relative_path,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
