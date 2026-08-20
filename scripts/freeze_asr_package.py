"""Freeze the local ASR model package into a deterministic SHA-256 manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from elderly_monitoring.modules.asr.integrity import (
    CHECKSUM_FILENAME,
    render_checksum_manifest,
    verify_model_package,
)
from elderly_monitoring.modules.asr.settings import ASRSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ASRSettings().model_root,
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().model_root.expanduser().resolve()
    destination = root / CHECKSUM_FILENAME
    temporary = root / f".{CHECKSUM_FILENAME}.tmp"
    temporary.write_text(
        render_checksum_manifest(root),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)
    manifest_sha256 = verify_model_package(root)
    print(f"model_root={root}")
    print(f"manifest={destination}")
    print(f"manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
