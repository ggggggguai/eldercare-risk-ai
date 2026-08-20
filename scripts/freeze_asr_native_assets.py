"""Freeze the FFmpeg/FFprobe binaries used by the ASR decoder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from elderly_monitoring.modules.asr.runtime_integrity import (
    render_native_assets_manifest,
    verify_native_assets,
)
from elderly_monitoring.modules.asr.settings import ASRSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ASRSettings().native_assets_manifest,
    )
    return parser.parse_args()


def main() -> int:
    settings = ASRSettings(native_assets_manifest=parse_args().output)
    destination = settings.native_assets_manifest.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(render_native_assets_manifest(settings), encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    verify_native_assets(settings)
    print(f"manifest={destination}")
    print(f"ffmpeg={settings.ffmpeg_path}")
    print(f"ffprobe={settings.ffprobe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
