from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.data_manifest import (
    write_fall_risk_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic fall-risk data asset manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing data/external.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
        help="Repository-relative or absolute output JSONL path.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable used for per-video metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output. The default refuses overwrite.",
    )
    args = parser.parse_args()

    result = write_fall_risk_manifest(
        args.repo_root,
        args.output,
        overwrite=args.overwrite,
        ffprobe_bin=args.ffprobe_bin,
    )
    output = args.output
    if output.is_absolute():
        try:
            output = output.resolve().relative_to(args.repo_root.resolve())
        except ValueError:
            pass
    summary = {**result.summary, "output": output.as_posix()}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
