#!/usr/bin/env python
"""Reject oversized, private or runtime-only files from a deployment branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_ROOTS = {
    "models",
    "data",
    "reports",
    "artifacts",
    "training",
    "tmp",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".npz",
    ".joblib",
    ".cbm",
    ".onnx",
    ".safetensors",
    ".wav",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".jpg",
    ".jpeg",
    ".png",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[dict[str, object]] = []
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        count += 1
        size = path.stat().st_size
        total += size
        reason = None
        if relative.parts[0] in FORBIDDEN_ROOTS:
            reason = "forbidden_runtime_root"
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            reason = "forbidden_binary_or_media_suffix"
        elif size > args.max_bytes:
            reason = "file_too_large"
        elif relative.name == ".env" or (
            relative.name.startswith(".env.") and relative.name != ".env.example"
        ):
            reason = "secret_environment_file"
        if reason:
            failures.append(
                {"path": relative.as_posix(), "size_bytes": size, "reason": reason}
            )
    result = {
        "status": "failed" if failures else "passed",
        "file_count": count,
        "total_bytes": total,
        "max_file_bytes": args.max_bytes,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
