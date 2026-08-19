#!/usr/bin/env python
"""Run the zero-video M0-CAM runtime deployability preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
    PortableContractError,
    preflight_camera_runtime,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _parse_args() -> argparse.Namespace:
    parser = _Parser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = preflight_camera_runtime(
            project_root=args.project_root,
            output_path=args.output,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["status"] == "ready" else 2
    except (PortableContractError, FileExistsError, OSError, ValueError) as exc:
        print(f"camera runtime preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
