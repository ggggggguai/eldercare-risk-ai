#!/usr/bin/env python
"""Restore fixed M0-CAM runtime assets from one verified offline archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
    PortableAssetBlockedError,
    PortableContractError,
    restore_camera_runtime_assets,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _parse_args() -> argparse.Namespace:
    parser = _Parser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = restore_camera_runtime_assets(
            project_root=args.project_root,
            archive_path=args.archive,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except PortableAssetBlockedError as exc:
        print(
            json.dumps(
                {"status": "blocked", "blocker_code": exc.blocker_code},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    except (PortableContractError, FileExistsError, OSError, ValueError) as exc:
        print(f"camera runtime asset restore failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
