"""Build deterministic DATA-005 Shenzhen V3.3.3 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from elderly_monitoring.datasets.adapters.shenzhen import build_shenzhen_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--manifests-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_shenzhen_artifacts(
        workspace_root=args.workspace_root,
        manifests_root=args.manifests_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    summary = {
        "artifact_sha256": dict(result.artifact_sha256),
        "dataset_id": "shenzhen_elderly",
        "participant_count": result.participant_count,
        "positive_row_count": result.positive_row_count,
        "row_count": result.row_count,
        "status": "pass",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
