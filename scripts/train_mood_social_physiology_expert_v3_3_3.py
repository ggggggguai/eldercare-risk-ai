"""Train the V3.3.3 PhysiologyExpert under the frozen nested split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from elderly_monitoring.modules.mental_health.mood_social.experts.physiology import (
    PhysiologyExpertError,
    train_physiology_expert,
)


DEFAULT_CONFIG = Path("configs/training/mood_social_physiology_expert_v3_3_3.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-manifest-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    config = (
        args.config.resolve()
        if args.config is not None
        else repository_root / DEFAULT_CONFIG
    )
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    try:
        result = train_physiology_expert(
            repository_root,
            config,
            report_dir=args.report_dir,
            model_path=args.model_path,
            model_manifest_path=args.model_manifest_path,
            overwrite=args.overwrite,
            command=command,
        )
    except (PhysiologyExpertError, OSError, ValueError, RuntimeError) as exc:
        print(f"MODEL-004 training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
