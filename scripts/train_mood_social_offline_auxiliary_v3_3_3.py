"""Train MODEL-006 PHQ-9 offline auxiliary models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.mood_social.offline_auxiliary import (
    OfflineAuxiliaryError,
    train_offline_auxiliary_models,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mood_social_offline_auxiliary_v3_3_3.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = _repository_root()
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        result = train_offline_auxiliary_models(
            root,
            config,
            overwrite=args.overwrite,
            command=sys.argv,
        )
    except OfflineAuxiliaryError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "task_id": result["task_id"],
                "run_id": result["run_id"],
                "model_path": str(result["model_path"]),
                "manifest_path": str(result["manifest_path"]),
                "report_directory": str(result["report_directory"]),
                "model_sha256": result["model_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "report_core_sha256": result["report_core_sha256"],
                "oof_row_count": result["oof_row_count"],
                "available_row_count": result["available_row_count"],
                "warning_count": result["warning_count"],
                "baseline_protection_sha256": result["baseline_protection_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
