from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from elderly_monitoring.modules.fall_risk.splits import build_splits_from_files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic, leakage-audited splits for four fall-risk tasks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
        help="Unified fall-risk manifest JSONL.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations/fall_risk"),
        help="Directory containing the configured label JSONL files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/fall_risk_splits_v1.yaml"),
        help="Versioned split protocol YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/fall_risk"),
        help="Output root; each task is written to its own version directory.",
    )
    parser.add_argument(
        "--overwrite-development",
        action="store_true",
        help="Replace existing non-frozen development outputs; frozen splits remain immutable.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="Required formal validator report when protocol_status is frozen.",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("configs/data/fall_risk_label_validation_v1.yaml"),
        help="Validator config whose SHA-256 is bound by a frozen formal report.",
    )
    args = parser.parse_args(argv)

    try:
        artifacts = build_splits_from_files(
            manifest_path=args.manifest,
            annotations_dir=args.annotations_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            overwrite_development=args.overwrite_development,
            validation_report_path=args.validation_report,
            validation_config_path=args.validation_config,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {
        split_name: {
            "status": artifact["metadata"]["status"],
            "split_id": artifact["metadata"]["split_id"],
            "eligible_sample_count": artifact["metadata"]["eligible_sample_count"],
            "blockers": artifact["metadata"]["blockers"],
        }
        for split_name, artifact in sorted(artifacts.items())
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
