from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.near_fall_label_publishing import (
    publish_near_fall_manual_labels_v3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish explicitly human-reviewed near-fall decisions as a v3 candidate. "
            "It never infers labels from rules, actions, directories or background."
        )
    )
    parser.add_argument(
        "--base-event-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "data/annotations/fall_risk/near_fall_manual_decisions_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-event-labels",
        type=Path,
        default=Path(
            "data/annotations/fall_risk/event_labels_v3.near_fall_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/fall_risk/near-fall-label-publication-v1.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_near_fall_manual_labels_v3(
            base_event_labels=args.base_event_labels,
            decisions=args.decisions,
            manifest=args.manifest,
            output_event_labels=args.output_event_labels,
            report=args.report,
            repo_root=args.repo_root,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
