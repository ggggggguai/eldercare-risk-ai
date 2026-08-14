from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SitStandContinuousConfig,
    prepare_sit_stand_continuous_dataset,
)
from elderly_monitoring.modules.fall_risk.sit_stand_event_labels import (
    validate_sit_stand_event_labels,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train/validation causal sit-stand windows; test pose stays locked."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review-log", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--context-sec", type=float, default=8.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.5)
    parser.add_argument("--min-observed-frames", type=int, default=16)
    parser.add_argument("--min-partial-observed-frames", type=int, default=6)
    parser.add_argument("--target-track-min-coverage", type=float, default=0.6)
    parser.add_argument("--target-track-min-dominance", type=float, default=1.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = _read_jsonl(args.labels)
        validate_sit_stand_event_labels(labels, _read_jsonl(args.review_log))

        def pose_reader(video_id: str) -> list[dict]:
            path = args.pose_dir / f"{video_id}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"pose JSONL not found for {video_id}: {path}")
            return _read_jsonl(path)

        result = prepare_sit_stand_continuous_dataset(
            labels,
            _read_jsonl(args.assignments),
            pose_reader=pose_reader,
            output_dir=args.output_dir,
            config=SitStandContinuousConfig(
                target_fps=args.target_fps,
                context_sec=args.context_sec,
                max_gap_sec=args.max_gap_sec,
                min_observed_frames=args.min_observed_frames,
                min_partial_observed_frames=args.min_partial_observed_frames,
                target_track_min_coverage=args.target_track_min_coverage,
                target_track_min_dominance=args.target_track_min_dominance,
            ),
            manifest=_read_jsonl(args.manifest),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
