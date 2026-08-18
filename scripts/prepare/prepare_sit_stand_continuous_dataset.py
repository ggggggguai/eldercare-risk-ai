from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SitStandContinuousConfig,
    SitStandSamplingConfig,
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
    parser.add_argument(
        "--pose-dir",
        type=Path,
        action="append",
        required=True,
        help="Pose directory; repeat for additional governed sources such as SCF.",
    )
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
    parser.add_argument(
        "--sampling-policy",
        choices=("single_offset_v1", "balanced_causal_v2"),
        default="single_offset_v1",
    )
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
    parser.add_argument(
        "--event-cutoff-fractions",
        type=float,
        nargs="+",
        default=(0.5, 0.75, 1.0),
        help="Event progress fractions used by balanced_causal_v2.",
    )
    parser.add_argument(
        "--post-event-offsets-sec",
        type=float,
        nargs="+",
        default=(0.5,),
        help="Post-event background offsets used by balanced_causal_v2.",
    )
    parser.add_argument(
        "--target-weight-shares",
        type=float,
        nargs=3,
        metavar=("BACKGROUND", "SIT_TO_STAND", "STAND_TO_SIT"),
        default=(0.5, 0.25, 0.25),
        help="Relative background, sit-to-stand and stand-to-sit loss weights.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = _read_jsonl(args.labels)
        validate_sit_stand_event_labels(labels, _read_jsonl(args.review_log))

        def pose_reader(video_id: str) -> list[dict]:
            candidates = [path / f"{video_id}.jsonl" for path in args.pose_dir]
            matches = [path for path in candidates if path.is_file()]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected exactly one pose JSONL for {video_id}; found {matches}"
                )
            return _read_jsonl(matches[0])

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
            sampling=(
                SitStandSamplingConfig(
                    event_cutoff_fractions=tuple(args.event_cutoff_fractions),
                    post_event_offsets_sec=tuple(args.post_event_offsets_sec),
                    auxiliary_weight=args.auxiliary_weight,
                    target_weight_shares=tuple(args.target_weight_shares),
                )
                if args.sampling_policy == "balanced_causal_v2"
                else None
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
