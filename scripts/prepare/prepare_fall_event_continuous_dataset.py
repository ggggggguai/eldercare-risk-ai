from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from elderly_monitoring.modules.fall_risk.fall_event_continuous_dataset import (
    ContinuousFallDatasetConfig,
    build_continuous_fall_dataset,
)


DEFAULT_GOVERNANCE = Path(
    "reports/fall_risk/fall_event_continuous_governance_v1/training_manifest.jsonl"
)
DEFAULT_POSE_ROOTS = (
    Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"),
    Path("reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        description="Materialize train/validation-only causal fall-event windows."
    )
    parser.add_argument("--governance-manifest", type=Path, default=DEFAULT_GOVERNANCE)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument("--pose-root", type=Path, action="append", dest="pose_roots")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-observed-frames", type=int, default=16)
    parser.add_argument("--min-partial-observed-frames", type=int, default=8)
    parser.add_argument("--min-valid-joint-ratio", type=float, default=0.50)
    parser.add_argument("--max-interpolated-joint-ratio", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_continuous_fall_dataset(
            _read_jsonl(args.governance_manifest),
            _read_jsonl(args.manifest),
            pose_roots=args.pose_roots or DEFAULT_POSE_ROOTS,
            output_dir=args.output_dir,
            config=ContinuousFallDatasetConfig(
                window_sec=args.window_sec,
                target_fps=args.target_fps,
                max_gap_sec=args.max_gap_sec,
                min_observed_frames=args.min_observed_frames,
                min_partial_observed_frames=args.min_partial_observed_frames,
                min_valid_joint_ratio=args.min_valid_joint_ratio,
                max_interpolated_joint_ratio=args.max_interpolated_joint_ratio,
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
