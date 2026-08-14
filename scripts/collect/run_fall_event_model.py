from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.fall_event_tcn import (
    FallEventTCNEnsemblePredictor,
    FallEventTCNPredictor,
)
from elderly_monitoring.modules.fall_risk.fall_event_training import (
    FallEventDatasetConfig,
    _resample_pose_records,
    _window_quality,
    build_fall_event_tensor,
)


DEFAULT_CHECKPOINTS = (
    Path(
        "reports/fall_risk/fall_event_proxy_v1/"
        "development-splitv3-e71a045-seed42/best_model.pt"
    ),
    Path(
        "reports/fall_risk/fall_event_proxy_v1/"
        "development-splitv3-e71a045-seed43/best_model.pt"
    ),
    Path(
        "reports/fall_risk/fall_event_proxy_v1/"
        "development-splitv3-e71a045-seed44/best_model.pt"
    ),
)
DEFAULT_ENSEMBLE_THRESHOLD = 0.42490479350090027


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the provisional fall-event TCN on candidate windows from a pose JSONL stream."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint path; repeat for mean-probability ensemble inference.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-observed-frames", type=int, default=12)
    parser.add_argument("--min-usable-frame-ratio", type=float, default=0.60)
    parser.add_argument("--min-core-joint-coverage", type=float, default=0.70)
    parser.add_argument(
        "--track-id",
        default=None,
        help="Optional track ID; by default the track with the most observed frames is used.",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_track(records: Sequence[dict[str, Any]], track_id: str | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("track_id", record.get("person_id", "unknown")))].append(
            dict(record)
        )
    if not grouped:
        return []
    selected = str(track_id) if track_id is not None else max(
        grouped,
        key=lambda value: len(grouped[value]),
    )
    if selected not in grouped:
        raise ValueError(f"requested track_id is not present in input: {selected}")
    return sorted(
        grouped[selected],
        key=lambda row: (
            float(row.get("timestamp_sec", 0.0)),
            int(row.get("frame_id", 0)),
        ),
    )


def _window_starts(start: float, end: float, window_sec: float, stride_sec: float) -> list[float]:
    if end <= start:
        return []
    if end - start <= window_sec:
        return [start]
    last = end - window_sec
    starts = list(np.arange(start, last + 1e-9, stride_sec, dtype=np.float64))
    if not starts or abs(float(starts[-1]) - last) > 1e-6:
        starts.append(last)
    return [float(value) for value in starts]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def run(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise FileNotFoundError(f"fall-event pose input not found: {args.input}")
    if args.window_sec <= 0 or args.stride_sec <= 0:
        raise ValueError("window-sec and stride-sec must be positive")
    config = FallEventDatasetConfig(
        target_fps=args.target_fps,
        window_sec=args.window_sec,
        max_gap_sec=args.max_gap_sec,
        min_observed_frames=args.min_observed_frames,
        min_usable_frame_ratio=args.min_usable_frame_ratio,
        min_core_joint_coverage=args.min_core_joint_coverage,
    )
    records = _select_track(_read_jsonl(args.input), args.track_id)
    if not records:
        raise ValueError("pose input contains no track records")
    timestamps = [float(row["timestamp_sec"]) for row in records if row.get("timestamp_sec") is not None]
    if not timestamps:
        raise ValueError("pose input contains no timestamp_sec values")
    using_default_ensemble = not args.checkpoint
    checkpoint_paths = args.checkpoint or list(DEFAULT_CHECKPOINTS)
    if len(checkpoint_paths) == 1:
        predictor = FallEventTCNPredictor(
            checkpoint_paths[0],
            device=args.device,
            threshold=args.threshold,
        )
    else:
        predictor = FallEventTCNEnsemblePredictor(
            checkpoint_paths,
            device=args.device,
            threshold=(
                DEFAULT_ENSEMBLE_THRESHOLD
                if args.threshold is None and using_default_ensemble
                else (0.5 if args.threshold is None else args.threshold)
            ),
        )
    outputs: list[dict[str, Any]] = []
    starts = _window_starts(min(timestamps), max(timestamps), config.window_sec, args.stride_sec)
    for window_index, start in enumerate(starts):
        slots = _resample_pose_records(
            records,
            start_time_sec=start,
            window_frames=config.window_frames,
            target_fps=config.target_fps,
            max_gap_sec=config.max_gap_sec,
        )
        observed = [row for row in slots if row is not None]
        quality = _window_quality(slots)
        base = {
            "window_index": window_index,
            "window_start_time_sec": round(start, 6),
            "window_end_time_exclusive": round(start + config.window_sec, 6),
            "track_id": str(records[0].get("track_id", records[0].get("person_id", "unknown"))),
            "observed_frame_count": len(observed),
            "quality": quality,
            "model_version": predictor.model_version,
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "status": "provisional_shadow",
        }
        if (
            len(observed) < config.min_observed_frames
            or quality["usable_frame_ratio"] < config.min_usable_frame_ratio
            or quality["core_joint_coverage"] < config.min_core_joint_coverage
        ):
            base.update(
                {
                    "status": "unavailable",
                    "reason": "insufficient_fall_event_quality",
                    "fall_event_score": None,
                    "fall_event_detected": False,
                }
            )
        else:
            tensor = build_fall_event_tensor(
                slots,
                window_frames=config.window_frames,
                target_fps=config.target_fps,
            )
            base.update(predictor.predict_tensor(tensor))
        outputs.append(base)
    _write_jsonl(args.output, outputs)
    print(json.dumps({
        "input": args.input.as_posix(),
        "input_sha256": _sha256(args.input),
        "output": args.output.as_posix(),
        "window_count": len(outputs),
        "status": "provisional_shadow",
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
