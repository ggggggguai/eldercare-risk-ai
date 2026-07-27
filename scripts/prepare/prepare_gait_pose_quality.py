from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.gait_training import (
    select_gait_training_labels,
)
from elderly_monitoring.modules.fall_risk.pose import run_yolov8_pose, write_jsonl
from elderly_monitoring.modules.fall_risk.pose_quality import process_pose_records


def build_pose_jobs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = select_gait_training_labels(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["video_id"])].append(row)

    jobs: list[dict[str, Any]] = []
    for video_id in sorted(grouped):
        labels = grouped[video_id]
        paths = {str(row.get("file_path", "")) for row in labels}
        if len(paths) != 1 or not next(iter(paths)):
            raise ValueError(f"{video_id} has inconsistent or missing source video paths")
        scenes = {str(row.get("scene", "unknown")) for row in labels}
        if len(scenes) != 1:
            raise ValueError(f"{video_id} has inconsistent scene labels")
        jobs.append(
            {
                "video_id": video_id,
                "video_path": next(iter(paths)),
                "scene": next(iter(scenes)),
                "max_frames": max(int(row["end_frame"]) for row in labels) + 1,
                "label_count": len(labels),
            }
        )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and quality-control pose inputs for labeled gait windows."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/gait_pose_quality"),
    )
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = _read_jsonl(args.labels)
        jobs = build_pose_jobs(rows)
        if args.max_videos is not None:
            if args.max_videos < 1:
                raise ValueError("max-videos must be positive")
            jobs = jobs[: args.max_videos]
        if not jobs:
            raise ValueError("no labeled gait videos were selected")
        from ultralytics import YOLO

        model = YOLO(args.model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = args.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        completed: list[dict[str, Any]] = []
        for index, job in enumerate(jobs, start=1):
            video_path = Path(job["video_path"])
            if not video_path.is_file():
                raise FileNotFoundError(f"source gait video not found: {video_path}")
            cleaned_path = args.output_dir / f"{job['video_id']}.jsonl"
            raw_path = raw_dir / f"{job['video_id']}.jsonl"
            if cleaned_path.exists() and not args.overwrite:
                completed.append({**job, "status": "skipped_existing"})
                print(f"[{index}/{len(jobs)}] skip {job['video_id']}", flush=True)
                continue
            print(f"[{index}/{len(jobs)}] pose {job['video_id']}", flush=True)
            raw_count = run_yolov8_pose(
                video_path=video_path,
                output_path=raw_path,
                model_name=args.model,
                scene_region=str(job["scene"]),
                person_id_prefix=job["video_id"],
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
                tracker_config=args.tracker,
                max_frames=int(job["max_frames"]),
                normalize_coordinates=True,
                device=args.device,
                model=model,
            )
            cleaned = process_pose_records(_read_jsonl(raw_path))
            partial = cleaned_path.with_suffix(f"{cleaned_path.suffix}.part")
            cleaned_count = write_jsonl(cleaned, partial)
            os.replace(partial, cleaned_path)
            completed.append(
                {
                    **job,
                    "status": "completed",
                    "raw_pose_count": raw_count,
                    "cleaned_pose_count": cleaned_count,
                }
            )
        summary = {
            "schema_version": "gait-pose-quality-preparation-v1",
            "labels_path": args.labels.as_posix(),
            "model": args.model,
            "device": args.device,
            "confidence": args.confidence,
            "iou": args.iou,
            "tracker": args.tracker,
            "jobs": completed,
        }
        summary_path = args.output_dir / "preparation_summary.json"
        partial = summary_path.with_suffix(f"{summary_path.suffix}.part")
        partial.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, summary_path)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "video_count": len(jobs)}))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
