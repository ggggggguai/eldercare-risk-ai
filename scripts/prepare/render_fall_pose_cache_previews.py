from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


COCO_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

TRACK_COLORS = (
    (72, 214, 118),
    (73, 174, 255),
    (255, 151, 76),
    (214, 113, 255),
    (255, 221, 87),
)

ACTION_LABELS = {
    "A008": "SIT DOWN",
    "A009": "STAND UP",
    "A042": "STAGGER",
    "A043": "FALL DOWN",
    "A080": "SQUAT",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render review videos from an existing cleaned fall-pose cache."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/fall_risk/pose_quality_y8n_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video-id",
        action="append",
        required=True,
        help="Manifest video_id to render; repeat for multiple samples.",
    )
    parser.add_argument("--width", type=int, default=960)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width < 320:
        print("error: --width must be at least 320", file=sys.stderr)
        return 2
    if len(set(args.video_id)) != len(args.video_id):
        print("error: duplicate --video-id", file=sys.stderr)
        return 2

    try:
        import cv2

        manifests = _select_manifest_rows(args.manifest, set(args.video_id))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        stills: list[tuple[str, Any]] = []
        for video_id in args.video_id:
            sample, still = _render_sample(
                cv2=cv2,
                manifest=manifests[video_id],
                cleaned_path=args.cache_dir / "cleaned" / f"{video_id}.jsonl",
                output_dir=args.output_dir,
                target_width=args.width,
            )
            samples.append(sample)
            stills.append((sample["label"], still))

        datasets = sorted(
            {str(manifests[video_id].get("dataset", "unknown")) for video_id in args.video_id}
        )
        dataset_slug = datasets[0] if len(datasets) == 1 else "mixed"
        contact_sheet = args.output_dir / f"{dataset_slug}-pose-preview-contact-sheet.jpg"
        _write_contact_sheet(cv2, stills, contact_sheet)
        _write_json(
            args.output_dir / "preview_manifest.json",
            {
                "schema_version": "fall-pose-cache-preview-v1",
                "generated_at": datetime.now(UTC).isoformat(),
                "samples": samples,
                "contact_sheet": str(contact_sheet.resolve()),
            },
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _select_manifest_rows(
    manifest_path: Path, selected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id", ""))
            if video_id in selected_ids:
                if video_id in selected:
                    raise ValueError(f"duplicate video_id in manifest: {video_id}")
                selected[video_id] = row
    missing = sorted(selected_ids - set(selected))
    if missing:
        raise ValueError(f"video_id not found in manifest: {', '.join(missing)}")
    return selected


def _read_cleaned_records(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"cleaned pose cache not found: {path}")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                frame_id = int(record["frame_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid frame_id at {path}:{line_number}") from exc
            by_frame[frame_id].append(record)
    return dict(by_frame)


def _render_sample(
    *,
    cv2: Any,
    manifest: Mapping[str, Any],
    cleaned_path: Path,
    output_dir: Path,
    target_width: int,
) -> tuple[dict[str, Any], Any]:
    video_id = str(manifest["video_id"])
    source_path = Path(str(manifest["path"]))
    if not source_path.is_file():
        raise ValueError(f"source video not found: {source_path}")
    records_by_frame = _read_cleaned_records(cleaned_path)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(manifest["fps"])
    target_height = max(2, round(source_height * target_width / source_width))
    target_height += target_height % 2
    raw_path = output_dir / f"{video_id}.raw.mp4"
    preview_path = output_dir / f"{video_id}.mp4"
    still_path = output_dir / f"{video_id}.jpg"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width, target_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create preview video: {raw_path}")

    frame_count = 0
    detected_frames = 0
    multi_person_frames = 0
    pose_records = 0
    low_quality_records = 0
    still = None
    still_score: tuple[int, int, int] | None = None
    still_target = max(0, int(manifest["frame_count"]) // 2)
    action_code = str(manifest.get("source_action_code", ""))
    label = _build_label(manifest)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            frame_records = records_by_frame.get(frame_count, [])
            if frame_records:
                detected_frames += 1
            if len(frame_records) > 1:
                multi_person_frames += 1
            pose_records += len(frame_records)
            frame_low_quality = sum(
                record.get("quality_state") != "usable" for record in frame_records
            )
            low_quality_records += frame_low_quality
            _draw_frame_overlay(
                cv2=cv2,
                frame=frame,
                records=frame_records,
                video_id=video_id,
                label=label,
                frame_id=frame_count,
            )
            writer.write(frame)
            candidate_score = (
                frame_low_quality,
                len(frame_records),
                -abs(frame_count - still_target),
            )
            if still_score is None or candidate_score > still_score:
                still = frame.copy()
                still_score = candidate_score
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"source video contains no readable frames: {source_path}")
    if still is None:
        raise RuntimeError(f"cannot select representative frame: {raw_path}")
    if not cv2.imwrite(str(still_path), still):
        raise RuntimeError(f"cannot write representative frame: {still_path}")
    _transcode_h264(raw_path, preview_path)

    return (
        {
            "video_id": video_id,
            "action_code": action_code,
            "label": label,
            "source_path": str(source_path),
            "cleaned_pose_path": str(cleaned_path),
            "preview_path": str(preview_path.resolve()),
            "raw_preview_path": str(raw_path.resolve()),
            "still_path": str(still_path.resolve()),
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "duration_sec": round(frame_count / fps, 6),
            "pose_records": pose_records,
            "detected_frame_coverage": round(detected_frames / frame_count, 6),
            "multi_person_frame_ratio": round(multi_person_frames / frame_count, 6),
            "low_quality_record_ratio": round(
                low_quality_records / pose_records if pose_records else 0.0, 6
            ),
        },
        still,
    )


def _build_label(manifest: Mapping[str, Any]) -> str:
    action_code = str(manifest.get("source_action_code", ""))
    if action_code:
        return f"{action_code} {ACTION_LABELS.get(action_code, 'UNKNOWN')}"

    original_event_id = str(manifest.get("original_event_id", ""))
    match = re.search(r"(?:^|_)act_(\d+)(?:_|$)", original_event_id)
    action_label = f"ACT{match.group(1)}" if match else "ACT UNKNOWN"
    subset = str(manifest.get("subset", "unknown")).upper()
    scene = str(manifest.get("scene_region", "unknown")).upper()
    return f"{subset} {action_label} | {scene}"


def _draw_frame_overlay(
    *,
    cv2: Any,
    frame: Any,
    records: Sequence[Mapping[str, Any]],
    video_id: str,
    label: str,
    frame_id: int,
) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 48), (16, 16, 16), -1)
    cv2.putText(
        frame,
        f"{label} | {_compact_video_id(video_id)} | frame {frame_id}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    for record in records:
        track_id = int(record.get("track_id") or 0)
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        bbox = record.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = (
                int(float(bbox[0]) * width),
                int(float(bbox[1]) * height),
                int(float(bbox[2]) * width),
                int(float(bbox[3]) * height),
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            state = str(record.get("quality_state", "unknown"))
            quality = float(record.get("core_keypoint_quality") or 0.0)
            cv2.putText(
                frame,
                f"T{track_id} {state} q={quality:.2f}",
                (max(2, x1), max(62, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )

        points: dict[str, tuple[int, int, str]] = {}
        for keypoint in record.get("keypoints", []):
            if not keypoint.get("valid"):
                continue
            x = keypoint.get("x_smooth")
            y = keypoint.get("y_smooth")
            if x is None or y is None:
                continue
            points[str(keypoint["name"])] = (
                int(float(x) * width),
                int(float(y) * height),
                str(keypoint.get("source", "observed")),
            )
        for start_name, end_name in COCO_EDGES:
            if start_name not in points or end_name not in points:
                continue
            cv2.line(
                frame,
                points[start_name][:2],
                points[end_name][:2],
                color,
                3,
                cv2.LINE_AA,
            )
        for x, y, source in points.values():
            point_color = (0, 214, 255) if source == "interpolated" else color
            cv2.circle(frame, (x, y), 4, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, point_color, -1, cv2.LINE_AA)


def _compact_video_id(video_id: str) -> str:
    match = re.search(
        r"_sbj_(\d+)_loc_(\d+)_act_(\d+)_side_([a-z])_attempt_(\d+)", video_id
    )
    if match:
        subject, location, action, side, attempt = match.groups()
        return (
            f"SBJ{int(subject):02d} LOC{int(location)} ACT{int(action)} "
            f"{side.upper()}{int(attempt)}"
        )
    return video_id if len(video_id) <= 48 else f"{video_id[:45]}..."


def _transcode_h264(raw_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create browser-compatible H.264 video")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {raw_path}: {completed.stderr.strip()}")


def _write_contact_sheet(cv2: Any, stills: Sequence[tuple[str, Any]], path: Path) -> None:
    import numpy as np

    cell_width = 480
    cell_height = 300
    columns = 2
    rows = (len(stills) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 22, dtype=np.uint8)
    for index, (label, still) in enumerate(stills):
        resized = cv2.resize(still, (cell_width, 270), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * cell_height
        sheet[y : y + 270, x : x + cell_width] = resized
        cv2.putText(
            sheet,
            label,
            (x + 12, y + 291),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write contact sheet: {path}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
