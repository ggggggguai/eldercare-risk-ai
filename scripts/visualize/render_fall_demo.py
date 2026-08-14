from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _event_for_time(
    events: list[dict[str, Any]],
    timestamp: float,
    offline_full_clip: bool = False,
) -> dict[str, Any] | None:
    if offline_full_clip:
        scored = [event for event in events if event.get("fall_event_score") is not None]
        return max(scored, key=lambda item: float(item.get("fall_event_score") or 0.0), default=None)
    completed = [
        event
        for event in events
        if event.get("status") == "provisional_shadow"
        and float(event.get("window_end_time_exclusive", 0.0)) <= timestamp
    ]
    if not completed:
        return None
    return max(completed, key=lambda item: float(item.get("window_end_time_exclusive", 0.0)))


def _draw_text(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(30)
    body_font = _font(24)
    x, y = 28, 24
    line_height = 38
    panel_height = 26 + line_height * len(lines)
    draw.rounded_rectangle((x - 12, y - 12, 610, y + panel_height), radius=12, fill=(10, 17, 28, 215))
    for index, (text, color) in enumerate(lines):
        draw.text((x, y + index * line_height), text, font=title_font if index == 0 else body_font, fill=(*color, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_score_chart(
    frame: np.ndarray,
    events: list[dict[str, Any]],
    timestamp: float,
    duration: float,
    threshold: float,
    official_onset_sec: float,
) -> None:
    height, width = frame.shape[:2]
    if width < 1000:
        return
    left, right = width - 620, width - 28
    top, bottom = 28, min(260, height - 80)
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (10, 17, 28), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    plot_left, plot_right = left + 50, right - 20
    plot_top, plot_bottom = top + 44, bottom - 36

    def point(time_value: float, score: float) -> tuple[int, int]:
        x = plot_left + int((plot_right - plot_left) * min(1.0, max(0.0, time_value / duration)))
        y = plot_bottom - int((plot_bottom - plot_top) * min(1.0, max(0.0, score)))
        return x, y

    cv2.putText(frame, "CAUSAL FALL SCORE", (left + 20, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(frame, (plot_left, plot_bottom), (plot_right, plot_bottom), (130, 140, 150), 1)
    cv2.line(frame, (plot_left, plot_top), (plot_left, plot_bottom), (130, 140, 150), 1)
    threshold_y = point(0.0, threshold)[1]
    for x in range(plot_left, plot_right, 18):
        cv2.line(frame, (x, threshold_y), (min(x + 10, plot_right), threshold_y), (50, 80, 235), 2)
    onset_x = point(official_onset_sec, 0.0)[0]
    cv2.line(frame, (onset_x, plot_top), (onset_x, plot_bottom), (40, 170, 255), 2)
    cv2.putText(frame, "official onset", (max(plot_left, onset_x - 70), plot_bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 170, 255), 1, cv2.LINE_AA)
    completed = [
        event
        for event in events
        if event.get("fall_event_score") is not None
        and float(event.get("window_end_time_exclusive", 0.0)) <= timestamp
    ]
    score_points = [
        point(float(event["window_end_time_exclusive"]), float(event["fall_event_score"]))
        for event in completed
    ]
    if len(score_points) > 1:
        cv2.polylines(frame, [np.asarray(score_points, dtype=np.int32)], False, (20, 210, 255), 3, cv2.LINE_AA)
    for chart_point in score_points:
        cv2.circle(frame, chart_point, 4, (20, 210, 255), -1)
    current_x = point(timestamp, 0.0)[0]
    cv2.line(frame, (current_x, plot_top), (current_x, plot_bottom), (20, 210, 255), 1)
    cv2.putText(frame, f"threshold {threshold:.2f}", (plot_left + 6, threshold_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 115, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "0", (plot_left - 28, plot_bottom + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 195, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "1", (plot_left - 28, plot_top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 195, 200), 1, cv2.LINE_AA)


def render(
    input_video: Path,
    pose_jsonl: Path,
    event_jsonl: Path,
    output_video: Path,
    summary_path: Path,
    output_width: int | None = None,
    official_onset_sec: float | None = None,
    offline_full_clip: bool = False,
    evaluation_label: str | None = None,
    evaluation_note: str | None = None,
) -> dict[str, Any]:
    poses = _read_jsonl(pose_jsonl)
    events = _read_jsonl(event_jsonl)
    pose_by_frame = {int(row["frame_id"]): row for row in poses}
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = output_width or source_width
    if width <= 0:
        raise ValueError("output width must be positive")
    scale = width / source_width
    height = int(round(source_height * scale))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {output_video}")

    trajectory: list[tuple[int, int]] = []
    max_score = max((float(event.get("fall_event_score") or 0.0) for event in events), default=0.0)
    max_quality = max((float(event.get("quality", {}).get("mean_joint_quality") or 0.0) for event in events), default=0.0)
    max_coverage = max((float(event.get("quality", {}).get("core_joint_coverage") or 0.0) for event in events), default=0.0)
    confidence = round(0.45 * max_quality + 0.35 * max_coverage + 0.20 * min(1.0, max_score + 0.2), 2)
    positive_end_times = [
        float(event["window_end_time_exclusive"])
        for event in events
        if event.get("fall_event_detected") and event.get("window_end_time_exclusive") is not None
    ]
    first_positive_end = min(positive_end_times, default=None)
    trigger_delay_sec = (
        round(first_positive_end - official_onset_sec, 4)
        if first_positive_end is not None and official_onset_sec is not None
        else None
    )
    duration = max(1.0 / fps, float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps)
    processed = 0
    event_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        timestamp = processed / fps
        row = pose_by_frame.get(processed)
        event = _event_for_time(events, timestamp, offline_full_clip)
        detected = bool(event and event.get("fall_event_detected"))
        bbox = row.get("bbox_pixels") or row.get("bbox") if row else None
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = (int(round(float(value) * scale)) for value in bbox[:4])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            trajectory.append(center)
            trajectory = trajectory[-45:]
            color = (
                (40, 45, 230)
                if detected
                else (70, 210, 90)
                if offline_full_clip and event is not None
                else (255, 190, 30)
            )
            if detected:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 220), -1)
                frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
                event_frames += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, "TRACK ID 1", (x1, max(38, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            if len(trajectory) > 1:
                cv2.polylines(frame, [np.asarray(trajectory, dtype=np.int32)], False, (255, 210, 20), 4, cv2.LINE_AA)

        score = float(event.get("fall_event_score") or 0.0) if event else 0.0
        status = (
            "跌倒事件"
            if detected
            else "非跌倒"
            if offline_full_clip and event is not None
            else "观察中"
        )
        lines = [
            (
                "NEGATIVE CHECK / OFFLINE FULL-CLIP"
                if offline_full_clip
                else "FALL RISK / PROVISIONAL SHADOW",
                (255, 255, 255),
            ),
            (f"跌倒分数：{score:.2f}    置信度：{confidence:.2f}", (255, 238, 120)),
            (f"目标轨迹：ID 1    状态：{status}", (255, 255, 255)),
        ]
        if evaluation_label:
            lines.append((evaluation_label, (150, 255, 170)))
        if evaluation_note:
            lines.append((evaluation_note, (255, 190, 120)))
        if trigger_delay_sec is not None:
            timing_label = f"首次触发：{first_positive_end:.2f}s    延迟：{trigger_delay_sec:+.2f}s"
            lines.append((timing_label, (255, 190, 120)))
        frame = _draw_text(frame, lines)
        if official_onset_sec is not None:
            _draw_score_chart(
                frame,
                events,
                timestamp,
                duration,
                0.42490479350090027,
                official_onset_sec,
            )
        bar_y = height - 34
        cv2.rectangle(frame, (28, bar_y), (width - 28, bar_y + 8), (80, 80, 80), -1)
        progress = min(1.0, (processed + 1) / max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        cv2.rectangle(frame, (28, bar_y), (28 + int((width - 56) * progress), bar_y + 8), (20, 210, 255), -1)
        cv2.circle(frame, (28 + int((width - 56) * progress), bar_y + 4), 12, (20, 210, 255), -1)
        cv2.putText(frame, "video progress", (34, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 210, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        processed += 1

    cap.release()
    writer.release()
    summary = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "frames": processed,
        "fps": fps,
        "output_size": [width, height],
        "track_id": 1,
        "max_fall_event_score": round(max_score, 4),
        "display_confidence": confidence,
        "event_frames": event_frames,
        "model_status": "provisional_shadow",
        "event_threshold": 0.42490479350090027,
        "official_onset_sec": official_onset_sec,
        "first_positive_window_end_sec": first_positive_end,
        "trigger_delay_sec": trigger_delay_sec,
        "display_mode": "offline_full_clip" if offline_full_clip else "causal_window_end",
        "evaluation_label": evaluation_label,
        "evaluation_note": evaluation_note,
        "note": "Candidate TCN overlay for demonstration; not a production-path or formal evaluation result.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fall-risk demonstration overlay video.")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--pose-jsonl", type=Path, required=True)
    parser.add_argument("--event-jsonl", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--official-onset-sec", type=float, default=None)
    parser.add_argument("--offline-full-clip", action="store_true")
    parser.add_argument("--evaluation-label", default=None)
    parser.add_argument("--evaluation-note", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.input_video,
                args.pose_jsonl,
                args.event_jsonl,
                args.output_video,
                args.summary,
                args.output_width,
                args.official_onset_sec,
                args.offline_full_clip,
                args.evaluation_label,
                args.evaluation_note,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
