"""跌倒风险视频链路中的人体检测与跟踪。

输出是逐帧人体框和 track_id 的 JSONL。后续模块可用这些记录计算速度、
轨迹连续性和场景区域证据。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class TrackObservation:
    frame_id: int
    person_id: str
    track_id: int
    bbox: list[float]
    scene_region: str
    track_confidence: float
    center: list[float]
    speed_px_per_sec: float | None
    timestamp_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bbox_center(bbox: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)]


def calculate_speed(
    current_center: list[float],
    previous_center: list[float] | None,
    fps: float,
) -> float | None:
    if previous_center is None or fps <= 0:
        return None
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    return round(((dx * dx + dy * dy) ** 0.5) * fps, 3)


def build_observation(
    *,
    frame_id: int,
    track_id: int,
    bbox: Iterable[float],
    confidence: float,
    scene_region: str,
    person_id_prefix: str,
    fps: float,
    previous_center: list[float] | None,
) -> TrackObservation:
    rounded_bbox = [round(float(value), 3) for value in bbox]
    center = bbox_center(rounded_bbox)
    # 像素速度只是相机坐标系下的 proxy，适合工程烟测和粗粒度运动特征，
    # 不能直接解释为真实世界步速。
    return TrackObservation(
        frame_id=frame_id,
        person_id=f"{person_id_prefix}_{track_id:03d}",
        track_id=track_id,
        bbox=rounded_bbox,
        scene_region=scene_region,
        track_confidence=round(float(confidence), 4),
        center=center,
        speed_px_per_sec=calculate_speed(center, previous_center, fps),
        timestamp_sec=round(frame_id / fps, 4) if fps > 0 else 0.0,
    )


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def run_yolov8_bytetrack(
    *,
    video_path: Path,
    output_path: Path,
    model_name: str = "yolov8n.pt",
    scene_region: str = "unknown",
    person_id_prefix: str = "elder",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    tracker_config: str = "bytetrack.yaml",
    max_frames: int | None = None,
) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Tracking requires the vision dependencies. Install the project with the vision extra."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    model = YOLO(model_name)
    # 按 track_id 记录上一帧中心点，避免把不同人的位置串起来计算速度。
    previous_centers: dict[int, list[float]] = {}

    def iter_records() -> Iterable[dict[str, Any]]:
        frame_count = 0
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=confidence_threshold,
            iou=iou_threshold,
            tracker=tracker_config,
            verbose=False,
        )
        for result in results:
            frame_id = int(getattr(result, "frame_id", frame_count) or frame_count)
            if max_frames is not None and frame_count >= max_frames:
                break
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy_values = boxes.xyxy.cpu().tolist()
                track_ids = boxes.id.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for bbox, raw_track_id, confidence in zip(xyxy_values, track_ids, confidences, strict=False):
                    track_id = int(raw_track_id)
                    observation = build_observation(
                        frame_id=frame_id,
                        track_id=track_id,
                        bbox=bbox,
                        confidence=float(confidence),
                        scene_region=scene_region,
                        person_id_prefix=person_id_prefix,
                        fps=fps,
                        previous_center=previous_centers.get(track_id),
                    )
                    previous_centers[track_id] = observation.center
                    yield observation.to_dict()
            frame_count += 1

    return write_jsonl(iter_records(), output_path)
