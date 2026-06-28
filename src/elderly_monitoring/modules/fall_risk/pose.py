"""跌倒风险视频链路中的姿态关键点提取。

当前实现用 YOLOv8 pose 作为可替换的工程后端。下游模块应依赖 JSONL
结构而不是具体模型，后续接入 RTMPose 时可以尽量不改消费者代码。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PoseObservation:
    frame_id: int
    person_id: str
    keypoints: list[PoseKeypoint]
    timestamp_sec: float
    track_id: int | None = None
    bbox: list[float] | None = None
    pose_confidence: float | None = None
    keypoint_quality: float | None = None
    scene_region: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keypoints"] = [keypoint.to_dict() for keypoint in self.keypoints]
        return payload


def build_keypoints(
    points_xy: Iterable[Iterable[float]],
    scores: Iterable[float],
    *,
    names: Iterable[str] = COCO_KEYPOINT_NAMES,
    normalize_by: tuple[float, float] | None = None,
) -> list[PoseKeypoint]:
    width, height = normalize_by or (0.0, 0.0)
    keypoints: list[PoseKeypoint] = []
    for name, point, score in zip(names, points_xy, scores, strict=False):
        x_raw, y_raw = [float(value) for value in point]
        # 默认归一化坐标，让规则 baseline 尽量少受视频分辨率影响；
        # 如需保留像素坐标，可以在入口参数中关闭归一化。
        x = x_raw / width if width > 0 else x_raw
        y = y_raw / height if height > 0 else y_raw
        keypoints.append(
            PoseKeypoint(
                name=str(name),
                x=round(x, 4),
                y=round(y, 4),
                score=round(float(score), 4),
            )
        )
    return keypoints


def keypoint_quality(keypoints: Iterable[PoseKeypoint], *, min_score: float = 0.30) -> float:
    """基于可见关键点覆盖率估计单帧姿态质量。"""
    keypoint_list = list(keypoints)
    if not keypoint_list:
        return 0.0
    visible_count = sum(1 for keypoint in keypoint_list if keypoint.score >= min_score)
    mean_score = sum(keypoint.score for keypoint in keypoint_list) / len(keypoint_list)
    coverage = visible_count / len(keypoint_list)
    return round((coverage * 0.6) + (mean_score * 0.4), 4)


def build_pose_observation(
    *,
    frame_id: int,
    person_id: str,
    keypoints: Iterable[PoseKeypoint],
    timestamp_sec: float,
    scene_region: str = "unknown",
    track_id: int | None = None,
    bbox: Iterable[float] | None = None,
    pose_confidence: float | None = None,
) -> PoseObservation:
    keypoint_list = list(keypoints)
    rounded_bbox = [round(float(value), 3) for value in bbox] if bbox is not None else None
    return PoseObservation(
        frame_id=frame_id,
        person_id=person_id,
        keypoints=keypoint_list,
        timestamp_sec=round(float(timestamp_sec), 4),
        track_id=track_id,
        bbox=rounded_bbox,
        pose_confidence=round(float(pose_confidence), 4) if pose_confidence is not None else None,
        keypoint_quality=keypoint_quality(keypoint_list),
        scene_region=scene_region,
    )


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def run_yolov8_pose(
    *,
    video_path: Path,
    output_path: Path,
    model_name: str = "yolov8n-pose.pt",
    scene_region: str = "unknown",
    person_id_prefix: str = "elder",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    tracker_config: str = "bytetrack.yaml",
    max_frames: int | None = None,
    normalize_coordinates: bool = True,
) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "姿态关键点提取需要视觉依赖，请使用 eldercare-ai 环境或安装 vision 依赖。"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    capture.release()

    model = YOLO(model_name)

    def iter_records() -> Iterable[dict[str, Any]]:
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            conf=confidence_threshold,
            iou=iou_threshold,
            tracker=tracker_config,
            verbose=False,
        )
        for frame_count, result in enumerate(results):
            if max_frames is not None and frame_count >= max_frames:
                break
            frame_id = int(getattr(result, "frame_id", frame_count) or frame_count)
            keypoints = result.keypoints
            boxes = result.boxes
            if keypoints is None or keypoints.xy is None:
                continue

            xy_values = keypoints.xy.cpu().tolist()
            score_values = keypoints.conf.cpu().tolist() if keypoints.conf is not None else []
            bbox_values = boxes.xyxy.cpu().tolist() if boxes is not None and boxes.xyxy is not None else []
            box_confidences = boxes.conf.cpu().tolist() if boxes is not None and boxes.conf is not None else []
            track_ids = boxes.id.cpu().tolist() if boxes is not None and boxes.id is not None else []

            for person_index, points_xy in enumerate(xy_values):
                scores = score_values[person_index] if person_index < len(score_values) else [0.0] * len(points_xy)
                bbox = bbox_values[person_index] if person_index < len(bbox_values) else None
                confidence = box_confidences[person_index] if person_index < len(box_confidences) else None
                raw_track_id = track_ids[person_index] if person_index < len(track_ids) else None
                track_id = int(raw_track_id) if raw_track_id is not None else person_index + 1
                person_id = f"{person_id_prefix}_{track_id:03d}"
                pose_keypoints = build_keypoints(
                    points_xy,
                    scores,
                    normalize_by=(width, height) if normalize_coordinates else None,
                )
                observation = build_pose_observation(
                    frame_id=frame_id,
                    person_id=person_id,
                    keypoints=pose_keypoints,
                    timestamp_sec=(frame_id / fps) if fps > 0 else 0.0,
                    scene_region=scene_region,
                    track_id=track_id,
                    bbox=bbox,
                    pose_confidence=confidence,
                )
                yield observation.to_dict()

    return write_jsonl(iter_records(), output_path)
