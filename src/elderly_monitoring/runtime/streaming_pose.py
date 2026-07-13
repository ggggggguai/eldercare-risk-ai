from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elderly_monitoring.modules.fall_risk.pose import PoseObservation, build_keypoints, build_pose_observation
from elderly_monitoring.modules.fall_risk.tracking import TrackObservation, bbox_center


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def adapt_yolo_pose_result(
    result: Any,
    *,
    frame_id: int,
    timestamp_sec: float,
    frame_size: tuple[float, float],
    scene_region: str,
    person_id_prefix: str = "elder",
) -> tuple[list[TrackObservation], list[PoseObservation]]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return [], []
    bboxes = _tolist(getattr(boxes, "xyxy", None))
    confidences = _tolist(getattr(boxes, "conf", None))
    track_ids = _tolist(getattr(boxes, "id", None))
    points_by_person = _tolist(getattr(keypoints, "xy", None))
    scores_by_person = _tolist(getattr(keypoints, "conf", None))
    width, height = frame_size
    tracks: list[TrackObservation] = []
    poses: list[PoseObservation] = []
    for index, bbox in enumerate(bboxes):
        if index >= len(track_ids) or track_ids[index] is None:
            continue
        track_id = int(track_ids[index])
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        rounded_bbox = [round(float(value), 3) for value in bbox[:4]]
        tracks.append(
            TrackObservation(
                frame_id=frame_id,
                person_id=f"{person_id_prefix}_{track_id:03d}",
                track_id=track_id,
                bbox=rounded_bbox,
                scene_region=scene_region,
                track_confidence=round(confidence, 4),
                center=bbox_center(rounded_bbox),
                speed_px_per_sec=None,
                timestamp_sec=round(timestamp_sec, 4),
            )
        )
        if index >= len(points_by_person) or not points_by_person[index]:
            continue
        points = points_by_person[index]
        scores = scores_by_person[index] if index < len(scores_by_person) else [0.0] * len(points)
        poses.append(
            build_pose_observation(
                frame_id=frame_id,
                person_id=f"{person_id_prefix}_{track_id:03d}",
                keypoints=build_keypoints(points, scores, normalize_by=(width, height)),
                timestamp_sec=timestamp_sec,
                scene_region=scene_region,
                track_id=track_id,
                bbox=rounded_bbox,
                pose_confidence=confidence,
            )
        )
    return tracks, poses


@dataclass(frozen=True)
class StreamingPoseResult:
    tracks: list[TrackObservation]
    poses: list[PoseObservation]
    primary_pose: PoseObservation | None
    window_reset: bool = False


class StreamingPoseTracker:
    def __init__(
        self,
        *,
        model_name: str = "yolov8n-pose.pt",
        model: Any | None = None,
        person_id: str,
        scene_region: str,
        lost_timeout_sec: float = 2.0,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_name)
        self.model = model
        self.person_id = person_id
        self.scene_region = scene_region
        self.lost_timeout_sec = lost_timeout_sec
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        self.primary_track_id: int | None = None
        self.primary_missing_since: float | None = None

    def process_frame(
        self, frame: Any, *, frame_id: int, timestamp_sec: float, frame_size: tuple[float, float] | None = None
    ) -> StreamingPoseResult:
        if frame_size is None:
            height, width = frame.shape[:2]
            frame_size = (float(width), float(height))
        raw = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            verbose=False,
        )
        result = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        tracks, poses = adapt_yolo_pose_result(
            result,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            frame_size=frame_size,
            scene_region=self.scene_region,
        )
        by_id = {pose.track_id: pose for pose in poses if pose.track_id is not None}
        reset = False
        selected: PoseObservation | None = None
        if self.primary_track_id in by_id:
            selected = by_id[self.primary_track_id]
            self.primary_missing_since = None
        elif self.primary_track_id is not None:
            if self.primary_missing_since is None:
                self.primary_missing_since = timestamp_sec
            if timestamp_sec - self.primary_missing_since >= self.lost_timeout_sec:
                self.primary_track_id = None
                self.primary_missing_since = None
                reset = True
        if self.primary_track_id is None and poses:
            selected = max(poses, key=lambda pose: _bbox_area(pose.bbox))
            self.primary_track_id = selected.track_id
        if selected is not None:
            selected = PoseObservation(**{**selected.__dict__, "person_id": self.person_id})
        return StreamingPoseResult(tracks=tracks, poses=poses, primary_pose=selected, window_reset=reset)

    def close(self) -> None:
        self.model = None


def _bbox_area(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
