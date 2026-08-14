from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from elderly_monitoring.modules.fall_risk.pose import (
    NORMALIZED_IMAGE_COORDINATES,
    PIXEL_IMAGE_COORDINATES,
    PoseObservation,
    build_keypoints,
    build_pose_observation,
    normalize_bbox,
)
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
    normalize_coordinates: bool = True,
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
    normalization_size = (
        (width, height) if normalize_coordinates and width > 0 and height > 0 else None
    )
    tracks: list[TrackObservation] = []
    poses: list[PoseObservation] = []
    for index, bbox in enumerate(bboxes):
        if index >= len(track_ids) or track_ids[index] is None:
            continue
        track_id = int(track_ids[index])
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        bbox_pixels = normalize_bbox(bbox[:4])
        if bbox_pixels is None:
            continue
        pose_bbox = normalize_bbox(
            bbox[:4],
            normalize_by=normalization_size,
        )
        tracks.append(
            TrackObservation(
                frame_id=frame_id,
                person_id=f"{person_id_prefix}_{track_id:03d}",
                track_id=track_id,
                bbox=bbox_pixels,
                scene_region=scene_region,
                track_confidence=round(confidence, 4),
                center=bbox_center(bbox_pixels),
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
                keypoints=build_keypoints(
                    points,
                    scores,
                    normalize_by=normalization_size,
                ),
                timestamp_sec=timestamp_sec,
                scene_region=scene_region,
                track_id=track_id,
                bbox=pose_bbox,
                bbox_pixels=bbox_pixels,
                coordinate_system=(
                    NORMALIZED_IMAGE_COORDINATES
                    if normalization_size is not None
                    else PIXEL_IMAGE_COORDINATES
                ),
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
    target_state: str = "unbound"
    target_reason: str = "no_candidates"
    target_changed: bool = False
    binding_verified: bool = False
    stage_timings_ms: dict[str, float] | None = None


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
        inference_size: int = 640,
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
        if inference_size < 32:
            raise ValueError("pose inference_size must be at least 32")
        self.inference_size = int(inference_size)
        self.primary_track_id: int | None = None
        self.primary_missing_since: float | None = None
        self.target_state = "unbound"
        self.target_reason = "not_started"
        self.target_change_count = 0

    def process_frame(
        self, frame: Any, *, frame_id: int, timestamp_sec: float, frame_size: tuple[float, float] | None = None
    ) -> StreamingPoseResult:
        if frame_size is None:
            height, width = frame.shape[:2]
            frame_size = (float(width), float(height))
        backend_started = time.perf_counter()
        raw = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            imgsz=self.inference_size,
            verbose=False,
        )
        backend_ms = _elapsed_ms(backend_started)
        result = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        adaptation_started = time.perf_counter()
        tracks, poses = adapt_yolo_pose_result(
            result,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            frame_size=frame_size,
            scene_region=self.scene_region,
        )
        adaptation_ms = _elapsed_ms(adaptation_started)
        selection_started = time.perf_counter()
        by_id = {pose.track_id: pose for pose in poses if pose.track_id is not None}
        reset = False
        target_changed = False
        selected: PoseObservation | None = None
        if self.primary_track_id in by_id:
            selected = by_id[self.primary_track_id]
            self.primary_missing_since = None
            self.target_state = "bound"
            self.target_reason = "bound_track_observed"
        elif self.primary_track_id is not None:
            if self.primary_missing_since is None:
                self.primary_missing_since = timestamp_sec
            if timestamp_sec - self.primary_missing_since >= self.lost_timeout_sec:
                previous_track_id = self.primary_track_id
                self.primary_track_id = None
                self.primary_missing_since = None
                reset = True
                if len(poses) == 1:
                    selected = poses[0]
                    self.primary_track_id = selected.track_id
                    target_changed = selected.track_id != previous_track_id
                    self.target_state = "bound"
                    self.target_reason = "target_replaced_after_ttl"
                elif len(poses) > 1:
                    self.target_state = "ambiguous"
                    self.target_reason = "multiple_replacement_candidates"
                else:
                    self.target_state = "unbound"
                    self.target_reason = "target_lost_timeout"
            else:
                self.target_state = "lost"
                self.target_reason = "bound_track_temporarily_missing"
        if self.primary_track_id is None and not reset:
            if len(poses) == 1:
                selected = poses[0]
                self.primary_track_id = selected.track_id
                self.target_state = "bound"
                self.target_reason = "single_candidate_stream_prior"
            elif len(poses) > 1:
                self.target_state = "ambiguous"
                self.target_reason = "multiple_unbound_candidates"
            else:
                self.target_state = "unbound"
                self.target_reason = "no_candidates"
        if target_changed:
            self.target_change_count += 1
        if selected is not None:
            selected = PoseObservation(**{**selected.__dict__, "person_id": self.person_id})
        selection_ms = _elapsed_ms(selection_started)
        return StreamingPoseResult(
            tracks=tracks,
            poses=poses,
            primary_pose=selected,
            window_reset=reset,
            target_state=self.target_state,
            target_reason=self.target_reason,
            target_changed=target_changed,
            # person_id is a stream-side prior; this is deliberately never a visual identity claim.
            binding_verified=False,
            stage_timings_ms={
                "detection_tracking_pose_backend": backend_ms,
                "result_adaptation": adaptation_ms,
                "target_selection": selection_ms,
            },
        )

    def close(self) -> None:
        self.model = None

    def reset(self) -> None:
        self.primary_track_id = None
        self.primary_missing_since = None
        self.target_state = "unbound"
        self.target_reason = "tracker_reset"
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    @property
    def target_diagnostics(self) -> dict[str, Any]:
        return {
            "state": self.target_state,
            "reason": self.target_reason,
            "primary_track_id": self.primary_track_id,
            "binding_verified": False,
            "target_change_count": self.target_change_count,
        }
def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)
