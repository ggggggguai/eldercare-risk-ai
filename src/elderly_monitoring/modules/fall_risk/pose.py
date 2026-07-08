"""跌倒风险视频链路中的姿态关键点提取。

当前实现保留 YOLOv8 pose 工程后端，并新增 RTMPose/MMPose 可选后端。
下游模块应依赖 JSONL 结构而不是具体模型。
"""

from __future__ import annotations

import importlib
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

DEFAULT_RTMPOSE_MODEL = "human"


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


def build_rtmpose_observations(
    predictions: Any,
    *,
    frame_id: int,
    timestamp_sec: float,
    frame_size: tuple[float, float],
    scene_region: str = "unknown",
    person_id_prefix: str = "elder",
    normalize_coordinates: bool = True,
) -> list[PoseObservation]:
    """将 MMPose/RTMPose 单帧预测适配成项目统一姿态记录。"""
    width, height = frame_size
    observations: list[PoseObservation] = []
    for person_index, prediction in enumerate(_iter_rtmpose_instances(predictions)):
        points_xy, scores = _extract_rtmpose_keypoints(prediction)
        if not points_xy:
            continue

        bbox = _extract_rtmpose_bbox(prediction)
        confidence = _extract_first_number(
            prediction,
            ("bbox_score", "bbox_scores", "bbox_confidence", "score"),
        )
        raw_track_id = _extract_first_number(prediction, ("track_id", "id"))
        track_id = int(raw_track_id) if raw_track_id is not None else person_index + 1
        person_id = str(prediction.get("person_id") or f"{person_id_prefix}_{track_id:03d}")
        pose_keypoints = build_keypoints(
            points_xy,
            scores,
            normalize_by=(width, height) if normalize_coordinates else None,
        )
        observations.append(
            build_pose_observation(
                frame_id=frame_id,
                person_id=person_id,
                keypoints=pose_keypoints,
                timestamp_sec=timestamp_sec,
                scene_region=scene_region,
                track_id=track_id,
                bbox=bbox,
                pose_confidence=confidence,
            )
        )
    return observations


def run_rtmpose_pose(
    *,
    video_path: Path,
    output_path: Path,
    pose_config: str | None = None,
    pose_checkpoint: str | None = None,
    device: str | None = None,
    scene_region: str = "unknown",
    person_id_prefix: str = "elder",
    confidence_threshold: float = 0.25,
    max_frames: int | None = None,
    normalize_coordinates: bool = True,
) -> int:
    MMPoseInferencer = _load_mmpose_inferencer()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "RTMPose 视频推理需要 OpenCV；请确认 eldercare-ai 环境已安装视觉依赖。"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    capture.release()

    try:
        inferencer = MMPoseInferencer(
            pose2d=pose_config or DEFAULT_RTMPOSE_MODEL,
            pose2d_weights=pose_checkpoint,
            device=device,
        )
    except ImportError as exc:
        raise _rtmpose_dependency_error(exc) from exc

    def iter_records() -> Iterable[dict[str, Any]]:
        try:
            results = inferencer(
                str(video_path),
                bbox_thr=confidence_threshold,
                show=False,
                return_vis=False,
            )
            for frame_count, result in enumerate(results):
                if max_frames is not None and frame_count >= max_frames:
                    break
                frame_id = int(_mapping_number(result, ("frame_id",), frame_count))
                predictions = _extract_rtmpose_predictions(result)
                observations = build_rtmpose_observations(
                    predictions,
                    frame_id=frame_id,
                    timestamp_sec=(frame_id / fps) if fps > 0 else 0.0,
                    frame_size=(width, height),
                    scene_region=scene_region,
                    person_id_prefix=person_id_prefix,
                    normalize_coordinates=normalize_coordinates,
                )
                for observation in observations:
                    yield observation.to_dict()
        except ImportError as exc:
            raise _rtmpose_dependency_error(exc) from exc

    return write_jsonl(iter_records(), output_path)


def _load_mmpose_inferencer() -> Any:
    try:
        module = importlib.import_module("mmpose.apis")
        # MMPose 运行时通常还会依赖 mmcv/mmengine。这里主动探测一次，
        # 让缺依赖错误在选择 RTMPose 后端时清晰暴露。
        importlib.import_module("mmcv")
        importlib.import_module("mmengine")
    except ImportError as exc:
        raise _rtmpose_dependency_error(exc) from exc

    inferencer = getattr(module, "MMPoseInferencer", None)
    if inferencer is None:
        raise RuntimeError("当前 MMPose 版本缺少 MMPoseInferencer，请升级 MMPose 后重试。")
    return inferencer


def _rtmpose_dependency_error(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "RTMPose 后端需要安装 MMPose、MMCV 和 MMEngine 相关依赖。"
        "当前环境缺少这些可选依赖，普通 YOLOv8-pose 单元测试不会受影响；"
        "请按 MMPose 官方文档在 eldercare-ai 环境中安装后再使用 --backend rtmpose。"
    )


def _extract_rtmpose_predictions(result: Any) -> Any:
    if isinstance(result, Mapping):
        predictions = result.get("predictions", result)
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            return predictions[0]
        return predictions
    return _object_to_builtin(getattr(result, "predictions", result))


def _iter_rtmpose_instances(predictions: Any) -> Iterable[Mapping[str, Any]]:
    normalized = _object_to_builtin(predictions)
    if isinstance(normalized, Mapping):
        pred_instances = normalized.get("pred_instances")
        if pred_instances is not None:
            yield from _iter_rtmpose_instances(pred_instances)
            return
        keypoints = normalized.get("keypoints")
        if _is_multi_person_keypoints(keypoints):
            scores = normalized.get("keypoint_scores", normalized.get("keypoint_score", []))
            bboxes = normalized.get("bboxes", normalized.get("bbox", normalized.get("bboxs", [])))
            bbox_scores = normalized.get("bbox_scores", normalized.get("bbox_score", []))
            track_ids = normalized.get("track_ids", normalized.get("track_id", []))
            person_ids = normalized.get("person_ids", normalized.get("person_id", []))
            for index, points_xy in enumerate(keypoints):
                yield {
                    "keypoints": points_xy,
                    "keypoint_scores": _item_at(scores, index, default=[]),
                    "bbox": _item_at(bboxes, index),
                    "bbox_score": _item_at(bbox_scores, index),
                    "track_id": _item_at(track_ids, index),
                    "person_id": _item_at(person_ids, index),
                }
            return
        yield normalized
        return

    if isinstance(normalized, list):
        for item in normalized:
            if isinstance(item, list):
                for nested_item in item:
                    if isinstance(nested_item, Mapping):
                        yield nested_item
                continue
            if isinstance(item, Mapping):
                yield item


def _extract_rtmpose_keypoints(prediction: Mapping[str, Any]) -> tuple[list[list[float]], list[float]]:
    points = _object_to_builtin(prediction.get("keypoints", []))
    scores = _object_to_builtin(prediction.get("keypoint_scores", prediction.get("keypoint_score", [])))
    if _is_multi_person_keypoints(points):
        points = points[0]
    if isinstance(scores, list) and scores and isinstance(scores[0], list):
        scores = scores[0]
    if not isinstance(points, list):
        return [], []
    if not isinstance(scores, list) or not scores:
        scores = [0.0] * len(points)
    return points, scores


def _extract_rtmpose_bbox(prediction: Mapping[str, Any]) -> list[float] | None:
    bbox = _object_to_builtin(prediction.get("bbox", prediction.get("bboxes")))
    if bbox is None:
        return None
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        bbox = bbox[0]
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    return [float(value) for value in bbox[:4]]


def _extract_first_number(prediction: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _object_to_builtin(prediction.get(key))
        if isinstance(value, list):
            while isinstance(value, list) and value:
                value = value[0]
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _mapping_number(mapping: Any, keys: Iterable[str], default: float) -> float:
    if not isinstance(mapping, Mapping):
        return default
    value = _extract_first_number(mapping, keys)
    return default if value is None else value


def _object_to_builtin(value: Any) -> Any:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _object_to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_object_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_object_to_builtin(item) for item in value]
    return value


def _is_multi_person_keypoints(value: Any) -> bool:
    value = _object_to_builtin(value)
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and bool(value[0])
        and isinstance(value[0][0], list)
    )


def _item_at(value: Any, index: int, default: Any = None) -> Any:
    value = _object_to_builtin(value)
    if isinstance(value, list) and index < len(value):
        return value[index]
    if index == 0 and value not in (None, []):
        return value
    return default


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
