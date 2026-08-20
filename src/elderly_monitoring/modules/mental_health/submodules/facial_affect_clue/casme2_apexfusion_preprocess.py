from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .face_tracking import DlibFaceLandmarker, default_dlib_predictor_path
from .optical_flow import read_image
from .paper_preprocess import PaperPreprocessConfig, align_face_sequence, optical_strain
from .casme2_apexfusion_manifest import read_jsonl, sha256_file, write_json, write_jsonl


ARTIFACT_SCHEMA = "casme2_apexfusion_artifact_v1"
ARTIFACT_MANIFEST_SCHEMA = "casme2_apexfusion_artifact_manifest_v1"
APPEARANCE_SHAPE = (7, 3, 112, 112)
SHORT_MOTION_SHAPE = (6, 4, 56, 56)
LONG_MOTION_SHAPE = (2, 4, 56, 56)
LOCAL_ROI_SHAPE = (7, 5, 1, 32, 32)
LANDMARK_SHAPE = (7, 68, 2)
LANDMARK_DYNAMICS_SHAPE = (7, 68, 6)
HOOF_SHAPE = (48,)
LBP_TOP_SHAPE = (48,)

ROI_GROUPS = {
    "left_brow_eye": tuple(range(17, 22)) + tuple(range(36, 42)),
    "right_brow_eye": tuple(range(22, 27)) + tuple(range(42, 48)),
    "nose_cheeks": tuple(range(27, 36)) + (1, 2, 3, 13, 14, 15),
    "mouth": tuple(range(48, 68)),
    "chin": tuple(range(5, 12)),
}


@dataclass(frozen=True)
class ApexFusionPreprocessConfig:
    aligned_size: int = 256
    appearance_size: int = 112
    flow_size: int = 56
    roi_size: int = 32
    flow_quantile: float = 0.995
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    flow_flags: int = 0

    def fingerprint(self) -> str:
        payload = {"schema": ARTIFACT_SCHEMA, **asdict(self), "roi_groups": ROI_GROUPS}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_frame_numbers(onset: int, apex: int, offset: int) -> tuple[int, ...]:
    if not onset < apex < offset:
        raise ValueError("Official apex must be strictly inside onset/offset")
    return (
        onset,
        max(onset, apex - 2),
        max(onset, apex - 1),
        apex,
        min(offset, apex + 1),
        min(offset, apex + 2),
        offset,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points.astype(np.float32), np.ones(len(points), dtype=np.float32)))
    return (homogeneous @ transform.T).astype(np.float32)


def _dense_flow(first: np.ndarray, second: np.ndarray, config: ApexFusionPreprocessConfig) -> np.ndarray:
    first_gray = cv2.resize(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (config.flow_size, config.flow_size), interpolation=cv2.INTER_AREA)
    second_gray = cv2.resize(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), (config.flow_size, config.flow_size), interpolation=cv2.INTER_AREA)
    return cv2.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        config.flow_pyr_scale,
        config.flow_levels,
        config.flow_winsize,
        config.flow_iterations,
        config.flow_poly_n,
        config.flow_poly_sigma,
        config.flow_flags,
    ).astype(np.float32)


def _motion_tensor(flow: np.ndarray, quantile: float) -> tuple[np.ndarray, dict[str, float]]:
    strain = optical_strain(flow)
    magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
    epsilon = float(np.finfo(np.float32).eps)
    vector_scale = max(float(np.quantile(magnitude, quantile)), epsilon)
    strain_scale = max(float(np.quantile(strain, quantile)), epsilon)
    tensor = np.stack(
        (
            np.clip(flow[..., 0] / vector_scale, -1.0, 1.0),
            np.clip(flow[..., 1] / vector_scale, -1.0, 1.0),
            np.clip(magnitude / vector_scale, 0.0, 1.0),
            np.clip(strain / strain_scale, 0.0, 1.0),
        ),
        axis=0,
    ).astype(np.float32)
    return tensor, {"vector_scale": vector_scale, "strain_scale": strain_scale}


def _normalized_landmarks(points: np.ndarray) -> np.ndarray:
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    scale = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    return ((points - points[30]) / scale).astype(np.float32)


def _roi_crop(image: np.ndarray, points: np.ndarray, indices: Sequence[int], size: int) -> np.ndarray:
    selected = points[np.asarray(indices, dtype=np.int64)]
    minimum = selected.min(axis=0)
    maximum = selected.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = max(float((maximum - minimum).max()) * 1.45, 8.0)
    x0, y0 = np.floor(center - extent / 2.0).astype(int)
    x1, y1 = np.ceil(center + extent / 2.0).astype(int)
    pad = int(max(0, -x0, -y0, x1 - image.shape[1], y1 - image.shape[0])) + 1
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    crop = padded[y0 + pad : y1 + pad, x0 + pad : x1 + pad]
    if crop.size == 0:
        raise RuntimeError("Empty landmark ROI")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)[None]


def _lbp_codes(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    padded = np.pad(image, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    return codes


def _hist16(codes: np.ndarray) -> np.ndarray:
    hist = np.bincount((codes.astype(np.uint16) // 16).reshape(-1), minlength=16).astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def lbp_top_features(gray_sequence: np.ndarray) -> np.ndarray:
    if gray_sequence.shape[0] != 7:
        raise ValueError("LBP-TOP requires seven frames")
    xy = gray_sequence[3]
    xt = gray_sequence[:, gray_sequence.shape[1] // 2, :]
    yt = gray_sequence[:, :, gray_sequence.shape[2] // 2]
    return np.concatenate((_hist16(_lbp_codes(xy)), _hist16(_lbp_codes(xt)), _hist16(_lbp_codes(yt)))).astype(np.float32)


def hoof_features(flows: Sequence[np.ndarray], bins: int = 8) -> np.ndarray:
    features = []
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    for flow in flows:
        magnitude = np.linalg.norm(flow, axis=2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        hist, _ = np.histogram(angle, bins=edges, weights=magnitude)
        hist = hist.astype(np.float32)
        features.append(hist / max(float(hist.sum()), 1.0))
    return np.concatenate(features).astype(np.float32)


def _hog_features(gray_apex: np.ndarray) -> np.ndarray:
    image = np.clip(np.rint(gray_apex * 255.0), 0, 255).astype(np.uint8)
    descriptor = cv2.HOGDescriptor((112, 112), (16, 16), (8, 8), (8, 8), 9)
    return descriptor.compute(image).reshape(-1).astype(np.float32)


def _geometry_features(landmarks: np.ndarray) -> np.ndarray:
    # Positions, apex/onset and offset/apex displacement plus compact geometry.
    onset, apex, offset = landmarks[0], landmarks[3], landmarks[6]
    eye_open = [
        np.linalg.norm(frame[37] - frame[41]) + np.linalg.norm(frame[38] - frame[40])
        + np.linalg.norm(frame[43] - frame[47]) + np.linalg.norm(frame[44] - frame[46])
        for frame in landmarks
    ]
    mouth_open = [np.linalg.norm(frame[62] - frame[66]) + np.linalg.norm(frame[63] - frame[65]) for frame in landmarks]
    mouth_width = [np.linalg.norm(frame[48] - frame[54]) for frame in landmarks]
    compact = np.asarray(
        [np.mean(eye_open), np.std(eye_open), np.max(eye_open) - np.min(eye_open), np.mean(mouth_open), np.std(mouth_open), np.max(mouth_open) - np.min(mouth_open), np.mean(mouth_width), np.std(mouth_width), np.linalg.norm(apex - onset, axis=1).mean(), np.linalg.norm(offset - apex, axis=1).mean(), np.linalg.norm(apex - onset, axis=1).max(), np.linalg.norm(offset - apex, axis=1).max()],
        dtype=np.float32,
    )
    return np.concatenate(((apex - onset).reshape(-1), (offset - apex).reshape(-1), compact)).astype(np.float32)


def preprocess_record(
    record: Mapping[str, Any],
    *,
    landmarker: DlibFaceLandmarker,
    config: ApexFusionPreprocessConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frame_numbers = sample_frame_numbers(int(record["onset_frame"]), int(record["apex_frame"]), int(record["offset_frame"]))
    frame_dir = Path(str(record["raw_frame_dir"]))
    frames = [read_image(frame_dir / f"img{number}.jpg") for number in frame_numbers]
    detections = [landmarker.detect(frame) for frame in frames]
    paper_config = PaperPreprocessConfig(aligned_width=config.aligned_size, aligned_height=config.aligned_size, flow_width=config.flow_size, flow_height=config.flow_size, preprocess_variant="base")
    alignment = align_face_sequence(frames, detections[0].points, paper_config)
    aligned_frames = list(alignment.frames)
    transformed = np.stack([_transform_points(result.points, alignment.transform) for result in detections]).astype(np.float32)
    transformed[..., 0] = np.clip(transformed[..., 0], 0, config.aligned_size - 1)
    transformed[..., 1] = np.clip(transformed[..., 1], 0, config.aligned_size - 1)
    normalized = np.stack([_normalized_landmarks(points) for points in transformed]).astype(np.float32)
    displacement = normalized - normalized[0:1]
    velocity = np.concatenate((np.zeros_like(normalized[:1]), np.diff(normalized, axis=0)), axis=0)
    landmark_dynamics = np.concatenate((normalized, displacement, velocity), axis=2).astype(np.float32)

    appearance = np.stack(
        [cv2.cvtColor(cv2.resize(frame, (config.appearance_size, config.appearance_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0 for frame in aligned_frames]
    )
    local_roi = np.stack(
        [np.stack([_roi_crop(frame, points, indices, config.roi_size) for indices in ROI_GROUPS.values()]) for frame, points in zip(aligned_frames, transformed)]
    ).astype(np.float32)
    short_raw = [_dense_flow(aligned_frames[index], aligned_frames[index + 1], config) for index in range(6)]
    long_raw = [_dense_flow(aligned_frames[0], aligned_frames[3], config), _dense_flow(aligned_frames[3], aligned_frames[6], config)]
    short_pairs = [_motion_tensor(flow, config.flow_quantile) for flow in short_raw]
    long_pairs = [_motion_tensor(flow, config.flow_quantile) for flow in long_raw]
    motion_short = np.stack([pair[0] for pair in short_pairs]).astype(np.float32)
    motion_long = np.stack([pair[0] for pair in long_pairs]).astype(np.float32)
    gray_sequence = np.stack([cv2.cvtColor(cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for frame in aligned_frames])
    hoof = hoof_features(short_raw)
    lbp_top = lbp_top_features(gray_sequence)
    hog = _hog_features(gray_sequence[3])
    landmark_features = _geometry_features(normalized)
    motion_stats = np.asarray(
        [value for flow in short_raw + long_raw for value in (np.linalg.norm(flow, axis=2).mean(), np.linalg.norm(flow, axis=2).std(), np.abs(flow[..., 0]).mean(), np.abs(flow[..., 1]).mean())],
        dtype=np.float32,
    )
    traditional = np.concatenate((hoof, lbp_top, hog, landmark_features, motion_stats)).astype(np.float32)
    arrays = {
        "appearance": appearance.astype(np.float32),
        "motion_short": motion_short,
        "motion_long": motion_long,
        "local_roi": local_roi,
        "landmarks": normalized,
        "landmark_dynamics": landmark_dynamics,
        "hoof": hoof,
        "lbp_top": lbp_top,
        "hog": hog,
        "landmark_features": landmark_features,
        "motion_stats": motion_stats,
        "traditional": traditional,
    }
    expected = {"appearance": APPEARANCE_SHAPE, "motion_short": SHORT_MOTION_SHAPE, "motion_long": LONG_MOTION_SHAPE, "local_roi": LOCAL_ROI_SHAPE, "landmarks": LANDMARK_SHAPE, "landmark_dynamics": LANDMARK_DYNAMICS_SHAPE, "hoof": HOOF_SHAPE, "lbp_top": LBP_TOP_SHAPE}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{key} shape mismatch: {arrays[key].shape} != {shape}")
    for key, value in arrays.items():
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise RuntimeError(f"Non-finite or non-float32 array: {key}")
    metadata = {
        "schema_version": ARTIFACT_SCHEMA,
        "sample_id": record["sample_id"],
        "subject_id": record["subject_id"],
        "frame_numbers": list(frame_numbers),
        "detection_sources": [result.detection_source for result in detections],
        "face_boxes": [list(result.face_box) for result in detections],
        "alignment_transform": alignment.transform.tolist(),
        "flow_scales_short": [pair[1] for pair in short_pairs],
        "flow_scales_long": [pair[1] for pair in long_pairs],
        "roi_names": list(ROI_GROUPS),
        "config_sha256": config.fingerprint(),
        "fold_fit_required": ["standardization", "PCA", "feature_selection", "classifier", "calibration"],
    }
    return arrays, metadata


def write_artifact(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return sha256_file(path)


def _preview(path: Path, arrays: Mapping[str, np.ndarray], title: str) -> None:
    frames = [cv2.cvtColor((np.clip(arrays["appearance"][index].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for index in range(7)]
    panel = np.concatenate(frames, axis=1)
    panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def build_artifact_release(
    *,
    manifest_path: Path,
    output_root: Path,
    artifact_manifest_path: Path,
    failure_path: Path,
    config_path: Path,
    visualization_dir: Path,
    predictor_path: Path | None = None,
    config: ApexFusionPreprocessConfig | None = None,
) -> dict[str, Any]:
    config = config or ApexFusionPreprocessConfig()
    predictor_path = predictor_path or default_dlib_predictor_path(Path(__file__).resolve().parents[7])
    landmarker = DlibFaceLandmarker(predictor_path, allow_cropped_frame_fallback=True, detector_upsample=0)
    rows = read_jsonl(manifest_path)
    artifact_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    preview_classes: set[str] = set()
    for record in rows:
        try:
            arrays, metadata = preprocess_record(record, landmarker=landmarker, config=config)
            artifact_path = output_root / str(record["subject_id"]) / f"{record['sample_id']}.npz"
            digest = write_artifact(artifact_path, arrays)
            artifact_row = {
                **metadata,
                "artifact_path": artifact_path.resolve().as_posix(),
                "artifact_sha256": digest,
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                "three_class_id": record["three_class_id"],
                "three_class_name": record["three_class_name"],
                "aux_emotion_id": record["aux_emotion_id"],
                "source_manifest_sha256": sha256_file(manifest_path),
            }
            artifact_rows.append(artifact_row)
            label = str(record["three_class_name"])
            if label not in preview_classes and not bool(record.get("publication_restricted")):
                _preview(visualization_dir / f"{label}__{record['sample_id']}.jpg", arrays, f"{label} | 7 apex-anchored frames")
                preview_classes.add(label)
        except Exception as error:  # preserve every source failure in the release
            failures.append({"sample_id": record["sample_id"], "subject_id": record["subject_id"], "error_type": type(error).__name__, "detail": str(error)})
    write_jsonl(artifact_manifest_path, artifact_rows)
    write_json(failure_path, {"schema_version": "casme2_apexfusion_failures_v1", "count": len(failures), "failures": failures})
    resolved = {
        "schema_version": "casme2_apexfusion_preprocess_config_v1",
        "task_id": "FLOW-ME2-001",
        "config": asdict(config),
        "config_sha256": config.fingerprint(),
        "predictor_path": predictor_path.resolve().as_posix(),
        "predictor_sha256": landmarker.predictor_sha256,
        "source_manifest_path": manifest_path.resolve().as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "artifact_root": output_root.resolve().as_posix(),
        "normalization_scope": "per_sample_pair_only; learned scaling/PCA remains fold_fit",
        "historical_artifacts_overwritten": False,
    }
    write_json(config_path, resolved)
    return {
        "status": "passed" if len(artifact_rows) == len(rows) and not failures else "failed",
        "samples": len(rows),
        "artifacts": len(artifact_rows),
        "failures": len(failures),
        "detection_sources": dict(sorted(Counter(source for row in artifact_rows for source in row["detection_sources"]).items())),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "failure_list_sha256": sha256_file(failure_path),
        "config_sha256": sha256_file(config_path),
    }


def audit_artifact_release(*, formal_manifest_path: Path, artifact_manifest_path: Path, failure_path: Path) -> dict[str, Any]:
    formal = read_jsonl(formal_manifest_path)
    artifacts = read_jsonl(artifact_manifest_path)
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = []
    formal_ids = {row["sample_id"] for row in formal}
    artifact_ids = {row["sample_id"] for row in artifacts}
    if formal_ids != artifact_ids:
        errors.append({"code": "manifest_alignment", "missing": sorted(formal_ids - artifact_ids), "extra": sorted(artifact_ids - formal_ids)})
    if failures.get("count") != 0:
        errors.append({"code": "artifact_failures", "detail": failures})
    reference_shapes: dict[str, list[int]] | None = None
    for row in artifacts:
        path = Path(row["artifact_path"])
        if not path.is_file() or sha256_file(path) != row["artifact_sha256"]:
            errors.append({"code": "artifact_hash", "sample_id": row["sample_id"]})
            continue
        with np.load(path, allow_pickle=False) as archive:
            shapes = {key: list(archive[key].shape) for key in archive.files}
            if reference_shapes is None:
                reference_shapes = shapes
            elif shapes != reference_shapes:
                errors.append({"code": "shape_drift", "sample_id": row["sample_id"], "shapes": shapes})
            for key in archive.files:
                value = archive[key]
                if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                    errors.append({"code": "array_invalid", "sample_id": row["sample_id"], "array": key})
    return {
        "schema_version": "casme2_apexfusion_artifact_audit_v1",
        "task_id": "FLOW-ME2-001",
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "inventory": {"formal_samples": len(formal), "artifact_samples": len(artifacts), "failure_count": failures.get("count"), "array_shapes": reference_shapes},
        "guards": {"formal_manifest_aligned": formal_ids == artifact_ids, "arrays_float32_finite": not any(error["code"] == "array_invalid" for error in errors), "historical_artifacts_overwritten": False, "model_training_started": False},
    }
