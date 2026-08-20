from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .face_tracking import FaceLandmarker, LandmarkResult
from .optical_flow import (
    FLOW_SCHEMA_VERSION,
    OpticalFlowConfig,
    align_to_reference,
    dense_farneback,
    motion_energy,
    normalize_three_channel_flow,
    read_image,
    resize_bgr,
    to_gray,
)


PREPROCESS_SCHEMA_VERSION = "mhssa_tgcn_preprocess_v1"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
REGION_LIST = (
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose",
    "outer_lip",
)
REGION_LANDMARKS = {
    "left_eyebrow": (21, 20, 19, 18, 17),
    "right_eyebrow": (22, 23, 24, 25, 26),
    "left_eye": (36, 37, 38, 39, 40, 41),
    "right_eye": (42, 43, 44, 45, 46, 47),
    "nose": (27, 28, 29, 30, 31, 32, 33, 34, 35),
    "outer_lip": (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59),
}
REGION_ID_DICT = {name: index for index, name in enumerate(REGION_LIST)}
MAX_PATCHES_PER_REGION = 12
PATCH_SIZE = 7
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
_NATURAL_NUMBER = re.compile(r"(\d+)")


@dataclass
class PreprocessResult:
    metadata: dict[str, Any]
    patches: np.ndarray
    region_ids: np.ndarray
    masks: np.ndarray
    keypoints: np.ndarray
    flow: np.ndarray
    landmarks: np.ndarray
    onset_preview: np.ndarray
    apex_preview: np.ndarray


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_NUMBER.split(path.name)
    )


def list_sequence_frames(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)
    return sorted(
        (
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_key,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float32), np.ones(len(points), dtype=np.float32))
    )
    return homogeneous @ transform.T


def _extract_patch(image: np.ndarray, center: np.ndarray) -> np.ndarray:
    radius = PATCH_SIZE // 2
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    padded = cv2.copyMakeBorder(
        image,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    x += radius
    y += radius
    patch = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
        raise RuntimeError(f"Invalid patch shape: {patch.shape}")
    return patch.astype(np.float32)


def build_model_tensors(
    flow: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patches: list[np.ndarray] = []
    region_ids: list[int] = []
    masks: list[float] = []
    keypoints: list[np.ndarray] = []
    for region_name in REGION_LIST:
        region_points = [landmarks[index] for index in REGION_LANDMARKS[region_name]]
        for point in region_points:
            patches.append(_extract_patch(flow, point).reshape(PATCH_DIM))
            region_ids.append(REGION_ID_DICT[region_name])
            masks.append(1.0)
            keypoints.append(point.astype(np.float32))
        for _ in range(MAX_PATCHES_PER_REGION - len(region_points)):
            patches.append(np.zeros(PATCH_DIM, dtype=np.float32))
            region_ids.append(REGION_ID_DICT[region_name])
            masks.append(0.0)
            keypoints.append(np.zeros(2, dtype=np.float32))
    arrays = (
        np.stack(patches).astype(np.float32),
        np.asarray(region_ids, dtype=np.int64),
        np.asarray(masks, dtype=np.float32),
        np.stack(keypoints).astype(np.float32),
    )
    expected_shapes = ((72, 147), (72,), (72,), (72, 2))
    if tuple(array.shape for array in arrays) != expected_shapes:
        raise RuntimeError(f"Unexpected model tensor shapes: {[a.shape for a in arrays]}")
    return arrays


class SequencePreprocessor:
    def __init__(
        self,
        landmarker: FaceLandmarker,
        flow_config: OpticalFlowConfig | None = None,
    ) -> None:
        self.landmarker = landmarker
        self.flow_config = flow_config or OpticalFlowConfig()

    def process(self, record: Mapping[str, Any]) -> PreprocessResult:
        frame_paths = list_sequence_frames(Path(str(record["frame_dir"])))
        if len(frame_paths) < 2:
            raise ValueError(f"{record['sample_id']} requires at least two frames")
        frames = [resize_bgr(read_image(path), self.flow_config) for path in frame_paths]
        reference = frames[0]
        reference_gray = to_gray(reference)
        landmark_result: LandmarkResult = self.landmarker.detect(reference)

        aligned_frames = [reference]
        alignment_converged = [True]
        alignment_correlations: list[float | None] = [1.0]
        energies = [0.0]
        raw_flows = [np.zeros((*reference_gray.shape, 2), dtype=np.float32)]
        for frame in frames[1:]:
            alignment = align_to_reference(reference_gray, frame, self.flow_config)
            aligned_frames.append(alignment.image)
            alignment_converged.append(alignment.converged)
            alignment_correlations.append(alignment.correlation)
            flow = dense_farneback(reference_gray, to_gray(alignment.image), self.flow_config)
            raw_flows.append(flow)
            energies.append(motion_energy(flow))

        apex_index = int(np.argmax(np.asarray(energies, dtype=np.float32)))
        if apex_index == 0:
            apex_index = 1
        normalized_flow = normalize_three_channel_flow(
            raw_flows[apex_index], self.flow_config
        )
        scale = np.array(
            [
                self.flow_config.flow_width / self.flow_config.aligned_width,
                self.flow_config.flow_height / self.flow_config.aligned_height,
            ],
            dtype=np.float32,
        )
        flow_landmarks = landmark_result.points.astype(np.float32) * scale
        flow_landmarks[:, 0] = np.clip(
            flow_landmarks[:, 0], 0, self.flow_config.flow_width - 1
        )
        flow_landmarks[:, 1] = np.clip(
            flow_landmarks[:, 1], 0, self.flow_config.flow_height - 1
        )
        patches, region_ids, masks, keypoints = build_model_tensors(
            normalized_flow, flow_landmarks
        )
        failed_alignments = int(sum(not value for value in alignment_converged[1:]))
        quality_status = (
            "degraded"
            if landmark_result.source.startswith("estimated_") or failed_alignments > 0
            else "pass"
        )
        metadata = {
            "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
            "flow_schema_version": FLOW_SCHEMA_VERSION,
            "flow_config_sha256": self.flow_config.fingerprint(),
            "sample_id": record["sample_id"],
            "source_dataset": record["source_dataset"],
            "sample_role": record["sample_role"],
            "subject_id": record["subject_id"],
            "sequence_id": record["sequence_id"],
            "label": record.get("label"),
            "label_name": record.get("label_name"),
            "frame_count": len(frame_paths),
            "onset_frame": frame_paths[0].name,
            "apex_frame": frame_paths[apex_index].name,
            "offset_frame": frame_paths[-1].name,
            "onset_index": 0,
            "apex_index": apex_index,
            "offset_index": len(frame_paths) - 1,
            "frame_annotation_source": "estimated",
            "evaluation_scope": "engineering_only",
            "paper_reproduction_claim": False,
            "landmark_source": landmark_result.source,
            "landmark_asset_sha256": getattr(
                self.landmarker, "predictor_sha256", None
            ),
            "face_detection_source": landmark_result.detection_source,
            "face_box": list(landmark_result.face_box),
            "alignment_method": "ecc_euclidean_v1",
            "alignment_converged_count": int(sum(alignment_converged)),
            "alignment_failed_count": failed_alignments,
            "alignment_min_correlation": min(
                value for value in alignment_correlations if value is not None
            ),
            "apex_motion_energy": float(energies[apex_index]),
            "patch_shape": [72, 147],
            "region_ids_shape": [72],
            "masks_shape": [72],
            "keypoints_shape": [72, 2],
            "flow_shape": list(normalized_flow.shape),
            "valid_patch_count": int(masks.sum()),
            "quality_status": quality_status,
            "rejection_reasons": [],
        }
        return PreprocessResult(
            metadata=metadata,
            patches=patches,
            region_ids=region_ids,
            masks=masks,
            keypoints=keypoints,
            flow=normalized_flow,
            landmarks=flow_landmarks,
            onset_preview=reference,
            apex_preview=aligned_frames[apex_index],
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preprocess_artifact(result: PreprocessResult, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            patches=result.patches,
            region_ids=result.region_ids,
            masks=result.masks,
            keypoints=result.keypoints,
            flow=result.flow,
            landmarks=result.landmarks,
            label=np.asarray(-1 if result.metadata["label"] is None else result.metadata["label"]),
        )
    temporary.replace(output_path)
    metadata = dict(result.metadata)
    metadata["artifact_path"] = output_path.resolve().as_posix()
    metadata["artifact_sha256"] = _sha256_file(output_path)
    return metadata


def audit_preprocess_artifacts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    total_bytes = 0
    flow_min = float("inf")
    flow_max = float("-inf")
    expected_shapes = {
        "patches": (72, 147),
        "region_ids": (72,),
        "masks": (72,),
        "keypoints": (72, 2),
        "flow": (32, 32, 3),
        "landmarks": (68, 2),
    }
    for record in records:
        sample_id = str(record["sample_id"])
        artifact_path = Path(str(record["artifact_path"]))
        if not artifact_path.is_file():
            issues.append({"sample_id": sample_id, "code": "missing_artifact"})
            continue
        total_bytes += artifact_path.stat().st_size
        actual_sha256 = _sha256_file(artifact_path)
        if actual_sha256 != record.get("artifact_sha256"):
            issues.append(
                {
                    "sample_id": sample_id,
                    "code": "artifact_sha256_mismatch",
                    "expected": record.get("artifact_sha256"),
                    "actual": actual_sha256,
                }
            )
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                for name, expected_shape in expected_shapes.items():
                    if name not in artifact or artifact[name].shape != expected_shape:
                        issues.append(
                            {
                                "sample_id": sample_id,
                                "code": "artifact_shape_mismatch",
                                "array": name,
                                "expected": list(expected_shape),
                                "actual": (
                                    list(artifact[name].shape)
                                    if name in artifact
                                    else None
                                ),
                            }
                        )
                for name in ("patches", "masks", "keypoints", "flow", "landmarks"):
                    if name in artifact and not np.all(np.isfinite(artifact[name])):
                        issues.append(
                            {
                                "sample_id": sample_id,
                                "code": "artifact_non_finite",
                                "array": name,
                            }
                        )
                if "masks" in artifact and float(artifact["masks"].sum()) != 43.0:
                    issues.append(
                        {
                            "sample_id": sample_id,
                            "code": "invalid_valid_patch_count",
                        }
                    )
                if "flow" in artifact:
                    flow = artifact["flow"]
                    flow_min = min(flow_min, float(flow.min()))
                    flow_max = max(flow_max, float(flow.max()))
                    if flow.min() < -1.0 or flow.max() > 1.0:
                        issues.append(
                            {"sample_id": sample_id, "code": "flow_out_of_range"}
                        )
                if "label" in artifact:
                    label = int(artifact["label"])
                    expected_label = -1 if record.get("label") is None else int(record["label"])
                    if label != expected_label:
                        issues.append(
                            {
                                "sample_id": sample_id,
                                "code": "artifact_label_mismatch",
                                "expected": expected_label,
                                "actual": label,
                            }
                        )
        except (OSError, ValueError, KeyError) as exc:
            issues.append(
                {
                    "sample_id": sample_id,
                    "code": "artifact_load_failed",
                    "detail": str(exc),
                }
            )
    return {
        "schema_version": "microexpression_flow_artifact_audit_v1",
        "task_id": "FLOW-ME-001",
        "status": "pass" if records and not issues else "fail",
        "artifact_count": len(records),
        "total_bytes": total_bytes,
        "flow_value_range": [
            None if flow_min == float("inf") else flow_min,
            None if flow_max == float("-inf") else flow_max,
        ],
        "issue_count": len(issues),
        "issues": issues,
    }


def write_preprocess_audit(report: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return _sha256_file(output_path)


def write_visualization(result: PreprocessResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onset = cv2.resize(result.onset_preview, (256, 256), interpolation=cv2.INTER_CUBIC)
    apex = cv2.resize(result.apex_preview, (256, 256), interpolation=cv2.INTER_CUBIC)
    u, v = result.flow[..., 0], result.flow[..., 1]
    magnitude = result.flow[..., 2]
    angle = cv2.phase(u, v, angleInDegrees=True)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * 255.0, 0, 255).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for point in result.landmarks[17:60]:
        cv2.circle(flow_bgr, tuple(np.rint(point).astype(int)), 1, (255, 255, 255), -1)
    flow_bgr = cv2.resize(flow_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack((onset, apex, flow_bgr))
    labels = ("onset", "aligned apex", "u/v/o + landmarks")
    for index, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (index * 256 + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    encoded_ok, encoded = cv2.imencode(".png", canvas)
    if not encoded_ok:
        raise RuntimeError("Unable to encode visualization")
    encoded.tofile(output_path)


def write_preprocess_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    summary_path: Path,
    config: OpticalFlowConfig,
    landmark_mode: str,
    landmark_asset_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    quality_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    detection_counts: dict[str, int] = {}
    for record in records:
        quality = str(record["quality_status"])
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        label = str(record.get("label_name"))
        label_counts[label] = label_counts.get(label, 0) + 1
        detection = str(record.get("face_detection_source"))
        detection_counts[detection] = detection_counts.get(detection, 0) + 1
    sample_ids = [str(record["sample_id"]) for record in records]
    frame_counts = [int(record["frame_count"]) for record in records]
    apex_indices = [int(record["apex_index"]) for record in records]
    artifact_paths = [Path(str(record["artifact_path"])) for record in records]
    validation = {
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
        "all_artifacts_available": all(path.is_file() for path in artifact_paths),
        "all_tensor_shapes_fixed": all(
            record["patch_shape"] == [72, 147]
            and record["region_ids_shape"] == [72]
            and record["masks_shape"] == [72]
            and record["keypoints_shape"] == [72, 2]
            and record["flow_shape"] == [config.flow_height, config.flow_width, 3]
            for record in records
        ),
        "all_valid_patch_counts_fixed": all(
            int(record["valid_patch_count"]) == 43 for record in records
        ),
        "all_landmark_assets_versioned": all(
            record.get("landmark_source") != "dlib_68_v1"
            or bool(record.get("landmark_asset_sha256"))
            for record in records
        ),
    }
    summary = {
        "schema_version": "microexpression_flow_summary_v1",
        "task_id": "FLOW-ME-001",
        "status": "pass" if records else "empty",
        "sample_count": len(records),
        "quality_counts": dict(sorted(quality_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "face_detection_counts": dict(sorted(detection_counts.items())),
        "landmark_mode": landmark_mode,
        "landmark_asset_sha256": landmark_asset_sha256,
        "flow_schema_version": FLOW_SCHEMA_VERSION,
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "flow_config": {
            "aligned_size": [config.aligned_height, config.aligned_width],
            "flow_size": [config.flow_height, config.flow_width],
            "normalization": "u=tanh(u/2), v=tanh(v/2), o=tanh(magnitude/2)",
            "sha256": config.fingerprint(),
        },
        "tensor_contract": {
            "patches": [72, 147],
            "region_ids": [72],
            "masks": [72],
            "keypoints": [72, 2],
            "flow": [config.flow_height, config.flow_width, 3],
            "valid_patches": 43,
        },
        "frame_statistics": {
            "total": sum(frame_counts),
            "min": min(frame_counts) if frame_counts else 0,
            "max": max(frame_counts) if frame_counts else 0,
        },
        "alignment": {
            "method": "ecc_euclidean_v1",
            "failed_frame_count": sum(
                int(record["alignment_failed_count"]) for record in records
            ),
        },
        "estimated_apex": {
            "min_index": min(apex_indices) if apex_indices else None,
            "max_index": max(apex_indices) if apex_indices else None,
        },
        "validation": validation,
        "manifest": {
            "path": manifest_path.resolve().as_posix(),
            "sha256": _sha256_file(manifest_path),
        },
    }
    if not all(validation.values()):
        summary["status"] = "fail"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary
