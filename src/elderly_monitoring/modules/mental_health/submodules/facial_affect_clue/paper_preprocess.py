from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

from .apex_spotting import (
    APEX_SCHEMA_VERSION,
    ROI_NAMES,
    ApexSpottingResult,
    DCRoIsConfig,
    spot_apex_dc_rois,
)
from .face_tracking import FaceLandmarker, LandmarkResult
from .optical_flow import read_image
from .preprocess import list_sequence_frames


PAPER_PREPROCESS_SCHEMA_VERSION = "mhssa_tgcn_paper_preprocess_v2"
PAPER_FLOW_SCHEMA_VERSION = "farneback_u_v_optical_strain_robust_v2"
OPTICAL_STRAIN_FORMULA_VERSION = "thesis_equation_3_2_central_gradient_v1"
PATCH_ORDER_VERSION = "thesis_muscle_propagation_43_v1"
FACE_ALIGNMENT_VERSION = "dlib_anchor_similarity_onset_fixed_256_v2"
PREPROCESS_VARIANTS = ("base", "denoise", "illumination", "combined")

REGION_LIST = (
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose",
    "outer_lip",
)

# Dlib uses zero-based indices. Each tuple follows the thesis' muscle-motion
# propagation direction rather than the predictor's contour enumeration.
ORDERED_REGION_LANDMARKS = {
    "left_eyebrow": (21, 20, 19, 18, 17),
    "right_eyebrow": (22, 23, 24, 25, 26),
    "left_eye": (39, 38, 40, 37, 41, 36),
    "right_eye": (42, 43, 47, 44, 46, 45),
    "nose": (27, 28, 29, 30, 31, 35, 32, 34, 33),
    "outer_lip": (48, 54, 49, 53, 59, 55, 50, 52, 58, 56, 51, 57),
}
ORDERED_LANDMARK_INDICES = tuple(
    index for region in REGION_LIST for index in ORDERED_REGION_LANDMARKS[region]
)
REGION_IDS = tuple(
    region_id
    for region_id, region in enumerate(REGION_LIST)
    for _ in ORDERED_REGION_LANDMARKS[region]
)
REGION_OFFSETS = (0, 5, 10, 16, 22, 31, 43)


@dataclass(frozen=True)
class PaperPreprocessConfig:
    aligned_width: int = 256
    aligned_height: int = 256
    flow_width: int = 32
    flow_height: int = 32
    patch_size: int = 5
    preprocess_variant: str = "combined"
    gamma: float = 0.8
    gaussian_kernel: int = 5
    median_kernel: int = 3
    flow_estimator: str = "farneback"
    normalization_quantile: float = 0.995
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def __post_init__(self) -> None:
        if (self.aligned_width, self.aligned_height) != (256, 256):
            raise ValueError("The paper-v2 face contract is fixed at 256x256")
        if self.patch_size != 5:
            raise ValueError("The SMIC paper-v2 patch contract is fixed at 5x5")
        if self.preprocess_variant not in PREPROCESS_VARIANTS:
            raise ValueError(f"Unknown preprocessing variant: {self.preprocess_variant}")
        if self.flow_estimator not in {"farneback", "tvl1"}:
            raise ValueError("flow_estimator must be 'farneback' or 'tvl1'")
        if not 0.5 <= self.normalization_quantile <= 1.0:
            raise ValueError("normalization_quantile must be in [0.5, 1.0]")
        for name, kernel in (
            ("gaussian_kernel", self.gaussian_kernel),
            ("median_kernel", self.median_kernel),
        ):
            if kernel < 3 or kernel % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 3")

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * 3

    def fingerprint(self, apex_config: DCRoIsConfig) -> str:
        payload = {
            "preprocess_schema_version": PAPER_PREPROCESS_SCHEMA_VERSION,
            "flow_schema_version": PAPER_FLOW_SCHEMA_VERSION,
            "optical_strain_formula_version": OPTICAL_STRAIN_FORMULA_VERSION,
            "patch_order_version": PATCH_ORDER_VERSION,
            "face_alignment_version": FACE_ALIGNMENT_VERSION,
            "config": asdict(self),
            "apex_config": asdict(apex_config),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FaceAlignmentResult:
    frames: tuple[np.ndarray, ...]
    landmarks: np.ndarray
    transform: np.ndarray
    frame_landmarks: tuple[np.ndarray, ...] = ()
    transforms: tuple[np.ndarray, ...] = ()


@dataclass
class PaperPreprocessResult:
    metadata: dict[str, Any]
    patches: np.ndarray
    region_ids: np.ndarray
    region_offsets: np.ndarray
    keypoints: np.ndarray
    ordered_landmark_indices: np.ndarray
    flow: np.ndarray
    landmarks: np.ndarray
    apex_frame_scores: np.ndarray
    apex_roi_scores: np.ndarray
    apex_roi_boxes: np.ndarray
    onset_preview: np.ndarray
    apex_preview: np.ndarray


class DenseFlowEstimator(Protocol):
    version: str

    def estimate(
        self,
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
    ) -> np.ndarray:
        ...


class FarnebackFlowEstimator:
    version = "opencv_farneback_v1"

    def __init__(self, config: PaperPreprocessConfig) -> None:
        self.config = config

    def estimate(
        self,
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
    ) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            reference_gray,
            target_gray,
            None,
            self.config.pyr_scale,
            self.config.levels,
            self.config.winsize,
            self.config.iterations,
            self.config.poly_n,
            self.config.poly_sigma,
            self.config.flags,
        ).astype(np.float32)


class TVL1FlowEstimator:
    version = "opencv_dual_tvl1_candidate_v1"

    def __init__(self) -> None:
        factory = None
        if hasattr(cv2, "optflow") and hasattr(
            cv2.optflow, "DualTVL1OpticalFlow_create"
        ):
            factory = cv2.optflow.DualTVL1OpticalFlow_create
        elif hasattr(cv2, "DualTVL1OpticalFlow_create"):
            factory = cv2.DualTVL1OpticalFlow_create
        if factory is None:
            raise RuntimeError(
                "TV-L1 requires an OpenCV build with the optflow contrib module"
            )
        self._estimator = factory()

    def estimate(
        self,
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
    ) -> np.ndarray:
        return self._estimator.calc(reference_gray, target_gray, None).astype(
            np.float32
        )


def create_flow_estimator(config: PaperPreprocessConfig) -> DenseFlowEstimator:
    if config.flow_estimator == "farneback":
        return FarnebackFlowEstimator(config)
    return TVL1FlowEstimator()


def _alignment_anchors(landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            landmarks[36:42].mean(axis=0),
            landmarks[42:48].mean(axis=0),
            landmarks[30],
            landmarks[[48, 54]].mean(axis=0),
        ),
        dtype=np.float32,
    )


def _canonical_alignment_anchors(config: PaperPreprocessConfig) -> np.ndarray:
    normalized = np.asarray(
        ((0.35, 0.38), (0.65, 0.38), (0.50, 0.56), (0.50, 0.73)),
        dtype=np.float32,
    )
    return normalized * np.asarray(
        (config.aligned_width - 1, config.aligned_height - 1), dtype=np.float32
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float32), np.ones(len(points), dtype=np.float32))
    )
    return (homogeneous @ transform.T).astype(np.float32)


def align_face_sequence(
    frames: Sequence[np.ndarray],
    onset_landmarks: np.ndarray,
    config: PaperPreprocessConfig,
) -> FaceAlignmentResult:
    if not frames:
        raise ValueError("Cannot align an empty sequence")
    transform, _ = cv2.estimateAffinePartial2D(
        _alignment_anchors(onset_landmarks),
        _canonical_alignment_anchors(config),
        method=cv2.LMEDS,
    )
    if transform is None or not np.all(np.isfinite(transform)):
        raise RuntimeError("Dlib anchor similarity alignment failed")
    aligned = tuple(
        cv2.warpAffine(
            frame,
            transform.astype(np.float32),
            (config.aligned_width, config.aligned_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        for frame in frames
    )
    landmarks = _transform_points(onset_landmarks, transform)
    landmarks[:, 0] = np.clip(landmarks[:, 0], 0, config.aligned_width - 1)
    landmarks[:, 1] = np.clip(landmarks[:, 1], 0, config.aligned_height - 1)
    typed_transform = transform.astype(np.float32)
    return FaceAlignmentResult(
        aligned,
        landmarks,
        typed_transform,
        tuple(landmarks.copy() for _ in frames),
        tuple(typed_transform.copy() for _ in frames),
    )


def apply_preprocessing_variant(
    image: np.ndarray,
    config: PaperPreprocessConfig,
) -> np.ndarray:
    result = image.copy()
    if config.preprocess_variant in {"denoise", "combined"}:
        result = cv2.GaussianBlur(
            result,
            (config.gaussian_kernel, config.gaussian_kernel),
            0,
        )
        result = cv2.medianBlur(result, config.median_kernel)
    if config.preprocess_variant in {"illumination", "combined"}:
        ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        luminance = cv2.equalizeHist(ycrcb[..., 0])
        normalized = np.power(
            luminance.astype(np.float32) / 255.0,
            config.gamma,
        )
        ycrcb[..., 0] = np.clip(np.rint(normalized * 255.0), 0, 255).astype(
            np.uint8
        )
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return result


def resize_vector_flow(
    flow: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    original_height, original_width = flow.shape[:2]
    resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    resized[..., 0] *= width / original_width
    resized[..., 1] *= height / original_height
    return resized


def optical_strain(flow: np.ndarray) -> np.ndarray:
    """Compute thesis equation 3.2 from horizontal and vertical flow."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    strain_squared = (
        np.square(du_dx)
        + np.square(dv_dy)
        + 0.5 * np.square(du_dy + dv_dx)
    )
    return np.sqrt(np.maximum(strain_squared, 0.0)).astype(np.float32)


def normalize_flow_channels(
    flow: np.ndarray,
    strain: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if strain.shape != flow.shape[:2]:
        raise ValueError("Strain shape must match the flow spatial shape")
    epsilon = float(np.finfo(np.float32).eps)
    vector_scale = max(
        float(np.quantile(np.abs(flow).reshape(-1), quantile)), epsilon
    )
    strain_scale = max(
        float(np.quantile(strain.reshape(-1), quantile)), epsilon
    )
    normalized = np.stack(
        (
            np.clip(flow[..., 0] / vector_scale, -1.0, 1.0),
            np.clip(flow[..., 1] / vector_scale, -1.0, 1.0),
            np.clip(strain / strain_scale, 0.0, 1.0),
        ),
        axis=2,
    ).astype(np.float32)
    return normalized, {
        "vector_scale": vector_scale,
        "strain_scale": strain_scale,
        "quantile": float(quantile),
    }


def _extract_patch(
    flow: np.ndarray,
    center: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    radius = patch_size // 2
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    padded = cv2.copyMakeBorder(
        flow,
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
    expected = (patch_size, patch_size, 3)
    if patch.shape != expected:
        raise RuntimeError(f"Expected patch {expected}, got {patch.shape}")
    return patch.astype(np.float32)


def build_paper_patch_tensors(
    flow: np.ndarray,
    landmarks: np.ndarray,
    patch_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if flow.ndim != 3 or flow.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 flow, got {flow.shape}")
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected 68x2 landmarks, got {landmarks.shape}")
    ordered_indices = np.asarray(ORDERED_LANDMARK_INDICES, dtype=np.int64)
    keypoints = landmarks[ordered_indices].astype(np.float32)
    patches = np.stack(
        [_extract_patch(flow, point, patch_size).reshape(-1) for point in keypoints]
    ).astype(np.float32)
    region_ids = np.asarray(REGION_IDS, dtype=np.int64)
    offsets = np.asarray(REGION_OFFSETS, dtype=np.int64)
    expected_patch_dim = patch_size * patch_size * 3
    if patches.shape != (43, expected_patch_dim):
        raise RuntimeError(f"Unexpected paper patch shape: {patches.shape}")
    return patches, region_ids, offsets, keypoints, ordered_indices


class PaperSequencePreprocessor:
    def __init__(
        self,
        landmarker: FaceLandmarker,
        config: PaperPreprocessConfig | None = None,
        apex_config: DCRoIsConfig | None = None,
        flow_estimator: DenseFlowEstimator | None = None,
    ) -> None:
        self.landmarker = landmarker
        self.config = config or PaperPreprocessConfig()
        self.apex_config = apex_config or DCRoIsConfig()
        self.flow_estimator = flow_estimator or create_flow_estimator(self.config)

    def process(self, record: Mapping[str, Any]) -> PaperPreprocessResult:
        frame_paths = list_sequence_frames(Path(str(record["frame_dir"])))
        if len(frame_paths) < 3:
            raise ValueError(f"{record['sample_id']} requires at least three frames")
        frames = [read_image(path) for path in frame_paths]
        landmark_result: LandmarkResult = self.landmarker.detect(frames[0])
        alignment = align_face_sequence(
            frames,
            landmark_result.points,
            self.config,
        )
        base_grays = tuple(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in alignment.frames
        )
        apex: ApexSpottingResult = spot_apex_dc_rois(
            base_grays,
            alignment.landmarks,
            self.apex_config,
        )
        apex_landmark_result: LandmarkResult = self.landmarker.detect(
            frames[apex.apex_index]
        )

        onset = apply_preprocessing_variant(alignment.frames[0], self.config)
        apex_frame = apply_preprocessing_variant(
            alignment.frames[apex.apex_index], self.config
        )
        raw_flow = self.flow_estimator.estimate(
            cv2.cvtColor(onset, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(apex_frame, cv2.COLOR_BGR2GRAY),
        )
        resized_flow = resize_vector_flow(
            raw_flow,
            self.config.flow_width,
            self.config.flow_height,
        )
        strain = optical_strain(resized_flow)
        normalized_flow, normalization = normalize_flow_channels(
            resized_flow,
            strain,
            self.config.normalization_quantile,
        )
        landmark_scale = np.asarray(
            (
                self.config.flow_width / self.config.aligned_width,
                self.config.flow_height / self.config.aligned_height,
            ),
            dtype=np.float32,
        )
        apex_landmarks = _transform_points(
            apex_landmark_result.points,
            alignment.transform,
        )
        apex_landmarks[:, 0] = np.clip(
            apex_landmarks[:, 0], 0, self.config.aligned_width - 1
        )
        apex_landmarks[:, 1] = np.clip(
            apex_landmarks[:, 1], 0, self.config.aligned_height - 1
        )
        flow_landmarks = apex_landmarks * landmark_scale
        flow_landmarks[:, 0] = np.clip(
            flow_landmarks[:, 0], 0, self.config.flow_width - 1
        )
        flow_landmarks[:, 1] = np.clip(
            flow_landmarks[:, 1], 0, self.config.flow_height - 1
        )
        patches, region_ids, region_offsets, keypoints, ordered_indices = (
            build_paper_patch_tensors(
                normalized_flow,
                flow_landmarks,
                self.config.patch_size,
            )
        )

        landmark_results = (landmark_result, apex_landmark_result)
        detection_sources = Counter(result.detection_source for result in landmark_results)
        quality_status = (
            "degraded"
            if any(result.source.startswith("estimated_") for result in landmark_results)
            or "cropped_frame_fallback" in detection_sources
            else "pass"
        )
        config_sha256 = self.config.fingerprint(self.apex_config)
        metadata = {
            "task_id": "FLOW-ME-002",
            "preprocess_schema_version": PAPER_PREPROCESS_SCHEMA_VERSION,
            "flow_schema_version": PAPER_FLOW_SCHEMA_VERSION,
            "apex_schema_version": APEX_SCHEMA_VERSION,
            "optical_strain_formula_version": OPTICAL_STRAIN_FORMULA_VERSION,
            "patch_order_version": PATCH_ORDER_VERSION,
            "face_alignment_version": FACE_ALIGNMENT_VERSION,
            "preprocess_config_sha256": config_sha256,
            "apex_config_sha256": self.apex_config.fingerprint(),
            "sample_id": record["sample_id"],
            "source_dataset": record["source_dataset"],
            "sample_role": record["sample_role"],
            "subject_id": record["subject_id"],
            "sequence_id": record["sequence_id"],
            "label": record.get("label"),
            "label_name": record.get("label_name"),
            "frame_count": len(frame_paths),
            "onset_frame": frame_paths[0].name,
            "apex_frame": frame_paths[apex.apex_index].name,
            "offset_frame": frame_paths[-1].name,
            "onset_index": 0,
            "apex_index": apex.apex_index,
            "offset_index": len(frame_paths) - 1,
            "apex_method": "dc_rois",
            "apex_implementation": APEX_SCHEMA_VERSION,
            "apex_score": float(apex.frame_scores[apex.apex_index]),
            "apex_boundary_status": (
                "near_onset"
                if apex.apex_index == 1
                else (
                    "near_offset"
                    if apex.apex_index == len(frame_paths) - 2
                    else "interior"
                )
            ),
            "apex_roi_score": {
                name: float(apex.roi_scores[apex.apex_index, index])
                for index, name in enumerate(ROI_NAMES)
            },
            "frame_annotation_source": "estimated_dc_rois",
            "face_normalization": "aligned_256x256",
            "alignment_method": FACE_ALIGNMENT_VERSION,
            "alignment_transform": alignment.transform.tolist(),
            "landmark_source": landmark_result.source,
            "landmark_asset_sha256": getattr(
                self.landmarker, "predictor_sha256", None
            ),
            "face_detection_source": landmark_result.detection_source,
            "face_detection_source_counts": dict(sorted(detection_sources.items())),
            "per_frame_landmark_alignment": False,
            "landmark_detection_frame_count": len(landmark_results),
            "temporal_correspondence": "fixed_onset_transform",
            "source_face_box": list(landmark_result.face_box),
            "denoise_variant": (
                "gaussian_median"
                if self.config.preprocess_variant in {"denoise", "combined"}
                else "none"
            ),
            "illumination_variant": (
                "histogram_equalization_gamma"
                if self.config.preprocess_variant in {"illumination", "combined"}
                else "none"
            ),
            "preprocess_variant": self.config.preprocess_variant,
            "flow_estimator": self.config.flow_estimator,
            "flow_estimator_version": self.flow_estimator.version,
            "flow_channels": "u_v_optical_strain",
            "flow_normalization": "per_sample_robust_clip",
            "flow_normalization_parameters": normalization,
            "aligned_shape": [
                self.config.aligned_height,
                self.config.aligned_width,
                3,
            ],
            "flow_shape": list(normalized_flow.shape),
            "patch_size": self.config.patch_size,
            "patch_dim": self.config.patch_dim,
            "patch_shape": list(patches.shape),
            "valid_patch_count": 43,
            "ordered_landmark_indices_zero_based": list(ORDERED_LANDMARK_INDICES),
            "region_offsets": list(REGION_OFFSETS),
            "paper_method_compliance": "reimplemented_farneback_approximation",
            "paper_reproduction_claim": False,
            "evaluation_scope": "offline_training_candidate",
            "historical_engineering_baseline": False,
            "deployment_eligible": False,
            "quality_status": quality_status,
            "rejection_reasons": [],
        }
        return PaperPreprocessResult(
            metadata=metadata,
            patches=patches,
            region_ids=region_ids,
            region_offsets=region_offsets,
            keypoints=keypoints,
            ordered_landmark_indices=ordered_indices,
            flow=normalized_flow,
            landmarks=flow_landmarks.astype(np.float32),
            apex_frame_scores=apex.frame_scores,
            apex_roi_scores=apex.roi_scores,
            apex_roi_boxes=apex.roi_boxes,
            onset_preview=onset,
            apex_preview=apex_frame,
        )


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_paper_artifact(
    result: PaperPreprocessResult,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            patches=result.patches,
            region_ids=result.region_ids,
            region_offsets=result.region_offsets,
            keypoints=result.keypoints,
            ordered_landmark_indices=result.ordered_landmark_indices,
            flow=result.flow,
            landmarks=result.landmarks,
            apex_frame_scores=result.apex_frame_scores,
            apex_roi_scores=result.apex_roi_scores,
            apex_roi_boxes=result.apex_roi_boxes,
            label=np.asarray(
                -1 if result.metadata["label"] is None else result.metadata["label"]
            ),
        )
    temporary.replace(output_path)
    metadata = dict(result.metadata)
    metadata["artifact_path"] = output_path.resolve().as_posix()
    metadata["artifact_sha256"] = sha256_file(output_path)
    return metadata


def audit_paper_artifacts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    expected_shapes = {
        "patches": (43, 75),
        "region_ids": (43,),
        "region_offsets": (7,),
        "keypoints": (43, 2),
        "ordered_landmark_indices": (43,),
        "flow": (32, 32, 3),
        "landmarks": (68, 2),
    }
    seen: set[str] = set()
    total_bytes = 0
    flow_min = float("inf")
    flow_max = float("-inf")
    for record in records:
        sample_id = str(record.get("sample_id"))
        if record.get("quality_status") != "pass":
            warnings.append(
                {
                    "sample_id": sample_id,
                    "code": "degraded_landmark_detection",
                    "face_detection_source_counts": record.get(
                        "face_detection_source_counts"
                    ),
                }
            )
        if record.get("apex_boundary_status") != "interior":
            warnings.append(
                {
                    "sample_id": sample_id,
                    "code": "apex_near_sequence_boundary",
                    "status": record.get("apex_boundary_status"),
                    "apex_index": record.get("apex_index"),
                    "frame_count": record.get("frame_count"),
                }
            )
        if float(record.get("apex_score", 0.0)) < 0.01:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "code": "low_dc_rois_difference",
                    "apex_score": record.get("apex_score"),
                }
            )
        if sample_id in seen:
            issues.append({"sample_id": sample_id, "code": "duplicate_sample_id"})
        seen.add(sample_id)
        path = Path(str(record.get("artifact_path", "")))
        if not path.is_file():
            issues.append({"sample_id": sample_id, "code": "missing_artifact"})
            continue
        total_bytes += path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_sha256 != record.get("artifact_sha256"):
            issues.append(
                {
                    "sample_id": sample_id,
                    "code": "artifact_sha256_mismatch",
                    "actual": actual_sha256,
                }
            )
        try:
            with np.load(path, allow_pickle=False) as artifact:
                for name, shape in expected_shapes.items():
                    actual = artifact[name].shape if name in artifact else None
                    if actual != shape:
                        issues.append(
                            {
                                "sample_id": sample_id,
                                "code": "artifact_shape_mismatch",
                                "array": name,
                                "expected": list(shape),
                                "actual": None if actual is None else list(actual),
                            }
                        )
                for name in (
                    "patches",
                    "keypoints",
                    "flow",
                    "landmarks",
                    "apex_frame_scores",
                    "apex_roi_scores",
                ):
                    if name in artifact and not np.all(np.isfinite(artifact[name])):
                        issues.append(
                            {
                                "sample_id": sample_id,
                                "code": "artifact_non_finite",
                                "array": name,
                            }
                        )
                if "ordered_landmark_indices" in artifact and not np.array_equal(
                    artifact["ordered_landmark_indices"],
                    np.asarray(ORDERED_LANDMARK_INDICES, dtype=np.int64),
                ):
                    issues.append(
                        {"sample_id": sample_id, "code": "patch_order_mismatch"}
                    )
                if "region_offsets" in artifact and not np.array_equal(
                    artifact["region_offsets"],
                    np.asarray(REGION_OFFSETS, dtype=np.int64),
                ):
                    issues.append(
                        {"sample_id": sample_id, "code": "region_offsets_mismatch"}
                    )
                if "flow" in artifact:
                    flow = artifact["flow"]
                    flow_min = min(flow_min, float(flow.min()))
                    flow_max = max(flow_max, float(flow.max()))
                    if float(flow.min()) < -1.000001 or float(flow.max()) > 1.000001:
                        issues.append(
                            {"sample_id": sample_id, "code": "flow_out_of_range"}
                        )
                    if float(flow[..., 2].min()) < -1e-7:
                        issues.append(
                            {"sample_id": sample_id, "code": "negative_optical_strain"}
                        )
                if "apex_frame_scores" in artifact:
                    apex_index = int(record.get("apex_index", -1))
                    frame_count = int(record.get("frame_count", -1))
                    scores = artifact["apex_frame_scores"]
                    if scores.shape != (frame_count,) or not 1 <= apex_index < frame_count - 1:
                        issues.append(
                            {"sample_id": sample_id, "code": "invalid_apex_contract"}
                        )
                if "label" in artifact:
                    expected_label = (
                        -1 if record.get("label") is None else int(record["label"])
                    )
                    if int(artifact["label"]) != expected_label:
                        issues.append(
                            {"sample_id": sample_id, "code": "label_mismatch"}
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
        "schema_version": "microexpression_paper_v2_artifact_audit_v1",
        "task_id": "FLOW-ME-002",
        "status": "pass" if records and not issues else "fail",
        "artifact_count": len(records),
        "unique_sample_count": len(seen),
        "total_bytes": total_bytes,
        "flow_value_range": [
            None if flow_min == float("inf") else flow_min,
            None if flow_max == float("-inf") else flow_max,
        ],
        "issue_count": len(issues),
        "issues": issues,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _write_json(payload: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return sha256_file(output_path)


def write_paper_reports(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    summary_path: Path,
    config: PaperPreprocessConfig,
    apex_config: DCRoIsConfig,
    landmark_mode: str,
    landmark_asset_sha256: str | None,
    input_manifest_path: Path,
    source_code_paths: Sequence[Path],
    config_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    code_hashes = {
        path.resolve().as_posix(): sha256_file(path)
        for path in source_code_paths
        if path.is_file()
    }
    summary: dict[str, Any] = {
        "schema_version": "microexpression_paper_v2_preprocess_summary_v1",
        "task_id": "FLOW-ME-002",
        "status": "pass" if records else "fail",
        "artifact_count": len(records),
        "sample_count": len({str(record["sample_id"]) for record in records}),
        "subject_count": len({str(record["subject_id"]) for record in records}),
        "label_distribution": dict(
            sorted(Counter(str(record.get("label_name")) for record in records).items())
        ),
        "quality_distribution": dict(
            sorted(Counter(str(record["quality_status"]) for record in records).items())
        ),
        "apex_index_range": (
            [
                min(int(record["apex_index"]) for record in records),
                max(int(record["apex_index"]) for record in records),
            ]
            if records
            else [None, None]
        ),
        "frame_count_total": sum(int(record["frame_count"]) for record in records),
        "frame_annotation_source": "estimated_dc_rois",
        "paper_method_compliance": "reimplemented_farneback_approximation",
        "preprocess_config": asdict(config),
        "apex_config": asdict(apex_config),
        "preprocess_config_sha256": config.fingerprint(apex_config),
        "landmark_mode": landmark_mode,
        "landmark_asset_sha256": landmark_asset_sha256,
        "input_manifest_path": input_manifest_path.resolve().as_posix(),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "artifact_manifest_path": manifest_path.resolve().as_posix(),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "source_code_sha256": code_hashes,
        "config_path": config_path.resolve().as_posix() if config_path else None,
        "config_file_sha256": (
            sha256_file(config_path) if config_path and config_path.is_file() else None
        ),
    }
    summary_sha256 = _write_json(summary, summary_path)
    summary["summary_path"] = summary_path.resolve().as_posix()
    summary["summary_sha256"] = summary_sha256
    return summary


def write_paper_audit(report: Mapping[str, Any], output_path: Path) -> str:
    return _write_json(report, output_path)


def _signed_channel_image(channel: np.ndarray) -> np.ndarray:
    normalized = np.clip((channel + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def write_paper_visualization(
    result: PaperPreprocessResult,
    output_path: Path,
) -> None:
    panel_size = 384
    onset = cv2.resize(result.onset_preview, (panel_size, panel_size))
    apex = cv2.resize(result.apex_preview, (panel_size, panel_size))
    scale = panel_size / 256.0
    colors = ((40, 180, 255), (255, 170, 40), (80, 220, 100))
    for index, box in enumerate(result.apex_roi_boxes):
        left, top, right, bottom = (int(round(float(value) * scale)) for value in box)
        cv2.rectangle(onset, (left, top), (right, bottom), colors[index], 2)
        cv2.putText(
            onset,
            ROI_NAMES[index],
            (left, max(15, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colors[index],
            1,
            cv2.LINE_AA,
        )

    region_colors = (
        (40, 80, 240),
        (40, 190, 240),
        (50, 210, 80),
        (220, 180, 40),
        (220, 80, 180),
        (170, 80, 240),
    )
    aligned_keypoints = result.keypoints * (panel_size / 32.0)
    for region_id, (start, stop) in enumerate(zip(REGION_OFFSETS[:-1], REGION_OFFSETS[1:])):
        points = np.rint(aligned_keypoints[start:stop]).astype(np.int32)
        cv2.polylines(apex, [points], False, region_colors[region_id], 1, cv2.LINE_AA)
        for local_index, point in enumerate(points, start=1):
            center = (int(point[0]), int(point[1]))
            cv2.circle(apex, center, 3, region_colors[region_id], -1, cv2.LINE_AA)
            cv2.putText(
                apex,
                str(local_index),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                region_colors[region_id],
                1,
                cv2.LINE_AA,
            )

    curve = np.full((panel_size, panel_size, 3), 255, dtype=np.uint8)
    scores = result.apex_frame_scores
    maximum = max(float(scores.max()), np.finfo(np.float32).eps)
    margin = 32
    if len(scores) > 1:
        xs = np.linspace(margin, panel_size - margin, len(scores))
        ys = panel_size - margin - scores / maximum * (panel_size - 2 * margin)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(curve, [points], False, (30, 90, 210), 2, cv2.LINE_AA)
        apex_index = int(result.metadata["apex_index"])
        cv2.line(
            curve,
            (int(xs[apex_index]), margin),
            (int(xs[apex_index]), panel_size - margin),
            (40, 170, 40),
            2,
        )
    cv2.putText(
        curve,
        f"D&C-RoIs apex={result.metadata['apex_index']}",
        (margin, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    u = cv2.resize(_signed_channel_image(result.flow[..., 0]), (panel_size, 128))
    v = cv2.resize(_signed_channel_image(result.flow[..., 1]), (panel_size, 128))
    strain_gray = np.clip(result.flow[..., 2] * 255.0, 0, 255).astype(np.uint8)
    strain = cv2.resize(
        cv2.applyColorMap(strain_gray, cv2.COLORMAP_INFERNO),
        (panel_size, 128),
    )
    flow_panel = np.vstack((u, v, strain))
    for y, label in ((18, "u"), (146, "v"), (274, "optical strain")):
        cv2.putText(
            flow_panel,
            label,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    canvas = np.vstack((np.hstack((onset, apex)), np.hstack((curve, flow_panel))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix, canvas)
    if not ok:
        raise RuntimeError(f"Unable to encode visualization: {output_path}")
    encoded.tofile(output_path)
