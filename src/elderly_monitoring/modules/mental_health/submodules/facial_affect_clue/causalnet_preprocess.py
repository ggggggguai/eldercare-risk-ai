from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .optical_flow import read_image
from .paper_preprocess import optical_strain
from .preprocess import list_sequence_frames


CAUSALNET_PREPROCESS_SCHEMA_VERSION = "causalnet_four_route_preprocess_v1"
CAUSALNET_FLOW_SCHEMA_VERSION = "farneback_u_v_optical_strain_v1"
CAUSALNET_DIRECTION_SCHEMA_VERSION = "hsv_direction_saturation_v1"
CAUSALNET_UPSTREAM_COMMIT = "7bdf163face030eeae4358fc7638cf538305acce"
CAUSALNET_UPSTREAM_REPOSITORY = "https://github.com/tony19980810/CausalNet"
CAUSALNET_UPSTREAM_LICENSE = "MIT"
LABEL_NAMES = {0: "negative", 1: "positive", 2: "surprise"}
SPATIAL_VARIANTS = ("source_compatible_roi", "full_face")

# Coordinates are the frozen 28x28 source-repository coordinates. They are
# (row, column) centers, and the upstream implementation crops 14x14 blocks.
UPSTREAM_ROI_CENTERS = (
    (9, 11),   # left eye
    (21, 10),  # left lip
    (15, 17),  # nose (kept for provenance; not used by the 2x2 collage)
    (10, 22),  # right eye
    (20, 22),  # right lip
)


@dataclass(frozen=True)
class CausalNetPreprocessConfig:
    image_size: int = 28
    crop_size: int = 14
    flow_width: int = 32
    flow_height: int = 32
    spatial_variant: str = "source_compatible_roi"
    flow_estimator: str = "farneback"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    magnitude_quantile: float = 0.995

    def __post_init__(self) -> None:
        if self.image_size != 28:
            raise ValueError("CausalNet input size is fixed at 28")
        if self.crop_size != 14:
            raise ValueError("CausalNet source crop size is fixed at 14")
        if self.spatial_variant not in SPATIAL_VARIANTS:
            raise ValueError(f"Unknown spatial variant: {self.spatial_variant}")
        if self.flow_estimator != "farneback":
            raise ValueError("FLOW-ME-004 currently freezes Farneback")
        if not 0.5 <= self.magnitude_quantile <= 1.0:
            raise ValueError("magnitude_quantile must be in [0.5, 1.0]")

    def fingerprint(self) -> str:
        payload = {
            "schema_version": CAUSALNET_PREPROCESS_SCHEMA_VERSION,
            "flow_schema_version": CAUSALNET_FLOW_SCHEMA_VERSION,
            "direction_schema_version": CAUSALNET_DIRECTION_SCHEMA_VERSION,
            "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
            "config": asdict(self),
            "upstream_roi_centers": UPSTREAM_ROI_CENTERS,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_code_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _alignment_transform(record: Mapping[str, Any]) -> np.ndarray:
    transform = np.asarray(record.get("alignment_transform"), dtype=np.float32)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid alignment_transform for {record['sample_id']}")
    return transform


def _aligned_frame(path: Path, transform: np.ndarray) -> np.ndarray:
    frame = read_image(path)
    return cv2.warpAffine(
        frame,
        transform,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _flow_config(config: CausalNetPreprocessConfig) -> dict[str, Any]:
    return {
        "pyr_scale": config.pyr_scale,
        "levels": config.levels,
        "winsize": config.winsize,
        "iterations": config.iterations,
        "poly_n": config.poly_n,
        "poly_sigma": config.poly_sigma,
        "flags": config.flags,
    }


def estimate_flow(reference: np.ndarray, target: np.ndarray, config: CausalNetPreprocessConfig) -> np.ndarray:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    raw = cv2.calcOpticalFlowFarneback(
        reference_gray,
        target_gray,
        None,
        **_flow_config(config),
    ).astype(np.float32)
    resized = cv2.resize(raw, (config.flow_width, config.flow_height), interpolation=cv2.INTER_AREA)
    resized[..., 0] *= config.flow_width / float(reference.shape[1])
    resized[..., 1] *= config.flow_height / float(reference.shape[0])
    if not np.isfinite(resized).all():
        raise ValueError("Optical flow contains NaN or Inf")
    return resized


def _normalize_flow(flow: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    if not np.isfinite(flow).all():
        raise ValueError("Optical flow contains NaN or Inf")
    magnitude = np.linalg.norm(flow, axis=2)
    scale = max(float(np.quantile(magnitude, quantile)), np.finfo(np.float32).eps)
    normalized = np.stack(
        (
            np.clip(flow[..., 0] / scale, -1.0, 1.0),
            np.clip(flow[..., 1] / scale, -1.0, 1.0),
            np.clip(optical_strain(flow) / scale, 0.0, 1.0),
        ),
        axis=2,
    ).astype(np.float32)
    return normalized, scale


def direction_map(flow: np.ndarray, magnitude_scale: float) -> np.ndarray:
    """Encode angle as HSV hue and magnitude as saturation, then return RGB [0, 1]."""
    if not np.isfinite(flow).all() or not np.isfinite(magnitude_scale):
        raise ValueError("Direction-map input contains NaN or Inf")
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hue = ((angle + np.pi) / (2.0 * np.pi) * 179.0).astype(np.uint8)
    magnitude = np.linalg.norm(flow, axis=2)
    saturation = np.clip(magnitude / max(magnitude_scale, np.finfo(np.float32).eps), 0.0, 1.0)
    hsv = np.stack(
        (hue, np.round(saturation * 255.0).astype(np.uint8), np.full_like(hue, 255)),
        axis=2,
    )
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb


def _crop_with_padding(image: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    row, col = center
    radius = size // 2
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
    row += radius
    col += radius
    crop = padded[row - radius : row + radius, col - radius : col + radius]
    if crop.shape[:2] != (size, size):
        raise RuntimeError(f"Expected crop {(size, size)}, got {crop.shape}")
    return crop


def source_compatible_roi(image: np.ndarray, config: CausalNetPreprocessConfig) -> np.ndarray:
    """Reproduce the upstream left-eye/lip + right-eye/lip 2x2 collage."""
    parts = [_crop_with_padding(image, center, config.crop_size) for center in UPSTREAM_ROI_CENTERS]
    top = cv2.hconcat([parts[0], parts[1]])
    bottom = cv2.hconcat([parts[3], parts[4]])
    collage = cv2.vconcat([top, bottom])
    return cv2.resize(collage, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)


def spatial_projection(image: np.ndarray, config: CausalNetPreprocessConfig) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    if config.spatial_variant == "source_compatible_roi":
        return source_compatible_roi(image, config)
    return cv2.resize(image, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)


def _pair_input(reference: np.ndarray, target: np.ndarray, config: CausalNetPreprocessConfig) -> tuple[np.ndarray, np.ndarray, float]:
    flow = estimate_flow(reference, target, config)
    normalized, scale = _normalize_flow(flow, config.magnitude_quantile)
    direction = direction_map(flow, scale)
    return spatial_projection(normalized, config), spatial_projection(direction, config), scale


def build_four_route_input(
    record: Mapping[str, Any],
    config: CausalNetPreprocessConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or CausalNetPreprocessConfig()
    frame_paths = list_sequence_frames(Path(str(record["frame_dir"])))
    onset = int(record["onset_index"])
    apex = int(record["apex_index"])
    offset = int(record["offset_index"])
    if not 0 <= onset < apex < offset < len(frame_paths):
        raise ValueError(f"Invalid key-frame order for {record['sample_id']}")
    transform = _alignment_transform(record)
    aligned = {
        index: _aligned_frame(frame_paths[index], transform)
        for index in (onset, apex, offset)
    }
    oa_flow, oa_direction, oa_scale = _pair_input(aligned[onset], aligned[apex], config)
    ao_flow, ao_direction, ao_scale = _pair_input(aligned[apex], aligned[offset], config)
    inputs = np.stack((oa_flow, ao_flow, oa_direction, ao_direction), axis=0)
    inputs = np.transpose(inputs, (0, 3, 1, 2)).astype(np.float32)
    if inputs.shape != (4, 3, config.image_size, config.image_size):
        raise RuntimeError(f"Unexpected CausalNet input shape: {inputs.shape}")
    if not np.isfinite(inputs).all():
        raise ValueError(f"CausalNet input contains NaN or Inf: {record['sample_id']}")
    metadata = {
        "task_id": "FLOW-ME-004",
        "schema_version": CAUSALNET_PREPROCESS_SCHEMA_VERSION,
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "source_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
        "sample_id": str(record["sample_id"]),
        "source_dataset": str(record["source_dataset"]),
        "sample_role": str(record.get("sample_role", "classification")),
        "subject_id": str(record["subject_id"]),
        "sequence_id": str(record["sequence_id"]),
        "label": int(record["label"]),
        "label_name": str(record.get("label_name", LABEL_NAMES[int(record["label"])])),
        "frame_dir": str(record["frame_dir"]),
        "frame_annotation_source": str(record.get("frame_annotation_source", "estimated")),
        "apex_method": str(record.get("apex_method", "dc_rois")),
        "onset_index": onset,
        "apex_index": apex,
        "offset_index": offset,
        "onset_frame": str(record.get("onset_frame", frame_paths[onset].name)),
        "apex_frame": str(record.get("apex_frame", frame_paths[apex].name)),
        "offset_frame": str(record.get("offset_frame", frame_paths[offset].name)),
        "alignment_transform": transform.tolist(),
        "spatial_variant": config.spatial_variant,
        "route_order": [
            "onset_to_apex_flow",
            "apex_to_offset_flow",
            "onset_to_apex_direction",
            "apex_to_offset_direction",
        ],
        "flow_estimator": "opencv_farneback",
        "flow_estimator_parameters": _flow_config(config),
        "flow_channels": ["u", "v", "optical_strain"],
        "flow_channel_definition": {
            "channel_0": "u / per_pair_magnitude_quantile_scale, clipped to [-1,1]",
            "channel_1": "v / per_pair_magnitude_quantile_scale, clipped to [-1,1]",
            "channel_2": "optical_strain / per_pair_magnitude_quantile_scale, clipped to [0,1]",
        },
        "direction_map_definition": {
            "angle_to_hue": "(atan2(v,u)+pi)/(2*pi)*179",
            "magnitude_to_saturation": "clip(norm(u,v)/per_pair_quantile_scale,0,1)*255",
            "value": 255,
            "opencv_conversion": "HSV uint8 -> RGB float32",
            "output_range": [0.0, 1.0],
        },
        "flow_normalization": "per_pair_quantile_clip",
        "flow_normalization_parameters": {
            "onset_to_apex_magnitude_scale": oa_scale,
            "apex_to_offset_magnitude_scale": ao_scale,
            "quantile": config.magnitude_quantile,
        },
        "input_shape": list(inputs.shape),
        "preprocess_config_sha256": config.fingerprint(),
        "preprocessing_config_hash": config.fingerprint(),
        "preprocess_config": asdict(config),
        "code_hash": None,
        "source_manifest_hash": None,
        "quality_status": "pass",
        "deployment_eligible": False,
        "evaluation_scope": "offline_training_candidate",
    }
    return inputs, metadata


def directory_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix()) if path.exists() else []
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(item)))
    return {"path": path.resolve().as_posix(), "file_count": len(files), "sha256": digest.hexdigest()}


def write_artifact(inputs: np.ndarray, metadata: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    np.savez_compressed(path, inputs=inputs, label=np.asarray(payload["label"], dtype=np.int64), metadata_json=np.asarray(json.dumps(payload, sort_keys=True)))
    result = dict(payload)
    result["artifact_path"] = path.resolve().as_posix()
    result["artifact_sha256"] = sha256_file(path)
    return result


def _as_json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record, ensure_ascii=False))


def audit_artifacts(records: Sequence[Mapping[str, Any]], *, expected_variant: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if sample_id in seen:
            issues.append({"code": "duplicate_sample_id", "sample_id": sample_id})
        seen.add(sample_id)
        try:
            with np.load(Path(str(record["artifact_path"])), allow_pickle=False) as artifact:
                inputs = np.asarray(artifact["inputs"])
                label = int(np.asarray(artifact["label"]).item())
                embedded = json.loads(str(np.asarray(artifact["metadata_json"]).item()))
            if inputs.shape != (4, 3, 28, 28):
                issues.append({"code": "invalid_shape", "sample_id": sample_id, "shape": list(inputs.shape)})
            if not np.isfinite(inputs).all():
                issues.append({"code": "non_finite", "sample_id": sample_id})
            if label != int(record["label"]):
                issues.append({"code": "label_mismatch", "sample_id": sample_id})
            if embedded.get("spatial_variant") != expected_variant:
                issues.append({"code": "variant_mismatch", "sample_id": sample_id})
            if embedded.get("route_order") != [
                "onset_to_apex_flow", "apex_to_offset_flow", "onset_to_apex_direction", "apex_to_offset_direction"
            ]:
                issues.append({"code": "route_order_mismatch", "sample_id": sample_id})
            if not int(embedded["onset_index"]) < int(embedded["apex_index"]) < int(embedded["offset_index"]):
                issues.append({"code": "keyframe_order", "sample_id": sample_id})
        except Exception as exc:
            issues.append({"code": "artifact_read_error", "sample_id": sample_id, "error": repr(exc)})
    return {"status": "pass" if not issues else "fail", "issue_count": len(issues), "issues": issues, "record_count": len(records)}
