from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .optical_flow import read_image
from .paper_preprocess import (
    FACE_ALIGNMENT_VERSION,
    PAPER_FLOW_SCHEMA_VERSION,
    PaperPreprocessConfig,
    apply_preprocessing_variant,
    create_flow_estimator,
    list_sequence_frames,
    optical_strain,
    resize_vector_flow,
)


DSTM_TEMPORAL_SCHEMA_VERSION = "dstm_temporal_flow_sequence_v2"
DSTM_TEMPORAL_FLOW_VERSION = "adjacent_farneback_u_v_optical_strain_onset_apex_v2"


@dataclass(frozen=True)
class DSTMTemporalConfig:
    """Configuration for the thesis onset-to-apex temporal artifact.

    The thesis does not publish the sequence-generation implementation, but
    section 4.2.2 fixes the interval as onset to apex. Each item is the flow
    from frame t to frame t+1 within that interval.
    """

    flow_width: int = 32
    flow_height: int = 32
    preprocess_variant: str = "combined"
    flow_estimator: str = "farneback"
    normalization_quantile: float = 0.995
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    sequence_start: str = "onset"
    sequence_end: str = "apex"
    pairing: str = "adjacent_frame"

    def __post_init__(self) -> None:
        if (self.flow_width, self.flow_height) != (32, 32):
            raise ValueError("DSTM temporal flow is frozen at 32x32")
        if self.preprocess_variant not in {"base", "denoise", "illumination", "combined"}:
            raise ValueError(f"Unsupported preprocessing variant: {self.preprocess_variant}")
        if self.flow_estimator not in {"farneback", "tvl1"}:
            raise ValueError("flow_estimator must be farneback or tvl1")
        if self.sequence_start != "onset" or self.sequence_end != "apex":
            raise ValueError("The DSTM primary artifact must cover onset to apex")
        if self.pairing != "adjacent_frame":
            raise ValueError("Only adjacent-frame temporal pairing is supported")

    def paper_config(self) -> PaperPreprocessConfig:
        return PaperPreprocessConfig(
            preprocess_variant=self.preprocess_variant,
            flow_estimator=self.flow_estimator,
            normalization_quantile=self.normalization_quantile,
            flow_width=self.flow_width,
            flow_height=self.flow_height,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = {
            "schema_version": DSTM_TEMPORAL_SCHEMA_VERSION,
            "flow_schema_version": DSTM_TEMPORAL_FLOW_VERSION,
            "paper_flow_schema_version": PAPER_FLOW_SCHEMA_VERSION,
            "face_alignment_version": FACE_ALIGNMENT_VERSION,
            "config": self.as_dict(),
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _alignment_transform(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("alignment_transform")
    if value is None:
        return np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    transform = np.asarray(value, dtype=np.float32)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid alignment transform for {record['sample_id']}")
    return transform


def _aligned_frames(
    frame_paths: Sequence[Path],
    record: Mapping[str, Any],
    config: DSTMTemporalConfig,
) -> tuple[np.ndarray, ...]:
    transform = _alignment_transform(record)
    paper_config = config.paper_config()
    aligned: list[np.ndarray] = []
    for path in frame_paths:
        frame = read_image(path)
        frame = cv2.warpAffine(
            frame,
            transform,
            (paper_config.aligned_width, paper_config.aligned_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        aligned.append(apply_preprocessing_variant(frame, paper_config))
    return tuple(aligned)


def _normalize_sequence(
    raw_flows: Sequence[np.ndarray],
    raw_strains: Sequence[np.ndarray],
    quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if not raw_flows or len(raw_flows) != len(raw_strains):
        raise ValueError("Temporal flow sequence is empty or mismatched")
    flow_values = np.concatenate([flow.reshape(-1, 2) for flow in raw_flows], axis=0)
    strain_values = np.concatenate([strain.reshape(-1) for strain in raw_strains])
    epsilon = float(np.finfo(np.float32).eps)
    vector_scale = max(float(np.quantile(np.abs(flow_values), quantile)), epsilon)
    strain_scale = max(float(np.quantile(strain_values, quantile)), epsilon)
    normalized: list[np.ndarray] = []
    for flow, strain in zip(raw_flows, raw_strains, strict=True):
        normalized.append(
            np.stack(
                (
                    np.clip(flow[..., 0] / vector_scale, -1.0, 1.0),
                    np.clip(flow[..., 1] / vector_scale, -1.0, 1.0),
                    np.clip(strain / strain_scale, 0.0, 1.0),
                ),
                axis=2,
            ).astype(np.float32)
        )
    return np.stack(normalized, axis=0), {
        "vector_scale": vector_scale,
        "strain_scale": strain_scale,
        "quantile": float(quantile),
    }


def build_dstm_temporal_sequence(
    record: Mapping[str, Any],
    *,
    config: DSTMTemporalConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or DSTMTemporalConfig()
    all_frame_paths = list_sequence_frames(Path(str(record["frame_dir"])))
    if len(all_frame_paths) < 2:
        raise ValueError(f"{record['sample_id']} requires at least two frames")
    onset_index = int(record.get("onset_index", 0) or 0)
    apex_index = int(record.get("apex_index", -1) or -1)
    offset_index = int(
        record.get("offset_index", len(all_frame_paths) - 1)
        or len(all_frame_paths) - 1
    )
    if not 0 <= onset_index < apex_index < len(all_frame_paths):
        raise ValueError(
            f"Invalid onset/apex interval for {record['sample_id']}: "
            f"{onset_index}/{apex_index}/{len(all_frame_paths)}"
        )
    frame_paths = all_frame_paths[onset_index : apex_index + 1]
    aligned = _aligned_frames(frame_paths, record, config)
    paper_config = config.paper_config()
    estimator = create_flow_estimator(paper_config)
    raw_flows: list[np.ndarray] = []
    raw_strains: list[np.ndarray] = []
    frame_pairs: list[tuple[int, int]] = []
    for local_index in range(len(aligned) - 1):
        reference = cv2.cvtColor(aligned[local_index], cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(aligned[local_index + 1], cv2.COLOR_BGR2GRAY)
        raw = estimator.estimate(reference, target)
        resized = resize_vector_flow(raw, config.flow_width, config.flow_height)
        raw_flows.append(resized)
        raw_strains.append(optical_strain(resized))
        source_index = onset_index + local_index
        frame_pairs.append((source_index, source_index + 1))
    normalized_hwc, normalization = _normalize_sequence(
        raw_flows, raw_strains, config.normalization_quantile
    )
    sequence = np.transpose(normalized_hwc, (0, 3, 1, 2)).astype(np.float32)
    if sequence.ndim != 4 or sequence.shape[1:] != (3, 32, 32):
        raise RuntimeError(f"Unexpected DSTM flow sequence shape: {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise RuntimeError(f"DSTM temporal sequence contains non-finite values: {record['sample_id']}")
    if not np.any(np.abs(sequence) > 1e-7):
        raise RuntimeError(f"DSTM temporal sequence is all zero: {record['sample_id']}")
    metadata = {
        "task_id": "MODEL-ME-006",
        "schema_version": DSTM_TEMPORAL_SCHEMA_VERSION,
        "flow_schema_version": DSTM_TEMPORAL_FLOW_VERSION,
        "source_paper_flow_schema_version": PAPER_FLOW_SCHEMA_VERSION,
        "sample_id": str(record["sample_id"]),
        "source_dataset": str(record["source_dataset"]),
        "subject_id": str(record["subject_id"]),
        "sequence_id": str(record["sequence_id"]),
        "frame_dir": str(record["frame_dir"]),
        "frame_count": len(all_frame_paths),
        "interval_frame_count": len(frame_paths),
        "sequence_length": int(sequence.shape[0]),
        "sequence_start": config.sequence_start,
        "sequence_end": config.sequence_end,
        "pairing": config.pairing,
        "frame_pairs": [[int(left), int(right)] for left, right in frame_pairs],
        "onset_index": onset_index,
        "apex_index": apex_index,
        "offset_index": offset_index,
        "flow_shape": list(sequence.shape[1:]),
        "flow_channels": "u_v_optical_strain",
        "flow_estimator": config.flow_estimator,
        "flow_estimator_version": estimator.version,
        "preprocess_variant": config.preprocess_variant,
        "face_alignment_version": FACE_ALIGNMENT_VERSION,
        "alignment_source": "FLOW-ME-002_fixed_onset_transform",
        "normalization": normalization,
        "config": config.as_dict(),
        "config_sha256": config.fingerprint(),
        "paper_method_compliance": "temporal_sequence_reimplementation",
        "paper_reproduction_claim": False,
        "deployment_eligible": False,
        "label": int(record["label"]),
        "label_name": str(record.get("label_name", "unknown")),
        "quality_status": str(record.get("quality_status", "unknown")),
        "apex_boundary_status": str(record.get("apex_boundary_status", "unknown")),
    }
    return sequence, metadata


def write_dstm_temporal_artifact(
    sequence: np.ndarray,
    metadata: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            flow_sequence=np.asarray(sequence, dtype=np.float32),
            frame_pairs=np.asarray(metadata["frame_pairs"], dtype=np.int64),
            label=np.asarray(int(metadata["label"]), dtype=np.int64),
        )
    temporary.replace(output_path)
    result = dict(metadata)
    result["artifact_path"] = output_path.resolve().as_posix()
    result["artifact_sha256"] = _sha256_file(output_path)
    result["artifact_bytes"] = output_path.stat().st_size
    return result


def audit_dstm_temporal_artifacts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: set[str] = set()
    lengths: list[int] = []
    nonzero_count = 0
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id in seen:
            issues.append({"sample_id": sample_id, "code": "duplicate_sample_id"})
        seen.add(sample_id)
        path = Path(str(record.get("artifact_path", "")))
        if not path.is_file():
            issues.append({"sample_id": sample_id, "code": "missing_artifact"})
            continue
        if _sha256_file(path) != record.get("artifact_sha256"):
            issues.append({"sample_id": sample_id, "code": "artifact_sha256_mismatch"})
        try:
            with np.load(path, allow_pickle=False) as artifact:
                required = {"flow_sequence", "frame_pairs", "label"}
                missing = required - set(artifact.files)
                if missing:
                    issues.append({"sample_id": sample_id, "code": "missing_arrays", "arrays": sorted(missing)})
                    continue
                sequence = np.asarray(artifact["flow_sequence"])
                pairs = np.asarray(artifact["frame_pairs"])
                if sequence.ndim != 4 or sequence.shape[1:] != (3, 32, 32):
                    issues.append({"sample_id": sample_id, "code": "shape_mismatch", "shape": list(sequence.shape)})
                if pairs.shape != (sequence.shape[0], 2):
                    issues.append({"sample_id": sample_id, "code": "frame_pair_shape_mismatch"})
                if not np.isfinite(sequence).all():
                    issues.append({"sample_id": sample_id, "code": "non_finite_sequence"})
                if not np.any(np.abs(sequence) > 1e-7):
                    issues.append({"sample_id": sample_id, "code": "all_zero_sequence"})
                onset_index = int(record.get("onset_index", 0))
                apex_index = int(record.get("apex_index", -1))
                if sequence.shape[0] != apex_index - onset_index:
                    issues.append({"sample_id": sample_id, "code": "incomplete_onset_apex_sequence"})
                expected_pairs = np.column_stack(
                    (
                        np.arange(onset_index, apex_index),
                        np.arange(onset_index + 1, apex_index + 1),
                    )
                )
                if not np.array_equal(pairs, expected_pairs):
                    issues.append({"sample_id": sample_id, "code": "non_adjacent_frame_pairs"})
                lengths.append(int(sequence.shape[0]))
                nonzero_count += int(np.any(np.abs(sequence) > 1e-7))
        except Exception as error:  # noqa: BLE001
            issues.append({"sample_id": sample_id, "code": "load_error", "error": repr(error)})
    return {
        "schema_version": DSTM_TEMPORAL_SCHEMA_VERSION,
        "task_id": "MODEL-ME-006",
        "status": "pass" if not issues else "failed",
        "sample_count": len(records),
        "unique_sample_count": len(seen),
        "nonzero_sequence_count": nonzero_count,
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "sequence_length_mean": float(np.mean(lengths)) if lengths else None,
        "issues": issues,
        "warnings": warnings,
    }
