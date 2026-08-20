from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .optical_flow import read_image
from .paper_preprocess import (
    PaperPreprocessConfig,
    apply_preprocessing_variant,
    build_paper_patch_tensors,
    create_flow_estimator,
    optical_strain,
    resize_vector_flow,
)
from .preprocess import list_sequence_frames


EVALUATION_VARIANT_SCHEMA_VERSION = "microexpression_eval_input_variants_v2"
PREPROCESS_CODES = {
    "P0": "base",
    "P1": "denoise",
    "P2": "illumination",
    "P3": "combined",
}
FLOW_ESTIMATORS = ("farneback", "tvl1")
THIRD_CHANNELS = ("optical_strain", "magnitude")


def evaluation_variant_id(
    preprocess_code: str,
    flow_estimator: str,
    third_channel: str,
) -> str:
    if preprocess_code not in PREPROCESS_CODES:
        raise ValueError(f"Unknown preprocessing code: {preprocess_code}")
    if flow_estimator not in FLOW_ESTIMATORS:
        raise ValueError(f"Unknown flow estimator: {flow_estimator}")
    if third_channel not in THIRD_CHANNELS:
        raise ValueError(f"Unknown third channel: {third_channel}")
    return f"{preprocess_code.lower()}_{flow_estimator}_{third_channel}"


def normalize_evaluation_flow(
    flow: np.ndarray,
    third_channel: np.ndarray,
    *,
    quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape HxWx2")
    if third_channel.shape != flow.shape[:2]:
        raise ValueError("third channel shape does not match flow")
    epsilon = float(np.finfo(np.float32).eps)
    vector_scale = max(
        float(np.quantile(np.abs(flow).reshape(-1), quantile)), epsilon
    )
    third_scale = max(
        float(np.quantile(np.abs(third_channel).reshape(-1), quantile)), epsilon
    )
    normalized = np.stack(
        (
            np.clip(flow[..., 0] / vector_scale, -1.0, 1.0),
            np.clip(flow[..., 1] / vector_scale, -1.0, 1.0),
            np.clip(third_channel / third_scale, 0.0, 1.0),
        ),
        axis=-1,
    ).astype(np.float32)
    return normalized, {
        "vector_scale": vector_scale,
        "third_channel_scale": third_scale,
        "quantile": float(quantile),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aligned_pair(
    sequence_record: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    frames = list_sequence_frames(Path(str(sequence_record["frame_dir"])))
    apex_index = int(frozen_record["apex_index"])
    if not 0 < apex_index < len(frames) - 1:
        raise ValueError(f"Invalid frozen apex index for {frozen_record['sample_id']}")
    transform = np.asarray(frozen_record["alignment_transform"], dtype=np.float32)
    if transform.shape != (2, 3):
        raise ValueError("Frozen alignment transform must have shape 2x3")

    def align(path: Path) -> np.ndarray:
        return cv2.warpAffine(
            read_image(path),
            transform,
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return align(frames[0]), align(frames[apex_index])


def build_evaluation_variants_for_sample(
    *,
    sequence_record: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    base_config: PaperPreprocessConfig,
    output_root: Path,
) -> list[dict[str, Any]]:
    onset_aligned, apex_aligned = _aligned_pair(sequence_record, frozen_record)
    frozen_artifact_path = Path(str(frozen_record["artifact_path"]))
    with np.load(frozen_artifact_path, allow_pickle=False) as frozen_artifact:
        landmarks = np.asarray(frozen_artifact["landmarks"], dtype=np.float32).copy()
        copied_arrays = {
            name: np.asarray(frozen_artifact[name]).copy()
            for name in (
                "apex_frame_scores",
                "apex_roi_scores",
                "apex_roi_boxes",
            )
        }
    records: list[dict[str, Any]] = []
    for preprocess_code, preprocess_variant in PREPROCESS_CODES.items():
        for flow_name in FLOW_ESTIMATORS:
            config = replace(
                base_config,
                preprocess_variant=preprocess_variant,
                flow_estimator=flow_name,
            )
            onset = apply_preprocessing_variant(onset_aligned, config)
            apex = apply_preprocessing_variant(apex_aligned, config)
            estimator = create_flow_estimator(config)
            raw_flow = estimator.estimate(
                cv2.cvtColor(onset, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(apex, cv2.COLOR_BGR2GRAY),
            )
            flow = resize_vector_flow(raw_flow, config.flow_width, config.flow_height)
            channel_values = {
                "optical_strain": optical_strain(flow),
                "magnitude": np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1])),
            }
            for channel_name, third_channel in channel_values.items():
                normalized, normalization = normalize_evaluation_flow(
                    flow, third_channel, quantile=config.normalization_quantile
                )
                patches, region_ids, offsets, keypoints, ordered = (
                    build_paper_patch_tensors(
                        normalized, landmarks, patch_size=config.patch_size
                    )
                )
                variant_id = evaluation_variant_id(
                    preprocess_code, flow_name, channel_name
                )
                artifact_path = (
                    output_root
                    / variant_id
                    / str(frozen_record["subject_id"])
                    / f"{frozen_record['sample_id']}.npz"
                ).resolve()
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = artifact_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        patches=patches,
                        region_ids=region_ids,
                        region_offsets=offsets,
                        keypoints=keypoints,
                        ordered_landmark_indices=ordered,
                        flow=normalized,
                        landmarks=landmarks,
                        label=np.asarray(frozen_record["label"], dtype=np.int32),
                        **copied_arrays,
                    )
                temporary.replace(artifact_path)
                record = dict(frozen_record)
                record.update(
                    {
                        "task_id": "EVAL-ME-002",
                        "evaluation_variant_schema_version": (
                            EVALUATION_VARIANT_SCHEMA_VERSION
                        ),
                        "evaluation_variant_id": variant_id,
                        "preprocess_code": preprocess_code,
                        "preprocess_variant": preprocess_variant,
                        "flow_estimator": flow_name,
                        "flow_estimator_version": estimator.version,
                        "third_channel": channel_name,
                        "flow_channels": f"u_v_{channel_name}",
                        "flow_normalization_parameters": normalization,
                        "artifact_path": artifact_path.as_posix(),
                        "artifact_sha256": _sha256_file(artifact_path),
                        "derived_from_frozen_artifact": frozen_artifact_path.resolve().as_posix(),
                        "derived_from_frozen_artifact_sha256": str(
                            frozen_record["artifact_sha256"]
                        ),
                        "apex_recomputed": False,
                        "alignment_recomputed": False,
                        "landmarks_recomputed": False,
                        "paper_reproduction_claim": False,
                    }
                )
                records.append(record)
    return records


def write_variant_manifests(
    records: Sequence[Mapping[str, Any]],
    *,
    report_root: Path,
    source_hashes: Mapping[str, str],
    config: PaperPreprocessConfig,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["evaluation_variant_id"]), []).append(record)
    report_root.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    for variant_id, variant_records in sorted(grouped.items()):
        manifest_path = report_root / f"artifact_manifest_{variant_id}_v2.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in sorted(
                    variant_records, key=lambda item: str(item["sample_id"])
                )
            ),
            encoding="utf-8",
        )
        variants[variant_id] = {
            "sample_count": len(variant_records),
            "manifest_path": manifest_path.resolve().as_posix(),
            "manifest_sha256": _sha256_file(manifest_path),
            "artifact_bytes": sum(
                Path(str(record["artifact_path"])).stat().st_size
                for record in variant_records
            ),
        }
    summary = {
        "schema_version": EVALUATION_VARIANT_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "pass" if all(
            value["sample_count"] == 164 for value in variants.values()
        ) else "fail",
        "variant_count": len(variants),
        "sample_count_per_variant": sorted(
            {value["sample_count"] for value in variants.values()}
        ),
        "source_hashes": dict(source_hashes),
        "base_config": asdict(config),
        "frozen_apex_alignment_landmarks": True,
        "variants": variants,
    }
    summary_path = report_root / "evaluation_input_variants_summary_v2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["summary_path"] = summary_path.resolve().as_posix()
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def audit_evaluation_variants(
    *,
    summary: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    frozen_by_id = {str(record["sample_id"]): record for record in frozen_records}
    expected_ids = set(frozen_by_id)
    artifact_count = 0
    p3_reference_exact_count = 0
    variant_results: dict[str, Any] = {}
    for variant_id, variant_summary in summary["variants"].items():
        manifest_path = Path(str(variant_summary["manifest_path"]))
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sample_ids = {str(record["sample_id"]) for record in records}
        if sample_ids != expected_ids:
            issues.append({"variant_id": variant_id, "code": "sample_id_mismatch"})
        variant_issue_count = 0
        for record in records:
            artifact_count += 1
            sample_id = str(record["sample_id"])
            artifact_path = Path(str(record["artifact_path"]))
            if not artifact_path.is_file():
                issues.append(
                    {
                        "variant_id": variant_id,
                        "sample_id": sample_id,
                        "code": "artifact_missing",
                    }
                )
                variant_issue_count += 1
                continue
            if _sha256_file(artifact_path) != record["artifact_sha256"]:
                issues.append(
                    {
                        "variant_id": variant_id,
                        "sample_id": sample_id,
                        "code": "artifact_hash_mismatch",
                    }
                )
                variant_issue_count += 1
            with np.load(artifact_path, allow_pickle=False) as artifact:
                expected_shapes = {
                    "patches": (43, 75),
                    "region_ids": (43,),
                    "region_offsets": (7,),
                    "keypoints": (43, 2),
                    "ordered_landmark_indices": (43,),
                    "flow": (32, 32, 3),
                    "landmarks": (68, 2),
                }
                for name, expected_shape in expected_shapes.items():
                    if name not in artifact or artifact[name].shape != expected_shape:
                        issues.append(
                            {
                                "variant_id": variant_id,
                                "sample_id": sample_id,
                                "code": f"invalid_{name}_shape",
                            }
                        )
                        variant_issue_count += 1
                    elif not np.isfinite(artifact[name]).all():
                        issues.append(
                            {
                                "variant_id": variant_id,
                                "sample_id": sample_id,
                                "code": f"non_finite_{name}",
                            }
                        )
                        variant_issue_count += 1
                if variant_id == "p3_farneback_optical_strain":
                    frozen_path = Path(str(frozen_by_id[sample_id]["artifact_path"]))
                    with np.load(frozen_path, allow_pickle=False) as frozen:
                        exact = all(
                            np.array_equal(artifact[name], frozen[name])
                            for name in (
                                "patches",
                                "region_ids",
                                "region_offsets",
                                "keypoints",
                                "ordered_landmark_indices",
                                "flow",
                                "landmarks",
                                "label",
                            )
                        )
                    if exact:
                        p3_reference_exact_count += 1
                    else:
                        issues.append(
                            {
                                "variant_id": variant_id,
                                "sample_id": sample_id,
                                "code": "p3_reference_numeric_drift",
                            }
                        )
                        variant_issue_count += 1
        variant_results[variant_id] = {
            "sample_count": len(records),
            "issue_count": variant_issue_count,
            "manifest_sha256_verified": (
                _sha256_file(manifest_path) == variant_summary["manifest_sha256"]
            ),
        }
    return {
        "schema_version": EVALUATION_VARIANT_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "pass" if not issues else "fail",
        "variant_count": len(variant_results),
        "artifact_count": artifact_count,
        "p3_reference_exact_count": p3_reference_exact_count,
        "p3_reference_expected_count": len(frozen_records),
        "issue_count": len(issues),
        "issues": issues,
        "variants": variant_results,
    }
