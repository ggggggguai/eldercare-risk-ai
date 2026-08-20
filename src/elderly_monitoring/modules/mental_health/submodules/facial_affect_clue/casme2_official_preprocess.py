from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_preprocess import (
    CAUSALNET_DIRECTION_SCHEMA_VERSION,
    CAUSALNET_FLOW_SCHEMA_VERSION,
    CAUSALNET_PREPROCESS_SCHEMA_VERSION,
    CAUSALNET_UPSTREAM_COMMIT,
    CAUSALNET_UPSTREAM_LICENSE,
    CAUSALNET_UPSTREAM_REPOSITORY,
    CausalNetPreprocessConfig,
    pair_input,
    sha256_file,
)
from .face_tracking import FaceLandmarker
from .optical_flow import read_image
from .paper_preprocess import (
    FACE_ALIGNMENT_VERSION,
    PaperPreprocessConfig,
    align_face_sequence,
)
from .preprocess import list_sequence_frames


CASME2_OFFICIAL_FLOW_SCHEMA_VERSION = "casme2_official_causalnet_flow_v1"
EXPECTED_ROUTE_ORDER = (
    "onset_to_apex_flow",
    "apex_to_offset_flow",
    "onset_to_apex_direction",
    "apex_to_offset_direction",
)
_FRAME_NUMBER = re.compile(r"(\d+)$")


def _frame_number(path: Path) -> int | None:
    match = _FRAME_NUMBER.search(path.stem)
    return int(match.group(1)) if match else None


def _key_frame_paths(record: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    raw = record.get("raw")
    if not isinstance(raw, Mapping) or not raw.get("available"):
        raise ValueError(f"{record.get('sample_id')} has no canonical RAW source")
    frame_dir = Path(str(raw["path"]))
    by_number: dict[int, Path] = {}
    for path in list_sequence_frames(frame_dir):
        number = _frame_number(path)
        if number is None:
            continue
        if number in by_number:
            raise ValueError(f"Duplicate frame number {number} in {frame_dir}")
        by_number[number] = path
    key_numbers = (
        int(record["onset_frame"]),
        int(record["apex_frame"]),
        int(record["offset_frame"]),
    )
    missing = [number for number in key_numbers if number not in by_number]
    if missing:
        raise FileNotFoundError(
            f"{record.get('sample_id')} is missing RAW key frames {missing}"
        )
    return tuple(by_number[number] for number in key_numbers)  # type: ignore[return-value]


def build_official_four_route_input(
    record: Mapping[str, Any],
    landmarker: FaceLandmarker,
    config: CausalNetPreprocessConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or CausalNetPreprocessConfig()
    if record.get("schema_version") != "casme2_official_manifest_v1":
        raise ValueError("FLOW-ME-003 requires the DATA-ME-004 official manifest")
    if record.get("flow_ready_with_official_apex") is not True:
        raise ValueError(f"{record.get('sample_id')} is not official-apex flow ready")
    if record.get("project_three_class_label") is not None:
        raise ValueError("FLOW-ME-003 must not assign project labels")
    onset_number = int(record["onset_frame"])
    apex_number = int(record["apex_frame"])
    offset_number = int(record["offset_frame"])
    if not onset_number < apex_number < offset_number:
        raise ValueError(f"Invalid official key-frame order for {record['sample_id']}")

    frame_paths = _key_frame_paths(record)
    frames = tuple(read_image(path) for path in frame_paths)
    onset_landmarks = landmarker.detect(frames[0])
    alignment = align_face_sequence(
        frames,
        onset_landmarks.points,
        PaperPreprocessConfig(preprocess_variant="base"),
    )
    oa_flow, oa_direction, oa_scale = pair_input(
        alignment.frames[0], alignment.frames[1], config
    )
    ao_flow, ao_direction, ao_scale = pair_input(
        alignment.frames[1], alignment.frames[2], config
    )
    inputs = np.stack(
        (oa_flow, ao_flow, oa_direction, ao_direction), axis=0
    )
    inputs = np.transpose(inputs, (0, 3, 1, 2)).astype(np.float32)
    if inputs.shape != (4, 3, 28, 28):
        raise RuntimeError(f"Unexpected CausalNet input shape: {inputs.shape}")
    if not np.isfinite(inputs).all():
        raise ValueError(f"Non-finite CausalNet input: {record['sample_id']}")

    metadata = {
        "task_id": "FLOW-ME-003",
        "schema_version": CASME2_OFFICIAL_FLOW_SCHEMA_VERSION,
        "causalnet_preprocess_schema_version": CAUSALNET_PREPROCESS_SCHEMA_VERSION,
        "flow_schema_version": CAUSALNET_FLOW_SCHEMA_VERSION,
        "direction_schema_version": CAUSALNET_DIRECTION_SCHEMA_VERSION,
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
        "sample_id": str(record["sample_id"]),
        "source_dataset": "casme2_official",
        "sample_role": "classification_candidate",
        "subject_id": str(record["subject_id"]),
        "sequence_id": str(record["sequence_id"]),
        "official_estimated_emotion": str(record["estimated_emotion"]),
        "official_objective_class": int(record["objective_class"]),
        "action_units": record.get("action_units"),
        "project_three_class_label": None,
        "stored_label": -1,
        "label_space": "unassigned_until_eval_crosswalk",
        "frame_source": "CASME2_RAW/CASME2-RAW",
        "frame_dir": str(record["raw"]["path"]),
        "frame_annotation_source": "official_coding_20140508",
        "apex_method": "official",
        "onset_frame_number": onset_number,
        "apex_frame_number": apex_number,
        "offset_frame_number": offset_number,
        "onset_frame": frame_paths[0].name,
        "apex_frame": frame_paths[1].name,
        "offset_frame": frame_paths[2].name,
        "onset_index": 0,
        "apex_index": 1,
        "offset_index": 2,
        "alignment_method": FACE_ALIGNMENT_VERSION,
        "alignment_transform": alignment.transform.tolist(),
        "landmark_source": onset_landmarks.source,
        "landmark_asset_sha256": getattr(landmarker, "predictor_sha256", None),
        "face_detection_source": onset_landmarks.detection_source,
        "source_face_box": list(onset_landmarks.face_box),
        "temporal_correspondence": "fixed_onset_transform",
        "spatial_variant": config.spatial_variant,
        "route_order": list(EXPECTED_ROUTE_ORDER),
        "flow_estimator": "opencv_farneback",
        "flow_channels": ["u", "v", "optical_strain"],
        "flow_normalization": "per_pair_quantile_clip",
        "flow_normalization_parameters": {
            "onset_to_apex_magnitude_scale": oa_scale,
            "apex_to_offset_magnitude_scale": ao_scale,
            "quantile": config.magnitude_quantile,
        },
        "direction_map_definition": {
            "angle_to_hue": "(atan2(v,u)+pi)/(2*pi)*179",
            "magnitude_to_saturation": "clip(norm(u,v)/pair_scale,0,1)*255",
            "value": 255,
            "opencv_conversion": "HSV uint8 -> RGB float32",
        },
        "input_shape": list(inputs.shape),
        "preprocess_config": asdict(config),
        "preprocess_config_sha256": config.fingerprint(),
        "preprocessing_config_hash": config.fingerprint(),
        "code_hash": None,
        "source_manifest_hash": None,
        "quality_status": "pass",
        "publication_restricted": bool(record["publication_restricted"]),
        "evaluation_scope": "offline_training_candidate",
        "deployment_eligible": False,
    }
    return inputs, metadata


def write_official_artifact(
    inputs: np.ndarray,
    metadata: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    np.savez_compressed(
        path,
        inputs=inputs,
        label=np.asarray(-1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return {
        **payload,
        "artifact_path": path.resolve().as_posix(),
        "artifact_sha256": sha256_file(path),
    }


def audit_official_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_sample_ids: Sequence[str],
    expected_variant: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected = set(expected_sample_ids)
    observed = {str(record.get("sample_id")) for record in records}
    if len(observed) != len(records):
        issues.append({"code": "duplicate_sample_id"})
    if observed != expected:
        issues.append(
            {
                "code": "coverage_mismatch",
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            }
        )
    for record in records:
        sample_id = str(record.get("sample_id"))
        try:
            with np.load(Path(str(record["artifact_path"])), allow_pickle=False) as item:
                inputs = np.asarray(item["inputs"])
                stored_label = int(np.asarray(item["label"]).item())
                embedded = json.loads(str(np.asarray(item["metadata_json"]).item()))
            if inputs.shape != (4, 3, 28, 28):
                issues.append({"code": "invalid_shape", "sample_id": sample_id})
            if not np.isfinite(inputs).all():
                issues.append({"code": "non_finite", "sample_id": sample_id})
            if stored_label != -1 or embedded.get("project_three_class_label") is not None:
                issues.append({"code": "premature_project_label", "sample_id": sample_id})
            if embedded.get("route_order") != list(EXPECTED_ROUTE_ORDER):
                issues.append({"code": "route_order_mismatch", "sample_id": sample_id})
            if embedded.get("spatial_variant") != expected_variant:
                issues.append({"code": "variant_mismatch", "sample_id": sample_id})
            if embedded.get("sample_id") != sample_id:
                issues.append({"code": "embedded_sample_id_mismatch", "sample_id": sample_id})
            if embedded.get("artifact_path") is not None:
                issues.append({"code": "recursive_artifact_path", "sample_id": sample_id})
            if embedded.get("frame_source") != "CASME2_RAW/CASME2-RAW":
                issues.append({"code": "non_canonical_frame_source", "sample_id": sample_id})
            if not (
                int(embedded["onset_frame_number"])
                < int(embedded["apex_frame_number"])
                < int(embedded["offset_frame_number"])
            ):
                issues.append({"code": "keyframe_order", "sample_id": sample_id})
            if sha256_file(Path(str(record["artifact_path"]))) != record.get(
                "artifact_sha256"
            ):
                issues.append({"code": "artifact_hash_mismatch", "sample_id": sample_id})
        except Exception as error:  # noqa: BLE001
            issues.append(
                {
                    "code": "artifact_read_error",
                    "sample_id": sample_id,
                    "error": repr(error),
                }
            )
    return {
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "expected_count": len(expected),
        "record_count": len(records),
    }
