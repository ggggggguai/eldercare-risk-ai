from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.face_tracking import (
    TemplateFaceLandmarker,
    canonical_landmarks_68,
    default_dlib_predictor_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.optical_flow import (
    OpticalFlowConfig,
    normalize_three_channel_flow,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.preprocess import (
    SequencePreprocessor,
    audit_preprocess_artifacts,
    build_model_tensors,
    write_preprocess_artifact,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def _synthetic_record(tmp_path: Path) -> dict[str, object]:
    frame_dir = tmp_path / "序列" / "s1_ne_01"
    for index, shift in enumerate((0, 2, 5), start=1):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.ellipse(image, (48, 50), (32, 39), 0, 0, 360, (110, 110, 110), -1)
        cv2.circle(image, (35, 41), 3, (220, 220, 220), -1)
        cv2.circle(image, (61, 41), 3, (220, 220, 220), -1)
        cv2.line(image, (38, 68), (58 + shift, 68), (230, 230, 230), 2)
        _write_image(frame_dir / f"frame{index}.bmp", image)
    return {
        "sample_id": "smic_hs__s01__s1_ne_01",
        "source_dataset": "smic_hs",
        "sample_role": "classification",
        "subject_id": "s01",
        "sequence_id": "s1_ne_01",
        "frame_dir": frame_dir.as_posix(),
        "label": 0,
        "label_name": "negative",
    }


def test_canonical_landmarks_match_dlib_schema() -> None:
    points = canonical_landmarks_68(128, 128)
    assert points.shape == (68, 2)
    assert np.all(points >= 0)
    assert np.all(points[:, 0] < 128)
    assert np.all(points[:, 1] < 128)


def test_versioned_dlib_predictor_asset_is_available() -> None:
    predictor = default_dlib_predictor_path()
    assert predictor.name == "shape_predictor_68_face_landmarks.dat"
    assert predictor.stat().st_size == 99_693_937


def test_three_channel_flow_has_fixed_shape_and_range() -> None:
    config = OpticalFlowConfig()
    flow = np.zeros((128, 128, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    flow[..., 1] = -2.0
    normalized = normalize_three_channel_flow(flow, config)

    assert normalized.shape == (32, 32, 3)
    assert normalized.dtype == np.float32
    assert np.all(np.isfinite(normalized))
    assert normalized[..., 0].min() >= -1.0
    assert normalized[..., 0].max() <= 1.0
    assert normalized[..., 1].min() >= -1.0
    assert normalized[..., 2].min() >= 0.0
    assert normalized[..., 2].max() <= 1.0


def test_model_tensor_contract_uses_43_valid_patches() -> None:
    flow = np.zeros((32, 32, 3), dtype=np.float32)
    landmarks = canonical_landmarks_68(32, 32)
    patches, region_ids, masks, keypoints = build_model_tensors(flow, landmarks)

    assert patches.shape == (72, 147)
    assert region_ids.shape == (72,)
    assert masks.shape == (72,)
    assert keypoints.shape == (72, 2)
    assert masks.sum() == 43
    assert np.all(keypoints[masks == 0] == 0)


def test_sequence_preprocess_is_deterministic_and_writes_npz(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    processor = SequencePreprocessor(TemplateFaceLandmarker(), OpticalFlowConfig())

    first = processor.process(record)
    second = processor.process(record)

    assert first.metadata["quality_status"] == "degraded"
    assert first.metadata["landmark_asset_sha256"] is None
    assert first.metadata["frame_annotation_source"] == "estimated"
    assert first.metadata["apex_index"] in (1, 2)
    assert first.metadata["patch_shape"] == [72, 147]
    assert np.array_equal(first.patches, second.patches)
    assert np.array_equal(first.keypoints, second.keypoints)
    assert np.array_equal(first.flow, second.flow)

    output = tmp_path / "artifact.npz"
    metadata = write_preprocess_artifact(first, output)
    assert output.is_file()
    assert len(metadata["artifact_sha256"]) == 64
    with np.load(output) as artifact:
        assert artifact["patches"].shape == (72, 147)
        assert artifact["flow"].shape == (32, 32, 3)

    audit = audit_preprocess_artifacts([metadata])
    assert audit["status"] == "pass"
    assert audit["artifact_count"] == 1
    assert audit["issue_count"] == 0
