from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.apex_spotting import (
    DCRoIsConfig,
    build_roi_boxes,
    divide_and_conquer_local_peak,
    spot_apex_dc_rois,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.face_tracking import (
    TemplateFaceLandmarker,
    canonical_landmarks_68,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_preprocess import (
    ORDERED_LANDMARK_INDICES,
    ORDERED_REGION_LANDMARKS,
    REGION_OFFSETS,
    PaperPreprocessConfig,
    PaperSequencePreprocessor,
    TVL1FlowEstimator,
    align_face_sequence,
    apply_preprocessing_variant,
    audit_paper_artifacts,
    build_paper_patch_tensors,
    normalize_flow_channels,
    optical_strain,
    write_paper_artifact,
    write_paper_visualization,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def _face_frame(size: int, mouth_shift: int = 0) -> np.ndarray:
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    cv2.ellipse(
        image,
        (size // 2, size // 2),
        (int(size * 0.34), int(size * 0.43)),
        0,
        0,
        360,
        (135, 135, 135),
        -1,
    )
    cv2.circle(image, (int(size * 0.37), int(size * 0.40)), 4, (20, 20, 20), -1)
    cv2.circle(image, (int(size * 0.63), int(size * 0.40)), 4, (20, 20, 20), -1)
    cv2.line(
        image,
        (int(size * 0.39), int(size * 0.73)),
        (int(size * 0.61) + mouth_shift, int(size * 0.73)),
        (230, 230, 230),
        3,
    )
    return image


def _synthetic_record(tmp_path: Path) -> dict[str, object]:
    frame_dir = tmp_path / "sequence" / "s1_ne_01"
    for index, shift in enumerate((0, 2, 8, 3), start=1):
        _write_image(frame_dir / f"frame{index}.bmp", _face_frame(112, shift))
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


def test_divide_and_conquer_peak_search_returns_local_maximum() -> None:
    index, trace = divide_and_conquer_local_peak([0.0, 0.1, 0.8, 0.4, 0.2])
    assert index == 2
    assert trace[0] == (1, 4, 2)


def test_dc_rois_peak_search_can_exclude_offset_frame() -> None:
    index, _ = divide_and_conquer_local_peak(
        [0.0, 0.1, 0.3, 0.8], minimum_index=1, maximum_index=2
    )
    assert index == 2


def test_dc_rois_uses_three_landmark_derived_regions() -> None:
    landmarks = canonical_landmarks_68(256, 256)
    boxes = build_roi_boxes(landmarks, (256, 256), DCRoIsConfig())
    assert boxes.shape == (3, 4)
    assert np.all(boxes[:, 2] > boxes[:, 0])
    assert np.all(boxes[:, 3] > boxes[:, 1])

    rng = np.random.default_rng(12)
    onset = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    frames = [onset.copy() for _ in range(5)]
    left, top, right, bottom = boxes[2]
    frames[1][top:bottom, left:right] = np.roll(
        onset[top:bottom, left:right], 1, axis=1
    )
    frames[2][top:bottom, left:right] = 255 - onset[top:bottom, left:right]
    frames[3][top:bottom, left:right] = np.roll(
        onset[top:bottom, left:right], 1, axis=0
    )
    result = spot_apex_dc_rois(frames, landmarks)
    assert result.apex_index == 2
    assert result.frame_scores.shape == (5,)
    assert result.roi_scores.shape == (5, 3)
    assert result.frame_scores[0] == pytest.approx(0.0)


def test_similarity_alignment_outputs_fixed_256_contract() -> None:
    frame = _face_frame(112)
    landmarks = canonical_landmarks_68(112, 112)
    result = align_face_sequence(
        [frame, frame.copy()], landmarks, PaperPreprocessConfig()
    )
    assert len(result.frames) == 2
    assert result.frames[0].shape == (256, 256, 3)
    assert result.landmarks.shape == (68, 2)
    assert result.transform.shape == (2, 3)
    assert np.all(result.landmarks >= 0)
    assert np.all(result.landmarks <= 255)


def test_preprocessing_variants_are_configurable_and_deterministic() -> None:
    image = _face_frame(112)
    outputs = {
        variant: apply_preprocessing_variant(
            image, PaperPreprocessConfig(preprocess_variant=variant)
        )
        for variant in ("base", "denoise", "illumination", "combined")
    }
    assert all(output.shape == image.shape for output in outputs.values())
    assert np.array_equal(outputs["base"], image)
    assert not np.array_equal(outputs["denoise"], outputs["base"])
    assert not np.array_equal(outputs["illumination"], outputs["base"])
    assert not np.array_equal(outputs["combined"], outputs["denoise"])
    repeated = apply_preprocessing_variant(
        image, PaperPreprocessConfig(preprocess_variant="combined")
    )
    assert np.array_equal(outputs["combined"], repeated)


def test_optical_strain_matches_thesis_equation_for_affine_flow() -> None:
    y, x = np.mgrid[0:17, 0:19].astype(np.float32)
    a, b, c, d = 0.25, -0.12, 0.08, 0.31
    flow = np.stack((a * x + b * y, c * x + d * y), axis=2)
    actual = optical_strain(flow)
    expected = np.sqrt(a**2 + d**2 + 0.5 * (b + c) ** 2)
    assert actual.shape == (17, 19)
    assert np.allclose(actual, expected, atol=1e-6)


def test_normalized_optical_strain_is_non_negative_and_bounded() -> None:
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[..., 0] = np.linspace(-3.0, 3.0, 32)[None, :]
    strain = optical_strain(flow)
    normalized, parameters = normalize_flow_channels(flow, strain, 0.995)
    assert normalized.shape == (32, 32, 3)
    assert normalized.dtype == np.float32
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0
    assert normalized[..., 2].min() >= 0.0
    assert parameters["vector_scale"] > 0
    assert parameters["strain_scale"] > 0


def test_paper_patch_contract_uses_43_ordered_5x5x3_patches() -> None:
    assert ORDERED_REGION_LANDMARKS["left_eyebrow"] == (21, 20, 19, 18, 17)
    assert ORDERED_REGION_LANDMARKS["right_eyebrow"] == (22, 23, 24, 25, 26)
    assert ORDERED_REGION_LANDMARKS["nose"][:4] == (27, 28, 29, 30)
    assert len(ORDERED_LANDMARK_INDICES) == len(set(ORDERED_LANDMARK_INDICES)) == 43

    flow = np.zeros((32, 32, 3), dtype=np.float32)
    landmarks = canonical_landmarks_68(32, 32)
    patches, region_ids, offsets, keypoints, ordered_indices = (
        build_paper_patch_tensors(flow, landmarks)
    )
    assert patches.shape == (43, 75)
    assert region_ids.shape == (43,)
    assert offsets.tolist() == list(REGION_OFFSETS)
    assert keypoints.shape == (43, 2)
    assert ordered_indices.tolist() == list(ORDERED_LANDMARK_INDICES)


def test_tvl1_candidate_interface_is_explicit() -> None:
    try:
        estimator = TVL1FlowEstimator()
    except RuntimeError as exc:
        assert "optflow contrib" in str(exc)
    else:
        first = np.zeros((16, 16), dtype=np.uint8)
        second = first.copy()
        assert estimator.estimate(first, second).shape == (16, 16, 2)


def test_paper_sequence_artifact_and_visualization_contract(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    processor = PaperSequencePreprocessor(
        TemplateFaceLandmarker(),
        PaperPreprocessConfig(preprocess_variant="combined"),
    )
    first = processor.process(record)
    second = processor.process(record)

    assert first.metadata["frame_annotation_source"] == "estimated_dc_rois"
    assert first.metadata["face_normalization"] == "aligned_256x256"
    assert first.metadata["flow_channels"] == "u_v_optical_strain"
    assert first.metadata["patch_shape"] == [43, 75]
    assert first.metadata["valid_patch_count"] == 43
    assert 1 <= first.metadata["apex_index"] < 4
    assert np.array_equal(first.patches, second.patches)
    assert np.array_equal(first.flow, second.flow)

    output = tmp_path / "paper_v2" / "artifact.npz"
    metadata = write_paper_artifact(first, output)
    visualization = tmp_path / "paper_v2" / "visualization.png"
    write_paper_visualization(first, visualization)
    assert output.is_file()
    assert visualization.is_file()
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["patches"].shape == (43, 75)
        assert artifact["flow"].shape == (32, 32, 3)
        assert artifact["apex_roi_scores"].shape == (4, 3)
    audit = audit_paper_artifacts([metadata])
    assert audit["status"] == "pass"
    assert audit["issue_count"] == 0
