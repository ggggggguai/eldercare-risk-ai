from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.microexpression_spotter import (
    MicroExpressionWindowSpotter,
    SpotterFeatureConfig,
    SpotterFeatureDataset,
    extract_window_features,
    feature_schema,
    list_frame_paths,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.spotting_evaluation import (
    binary_metrics,
    build_spotting_splits,
    select_threshold,
    validate_spotting_splits,
)


def _write_frame(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    encoded.tofile(str(path))


def test_window_features_are_fixed_shape_and_finite(tmp_path: Path) -> None:
    paths = []
    for index in range(12):
        image = np.zeros((80, 72), dtype=np.uint8)
        cv2.rectangle(image, (12 + index, 24), (30 + index, 42), 180, -1)
        path = tmp_path / f"reg_image{100 + index}.bmp"
        _write_frame(path, image)
        paths.append(path)
    config = SpotterFeatureConfig(image_size=32, temporal_bins=4, spatial_grid=4)
    features = extract_window_features(paths, config)
    assert features.shape == (84,)
    assert np.isfinite(features).all()
    assert np.any(features > 0)
    assert feature_schema(config)["uses_frame_count_as_feature"] is False


def test_frame_paths_use_numeric_order(tmp_path: Path) -> None:
    for name in ("reg_image10.bmp", "reg_image2.bmp", "reg_image1.bmp"):
        _write_frame(tmp_path / name, np.zeros((20, 20), dtype=np.uint8))
    assert [path.name for path in list_frame_paths(tmp_path)] == [
        "reg_image1.bmp",
        "reg_image2.bmp",
        "reg_image10.bmp",
    ]


def _records(tmp_path: Path) -> list[dict[str, object]]:
    rows = []
    for subject_index in range(1, 7):
        subject = f"s{subject_index:02d}"
        for label in (0, 1):
            path = tmp_path / f"{subject}_{label}.npz"
            np.savez_compressed(path, features=np.full(10, subject_index + label, dtype=np.float32))
            rows.append(
                {
                    "sample_id": f"{subject}_{label}",
                    "subject_id": subject,
                    "source_dataset": "smic_hs",
                    "label": label,
                    "feature_path": str(path),
                }
            )
    return rows


def test_spotting_splits_are_subject_isolated(tmp_path: Path) -> None:
    split = build_spotting_splits(_records(tmp_path), validation_subject_count=2)
    validate_spotting_splits(split)
    assert len(split["folds"]) == 6
    assert {fold["test_subject"] for fold in split["folds"]} == {
        f"s{index:02d}" for index in range(1, 7)
    }


def test_spotting_split_rejects_test_selection(tmp_path: Path) -> None:
    split = build_spotting_splits(_records(tmp_path), validation_subject_count=2)
    broken = deepcopy(split)
    broken["folds"][0]["selection_authority"]["test_used_for_selection"] = True
    with pytest.raises(ValueError, match="selection leakage"):
        validate_spotting_splits(broken)


def test_validation_threshold_is_selected_without_test_data() -> None:
    selected = select_threshold([0, 0, 1, 1], [0.10, 0.35, 0.65, 0.90])
    assert 0.35 < selected["threshold"] <= 0.65
    assert selected["metrics"]["macro_f1"] == 1.0
    metrics = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert metrics["balanced_accuracy"] == 1.0


def test_spotter_dataset_and_model_contract(tmp_path: Path) -> None:
    dataset = SpotterFeatureDataset(_records(tmp_path))
    model = MicroExpressionWindowSpotter(input_dim=10)
    logits = model(dataset.features)
    assert logits.shape == (12,)
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="Expected"):
        model(dataset.features[0])
