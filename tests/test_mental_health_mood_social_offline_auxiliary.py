from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.offline_auxiliary import (
    DATASET_ORDER,
    GRADE_BANDS,
    MODEL_FAMILY_BY_TASK,
    MULTICLASS_DATASETS,
    OfflineAuxiliaryError,
    OfflineFoldPreprocessor,
    build_task_frame,
    dataset_spec,
    deterministic_candidates,
    load_offline_auxiliary_config,
    load_offline_auxiliary_inputs,
    materialize_features,
    multiclass_metrics,
    multiclass_sample_weights,
    phq9_grade,
    regression_metrics,
    regression_sample_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_offline_auxiliary_v3_3_3.yaml"


def test_config_freezes_dataset_task_and_search_boundaries() -> None:
    config = load_offline_auxiliary_config(CONFIG, repository_root=ROOT)
    assert tuple(config.payload["datasets"]) == DATASET_ORDER
    assert len(config.lightgbm_candidates) == 8
    assert len(config.elasticnet_candidates) == 20
    assert len(config.catboost_candidates) == 16
    assert config.payload["output"]["model_status"] == "offline_auxiliary_not_active"
    assert not config.payload["output"]["enter_fusion"]
    assert not config.payload["output"]["enter_api"]


def test_dataset_specs_use_only_frozen_schema_features() -> None:
    assert tuple(MODEL_FAMILY_BY_TASK) == (
        "psyche_d::regression",
        "psyche_d::multiclass",
        "resilient::regression",
        "nhanes::regression",
        "nhanes::multiclass",
        "shenzhen_elderly::regression",
        "nhanes_ssq_2005_2008::regression",
        "nhanes_ssq_2005_2008::multiclass",
    )
    for dataset_id in DATASET_ORDER:
        spec = dataset_spec(dataset_id)
        assert spec.input_features
        assert all(
            feature.startswith(
                ("activity.", "sleep.", "physiology.", "social_context.")
            )
            for feature in spec.input_features
        )
        assert not any(
            token in feature.lower()
            for feature in spec.input_features
            for token in ("phq", "target", "source__", "feature_mask")
        )
        assert spec.train_multiclass == (dataset_id in MULTICLASS_DATASETS)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, 0),
        (4, 0),
        (5, 1),
        (9, 1),
        (10, 2),
        (14, 2),
        (15, 3),
        (19, 3),
        (20, 4),
        (27, 4),
    ],
)
def test_phq9_grade_boundaries(score: int, expected: int) -> None:
    assert phq9_grade(score) == expected


def test_phq9_grade_rejects_out_of_range() -> None:
    with pytest.raises(OfflineAuxiliaryError):
        phq9_grade(-1)
    with pytest.raises(OfflineAuxiliaryError):
        phq9_grade(28)


def test_fold_preprocessor_fits_train_only_and_handles_unknown() -> None:
    train = pd.DataFrame(
        {
            "numeric": [1.0, 3.0, np.nan],
            "category": ["a", "b", None],
            "constant": [7.0, 7.0, 7.0],
        }
    )
    preprocessor = OfflineFoldPreprocessor.fit(
        train,
        input_features=("numeric", "category", "constant"),
        categorical_features=("category",),
        standardize_numeric=True,
    )
    assert preprocessor.selected_features == ("numeric", "category")
    assert preprocessor.numeric_medians["numeric"] == 2.0
    transformed = preprocessor.transform(
        pd.DataFrame({"numeric": [np.nan], "category": ["outside"], "constant": [9.0]})
    )
    assert transformed.loc[0, "numeric"] == 0.0
    assert transformed.loc[0, "category==__UNKNOWN__"] == 1.0
    assert transformed.loc[0, "category==__MISSING__"] == 0.0


def test_regression_weights_equalize_participants_and_windows() -> None:
    frame = pd.DataFrame({"global_participant_id": ["a", "a", "b"]})
    weight = regression_sample_weights(frame)
    assert weight.mean() == pytest.approx(1.0)
    assert weight[:2].sum() == pytest.approx(weight[2])


def test_multiclass_weights_equalize_classes_participants_and_windows() -> None:
    frame = pd.DataFrame(
        {
            "global_participant_id": ["a", "a", "b", "c"],
            "target_grade": [0, 0, 0, 1],
        }
    )
    weight = multiclass_sample_weights(frame)
    assert weight.mean() == pytest.approx(1.0)
    assert weight[:3].sum() == pytest.approx(weight[3])
    assert weight[:2].sum() == pytest.approx(weight[2])


def test_regression_metrics_report_normalized_and_score_units() -> None:
    metrics = regression_metrics([0.0, 1.0], [0.1, 0.8])
    assert metrics["mae_normalized"] == pytest.approx(0.15)
    assert metrics["rmse_normalized"] == pytest.approx(np.sqrt(0.025))
    assert metrics["mae_score"] == pytest.approx(4.05)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_multiclass_metrics_use_all_five_classes() -> None:
    target = np.arange(5, dtype="int64")
    probabilities = np.eye(5, dtype="float64")
    metrics = multiclass_metrics(target, probabilities)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["ordinal_mae"] == pytest.approx(0.0)
    assert len(metrics["per_class"]) == 5
    assert np.asarray(metrics["confusion_matrix"]).shape == (5, 5)


def test_deterministic_candidate_subset_is_stable_and_unique() -> None:
    grid = {"a": [1, 2], "b": [3, 4, 5]}
    first = deterministic_candidates(grid, candidate_count=4, seed=17, prefix="x")
    second = deterministic_candidates(grid, candidate_count=4, seed=17, prefix="x")
    assert first == second
    assert len({row["candidate_id"] for row in first}) == 4
    assert all(row["params"]["a"] in {1, 2} for row in first)


def test_real_inputs_match_split_and_target_bands() -> None:
    config = load_offline_auxiliary_config(CONFIG, repository_root=ROOT)
    inputs = load_offline_auxiliary_inputs(config)
    assert len(inputs.assignments) == 15361
    for dataset_id in DATASET_ORDER:
        frame = build_task_frame(inputs, dataset_id)
        assert frame["global_participant_id"].nunique() == int(
            inputs.input_bindings[dataset_id]["participant_count"]
        )
        assert set(frame["outer_fold"]) == set(range(5))
        assert frame["target_normalized"].between(0, 1).all()
        assert set(frame["target_grade"]).issubset(set(range(len(GRADE_BANDS))))


def test_fold_ecdf_and_masks_materialize_without_label_inputs() -> None:
    config = load_offline_auxiliary_config(CONFIG, repository_root=ROOT)
    inputs = load_offline_auxiliary_inputs(config)
    spec = dataset_spec("psyche_d")
    frame = build_task_frame(inputs, "psyche_d")
    train = frame[frame["outer_fold"].ne(0)]
    from elderly_monitoring.modules.mental_health.mood_social import (
        offline_auxiliary as module,
    )

    ecdf = module._fit_activity_ecdf(
        inputs,
        spec,
        set(train["global_participant_id"].astype(str)),
    )
    materialized = materialize_features(frame.head(20), spec, activity_ecdf=ecdf)
    assert tuple(materialized.columns) == spec.input_features
    assert "activity.activity_volume_norm" in materialized
    assert not any(column.startswith("feature_mask.") for column in materialized)
    assert not any("phq" in column.lower() for column in materialized)


def test_current_config_rejects_tampered_scale_input_policy(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "scale_items_are_model_inputs: false",
        "scale_items_are_model_inputs: true",
    )
    path = tmp_path / "tampered.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(OfflineAuxiliaryError, match="feature boundary"):
        load_offline_auxiliary_config(path, repository_root=ROOT)
