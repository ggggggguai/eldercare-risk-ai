from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.experts import (
    ActivityExpertBundle,
    activity,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.activity import (
    ACTIVITY_DATASET_IDS,
    ACTIVITY_FEATURE_COLUMNS,
    ACTIVITY_MASK_COLUMNS,
    EXPECTED_SPLIT_SHA256,
    ActivityExpertError,
    NumericFoldPreprocessor,
    ProbabilityCalibrator,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    load_activity_training_config,
    load_activity_training_inputs,
    participant_set_sha256,
    sample_weight_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "training" / "mood_social_activity_expert_v3_3_3.yaml"
)


def test_activity_expert_is_exported_from_experts_package() -> None:
    assert ActivityExpertBundle is activity.ActivityExpertBundle


def _empty_activity_frame(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{
                feature: pd.Series([pd.NA] * row_count, dtype="Float64")
                for feature in ACTIVITY_FEATURE_COLUMNS
            },
            **{
                mask: pd.Series([0] * row_count, dtype="Int8")
                for mask in ACTIVITY_MASK_COLUMNS
            },
        }
    )


def test_training_config_and_candidate_traversal_are_frozen() -> None:
    config = load_activity_training_config(CONFIG_PATH)

    first = deterministic_search_candidates(
        config.search_space,
        random_seed=config.random_seed,
        candidate_count=config.search_candidate_count,
    )
    second = deterministic_search_candidates(
        config.search_space,
        random_seed=config.random_seed,
        candidate_count=config.search_candidate_count,
    )

    assert config.run_id == "MH-20260731-001"
    assert config.split_sha256 == EXPECTED_SPLIT_SHA256
    assert config.dataset_ids == ACTIVITY_DATASET_IDS
    assert first == second
    assert len(first) == len({item["candidate_id"] for item in first}) == 32
    assert {
        name: set(item["params"][name] for item in first)
        for name in config.search_space
    } == {name: set(values) for name, values in config.search_space.items()}


def test_sample_weights_follow_all_four_frozen_equalities() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": [
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "b",
                "b",
            ],
            "global_participant_id": [
                "a::p1",
                "a::p1",
                "a::p2",
                "a::p3",
                "a::p3",
                "a::p3",
                "b::p1",
                "b::p2",
            ],
            "binary_target": [0, 0, 0, 1, 1, 1, 0, 0],
        }
    )

    weight = compute_training_sample_weights(frame)
    audit = sample_weight_audit(frame, weight)

    assert weight.sum() == pytest.approx(len(frame))
    assert audit["by_dataset"]["a"]["total"] == pytest.approx(
        audit["by_dataset"]["b"]["total"]
    )
    assert audit["by_dataset"]["a"]["by_class"]["0"] == pytest.approx(
        audit["by_dataset"]["a"]["by_class"]["1"]
    )
    assert weight[0] == pytest.approx(weight[1])
    assert weight[0] + weight[1] == pytest.approx(weight[2])
    assert weight[3] == pytest.approx(weight[4])
    assert weight[4] == pytest.approx(weight[5])
    assert weight[6] == pytest.approx(weight[7])


def test_fold_preprocessor_uses_only_training_values() -> None:
    train = _empty_activity_frame(3)
    volume = "activity.activity_volume_norm"
    volume_mask = f"feature_mask.{volume}"
    valid_days = "activity.valid_days"
    valid_days_mask = f"feature_mask.{valid_days}"
    amplitude = "activity.relative_amplitude"
    amplitude_mask = f"feature_mask.{amplitude}"
    train[volume] = pd.Series([0.2, 0.6, 0.8], dtype="Float64")
    train[volume_mask] = pd.Series([1, 1, 1], dtype="Int8")
    train[valid_days] = pd.Series([3.0, 5.0, 7.0], dtype="Float64")
    train[valid_days_mask] = pd.Series([1, 1, 1], dtype="Int8")
    train[amplitude] = pd.Series([0.5, 0.5, 0.5], dtype="Float64")
    train[amplitude_mask] = pd.Series([1, 1, 1], dtype="Int8")
    validation = _empty_activity_frame(1)
    validation[volume] = pd.Series([0.99], dtype="Float64")
    validation[volume_mask] = pd.Series([1], dtype="Int8")

    preprocessor = NumericFoldPreprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.selected_features == (volume, valid_days)
    assert preprocessor.medians == pytest.approx((0.6, 5.0))
    assert list(transformed.columns) == [volume, valid_days]
    assert transformed.iloc[0].tolist() == pytest.approx([0.99, 5.0])
    assert amplitude not in preprocessor.selected_features


def test_fold_preprocessor_rejects_value_mask_mismatch() -> None:
    frame = _empty_activity_frame(2)
    feature = "activity.activity_volume_norm"
    frame[feature] = pd.Series([0.2, 0.4], dtype="Float64")

    with pytest.raises(ActivityExpertError, match="value-mask relation"):
        NumericFoldPreprocessor.fit(frame)


def test_probability_calibration_selection_rule() -> None:
    raw = np.linspace(0.05, 0.95, 40)
    target = np.asarray([0] * 20 + [1] * 20)
    weight = np.ones(40)

    isotonic = ProbabilityCalibrator.fit(
        raw,
        target,
        weight,
        positive_participant_count=200,
        isotonic_min_positive_participants=200,
    )
    platt = ProbabilityCalibrator.fit(
        raw,
        target,
        weight,
        positive_participant_count=199,
        isotonic_min_positive_participants=200,
    )

    assert isotonic.method == "isotonic"
    assert platt.method == "platt"
    assert np.all((0 <= isotonic.predict(raw)) & (isotonic.predict(raw) <= 1))
    assert np.all((0 <= platt.predict(raw)) & (platt.predict(raw) <= 1))


def test_binary_metrics_include_frozen_outputs_and_equal_width_ece() -> None:
    result = binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        decision_threshold=0.5,
        ece_bin_count=2,
    )

    assert result["auprc"] == pytest.approx(1.0)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["brier_score"] == pytest.approx(0.085)
    assert result["ece"] == pytest.approx(0.25)
    assert len(result["calibration_curve"]) == 2


def test_activity_inputs_revalidate_all_five_bound_sources() -> None:
    inputs = load_activity_training_inputs(REPOSITORY_ROOT)

    assert inputs.split_manifest_sha256 == EXPECTED_SPLIT_SHA256
    assert tuple(inputs.frames) == ACTIVITY_DATASET_IDS
    assert {name: len(frame) for name, frame in inputs.frames.items()} == {
        "psyche_d": 10_866,
        "resilient": 73,
        "nhanes": 2_775,
    }
    assert len(inputs.assignments) == 6_884
    assert set(
        inputs.split_manifest["inputs"][index]["dataset_id"] for index in range(5)
    ) == {
        "psyche_d",
        "resilient",
        "nhanes",
        "shenzhen_elderly",
        "nhanes_ssq_2005_2008",
    }
    assert all(
        frame["activity.activity_volume_norm"].isna().all()
        for frame in inputs.frames.values()
    )


def test_real_inner_fold_ecdf_and_preprocessing_exclude_validation_participants() -> (
    None
):
    inputs = load_activity_training_inputs(REPOSITORY_ROOT)
    assignments = activity._assignment_lookup(inputs.assignments)
    outer_train = {
        key for key, value in assignments.items() if value["outer_fold"] != 0
    }
    prepared = activity._prepare_inner_fold(
        inputs,
        assignments,
        outer_fold=0,
        inner_fold=0,
        outer_train_keys=outer_train,
    )
    training_participants = set(
        prepared.train_table["global_participant_id"].astype(str)
    )
    validation_participants = set(
        prepared.validation_table["global_participant_id"].astype(str)
    )

    assert training_participants.isdisjoint(validation_participants)
    available_participants = training_participants | validation_participants
    assert available_participants < outer_train
    unavailable = outer_train - available_participants
    assert all(key.startswith("psyche_d::") for key in unavailable)
    assert (
        inputs.frames["psyche_d"]
        .loc[
            inputs.frames["psyche_d"]["global_participant_id"]
            .astype(str)
            .isin(unavailable),
            "x_source_mask",
        ]
        .eq(0)
        .all()
    )
    assert prepared.preprocessor.selected_features == (
        "activity.activity_volume_norm",
        "activity.relative_amplitude",
        "activity.interdaily_stability",
        "activity.intradaily_variability",
        "activity.activity_variability",
        "activity.valid_days",
    )
    inner_validation = {
        key
        for key in outer_train
        if assignments[key]["inner_validation_fold_by_outer_fold"]["0"] == 0
    }
    inner_training = outer_train - inner_validation
    for dataset_id in ACTIVITY_DATASET_IDS:
        expected = {key for key in inner_training if key.startswith(f"{dataset_id}::")}
        ecdf = prepared.ecdf_by_dataset[dataset_id]
        assert ecdf.training_participant_count == len(expected)
        assert ecdf.training_participant_sha256 == participant_set_sha256(expected)
        assert (
            not set(
                prepared.validation_table.loc[
                    prepared.validation_table["dataset_id"].eq(dataset_id),
                    "global_participant_id",
                ].astype(str)
            )
            & expected
        )
