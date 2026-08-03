from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.experts import (
    SleepExpertBundle,
)
from elderly_monitoring.modules.mental_health.mood_social.experts import (
    sleep,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.sleep import (
    EXPECTED_SPLIT_SHA256,
    SLEEP_DATASET_IDS,
    SLEEP_FEATURE_COLUMNS,
    SLEEP_MASK_COLUMNS,
    SleepExpertError,
    SleepNumericFoldPreprocessor,
    SleepProbabilityCalibrator,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    load_sleep_training_config,
    load_sleep_training_inputs,
    sample_weight_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "training" / "mood_social_sleep_expert_v3_3_3.yaml"
)


def _empty_sleep_frame(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{
                feature: pd.Series([pd.NA] * row_count, dtype="Float64")
                for feature in SLEEP_FEATURE_COLUMNS
            },
            **{
                mask: pd.Series([0] * row_count, dtype="Int8")
                for mask in SLEEP_MASK_COLUMNS
            },
        }
    )


def test_sleep_expert_is_exported_from_experts_package() -> None:
    assert SleepExpertBundle is sleep.SleepExpertBundle


def test_training_config_and_candidate_traversal_are_frozen() -> None:
    config = load_sleep_training_config(CONFIG_PATH)

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

    assert config.run_id == "MH-20260731-002"
    assert config.split_sha256 == EXPECTED_SPLIT_SHA256
    assert config.dataset_ids == SLEEP_DATASET_IDS
    assert first == second
    assert len(first) == len({item["candidate_id"] for item in first}) == 32
    assert {
        name: set(item["params"][name] for item in first)
        for name in config.search_space
    } == {name: set(values) for name, values in config.search_space.items()}


def test_sample_weights_follow_all_four_frozen_equalities() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "a", "a", "a", "a", "a", "b", "b"],
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
    train = _empty_sleep_frame(3)
    duration = "sleep.sleep_duration_norm"
    duration_mask = f"feature_mask.{duration}"
    valid_nights = "sleep.valid_nights"
    valid_nights_mask = f"feature_mask.{valid_nights}"
    fragmentation = "sleep.sleep_fragmentation"
    fragmentation_mask = f"feature_mask.{fragmentation}"
    train[duration] = pd.Series([0.2, 0.6, 0.8], dtype="Float64")
    train[duration_mask] = pd.Series([1, 1, 1], dtype="Int8")
    train[valid_nights] = pd.Series([3.0, 5.0, 7.0], dtype="Float64")
    train[valid_nights_mask] = pd.Series([1, 1, 1], dtype="Int8")
    train[fragmentation] = pd.Series([0.5, 0.5, 0.5], dtype="Float64")
    train[fragmentation_mask] = pd.Series([1, 1, 1], dtype="Int8")
    validation = _empty_sleep_frame(1)
    validation[duration] = pd.Series([0.99], dtype="Float64")
    validation[duration_mask] = pd.Series([1], dtype="Int8")

    preprocessor = SleepNumericFoldPreprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.selected_features == (duration, valid_nights)
    assert preprocessor.medians == pytest.approx((0.6, 5.0))
    assert list(transformed.columns) == [duration, valid_nights]
    assert transformed.iloc[0].tolist() == pytest.approx([0.99, 5.0])
    assert fragmentation not in preprocessor.selected_features


def test_fold_preprocessor_rejects_value_mask_mismatch() -> None:
    frame = _empty_sleep_frame(2)
    frame["sleep.sleep_duration_norm"] = pd.Series([0.2, 0.4], dtype="Float64")

    with pytest.raises(SleepExpertError, match="value-mask relation"):
        SleepNumericFoldPreprocessor.fit(frame)


def test_probability_calibration_selection_rule() -> None:
    raw = np.linspace(0.05, 0.95, 40)
    target = np.asarray([0] * 20 + [1] * 20)
    weight = np.ones(40)

    isotonic = SleepProbabilityCalibrator.fit(
        raw,
        target,
        weight,
        positive_participant_count=200,
        isotonic_min_positive_participants=200,
    )
    platt = SleepProbabilityCalibrator.fit(
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


def test_sleep_inputs_revalidate_all_five_bound_sources() -> None:
    inputs = load_sleep_training_inputs(REPOSITORY_ROOT)

    assert inputs.split_manifest_sha256 == EXPECTED_SPLIT_SHA256
    assert tuple(inputs.frames) == SLEEP_DATASET_IDS
    assert {name: len(frame) for name, frame in inputs.frames.items()} == {
        "psyche_d": 10_866,
        "resilient": 73,
        "nhanes": 2_775,
    }
    assert len(inputs.assignments) == 6_884
    assert {item["dataset_id"] for item in inputs.split_manifest["inputs"]} == {
        "psyche_d",
        "resilient",
        "nhanes",
        "shenzhen_elderly",
        "nhanes_ssq_2005_2008",
    }
    assert {
        dataset_id: int(
            inputs.frames[dataset_id][list(SLEEP_MASK_COLUMNS)]
            .apply(pd.to_numeric, errors="raise")
            .sum(axis=1)
            .gt(0)
            .sum()
        )
        for dataset_id in SLEEP_DATASET_IDS
    } == {
        "psyche_d": 10_745,
        "resilient": 71,
        "nhanes": 2_775,
    }


def test_real_inner_fold_preprocessing_excludes_validation_participants() -> None:
    inputs = load_sleep_training_inputs(REPOSITORY_ROOT)
    assignments = sleep._assignment_lookup(inputs.assignments)
    outer_train = {
        key for key, value in assignments.items() if value["outer_fold"] != 0
    }
    prepared = sleep._prepare_inner_fold(
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
    assert training_participants | validation_participants <= outer_train
    assert not hasattr(prepared, "ecdf_by_dataset")
    assert prepared.preprocessor.selected_features == tuple(
        feature
        for feature in SLEEP_FEATURE_COLUMNS
        if feature != "sleep.night_exit_count_mean"
    )

    values, _ = sleep._masked_sleep_values(prepared.train_table)
    expected_medians = []
    for feature in prepared.preprocessor.selected_features:
        column = values[:, SLEEP_FEATURE_COLUMNS.index(feature)]
        expected_medians.append(float(np.median(column[np.isfinite(column)])))
    assert prepared.preprocessor.medians == pytest.approx(expected_medians)


def test_partition_transform_is_identity_for_canonical_sleep_fields() -> None:
    inputs = load_sleep_training_inputs(REPOSITORY_ROOT)
    keys = {
        str(frame.iloc[0]["global_participant_id"]) for frame in inputs.frames.values()
    }
    transformed = sleep._transform_partition(inputs, keys)

    assert set(transformed["global_participant_id"].astype(str)) == keys
    for dataset_id in SLEEP_DATASET_IDS:
        actual = transformed[transformed["dataset_id"].eq(dataset_id)].iloc[0]
        source = inputs.frames[dataset_id].loc[int(actual["canonical_row_index"])]
        for column in (*SLEEP_FEATURE_COLUMNS, *SLEEP_MASK_COLUMNS):
            if pd.isna(source[column]):
                assert pd.isna(actual[column])
            else:
                assert actual[column] == source[column]
