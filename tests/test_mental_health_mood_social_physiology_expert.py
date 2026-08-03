from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.experts import (
    PhysiologyExpertBundle,
)
from elderly_monitoring.modules.mental_health.mood_social.experts import (
    physiology,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.physiology import (
    ALL_FROZEN_DATASET_IDS,
    EXPECTED_SPLIT_SHA256,
    EXCLUDED_DATASET_IDS,
    PHYSIOLOGY_DATASET_IDS,
    PHYSIOLOGY_FEATURE_COLUMNS,
    PHYSIOLOGY_MASK_COLUMNS,
    UPSTREAM_EXPERT_GUARDS,
    PhysiologyExpertError,
    PhysiologyNumericFoldPreprocessor,
    PhysiologyProbabilityCalibrator,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    load_physiology_training_config,
    load_physiology_training_inputs,
    sample_weight_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "mood_social_physiology_expert_v3_3_3.yaml"
)


def _empty_physiology_frame(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{
                feature: pd.Series([pd.NA] * row_count, dtype="Float64")
                for feature in PHYSIOLOGY_FEATURE_COLUMNS
            },
            **{
                mask: pd.Series([0] * row_count, dtype="Int8")
                for mask in PHYSIOLOGY_MASK_COLUMNS
            },
        }
    )


def test_physiology_expert_is_exported_from_experts_package() -> None:
    assert PhysiologyExpertBundle is physiology.PhysiologyExpertBundle


def test_training_config_and_complete_candidate_grid_are_frozen() -> None:
    config = load_physiology_training_config(CONFIG_PATH)

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

    assert config.run_id == "MH-20260731-004"
    assert config.split_sha256 == EXPECTED_SPLIT_SHA256
    assert config.dataset_ids == ("resilient",)
    assert config.calibration_method == "platt"
    assert config.logistic_solver == "saga"
    assert config.logistic_max_iter == 5000
    assert first == second
    assert len(first) == len({item["candidate_id"] for item in first}) == 20
    assert {(item["params"]["C"], item["params"]["l1_ratio"]) for item in first} == set(
        itertools.product((0.01, 0.1, 1, 10), (0, 0.25, 0.5, 0.75, 1))
    )


def test_complete_grid_rejects_partial_candidate_count() -> None:
    config = load_physiology_training_config(CONFIG_PATH)

    with pytest.raises(PhysiologyExpertError, match="complete grid"):
        deterministic_search_candidates(
            config.search_space,
            random_seed=config.random_seed,
            candidate_count=19,
        )


def test_sample_weights_follow_frozen_class_and_participant_equalities() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["resilient"] * 7,
            "global_participant_id": ["p1", "p1", "p2", "p3", "p3", "p3", "p4"],
            "binary_target": [0, 0, 0, 1, 1, 1, 1],
        }
    )

    weight = compute_training_sample_weights(frame)
    audit = sample_weight_audit(frame, weight)

    assert weight.sum() == pytest.approx(len(frame))
    source = audit["by_dataset"]["resilient"]
    assert source["by_class"]["0"] == pytest.approx(source["by_class"]["1"])
    assert weight[0] == pytest.approx(weight[1])
    assert weight[0] + weight[1] == pytest.approx(weight[2])
    assert weight[3] == pytest.approx(weight[4]) == pytest.approx(weight[5])
    assert weight[3] + weight[4] + weight[5] == pytest.approx(weight[6])


def test_fold_preprocessor_fits_median_mean_and_scale_on_training_only() -> None:
    train = _empty_physiology_frame(4)
    heart_rate = "physiology.heart_rate_mean_bpm"
    heart_rate_mask = f"feature_mask.{heart_rate}"
    respiration = "physiology.respiration_rate_mean_bpm"
    respiration_mask = f"feature_mask.{respiration}"
    constant = "physiology.heart_rate_min_bpm"
    constant_mask = f"feature_mask.{constant}"
    train[heart_rate] = pd.Series([50.0, 60.0, pd.NA, 80.0], dtype="Float64")
    train[heart_rate_mask] = pd.Series([1, 1, 0, 1], dtype="Int8")
    train[respiration] = pd.Series([10.0, 12.0, 14.0, 16.0], dtype="Float64")
    train[respiration_mask] = pd.Series([1, 1, 1, 1], dtype="Int8")
    train[constant] = pd.Series([40.0] * 4, dtype="Float64")
    train[constant_mask] = pd.Series([1] * 4, dtype="Int8")
    validation = _empty_physiology_frame(1)
    validation[heart_rate] = pd.Series([1000.0], dtype="Float64")
    validation[heart_rate_mask] = pd.Series([1], dtype="Int8")

    preprocessor = PhysiologyNumericFoldPreprocessor.fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert preprocessor.selected_features == (heart_rate, respiration)
    assert preprocessor.medians == pytest.approx((60.0, 13.0))
    assert transformed_train.mean().to_numpy() == pytest.approx([0.0, 0.0])
    assert transformed_train.std(ddof=0).to_numpy() == pytest.approx([1.0, 1.0])
    assert transformed_validation.loc[0, respiration] == pytest.approx(
        (13.0 - preprocessor.means[1]) / preprocessor.scales[1]
    )
    assert transformed_validation.loc[0, heart_rate] > 10
    assert constant not in preprocessor.selected_features


def test_fold_preprocessor_rejects_value_mask_mismatch() -> None:
    frame = _empty_physiology_frame(2)
    frame["physiology.heart_rate_mean_bpm"] = pd.Series([60.0, 70.0], dtype="Float64")

    with pytest.raises(PhysiologyExpertError, match="value-mask relation"):
        PhysiologyNumericFoldPreprocessor.fit(frame)


def test_probability_calibrator_is_always_platt() -> None:
    raw = np.linspace(0.05, 0.95, 40)
    target = np.asarray([0] * 20 + [1] * 20)
    calibrator = PhysiologyProbabilityCalibrator.fit(
        raw,
        target,
        np.ones(40),
        positive_participant_count=20,
    )

    assert calibrator.method == "platt"
    predicted = calibrator.predict(raw)
    assert np.all((0 <= predicted) & (predicted <= 1))


def test_binary_metrics_preserve_null_auprc_for_no_positive_fold() -> None:
    result = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])

    assert result["auprc"] is None
    assert result["auroc"] is None
    assert result["sensitivity"] is None
    assert result["positive_row_count"] == 0


def test_inputs_validate_five_sources_and_three_upstream_experts() -> None:
    inputs = load_physiology_training_inputs(REPOSITORY_ROOT)
    frame = inputs.frames["resilient"]
    masks = frame[list(PHYSIOLOGY_MASK_COLUMNS)].apply(pd.to_numeric, errors="raise")

    assert inputs.split_manifest_sha256 == EXPECTED_SPLIT_SHA256
    assert tuple(inputs.frames) == PHYSIOLOGY_DATASET_IDS == ("resilient",)
    assert tuple(inputs.input_bindings) == ALL_FROZEN_DATASET_IDS
    assert tuple(inputs.upstream_expert_bindings) == ("activity", "sleep", "joint")
    assert len(inputs.assignments) == len(frame) == 73
    assert int(frame["binary_target"].sum()) == 10
    assert int(masks.sum(axis=1).gt(0).sum()) == 71
    assert int(frame.loc[masks.sum(axis=1).gt(0), "binary_target"].sum()) == 9
    assert set(EXCLUDED_DATASET_IDS).isdisjoint(inputs.frames)
    assert {
        name: {key: binding[key] for key in ("model_sha256", "manifest_sha256")}
        for name, binding in inputs.upstream_expert_bindings.items()
    } == {
        name: {key: guard[key] for key in ("model_sha256", "manifest_sha256")}
        for name, guard in UPSTREAM_EXPERT_GUARDS.items()
    }


def test_real_fold_preprocessing_is_leakage_safe_and_selects_seven_fields() -> None:
    inputs = load_physiology_training_inputs(REPOSITORY_ROOT)
    assignments = physiology._assignment_lookup(inputs.assignments)
    outer_train = {
        key for key, value in assignments.items() if value["outer_fold"] != 0
    }
    prepared = physiology._prepare_inner_fold(
        inputs,
        assignments,
        outer_fold=0,
        inner_fold=0,
        outer_train_keys=outer_train,
    )
    training_participants = set(prepared.train_table["global_participant_id"])
    validation_participants = set(prepared.validation_table["global_participant_id"])

    assert training_participants.isdisjoint(validation_participants)
    assert training_participants | validation_participants <= outer_train
    assert prepared.preprocessor.selected_features == tuple(
        feature
        for feature in PHYSIOLOGY_FEATURE_COLUMNS
        if feature
        not in {
            "physiology.respiratory_abnormal_ratio",
            "physiology.valid_nights",
        }
    )
    assert prepared.train_matrix.mean().to_numpy() == pytest.approx(
        np.zeros(7), abs=1e-12
    )
    assert prepared.train_matrix.std(ddof=0).to_numpy() == pytest.approx(
        np.ones(7), abs=1e-12
    )


def test_real_outer_and_empty_positive_inner_fold_counts_are_frozen() -> None:
    inputs = load_physiology_training_inputs(REPOSITORY_ROOT)
    assignments = physiology._assignment_lookup(inputs.assignments)
    outer_counts: list[tuple[int, int]] = []
    empty_positive_inner: list[tuple[int, int]] = []
    all_keys = set(assignments)
    for outer_fold in range(5):
        test_keys = {
            key
            for key, value in assignments.items()
            if value["outer_fold"] == outer_fold
        }
        test = physiology._available_physiology_rows(
            physiology._transform_partition(inputs, test_keys)
        )
        outer_counts.append((len(test), int(test["binary_target"].sum())))
        outer_train = all_keys - test_keys
        for inner_fold in range(5):
            prepared = physiology._prepare_inner_fold(
                inputs,
                assignments,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_train_keys=outer_train,
            )
            if int(prepared.validation_target.sum()) == 0:
                empty_positive_inner.append((outer_fold, inner_fold))

    assert outer_counts == [(15, 2), (15, 2), (15, 2), (12, 1), (14, 2)]
    assert empty_positive_inner == [(2, 3), (4, 0)]


def test_partition_transform_preserves_only_resilient_physiology_fields() -> None:
    inputs = load_physiology_training_inputs(REPOSITORY_ROOT)
    key = str(inputs.frames["resilient"].iloc[0]["global_participant_id"])
    transformed = physiology._transform_partition(inputs, {key})
    actual = transformed.iloc[0]
    source = inputs.frames["resilient"].loc[int(actual["canonical_row_index"])]

    assert set(transformed["dataset_id"]) == {"resilient"}
    assert not any(column.startswith("activity.") for column in transformed.columns)
    assert not any(column.startswith("sleep.") for column in transformed.columns)
    for column in (*PHYSIOLOGY_FEATURE_COLUMNS, *PHYSIOLOGY_MASK_COLUMNS):
        if pd.isna(source[column]):
            assert pd.isna(actual[column])
        else:
            assert actual[column] == source[column]
