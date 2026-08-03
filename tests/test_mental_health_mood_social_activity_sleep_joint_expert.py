from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.experts import (
    ActivitySleepJointExpertBundle,
    joint,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.activity import (
    ACTIVITY_FEATURE_COLUMNS,
    ACTIVITY_MASK_COLUMNS,
    ProbabilityCalibrator,
    compute_training_sample_weights,
    deterministic_search_candidates,
    participant_set_sha256,
    sample_weight_audit,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.joint import (
    EXPECTED_SPLIT_SHA256,
    JOINT_DATASET_IDS,
    JOINT_FEATURE_COLUMNS,
    JOINT_MASK_COLUMNS,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    UPSTREAM_EXPERT_GUARDS,
    ActivitySleepJointExpertError,
    JointNumericFoldPreprocessor,
    load_joint_training_config,
    load_joint_training_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.sleep import (
    SLEEP_FEATURE_COLUMNS,
    SLEEP_MASK_COLUMNS,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import SPLIT_ID


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "mood_social_activity_sleep_joint_expert_v3_3_3.yaml"
)


class _FixedClassifier:
    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(matrix), 0.4, dtype="float64")
        return np.column_stack([1.0 - probability, probability])


def _empty_joint_frame(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{
                feature: pd.Series([pd.NA] * row_count, dtype="Float64")
                for feature in JOINT_FEATURE_COLUMNS
            },
            **{
                mask: pd.Series([0] * row_count, dtype="Int8")
                for mask in JOINT_MASK_COLUMNS
            },
        }
    )


def test_joint_expert_is_exported_from_experts_package() -> None:
    assert ActivitySleepJointExpertBundle is joint.ActivitySleepJointExpertBundle


def test_training_config_candidate_traversal_and_guards_are_frozen() -> None:
    config = load_joint_training_config(CONFIG_PATH)
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

    assert config.run_id == "MH-20260731-003"
    assert config.split_sha256 == EXPECTED_SPLIT_SHA256
    assert config.dataset_ids == JOINT_DATASET_IDS
    assert config.upstream_expert_guards == UPSTREAM_EXPERT_GUARDS
    assert first == second
    assert len(first) == len({item["candidate_id"] for item in first}) == 32
    assert {
        name: set(item["params"][name] for item in first)
        for name in config.search_space
    } == {name: set(values) for name, values in config.search_space.items()}


def test_joint_feature_boundary_excludes_masks_coverage_and_probabilities() -> None:
    assert JOINT_FEATURE_COLUMNS == ACTIVITY_FEATURE_COLUMNS + SLEEP_FEATURE_COLUMNS
    assert JOINT_MASK_COLUMNS == ACTIVITY_MASK_COLUMNS + SLEEP_MASK_COLUMNS
    assert not set(JOINT_FEATURE_COLUMNS) & set(JOINT_MASK_COLUMNS)
    assert not any("probability" in name for name in JOINT_FEATURE_COLUMNS)
    assert not any("feature_coverage" in name for name in JOINT_FEATURE_COLUMNS)


def test_sample_weights_retain_all_four_frozen_equalities() -> None:
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
    assert weight[3] == pytest.approx(weight[4]) == pytest.approx(weight[5])
    assert weight[6] == pytest.approx(weight[7])


def test_joint_preprocessor_uses_training_values_from_both_domains() -> None:
    train = _empty_joint_frame(3)
    activity = ACTIVITY_FEATURE_COLUMNS[0]
    sleep = SLEEP_FEATURE_COLUMNS[0]
    constant = SLEEP_FEATURE_COLUMNS[1]
    train[activity] = pd.Series([0.2, 0.6, 0.8], dtype="Float64")
    train[f"feature_mask.{activity}"] = pd.Series([1, 1, 1], dtype="Int8")
    train[sleep] = pd.Series([6.0, 7.0, 8.0], dtype="Float64")
    train[f"feature_mask.{sleep}"] = pd.Series([1, 1, 1], dtype="Int8")
    train[constant] = pd.Series([0.5, 0.5, 0.5], dtype="Float64")
    train[f"feature_mask.{constant}"] = pd.Series([1, 1, 1], dtype="Int8")
    validation = _empty_joint_frame(1)
    validation[activity] = pd.Series([0.99], dtype="Float64")
    validation[f"feature_mask.{activity}"] = pd.Series([1], dtype="Int8")
    validation[sleep] = pd.Series([pd.NA], dtype="Float64")

    preprocessor = JointNumericFoldPreprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.selected_features == (activity, sleep)
    assert preprocessor.medians == pytest.approx((0.6, 7.0))
    assert transformed.iloc[0].tolist() == pytest.approx([0.99, 7.0])
    assert constant not in preprocessor.selected_features


def test_joint_value_mask_mismatch_is_rejected() -> None:
    frame = _empty_joint_frame(2)
    frame[ACTIVITY_FEATURE_COLUMNS[0]] = pd.Series([0.2, 0.4], dtype="Float64")

    with pytest.raises(ActivitySleepJointExpertError, match="value-mask relation"):
        JointNumericFoldPreprocessor.fit(frame)


def test_joint_availability_requires_real_activity_and_sleep_inputs() -> None:
    frame = _empty_joint_frame(4)
    activity = ACTIVITY_FEATURE_COLUMNS[0]
    sleep = SLEEP_FEATURE_COLUMNS[0]
    frame.loc[[1, 3], activity] = 0.5
    frame.loc[[1, 3], f"feature_mask.{activity}"] = 1
    frame.loc[[2, 3], sleep] = 7.0
    frame.loc[[2, 3], f"feature_mask.{sleep}"] = 1

    available = joint._available_joint_rows(frame)

    assert available.index.tolist() == [0]
    assert len(available) == 1
    assert available.iloc[0][activity] == pytest.approx(0.5)
    assert available.iloc[0][sleep] == pytest.approx(7.0)


def test_bundle_predict_enforces_two_sided_mask_without_learning_masks() -> None:
    activity = ACTIVITY_FEATURE_COLUMNS[0]
    sleep = SLEEP_FEATURE_COLUMNS[0]
    observed_counts = tuple(
        3 if feature in (activity, sleep) else 0 for feature in JOINT_FEATURE_COLUMNS
    )
    unique_counts = tuple(
        3 if feature in (activity, sleep) else 0 for feature in JOINT_FEATURE_COLUMNS
    )
    preprocessor = JointNumericFoldPreprocessor(
        input_features=JOINT_FEATURE_COLUMNS,
        selected_features=(activity, sleep),
        medians=(0.5, 7.0),
        observed_counts=observed_counts,
        unique_counts=unique_counts,
    )
    raw = np.linspace(0.1, 0.9, 20)
    target = np.asarray([0] * 10 + [1] * 10)
    calibrator = ProbabilityCalibrator.fit(
        raw,
        target,
        np.ones(20),
        positive_participant_count=10,
        isotonic_min_positive_participants=200,
    )
    bundle = ActivitySleepJointExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=joint.EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=JOINT_DATASET_IDS,
        input_feature_names=JOINT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=_FixedClassifier(),
        calibrator=calibrator,
        ecdf_by_dataset={name: object() for name in JOINT_DATASET_IDS},
        upstream_expert_guards=UPSTREAM_EXPERT_GUARDS,
        hyperparameters={"n_estimators": 1},
        training_scope="unit_test",
        training_participant_sha256=participant_set_sha256({"psyche_d::p1"}),
        training_participant_count=1,
        training_row_count=1,
        run_id="MH-20260731-003",
    )
    frame = _empty_joint_frame(4)
    frame.loc[[1, 3], activity] = 0.5
    frame.loc[[1, 3], f"feature_mask.{activity}"] = 1
    frame.loc[[2, 3], sleep] = 7.0
    frame.loc[[2, 3], f"feature_mask.{sleep}"] = 1

    predicted = bundle.predict(frame)

    assert predicted["expert_mask"].tolist() == [0, 0, 0, 1]
    assert predicted.loc[:2, "raw_probability"].isna().all()
    assert predicted.loc[:2, "calibrated_probability"].isna().all()
    assert predicted.loc[3, "raw_probability"] == pytest.approx(0.4)
    assert 0 <= predicted.loc[3, "calibrated_probability"] <= 1


def test_joint_inputs_revalidate_five_sources_and_upstream_artifacts() -> None:
    inputs = load_joint_training_inputs(REPOSITORY_ROOT)

    assert inputs.split_manifest_sha256 == EXPECTED_SPLIT_SHA256
    assert tuple(inputs.frames) == JOINT_DATASET_IDS
    assert {name: len(frame) for name, frame in inputs.frames.items()} == {
        "psyche_d": 10_866,
        "resilient": 73,
        "nhanes": 2_775,
    }
    assert len(inputs.assignments) == 6_884
    assert set(inputs.upstream_expert_bindings) == {"activity", "sleep"}
    assert all(
        inputs.upstream_expert_bindings[name]["model_sha256"]
        == UPSTREAM_EXPERT_GUARDS[name]["model_sha256"]
        for name in ("activity", "sleep")
    )
    changed = {name: dict(guard) for name, guard in UPSTREAM_EXPERT_GUARDS.items()}
    changed["activity"]["model_sha256"] = "0" * 64
    with pytest.raises(ActivitySleepJointExpertError, match="policy changed"):
        load_joint_training_inputs(
            REPOSITORY_ROOT,
            upstream_expert_guards=changed,
        )


def test_full_joint_transform_preserves_identity_sleep_and_expected_eligibility() -> (
    None
):
    inputs = load_joint_training_inputs(REPOSITORY_ROOT)
    assignments = joint._assignment_lookup(inputs.assignments)
    all_keys = set(assignments)
    ecdf = joint._fit_ecdfs(inputs, all_keys)
    transformed = joint._transform_partition(inputs, ecdf, all_keys)
    available = joint._available_joint_rows(transformed)

    assert len(transformed) == 13_714
    assert len(available) == 13_585
    assert available["global_participant_id"].nunique() == 6_847
    assert int(available["binary_target"].sum()) == 3_681
    assert available.groupby("dataset_id", sort=False).size().to_dict() == {
        "psyche_d": 10_739,
        "resilient": 71,
        "nhanes": 2_775,
    }
    for dataset_id in JOINT_DATASET_IDS:
        actual = transformed[transformed["dataset_id"].eq(dataset_id)].iloc[0]
        source = inputs.frames[dataset_id].loc[int(actual["canonical_row_index"])]
        for column in (*SLEEP_FEATURE_COLUMNS, *SLEEP_MASK_COLUMNS):
            if pd.isna(source[column]):
                assert pd.isna(actual[column])
            else:
                assert actual[column] == source[column]


def test_real_inner_fold_fits_ecdf_and_preprocessor_on_training_only() -> None:
    inputs = load_joint_training_inputs(REPOSITORY_ROOT)
    assignments = joint._assignment_lookup(inputs.assignments)
    outer_train = {
        key for key, value in assignments.items() if value["outer_fold"] != 0
    }
    prepared = joint._prepare_inner_fold(
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
    inner_validation = {
        key
        for key in outer_train
        if assignments[key]["inner_validation_fold_by_outer_fold"]["0"] == 0
    }
    inner_training = outer_train - inner_validation

    assert training_participants.isdisjoint(validation_participants)
    assert training_participants | validation_participants <= outer_train
    assert prepared.preprocessor.selected_features == (
        "activity.activity_volume_norm",
        "activity.relative_amplitude",
        "activity.interdaily_stability",
        "activity.intradaily_variability",
        "activity.activity_variability",
        "activity.valid_days",
        *(
            feature
            for feature in SLEEP_FEATURE_COLUMNS
            if feature != "sleep.night_exit_count_mean"
        ),
    )
    for dataset_id in JOINT_DATASET_IDS:
        expected = {key for key in inner_training if key.startswith(f"{dataset_id}::")}
        fitted = prepared.ecdf_by_dataset[dataset_id]
        assert fitted.training_participant_count == len(expected)
        assert fitted.training_participant_sha256 == participant_set_sha256(expected)
