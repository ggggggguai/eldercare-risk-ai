from __future__ import annotations

import itertools
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.experts import (
    SocialContextExpertBundle,
)
from elderly_monitoring.modules.mental_health.mood_social.experts import social_context
from elderly_monitoring.modules.mental_health.mood_social.experts.social_context import (
    EXCLUDED_DATASET_IDS,
    EXPECTED_SPLIT_SHA256,
    MISSING_CATEGORY,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    SOCIAL_CONTEXT_CATEGORICAL_COLUMNS,
    SOCIAL_CONTEXT_DATASET_IDS,
    SOCIAL_CONTEXT_FEATURE_COLUMNS,
    SOCIAL_CONTEXT_MASK_COLUMNS,
    SOCIAL_CONTEXT_NUMERIC_COLUMNS,
    UPSTREAM_EXPERT_GUARDS,
    SocialContextExpertError,
    SocialContextFoldPreprocessor,
    SocialContextProbabilityCalibrator,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    load_social_context_training_config,
    load_social_context_training_inputs,
    sample_weight_audit,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "mood_social_social_context_expert_v3_3_3.yaml"
)
EXPECTED_SELECTED_FEATURES = (
    "social_context.age_group",
    "social_context.sex",
    "social_context.marital_status",
    "social_context.self_rated_health",
    "social_context.education_level",
    "social_context.economic_status",
)


def _empty_social_frame(row_count: int) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for feature in SOCIAL_CONTEXT_FEATURE_COLUMNS:
        dtype = "string" if feature in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS else "Float64"
        values[feature] = pd.Series([pd.NA] * row_count, dtype=dtype)
    return pd.DataFrame(
        {
            **values,
            **{
                mask: pd.Series([0] * row_count, dtype="Int8")
                for mask in SOCIAL_CONTEXT_MASK_COLUMNS
            },
        }
    )


@pytest.fixture(scope="module")
def real_inputs() -> social_context.SocialContextTrainingInputs:
    return load_social_context_training_inputs(REPOSITORY_ROOT)


def test_social_context_expert_is_exported_from_experts_package() -> None:
    assert SocialContextExpertBundle is social_context.SocialContextExpertBundle


def test_feature_boundary_contains_only_frozen_profile_fields() -> None:
    assert len(SOCIAL_CONTEXT_FEATURE_COLUMNS) == 10
    assert len(SOCIAL_CONTEXT_CATEGORICAL_COLUMNS) == 6
    assert len(SOCIAL_CONTEXT_NUMERIC_COLUMNS) == 4
    assert all(
        name.startswith("social_context.") for name in SOCIAL_CONTEXT_FEATURE_COLUMNS
    )
    assert all(
        name.startswith("feature_mask.social_context.")
        for name in SOCIAL_CONTEXT_MASK_COLUMNS
    )
    forbidden = ("activity.", "sleep.", "physiology.", "social_contact.", "source__")
    assert not any(
        name.startswith(forbidden) for name in SOCIAL_CONTEXT_FEATURE_COLUMNS
    )


def test_training_config_and_complete_catboost_grid_are_frozen() -> None:
    config = load_social_context_training_config(CONFIG_PATH)
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

    assert config.run_id == "MH-20260801-005"
    assert config.split_sha256 == EXPECTED_SPLIT_SHA256
    assert config.dataset_ids == SOCIAL_CONTEXT_DATASET_IDS
    assert config.calibration_positive_participant_threshold == 200
    assert config.catboost_thread_count == 1
    assert first == second
    assert len(first) == len({item["candidate_id"] for item in first}) == 16
    assert {
        (
            item["params"]["iterations"],
            item["params"]["depth"],
            item["params"]["learning_rate"],
            item["params"]["l2_leaf_reg"],
        )
        for item in first
    } == set(itertools.product((300, 600), (4, 6), (0.03, 0.05), (3, 10)))


def test_complete_grid_rejects_partial_candidate_count() -> None:
    config = load_social_context_training_config(CONFIG_PATH)
    with pytest.raises(SocialContextExpertError, match="complete grid"):
        deterministic_search_candidates(
            config.search_space,
            random_seed=config.random_seed,
            candidate_count=15,
        )


def test_catboost_metadata_is_deterministic_and_scope_specific() -> None:
    config = load_social_context_training_config(CONFIG_PATH)
    params = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10,
    }
    production_first = social_context._deterministic_catboost_metadata(
        config,
        params,
        training_identity="production_aggregate",
    )
    production_second = social_context._deterministic_catboost_metadata(
        config,
        params,
        training_identity="production_aggregate",
    )
    outer = social_context._deterministic_catboost_metadata(
        config,
        params,
        training_identity="outer_fold_0_aggregate",
    )

    assert production_first == production_second
    assert production_first["train_finish_time"] == "1970-01-01T00:00:00Z"
    assert len(production_first["model_guid"].split("-")) == 4
    assert production_first["eldercare_training_identity"] == "production_aggregate"
    assert (
        production_first["eldercare_training_sha256"]
        != outer["eldercare_training_sha256"]
    )


def test_sample_weights_follow_all_four_frozen_equalities() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a"] * 7 + ["b"] * 4,
            "global_participant_id": [
                "a0",
                "a0",
                "a1",
                "a2",
                "a2",
                "a2",
                "a3",
                "b0",
                "b1",
                "b2",
                "b3",
            ],
            "binary_target": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
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
    assert weight[3] + weight[4] + weight[5] == pytest.approx(weight[6])


def test_fold_preprocessor_fits_vocabulary_and_median_on_training_only() -> None:
    train = _empty_social_frame(4)
    age = "social_context.age_group"
    age_mask = f"feature_mask.{age}"
    health = "social_context.self_rated_health"
    health_mask = f"feature_mask.{health}"
    sex = "social_context.sex"
    sex_mask = f"feature_mask.{sex}"
    train[age] = pd.Series(["60_69", "70_79", pd.NA, "60_69"], dtype="string")
    train[age_mask] = pd.Series([1, 1, 0, 1], dtype="Int8")
    train[health] = pd.Series([1, 3, pd.NA, 5], dtype="Float64")
    train[health_mask] = pd.Series([1, 1, 0, 1], dtype="Int8")
    train[sex] = pd.Series(["female"] * 4, dtype="string")
    train[sex_mask] = pd.Series([1] * 4, dtype="Int8")
    validation = _empty_social_frame(2)
    validation[age] = pd.Series(["80_plus", pd.NA], dtype="string")
    validation[age_mask] = pd.Series([1, 0], dtype="Int8")

    preprocessor = SocialContextFoldPreprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.selected_features == (age, health)
    assert preprocessor.category_levels[age] == ("60_69", "70_79")
    assert preprocessor.medians == (None, 3.0)
    assert transformed[age].tolist() == [MISSING_CATEGORY, MISSING_CATEGORY]
    assert transformed[health].tolist() == pytest.approx([3.0, 3.0])
    assert sex not in preprocessor.selected_features


def test_fold_preprocessor_rejects_value_mask_mismatch() -> None:
    frame = _empty_social_frame(2)
    frame["social_context.sex"] = pd.Series(["female", "male"], dtype="string")

    with pytest.raises(SocialContextExpertError, match="value-mask relation"):
        SocialContextFoldPreprocessor.fit(frame)


def test_probability_calibration_follows_positive_participant_rule() -> None:
    raw = np.linspace(0.01, 0.99, 400)
    target = np.asarray([0] * 200 + [1] * 200)
    isotonic = SocialContextProbabilityCalibrator.fit(
        raw,
        target,
        np.ones(400),
        positive_participant_count=200,
        isotonic_min_positive_participants=200,
    )
    platt = SocialContextProbabilityCalibrator.fit(
        raw[:40],
        np.asarray([0] * 20 + [1] * 20),
        np.ones(40),
        positive_participant_count=20,
        isotonic_min_positive_participants=200,
    )

    assert isotonic.method == "isotonic"
    assert platt.method == "platt"
    assert np.all((0 <= isotonic.predict(raw)) & (isotonic.predict(raw) <= 1))
    assert np.all((0 <= platt.predict(raw[:40])) & (platt.predict(raw[:40]) <= 1))


def test_binary_metrics_preserve_null_class_metrics() -> None:
    result = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["auprc"] is None
    assert result["auroc"] is None
    assert result["sensitivity"] is None
    assert result["positive_row_count"] == 0


def test_inputs_validate_four_sources_and_four_upstream_experts(
    real_inputs: social_context.SocialContextTrainingInputs,
) -> None:
    assert real_inputs.split_manifest_sha256 == EXPECTED_SPLIT_SHA256
    assert tuple(real_inputs.frames) == SOCIAL_CONTEXT_DATASET_IDS
    assert tuple(real_inputs.input_bindings) == SOCIAL_CONTEXT_DATASET_IDS
    assert tuple(real_inputs.upstream_expert_bindings) == (
        "activity",
        "sleep",
        "joint",
        "physiology",
    )
    assert len(real_inputs.assignments) == 11_325
    assert sum(len(frame) for frame in real_inputs.frames.values()) == 11_325
    assert (
        sum(int(frame["binary_target"].sum()) for frame in real_inputs.frames.values())
        == 635
    )
    assert set(EXCLUDED_DATASET_IDS).isdisjoint(real_inputs.frames)
    for frame in real_inputs.frames.values():
        masks = frame[list(SOCIAL_CONTEXT_MASK_COLUMNS)].apply(
            pd.to_numeric, errors="raise"
        )
        assert masks.sum(axis=1).gt(0).all()
    assert {
        name: {key: binding[key] for key in ("model_sha256", "manifest_sha256")}
        for name, binding in real_inputs.upstream_expert_bindings.items()
    } == {
        name: {key: guard[key] for key in ("model_sha256", "manifest_sha256")}
        for name, guard in UPSTREAM_EXPERT_GUARDS.items()
    }


def test_real_inner_fold_preprocessing_is_leakage_safe(
    real_inputs: social_context.SocialContextTrainingInputs,
) -> None:
    assignments = social_context._assignment_lookup(real_inputs.assignments)
    outer_train = {
        key for key, value in assignments.items() if value["outer_fold"] != 0
    }
    prepared = social_context._prepare_inner_fold(
        real_inputs,
        assignments,
        outer_fold=0,
        inner_fold=0,
        outer_train_keys=outer_train,
    )
    training_participants = set(prepared.train_table["global_participant_id"])
    validation_participants = set(prepared.validation_table["global_participant_id"])

    assert training_participants.isdisjoint(validation_participants)
    assert training_participants | validation_participants == outer_train
    assert prepared.preprocessor.selected_features == EXPECTED_SELECTED_FEATURES
    assert prepared.train_matrix.columns.tolist() == list(EXPECTED_SELECTED_FEATURES)
    assert prepared.validation_matrix.columns.tolist() == list(
        EXPECTED_SELECTED_FEATURES
    )


def test_bundle_predict_requires_real_selected_profile_input() -> None:
    train = _empty_social_frame(8)
    age = "social_context.age_group"
    age_mask = f"feature_mask.{age}"
    health = "social_context.self_rated_health"
    health_mask = f"feature_mask.{health}"
    train[age] = pd.Series(["60_69", "70_79", "60_69", "70_79"] * 2, dtype="string")
    train[age_mask] = pd.Series([1] * 8, dtype="Int8")
    train[health] = pd.Series([1, 2, 3, 4, 2, 3, 4, 5], dtype="Float64")
    train[health_mask] = pd.Series([1] * 8, dtype="Int8")
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    preprocessor = SocialContextFoldPreprocessor.fit(train)
    matrix = preprocessor.transform(train)
    classifier = CatBoostClassifier(
        iterations=20,
        depth=2,
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_seed=20260728,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
        bootstrap_type="No",
        random_strength=0,
    )
    classifier.fit(matrix, target, cat_features=[0], verbose=False)
    raw = classifier.predict_proba(matrix)[:, 1]
    calibrator = SocialContextProbabilityCalibrator.fit(
        raw,
        target,
        np.ones(8),
        positive_participant_count=4,
        isotonic_min_positive_participants=200,
    )
    bundle = SocialContextExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=social_context.EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=social_context.SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=SOCIAL_CONTEXT_DATASET_IDS,
        input_feature_names=SOCIAL_CONTEXT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        hyperparameters={"iterations": 20, "depth": 2},
        training_scope="test",
        training_participant_sha256="0" * 64,
        training_participant_count=8,
        training_row_count=8,
        run_id="test",
    )
    query = _empty_social_frame(2)
    query[age] = pd.Series(["60_69", pd.NA], dtype="string")
    query[age_mask] = pd.Series([1, 0], dtype="Int8")
    predicted = bundle.predict(query)

    assert predicted["expert_mask"].tolist() == [1, 0]
    assert 0 <= predicted.loc[0, "calibrated_probability"] <= 1
    assert pd.isna(predicted.loc[1, "raw_probability"])
    assert pd.isna(predicted.loc[1, "calibrated_probability"])
