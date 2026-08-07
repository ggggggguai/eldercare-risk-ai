from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (
    ForecastExperimentError,
    build_forecast_pair,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (
    ElasticNetPreprocessor,
    calibration_bins,
    participant_class_weights,
    participant_equal_weights,
    select_threshold,
    weighted_metrics,
)
from elderly_monitoring.datasets.adapters.psyche_d import (
    LABEL_FIELDS,
    PSYCHE_D_SOURCE_FIELDS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parents[1]
MAPPING_PATH = (
    WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D" / "p0_feature_mapping_v1.yaml"
)


def _split_fixture(path: Path, participants: list[str]) -> Path:
    rows = [
        {
            "dataset_id": "psyche_d",
            "global_participant_id": f"psyche_d::{participant}",
            "outer_fold": index % 5,
            "inner_validation_fold_by_outer_fold": {
                str(fold): None for fold in range(5)
            },
        }
        for index, participant in enumerate(participants)
    ]
    payload = {
        "manifest_version": "mood-social-nested-participant-split-manifest-v1",
        "protocol": {
            "random_seed": 20260728,
            "global_participant_key_format": "dataset_id::participant_id",
        },
        "participant_assignments": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source(participants: list[str] = ["alpha_with_gap", "beta"]) -> pd.DataFrame:
    rows: list[dict[str, float | int | None]] = []
    index: list[str] = []
    for participant_index, participant in enumerate(participants):
        for month in (1, 2, 3, 4):
            row: dict[str, float | int | None] = {
                field: float(participant_index + month)
                for field in PSYCHE_D_SOURCE_FIELDS
            }
            row.update(
                {
                    "phq9_score_start": 1.0,
                    "phq9_score_end": 12.0 if participant_index == 0 else 2.0,
                    "phq9_cat_start": 0.0,
                    "phq9_cat_end": 2.0 if participant_index == 0 else 0.0,
                }
            )
            if month != 3:
                row.update({field: None for field in LABEL_FIELDS})
            rows.append(row)
            index.append(f"{participant}_{month}")
    return pd.DataFrame(rows, index=pd.Index(index))


def test_forecast_pair_uses_exact_months_and_audit_schema(tmp_path: Path) -> None:
    split_path = _split_fixture(tmp_path / "split.json", ["alpha_with_gap", "beta"])
    one, two, common = build_forecast_pair(
        _source(),
        mapping_path=MAPPING_PATH,
        split_path=split_path,
        expected_1m=2,
        expected_2m=2,
        expected_common=2,
    )

    assert set(one["feature_month_slot"]) == {2}
    assert set(two["feature_month_slot"]) == {1}
    assert set(one["task_id"]) == {"forecast_1m"}
    assert set(two["task_id"]) == {"forecast_2m"}
    assert set(common["target_window_id"]) == set(one["target_window_id"])
    assert all(one["target_window_id"].str.contains("::m3"))
    assert all(one["feature_window_id"].str.contains("::m2"))
    assert all(two["feature_window_id"].str.contains("::m1"))
    assert list(one["feature_start_time"].unique()) == [None]
    assert not set(LABEL_FIELDS).intersection(one.columns)


def test_target_and_future_month_fields_do_not_enter_features(tmp_path: Path) -> None:
    split_path = _split_fixture(tmp_path / "split.json", ["alpha_with_gap", "beta"])
    source = _source()
    baseline, _, _ = build_forecast_pair(
        source, mapping_path=MAPPING_PATH, split_path=split_path
    )
    changed = source.copy()
    changed.loc["alpha_with_gap_3", PSYCHE_D_SOURCE_FIELDS[0]] = 999999.0
    changed.loc["alpha_with_gap_4", PSYCHE_D_SOURCE_FIELDS[1]] = 888888.0
    altered, _, _ = build_forecast_pair(
        changed, mapping_path=MAPPING_PATH, split_path=split_path
    )
    pd.testing.assert_frame_equal(
        baseline.sort_index(axis=1), altered.sort_index(axis=1), check_dtype=False
    )


def test_duplicate_participant_month_fails(tmp_path: Path) -> None:
    split_path = _split_fixture(tmp_path / "split.json", ["alpha_with_gap", "beta"])
    source = pd.concat([_source(), _source().iloc[[0]]])
    with pytest.raises(ForecastExperimentError, match="sample index must be unique"):
        build_forecast_pair(source, mapping_path=MAPPING_PATH, split_path=split_path)


def test_invalid_complete_label_month_fails(tmp_path: Path) -> None:
    split_path = _split_fixture(tmp_path / "split.json", ["alpha_with_gap", "beta"])
    source = _source()
    source.loc["alpha_with_gap_1", list(LABEL_FIELDS)] = [1, 2, 0, 0]
    with pytest.raises(ForecastExperimentError, match="nominal months"):
        build_forecast_pair(source, mapping_path=MAPPING_PATH, split_path=split_path)


def test_forecast_weights_keep_training_and_evaluation_policies_separate() -> None:
    frame = pd.DataFrame(
        {
            "global_participant_id": ["p0", "p0", "p1", "p2", "p2"],
            "future_binary_target": [0, 0, 0, 1, 1],
        }
    )
    train = participant_class_weights(frame)
    evaluation = participant_equal_weights(frame)
    assert train[frame.future_binary_target.eq(0)].sum() == pytest.approx(0.5)
    assert train[frame.future_binary_target.eq(1)].sum() == pytest.approx(0.5)
    assert evaluation[frame.global_participant_id.eq("p0")].sum() == pytest.approx(
        1 / 3
    )
    assert evaluation[frame.global_participant_id.eq("p1")].sum() == pytest.approx(
        1 / 3
    )
    assert evaluation[frame.global_participant_id.eq("p2")].sum() == pytest.approx(
        1 / 3
    )


def test_elasticnet_preprocessor_preserves_missing_indicators() -> None:
    names = tuple(PSYCHE_D_SOURCE_FIELDS)
    train = pd.DataFrame({name: [1.0, 3.0] for name in names})
    train.loc[0, names[0]] = None
    train[names[1]] = None
    validation = pd.DataFrame({name: [2.0] for name in names})
    validation.loc[0, names[0]] = None
    transformed = ElasticNetPreprocessor(names).fit(train).transform(validation)
    assert transformed.shape == (1, 54)
    assert transformed[0, 27] == 1.0
    assert transformed[0, 28] == 1.0
    assert transformed[0, 1] == 0.0


def test_threshold_selection_prefers_sensitivity_then_lower_value() -> None:
    y = [0, 0, 1, 1]
    probability = [0.2, 0.4, 0.4, 0.8]
    threshold, record = select_threshold(y, probability, [0.25] * 4)
    assert threshold in set(probability) | {0.0, 1.0}
    assert record["operator"] == "greater_than_or_equal"
    assert record["candidate_count"] == len(set(probability) | {0.0, 1.0})


def test_metrics_support_fold_specific_thresholds_and_tie_preserving_bins() -> None:
    y = [0, 1, 0, 1]
    probability = [0.1, 0.4, 0.4, 0.9]
    weights = [0.25] * 4
    metrics = weighted_metrics(
        y, probability, probability, [0.2, 0.3, 0.5, 0.8], weights
    )
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    bins = calibration_bins(y, probability, weights)
    tied = [
        row for row in bins if row["probability_min"] <= 0.4 <= row["probability_max"]
    ]
    assert len(tied) == 1
    assert tied[0]["row_count"] >= 2


def test_forecast_remains_isolated_from_v33_package_and_backend() -> None:
    acceptance_path = (
        REPOSITORY_ROOT
        / "models"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "acceptance.json"
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["status"] == "passed"
    assert acceptance["model_version"] == "mood-fusion-v3.3.3"
    assert (
        acceptance["package_manifest_sha256"]
        == "b06ba3fc042e07eb17738a21b3683f3029a3adeae6aadd16705b9eb51a0a08d8"
    )
    assert (
        acceptance["package_checksums_sha256"]
        == "e40ac1b96c3e4075a1f62523f446f7e1717cefa7d75a4a59128d11611e4f4ce3"
    )
    pipeline = (
        REPOSITORY_ROOT
        / "src"
        / "elderly_monitoring"
        / "modules"
        / "mental_health"
        / "mood_social"
        / "pipeline.py"
    ).read_text(encoding="utf-8")
    prohibited = ("forecast_1m", "forecast_2m", "mood_forecast", "forecast_v3.4")
    assert not any(token in pipeline for token in prohibited)
    backend_root = WORKSPACE_ROOT / "backend"
    for path in backend_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in prohibited), path


@pytest.mark.integration
def test_frozen_real_source_counts() -> None:
    source_path = (
        WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D" / "anon_processed_df_parquet"
    )
    split_path = (
        REPOSITORY_ROOT
        / "data"
        / "processed"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "splits"
        / "split_manifest.json"
    )
    if not source_path.exists() or not split_path.exists():
        pytest.skip("frozen PSYCHE-D source or split is unavailable")
    source = pd.read_parquet(
        source_path, columns=[*PSYCHE_D_SOURCE_FIELDS, *LABEL_FIELDS]
    )
    one, two, common = build_forecast_pair(
        source,
        mapping_path=MAPPING_PATH,
        split_path=split_path,
        expected_1m=9393,
        expected_2m=9280,
        expected_common=8494,
    )
    assert len(source) == 35694
    assert one["global_participant_id"].nunique() == 3635
    assert two["global_participant_id"].nunique() == 3593
