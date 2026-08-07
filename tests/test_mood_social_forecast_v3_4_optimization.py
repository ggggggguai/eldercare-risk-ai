from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from elderly_monitoring.modules.mental_health.mood_social.forecast_optimization import (
    SelectionCandidate,
    build_history_forecast_samples,
    evaluate_inner_candidate,
    history_feature_groups,
    strict_auxiliary_inner_views,
)
from elderly_monitoring.datasets.adapters.psyche_d import PSYCHE_D_SOURCE_FIELDS


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]


def _mapping(path: Path) -> Path:
    payload = {
        "features": {
            "steps": [{"source_field": name} for name in PSYCHE_D_SOURCE_FIELDS[:10]],
            "sleep": [{"source_field": name} for name in PSYCHE_D_SOURCE_FIELDS[10:]],
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _split(path: Path, participants: list[str]) -> Path:
    rows = [
        {
            "dataset_id": "psyche_d",
            "global_participant_id": f"psyche_d::{participant}",
            "outer_fold": index % 5,
        }
        for index, participant in enumerate(participants)
    ]
    path.write_text(
        json.dumps(
            {
                "manifest_version": "mood-social-nested-participant-split-manifest-v1",
                "protocol": {
                    "random_seed": 20260728,
                    "global_participant_key_format": "dataset_id::participant_id",
                },
                "participant_assignments": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def _source(participants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    index: list[str] = []
    for participant_index, participant in enumerate(participants):
        for month in range(5):
            row = {
                name: float(participant_index * 100 + field_index + month)
                for field_index, name in enumerate(PSYCHE_D_SOURCE_FIELDS)
            }
            row.update(
                {
                    "phq9_score_start": np.nan,
                    "phq9_score_end": np.nan,
                    "phq9_cat_start": np.nan,
                    "phq9_cat_end": np.nan,
                }
            )
            if month == 3:
                score = float((participant_index * 3) % 28)
                row.update(
                    {
                        "phq9_score_start": score,
                        "phq9_score_end": score,
                        "phq9_cat_start": float(min(int(score // 5), 4)),
                        "phq9_cat_end": float(min(int(score // 5), 4)),
                    }
                )
            rows.append(row)
            index.append(f"{participant}_{month}")
    return pd.DataFrame(rows, index=index)


def _build(tmp_path: Path, task_id: str = "forecast_1m") -> pd.DataFrame:
    participants = [f"person_{index}" for index in range(15)]
    return build_history_forecast_samples(
        _source(participants),
        task_id=task_id,
        mapping_path=_mapping(tmp_path / "mapping.yaml"),
        split_path=_split(tmp_path / "split.json", participants),
        expected_count=len(participants),
    )


def test_history_feature_groups_have_frozen_counts_and_formulas(tmp_path: Path) -> None:
    samples = _build(tmp_path)
    groups = history_feature_groups(PSYCHE_D_SOURCE_FIELDS)
    assert {name: len(values) for name, values in groups.items()} == {
        "anchor_only": 54,
        "anchor_delta": 162,
        "anchor_delta_rolling": 432,
    }
    field = PSYCHE_D_SOURCE_FIELDS[0]
    row = samples.loc[samples["participant_id"].eq("person_0")].iloc[0]
    assert row["feature_month_slot"] == 2
    assert row[f"{field}__anchor"] == pytest.approx(2.0)
    assert row[f"{field}__delta_1"] == pytest.approx(1.0)
    assert row[f"{field}__delta_2"] == pytest.approx(2.0)
    assert row[f"{field}__rolling_mean_3"] == pytest.approx(1.0)
    assert row[f"{field}__rolling_std_3"] == pytest.approx(np.std([0.0, 1.0, 2.0]))
    assert row[f"{field}__rolling_slope_3"] == pytest.approx(1.0)
    assert row[f"{field}__history_available_count"] == 3
    assert row[f"{field}__months_since_last_valid"] == 0


def test_target_and_future_mutations_cannot_change_history_features(
    tmp_path: Path,
) -> None:
    participants = ["alpha", "beta"]
    mapping = _mapping(tmp_path / "mapping.yaml")
    split = _split(tmp_path / "split.json", participants)
    source = _source(participants)
    before = build_history_forecast_samples(
        source, task_id="forecast_1m", mapping_path=mapping, split_path=split
    )
    mutated = source.copy()
    mutated.loc["alpha_3", list(PSYCHE_D_SOURCE_FIELDS)] = 999999.0
    mutated.loc["alpha_4", list(PSYCHE_D_SOURCE_FIELDS)] = -999999.0
    mutated.loc["alpha_3", "phq9_score_end"] = 27.0
    after = build_history_forecast_samples(
        mutated, task_id="forecast_1m", mapping_path=mapping, split_path=split
    )
    features = history_feature_groups(PSYCHE_D_SOURCE_FIELDS)["anchor_delta_rolling"]
    pd.testing.assert_frame_equal(before[list(features)], after[list(features)])


def test_missing_history_stays_null_and_has_explicit_masks(tmp_path: Path) -> None:
    participants = ["alpha", "beta"]
    source = _source(participants)
    field = PSYCHE_D_SOURCE_FIELDS[0]
    source.loc[["alpha_0", "alpha_1", "alpha_2"], field] = np.nan
    source.loc["alpha_3", field] = 12345.0
    samples = build_history_forecast_samples(
        source,
        task_id="forecast_1m",
        mapping_path=_mapping(tmp_path / "mapping.yaml"),
        split_path=_split(tmp_path / "split.json", participants),
    )
    row = samples.loc[samples["participant_id"].eq("alpha")].iloc[0]
    assert np.isnan(row[f"{field}__anchor"])
    assert row[f"{field}__anchor_missing"] == 1
    assert np.isnan(row[f"{field}__delta_1"])
    assert row[f"{field}__history_available_count"] == 0
    assert np.isnan(row[f"{field}__months_since_last_valid"])
    assert row[f"{field}__months_since_last_valid_missing"] == 1


def test_auxiliary_train_values_are_strict_nested_oof(tmp_path: Path) -> None:
    samples = _build(tmp_path)
    assignments = {
        participant: index % 5
        for index, participant in enumerate(
            sorted(samples["global_participant_id"].unique())
        )
    }
    features = history_feature_groups(PSYCHE_D_SOURCE_FIELDS)["anchor_only"]
    train, validation = strict_auxiliary_inner_views(
        samples, assignments, features, "phq9_score", validation_fold=0
    )
    assert train["__auxiliary_prediction"].notna().all()
    assert validation["__auxiliary_prediction"].notna().all()
    assert set(train["global_participant_id"]).isdisjoint(
        set(validation["global_participant_id"])
    )


def test_candidate_selection_generates_complete_inner_oof(tmp_path: Path) -> None:
    samples = _build(tmp_path)
    assignments = {
        participant: index % 5
        for index, participant in enumerate(
            sorted(samples["global_participant_id"].unique())
        )
    }
    features = history_feature_groups(PSYCHE_D_SOURCE_FIELDS)["anchor_only"]
    candidate = SelectionCandidate(
        feature_group="anchor_only",
        model_family="elasticnet_logistic",
        params={"C": 0.1, "l1_ratio": 0.0},
        auxiliary_mode="none",
    )
    predictions, metrics = evaluate_inner_candidate(
        samples, assignments, features, candidate
    )
    assert len(predictions) == len(samples)
    assert predictions["target_window_id"].is_unique
    assert set(predictions["inner_fold_id"]) == set(range(5))
    assert 0.0 <= metrics["auprc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_frozen_source_history_counts_and_feature_boundaries() -> None:
    source_root = WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D"
    source = pd.read_parquet(source_root / "anon_processed_df_parquet")
    mapping = source_root / "p0_feature_mapping_v1.yaml"
    split = (
        ALGORITHM_ROOT
        / "data"
        / "processed"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "splits"
        / "split_manifest.json"
    )
    one = build_history_forecast_samples(
        source,
        task_id="forecast_1m",
        mapping_path=mapping,
        split_path=split,
        expected_count=9393,
    )
    two = build_history_forecast_samples(
        source,
        task_id="forecast_2m",
        mapping_path=mapping,
        split_path=split,
        expected_count=9280,
    )
    assert one["global_participant_id"].nunique() == 3635
    assert two["global_participant_id"].nunique() == 3593
    assert (
        one["feature_month_slot"] == one["label_month_slot"] - one["nominal_gap_months"]
    ).all()
    assert (
        two["feature_month_slot"] == two["label_month_slot"] - two["nominal_gap_months"]
    ).all()
    all_features = history_feature_groups(PSYCHE_D_SOURCE_FIELDS)[
        "anchor_delta_rolling"
    ]
    assert len(all_features) == 432
    assert not any("phq9" in name.lower() for name in all_features)
