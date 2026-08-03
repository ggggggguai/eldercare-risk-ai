"""Focused FUSION-001 alignment and mask tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.oof_table import (
    EXPERT_ORDER,
    TREND_ORDER,
    _auroc_reliability,
    _feature_observed,
    _reliability_maps,
    load_fusion_oof_config,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/mental_health/mood_social/MH-20260802-009"


def test_config_freezes_fusion_boundary() -> None:
    config = load_fusion_oof_config(
        ROOT / "configs/training/mood_social_fusion_oof_table_v3_3_3.yaml",
        repository_root=ROOT,
    )
    assert config.payload["task_id"] == "FUSION-001"
    assert config.payload["run_id"] == "MH-20260802-009"
    assert not any(config.payload["production_boundary"].values())
    assert (
        config.payload["confidence"]["probability_representation"]
        == "current_calibrated"
    )


def test_activity_ecdf_input_uses_x_source_mask() -> None:
    row = {
        "x_source_mask": 1,
        "x_source_value": 12.0,
        "activity.activity_volume_norm": np.nan,
    }
    assert _feature_observed(row, "activity.activity_volume_norm")


def test_categorical_feature_presence_is_supported() -> None:
    row = {
        "feature_mask.social_context.marital_status": 1,
        "social_context.marital_status": "partnered",
    }
    assert _feature_observed(row, "social_context.marital_status")
    row["social_context.marital_status"] = None
    assert not _feature_observed(row, "social_context.marital_status")


def test_auroc_reliability_is_excess_over_chance() -> None:
    frame = pd.DataFrame(
        {
            "binary_target": [0, 0, 1, 1],
            "calibrated_probability": [0.1, 0.2, 0.8, 0.9],
        }
    )
    assert _auroc_reliability(frame) == 1.0


def test_auroc_reliability_single_class_is_zero() -> None:
    frame = pd.DataFrame(
        {
            "binary_target": [0, 0],
            "calibrated_probability": [0.1, 0.2],
        }
    )
    assert _auroc_reliability(frame) == 0.0


def test_source_reliability_falls_back_for_small_source() -> None:
    rows = []
    for index in range(150):
        rows.append(
            {
                "outer_fold": index % 5,
                "dataset_id": "large",
                "expert_mask": 1,
                "binary_target": index % 2,
                "calibrated_probability": 0.1 if index % 2 == 0 else 0.9,
            }
        )
    for index in range(10):
        rows.append(
            {
                "outer_fold": index % 5,
                "dataset_id": "small",
                "expert_mask": 1,
                "binary_target": index % 2,
                "calibrated_probability": 0.1 if index % 2 == 0 else 0.9,
            }
        )
    reliability, scopes = _reliability_maps(pd.DataFrame(rows))
    assert scopes[(0, "large")] == "source_excluded_outer_fold"
    assert scopes[(0, "small")] == "expert_pooled_excluded_outer_fold"
    assert 0 <= reliability[(0, "small")] <= 1


def test_formal_table_has_all_branch_probability_layers() -> None:
    table = pd.read_parquet(REPORT / "fusion_oof_table.parquet")
    assert len(table) == 22191
    for expert in EXPERT_ORDER:
        assert f"expert_{expert}_raw_probability" in table
        assert f"expert_{expert}_current_probability" in table
        assert f"expert_{expert}_confidence" in table
        assert f"expert_{expert}_expert_mask" in table
    for branch in TREND_ORDER:
        assert f"trend_{branch}_personal_change_evidence" in table
        assert f"trend_{branch}_personal_change_mask" in table
        assert f"trend_{branch}_reliability" in table


def test_unavailable_expert_rows_keep_null_probabilities_and_zero_confidence() -> None:
    table = pd.read_parquet(REPORT / "fusion_oof_table.parquet")
    for expert in EXPERT_ORDER:
        prefix = f"expert_{expert}_"
        unavailable = table[prefix + "expert_mask"].eq(0)
        assert table.loc[unavailable, prefix + "raw_probability"].isna().all()
        assert table.loc[unavailable, prefix + "current_probability"].isna().all()
        assert table.loc[unavailable, prefix + "confidence"].eq(0).all()


def test_unavailable_personal_change_rows_keep_null_evidence() -> None:
    table = pd.read_parquet(REPORT / "fusion_oof_table.parquet")
    for branch in TREND_ORDER:
        prefix = f"trend_{branch}_"
        unavailable = table[prefix + "personal_change_mask"].eq(0)
        assert table.loc[unavailable, prefix + "personal_change_evidence"].isna().all()


def test_manifest_and_diagnostics_exclude_model006_and_stacking() -> None:
    manifest = json.loads(
        (REPORT / "fusion_table_manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (REPORT / "alignment_diagnostics.json").read_text(encoding="utf-8")
    )
    assert manifest["model006_included"] is False
    assert diagnostics["model006_columns_consumed"] == []
    assert diagnostics["stacking_fitted"] is False
    assert diagnostics["threshold_selected"] is False


def test_four_mask_layers_are_distinct_columns() -> None:
    table = pd.read_parquet(REPORT / "fusion_oof_table.parquet")
    assert table["feature_mask_json"].notna().all()
    assert "day_mask_json" in table
    assert "expert_activity_expert_mask" in table
    assert "trend_activity_personal_change_mask" in table
