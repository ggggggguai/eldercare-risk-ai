"""OPT-WORKPOINT-001 artifact and replay checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.workpoint_v3_3_4 import (
    load_workpoint_config,
    select_workpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/mood_social_workpoint_v3_3_4.yaml"
REPORT = ROOT / "reports/mental_health/mood_social/MH-20260804-022"
MANIFEST = (
    ROOT
    / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-022/competition_workpoint_manifest.json"
)


def test_workpoint_artifact_is_competition_demo_only() -> None:
    _, payload, _ = load_workpoint_config(CONFIG, repository_root=ROOT)
    assert payload["scope"] == "competition_demo"
    assert payload["production_threshold_selected"] is False
    result = select_workpoint(CONFIG, repository_root=ROOT, overwrite=True)
    assert result["selected_source"] == "current_online_baseline"
    assert result["selected_threshold"] == 0.48


def test_workpoint_curves_cover_both_sources_and_all_grid_points() -> None:
    curve = pd.read_parquet(REPORT / "threshold_curve.parquet")
    assert set(curve["source"]) == {"current_online_baseline", "v334_candidate"}
    assert curve.groupby("source").size().to_dict() == {
        "current_online_baseline": 99,
        "v334_candidate": 99,
    }
    assert curve["threshold"].between(0.01, 0.99).all()


def test_replay_proves_smoothing_upgrade_hysteresis_and_interruption() -> None:
    replay = pd.read_parquet(REPORT / "stability_replay.parquet").set_index("case")
    assert int(replay.loc["one_day_spike", "stabilized_level"]) == 0
    assert int(replay.loc["upgrade_after_two_days", "stabilized_level"]) == 1
    assert replay.loc["evidence_interruption", "interruption_result_available"] is False
    assert (
        replay.loc["evidence_interruption", "interruption_evidence_status"]
        == "insufficient_evidence"
    )
    assert int(replay.loc["downgrade_hysteresis", "stabilized_level"]) == 1


def test_manifest_binds_report_and_never_marks_production_threshold() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["task_id"] == "OPT-WORKPOINT-001"
    assert manifest["scope"] == "competition_demo"
    assert manifest["production_threshold_selected"] is False
    assert manifest["selected_source"] == "current_online_baseline"
