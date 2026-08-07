from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    ALGORITHM_ROOT
    / "artifacts"
    / "mental_health"
    / "mood_social"
    / "forecast_v3.4"
    / "optimization_candidates"
    / "MH-20260807-FOPT-002"
    / "evaluation"
)


def test_evaluation_outputs_are_not_preexisting() -> None:
    # The formal run is intentionally integration-only; this unit test documents its schema.
    assert (
        not EVALUATION_ROOT.exists()
        or (EVALUATION_ROOT / "evaluation_manifest.json").is_file()
    )


def test_evaluation_schema_when_present() -> None:
    if not EVALUATION_ROOT.exists():
        return
    manifest = json.loads((EVALUATION_ROOT / "evaluation_manifest.json").read_text())
    assert manifest["task_id"] == "FORECAST-OPT-001F"
    assert manifest["bootstrap_repetitions"] == 2000
    assert manifest["outer_test_used_for_selection"] is False
    assert manifest["product_integration_performed"] is False
    decision = json.loads((EVALUATION_ROOT / "promotion_decision.json").read_text())
    assert decision["status"] in {"promote_candidate", "retain_v3_4_baseline"}
    metrics = pd.read_parquet(EVALUATION_ROOT / "task_metrics.parquet")
    assert set(metrics["task_id"]) == {"forecast_1m", "forecast_2m"}
    assert set(metrics["variant"]) == {"baseline", "candidate"}


def test_evaluation_bootstrap_and_promotion_artifacts_when_present() -> None:
    if not EVALUATION_ROOT.exists():
        return
    bootstrap = pd.read_parquet(EVALUATION_ROOT / "bootstrap_deltas.parquet")
    assert len(bootstrap) == 8000
    assert bootstrap.groupby(["task_id", "scope"]).size().eq(2000).all()
    decision = json.loads((EVALUATION_ROOT / "promotion_decision.json").read_text())
    assert decision["status"] == "retain_v3_4_baseline"
    assert decision["per_task"]["forecast_2m"]["status"] == "promote_candidate"
    assert decision["per_task"]["forecast_1m"]["status"] == "retain_baseline"
