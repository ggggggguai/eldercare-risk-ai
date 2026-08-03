from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social import (
    EVALUATION_VERSION,
    MoodSocialEvaluationError,
    run_expert_evaluation,
)
from elderly_monitoring.modules.mental_health.mood_social.evaluation import (
    EXPERT_ORDER,
    LOSO_EXPERT_ORDER,
    binary_metric_summary,
    calibration_sample_weights,
    cross_fit_calibrator,
    load_evaluation_config,
    threshold_curve,
    validate_baseline_protection,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "evaluation" / "mood_social_expert_audit_v3_3_3.yaml"
)
REPORT_DIR = (
    REPOSITORY_ROOT / "reports" / "mental_health" / "mood_social" / "MH-20260801-006"
)
VARIANTS = (
    "raw",
    "active_calibrated",
    "cf_platt__frozen_four_level",
    "cf_platt__natural_sample",
    "cf_platt__target_prior_0p05",
    "cf_platt__target_prior_0p10",
    "cf_platt__target_prior_0p20",
    "cf_platt__target_prior_0p30",
    "cf_isotonic__frozen_four_level",
    "cf_isotonic__natural_sample",
    "cf_isotonic__target_prior_0p05",
    "cf_isotonic__target_prior_0p10",
    "cf_isotonic__target_prior_0p20",
    "cf_isotonic__target_prior_0p30",
)


def _calibration_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for index, target in enumerate((0, 0, 1, 1)):
            rows.append(
                {
                    "dataset_id": "a" if index % 2 == 0 else "b",
                    "global_participant_id": f"participant-{fold}-{index}",
                    "binary_target": target,
                    "outer_fold": fold,
                    "raw_probability": 0.08 + 0.18 * index + 0.01 * fold,
                }
            )
    return pd.DataFrame(rows)


def test_evaluation_api_is_exported() -> None:
    assert EVALUATION_VERSION == "mood-social-expert-audit-v3.3.3-v1"
    assert callable(run_expert_evaluation)


def test_config_freezes_read_only_scope_and_all_candidate_axes() -> None:
    config = load_evaluation_config(CONFIG_PATH)

    assert tuple(config["baseline_protection"]) == EXPERT_ORDER
    assert tuple(config["leave_one_source_out"]["experts"]) == LOSO_EXPERT_ORDER
    assert config["calibration"]["methods"] == ["platt", "isotonic"]
    assert config["calibration"]["weight_schemes"] == [
        "frozen_four_level",
        "natural_sample",
    ]
    assert config["calibration"]["target_prior_sensitivity"] == [
        0.05,
        0.1,
        0.2,
        0.3,
    ]
    assert not any(
        config["reporting"][field]
        for field in (
            "choose_production_threshold",
            "modify_attention_level_boundaries",
            "publish_diagnostic_models",
            "overwrite_active_artifacts",
        )
    )


def test_config_rejects_threshold_or_active_artifact_mutation(tmp_path: Path) -> None:
    payload = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "choose_production_threshold: false",
        "choose_production_threshold: true",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text(payload, encoding="utf-8")

    with pytest.raises(MoodSocialEvaluationError, match="read-only"):
        load_evaluation_config(changed)


def test_calibration_weights_separate_training_balance_from_natural_rows() -> None:
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
    natural = calibration_sample_weights(frame, "natural_sample")
    balanced = calibration_sample_weights(frame, "frozen_four_level")
    prior = calibration_sample_weights(frame, "target_prior", target_prior=0.2)

    assert np.array_equal(natural, np.ones(len(frame)))
    assert balanced.sum() == pytest.approx(len(frame))
    assert np.average(frame["binary_target"], weights=balanced) == pytest.approx(0.5)
    assert np.average(frame["binary_target"], weights=prior) == pytest.approx(0.2)
    assert balanced[0] == pytest.approx(balanced[1])
    assert balanced[0] + balanced[1] == pytest.approx(balanced[2])


@pytest.mark.parametrize("method", ["platt", "isotonic"])
@pytest.mark.parametrize(
    ("scheme", "prior"),
    [
        ("frozen_four_level", None),
        ("natural_sample", None),
        ("target_prior", 0.2),
    ],
)
def test_cross_fitted_calibrators_cover_each_outer_fold_without_participant_overlap(
    method: str,
    scheme: str,
    prior: float | None,
) -> None:
    frame = _calibration_frame()
    probability, audits = cross_fit_calibrator(
        frame,
        method=method,
        weight_scheme=scheme,
        target_prior=prior,
    )

    assert len(probability) == len(frame)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert [item["outer_fold"] for item in audits] == [0, 1, 2, 3, 4]
    assert sum(item["test_row_count"] for item in audits) == len(frame)
    assert all(
        item["train_participant_sha256"] != item["test_participant_sha256"]
        for item in audits
    )
    if prior is not None:
        assert all(
            item["weighted_fit_prevalence"] == pytest.approx(prior) for item in audits
        )


def test_binary_metrics_and_threshold_curve_are_complete() -> None:
    target = np.array([0, 0, 1, 1], dtype="int64")
    probability = np.array([0.1, 0.4, 0.6, 0.9], dtype="float64")

    metrics = binary_metric_summary(target, probability)
    curve = threshold_curve(target, probability)

    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["confusion"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }
    assert len(curve) == len(np.unique(probability)) + 2
    assert curve["threshold"].is_monotonic_decreasing
    assert curve["true_positive"].is_monotonic_increasing
    assert curve["false_positive"].is_monotonic_increasing


def test_formal_report_preserves_all_five_frozen_baselines() -> None:
    config = load_evaluation_config(CONFIG_PATH)
    protection = validate_baseline_protection(REPOSITORY_ROOT, config)

    assert protection["split"]["sha256"] == config["split"]["sha256"]
    assert (
        protection["feature_schema"]["sha256"]
        == config["split"]["feature_schema_sha256"]
    )
    assert set(protection["experts"]) == set(EXPERT_ORDER)
    assert all(
        protection["experts"][expert]["model"]["sha256"]
        == config["baseline_protection"][expert]["model_sha256"]
        for expert in EXPERT_ORDER
    )


def test_formal_calibration_and_loso_outputs_obey_frozen_boundaries() -> None:
    candidates = pd.read_parquet(REPORT_DIR / "calibration_candidates.parquet")
    metrics = pd.read_parquet(REPORT_DIR / "metric_summary.parquet")
    loso = pd.read_parquet(REPORT_DIR / "leave_one_source_predictions.parquet")
    audits = json.loads(
        (REPORT_DIR / "leave_one_source_preprocessing.json").read_text(encoding="utf-8")
    )

    assert len(metrics) == len(EXPERT_ORDER) * len(VARIANTS)
    assert set(metrics["probability_variant"]) == set(VARIANTS)
    assert not candidates.duplicated(["expert", "prediction_id"]).any()
    assert not loso.duplicated(["expert", "prediction_id"]).any()
    assert loso["dataset_id"].eq(loso["heldout_source"]).all()
    assert loso["raw_probability"].between(0.0, 1.0).all()
    for expert, records in audits["experts"].items():
        assert len(records) == (20 if expert == "social_context" else 15)
        for record in records:
            assert record["heldout_source"] not in record["training_source_ids"]
            assert (
                record["training_participant_sha256"]
                != record["test_participant_sha256"]
            )
            assert record["target_source_adaptation"]["labels_used"] is False
            if expert in {"activity", "joint"}:
                assert (
                    record["target_source_adaptation"]["test_participants_excluded"]
                    is True
                )


def test_formal_reader_artifact_is_bounded_and_source_backed() -> None:
    artifact = json.loads((REPORT_DIR / "artifact.json").read_text(encoding="utf-8"))
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]

    assert artifact["surface"] == "report"
    assert snapshot["status"] == "ready"
    assert manifest["blocks"][0] == {
        "id": "title",
        "type": "markdown",
        "body": f"# {manifest['title']}",
    }
    assert any(block["type"] == "chart" for block in manifest["blocks"])
    assert all(source["query"]["sql"] for source in manifest["sources"])
    assert sum(len(rows) for rows in snapshot["datasets"].values()) <= 2000


def test_independent_formal_validator_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts"
                / "validate_mood_social_expert_audit_v3_3_3.py"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["status"] == "pass"
    assert payload["check_count"] >= 1500
    assert payload["probability_variant_count_per_expert"] == 14
