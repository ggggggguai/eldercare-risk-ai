"""OPT-RUNTIME-001 deterministic sequence and boundary tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from elderly_monitoring.modules.mental_health.mood_social.runtime_policy import (
    DEFAULT_RUNTIME_POLICY_PATH,
    RUNTIME_POLICY_DEPLOYMENT_STATUS,
    RUNTIME_POLICY_SCOPE,
    RuntimePolicyConfigError,
    RuntimePolicyObservation,
    evaluate_runtime_policy,
    load_runtime_policy_config,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MOOD_SOCIAL_MODEL_VERSION,
)


START = date(2026, 7, 1)


def _observation(
    offset: int,
    value: float | None,
    *,
    model_version: str = MOOD_SOCIAL_MODEL_VERSION,
    strong_rule_level: int | None = None,
    strong_rule_code: str | None = None,
) -> RuntimePolicyObservation:
    return RuntimePolicyObservation(
        date=START + timedelta(days=offset),
        model_version=model_version,
        available=value is not None,
        attention_index=value,
        strong_rule_level=strong_rule_level,
        strong_rule_code=strong_rule_code,
    )


def _series(values: list[float | None]) -> tuple[RuntimePolicyObservation, ...]:
    return tuple(_observation(index, value) for index, value in enumerate(values))


def test_config_freezes_offline_demo_scope_and_four_workpoints() -> None:
    config = load_runtime_policy_config()
    assert config.scope == RUNTIME_POLICY_SCOPE == "offline_demo"
    assert config.deployment_status == RUNTIME_POLICY_DEPLOYMENT_STATUS
    assert config.expected_model_version == MOOD_SOCIAL_MODEL_VERSION
    assert config.thresholds == (0.25, 0.45, 0.65)
    assert [point.level for point in config.workpoints] == [0, 1, 2, 3]
    assert {point.scope for point in config.workpoints} == {"offline_demo"}
    assert len(config.config_sha256) == 64
    with pytest.raises(RuntimePolicyConfigError, match="scope"):
        evaluate_runtime_policy(
            _series([0.2, 0.2, 0.2]),
            config=replace(config, scope="production"),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("policy", "scope"), "production"),
        (("policy", "deployment_status"), "production_validated"),
        (("workpoints", 1, "scope"), "production"),
        (("workpoints", 1, "lower_bound_inclusive"), 0.24),
        (("smoothing", "window_calendar_days"), 2),
    ],
)
def test_config_rejects_scope_threshold_and_policy_drift(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    payload = yaml.safe_load(DEFAULT_RUNTIME_POLICY_PATH.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    drifted = tmp_path / "runtime-policy.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimePolicyConfigError, match="drifted"):
        load_runtime_policy_config(drifted)


def test_three_day_trailing_mean_suppresses_a_current_day_spike() -> None:
    result = evaluate_runtime_policy(_series([0.10, 0.40, 0.70]))
    assert result.available is True
    assert result.raw_attention_index == pytest.approx(0.70)
    assert result.raw_level == 3
    assert result.smoothed_attention_index == pytest.approx(0.40)
    assert result.smoothed_level == 1
    assert result.stabilized_level == 1
    assert result.reason_codes == ("initialized_from_smoothed_window",)


def test_upgrade_requires_two_consecutive_qualifying_policy_days() -> None:
    pending = evaluate_runtime_policy(_series([0.20, 0.20, 0.20, 0.35, 0.35]))
    assert pending.stabilized_level == 0
    assert pending.pending_upgrade_level == 1
    assert pending.consecutive_upgrade_days == 1
    assert pending.reason_codes == ("upgrade_pending",)

    upgraded = evaluate_runtime_policy(
        _series([0.20, 0.20, 0.20, 0.35, 0.35, 0.35])
    )
    assert upgraded.stabilized_level == 1
    assert upgraded.pending_upgrade_level is None
    assert upgraded.consecutive_upgrade_days == 0
    assert upgraded.reason_codes == ("upgraded_after_consecutive_days",)


def test_one_day_spike_does_not_escalate() -> None:
    result = evaluate_runtime_policy(
        _series([0.20, 0.20, 0.20, 0.80, 0.20, 0.20])
    )
    assert result.smoothed_attention_index == pytest.approx(0.40)
    assert result.stabilized_level == 0
    assert result.consecutive_upgrade_days == 0
    assert result.raw_attention_index == pytest.approx(0.20)


def test_upgrade_margin_and_downgrade_margin_form_hysteresis_bands() -> None:
    upgrade_held = evaluate_runtime_policy(
        _series([0.20, 0.20, 0.20, 0.27, 0.27, 0.27])
    )
    assert upgrade_held.smoothed_level == 1
    assert upgrade_held.stabilized_level == 0
    assert upgrade_held.reason_codes == ("held_by_upgrade_hysteresis",)

    downgrade_held = evaluate_runtime_policy(
        _series([0.50, 0.50, 0.50, 0.41, 0.41, 0.41])
    )
    assert downgrade_held.smoothed_level == 1
    assert downgrade_held.stabilized_level == 2
    assert downgrade_held.reason_codes == ("held_by_downgrade_hysteresis",)

    downgraded = evaluate_runtime_policy(
        _series([0.50, 0.50, 0.50, 0.39, 0.39, 0.39])
    )
    assert downgraded.stabilized_level == 1
    assert downgraded.reason_codes == ("downgraded_after_hysteresis",)


def test_missing_or_unavailable_day_breaks_the_contiguous_window() -> None:
    missing = (_observation(0, 0.4), _observation(1, 0.4), _observation(3, 0.4))
    missing_result = evaluate_runtime_policy(missing)
    assert missing_result.available is False
    assert missing_result.evidence_status == "insufficient_evidence"
    assert missing_result.evidence_days == 1
    assert missing_result.stabilized_level is None

    unavailable = evaluate_runtime_policy(_series([0.4, 0.4, None, 0.4, 0.4]))
    assert unavailable.available is False
    assert unavailable.evidence_days == 2
    assert unavailable.stabilized_level is None


def test_current_unavailable_has_explicit_insufficient_evidence_state() -> None:
    result = evaluate_runtime_policy(_series([0.5, 0.5, 0.5, None]))
    assert result.available is False
    assert result.evidence_status == "insufficient_evidence"
    assert result.raw_attention_index is None
    assert result.smoothed_attention_index is None
    assert result.stabilized_level is None
    assert "insufficient_current_evidence" in result.reason_codes


def test_strong_rule_passthrough_never_changes_the_raw_probability() -> None:
    available = (
        _observation(0, 0.10),
        _observation(1, 0.10),
        _observation(
            2,
            0.10,
            strong_rule_level=3,
            strong_rule_code="confirmed_urgent_event",
        ),
    )
    result = evaluate_runtime_policy(available)
    assert result.raw_attention_index == pytest.approx(0.10)
    assert result.raw_level == 0
    assert result.stabilized_level == 3
    assert result.strong_rule_applied is True
    assert result.reason_codes == ("strong_rule_passthrough",)

    unavailable = evaluate_runtime_policy(
        (
            _observation(
                0,
                None,
                strong_rule_level=2,
                strong_rule_code="explicit_safety_rule",
            ),
        )
    )
    assert unavailable.available is True
    assert unavailable.evidence_status == "strong_rule_only"
    assert unavailable.raw_attention_index is None
    assert unavailable.stabilized_level == 2


def test_other_model_versions_are_filtered_and_reported() -> None:
    observations = (
        _observation(0, 0.3),
        _observation(1, 0.9, model_version="mood-fusion-v3.3.2"),
        _observation(1, 0.3),
        _observation(2, 0.3),
    )
    result = evaluate_runtime_policy(observations)
    assert result.available is True
    assert result.smoothed_attention_index == pytest.approx(0.3)
    assert "ignored_other_model_versions" in result.reason_codes


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (0.249999, 0), (0.25, 1), (0.45, 2), (0.65, 3), (1.0, 3)],
)
def test_frozen_workpoint_boundaries(value: float, expected: int) -> None:
    result = evaluate_runtime_policy(_series([value, value, value]))
    assert result.raw_level == expected
    assert result.smoothed_level == expected
    assert result.stabilized_level == expected


def test_evaluation_is_idempotent_and_inputs_are_immutable() -> None:
    observations = _series([0.2, 0.3, 0.4])
    first = evaluate_runtime_policy(observations)
    second = evaluate_runtime_policy(observations)
    assert first == second
    assert observations == _series([0.2, 0.3, 0.4])
    with pytest.raises(FrozenInstanceError):
        observations[0].attention_index = 0.9  # type: ignore[misc]


def test_runtime_policy_has_no_forbidden_training_or_http_dependencies() -> None:
    source = Path(
        "src/elderly_monitoring/modules/mental_health/mood_social/runtime_policy.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    assert "dataset_id" not in lowered
    assert "phq-9" not in lowered
    assert "model-006" not in lowered
    assert "scorecard" not in lowered
    assert "moodsocialinferresponse" not in source


def test_observation_rejects_probability_and_strong_rule_contract_drift() -> None:
    with pytest.raises(ValueError, match="within"):
        _observation(0, 1.01)
    with pytest.raises(TypeError, match="numeric"):
        _observation(0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason code"):
        _observation(0, 0.2, strong_rule_level=3)
    with pytest.raises(ValueError, match="not allowed"):
        evaluate_runtime_policy(
            (_observation(0, 0.2, strong_rule_level=1, strong_rule_code="weak"),)
        )
