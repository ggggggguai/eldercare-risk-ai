"""Stateless offline-demo stabilization for mood-social daily results.

The policy is intentionally separate from the frozen V3 HTTP response. It
never changes model probabilities or the contract-defined ``attention_level``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from elderly_monitoring.common.config import load_yaml
from elderly_monitoring.modules.mental_health.mood_social.config import PROJECT_ROOT
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MOOD_SOCIAL_MODEL_VERSION,
)


RUNTIME_POLICY_VERSION = "mood-social-runtime-policy-v1"
RUNTIME_POLICY_SCOPE = "offline_demo"
RUNTIME_POLICY_DEPLOYMENT_STATUS = "not_production_validated"
DEFAULT_RUNTIME_POLICY_PATH = (
    PROJECT_ROOT
    / "configs/runtime/mood_social_runtime_policy_v3_3_3.yaml"
)
_EXPECTED_THRESHOLDS = (0.25, 0.45, 0.65)

EvidenceStatus = Literal[
    "sufficient",
    "insufficient_evidence",
    "strong_rule_only",
]


class RuntimePolicyConfigError(ValueError):
    """Raised when the frozen offline-demo policy configuration drifts."""


@dataclass(frozen=True)
class RuntimeWorkpoint:
    level: int
    lower_bound_inclusive: float
    upper_bound_exclusive: float | None
    scope: str


@dataclass(frozen=True)
class RuntimePolicyConfig:
    policy_version: str
    scope: str
    deployment_status: str
    expected_model_version: str
    workpoints: tuple[RuntimeWorkpoint, ...]
    smoothing_method: str
    smoothing_window_calendar_days: int
    minimum_available_days: int
    require_contiguous_days: bool
    consecutive_qualifying_days: int
    allow_level_skips: bool
    upgrade_margin: float
    downgrade_margin: float
    strong_rule_passthrough_enabled: bool
    strong_rule_allowed_target_levels: tuple[int, ...]
    strong_rule_require_reason_code: bool
    missing_day_breaks_sequence: bool
    unavailable_day_breaks_sequence: bool
    same_model_version_only: bool

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return tuple(
            point.lower_bound_inclusive for point in self.workpoints[1:]
        )  # type: ignore[return-value]

    @property
    def config_sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RuntimePolicyObservation:
    date: date
    model_version: str
    available: bool
    attention_index: float | None
    strong_rule_level: int | None = None
    strong_rule_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.date, date):
            raise TypeError("date must be a datetime.date")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("model_version must be non-empty")
        if not isinstance(self.available, bool):
            raise TypeError("available must be bool")
        if self.available:
            if self.attention_index is None:
                raise ValueError("available observations require attention_index")
            if isinstance(self.attention_index, bool) or not isinstance(
                self.attention_index, Real
            ):
                raise TypeError("attention_index must be numeric")
            value = float(self.attention_index)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("attention_index must be finite and within [0, 1]")
        elif self.attention_index is not None:
            raise ValueError("unavailable observations must not carry attention_index")
        if self.strong_rule_level is None:
            if self.strong_rule_code is not None:
                raise ValueError("strong_rule_code requires strong_rule_level")
        else:
            if isinstance(self.strong_rule_level, bool) or not isinstance(
                self.strong_rule_level, int
            ):
                raise TypeError("strong_rule_level must be an integer")
            if not isinstance(self.strong_rule_code, str) or not self.strong_rule_code:
                raise ValueError("strong rules require a non-empty reason code")


@dataclass(frozen=True)
class RuntimePolicyResult:
    policy_version: str
    scope: Literal["offline_demo"]
    deployment_status: Literal["not_production_validated"]
    config_sha256: str
    target_date: date
    model_version: str
    available: bool
    evidence_status: EvidenceStatus
    evidence_days: int
    raw_attention_index: float | None
    smoothed_attention_index: float | None
    raw_level: int | None
    smoothed_level: int | None
    stabilized_level: int | None
    strong_rule_applied: bool
    strong_rule_code: str | None
    pending_upgrade_level: int | None
    consecutive_upgrade_days: int
    reason_codes: tuple[str, ...]


def load_runtime_policy_config(
    path: str | Path = DEFAULT_RUNTIME_POLICY_PATH,
) -> RuntimePolicyConfig:
    """Load and strictly validate the OPT-RUNTIME-001 offline-demo policy."""

    source = Path(path)
    root = _mapping(load_yaml(source), str(source))
    _exact_keys(
        root,
        {"policy", "workpoints", "smoothing", "escalation", "hysteresis", "strong_rules", "evidence"},
        str(source),
    )
    policy = _mapping(root["policy"], f"{source}: policy")
    _exact_keys(
        policy,
        {"policy_version", "scope", "deployment_status", "expected_model_version"},
        f"{source}: policy",
    )

    raw_workpoints = root["workpoints"]
    if not isinstance(raw_workpoints, list):
        raise RuntimePolicyConfigError(f"{source}: workpoints must be a list")
    workpoints: list[RuntimeWorkpoint] = []
    for index, raw in enumerate(raw_workpoints):
        location = f"{source}: workpoints[{index}]"
        values = _mapping(raw, location)
        _exact_keys(
            values,
            {"level", "lower_bound_inclusive", "upper_bound_exclusive", "scope"},
            location,
        )
        upper = values["upper_bound_exclusive"]
        workpoints.append(
            RuntimeWorkpoint(
                level=_integer(values["level"], f"{location}.level"),
                lower_bound_inclusive=_number(
                    values["lower_bound_inclusive"],
                    f"{location}.lower_bound_inclusive",
                ),
                upper_bound_exclusive=(
                    None
                    if upper is None
                    else _number(upper, f"{location}.upper_bound_exclusive")
                ),
                scope=_string(values["scope"], f"{location}.scope"),
            )
        )

    smoothing = _section(
        root,
        "smoothing",
        {"method", "window_calendar_days", "minimum_available_days", "require_contiguous_days"},
        source,
    )
    escalation = _section(
        root,
        "escalation",
        {"consecutive_qualifying_days", "allow_level_skips"},
        source,
    )
    hysteresis = _section(
        root,
        "hysteresis",
        {"upgrade_margin", "downgrade_margin"},
        source,
    )
    strong_rules = _section(
        root,
        "strong_rules",
        {"passthrough_enabled", "allowed_target_levels", "require_reason_code"},
        source,
    )
    evidence = _section(
        root,
        "evidence",
        {"missing_day_breaks_sequence", "unavailable_day_breaks_sequence", "same_model_version_only"},
        source,
    )
    allowed_levels = strong_rules["allowed_target_levels"]
    if not isinstance(allowed_levels, list):
        raise RuntimePolicyConfigError(
            f"{source}: strong_rules.allowed_target_levels must be a list"
        )

    config = RuntimePolicyConfig(
        policy_version=_string(policy["policy_version"], "policy.policy_version"),
        scope=_string(policy["scope"], "policy.scope"),
        deployment_status=_string(
            policy["deployment_status"], "policy.deployment_status"
        ),
        expected_model_version=_string(
            policy["expected_model_version"], "policy.expected_model_version"
        ),
        workpoints=tuple(workpoints),
        smoothing_method=_string(smoothing["method"], "smoothing.method"),
        smoothing_window_calendar_days=_integer(
            smoothing["window_calendar_days"], "smoothing.window_calendar_days"
        ),
        minimum_available_days=_integer(
            smoothing["minimum_available_days"], "smoothing.minimum_available_days"
        ),
        require_contiguous_days=_boolean(
            smoothing["require_contiguous_days"], "smoothing.require_contiguous_days"
        ),
        consecutive_qualifying_days=_integer(
            escalation["consecutive_qualifying_days"],
            "escalation.consecutive_qualifying_days",
        ),
        allow_level_skips=_boolean(
            escalation["allow_level_skips"], "escalation.allow_level_skips"
        ),
        upgrade_margin=_number(hysteresis["upgrade_margin"], "hysteresis.upgrade_margin"),
        downgrade_margin=_number(
            hysteresis["downgrade_margin"], "hysteresis.downgrade_margin"
        ),
        strong_rule_passthrough_enabled=_boolean(
            strong_rules["passthrough_enabled"], "strong_rules.passthrough_enabled"
        ),
        strong_rule_allowed_target_levels=tuple(
            _integer(value, "strong_rules.allowed_target_levels")
            for value in allowed_levels
        ),
        strong_rule_require_reason_code=_boolean(
            strong_rules["require_reason_code"], "strong_rules.require_reason_code"
        ),
        missing_day_breaks_sequence=_boolean(
            evidence["missing_day_breaks_sequence"],
            "evidence.missing_day_breaks_sequence",
        ),
        unavailable_day_breaks_sequence=_boolean(
            evidence["unavailable_day_breaks_sequence"],
            "evidence.unavailable_day_breaks_sequence",
        ),
        same_model_version_only=_boolean(
            evidence["same_model_version_only"], "evidence.same_model_version_only"
        ),
    )
    _validate_frozen_config(config, source)
    return config


def evaluate_runtime_policy(
    observations: Sequence[RuntimePolicyObservation],
    *,
    target_date: date | None = None,
    config: RuntimePolicyConfig | None = None,
) -> RuntimePolicyResult:
    """Replay daily observations and return the deterministic target-day decision."""

    policy = config or load_runtime_policy_config()
    _validate_frozen_config(policy, Path("<runtime-policy-config>"))
    if not observations and target_date is None:
        raise ValueError("target_date is required when observations are empty")
    target = target_date or max(item.date for item in observations)
    if not isinstance(target, date):
        raise TypeError("target_date must be a datetime.date")
    relevant = sorted(
        (
            item
            for item in observations
            if item.model_version == policy.expected_model_version
            and item.date <= target
        ),
        key=lambda item: item.date,
    )
    relevant_dates = [item.date for item in relevant]
    if len(relevant_dates) != len(set(relevant_dates)):
        raise ValueError("same-model observations must not repeat a date")

    ignored_other_versions = any(
        item.model_version != policy.expected_model_version and item.date <= target
        for item in observations
    )
    by_date = {item.date: item for item in relevant}
    if not relevant:
        return _insufficient_result(
            target,
            policy,
            None,
            ignored_other_versions=ignored_other_versions,
        )

    trailing: deque[float] = deque(maxlen=policy.smoothing_window_calendar_days)
    previous_date: date | None = None
    stable_level: int | None = None
    pending_level: int | None = None
    pending_days = 0
    target_result: RuntimePolicyResult | None = None

    first_date = relevant[0].date
    day = first_date
    while day <= target:
        observation = by_date.get(day)
        if observation is None:
            trailing.clear()
            pending_level = None
            pending_days = 0
            if day == target:
                target_result = _insufficient_result(
                    target,
                    policy,
                    None,
                    ignored_other_versions=ignored_other_versions,
                )
            day += timedelta(days=1)
            previous_date = None
            continue

        if previous_date is None or observation.date != previous_date + timedelta(days=1):
            trailing.clear()
            pending_level = None
            pending_days = 0
        previous_date = observation.date

        raw_index = (
            float(observation.attention_index) if observation.available else None
        )
        raw_level = _level(raw_index, policy.thresholds) if raw_index is not None else None

        if observation.available:
            trailing.append(raw_index)  # type: ignore[arg-type]
        else:
            trailing.clear()
            pending_level = None
            pending_days = 0

        if observation.strong_rule_level is not None:
            if not policy.strong_rule_passthrough_enabled:
                raise ValueError("strong-rule input is disabled by policy")
            if observation.strong_rule_level not in policy.strong_rule_allowed_target_levels:
                raise ValueError("strong_rule_level is not allowed by policy")
            stable_level = max(stable_level or 0, observation.strong_rule_level)
            smoothed = (
                sum(trailing) / len(trailing)
                if len(trailing) == policy.smoothing_window_calendar_days
                else None
            )
            target_result = _result(
                policy=policy,
                target_date=observation.date,
                available=True,
                evidence_status=("sufficient" if smoothed is not None else "strong_rule_only"),
                evidence_days=len(trailing),
                raw_attention_index=raw_index,
                smoothed_attention_index=smoothed,
                raw_level=raw_level,
                smoothed_level=(
                    _level(smoothed, policy.thresholds) if smoothed is not None else None
                ),
                stabilized_level=stable_level,
                strong_rule_applied=True,
                strong_rule_code=observation.strong_rule_code,
                pending_upgrade_level=None,
                consecutive_upgrade_days=0,
                reason_codes=_reasons(
                    "strong_rule_passthrough",
                    "ignored_other_model_versions" if ignored_other_versions else None,
                ),
            )
            pending_level = None
            pending_days = 0
        elif len(trailing) < policy.minimum_available_days:
            target_result = _insufficient_result(
                observation.date,
                policy,
                observation,
                evidence_days=len(trailing),
                ignored_other_versions=ignored_other_versions,
            )
        else:
            smoothed = sum(trailing) / len(trailing)
            smoothed_level = _level(smoothed, policy.thresholds)
            reason: str
            if stable_level is None:
                stable_level = smoothed_level
                pending_level = None
                pending_days = 0
                reason = "initialized_from_smoothed_window"
            else:
                upgrade_target = _upgrade_target(
                    smoothed,
                    raw_index,
                    stable_level,
                    policy,
                )
                if upgrade_target > stable_level:
                    if pending_level == upgrade_target:
                        pending_days += 1
                    else:
                        pending_level = upgrade_target
                        pending_days = 1
                    if pending_days >= policy.consecutive_qualifying_days:
                        stable_level = upgrade_target
                        pending_level = None
                        pending_days = 0
                        reason = "upgraded_after_consecutive_days"
                    else:
                        reason = "upgrade_pending"
                else:
                    pending_level = None
                    pending_days = 0
                    downgrade_target = _downgrade_target(smoothed, stable_level, policy)
                    if downgrade_target < stable_level:
                        stable_level = downgrade_target
                        reason = "downgraded_after_hysteresis"
                    elif smoothed_level > stable_level:
                        reason = "held_by_upgrade_hysteresis"
                    elif smoothed_level < stable_level:
                        reason = "held_by_downgrade_hysteresis"
                    else:
                        reason = "stable"
            target_result = _result(
                policy=policy,
                target_date=observation.date,
                available=True,
                evidence_status="sufficient",
                evidence_days=len(trailing),
                raw_attention_index=raw_index,
                smoothed_attention_index=smoothed,
                raw_level=raw_level,
                smoothed_level=smoothed_level,
                stabilized_level=stable_level,
                strong_rule_applied=False,
                strong_rule_code=None,
                pending_upgrade_level=pending_level,
                consecutive_upgrade_days=pending_days,
                reason_codes=_reasons(
                    reason,
                    "ignored_other_model_versions" if ignored_other_versions else None,
                ),
            )
        day += timedelta(days=1)

    if target_result is None or target_result.target_date != target:
        return _insufficient_result(
            target,
            policy,
            by_date.get(target),
            ignored_other_versions=ignored_other_versions,
        )
    return target_result


def _upgrade_target(
    smoothed_value: float,
    current_value: float,
    stable_level: int,
    config: RuntimePolicyConfig,
) -> int:
    target = stable_level
    for level, threshold in enumerate(config.thresholds, start=1):
        boundary = threshold + config.upgrade_margin
        if (
            level > stable_level
            and smoothed_value >= boundary
            and current_value >= boundary
        ):
            target = level
    if not config.allow_level_skips and target > stable_level + 1:
        return stable_level + 1
    return target


def _downgrade_target(
    value: float,
    stable_level: int,
    config: RuntimePolicyConfig,
) -> int:
    target = 0
    for level, threshold in enumerate(config.thresholds, start=1):
        if value >= threshold - config.downgrade_margin:
            target = level
    return min(stable_level, target)


def _level(value: float, thresholds: tuple[float, float, float]) -> int:
    first, second, third = thresholds
    if value < first:
        return 0
    if value < second:
        return 1
    if value < third:
        return 2
    return 3


def _insufficient_result(
    target_date: date,
    policy: RuntimePolicyConfig,
    observation: RuntimePolicyObservation | None,
    *,
    evidence_days: int = 0,
    ignored_other_versions: bool = False,
) -> RuntimePolicyResult:
    raw_index = (
        float(observation.attention_index)
        if observation is not None and observation.available
        else None
    )
    return _result(
        policy=policy,
        target_date=target_date,
        available=False,
        evidence_status="insufficient_evidence",
        evidence_days=evidence_days,
        raw_attention_index=raw_index,
        smoothed_attention_index=None,
        raw_level=_level(raw_index, policy.thresholds) if raw_index is not None else None,
        smoothed_level=None,
        stabilized_level=None,
        strong_rule_applied=False,
        strong_rule_code=None,
        pending_upgrade_level=None,
        consecutive_upgrade_days=0,
        reason_codes=_reasons(
            "insufficient_contiguous_days" if raw_index is not None else "insufficient_current_evidence",
            "ignored_other_model_versions" if ignored_other_versions else None,
        ),
    )


def _result(
    *,
    policy: RuntimePolicyConfig,
    target_date: date,
    available: bool,
    evidence_status: EvidenceStatus,
    evidence_days: int,
    raw_attention_index: float | None,
    smoothed_attention_index: float | None,
    raw_level: int | None,
    smoothed_level: int | None,
    stabilized_level: int | None,
    strong_rule_applied: bool,
    strong_rule_code: str | None,
    pending_upgrade_level: int | None,
    consecutive_upgrade_days: int,
    reason_codes: tuple[str, ...],
) -> RuntimePolicyResult:
    return RuntimePolicyResult(
        policy_version=policy.policy_version,
        scope=RUNTIME_POLICY_SCOPE,
        deployment_status=RUNTIME_POLICY_DEPLOYMENT_STATUS,
        config_sha256=policy.config_sha256,
        target_date=target_date,
        model_version=policy.expected_model_version,
        available=available,
        evidence_status=evidence_status,
        evidence_days=evidence_days,
        raw_attention_index=raw_attention_index,
        smoothed_attention_index=smoothed_attention_index,
        raw_level=raw_level,
        smoothed_level=smoothed_level,
        stabilized_level=stabilized_level,
        strong_rule_applied=strong_rule_applied,
        strong_rule_code=strong_rule_code,
        pending_upgrade_level=pending_upgrade_level,
        consecutive_upgrade_days=consecutive_upgrade_days,
        reason_codes=reason_codes,
    )


def _validate_frozen_config(config: RuntimePolicyConfig, source: Path) -> None:
    expected_ranges = (
        (0, 0.0, 0.25),
        (1, 0.25, 0.45),
        (2, 0.45, 0.65),
        (3, 0.65, None),
    )
    actual_ranges = tuple(
        (point.level, point.lower_bound_inclusive, point.upper_bound_exclusive)
        for point in config.workpoints
    )
    checks = (
        (config.policy_version == RUNTIME_POLICY_VERSION, "policy_version"),
        (config.scope == RUNTIME_POLICY_SCOPE, "scope"),
        (
            config.deployment_status == RUNTIME_POLICY_DEPLOYMENT_STATUS,
            "deployment_status",
        ),
        (config.expected_model_version == MOOD_SOCIAL_MODEL_VERSION, "model_version"),
        (actual_ranges == expected_ranges, "workpoints"),
        (
            all(point.scope == RUNTIME_POLICY_SCOPE for point in config.workpoints),
            "workpoint scopes",
        ),
        (config.thresholds == _EXPECTED_THRESHOLDS, "thresholds"),
        (config.smoothing_method == "trailing_arithmetic_mean", "smoothing method"),
        (config.smoothing_window_calendar_days == 3, "smoothing window"),
        (config.minimum_available_days == 3, "minimum evidence"),
        (config.require_contiguous_days, "contiguous-day requirement"),
        (config.consecutive_qualifying_days == 2, "upgrade persistence"),
        (config.allow_level_skips, "level-skip setting"),
        (config.upgrade_margin == 0.03, "upgrade margin"),
        (config.downgrade_margin == 0.05, "downgrade margin"),
        (config.strong_rule_passthrough_enabled, "strong-rule passthrough"),
        (config.strong_rule_allowed_target_levels == (2, 3), "strong-rule levels"),
        (config.strong_rule_require_reason_code, "strong-rule reason code"),
        (config.missing_day_breaks_sequence, "missing-day behavior"),
        (config.unavailable_day_breaks_sequence, "unavailable-day behavior"),
        (config.same_model_version_only, "model-version filtering"),
    )
    for valid, label in checks:
        if not valid:
            raise RuntimePolicyConfigError(
                f"{source}: frozen offline-demo {label} has drifted"
            )


def _section(
    root: Mapping[str, Any],
    key: str,
    expected_keys: set[str],
    source: Path,
) -> Mapping[str, Any]:
    location = f"{source}: {key}"
    values = _mapping(root[key], location)
    _exact_keys(values, expected_keys, location)
    return values


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimePolicyConfigError(f"{location} must be a mapping")
    return value


def _exact_keys(values: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(values) != expected:
        raise RuntimePolicyConfigError(
            f"{location} keys must be exactly {sorted(expected)}"
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimePolicyConfigError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimePolicyConfigError(f"{location} must be an integer")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimePolicyConfigError(f"{location} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimePolicyConfigError(f"{location} must be finite")
    return parsed


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimePolicyConfigError(f"{location} must be boolean")
    return value


def _reasons(*values: str | None) -> tuple[str, ...]:
    return tuple(value for value in values if value is not None)


__all__ = [
    "DEFAULT_RUNTIME_POLICY_PATH",
    "RUNTIME_POLICY_DEPLOYMENT_STATUS",
    "RUNTIME_POLICY_SCOPE",
    "RUNTIME_POLICY_VERSION",
    "RuntimePolicyConfig",
    "RuntimePolicyConfigError",
    "RuntimePolicyObservation",
    "RuntimePolicyResult",
    "RuntimeWorkpoint",
    "evaluate_runtime_policy",
    "load_runtime_policy_config",
]
