"""V3.3.3 PersonalTrendExpert training and deterministic runtime scoring.

The three branches estimate personal change from prior natural-day evidence.
Activity and sleep use PSYCHE-D nominal-month sequences as an explicitly
documented transfer proxy; social uses a Shenzhen marital-status proxy because
no frozen S10 longitudinal PHQ-9 cohort exists.  PHQ-9 values are targets only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
import sklearn
import yaml

from elderly_monitoring.datasets.adapters.psyche_d import (
    TrainingFoldECDF as PsycheDTrainingFoldECDF,
)
from elderly_monitoring.modules.mental_health.mood_social.evaluation import (
    load_evaluation_config,
    validate_baseline_protection,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    map_mood_social_features,
    map_mood_social_trend_days,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_DAILY_FEATURE_SPECS,
    ACTIVITY_FEATURE_SPECS,
    CAMERA_GAIT_FEATURE_SPECS,
    FEATURE_SCHEMA_VERSION,
    SLEEP_DAILY_FEATURE_SPECS,
    SOCIAL_CONTACT_DAILY_FEATURE_SPECS,
    CameraGaitDay,
    DailyMappedFeatures,
    FeatureSpec,
    RiskDirection,
)
from elderly_monitoring.modules.mental_health.mood_social.offline_auxiliary import (
    BASELINE_PROTECTION_SHA256,
    EVALUATION_CORE_SHA256,
    EVALUATION_RUN_ID,
    FEATURE_SCHEMA_SHA256,
    INNER_FOLD_COUNT,
    OUTER_FOLD_COUNT,
    SPLIT_ID,
    SPLIT_SHA256,
    OfflineAuxiliaryInputs,
    build_task_frame,
    load_offline_auxiliary_config,
    load_offline_auxiliary_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
)


TASK_ID = "TREND-001"
RUN_ID = "MH-20260801-008"
TRAINING_VERSION = "mood-social-personal-trend-v3.3.3-v1"
BUNDLE_VERSION = "mood-social-personal-trend-bundle-v1"
MANIFEST_VERSION = "mood-social-personal-trend-manifest-v1"
OOF_VERSION = "mood-social-personal-trend-oof-v1"
METRICS_VERSION = "mood-social-personal-trend-metrics-v1"
SEARCH_VERSION = "mood-social-personal-trend-search-v1"
RANDOM_SEED = 20260728
COMPONENT_ORDER = (
    "robust_deviation",
    "risk_slope",
    "isolation_forest",
    "change_point",
    "persistence",
    "history_coverage",
)
BRANCH_ORDER = ("activity", "sleep", "social")
C_VALUES = (0.01, 0.1, 1.0, 10.0)
MODEL006_MODEL_SHA256 = (
    "35b36840edf542349a128f770cb02fdcbdb87a371f606ee053ff1bb201d93835"
)
MODEL006_MANIFEST_SHA256 = (
    "134d29c0174d483c0c343e20c48c9f0205c53511100199f69c12716f3a0c8806"
)
MODEL006_REPORT_CORE_SHA256 = (
    "a576f1d0835487e32d56678e1bb3bd61cc66afa97e332c69afbc6e86f637d883"
)
_EPSILON = 1e-12
_CIRCULAR_PERIOD = 1440.0


class PersonalTrendError(RuntimeError):
    """Raised when TREND-001 invariants or protected inputs are invalid."""


@dataclass(frozen=True)
class PersonalTrendTrainingConfig:
    config_path: Path
    config_sha256: str
    payload: Mapping[str, Any]
    repository_root: Path
    report_directory: Path
    model_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class TrendObservation:
    """One domain observation on an ordered natural/proxy time position."""

    position: float
    calendar_date: date | None
    values: Mapping[str, float | None]


@dataclass(frozen=True)
class TrendComponents:
    values: tuple[float, ...]
    personal_change_mask: int
    reliability: float
    valid_history_days: int
    stable_baseline_ready: bool
    accepted_history_positions: tuple[float, ...]
    current_feature_count: int
    baseline_feature_count: int
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.values) != len(COMPONENT_ORDER):
            raise PersonalTrendError("personal trend component length changed")
        if self.personal_change_mask not in (0, 1):
            raise PersonalTrendError("personal_change_mask must be binary")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.values
        ):
            raise PersonalTrendError("personal trend components must be finite 0..1")
        if not math.isfinite(self.reliability) or not 0 <= self.reliability <= 1:
            raise PersonalTrendError("personal trend reliability must be finite 0..1")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(COMPONENT_ORDER, self.values, strict=True))


@dataclass(frozen=True)
class TrendPreprocessor:
    input_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    medians: Mapping[str, float]
    means: Mapping[str, float]
    scales: Mapping[str, float]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        input_features: Sequence[str],
    ) -> "TrendPreprocessor":
        features = tuple(input_features)
        if not features or len(set(features)) != len(features):
            raise PersonalTrendError("trend preprocessor feature order is invalid")
        missing = set(features) - set(frame.columns)
        if missing:
            raise PersonalTrendError(f"trend preprocessing columns missing: {missing}")
        selected: list[str] = []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                dtype="float64"
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            fill = float(np.median(finite))
            filled = np.where(np.isfinite(values), values, fill)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            if scale <= _EPSILON:
                continue
            selected.append(feature)
            medians[feature] = fill
            means[feature] = mean
            scales[feature] = scale
        if not selected:
            raise PersonalTrendError("trend training fold has no non-constant feature")
        return cls(features, tuple(selected), medians, means, scales)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.input_features) - set(frame.columns)
        if missing:
            raise PersonalTrendError(f"trend inference columns missing: {missing}")
        columns: list[np.ndarray] = []
        for feature in self.selected_features:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                dtype="float64"
            )
            filled = np.where(np.isfinite(values), values, self.medians[feature])
            columns.append((filled - self.means[feature]) / self.scales[feature])
        matrix = np.column_stack(columns)
        if not np.isfinite(matrix).all():
            raise PersonalTrendError("trend preprocessing produced non-finite values")
        return matrix

    def to_manifest(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "selected_features": list(self.selected_features),
            "medians": dict(self.medians),
            "means": dict(self.means),
            "scales": dict(self.scales),
        }


@dataclass(frozen=True)
class PersonalTrendBranchModel:
    branch: str
    input_features: tuple[str, ...]
    preprocessor: TrendPreprocessor
    estimator: LogisticRegression
    selected_c: float
    training_participant_count: int
    training_row_count: int
    training_participant_sha256: str
    evidence_scope: str

    def predict_probability(self, components: TrendComponents) -> float | None:
        if components.personal_change_mask == 0:
            return None
        frame = pd.DataFrame([components.as_dict()])
        probability = float(
            self.estimator.predict_proba(self.preprocessor.transform(frame))[0, 1]
        )
        if not math.isfinite(probability):
            raise PersonalTrendError("trend branch produced non-finite probability")
        return min(max(probability, 0.0), 1.0)


@dataclass(frozen=True)
class PersonalTrendBranchResult:
    branch: str
    probability: float | None
    personal_change_mask: int
    reliability: float
    components: Mapping[str, float]
    details: Mapping[str, Any]
    evidence_scope: str


@dataclass(frozen=True)
class PersonalTrendBundle:
    bundle_version: str
    training_version: str
    run_id: str
    split_id: str
    split_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    branches: Mapping[str, PersonalTrendBranchModel]
    production_activity_ecdf: PsycheDTrainingFoldECDF
    social_proxy_transfer: str

    def validate(self) -> None:
        if (
            self.bundle_version != BUNDLE_VERSION
            or self.training_version != TRAINING_VERSION
            or self.run_id != RUN_ID
            or self.split_id != SPLIT_ID
            or self.split_sha256 != SPLIT_SHA256
            or self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
            or tuple(self.branches) != BRANCH_ORDER
        ):
            raise PersonalTrendError("personal trend bundle binding changed")
        for branch, model in self.branches.items():
            if model.branch != branch or model.training_row_count <= 0:
                raise PersonalTrendError("personal trend branch model is invalid")
        if (
            self.social_proxy_transfer
            != "cross_sectional_profile_to_s10_personal_change"
        ):
            raise PersonalTrendError("social proxy transfer binding changed")

    def predict_request(
        self,
        request: MoodSocialInferRequest | Mapping[str, Any],
    ) -> Mapping[str, PersonalTrendBranchResult]:
        self.validate()
        parsed = (
            request
            if isinstance(request, MoodSocialInferRequest)
            else MoodSocialInferRequest.model_validate(request)
        )
        days = map_mood_social_trend_days(parsed)
        mapped = map_mood_social_features(parsed)
        activity_components = _runtime_activity_components(
            days,
            mapped.trend_context.camera_gait_days,
        )
        components = {
            "activity": activity_components,
            "sleep": compute_trend_components(
                _daily_observations(days, "sleep"),
                SLEEP_DAILY_FEATURE_SPECS,
            ),
            "social": compute_trend_components(
                _daily_observations(days, "social"),
                _social_direction_specs(),
            ),
        }
        results: dict[str, PersonalTrendBranchResult] = {}
        for branch in BRANCH_ORDER:
            value = components[branch]
            model = self.branches[branch]
            results[branch] = PersonalTrendBranchResult(
                branch=branch,
                probability=model.predict_probability(value),
                personal_change_mask=value.personal_change_mask,
                reliability=value.reliability,
                components=value.as_dict(),
                details=value.details,
                evidence_scope=model.evidence_scope,
            )
        return results


def compute_trend_components(
    observations: Sequence[TrendObservation],
    specs: Sequence[FeatureSpec],
) -> TrendComponents:
    """Compute the six frozen PersonalTrend components without label inputs."""

    ordered = tuple(sorted(observations, key=lambda item: item.position))
    if not ordered:
        return _unavailable_components("no_observations")
    if len({item.position for item in ordered}) != len(ordered):
        raise PersonalTrendError("trend observations repeat a time position")
    spec_map = {
        spec.name: spec
        for spec in specs
        if spec.risk_direction
        not in (RiskDirection.MODEL_LEARNED, RiskDirection.COVERAGE_ONLY)
    }
    if not spec_map:
        return _unavailable_components("no_directional_features")
    current = ordered[-1]
    history = _lookback_history(ordered[:-1], current)
    accepted = _baseline_replay(history, spec_map)
    references = _reference_map(accepted, spec_map)
    current_scores = _observation_feature_scores(current, references, spec_map)
    eligible_feature_names = tuple(
        name
        for name, reference in references.items()
        if reference["count"] >= 3
        and _finite_value(current.values.get(name)) is not None
    )
    current_feature_count = sum(
        _finite_value(current.values.get(name)) is not None for name in spec_map
    )
    if not eligible_feature_names:
        return TrendComponents(
            values=(0.0,) * len(COMPONENT_ORDER),
            personal_change_mask=0,
            reliability=0.0,
            valid_history_days=len(accepted),
            stable_baseline_ready=len(accepted) >= 7,
            accepted_history_positions=tuple(item.position for item in accepted),
            current_feature_count=int(current_feature_count),
            baseline_feature_count=0,
            details={
                "reason": "insufficient_feature_history",
                "feature_scores": {},
                "excluded_history_positions": _excluded_positions(history, accepted),
            },
        )
    robust_deviation = max(
        current_scores[name]["score"] for name in eligible_feature_names
    )
    risk_slope = _risk_slope(accepted, current, references, spec_map)
    isolation = _isolation_component(accepted, current, references, spec_map)
    change_point = _change_point_component(accepted, current, references, spec_map)
    persistence = _persistence_component(ordered, spec_map)
    coverage = min(len(accepted) / 7.0, 1.0)
    reliability = _trend_reliability(
        current,
        accepted,
        spec_map,
        robust_deviation,
        coverage,
    )
    values = tuple(
        _clip01(value)
        for value in (
            robust_deviation,
            risk_slope,
            isolation,
            change_point,
            persistence,
            coverage,
        )
    )
    return TrendComponents(
        values=values,
        personal_change_mask=1,
        reliability=reliability,
        valid_history_days=len(accepted),
        stable_baseline_ready=len(accepted) >= 7,
        accepted_history_positions=tuple(item.position for item in accepted),
        current_feature_count=int(current_feature_count),
        baseline_feature_count=len(eligible_feature_names),
        details={
            "reason": "available",
            "feature_scores": {
                name: current_scores[name] for name in eligible_feature_names
            },
            "excluded_history_positions": _excluded_positions(history, accepted),
            "baseline_policy": {
                "lookback_calendar_days": 28,
                "initial_days": 3,
                "stable_days": 7,
                "max_valid_days": 14,
                "abnormal_day_threshold": 0.6,
                "domain_summary": "maximum",
            },
        },
    )


def _unavailable_components(reason: str) -> TrendComponents:
    return TrendComponents(
        values=(0.0,) * len(COMPONENT_ORDER),
        personal_change_mask=0,
        reliability=0.0,
        valid_history_days=0,
        stable_baseline_ready=False,
        accepted_history_positions=(),
        current_feature_count=0,
        baseline_feature_count=0,
        details={"reason": reason, "feature_scores": {}},
    )


def _lookback_history(
    history: Sequence[TrendObservation],
    current: TrendObservation,
) -> tuple[TrendObservation, ...]:
    if current.calendar_date is not None:
        selected = tuple(
            item
            for item in history
            if item.calendar_date is not None
            and 1 <= (current.calendar_date - item.calendar_date).days <= 28
        )
    else:
        selected = tuple(
            item for item in history if 0 < current.position - item.position <= 28
        )
    return selected


def _baseline_replay(
    history: Sequence[TrendObservation],
    specs: Mapping[str, FeatureSpec],
) -> tuple[TrendObservation, ...]:
    accepted: list[TrendObservation] = []
    for observation in history:
        if not any(
            _finite_value(observation.values.get(name)) is not None for name in specs
        ):
            continue
        if len(accepted) < 3:
            accepted.append(observation)
            continue
        references = _reference_map(accepted[-14:], specs)
        scores = _observation_feature_scores(observation, references, specs)
        domain_score = max(
            (value["score"] for value in scores.values()),
            default=None,
        )
        if domain_score is not None and domain_score >= 0.6:
            continue
        accepted.append(observation)
        if len(accepted) > 14:
            accepted = accepted[-14:]
    return tuple(accepted[-14:])


def _excluded_positions(
    history: Sequence[TrendObservation],
    accepted: Sequence[TrendObservation],
) -> list[float]:
    accepted_ids = {id(item) for item in accepted}
    return [item.position for item in history if id(item) not in accepted_ids]


def _reference_map(
    history: Sequence[TrendObservation],
    specs: Mapping[str, FeatureSpec],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        values = [
            value
            for item in history
            for value in [_finite_value(item.values.get(name))]
            if value is not None
        ]
        if not values:
            references[name] = {"count": 0}
        elif spec.risk_direction == RiskDirection.CIRCULAR_TWO_SIDED:
            references[name] = _circular_reference(values)
        else:
            references[name] = _linear_reference(values)
    return references


def _linear_reference(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype="float64")
    center = float(np.median(array))
    lower = float(np.quantile(array, 0.10, method="linear"))
    upper = float(np.quantile(array, 0.90, method="linear"))
    mad = float(np.median(np.abs(array - center)))
    scale = max(
        1.4826 * mad,
        (upper - lower) / 2.563,
        0.05 * abs(center),
        0.05,
    )
    return {
        "count": int(array.size),
        "center": center,
        "lower": lower,
        "upper": upper,
        "scale": float(scale),
        "circular": False,
    }


def _circular_reference(values: Sequence[float]) -> dict[str, Any]:
    array = np.mod(np.asarray(values, dtype="float64"), _CIRCULAR_PERIOD)
    distances = np.asarray(
        [
            sum(_circular_distance(candidate, other) for other in array)
            for candidate in array
        ],
        dtype="float64",
    )
    center = float(array[int(np.argmin(distances))])
    radial = np.asarray(
        [_circular_distance(value, center) for value in array],
        dtype="float64",
    )
    q90 = float(np.quantile(radial, 0.90, method="linear"))
    mad = float(np.median(radial))
    phase_floor = 0.05 * _CIRCULAR_PERIOD
    scale = max(1.4826 * mad, q90 / 1.645, phase_floor, 0.05)
    return {
        "count": int(array.size),
        "center": center,
        "lower": 0.0,
        "upper": q90,
        "scale": float(scale),
        "circular": True,
    }


def _observation_feature_scores(
    observation: TrendObservation,
    references: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, FeatureSpec],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        current = _finite_value(observation.values.get(name))
        reference = references.get(name, {})
        if current is None or int(reference.get("count", 0)) < 3:
            continue
        scores[name] = _feature_deviation(current, reference, spec.risk_direction)
    return scores


def _feature_deviation(
    current: float,
    reference: Mapping[str, Any],
    direction: RiskDirection,
) -> dict[str, float]:
    center = float(reference["center"])
    scale = float(reference["scale"])
    if bool(reference["circular"]):
        risk_delta = _circular_distance(current, center)
        quantile_distance = max(risk_delta - float(reference["upper"]), 0.0)
        relative_base = _CIRCULAR_PERIOD
    else:
        delta = current - center
        if direction == RiskDirection.INCREASE:
            risk_delta = max(delta, 0.0)
            quantile_distance = max(current - float(reference["upper"]), 0.0)
        elif direction == RiskDirection.DECREASE:
            risk_delta = max(-delta, 0.0)
            quantile_distance = max(float(reference["lower"]) - current, 0.0)
        else:
            risk_delta = abs(delta)
            quantile_distance = max(
                float(reference["lower"]) - current,
                current - float(reference["upper"]),
                0.0,
            )
        relative_base = max(abs(center), 0.05)
    standardized = _clip01((risk_delta / scale) / 2.0)
    relative = _clip01((risk_delta / relative_base) / 0.50)
    quantile = _clip01(quantile_distance / scale)
    return {
        "score": max(standardized, relative, quantile),
        "standardized": standardized,
        "relative": relative,
        "quantile": quantile,
        "current": current,
        "center": center,
        "scale": scale,
    }


def _risk_slope(
    history: Sequence[TrendObservation],
    current: TrendObservation,
    references: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, FeatureSpec],
) -> float:
    scores: list[float] = []
    for name, spec in specs.items():
        reference = references.get(name, {})
        if int(reference.get("count", 0)) < 2:
            continue
        points = [
            (item.position, value)
            for item in (*history, current)
            for value in [_finite_value(item.values.get(name))]
            if value is not None
        ]
        if len(points) < 3:
            continue
        pair_slopes: list[float] = []
        for left_index, (left_x, left_y) in enumerate(points[:-1]):
            for right_x, right_y in points[left_index + 1 :]:
                spacing = right_x - left_x
                if spacing <= 0:
                    continue
                if spec.risk_direction == RiskDirection.CIRCULAR_TWO_SIDED:
                    change = _circular_distance(right_y, left_y)
                else:
                    change = right_y - left_y
                pair_slopes.append(change / spacing)
        if not pair_slopes:
            continue
        slope = float(np.median(np.asarray(pair_slopes, dtype="float64")))
        if spec.risk_direction == RiskDirection.DECREASE:
            directed = max(-slope, 0.0)
        elif spec.risk_direction == RiskDirection.INCREASE:
            directed = max(slope, 0.0)
        else:
            directed = abs(slope)
        scores.append(_clip01(directed / float(reference["scale"])))
    return max(scores, default=0.0)


def _isolation_component(
    history: Sequence[TrendObservation],
    current: TrendObservation,
    references: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, FeatureSpec],
) -> float:
    if len(history) < 7:
        return 0.0
    common = [
        name
        for name in specs
        if _finite_value(current.values.get(name)) is not None
        and all(_finite_value(item.values.get(name)) is not None for item in history)
        and int(references.get(name, {}).get("count", 0)) >= 7
    ]
    if len(common) < 2:
        return 0.0
    train = np.asarray(
        [
            [
                _signed_standardized(
                    float(item.values[name]), references[name], specs[name]
                )
                for name in common
            ]
            for item in history
        ],
        dtype="float64",
    )
    current_matrix = np.asarray(
        [
            [
                _signed_standardized(
                    float(current.values[name]), references[name], specs[name]
                )
                for name in common
            ]
        ],
        dtype="float64",
    )
    estimator = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    estimator.fit(train)
    reference_scores = -estimator.score_samples(train)
    current_score = float(-estimator.score_samples(current_matrix)[0])
    return float(
        np.searchsorted(np.sort(reference_scores), current_score, side="right")
        / len(reference_scores)
    )


def _signed_standardized(
    value: float,
    reference: Mapping[str, Any],
    spec: FeatureSpec,
) -> float:
    center = float(reference["center"])
    if spec.risk_direction == RiskDirection.CIRCULAR_TWO_SIDED:
        delta = _circular_signed_delta(value, center)
    else:
        delta = value - center
    return delta / float(reference["scale"])


def _change_point_component(
    history: Sequence[TrendObservation],
    current: TrendObservation,
    references: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, FeatureSpec],
) -> float:
    scores: list[float] = []
    for name, spec in specs.items():
        values = [
            value
            for item in (*history, current)
            for value in [_finite_value(item.values.get(name))]
            if value is not None
        ]
        if len(values) < 6 or name not in references:
            continue
        earlier = values[:-3]
        recent = values[-3:]
        if spec.risk_direction == RiskDirection.CIRCULAR_TWO_SIDED:
            earlier_center = float(_circular_reference(earlier)["center"])
            recent_center = float(_circular_reference(recent)["center"])
            shift = _circular_distance(recent_center, earlier_center)
        else:
            earlier_center = float(np.median(np.asarray(earlier, dtype="float64")))
            recent_center = float(np.median(np.asarray(recent, dtype="float64")))
            delta = recent_center - earlier_center
            if spec.risk_direction == RiskDirection.DECREASE:
                shift = max(-delta, 0.0)
            elif spec.risk_direction == RiskDirection.INCREASE:
                shift = max(delta, 0.0)
            else:
                shift = abs(delta)
        scores.append(_clip01(shift / (2.0 * float(references[name]["scale"]))))
    return max(scores, default=0.0)


def _persistence_component(
    observations: Sequence[TrendObservation],
    specs: Mapping[str, FeatureSpec],
) -> float:
    count = 0
    previous: TrendObservation | None = None
    for index in range(len(observations) - 1, -1, -1):
        observation = observations[index]
        if previous is not None and not _consecutive(observation, previous):
            break
        prior = _lookback_history(observations[:index], observation)
        accepted = _baseline_replay(prior, specs)
        scores = _observation_feature_scores(
            observation,
            _reference_map(accepted, specs),
            specs,
        )
        domain_score = max((item["score"] for item in scores.values()), default=None)
        if domain_score is None or domain_score < 0.6:
            break
        count += 1
        previous = observation
        if count >= 7:
            break
    return min(count / 7.0, 1.0)


def _consecutive(earlier: TrendObservation, later: TrendObservation) -> bool:
    if earlier.calendar_date is not None and later.calendar_date is not None:
        return (later.calendar_date - earlier.calendar_date).days == 1
    return math.isclose(later.position - earlier.position, 1.0, abs_tol=0.0)


def _trend_reliability(
    current: TrendObservation,
    accepted: Sequence[TrendObservation],
    specs: Mapping[str, FeatureSpec],
    full_deviation: float,
    coverage: float,
) -> float:
    if len(accepted) < 3:
        return 0.0
    if len(accepted) == 3:
        stability = 0.5
    else:
        leave_one_out: list[float] = []
        for index in range(len(accepted)):
            subset = (*accepted[:index], *accepted[index + 1 :])
            scores = _observation_feature_scores(
                current,
                _reference_map(subset, specs),
                specs,
            )
            if scores:
                leave_one_out.append(max(item["score"] for item in scores.values()))
        if not leave_one_out:
            stability = 0.0
        else:
            stability = 1.0 - abs(full_deviation - float(median(leave_one_out)))
    return _clip01(coverage * _clip01(stability))


def _circular_distance(left: float, right: float) -> float:
    delta = abs((left - right) % _CIRCULAR_PERIOD)
    return min(delta, _CIRCULAR_PERIOD - delta)


def _circular_signed_delta(value: float, center: float) -> float:
    return ((value - center + _CIRCULAR_PERIOD / 2.0) % _CIRCULAR_PERIOD) - (
        _CIRCULAR_PERIOD / 2.0
    )


def _finite_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _daily_observations(
    days: Sequence[DailyMappedFeatures],
    domain: str,
) -> tuple[TrendObservation, ...]:
    observations: list[TrendObservation] = []
    for item in days:
        vector = getattr(item, domain)
        values = {
            name: (
                _finite_value(vector.value(name)) if vector.mask(name) == 1 else None
            )
            for name in vector.feature_names
        }
        observations.append(
            TrendObservation(
                position=float(item.date.toordinal()),
                calendar_date=item.date,
                values=values,
            )
        )
    return tuple(observations)


def _social_direction_specs() -> tuple[FeatureSpec, ...]:
    return tuple(
        spec
        for spec in SOCIAL_CONTACT_DAILY_FEATURE_SPECS
        if spec.risk_direction != RiskDirection.MODEL_LEARNED
    )


def _runtime_activity_components(
    days: Sequence[DailyMappedFeatures],
    gait_days: Sequence[CameraGaitDay],
) -> TrendComponents:
    candidates = [
        compute_trend_components(
            _daily_observations(days, "activity"),
            ACTIVITY_DAILY_FEATURE_SPECS,
        )
    ]
    date_order = tuple(item.date for item in days)
    candidates.append(_walking_speed_components(date_order, gait_days))
    by_scene: dict[tuple[str, str], dict[date, CameraGaitDay]] = {}
    for item in gait_days:
        by_scene.setdefault((item.camera_id, item.scene_version), {})[item.date] = item
    for scene_key in sorted(by_scene):
        scene_days = by_scene[scene_key]
        non_speed_specs = tuple(
            spec
            for spec in CAMERA_GAIT_FEATURE_SPECS
            if spec.name != "gait_speed_image_norm_per_sec"
        )
        observations = tuple(
            TrendObservation(
                position=float(day.toordinal()),
                calendar_date=day,
                values={
                    spec.name: (
                        _finite_value(getattr(scene_days[day], spec.name))
                        if day in scene_days
                        else None
                    )
                    for spec in non_speed_specs
                },
            )
            for day in date_order
        )
        value = compute_trend_components(observations, non_speed_specs)
        value_details = dict(value.details)
        value_details["camera_id"] = scene_key[0]
        value_details["scene_version"] = scene_key[1]
        candidates.append(
            TrendComponents(
                values=value.values,
                personal_change_mask=value.personal_change_mask,
                reliability=value.reliability,
                valid_history_days=value.valid_history_days,
                stable_baseline_ready=value.stable_baseline_ready,
                accepted_history_positions=value.accepted_history_positions,
                current_feature_count=value.current_feature_count,
                baseline_feature_count=value.baseline_feature_count,
                details=value_details,
            )
        )
    available = [item for item in candidates if item.personal_change_mask == 1]
    if not available:
        primary = candidates[0]
        return TrendComponents(
            values=primary.values,
            personal_change_mask=0,
            reliability=0.0,
            valid_history_days=max(item.valid_history_days for item in candidates),
            stable_baseline_ready=any(
                item.stable_baseline_ready for item in candidates
            ),
            accepted_history_positions=primary.accepted_history_positions,
            current_feature_count=sum(
                item.current_feature_count for item in candidates
            ),
            baseline_feature_count=0,
            details={
                "reason": "insufficient_activity_or_scene_matched_gait_history",
                "subdomains": [item.details for item in candidates],
                "gait_partition": "camera_id_plus_scene_version",
            },
        )
    return TrendComponents(
        values=tuple(
            max(item.values[index] for item in available)
            for index in range(len(COMPONENT_ORDER))
        ),
        personal_change_mask=1,
        reliability=max(item.reliability for item in available),
        valid_history_days=max(item.valid_history_days for item in available),
        stable_baseline_ready=any(item.stable_baseline_ready for item in available),
        accepted_history_positions=max(
            (item.accepted_history_positions for item in available),
            key=len,
        ),
        current_feature_count=sum(item.current_feature_count for item in available),
        baseline_feature_count=sum(item.baseline_feature_count for item in available),
        details={
            "reason": "available",
            "subdomains": [item.details for item in candidates],
            "aggregation": "componentwise_maximum",
            "gait_partition": "camera_id_plus_scene_version",
            "walking_speed_aggregation": "q10_q90_then_same_day_scene_median",
        },
    )


def _walking_speed_components(
    date_order: Sequence[date],
    gait_days: Sequence[CameraGaitDay],
) -> TrendComponents:
    speed_spec = CAMERA_GAIT_FEATURE_SPECS[0]
    by_scene: dict[tuple[str, str], dict[date, float]] = {}
    for item in gait_days:
        speed = _finite_value(item.gait_speed_image_norm_per_sec)
        if speed is not None:
            by_scene.setdefault((item.camera_id, item.scene_version), {})[item.date] = (
                speed
            )
    normalized_by_date: dict[date, list[float]] = {day: [] for day in date_order}
    current_scene_context: list[tuple[float, float, tuple[TrendObservation, ...]]] = []
    accepted_dates: set[date] = set()
    for scene_key in sorted(by_scene):
        scene_values = by_scene[scene_key]
        observations = tuple(
            TrendObservation(
                position=float(day.toordinal()),
                calendar_date=day,
                values={speed_spec.name: scene_values.get(day)},
            )
            for day in date_order
        )
        for index, observation in enumerate(observations):
            current_speed = _finite_value(observation.values.get(speed_spec.name))
            if current_speed is None:
                continue
            prior = _lookback_history(observations[:index], observation)
            accepted = _baseline_replay(prior, {speed_spec.name: speed_spec})
            normalized = _walking_speed_normalize(current_speed, accepted, speed_spec)
            if normalized is None:
                continue
            assert observation.calendar_date is not None
            normalized_by_date[observation.calendar_date].append(normalized)
            if index == len(observations) - 1:
                current_scene_context.append((normalized, current_speed, accepted))
                accepted_dates.update(
                    item.calendar_date
                    for item in accepted
                    if item.calendar_date is not None
                )
    normalized_series = [
        (day, float(np.median(values)))
        for day in date_order
        for values in [normalized_by_date[day]]
        if values
    ]
    if (
        not current_scene_context
        or not normalized_series
        or normalized_series[-1][0] != date_order[-1]
    ):
        return _unavailable_components("walking_speed_q10_q90_unavailable")
    current_norm = normalized_series[-1][1]
    robust = _clip01(1.0 - current_norm)
    slope = _normalized_speed_slope(normalized_series)
    change_point = _normalized_speed_change_point(normalized_series)
    persistence = _normalized_speed_persistence(normalized_series)
    valid_history_days = len(accepted_dates)
    coverage = min(valid_history_days / 7.0, 1.0)
    reliability = _walking_speed_reliability(
        current_scene_context,
        robust,
        coverage,
        speed_spec,
    )
    return TrendComponents(
        values=(robust, slope, 0.0, change_point, persistence, coverage),
        personal_change_mask=1,
        reliability=reliability,
        valid_history_days=valid_history_days,
        stable_baseline_ready=valid_history_days >= 7,
        accepted_history_positions=tuple(
            float(day.toordinal()) for day in sorted(accepted_dates)
        ),
        current_feature_count=len(current_scene_context),
        baseline_feature_count=1,
        details={
            "reason": "available",
            "feature": "walking_speed_norm_camera",
            "current_normalized_scene_median": current_norm,
            "current_available_scene_count": len(current_scene_context),
            "formula": "clip((v_today-Q10_history)/(Q90_history-Q10_history),0,1)",
            "history_partition": "camera_id_plus_scene_version",
            "same_day_aggregation": "median",
            "quantile_interpolation": "linear",
            "denominator_epsilon": 1.0e-6,
        },
    )


def _walking_speed_normalize(
    current_speed: float,
    accepted: Sequence[TrendObservation],
    speed_spec: FeatureSpec,
) -> float | None:
    values = [
        value
        for item in accepted[-14:]
        for value in [_finite_value(item.values.get(speed_spec.name))]
        if value is not None
    ]
    if len(values) < 3:
        return None
    array = np.asarray(values, dtype="float64")
    lower = float(np.quantile(array, 0.10, method="linear"))
    upper = float(np.quantile(array, 0.90, method="linear"))
    denominator = upper - lower
    if denominator <= 1.0e-6:
        return None
    return _clip01((current_speed - lower) / denominator)


def _normalized_speed_slope(series: Sequence[tuple[date, float]]) -> float:
    if len(series) < 3:
        return 0.0
    slopes = [
        (right_value - left_value) / (right_day - left_day).days
        for left_index, (left_day, left_value) in enumerate(series[:-1])
        for right_day, right_value in series[left_index + 1 :]
        if (right_day - left_day).days > 0
    ]
    return _clip01(max(-float(np.median(slopes)), 0.0)) if slopes else 0.0


def _normalized_speed_change_point(series: Sequence[tuple[date, float]]) -> float:
    if len(series) < 6:
        return 0.0
    values = [value for _, value in series]
    shift = max(
        float(np.median(values[:-3])) - float(np.median(values[-3:])),
        0.0,
    )
    return _clip01(shift / 2.0)


def _normalized_speed_persistence(series: Sequence[tuple[date, float]]) -> float:
    count = 0
    later_day: date | None = None
    for day, normalized in reversed(series):
        if later_day is not None and (later_day - day).days != 1:
            break
        if 1.0 - normalized < 0.6:
            break
        count += 1
        later_day = day
        if count >= 7:
            break
    return min(count / 7.0, 1.0)


def _walking_speed_reliability(
    scene_context: Sequence[tuple[float, float, tuple[TrendObservation, ...]]],
    full_deviation: float,
    coverage: float,
    speed_spec: FeatureSpec,
) -> float:
    minimum = min(len(history) for _, _, history in scene_context)
    if minimum == 3:
        stability = 0.5
    else:
        leave_one_out: list[float] = []
        for _, current_speed, history in scene_context:
            for index in range(len(history)):
                subset = (*history[:index], *history[index + 1 :])
                normalized = _walking_speed_normalize(
                    current_speed,
                    subset,
                    speed_spec,
                )
                if normalized is not None:
                    leave_one_out.append(1.0 - normalized)
        stability = (
            0.0
            if not leave_one_out
            else 1.0 - abs(full_deviation - float(np.median(np.asarray(leave_one_out))))
        )
    return _clip01(coverage * _clip01(stability))


def load_personal_trend_config(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> PersonalTrendTrainingConfig:
    path = Path(config_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else path.parents[2]
    )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PersonalTrendError("TREND-001 config is unreadable") from exc
    if not isinstance(payload, dict):
        raise PersonalTrendError("TREND-001 config root must be a mapping")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("run_id") != RUN_ID
        or payload.get("training_version") != TRAINING_VERSION
        or payload.get("frozen_document_version") != "V3.3.3"
        or int(payload.get("random_seed", -1)) != RANDOM_SEED
    ):
        raise PersonalTrendError("TREND-001 identity binding changed")
    schema = payload.get("feature_schema", {})
    split = payload.get("split", {})
    if (
        schema.get("version") != FEATURE_SCHEMA_VERSION
        or schema.get("sha256") != FEATURE_SCHEMA_SHA256
        or split.get("split_id") != SPLIT_ID
        or split.get("sha256") != SPLIT_SHA256
        or int(split.get("outer_folds", -1)) != OUTER_FOLD_COUNT
        or int(split.get("inner_folds", -1)) != INNER_FOLD_COUNT
    ):
        raise PersonalTrendError("TREND-001 schema/split binding changed")
    baseline = payload.get("baseline", {})
    scale = baseline.get("scale", {})
    deviation = baseline.get("deviation", {})
    circular = baseline.get("circular", {})
    if (
        int(baseline.get("history_lookback_calendar_days", -1)) != 28
        or int(baseline.get("initial_days", -1)) != 3
        or int(baseline.get("stable_days", -1)) != 7
        or int(baseline.get("max_valid_days", -1)) != 14
        or float(baseline.get("lower_quantile", -1)) != 0.10
        or float(baseline.get("upper_quantile", -1)) != 0.90
        or float(baseline.get("abnormal_score_threshold", -1)) != 0.60
        or float(scale.get("mad_multiplier", -1)) != 1.4826
        or float(scale.get("quantile_divisor", -1)) != 2.563
        or float(scale.get("relative_floor_multiplier", -1)) != 0.05
        or float(scale.get("absolute_floor", -1)) != 0.05
        or float(deviation.get("standardized_divisor", -1)) != 2.0
        or float(deviation.get("relative_change_denominator", -1)) != 0.50
        or deviation.get("domain_summary") != "maximum"
        or float(circular.get("period_minutes", -1)) != 1440.0
        or circular.get("center") != "circular_medoid"
    ):
        raise PersonalTrendError("TREND-001 baseline policy changed")
    components = payload.get("components", {})
    if tuple(components.get("order", ())) != COMPONENT_ORDER:
        raise PersonalTrendError("TREND-001 component order changed")
    slope = components.get("slope", {})
    isolation = components.get("isolation_forest", {})
    change_point = components.get("change_point", {})
    if (
        slope.get("method") != "theil_sen"
        or int(slope.get("minimum_total_points", -1)) != 3
        or not bool(slope.get("use_calendar_day_spacing"))
        or not bool(isolation.get("stable_baseline_only"))
        or int(isolation.get("n_estimators", -1)) != 100
        or float(isolation.get("contamination", -1)) != 0.15
        or int(isolation.get("random_seed", -1)) != RANDOM_SEED
        or int(isolation.get("minimum_common_features", -1)) != 2
        or int(change_point.get("minimum_valid_points", -1)) != 6
        or int(change_point.get("recent_window_points", -1)) != 3
    ):
        raise PersonalTrendError("TREND-001 component policy changed")
    logistic = payload.get("logistic", {})
    if (
        tuple(float(value) for value in logistic.get("c_values", ())) != C_VALUES
        or logistic.get("solver") != "lbfgs"
        or logistic.get("penalty") != "l2"
        or int(logistic.get("max_iter", -1)) != 5000
        or float(logistic.get("tolerance", -1)) != 1.0e-8
    ):
        raise PersonalTrendError("TREND-001 Logistic search changed")
    branches = payload.get("branches", {})
    if tuple(branches) != BRANCH_ORDER:
        raise PersonalTrendError("TREND-001 branch order changed")
    social = branches["social"]
    if (
        social.get("training_dataset") != "shenzhen_elderly"
        or social.get("proxy_input") != "social_context.marital_status"
        or social.get("runtime_input") != "robust_deviation"
        or bool(social.get("direct_s10_phq9_validation"))
    ):
        raise PersonalTrendError("TREND-001 social proxy boundary changed")
    boundary = payload.get("production_boundary", {})
    forbidden_true = (
        "phq9_scores_as_inputs",
        "phq9_grades_as_inputs",
        "scale_items_as_inputs",
        "publisher_labels_as_inputs",
        "offline_auxiliary_predictions_as_inputs",
        "masks_as_risk_inputs",
        "selects_production_threshold",
        "changes_active_experts",
        "changes_http_behavior",
    )
    if any(bool(boundary.get(name)) for name in forbidden_true):
        raise PersonalTrendError("TREND-001 production boundary changed")
    output = payload.get("output", {})
    return PersonalTrendTrainingConfig(
        config_path=path,
        config_sha256=_sha256_file(path),
        payload=payload,
        repository_root=root,
        report_directory=root / str(output["report_directory"]),
        model_path=root / str(output["model_path"]),
        manifest_path=root / str(output["manifest_path"]),
    )


def load_personal_trend_inputs(
    config: PersonalTrendTrainingConfig,
) -> OfflineAuxiliaryInputs:
    root = config.repository_root
    upstream = config.payload.get("upstream", {})
    if (
        upstream.get("evaluation_run_id") != EVALUATION_RUN_ID
        or upstream.get("evaluation_core_sha256") != EVALUATION_CORE_SHA256
        or upstream.get("baseline_protection_sha256") != BASELINE_PROTECTION_SHA256
    ):
        raise PersonalTrendError("TREND-001 EVAL/baseline binding changed")
    offline_path = root / str(upstream.get("offline_config_path", ""))
    offline_config = load_offline_auxiliary_config(
        offline_path,
        repository_root=root,
    )
    inputs = load_offline_auxiliary_inputs(offline_config)
    protection_sha = _canonical_payload_sha256(inputs.upstream_protection)
    if protection_sha != BASELINE_PROTECTION_SHA256:
        raise PersonalTrendError("five-expert protection summary drifted")
    model006 = upstream.get("offline_auxiliary", {})
    model_path = root / str(model006.get("model_path", ""))
    manifest_path = root / str(model006.get("manifest_path", ""))
    report_path = root / str(model006.get("report_path", ""))
    if (
        model006.get("model_sha256") != MODEL006_MODEL_SHA256
        or model006.get("manifest_sha256") != MODEL006_MANIFEST_SHA256
        or model006.get("report_core_sha256") != MODEL006_REPORT_CORE_SHA256
        or not model_path.is_file()
        or _sha256_file(model_path) != MODEL006_MODEL_SHA256
        or not manifest_path.is_file()
        or _sha256_file(manifest_path) != MODEL006_MANIFEST_SHA256
        or _report_core_sha256(report_path) != MODEL006_REPORT_CORE_SHA256
    ):
        raise PersonalTrendError("MODEL-006 protected output drifted")
    return inputs


def _fit_psyche_activity_ecdf(
    inputs: OfflineAuxiliaryInputs,
    training_participant_ids: set[str],
) -> PsycheDTrainingFoldECDF:
    binding = inputs.input_bindings["psyche_d"]
    participants = sorted(training_participant_ids, key=lambda value: value.encode())
    frame = inputs.frames["psyche_d"]
    selected = frame["global_participant_id"].astype(str).isin(participants)
    values = pd.to_numeric(
        frame.loc[selected, "source__steps_awake_mean"], errors="coerce"
    ).to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise PersonalTrendError("activity ECDF training fold has no finite values")
    return PsycheDTrainingFoldECDF(
        split_id=SPLIT_ID,
        canonical_artifact_sha256=str(binding["canonical_sha256"]),
        canonical_frame_sha256=str(binding["canonical_frame_sha256"]),
        training_participant_sha256=_adapter_participant_set_sha256(participants),
        training_participant_count=len(participants),
        sorted_training_values=tuple(float(value) for value in np.sort(finite)),
    )


def build_personal_trend_training_frame(
    inputs: OfflineAuxiliaryInputs,
    branch: str,
    *,
    activity_ecdf_training_participant_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build label-bearing proxy rows; label columns never enter components."""

    if branch not in BRANCH_ORDER:
        raise PersonalTrendError(f"unsupported personal trend branch: {branch}")
    if branch == "social":
        if activity_ecdf_training_participant_ids is not None:
            raise PersonalTrendError("social proxy must not receive activity ECDF")
        return _build_social_proxy_frame(inputs)
    task = build_task_frame(inputs, "psyche_d")
    task = task.loc[
        :,
        [
            "canonical_row_index",
            "global_participant_id",
            "nominal_month",
            "outer_fold",
            "inner_validation_fold_by_outer_fold",
            "binary_target",
        ],
    ]
    source = inputs.frames["psyche_d"]
    if branch == "activity":
        if not activity_ecdf_training_participant_ids:
            raise PersonalTrendError("activity trend requires fold-fitted ECDF")
        ecdf = _fit_psyche_activity_ecdf(
            inputs,
            activity_ecdf_training_participant_ids,
        )
        feature_rows = _psyche_activity_feature_rows(source, ecdf)
    else:
        if activity_ecdf_training_participant_ids is not None:
            raise PersonalTrendError("sleep trend must not receive activity ECDF")
        feature_rows = _psyche_feature_rows(source, branch)
    records: list[dict[str, Any]] = []
    for participant_id, participant in task.groupby(
        "global_participant_id", sort=False
    ):
        participant = participant.sort_values("nominal_month", kind="stable")
        observations: list[TrendObservation] = []
        for sequence_index, row in enumerate(participant.itertuples(index=False)):
            row_index = int(row.canonical_row_index)
            values, specs = _psyche_observation_values(
                feature_rows[row_index],
                branch,
            )
            observations.append(
                TrendObservation(
                    position=float(sequence_index),
                    calendar_date=None,
                    values=values,
                )
            )
            if len(observations) <= 3:
                components = TrendComponents(
                    values=(0.0,) * len(COMPONENT_ORDER),
                    personal_change_mask=0,
                    reliability=0.0,
                    valid_history_days=max(len(observations) - 1, 0),
                    stable_baseline_ready=False,
                    accepted_history_positions=tuple(
                        item.position for item in observations[:-1]
                    ),
                    current_feature_count=sum(
                        value is not None for value in values.values()
                    ),
                    baseline_feature_count=0,
                    details={
                        "reason": "insufficient_feature_history",
                        "feature_scores": {},
                    },
                )
            elif len(observations) == 4:
                components = _compute_three_history_proxy_components(
                    observations,
                    specs,
                )
            else:
                components = compute_trend_components(observations, specs)
            record = {
                "branch": branch,
                "dataset_id": "psyche_d",
                "prediction_id": f"{branch}::psyche_d::row={row_index}",
                "canonical_row_index": row_index,
                "global_participant_id": str(participant_id),
                "outer_fold": int(row.outer_fold),
                "inner_validation_fold_by_outer_fold": (
                    row.inner_validation_fold_by_outer_fold
                ),
                "target": int(row.binary_target),
                "available": bool(components.personal_change_mask),
                "personal_change_mask": int(components.personal_change_mask),
                "reliability": float(components.reliability),
                "valid_history_days": int(components.valid_history_days),
                "timescale_semantics": "timescale_proxy_transfer",
            }
            record.update(components.as_dict())
            records.append(record)
    result = pd.DataFrame(records)
    if len(result) != len(task) or result["prediction_id"].duplicated().any():
        raise PersonalTrendError(f"{branch} proxy row coverage changed")
    return result.sort_values("canonical_row_index", kind="stable").reset_index(
        drop=True
    )


def _compute_three_history_proxy_components(
    observations: Sequence[TrendObservation],
    specs: Sequence[FeatureSpec],
) -> TrendComponents:
    if len(observations) != 4:
        raise PersonalTrendError("three-history proxy requires exactly four rows")
    spec_map = {
        spec.name: spec
        for spec in specs
        if spec.risk_direction
        not in (RiskDirection.MODEL_LEARNED, RiskDirection.COVERAGE_ONLY)
    }
    history = tuple(observations[:3])
    current = observations[3]
    references = _reference_map(history, spec_map)
    scores = _observation_feature_scores(current, references, spec_map)
    if not scores:
        return TrendComponents(
            values=(0.0,) * len(COMPONENT_ORDER),
            personal_change_mask=0,
            reliability=0.0,
            valid_history_days=3,
            stable_baseline_ready=False,
            accepted_history_positions=tuple(item.position for item in history),
            current_feature_count=sum(
                _finite_value(current.values.get(name)) is not None for name in spec_map
            ),
            baseline_feature_count=0,
            details={"reason": "insufficient_feature_history", "feature_scores": {}},
        )
    robust = max(item["score"] for item in scores.values())
    slope = _risk_slope(history, current, references, spec_map)
    coverage = 3.0 / 7.0
    values = (
        robust,
        slope,
        0.0,
        0.0,
        (1.0 / 7.0) if robust >= 0.6 else 0.0,
        coverage,
    )
    return TrendComponents(
        values=tuple(_clip01(value) for value in values),
        personal_change_mask=1,
        reliability=coverage * 0.5,
        valid_history_days=3,
        stable_baseline_ready=False,
        accepted_history_positions=tuple(item.position for item in history),
        current_feature_count=sum(
            _finite_value(current.values.get(name)) is not None for name in spec_map
        ),
        baseline_feature_count=len(scores),
        details={
            "reason": "available",
            "feature_scores": scores,
            "timescale_proxy": True,
        },
    )


def _psyche_observation_values(
    row: Mapping[str, Any],
    branch: str,
) -> tuple[dict[str, float | None], tuple[FeatureSpec, ...]]:
    if branch == "activity":
        specs = tuple(
            spec
            for spec in ACTIVITY_FEATURE_SPECS
            if spec.risk_direction
            not in (RiskDirection.MODEL_LEARNED, RiskDirection.COVERAGE_ONLY)
        )
        values = {
            spec.name: _masked_canonical_value(row, f"activity.{spec.name}")
            for spec in specs
        }
        return values, specs
    specs = SLEEP_DAILY_FEATURE_SPECS
    values = {
        "sleep_duration_norm": _masked_canonical_value(
            row, "sleep.sleep_duration_norm"
        ),
        "time_in_bed_norm": _masked_canonical_value(row, "sleep.time_in_bed_norm"),
        "sleep_efficiency": _masked_canonical_value(row, "sleep.sleep_efficiency"),
        "sleep_onset_minute_of_day": _sin_cos_to_minute(
            _masked_canonical_value(row, "sleep.sleep_onset_sin"),
            _masked_canonical_value(row, "sleep.sleep_onset_cos"),
        ),
        "wake_time_minute_of_day": _sin_cos_to_minute(
            _masked_canonical_value(row, "sleep.wake_time_sin"),
            _masked_canonical_value(row, "sleep.wake_time_cos"),
        ),
        "sleep_midpoint_minute_of_day": _sin_cos_to_minute(
            _masked_canonical_value(row, "sleep.sleep_midpoint_sin"),
            _masked_canonical_value(row, "sleep.sleep_midpoint_cos"),
        ),
        "sleep_fragmentation": _masked_canonical_value(
            row, "sleep.sleep_fragmentation"
        ),
        "night_exit_count": _masked_canonical_value(row, "sleep.night_exit_count_mean"),
        "night_exit_minutes_norm": None,
    }
    return values, specs


def _masked_canonical_value(row: Mapping[str, Any], column: str) -> float | None:
    mask = _finite_value(row.get(f"feature_mask.{column}"))
    if mask != 1.0:
        return None
    return _finite_value(row.get(column))


def _psyche_feature_rows(
    frame: pd.DataFrame,
    branch: str,
) -> list[dict[str, Any]]:
    if branch == "activity":
        features = [
            f"activity.{spec.name}"
            for spec in ACTIVITY_FEATURE_SPECS
            if spec.risk_direction
            not in (RiskDirection.MODEL_LEARNED, RiskDirection.COVERAGE_ONLY)
        ]
    else:
        features = [
            "sleep.sleep_duration_norm",
            "sleep.time_in_bed_norm",
            "sleep.sleep_efficiency",
            "sleep.sleep_onset_sin",
            "sleep.sleep_onset_cos",
            "sleep.wake_time_sin",
            "sleep.wake_time_cos",
            "sleep.sleep_midpoint_sin",
            "sleep.sleep_midpoint_cos",
            "sleep.sleep_fragmentation",
            "sleep.night_exit_count_mean",
        ]
    columns = [
        item for feature in features for item in (feature, f"feature_mask.{feature}")
    ]
    return frame.loc[:, columns].to_dict(orient="records")


def _psyche_activity_feature_rows(
    frame: pd.DataFrame,
    ecdf: PsycheDTrainingFoldECDF,
) -> list[dict[str, Any]]:
    rows = _psyche_feature_rows(frame, "activity")
    raw = pd.to_numeric(frame[ecdf.source_field], errors="coerce").to_numpy(
        dtype="float64"
    )
    training = np.asarray(ecdf.sorted_training_values, dtype="float64")
    values = np.full(len(frame), np.nan, dtype="float64")
    present = np.isfinite(raw)
    values[present] = np.searchsorted(
        training,
        raw[present],
        side="right",
    ) / len(training)
    target = ecdf.target_field
    mask = ecdf.target_mask_field
    for index, row in enumerate(rows):
        row[target] = float(values[index]) if present[index] else None
        row[mask] = int(present[index])
    return rows


def _adapter_participant_set_sha256(participant_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for participant_id in sorted(
        participant_ids,
        key=lambda value: value.encode("utf-8"),
    ):
        digest.update(participant_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sin_cos_to_minute(
    sin_value: float | None,
    cos_value: float | None,
) -> float | None:
    if sin_value is None or cos_value is None:
        return None
    angle = math.atan2(sin_value, cos_value) % (2.0 * math.pi)
    return angle * _CIRCULAR_PERIOD / (2.0 * math.pi)


def _build_social_proxy_frame(inputs: OfflineAuxiliaryInputs) -> pd.DataFrame:
    task = build_task_frame(inputs, "shenzhen_elderly")
    records: list[dict[str, Any]] = []
    for row in task.itertuples(index=False):
        row_index = int(row.canonical_row_index)
        source = inputs.frames["shenzhen_elderly"].iloc[row_index]
        mask = _finite_value(source.get("feature_mask.social_context.marital_status"))
        marital = source.get("social_context.marital_status")
        available = mask == 1.0 and marital in {"partnered", "not_partnered"}
        robust = (
            0.0
            if marital == "partnered"
            else 1.0
            if marital == "not_partnered"
            else 0.0
        )
        records.append(
            {
                "branch": "social",
                "dataset_id": "shenzhen_elderly",
                "prediction_id": f"social::shenzhen_elderly::row={row_index}",
                "canonical_row_index": row_index,
                "global_participant_id": str(row.global_participant_id),
                "outer_fold": int(row.outer_fold),
                "inner_validation_fold_by_outer_fold": (
                    row.inner_validation_fold_by_outer_fold
                ),
                "target": int(row.binary_target),
                "available": bool(available),
                "personal_change_mask": int(available),
                "reliability": 0.0,
                "valid_history_days": 0,
                "timescale_semantics": (
                    "cross_sectional_profile_to_s10_personal_change_proxy"
                ),
                "robust_deviation": robust,
                "risk_slope": 0.0,
                "isolation_forest": 0.0,
                "change_point": 0.0,
                "persistence": 0.0,
                "history_coverage": 0.0,
            }
        )
    result = pd.DataFrame(records)
    if len(result) != len(task) or result["prediction_id"].duplicated().any():
        raise PersonalTrendError("social proxy row coverage changed")
    return result.sort_values("canonical_row_index", kind="stable").reset_index(
        drop=True
    )


def train_personal_trend_expert(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train, verify, and atomically publish the three TREND-001 branches."""

    started = datetime.now(timezone.utc)
    root = Path(repository_root).resolve()
    config = load_personal_trend_config(config_path, repository_root=root)
    _validate_output_paths(config)
    _assert_outputs_available(config, overwrite=overwrite)
    inputs = load_personal_trend_inputs(config)
    protection_before = _canonical_payload_sha256(inputs.upstream_protection)
    branches: dict[str, PersonalTrendBranchModel] = {}
    oof_parts: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    preprocessing: list[dict[str, Any]] = []

    for branch in BRANCH_ORDER:
        result = _train_branch(config, inputs, branch)
        branches[branch] = result["model"]
        oof_parts.append(result["oof"])
        metrics.append(result["metrics"])
        searches.append(result["search"])
        preprocessing.append(result["preprocessing"])

    psyche_ids = set(
        inputs.assignments.loc[
            inputs.assignments["dataset_id"].astype(str).eq("psyche_d"),
            "global_participant_id",
        ].astype(str)
    )
    production_ecdf = _fit_psyche_activity_ecdf(inputs, psyche_ids)
    bundle = PersonalTrendBundle(
        bundle_version=BUNDLE_VERSION,
        training_version=TRAINING_VERSION,
        run_id=RUN_ID,
        split_id=SPLIT_ID,
        split_sha256=SPLIT_SHA256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        branches=branches,
        production_activity_ecdf=production_ecdf,
        social_proxy_transfer="cross_sectional_profile_to_s10_personal_change",
    )
    bundle.validate()
    all_oof = pd.concat(oof_parts, ignore_index=True)
    all_oof["_branch_order"] = all_oof["branch"].map(
        {name: index for index, name in enumerate(BRANCH_ORDER)}
    )
    all_oof = all_oof.sort_values(
        ["_branch_order", "outer_fold", "canonical_row_index"], kind="stable"
    ).drop(columns="_branch_order")
    all_oof = all_oof.reset_index(drop=True)
    protection_after = validate_baseline_protection(
        root,
        load_evaluation_config(
            root
            / str(
                load_offline_auxiliary_config(
                    root / str(config.payload["upstream"]["offline_config_path"]),
                    repository_root=root,
                ).payload["upstream_evaluation"]["config_path"]
            )
        ),
    )
    if _canonical_payload_sha256(protection_after) != protection_before:
        raise PersonalTrendError("active expert protection changed during TREND-001")
    return _publish_training_run(
        config,
        inputs,
        bundle,
        all_oof,
        metrics,
        searches,
        preprocessing,
        protection_before,
        started=started,
        ended=datetime.now(timezone.utc),
        overwrite=overwrite,
        command=command or (),
    )


def _train_branch(
    config: PersonalTrendTrainingConfig,
    inputs: OfflineAuxiliaryInputs,
    branch: str,
) -> dict[str, Any]:
    dataset_id = "shenzhen_elderly" if branch == "social" else "psyche_d"
    assignments = inputs.assignments[
        inputs.assignments["dataset_id"].astype(str).eq(dataset_id)
    ].copy()
    participant_ids = set(assignments["global_participant_id"].astype(str))
    if not participant_ids:
        raise PersonalTrendError(f"{branch} has no assigned participants")
    input_features = ("robust_deviation",) if branch == "social" else COMPONENT_ORDER
    static_frame = (
        build_personal_trend_training_frame(inputs, branch)
        if branch != "activity"
        else None
    )

    def frame_for(training_ids: set[str]) -> pd.DataFrame:
        if static_frame is not None:
            return static_frame.copy()
        return _branch_frame_for_training_ids(inputs, branch, training_ids)

    outer_oof: list[pd.DataFrame] = []
    outer_searches: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []

    for outer_fold in range(OUTER_FOLD_COUNT):
        outer_train_ids = set(
            assignments.loc[
                assignments["outer_fold"].astype(int).ne(outer_fold),
                "global_participant_id",
            ].astype(str)
        )
        outer_test_ids = participant_ids - outer_train_ids
        if not outer_train_ids or not outer_test_ids:
            raise PersonalTrendError("TREND-001 outer partition is empty")
        candidate_predictions: dict[float, list[np.ndarray]] = {
            value: [] for value in C_VALUES
        }
        candidate_targets: dict[float, list[np.ndarray]] = {
            value: [] for value in C_VALUES
        }
        candidate_weights: dict[float, list[np.ndarray]] = {
            value: [] for value in C_VALUES
        }
        inner_manifests: list[dict[str, Any]] = []
        for inner_fold in range(INNER_FOLD_COUNT):
            inner_validation_ids = {
                str(row.global_participant_id)
                for row in assignments.itertuples(index=False)
                if str(row.global_participant_id) in outer_train_ids
                and _inner_fold_value(
                    row.inner_validation_fold_by_outer_fold,
                    outer_fold,
                )
                == inner_fold
            }
            inner_train_ids = outer_train_ids - inner_validation_ids
            full = frame_for(inner_train_ids)
            train = _available_rows(
                full[full["global_participant_id"].isin(inner_train_ids)]
            )
            validation = _available_rows(
                full[full["global_participant_id"].isin(inner_validation_ids)]
            )
            _require_binary_partitions(train, validation, branch=branch)
            preprocessor = TrendPreprocessor.fit(
                train,
                input_features=input_features,
            )
            train_matrix = preprocessor.transform(train)
            validation_matrix = preprocessor.transform(validation)
            train_target = train["target"].to_numpy(dtype="int64")
            validation_target = validation["target"].to_numpy(dtype="int64")
            train_weight = personal_trend_sample_weights(train)
            validation_weight = personal_trend_sample_weights(validation)
            for value in C_VALUES:
                estimator = _fit_logistic(
                    train_matrix,
                    train_target,
                    train_weight,
                    c_value=value,
                )
                prediction = estimator.predict_proba(validation_matrix)[:, 1]
                candidate_predictions[value].append(prediction)
                candidate_targets[value].append(validation_target)
                candidate_weights[value].append(validation_weight)
            inner_manifests.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "fit_participant_count": len(inner_train_ids),
                    "fit_participant_sha256": _string_set_sha256(inner_train_ids),
                    "validation_participant_count": len(inner_validation_ids),
                    "validation_participant_sha256": _string_set_sha256(
                        inner_validation_ids
                    ),
                    "fit_available_rows": len(train),
                    "validation_available_rows": len(validation),
                    "preprocessor": preprocessor.to_manifest(),
                    "activity_ecdf_fit_scope": (
                        "inner_train_participants" if branch == "activity" else None
                    ),
                }
            )
        candidate_rows: list[dict[str, Any]] = []
        for value in C_VALUES:
            target = np.concatenate(candidate_targets[value])
            prediction = np.concatenate(candidate_predictions[value])
            weight = np.concatenate(candidate_weights[value])
            scores = _binary_metrics(target, prediction, weight)
            candidate_rows.append(
                {
                    "candidate_id": f"l2-c={value:g}",
                    "c": value,
                    "pooled_inner_oof_auprc": scores["weighted_auprc"],
                    "pooled_inner_oof_brier": scores["weighted_brier"],
                    "pooled_inner_available_rows": int(target.size),
                }
            )
        selected = sorted(
            candidate_rows,
            key=lambda row: (
                -float(row["pooled_inner_oof_auprc"]),
                float(row["pooled_inner_oof_brier"]),
                str(row["candidate_id"]),
            ),
        )[0]
        outer_full = frame_for(outer_train_ids)
        outer_train = _available_rows(
            outer_full[outer_full["global_participant_id"].isin(outer_train_ids)]
        )
        outer_test_all = outer_full[
            outer_full["global_participant_id"].isin(outer_test_ids)
        ].copy()
        outer_test = _available_rows(outer_test_all)
        _require_binary_partitions(outer_train, outer_test, branch=branch)
        preprocessor = TrendPreprocessor.fit(
            outer_train,
            input_features=input_features,
        )
        estimator = _fit_logistic(
            preprocessor.transform(outer_train),
            outer_train["target"].to_numpy(dtype="int64"),
            personal_trend_sample_weights(outer_train),
            c_value=float(selected["c"]),
        )
        outer_test_all["probability"] = np.nan
        outer_test_all.loc[outer_test.index, "probability"] = estimator.predict_proba(
            preprocessor.transform(outer_test)
        )[:, 1]
        outer_test_all["outer_fold"] = outer_fold
        outer_oof.append(outer_test_all)
        outer_searches.append(
            {
                "outer_fold": outer_fold,
                "outer_train_participant_count": len(outer_train_ids),
                "outer_test_participant_count": len(outer_test_ids),
                "candidates": candidate_rows,
                "selected_candidate_id": selected["candidate_id"],
                "selected_c": selected["c"],
                "outer_test_role": "evaluation_only",
            }
        )
        preprocessing_rows.extend(inner_manifests)
        preprocessing_rows.append(
            {
                "outer_fold": outer_fold,
                "scope": "outer_train",
                "fit_participant_count": len(outer_train_ids),
                "fit_participant_sha256": _string_set_sha256(outer_train_ids),
                "fit_available_rows": len(outer_train),
                "preprocessor": preprocessor.to_manifest(),
                "activity_ecdf_fit_scope": (
                    "outer_train_participants" if branch == "activity" else None
                ),
            }
        )

    oof = pd.concat(outer_oof, ignore_index=True)
    oof = oof.sort_values(
        ["outer_fold", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    if len(oof) != len(frame_for(participant_ids)):
        raise PersonalTrendError(f"{branch} OOF coverage changed")
    if oof["prediction_id"].duplicated().any():
        raise PersonalTrendError(f"{branch} OOF IDs are duplicated")
    if oof.loc[oof["available"], "probability"].isna().any():
        raise PersonalTrendError(f"{branch} available OOF probability is null")
    if oof.loc[~oof["available"], "probability"].notna().any():
        raise PersonalTrendError(f"{branch} unavailable OOF probability is non-null")

    production_candidate = _select_production_c(outer_searches)
    production_frame = frame_for(participant_ids)
    production_rows = _available_rows(production_frame)
    _require_binary_partitions(production_rows, production_rows, branch=branch)
    production_preprocessor = TrendPreprocessor.fit(
        production_rows,
        input_features=input_features,
    )
    production_estimator = _fit_logistic(
        production_preprocessor.transform(production_rows),
        production_rows["target"].to_numpy(dtype="int64"),
        personal_trend_sample_weights(production_rows),
        c_value=float(production_candidate["c"]),
    )
    evidence_scope = (
        "engineering_proxy_no_direct_s10_phq9_validation"
        if branch == "social"
        else "proxy_label_supported_timescale_transfer"
    )
    model = PersonalTrendBranchModel(
        branch=branch,
        input_features=tuple(input_features),
        preprocessor=production_preprocessor,
        estimator=production_estimator,
        selected_c=float(production_candidate["c"]),
        training_participant_count=len(participant_ids),
        training_row_count=len(production_rows),
        training_participant_sha256=_string_set_sha256(participant_ids),
        evidence_scope=evidence_scope,
    )
    available_oof = _available_rows(oof)
    metric = _binary_metrics(
        available_oof["target"].to_numpy(dtype="int64"),
        available_oof["probability"].to_numpy(dtype="float64"),
        personal_trend_sample_weights(available_oof),
    )
    metric.update(
        {
            "branch": branch,
            "dataset_id": dataset_id,
            "row_count": len(oof),
            "participant_count": int(oof["global_participant_id"].nunique()),
            "available_row_count": len(available_oof),
            "unavailable_row_count": int((~oof["available"]).sum()),
            "positive_available_rows": int(available_oof["target"].sum()),
            "selected_c": model.selected_c,
            "evidence_scope": evidence_scope,
            "threshold_status": "0.5_descriptive_only_not_production_selected",
        }
    )
    preprocessing_rows.append(
        {
            "scope": "production",
            "fit_participant_count": len(participant_ids),
            "fit_participant_sha256": _string_set_sha256(participant_ids),
            "fit_available_rows": len(production_rows),
            "preprocessor": production_preprocessor.to_manifest(),
            "activity_ecdf_fit_scope": (
                "all_psyche_d_participants" if branch == "activity" else None
            ),
        }
    )
    return {
        "model": model,
        "oof": oof,
        "metrics": metric,
        "search": {
            "branch": branch,
            "candidate_order": [f"l2-c={value:g}" for value in C_VALUES],
            "outer_searches": outer_searches,
            "production_selection": production_candidate,
        },
        "preprocessing": {"branch": branch, "folds": preprocessing_rows},
    }


def _branch_frame_for_training_ids(
    inputs: OfflineAuxiliaryInputs,
    branch: str,
    training_ids: set[str],
) -> pd.DataFrame:
    return build_personal_trend_training_frame(
        inputs,
        branch,
        activity_ecdf_training_participant_ids=(
            training_ids if branch == "activity" else None
        ),
    )


def _available_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["available"].astype(bool)].copy()


def _require_binary_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    branch: str,
) -> None:
    if train.empty or validation.empty:
        raise PersonalTrendError(f"{branch} fold has no available rows")
    if set(train["target"].astype(int)) != {0, 1}:
        raise PersonalTrendError(f"{branch} training fold is not binary")
    if set(validation["target"].astype(int)) != {0, 1}:
        raise PersonalTrendError(f"{branch} validation fold is not binary")


def _inner_fold_value(value: Any, outer_fold: int) -> int | None:
    if not isinstance(value, Mapping):
        raise PersonalTrendError("inner-fold mapping is invalid")
    result = value.get(str(outer_fold), value.get(outer_fold))
    if result is None:
        return None
    converted = int(result)
    if converted not in range(INNER_FOLD_COUNT):
        raise PersonalTrendError("inner-fold value is outside 0..4")
    return converted


def personal_trend_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal class, participant-class, and repeated-window contribution."""

    required = {"global_participant_id", "target"}
    if frame.empty or not required.issubset(frame.columns):
        raise PersonalTrendError("trend sample-weight frame is invalid")
    target = pd.to_numeric(frame["target"], errors="raise").astype("int64")
    if set(target.tolist()) - {0, 1}:
        raise PersonalTrendError("trend sample-weight target is not binary")
    participants = frame["global_participant_id"].astype(str)
    weights = np.zeros(len(frame), dtype="float64")
    present_classes = sorted(set(target.tolist()))
    class_mass = 1.0 / len(present_classes)
    for target_value in present_classes:
        class_indices = np.flatnonzero(target.to_numpy() == target_value)
        class_participants = participants.iloc[class_indices]
        participant_counts = class_participants.value_counts()
        participant_mass = class_mass / len(participant_counts)
        for index, participant_id in zip(
            class_indices,
            class_participants.tolist(),
            strict=True,
        ):
            weights[index] = participant_mass / float(
                participant_counts[participant_id]
            )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise PersonalTrendError("trend sample weights are invalid")
    return weights * (len(weights) / weights.sum())


def _fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    c_value: float,
) -> LogisticRegression:
    estimator = LogisticRegression(
        C=float(c_value),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=5000,
        tol=1.0e-8,
        random_state=RANDOM_SEED,
    )
    estimator.fit(matrix, target, sample_weight=sample_weight)
    return estimator


def _binary_metrics(
    target: Sequence[int],
    probability: Sequence[float],
    sample_weight: Sequence[float],
) -> dict[str, float]:
    y = np.asarray(target, dtype="int64")
    p = np.asarray(probability, dtype="float64")
    weight = np.asarray(sample_weight, dtype="float64")
    if (
        y.ndim != 1
        or y.size == 0
        or p.shape != y.shape
        or weight.shape != y.shape
        or not np.isfinite(p).all()
        or not np.isfinite(weight).all()
        or np.any((p < 0) | (p > 1))
        or set(y.tolist()) != {0, 1}
    ):
        raise PersonalTrendError("binary metric inputs are invalid")
    hard = (p >= 0.5).astype("int64")
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, hard)),
        "f1_at_0_5": float(f1_score(y, hard, zero_division=0)),
        "weighted_auroc": float(roc_auc_score(y, p, sample_weight=weight)),
        "weighted_auprc": float(average_precision_score(y, p, sample_weight=weight)),
        "weighted_brier": float(np.average(np.square(p - y), weights=weight)),
        "weighted_log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weight)),
    }


def _select_production_c(
    outer_searches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for value in C_VALUES:
        candidate_id = f"l2-c={value:g}"
        matches = [
            candidate
            for outer in outer_searches
            for candidate in outer["candidates"]
            if candidate["candidate_id"] == candidate_id
        ]
        if len(matches) != OUTER_FOLD_COUNT:
            raise PersonalTrendError("production C aggregation is incomplete")
        rows.append(
            {
                "candidate_id": candidate_id,
                "c": value,
                "mean_outer_inner_auprc": float(
                    np.mean([row["pooled_inner_oof_auprc"] for row in matches])
                ),
                "mean_outer_inner_brier": float(
                    np.mean([row["pooled_inner_oof_brier"] for row in matches])
                ),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_outer_inner_auprc"]),
            float(row["mean_outer_inner_brier"]),
            str(row["candidate_id"]),
        ),
    )[0]
    return {**selected, "candidate_aggregation": rows}


def _publish_training_run(
    config: PersonalTrendTrainingConfig,
    inputs: OfflineAuxiliaryInputs,
    bundle: PersonalTrendBundle,
    oof: pd.DataFrame,
    metrics: Sequence[Mapping[str, Any]],
    searches: Sequence[Mapping[str, Any]],
    preprocessing: Sequence[Mapping[str, Any]],
    protection_sha256: str,
    *,
    started: datetime,
    ended: datetime,
    overwrite: bool,
    command: Sequence[str],
) -> dict[str, Any]:
    root = config.repository_root
    stage_root = Path(tempfile.mkdtemp(prefix="trend-001-stage-", dir=root))
    try:
        stage_model = stage_root / config.model_path.name
        stage_manifest = stage_root / config.manifest_path.name
        stage_report = stage_root / "report"
        stage_report.mkdir(parents=True)
        joblib.dump(bundle, stage_model, compress=0, protocol=4)
        loaded = joblib.load(stage_model)
        if not isinstance(loaded, PersonalTrendBundle):
            raise PersonalTrendError("staged PersonalTrend bundle cannot be reloaded")
        loaded.validate()
        model_sha = _sha256_file(stage_model)

        published_oof = oof.copy()
        published_oof.insert(0, "oof_version", OOF_VERSION)
        output_columns = [
            "oof_version",
            "branch",
            "dataset_id",
            "prediction_id",
            "canonical_row_index",
            "global_participant_id",
            "outer_fold",
            "target",
            "available",
            "personal_change_mask",
            "reliability",
            "valid_history_days",
            "timescale_semantics",
            *COMPONENT_ORDER,
            "probability",
        ]
        _write_parquet(
            published_oof.loc[:, output_columns],
            stage_report / "oof_predictions.parquet",
        )
        metrics_payload = {
            "version": METRICS_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "threshold_selection": "none",
            "branches": list(metrics),
        }
        search_payload = {
            "version": SEARCH_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "outer_test_fold_role": "evaluation_only",
            "selection": (
                "pooled_inner_oof_auprc_desc_then_brier_asc_then_candidate_id"
            ),
            "branches": list(searches),
        }
        preprocessing_payload = {
            "version": "mood-social-personal-trend-preprocessing-v1",
            "fold_fit_only": True,
            "activity_ecdf_fold_fit_only": True,
            "branches": list(preprocessing),
        }
        baseline_payload = {
            "version": "mood-social-personal-trend-baseline-policy-v1",
            "baseline": config.payload["baseline"],
            "components": config.payload["components"],
            "reliability": config.payload["reliability"],
            "runtime": {
                "history_window": "D-28_through_D-1",
                "current_day": "D",
                "missing_day_forward_fill": False,
                "gait_partition": "person_plus_camera_id_plus_scene_version",
                "activity_subdomain_aggregation": "componentwise_maximum",
                "walking_speed_normalization": (
                    "per_scene_clip_q10_q90_then_same_day_non_null_median"
                ),
                "walking_speed_denominator_epsilon": 1.0e-6,
            },
        }
        warning_payload = {
            "version": "mood-social-personal-trend-warnings-v1",
            "warnings": [
                {
                    "code": "PSYCHE_D_NOMINAL_MONTH_TIMESCALE_PROXY",
                    "branches": ["activity", "sleep"],
                    "message": (
                        "PSYCHE-D nominal-month sequences proxy natural-day "
                        "personal change; only fourth-and-later observations can "
                        "satisfy the three-history-day mask."
                    ),
                },
                {
                    "code": "NO_DIRECT_S10_PHQ9_VALIDATION",
                    "branches": ["social"],
                    "message": (
                        "The social branch calibrates Shenzhen marital-status "
                        "profile association onto runtime S10 robust deviation; "
                        "this is an engineering proxy, not direct longitudinal "
                        "S10/PHQ-9 validation."
                    ),
                },
                {
                    "code": "NO_PRODUCTION_THRESHOLD_SELECTED",
                    "branches": list(BRANCH_ORDER),
                    "message": "TREND-001 reports continuous probabilities only.",
                },
            ],
        }
        protection_payload = {
            "version": "mood-social-personal-trend-upstream-protection-v1",
            "split_sha256": SPLIT_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "evaluation_core_sha256": EVALUATION_CORE_SHA256,
            "baseline_protection_sha256": protection_sha256,
            "model006_model_sha256": MODEL006_MODEL_SHA256,
            "model006_manifest_sha256": MODEL006_MANIFEST_SHA256,
            "model006_report_core_sha256": MODEL006_REPORT_CORE_SHA256,
            "active_expert_protection": inputs.upstream_protection,
        }
        internal_manifest = {
            "version": MANIFEST_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "training_version": TRAINING_VERSION,
            "bundle_version": BUNDLE_VERSION,
            "model_sha256": model_sha,
            "config_sha256": config.config_sha256,
            "split_id": SPLIT_ID,
            "split_sha256": SPLIT_SHA256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "component_order": list(COMPONENT_ORDER),
            "branch_order": list(BRANCH_ORDER),
            "production_boundary": config.payload["production_boundary"],
            "branches": {
                name: {
                    "input_features": list(model.input_features),
                    "selected_c": model.selected_c,
                    "training_participant_count": model.training_participant_count,
                    "training_row_count": model.training_row_count,
                    "training_participant_sha256": (model.training_participant_sha256),
                    "evidence_scope": model.evidence_scope,
                    "preprocessor": model.preprocessor.to_manifest(),
                }
                for name, model in bundle.branches.items()
            },
            "production_activity_ecdf": bundle.production_activity_ecdf.to_dict(),
        }
        _write_json(stage_report / "metrics.json", metrics_payload)
        _write_json(stage_report / "search_results.json", search_payload)
        _write_json(stage_report / "preprocessing.json", preprocessing_payload)
        _write_json(stage_report / "baseline_policy.json", baseline_payload)
        _write_json(stage_report / "warnings.json", warning_payload)
        _write_json(stage_report / "upstream_protection.json", protection_payload)
        _write_json(stage_report / "training_manifest.json", internal_manifest)
        (stage_report / "model_card.md").write_text(
            _model_card(metrics), encoding="utf-8", newline="\n"
        )
        report_core = _deterministic_report_core_sha256(stage_report)
        external_manifest = {
            **internal_manifest,
            "report_core_sha256": report_core,
            "oof_sha256": _sha256_file(stage_report / "oof_predictions.parquet"),
            "metrics_sha256": _sha256_file(stage_report / "metrics.json"),
            "search_sha256": _sha256_file(stage_report / "search_results.json"),
            "baseline_policy_sha256": _sha256_file(
                stage_report / "baseline_policy.json"
            ),
        }
        _write_json(stage_manifest, external_manifest)
        manifest_sha = _sha256_file(stage_manifest)
        core_rows = _tree_hash_rows(stage_report)
        _write_json(
            stage_report / "artifacts.json",
            {
                "version": "mood-social-personal-trend-artifacts-v1",
                "artifact_count": len(core_rows),
                "artifacts": core_rows,
                "core_sha256": report_core,
                "report_core_sha256": report_core,
            },
        )
        _write_json(
            stage_report / "run.json",
            {
                "version": "mood-social-personal-trend-run-v1",
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "command": list(command),
                "python": sys.version,
                "platform": platform.platform(),
                "git": _git_state(root),
                "libraries": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "pyarrow": pyarrow.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
            },
        )
        _atomic_publish_file(stage_model, config.model_path, overwrite=overwrite)
        _atomic_publish_directory(
            stage_report,
            config.report_directory,
            overwrite=overwrite,
        )
        _atomic_publish_file(
            stage_manifest,
            config.manifest_path,
            overwrite=overwrite,
        )
        return {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "model_path": config.model_path.as_posix(),
            "model_sha256": model_sha,
            "manifest_path": config.manifest_path.as_posix(),
            "manifest_sha256": manifest_sha,
            "report_directory": config.report_directory.as_posix(),
            "report_core_sha256": report_core,
            "oof_row_count": len(oof),
            "available_oof_row_count": int(oof["available"].sum()),
            "metrics": list(metrics),
            "upstream_protection_sha256": protection_sha256,
        }
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def load_personal_trend_bundle(path: str | Path) -> PersonalTrendBundle:
    try:
        bundle = joblib.load(Path(path))
    except (OSError, ValueError, TypeError) as exc:
        raise PersonalTrendError("PersonalTrend bundle is unreadable") from exc
    if not isinstance(bundle, PersonalTrendBundle):
        raise PersonalTrendError("PersonalTrend artifact has the wrong type")
    bundle.validate()
    return bundle


def _model_card(metrics: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        "# PersonalTrendExpert V3.3.3",
        "",
        "运行：`MH-20260801-008`（TREND-001）。该工件尚未接入融合或 HTTP。",
        "",
        "| 分支 | 严格 OOF 可用行 | AUROC | AUPRC | Brier | 证据范围 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        rows.append(
            "| {branch} | {count} | {auroc:.6f} | {auprc:.6f} | "
            "{brier:.6f} | {scope} |".format(
                branch=item["branch"],
                count=item["available_row_count"],
                auroc=item["auroc"],
                auprc=item["auprc"],
                brier=item["brier"],
                scope=item["evidence_scope"],
            )
        )
    rows.extend(
        [
            "",
            "活动与睡眠是 PSYCHE-D 名义月份到自然日趋势的代理迁移。社会分支没有 S10 纵向 PHQ-9 配对数据，使用深圳老年人婚姻状态横断面关联作工程代理；不得解释为直接 S10 验证。",
            "",
            "量表分数、量表等级、量表条目、发布方标签和 MODEL-006 预测均未作为 PersonalTrend 风险输入。",
            "",
        ]
    )
    return "\n".join(rows)


def _validate_output_paths(config: PersonalTrendTrainingConfig) -> None:
    root = config.repository_root
    for path in (config.model_path, config.manifest_path, config.report_directory):
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise PersonalTrendError(
                "TREND-001 output escapes repository root"
            ) from exc
    protected_names = {
        "activity_expert.joblib",
        "sleep_expert.joblib",
        "activity_sleep_joint_expert.joblib",
        "physiology_expert.joblib",
        "social_context_expert.joblib",
        "offline_auxiliary_models.joblib",
    }
    if config.model_path.name in protected_names:
        raise PersonalTrendError("TREND-001 output overlaps a protected model")
    if config.model_path == config.manifest_path:
        raise PersonalTrendError("TREND-001 model and manifest paths overlap")


def _assert_outputs_available(
    config: PersonalTrendTrainingConfig,
    *,
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in (config.model_path, config.manifest_path, config.report_directory)
        if path.exists()
    ]
    if existing and not overwrite:
        raise PersonalTrendError(
            "TREND-001 output already exists; pass --overwrite explicitly: "
            + ", ".join(path.as_posix() for path in existing)
        )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        version="2.6",
        data_page_version="1.0",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=4096,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def _atomic_publish_file(stage: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise PersonalTrendError(f"output already exists: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix="trend-001-backup-", dir=destination.parent)
        )
        backup = backup_root / destination.name
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None:
            shutil.rmtree(backup.parent, ignore_errors=True)


def _atomic_publish_directory(
    stage: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise PersonalTrendError(f"output already exists: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix="trend-001-backup-", dir=destination.parent)
        )
        backup = backup_root / destination.name
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None:
            shutil.rmtree(backup.parent, ignore_errors=True)


def _tree_hash_rows(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(directory).as_posix().encode("utf-8"),
        )
    ]


def _deterministic_report_core_sha256(report: Path) -> str:
    rows = [
        row
        for row in _tree_hash_rows(report)
        if row["path"] not in {"run.json", "artifacts.json"}
    ]
    return _rows_sha256(rows)


def _report_core_sha256(report_path: Path) -> str:
    artifact_path = report_path / "artifacts.json"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PersonalTrendError("upstream report artifacts are unreadable") from exc
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or int(payload.get("artifact_count", -1)) != len(
        rows
    ):
        raise PersonalTrendError("upstream report artifact manifest is invalid")
    for row in rows:
        path = report_path / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise PersonalTrendError("upstream report artifact drifted")
    core = str(payload.get("report_core_sha256", payload.get("core_sha256", "")))
    if not _is_sha256(core):
        raise PersonalTrendError("upstream report core hash is invalid")
    return core


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def _string_set_sha256(values: Sequence[str] | set[str]) -> str:
    ordered = sorted(
        set(str(value) for value in values), key=lambda value: value.encode()
    )
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


__all__ = [
    "BRANCH_ORDER",
    "BUNDLE_VERSION",
    "COMPONENT_ORDER",
    "PersonalTrendBranchResult",
    "PersonalTrendBundle",
    "PersonalTrendError",
    "PersonalTrendTrainingConfig",
    "TrendComponents",
    "TrendObservation",
    "build_personal_trend_training_frame",
    "compute_trend_components",
    "load_personal_trend_bundle",
    "load_personal_trend_config",
    "load_personal_trend_inputs",
    "personal_trend_sample_weights",
    "train_personal_trend_expert",
]
