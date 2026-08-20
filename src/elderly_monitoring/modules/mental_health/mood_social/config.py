from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import time
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from elderly_monitoring.common.config import load_yaml


CURRENT_CONFIG_VERSION = "3.3.3"
PROJECT_ROOT_ENV = "ELDERLY_MONITORING_PROJECT_ROOT"
CONFIG_PATH_ENV = "MOOD_SOCIAL_CONFIG_PATH"
_CURRENT_CONFIG_RELATIVE_PATH = (
    Path("configs") / "modules" / "mood_social_v3_3_3.yaml"
)


def _discover_project_root() -> Path:
    """Locate external runtime assets in source and installed layouts."""

    configured_root = os.environ.get(PROJECT_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[5]
    candidates = (source_root, Path.cwd().resolve(), *Path.cwd().resolve().parents)
    for candidate in candidates:
        if (candidate / _CURRENT_CONFIG_RELATIVE_PATH).is_file():
            return candidate
    return source_root


PROJECT_ROOT = _discover_project_root()
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / _CURRENT_CONFIG_RELATIVE_PATH
)
HISTORICAL_V3_3_2_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "modules" / "mood_social_v3_3_2.yaml"
)

ACTIVE_SCORE_WEIGHT_FIELDS = (
    "center_motion_score",
    "pose_motion_score",
    "zone_transition_score",
    "posture_change_score",
)
MASK_LAYERS = (
    "feature_mask",
    "day_mask",
    "expert_mask",
    "personal_change_mask",
)
PERSONAL_TREND_DOMAINS = ("activity", "sleep", "social")
DEVICE_SOURCE_CODES = {
    "C": "camera",
    "S": "sleep_device",
    "T": "s10",
}
SUPPORTED_DEVICE_COMBINATIONS = (
    ("camera",),
    ("sleep_device",),
    ("s10",),
    ("camera", "sleep_device"),
    ("camera", "s10"),
    ("sleep_device", "s10"),
    ("camera", "sleep_device", "s10"),
)
MODEL_ARTIFACT_FILES = {
    "activity_expert": "activity_expert.joblib",
    "sleep_expert": "sleep_expert.joblib",
    "physiology_expert": "physiology_expert.joblib",
    "activity_sleep_joint_expert": "activity_sleep_joint_expert.joblib",
    "social_context_expert": "social_context_expert.joblib",
    "personal_trend": "personal_trend.joblib",
    "mood_fusion": "mood_fusion.joblib",
    "feature_schema": "feature_schema.json",
    "manifest": "manifest.json",
    "split_manifest": "split_manifest.json",
    "metrics": "metrics.json",
    "factor_labels_zh_cn": "factor_labels.zh-CN.json",
    "checksums": "SHA256SUMS",
}


@dataclass(frozen=True)
class SchemaConfig:
    request: str
    response: str
    error: str
    response_module: str


@dataclass(frozen=True)
class RuntimeConfig:
    timezone: str
    update_frequency: str
    inference_window_days: int
    fallback_strategy: str
    random_seed: int

    @property
    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class BaselineConfig:
    history_lookback_calendar_days: int
    initial_days: int
    stable_days: int
    max_valid_days: int
    center_method: str
    quantile_interpolation: str
    lower_quantile: float
    upper_quantile: float
    mad_scale_factor: float
    quantile_scale_divisor: float
    relative_scale_fraction: float
    absolute_scale_floor: float
    standardized_component_divisor: float
    relative_denominator_floor: float
    relative_change_scale: float
    deviation_aggregation: str
    abnormal_score_threshold: float
    exclude_current_day: bool
    exclude_imputed_values: bool
    exclude_abnormal_domain_days_after_initialization: bool


@dataclass(frozen=True)
class ActiveScoreWeights:
    center_motion_score: float
    pose_motion_score: float
    zone_transition_score: float
    posture_change_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: getattr(self, field)
            for field in ACTIVE_SCORE_WEIGHT_FIELDS
        }


@dataclass(frozen=True)
class WalkingSpeedConfig:
    history_lookback_calendar_days: int
    min_history_days: int
    max_valid_days: int
    lower_quantile: float
    upper_quantile: float
    denominator_epsilon: float


@dataclass(frozen=True)
class CameraConfig:
    daytime_start: str
    daytime_end: str
    daytime_minutes: int
    activity_window_seconds: int
    active_score_weights: ActiveScoreWeights
    active_score_threshold: float
    low_activity_score_threshold: float
    sedentary_min_minutes: int
    walking_speed: WalkingSpeedConfig

    @property
    def daytime_start_time(self) -> time:
        return _parse_clock(self.daytime_start, "camera.daytime_start")

    @property
    def daytime_end_time(self) -> time:
        return _parse_clock(self.daytime_end, "camera.daytime_end")


@dataclass(frozen=True)
class MaskingConfig:
    layers: tuple[str, str, str, str]
    imputed_values_activate_masks: bool
    mask_values_are_risk_features: bool


@dataclass(frozen=True)
class SlopeConfig:
    method: str
    minimum_total_points: int
    use_calendar_day_spacing: bool


@dataclass(frozen=True)
class IsolationForestConfig:
    stable_baseline_only: bool
    n_estimators: int
    contamination: float
    random_seed: int
    minimum_common_features: int


@dataclass(frozen=True)
class ChangePointConfig:
    minimum_valid_points: int
    recent_window_points: int


@dataclass(frozen=True)
class PersonalTrendConfig:
    domains: tuple[str, str, str]
    slope: SlopeConfig
    missing_day_breaks_sequence: bool
    isolation_forest: IsolationForestConfig
    change_point: ChangePointConfig


@dataclass(frozen=True)
class EffectiveEvidenceConfig:
    supervised_term: str
    personal_change_term: str
    unavailable_at_or_below: float
    run_intercept_when_unavailable: bool


@dataclass(frozen=True)
class FusionConfig:
    method: str
    probability_clip_epsilon: float
    confidence_mode: str
    independent_confidence_risk_term: bool
    mask_values_are_risk_features: bool
    effective_evidence: EffectiveEvidenceConfig


@dataclass(frozen=True)
class AttentionConfig:
    thresholds: tuple[float, float, float]
    score_rounding: str


@dataclass(frozen=True)
class TrendConfig:
    max_history_results: int
    include_current_result: bool
    same_model_version_only: bool
    minimum_points: int
    method: str
    use_calendar_day_spacing: bool
    rising_threshold_per_day: float
    falling_threshold_per_day: float


@dataclass(frozen=True)
class DeviceConfig:
    source_codes: Mapping[str, str]
    supported_combinations: tuple[tuple[str, ...], ...]
    profile_is_device: bool


@dataclass(frozen=True)
class ModelArtifactsConfig:
    version: str
    directory: Path
    artifacts: Mapping[str, str]

    def path_for(self, artifact: str) -> Path:
        try:
            filename = self.artifacts[artifact]
        except KeyError as exc:
            raise KeyError(f"unknown mood-social artifact: {artifact}") from exc
        return self.directory / filename


@dataclass(frozen=True)
class MoodSocialConfig:
    module: str
    version: str
    schema: SchemaConfig
    runtime: RuntimeConfig
    baseline: BaselineConfig
    camera: CameraConfig
    masks: MaskingConfig
    personal_trend: PersonalTrendConfig
    fusion: FusionConfig
    attention: AttentionConfig
    trend: TrendConfig
    devices: DeviceConfig
    models: ModelArtifactsConfig

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return self.attention.thresholds


def load_mood_social_config(path: str | Path | None = None) -> MoodSocialConfig:
    """Load the current V3.3.3 configuration.

    V3.3.2 remains on disk for audit only. Passing it explicitly is rejected
    rather than silently upgrading historical semantics.
    """

    configured_path = os.environ.get(CONFIG_PATH_ENV)
    if path is not None:
        config_path = Path(path)
    elif configured_path:
        config_path = Path(configured_path).expanduser()
    else:
        config_path = _discover_project_root() / _CURRENT_CONFIG_RELATIVE_PATH

    project_root = _discover_project_root()
    return mood_social_config_from_mapping(
        load_yaml(config_path),
        source=config_path,
        project_root=project_root,
    )


def mood_social_config_from_mapping(
    root: Mapping[str, Any],
    *,
    source: str | Path = "configuration",
    project_root: str | Path = PROJECT_ROOT,
) -> MoodSocialConfig:
    if not isinstance(root, Mapping):
        raise ValueError(f"{source}: configuration root must be a mapping")

    module = _string(root, "module", source)
    version = _string(root, "version", source)
    _require_frozen_value(module, "mood_social", f"{source}: module")
    _require_frozen_value(
        version,
        CURRENT_CONFIG_VERSION,
        f"{source}: version",
    )
    _require_exact_keys(
        root,
        {
            "module",
            "version",
            "schema",
            "runtime",
            "baseline",
            "camera",
            "masks",
            "personal_trend",
            "fusion",
            "attention",
            "trend",
            "devices",
            "models",
        },
        source,
    )

    schema = _schema_config(_mapping(root, "schema", source), source=source)
    runtime = _runtime_config(_mapping(root, "runtime", source), source=source)
    baseline = _baseline_config(_mapping(root, "baseline", source), source=source)
    camera = _camera_config(
        _mapping(root, "camera", source),
        source=source,
        baseline=baseline,
    )
    masks = _masking_config(_mapping(root, "masks", source), source=source)
    personal_trend = _personal_trend_config(
        _mapping(root, "personal_trend", source),
        source=source,
        runtime=runtime,
    )
    fusion = _fusion_config(_mapping(root, "fusion", source), source=source)
    attention = _attention_config(
        _mapping(root, "attention", source),
        source=source,
    )
    trend = _trend_config(_mapping(root, "trend", source), source=source)
    devices = _device_config(_mapping(root, "devices", source), source=source)
    models = _model_artifacts_config(
        _mapping(root, "models", source),
        source=source,
        project_root=Path(project_root),
    )
    return MoodSocialConfig(
        module=module,
        version=version,
        schema=schema,
        runtime=runtime,
        baseline=baseline,
        camera=camera,
        masks=masks,
        personal_trend=personal_trend,
        fusion=fusion,
        attention=attention,
        trend=trend,
        devices=devices,
        models=models,
    )


def _schema_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> SchemaConfig:
    path = f"{source}: schema"
    _require_exact_keys(
        values,
        {"request", "response", "error", "response_module"},
        path,
    )
    config = SchemaConfig(
        request=_string(values, "request", path),
        response=_string(values, "response", path),
        error=_string(values, "error", path),
        response_module=_string(values, "response_module", path),
    )
    expected = (
        "mood_social_infer_request_v3",
        "mood_social_infer_response_v3",
        "mood_social_error_v1",
        "mood_social_attention",
    )
    if (
        config.request,
        config.response,
        config.error,
        config.response_module,
    ) != expected:
        raise ValueError(
            f"{path} must use request/response V3, error V1, and "
            "mood_social_attention"
        )
    return config


def _runtime_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> RuntimeConfig:
    path = f"{source}: runtime"
    _require_exact_keys(
        values,
        {
            "timezone",
            "update_frequency",
            "inference_window_days",
            "fallback_strategy",
            "random_seed",
        },
        path,
    )
    timezone = _string(values, "timezone", path)
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"{path}.timezone is unknown: {timezone!r}") from exc
    config = RuntimeConfig(
        timezone=timezone,
        update_frequency=_string(values, "update_frequency", path),
        inference_window_days=_integer(values, "inference_window_days", path),
        fallback_strategy=_string(values, "fallback_strategy", path),
        random_seed=_integer(values, "random_seed", path),
    )
    expected = (
        "Asia/Shanghai",
        "daily",
        7,
        "unavailable",
        20260728,
    )
    if (
        config.timezone,
        config.update_frequency,
        config.inference_window_days,
        config.fallback_strategy,
        config.random_seed,
    ) != expected:
        raise ValueError(
            f"{path} must remain frozen at "
            "Asia/Shanghai, daily, 7, unavailable, seed 20260728"
        )
    return config


def _baseline_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> BaselineConfig:
    path = f"{source}: baseline"
    _require_exact_keys(
        values,
        {
            "history_lookback_calendar_days",
            "initial_days",
            "stable_days",
            "max_valid_days",
            "center_method",
            "quantile_interpolation",
            "lower_quantile",
            "upper_quantile",
            "mad_scale_factor",
            "quantile_scale_divisor",
            "relative_scale_fraction",
            "absolute_scale_floor",
            "standardized_component_divisor",
            "relative_denominator_floor",
            "relative_change_scale",
            "deviation_aggregation",
            "abnormal_score_threshold",
            "exclude_current_day",
            "exclude_imputed_values",
            "exclude_abnormal_domain_days_after_initialization",
        },
        path,
    )
    config = BaselineConfig(
        history_lookback_calendar_days=_integer(
            values,
            "history_lookback_calendar_days",
            path,
        ),
        initial_days=_integer(values, "initial_days", path),
        stable_days=_integer(values, "stable_days", path),
        max_valid_days=_integer(values, "max_valid_days", path),
        center_method=_string(values, "center_method", path),
        quantile_interpolation=_string(values, "quantile_interpolation", path),
        lower_quantile=_number(values, "lower_quantile", path),
        upper_quantile=_number(values, "upper_quantile", path),
        mad_scale_factor=_number(values, "mad_scale_factor", path),
        quantile_scale_divisor=_number(values, "quantile_scale_divisor", path),
        relative_scale_fraction=_number(values, "relative_scale_fraction", path),
        absolute_scale_floor=_number(values, "absolute_scale_floor", path),
        standardized_component_divisor=_number(
            values,
            "standardized_component_divisor",
            path,
        ),
        relative_denominator_floor=_number(
            values,
            "relative_denominator_floor",
            path,
        ),
        relative_change_scale=_number(values, "relative_change_scale", path),
        deviation_aggregation=_string(values, "deviation_aggregation", path),
        abnormal_score_threshold=_number(
            values,
            "abnormal_score_threshold",
            path,
        ),
        exclude_current_day=_boolean(values, "exclude_current_day", path),
        exclude_imputed_values=_boolean(
            values,
            "exclude_imputed_values",
            path,
        ),
        exclude_abnormal_domain_days_after_initialization=_boolean(
            values,
            "exclude_abnormal_domain_days_after_initialization",
            path,
        ),
    )
    if (
        config.history_lookback_calendar_days,
        config.initial_days,
        config.stable_days,
        config.max_valid_days,
    ) != (28, 3, 7, 14):
        raise ValueError(f"{path} days must remain frozen at 28/3/7/14")
    if (
        config.center_method,
        config.quantile_interpolation,
        config.lower_quantile,
        config.upper_quantile,
        config.mad_scale_factor,
        config.quantile_scale_divisor,
        config.relative_scale_fraction,
        config.absolute_scale_floor,
        config.standardized_component_divisor,
        config.relative_denominator_floor,
        config.relative_change_scale,
        config.deviation_aggregation,
        config.abnormal_score_threshold,
    ) != (
        "median",
        "linear",
        0.10,
        0.90,
        1.4826,
        2.563,
        0.05,
        0.05,
        2.0,
        0.05,
        0.50,
        "max",
        0.60,
    ):
        raise ValueError(f"{path} robust deviation parameters must match V3.3.3")
    if not (
        config.exclude_current_day
        and config.exclude_imputed_values
        and config.exclude_abnormal_domain_days_after_initialization
    ):
        raise ValueError(
            f"{path} must exclude current, imputed, and post-start abnormal "
            "domain days"
        )
    return config


def _camera_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
    baseline: BaselineConfig,
) -> CameraConfig:
    path = f"{source}: camera"
    _require_exact_keys(
        values,
        {
            "daytime_start",
            "daytime_end",
            "daytime_minutes",
            "activity_window_seconds",
            "active_score_weights",
            "active_score_threshold",
            "low_activity_score_threshold",
            "sedentary_min_minutes",
            "walking_speed",
        },
        path,
    )
    daytime_start = _string(values, "daytime_start", path)
    daytime_end = _string(values, "daytime_end", path)
    start_minutes = _clock_minutes(daytime_start, f"{path}.daytime_start")
    end_minutes = _clock_minutes(daytime_end, f"{path}.daytime_end")
    daytime_minutes = _integer(values, "daytime_minutes", path)
    activity_window_seconds = _integer(values, "activity_window_seconds", path)
    active_threshold = _number(values, "active_score_threshold", path)
    low_threshold = _number(values, "low_activity_score_threshold", path)
    sedentary_min_minutes = _integer(values, "sedentary_min_minutes", path)

    if (
        daytime_start,
        daytime_end,
        daytime_minutes,
        activity_window_seconds,
        active_threshold,
        low_threshold,
        sedentary_min_minutes,
    ) != ("06:00", "18:00", 720, 10, 0.40, 0.20, 30):
        raise ValueError(f"{path} observation parameters must match V3.3.3")
    if end_minutes - start_minutes != daytime_minutes:
        raise ValueError(f"{path}: clock range must equal daytime_minutes")

    weight_values = _mapping(values, "active_score_weights", path)
    _require_exact_keys(
        weight_values,
        set(ACTIVE_SCORE_WEIGHT_FIELDS),
        f"{path}.active_score_weights",
    )
    weights = ActiveScoreWeights(
        **{
            field: _number(
                weight_values,
                field,
                f"{path}.active_score_weights",
            )
            for field in ACTIVE_SCORE_WEIGHT_FIELDS
        }
    )
    expected_weights = {
        "center_motion_score": 0.55,
        "pose_motion_score": 0.30,
        "zone_transition_score": 0.10,
        "posture_change_score": 0.05,
    }
    if weights.as_dict() != expected_weights:
        raise ValueError(
            f"{path}.active_score_weights must remain frozen at "
            "0.55/0.30/0.10/0.05"
        )
    if not math.isclose(sum(weights.as_dict().values()), 1.0, abs_tol=1e-12):
        raise ValueError(f"{path}.active_score_weights must sum to 1")

    walking = _walking_speed_config(
        _mapping(values, "walking_speed", path),
        path=f"{path}.walking_speed",
        baseline=baseline,
    )
    return CameraConfig(
        daytime_start=daytime_start,
        daytime_end=daytime_end,
        daytime_minutes=daytime_minutes,
        activity_window_seconds=activity_window_seconds,
        active_score_weights=weights,
        active_score_threshold=active_threshold,
        low_activity_score_threshold=low_threshold,
        sedentary_min_minutes=sedentary_min_minutes,
        walking_speed=walking,
    )


def _walking_speed_config(
    values: Mapping[str, Any],
    *,
    path: str,
    baseline: BaselineConfig,
) -> WalkingSpeedConfig:
    _require_exact_keys(
        values,
        {
            "history_lookback_calendar_days",
            "min_history_days",
            "max_valid_days",
            "lower_quantile",
            "upper_quantile",
            "denominator_epsilon",
        },
        path,
    )
    config = WalkingSpeedConfig(
        history_lookback_calendar_days=_integer(
            values,
            "history_lookback_calendar_days",
            path,
        ),
        min_history_days=_integer(values, "min_history_days", path),
        max_valid_days=_integer(values, "max_valid_days", path),
        lower_quantile=_number(values, "lower_quantile", path),
        upper_quantile=_number(values, "upper_quantile", path),
        denominator_epsilon=_number(values, "denominator_epsilon", path),
    )
    if (
        config.history_lookback_calendar_days,
        config.min_history_days,
        config.max_valid_days,
        config.lower_quantile,
        config.upper_quantile,
        config.denominator_epsilon,
    ) != (28, 3, 14, 0.10, 0.90, 1e-6):
        raise ValueError(
            f"{path} must use D-28, 3 minimum days, 14 valid days, "
            "Q10/Q90, and epsilon 1e-6"
        )
    if (
        config.history_lookback_calendar_days
        != baseline.history_lookback_calendar_days
        or config.max_valid_days != baseline.max_valid_days
    ):
        raise ValueError(f"{path} history bounds must match baseline")
    return config


def _masking_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> MaskingConfig:
    path = f"{source}: masks"
    _require_exact_keys(
        values,
        {
            "layers",
            "imputed_values_activate_masks",
            "mask_values_are_risk_features",
        },
        path,
    )
    layers = _string_sequence(values, "layers", path)
    if layers != MASK_LAYERS:
        raise ValueError(f"{path}.layers must contain the four V3.3.3 masks")
    config = MaskingConfig(
        layers=(layers[0], layers[1], layers[2], layers[3]),
        imputed_values_activate_masks=_boolean(
            values,
            "imputed_values_activate_masks",
            path,
        ),
        mask_values_are_risk_features=_boolean(
            values,
            "mask_values_are_risk_features",
            path,
        ),
    )
    if (
        config.imputed_values_activate_masks
        or config.mask_values_are_risk_features
    ):
        raise ValueError(
            f"{path} imputed values cannot activate masks and masks cannot be "
            "risk features"
        )
    return config


def _personal_trend_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
    runtime: RuntimeConfig,
) -> PersonalTrendConfig:
    path = f"{source}: personal_trend"
    _require_exact_keys(
        values,
        {
            "domains",
            "slope",
            "persistence",
            "isolation_forest",
            "change_point",
        },
        path,
    )
    domains = _string_sequence(values, "domains", path)
    if domains != PERSONAL_TREND_DOMAINS:
        raise ValueError(f"{path}.domains must remain activity/sleep/social")

    slope_values = _mapping(values, "slope", path)
    slope_path = f"{path}.slope"
    _require_exact_keys(
        slope_values,
        {"method", "minimum_total_points", "use_calendar_day_spacing"},
        slope_path,
    )
    slope = SlopeConfig(
        method=_string(slope_values, "method", slope_path),
        minimum_total_points=_integer(
            slope_values,
            "minimum_total_points",
            slope_path,
        ),
        use_calendar_day_spacing=_boolean(
            slope_values,
            "use_calendar_day_spacing",
            slope_path,
        ),
    )
    if (
        slope.method,
        slope.minimum_total_points,
        slope.use_calendar_day_spacing,
    ) != ("theil_sen", 3, True):
        raise ValueError(
            f"{slope_path} must use deterministic Theil-Sen with at least "
            "three calendar-spaced points"
        )

    persistence_values = _mapping(values, "persistence", path)
    persistence_path = f"{path}.persistence"
    _require_exact_keys(
        persistence_values,
        {"missing_day_breaks_sequence"},
        persistence_path,
    )
    missing_day_breaks_sequence = _boolean(
        persistence_values,
        "missing_day_breaks_sequence",
        persistence_path,
    )
    _require_frozen_value(
        missing_day_breaks_sequence,
        True,
        f"{persistence_path}.missing_day_breaks_sequence",
    )

    isolation_values = _mapping(values, "isolation_forest", path)
    isolation_path = f"{path}.isolation_forest"
    _require_exact_keys(
        isolation_values,
        {
            "stable_baseline_only",
            "n_estimators",
            "contamination",
            "random_seed",
            "minimum_common_features",
        },
        isolation_path,
    )
    isolation = IsolationForestConfig(
        stable_baseline_only=_boolean(
            isolation_values,
            "stable_baseline_only",
            isolation_path,
        ),
        n_estimators=_integer(
            isolation_values,
            "n_estimators",
            isolation_path,
        ),
        contamination=_number(
            isolation_values,
            "contamination",
            isolation_path,
        ),
        random_seed=_integer(
            isolation_values,
            "random_seed",
            isolation_path,
        ),
        minimum_common_features=_integer(
            isolation_values,
            "minimum_common_features",
            isolation_path,
        ),
    )
    if (
        isolation.stable_baseline_only,
        isolation.n_estimators,
        isolation.contamination,
        isolation.random_seed,
        isolation.minimum_common_features,
    ) != (True, 100, 0.15, runtime.random_seed, 2):
        raise ValueError(
            f"{isolation_path} must remain stable-only, 100 trees, "
            "contamination 0.15, seed 20260728, and two common features"
        )

    change_values = _mapping(values, "change_point", path)
    change_path = f"{path}.change_point"
    _require_exact_keys(
        change_values,
        {"minimum_valid_points", "recent_window_points"},
        change_path,
    )
    change_point = ChangePointConfig(
        minimum_valid_points=_integer(
            change_values,
            "minimum_valid_points",
            change_path,
        ),
        recent_window_points=_integer(
            change_values,
            "recent_window_points",
            change_path,
        ),
    )
    if (
        change_point.minimum_valid_points,
        change_point.recent_window_points,
    ) != (6, 3):
        raise ValueError(
            f"{change_path} must use six valid points and a three-point "
            "recent window"
        )
    return PersonalTrendConfig(
        domains=(domains[0], domains[1], domains[2]),
        slope=slope,
        missing_day_breaks_sequence=missing_day_breaks_sequence,
        isolation_forest=isolation,
        change_point=change_point,
    )


def _fusion_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> FusionConfig:
    path = f"{source}: fusion"
    _require_exact_keys(
        values,
        {
            "method",
            "probability_clip_epsilon",
            "confidence_mode",
            "independent_confidence_risk_term",
            "mask_values_are_risk_features",
            "effective_evidence",
        },
        path,
    )
    evidence_values = _mapping(values, "effective_evidence", path)
    evidence_path = f"{path}.effective_evidence"
    _require_exact_keys(
        evidence_values,
        {
            "supervised_term",
            "personal_change_term",
            "unavailable_at_or_below",
            "run_intercept_when_unavailable",
        },
        evidence_path,
    )
    evidence = EffectiveEvidenceConfig(
        supervised_term=_string(
            evidence_values,
            "supervised_term",
            evidence_path,
        ),
        personal_change_term=_string(
            evidence_values,
            "personal_change_term",
            evidence_path,
        ),
        unavailable_at_or_below=_number(
            evidence_values,
            "unavailable_at_or_below",
            evidence_path,
        ),
        run_intercept_when_unavailable=_boolean(
            evidence_values,
            "run_intercept_when_unavailable",
            evidence_path,
        ),
    )
    config = FusionConfig(
        method=_string(values, "method", path),
        probability_clip_epsilon=_number(
            values,
            "probability_clip_epsilon",
            path,
        ),
        confidence_mode=_string(values, "confidence_mode", path),
        independent_confidence_risk_term=_boolean(
            values,
            "independent_confidence_risk_term",
            path,
        ),
        mask_values_are_risk_features=_boolean(
            values,
            "mask_values_are_risk_features",
            path,
        ),
        effective_evidence=evidence,
    )
    if (
        config.method,
        config.probability_clip_epsilon,
        config.confidence_mode,
        config.independent_confidence_risk_term,
        config.mask_values_are_risk_features,
        config.effective_evidence.supervised_term,
        config.effective_evidence.personal_change_term,
        config.effective_evidence.unavailable_at_or_below,
        config.effective_evidence.run_intercept_when_unavailable,
    ) != (
        "masked_logistic_stacking",
        1e-6,
        "decay_only",
        False,
        False,
        "expert_mask_times_confidence",
        "personal_change_mask_times_reliability",
        0.0,
        False,
    ):
        raise ValueError(
            f"{path} must use V3.3.3 masked logistic stacking, decay-only "
            "confidence, and effective-evidence hard gating"
        )
    return config


def _attention_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> AttentionConfig:
    path = f"{source}: attention"
    _require_exact_keys(values, {"thresholds", "score_rounding"}, path)
    threshold_values = _mapping(values, "thresholds", path)
    threshold_path = f"{path}.thresholds"
    _require_exact_keys(
        threshold_values,
        {"level_1", "level_2", "level_3"},
        threshold_path,
    )
    thresholds = tuple(
        _number(threshold_values, field, threshold_path)
        for field in ("level_1", "level_2", "level_3")
    )
    score_rounding = _string(values, "score_rounding", path)
    if thresholds != (0.25, 0.45, 0.65) or score_rounding != "half_up":
        raise ValueError(
            f"{path} must use thresholds 0.25/0.45/0.65 and half-up scores"
        )
    return AttentionConfig(
        thresholds=(thresholds[0], thresholds[1], thresholds[2]),
        score_rounding=score_rounding,
    )


def _trend_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> TrendConfig:
    path = f"{source}: trend"
    _require_exact_keys(
        values,
        {
            "max_history_results",
            "include_current_result",
            "same_model_version_only",
            "minimum_points",
            "method",
            "use_calendar_day_spacing",
            "rising_threshold_per_day",
            "falling_threshold_per_day",
        },
        path,
    )
    config = TrendConfig(
        max_history_results=_integer(values, "max_history_results", path),
        include_current_result=_boolean(
            values,
            "include_current_result",
            path,
        ),
        same_model_version_only=_boolean(
            values,
            "same_model_version_only",
            path,
        ),
        minimum_points=_integer(values, "minimum_points", path),
        method=_string(values, "method", path),
        use_calendar_day_spacing=_boolean(
            values,
            "use_calendar_day_spacing",
            path,
        ),
        rising_threshold_per_day=_number(
            values,
            "rising_threshold_per_day",
            path,
        ),
        falling_threshold_per_day=_number(
            values,
            "falling_threshold_per_day",
            path,
        ),
    )
    if (
        config.max_history_results,
        config.include_current_result,
        config.same_model_version_only,
        config.minimum_points,
        config.method,
        config.use_calendar_day_spacing,
        config.rising_threshold_per_day,
        config.falling_threshold_per_day,
    ) != (
        6,
        True,
        True,
        3,
        "ordinary_least_squares",
        True,
        0.03,
        -0.03,
    ):
        raise ValueError(
            f"{path} must use current plus six same-version results, "
            "three-point calendar OLS, and +/-0.03 per day"
        )
    return config


def _device_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
) -> DeviceConfig:
    path = f"{source}: devices"
    _require_exact_keys(
        values,
        {"source_codes", "supported_combinations", "profile_is_device"},
        path,
    )
    source_values = _mapping(values, "source_codes", path)
    _require_exact_keys(source_values, set(DEVICE_SOURCE_CODES), f"{path}.source_codes")
    source_codes = {
        code: _string(source_values, code, f"{path}.source_codes")
        for code in DEVICE_SOURCE_CODES
    }
    if source_codes != DEVICE_SOURCE_CODES:
        raise ValueError(f"{path}.source_codes must remain C/S/T")

    combinations_raw = _sequence(values, "supported_combinations", path)
    combinations: list[tuple[str, ...]] = []
    for index, raw_combination in enumerate(combinations_raw):
        combination_path = f"{path}.supported_combinations[{index}]"
        if not isinstance(raw_combination, Sequence) or isinstance(
            raw_combination,
            (str, bytes),
        ):
            raise ValueError(f"{combination_path} must be an array of sources")
        combination = tuple(
            _non_empty_string(item, combination_path)
            for item in raw_combination
        )
        combinations.append(combination)
    combination_tuple = tuple(combinations)
    if combination_tuple != SUPPORTED_DEVICE_COMBINATIONS:
        raise ValueError(
            f"{path}.supported_combinations must contain the seven C/S/T "
            "combinations"
        )
    profile_is_device = _boolean(values, "profile_is_device", path)
    _require_frozen_value(
        profile_is_device,
        False,
        f"{path}.profile_is_device",
    )
    return DeviceConfig(
        source_codes=MappingProxyType(source_codes),
        supported_combinations=combination_tuple,
        profile_is_device=profile_is_device,
    )


def _model_artifacts_config(
    values: Mapping[str, Any],
    *,
    source: str | Path,
    project_root: Path,
) -> ModelArtifactsConfig:
    path = f"{source}: models"
    _require_exact_keys(values, {"version", "directory", "artifacts"}, path)
    version = _string(values, "version", path)
    directory_value = _string(values, "directory", path)
    _require_frozen_value(version, "mood-fusion-v3.3.3", f"{path}.version")
    _require_frozen_value(
        directory_value.replace("\\", "/").rstrip("/"),
        "models/mental_health/mood_social/v3.3.3",
        f"{path}.directory",
    )

    artifact_values = _mapping(values, "artifacts", path)
    _require_exact_keys(
        artifact_values,
        set(MODEL_ARTIFACT_FILES),
        f"{path}.artifacts",
    )
    artifacts = {
        key: _string(artifact_values, key, f"{path}.artifacts")
        for key in MODEL_ARTIFACT_FILES
    }
    if artifacts != MODEL_ARTIFACT_FILES:
        raise ValueError(f"{path}.artifacts must match the mood-social manifest")
    return ModelArtifactsConfig(
        version=version,
        directory=project_root / Path(directory_value),
        artifacts=MappingProxyType(artifacts),
    )


def _mapping(
    values: Mapping[str, Any],
    key: str,
    parent: str | Path,
) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent}.{key} must be a mapping")
    return value


def _sequence(
    values: Mapping[str, Any],
    key: str,
    parent: str | Path,
) -> Sequence[Any]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{parent}.{key} must be an array")
    return value


def _string_sequence(
    values: Mapping[str, Any],
    key: str,
    parent: str | Path,
) -> tuple[str, ...]:
    path = f"{parent}.{key}"
    return tuple(
        _non_empty_string(item, path)
        for item in _sequence(values, key, parent)
    )


def _non_empty_string(value: Any, path: str | Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} entries must be non-empty strings")
    return value.strip()


def _string(values: Mapping[str, Any], key: str, parent: str | Path) -> str:
    return _non_empty_string(values.get(key), f"{parent}.{key}")


def _integer(values: Mapping[str, Any], key: str, parent: str | Path) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{parent}.{key} must be an integer")
    return value


def _number(values: Mapping[str, Any], key: str, parent: str | Path) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{parent}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{parent}.{key} must be a finite number")
    return number


def _boolean(values: Mapping[str, Any], key: str, parent: str | Path) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{parent}.{key} must be a boolean")
    return value


def _parse_clock(value: str, path: str) -> time:
    if len(value) != 5 or value[2] != ":":
        raise ValueError(f"{path} must use HH:MM format")
    try:
        parsed = time(hour=int(value[:2]), minute=int(value[3:]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must use a valid HH:MM time") from exc
    if parsed.strftime("%H:%M") != value:
        raise ValueError(f"{path} must use zero-padded HH:MM format")
    return parsed


def _clock_minutes(value: str, path: str) -> int:
    parsed = _parse_clock(value, path)
    return parsed.hour * 60 + parsed.minute


def _require_exact_keys(
    values: Mapping[str, Any],
    expected: set[str],
    path: str | Path,
) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} keys mismatch; missing={missing}, extra={extra}")


def _require_frozen_value(actual: object, expected: object, path: str) -> None:
    if actual != expected:
        raise ValueError(f"{path} must remain frozen at {expected!r}")
