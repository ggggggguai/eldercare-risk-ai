"""Deployable V3.4 mood-risk forecast runtime.

The V3.3.3 pipeline estimates the current cross-sectional attention state.  This
module is deliberately separate: it predicts PHQ-9 high-risk screening status
one and two months ahead.  Strictly historical PHQ-9 assessments select the
strong state-aware route; otherwise the runtime falls back to passive signals.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - handled as an unavailable forecast branch
    CatBoostClassifier = None  # type: ignore[assignment,misc]


FORECAST_MODEL_VERSION = "mood-forecast-v3.4.0"
FORECAST_ARTIFACT_RUN_ID = "MH-20260810-FDEP-001"
DEFAULT_PACKAGE = (
    Path(__file__).resolve().parents[5]
    / "models/mental_health/mood_social/v3.4.0/packages"
    / FORECAST_ARTIFACT_RUN_ID
)
HORIZONS = ("forecast_1m", "forecast_2m")
HISTORY_FEATURES = (
    "historical_phq9_last_score",
    "historical_phq9_age_months",
    "historical_phq9_count",
    "historical_phq9_mean",
    "historical_phq9_median",
    "historical_phq9_std",
    "historical_phq9_min",
    "historical_phq9_max",
    "historical_phq9_slope",
    "historical_phq9_last_delta",
)


class MoodForecastUnavailableError(RuntimeError):
    """Raised when the frozen V3.4 package cannot be loaded safely."""


@dataclass(frozen=True)
class _LoadedPackage:
    root: Path
    manifest: Mapping[str, Any]
    models: Mapping[str, Mapping[str, tuple[Any, ...]]]


def _package_directory() -> Path:
    override = os.getenv("MOOD_SOCIAL_FORECAST_PACKAGE_DIR")
    return Path(override).resolve() if override else DEFAULT_PACKAGE.resolve()


@lru_cache(maxsize=4)
def _load_package(package_directory: str) -> _LoadedPackage:
    if CatBoostClassifier is None:
        raise MoodForecastUnavailableError("CatBoost is not installed")
    root = Path(package_directory)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MoodForecastUnavailableError("forecast manifest is unavailable") from exc
    if (
        manifest.get("model_version") != FORECAST_MODEL_VERSION
        or manifest.get("artifact_run_id") != FORECAST_ARTIFACT_RUN_ID
    ):
        raise MoodForecastUnavailableError("forecast package identity mismatch")

    loaded: dict[str, dict[str, tuple[Any, ...]]] = {}
    try:
        for task_id in HORIZONS:
            loaded[task_id] = {}
            for mode in ("state_aware", "passive_fallback"):
                task_files = manifest["model_files"][task_id][mode]
                if not isinstance(task_files, list) or not task_files:
                    raise ValueError("empty model ensemble")
                models = []
                for relative in task_files:
                    model_path = (root / str(relative)).resolve()
                    if root not in model_path.parents:
                        raise ValueError("model path escapes package")
                    model = CatBoostClassifier()
                    model.load_model(str(model_path))
                    models.append(model)
                loaded[task_id][mode] = tuple(models)
    except (KeyError, OSError, ValueError) as exc:
        raise MoodForecastUnavailableError("forecast ensemble is incomplete") from exc
    return _LoadedPackage(root=root, manifest=manifest, models=loaded)


def clear_forecast_runtime_cache() -> None:
    _load_package.cache_clear()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="python")
    raise TypeError("forecast request must be a mapping or Pydantic model")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _history_features(
    assessments: Sequence[Mapping[str, Any]], target_date: date
) -> dict[str, float]:
    if not assessments:
        return {name: float("nan") for name in HISTORY_FEATURES}
    ordered = sorted(assessments, key=lambda item: item["date"])
    dates = [
        value if isinstance(value := item["date"], date) else date.fromisoformat(value)
        for item in ordered
    ]
    scores = np.asarray([float(item["score"]) for item in ordered], dtype="float64")
    month_positions = np.asarray(
        [(value - dates[0]).days / 30.4375 for value in dates], dtype="float64"
    )
    slope = (
        float(np.polyfit(month_positions, scores, 1)[0])
        if len(scores) >= 2 and np.unique(month_positions).size >= 2
        else 0.0
    )
    return {
        "historical_phq9_last_score": float(scores[-1]),
        "historical_phq9_age_months": float(
            max(0, (target_date - dates[-1]).days) / 30.4375
        ),
        "historical_phq9_count": float(len(scores)),
        "historical_phq9_mean": float(scores.mean()),
        "historical_phq9_median": float(np.median(scores)),
        "historical_phq9_std": float(scores.std(ddof=0)),
        "historical_phq9_min": float(scores.min()),
        "historical_phq9_max": float(scores.max()),
        "historical_phq9_slope": slope,
        "historical_phq9_last_delta": (
            float(scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0
        ),
    }


def _daily_signal(record: Mapping[str, Any], base_name: str) -> float | None:
    activity = record.get("activity") or {}
    sleep = record.get("sleep") or {}
    if base_name.startswith("steps"):
        value = _finite_number(activity.get("weighted_daytime_activity"))
        if value is None:
            value = _finite_number(activity.get("daytime_active_minutes"))
        # PSYCHE-D uses step-scale activity.  The service only exposes activity
        # minutes, so the deployment adapter uses a frozen 80 steps/minute scale.
        return None if value is None else value * 80.0
    if base_name == "sleep" or "sleep_asleep" in base_name:
        return _finite_number(sleep.get("sleep_minutes"))
    if "sleep_in_bed" in base_name:
        return _finite_number(sleep.get("in_bed_minutes"))
    if "sleep_ratio_asleep_in_bed" in base_name:
        return _finite_number(sleep.get("sleep_efficiency"))
    if "sleep_main_start_hour" in base_name:
        value = _finite_number(sleep.get("sleep_onset_minute_of_day"))
        return None if value is None else value / 60.0
    return None


def _period_stat(records: Sequence[Mapping[str, Any]], base_name: str) -> float | None:
    pairs = [
        (record["date"], value)
        for record in records
        if (value := _daily_signal(record, base_name)) is not None
    ]
    if not pairs:
        return None
    values = np.asarray([item[1] for item in pairs], dtype="float64")
    if base_name.endswith("_iqr"):
        return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if "weekday_mean" in base_name or "weekend_mean" in base_name:
        weekend = "weekend_mean" in base_name
        chosen = [
            value
            for raw_date, value in pairs
            if ((raw_date.weekday() >= 5) if weekend else (raw_date.weekday() < 5))
        ]
        return float(np.mean(chosen)) if chosen else None
    if "sum_recent" in base_name:
        return float(values[-7:].sum())
    if "rolling_6_max_recent" in base_name:
        window = values[-12:]
        if len(window) < 6:
            return float(window.max())
        return float(
            max(np.sum(window[index : index + 6]) for index in range(len(window) - 5))
        )
    if "rolling_6_median_recent" in base_name:
        return float(np.median(values[-7:]))
    if "mean_recent" in base_name:
        return float(values[-7:].mean())
    return float(values.mean())


def _passive_features(
    request: Mapping[str, Any], feature_names: Sequence[str]
) -> dict[str, float]:
    target = request["target_date"]
    target_date = target if isinstance(target, date) else date.fromisoformat(target)
    records = [
        *request.get("history_daily_features", []),
        request["current_daily_features"],
    ]
    normalized = []
    for raw in records:
        item = dict(_as_mapping(raw))
        raw_date = item["date"]
        item["date"] = (
            raw_date if isinstance(raw_date, date) else date.fromisoformat(raw_date)
        )
        normalized.append(item)
    normalized.sort(key=lambda item: item["date"])

    current = normalized[-1]
    numeric_current = 0
    for domain in ("activity", "sleep", "physiology", "social"):
        for value in (current.get(domain) or {}).values():
            numeric_current += int(_finite_number(value) is not None)
    observed = [
        item
        for item in normalized
        if any(item.get(name) for name in ("activity", "sleep"))
    ]
    history_observed = [item for item in observed if item["date"] < target_date]
    span_days = (
        (history_observed[-1]["date"] - history_observed[0]["date"]).days
        if len(history_observed) >= 2
        else 0
    )
    since_last = (
        (target_date - history_observed[-1]["date"]).days / 30.4375
        if history_observed
        else 1.0
    )
    bases = sorted(
        {
            name.split("__", 2)[1]
            for name in feature_names
            if name.startswith("short__") and name.count("__") >= 2
        }
    )
    split_date = date.fromordinal(target_date.toordinal() - 14)
    previous_records = [item for item in normalized if item["date"] < split_date]
    recent_records = [item for item in normalized if item["date"] >= split_date]
    base_values: dict[str, dict[str, float | None]] = {}
    for base in bases:
        previous = _period_stat(previous_records, base)
        recent = _period_stat(recent_records, base)
        delta = None if previous is None or recent is None else recent - previous
        scale = None if previous is None else max(abs(previous) * 0.1, 1.0)
        base_values[base] = {
            "anchor": recent,
            "delta_1": delta,
            "local_slope": delta,
            "personal_z": None if delta is None or scale is None else delta / scale,
        }

    values = {name: float("nan") for name in feature_names}
    direct = {
        "feature_nonmissing_count": float(numeric_current),
        "history_observed_month_count": float(int(bool(history_observed))),
        "history_month_span": float(span_days / 30.4375),
        "history_months_since_last_any_record": float(since_last),
        "baseline_available_count": float(
            sum(_period_stat(normalized, base) is not None for base in bases)
        ),
        "cold_start_field_count": float(
            sum(base_values[base]["delta_1"] is None for base in bases)
        ),
    }
    values.update({key: value for key, value in direct.items() if key in values})
    for name in feature_names:
        if not name.startswith("short__"):
            continue
        _, base, suffix = name.split("__", 2)
        if suffix in base_values.get(base, {}):
            raw = base_values[base][suffix]
            values[name] = float("nan") if raw is None else float(raw)

    aggregate: dict[str, list[float]] = {
        key: [] for key in ("delta_1", "local_slope", "personal_z")
    }
    for base in bases:
        for key in aggregate:
            raw = base_values[base][key]
            if raw is not None and math.isfinite(float(raw)):
                aggregate[key].append(float(raw))
    for key, items in aggregate.items():
        for statistic, value in (
            ("mean", np.mean(items) if items else np.nan),
            ("median", np.median(items) if items else np.nan),
            ("abs_mean", np.mean(np.abs(items)) if items else np.nan),
            ("missing_count", len(bases) - len(items)),
        ):
            name = f"opt003__{key}__{statistic}"
            if name in values:
                values[name] = float(value)
    return values


def _risk_level(probability: float, threshold: float) -> int:
    if probability < 0.25:
        return 0
    if probability < threshold:
        return 1
    if probability < 0.75:
        return 2
    return 3


def unavailable_forecast(reason: str) -> dict[str, Any]:
    return {
        "model_version": FORECAST_MODEL_VERSION,
        "artifact_run_id": FORECAST_ARTIFACT_RUN_ID,
        "available": False,
        "mode": "insufficient_data",
        "historical_phq9_used": False,
        "historical_phq9_count": 0,
        "used_sources": [],
        "horizons": [],
        "evidence_scope": "unavailable",
        "evaluated_auprc_1m": None,
        "evaluated_auprc_2m": None,
        "summary": "未来风险预测暂不可用，当前横截面状态判断仍可正常返回。",
        "limitations": [reason, "该结果不构成医学诊断。"],
        "diagnosis": False,
    }


def predict_mood_forecast(request: Any) -> dict[str, Any]:
    parsed = _as_mapping(request)
    try:
        package = _load_package(str(_package_directory()))
    except MoodForecastUnavailableError as exc:
        return unavailable_forecast(str(exc))

    raw_assessments = [
        dict(_as_mapping(item))
        for item in parsed.get("historical_phq9_assessments", [])
    ]
    mode = "state_aware" if raw_assessments else "passive_fallback"
    daily_records = [
        *parsed.get("history_daily_features", []),
        parsed["current_daily_features"],
    ]
    passive_observed = any(
        _as_mapping(item).get("activity") is not None
        or _as_mapping(item).get("sleep") is not None
        for item in daily_records
    )
    if mode == "passive_fallback" and not passive_observed:
        return unavailable_forecast(
            "passive forecast requires at least one activity or sleep record"
        )
    target = parsed["target_date"]
    target_date = target if isinstance(target, date) else date.fromisoformat(target)
    history = _history_features(raw_assessments, target_date)
    horizons = []
    for task_id in HORIZONS:
        feature_names = package.manifest["feature_names"][task_id][mode]
        values = _passive_features(parsed, feature_names)
        if mode == "state_aware":
            values.update(history)
        frame = pd.DataFrame(
            [[values.get(name, np.nan) for name in feature_names]],
            columns=feature_names,
        )
        probabilities = [
            float(model.predict_proba(frame)[:, 1][0])
            for model in package.models[task_id][mode]
        ]
        probability = float(np.mean(probabilities))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            return unavailable_forecast(
                "forecast model produced an invalid probability"
            )
        threshold = float(package.manifest["thresholds"][task_id][mode])
        horizons.append(
            {
                "task_id": task_id,
                "months_ahead": 1 if task_id == "forecast_1m" else 2,
                "risk_probability": probability,
                "risk_score": int(math.floor(probability * 100.0 + 0.5)),
                "risk_level": _risk_level(probability, threshold),
                "high_risk_threshold": threshold,
                "predicted_high_risk": probability >= threshold,
            }
        )

    evidence = package.manifest["evidence"][mode]
    used_sources = []
    daily = daily_records
    if any(_as_mapping(item).get("activity") is not None for item in daily):
        used_sources.append("camera")
    if any(_as_mapping(item).get("sleep") is not None for item in daily):
        used_sources.append("sleep_device")
    if mode == "state_aware":
        used_sources.append("historical_phq9")
    summary = (
        "已使用严格早于预测日的历史 PHQ-9 与被动活动/睡眠信号预测未来1个月和2个月风险。"
        if mode == "state_aware"
        else "当前无历史 PHQ-9，已自动使用纯被动活动/睡眠回退模型预测未来风险。"
    )
    limitations = [
        "线上包为冻结结构的全量重训模型；展示指标来自同路线严格参与者隔离 OOF，不是独立外部验证。",
        "日级活动字段通过冻结适配器转换为 PSYCHE-D 月级特征语义，真实部署后应继续做分布漂移监测。",
        "该结果仅用于关注趋势和筛查提醒，不构成医学诊断。",
    ]
    if mode == "passive_fallback":
        limitations.insert(0, "未使用历史 PHQ-9，不能引用状态感知模型约0.78的AUPRC。")
    return {
        "model_version": FORECAST_MODEL_VERSION,
        "artifact_run_id": FORECAST_ARTIFACT_RUN_ID,
        "available": True,
        "mode": mode,
        "historical_phq9_used": mode == "state_aware",
        "historical_phq9_count": len(raw_assessments),
        "used_sources": used_sources,
        "horizons": horizons,
        "evidence_scope": evidence["scope"],
        "evaluated_auprc_1m": float(evidence["forecast_1m_auprc"]),
        "evaluated_auprc_2m": float(evidence["forecast_2m_auprc"]),
        "summary": summary,
        "limitations": limitations,
        "diagnosis": False,
    }


__all__ = [
    "FORECAST_ARTIFACT_RUN_ID",
    "FORECAST_MODEL_VERSION",
    "MoodForecastUnavailableError",
    "clear_forecast_runtime_cache",
    "predict_mood_forecast",
    "unavailable_forecast",
]
