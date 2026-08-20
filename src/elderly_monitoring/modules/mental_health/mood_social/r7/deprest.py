"""Leakage-safe DepreST-CAT adapter for the R7 social-contact expert.

The source notebooks include assessment-day events and hundreds of precomputed
features.  R7 instead rebuilds a compact, S10-compatible calls-only table from
raw events.  Calendar-day zero is excluded because the source survey timestamp
is local-naive while call timestamps are Unix UTC and the source timezone is
not documented.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SURVEY_FILE = "surveysDepreST-CAT.csv"
CALL_FILE = "callLogsDepreST-CAT2021.csv"
TEXT_FILE = "textLogsDepreST-CAT2021.csv"
KNOWN_CALL_DIRECTIONS = frozenset({1, 2})
MAIN_WINDOWS = (14, 28)
SENSITIVITY_WINDOWS = (56, 112)

SOCIAL_MAIN_FEATURES = (
    "social.call_count_14d",
    "social.connected_call_count_14d",
    "social.call_duration_minutes_14d",
    "social.mean_connected_duration_minutes_14d",
    "social.active_call_days_14d",
    "social.no_call_days_14d",
    "social.longest_no_call_streak_14d",
    "social.active_contact_count_14d",
    "social.contact_entropy_14d",
    "social.contact_concentration_14d",
    "social.zero_duration_ratio_14d",
    "social.daily_call_count_mad_14d",
    "social.call_count_28d",
    "social.connected_call_count_28d",
    "social.call_duration_minutes_28d",
    "social.active_call_days_28d",
    "social.no_call_days_28d",
    "social.active_contact_count_28d",
    "social.recent_vs_prior_call_count_delta",
    "social.recent_vs_prior_call_count_ratio",
    "social.recent_vs_prior_duration_delta_minutes",
    "social.recent_vs_prior_active_days_delta",
)

SOCIAL_SENSITIVITY_FEATURES = tuple(
    f"social.{metric}_{window}d"
    for window in SENSITIVITY_WINDOWS
    for metric in (
        "call_count",
        "connected_call_count",
        "call_duration_minutes",
        "active_call_days",
        "active_contact_count",
    )
)

# Exact subset reproducible from the current S10/miniprogram daily call schema.
# Contact entropy/concentration remain research-only until the runtime contract
# exposes a same-semantics generator.
SOCIAL_PRODUCTION_FEATURES = tuple(
    name
    for name in SOCIAL_MAIN_FEATURES
    if not any(
        token in name
        for token in (
            "contact_entropy",
            "contact_concentration",
        )
    )
)


def pseudonymize_participant_id(value: object) -> str:
    digest = hashlib.sha256(f"deprest-cat-r7|{value}".encode("utf-8")).hexdigest()
    return f"deprest_r7::{digest[:24]}"


@dataclass(frozen=True)
class DeprestPaths:
    root: Path
    survey: Path
    calls: Path
    texts: Path
    readme: Path


def locate_deprest_root(project_root: Path) -> DeprestPaths:
    candidates = list(Path(project_root).rglob(SURVEY_FILE))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one local {SURVEY_FILE}, found {len(candidates)}"
        )
    root = candidates[0].parent.resolve()
    paths = DeprestPaths(
        root=root,
        survey=root / SURVEY_FILE,
        calls=root / CALL_FILE,
        texts=root / TEXT_FILE,
        readme=root / "README.md",
    )
    for path in (paths.survey, paths.calls, paths.texts, paths.readme):
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _parse_survey_timestamp(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.strip('"')
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if parsed.isna().any():
        raise ValueError("DepreST-CAT contains unparseable survey timestamps")
    return parsed


def load_surveys(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"id", "PHQ - Total", "Age", "Timestamp"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"DepreST-CAT survey columns missing: {missing}")
    if raw["id"].isna().any() or raw["id"].duplicated().any():
        raise ValueError("DepreST-CAT surveys must contain one non-null row per id")
    score = pd.to_numeric(raw["PHQ - Total"], errors="raise")
    if not score.between(0, 27).all():
        raise ValueError("DepreST-CAT PHQ total is outside 0..27")
    result = pd.DataFrame(
        {
            "participant_id": raw["id"].map(pseudonymize_participant_id),
            "assessment_time_source_naive": _parse_survey_timestamp(raw["Timestamp"]),
            "age_group": raw["Age"].fillna("unknown").astype(str).replace({"-": "unknown"}),
            "phq9_score_target": score.astype(int),
        }
    )
    result["phq9_ge5_target"] = result["phq9_score_target"].ge(5).astype(int)
    result["phq9_ge10_target"] = result["phq9_score_target"].ge(10).astype(int)
    return result.sort_values("participant_id", kind="stable").reset_index(drop=True)


def load_call_events(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"id", "Direction", "Duration", "UnixTimestamp", "Contact"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"DepreST-CAT call columns missing: {missing}")
    if raw["id"].isna().any():
        raise ValueError("DepreST-CAT call log contains missing participant id")
    timestamp = pd.to_numeric(raw["UnixTimestamp"], errors="coerce")
    duration = pd.to_numeric(raw["Duration"], errors="coerce")
    direction = pd.to_numeric(raw["Direction"], errors="coerce")
    if timestamp.isna().any() or duration.isna().any() or direction.isna().any():
        raise ValueError("DepreST-CAT call log contains non-numeric core values")
    if duration.lt(0).any():
        raise ValueError("DepreST-CAT call duration contains negative values")
    result = pd.DataFrame(
        {
            "participant_id": raw["id"].map(pseudonymize_participant_id),
            "event_time_utc": pd.to_datetime(timestamp, unit="ms", utc=True, errors="coerce"),
            "direction_code": direction.astype(int),
            "duration_seconds": duration.astype(float),
            "contact_hash": raw["Contact"].astype("string"),
        }
    )
    if result["event_time_utc"].isna().any():
        raise ValueError("DepreST-CAT call log contains invalid Unix timestamps")
    return result


def _entropy(values: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if counts.size < 2 or counts.sum() <= 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _concentration(values: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(np.square(probabilities).sum())


def _longest_zero_streak(counts: pd.Series) -> int:
    best = current = 0
    for value in counts.to_numpy():
        current = current + 1 if int(value) == 0 else 0
        best = max(best, current)
    return int(best)


def _mad(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    if not len(array):
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _window_summary(events: pd.DataFrame, assessment_day: date, days: int) -> dict[str, float]:
    end = pd.Timestamp(assessment_day, tz="UTC")
    start = end - pd.Timedelta(days=days)
    selected = events.loc[
        events["event_time_utc"].ge(start) & events["event_time_utc"].lt(end)
    ].copy()
    calendar = pd.date_range(start=start, periods=days, freq="D", tz="UTC")
    daily = selected.groupby(selected["event_time_utc"].dt.floor("D")).size().reindex(calendar, fill_value=0)
    connected = selected.loc[selected["duration_seconds"].gt(0)]
    contacts = connected["contact_hash"].dropna().astype(str).tolist()
    duration_minutes = float(connected["duration_seconds"].sum() / 60.0)
    return {
        "call_count": float(len(selected)),
        "connected_call_count": float(len(connected)),
        "call_duration_minutes": duration_minutes,
        "mean_connected_duration_minutes": (
            float(duration_minutes / len(connected)) if len(connected) else 0.0
        ),
        "active_call_days": float(daily.gt(0).sum()),
        "no_call_days": float(daily.eq(0).sum()),
        "longest_no_call_streak": float(_longest_zero_streak(daily)),
        "active_contact_count": float(len(set(contacts))),
        "contact_entropy": _entropy(contacts),
        "contact_concentration": _concentration(contacts),
        "zero_duration_ratio": float(selected["duration_seconds"].eq(0).mean()) if len(selected) else 0.0,
        "daily_call_count_mad": _mad(daily),
    }


def _prior_window_summary(events: pd.DataFrame, assessment_day: date) -> dict[str, float]:
    end = pd.Timestamp(assessment_day, tz="UTC") - pd.Timedelta(days=14)
    start = end - pd.Timedelta(days=14)
    selected = events.loc[
        events["event_time_utc"].ge(start) & events["event_time_utc"].lt(end)
    ].copy()
    calendar = pd.date_range(start=start, periods=14, freq="D", tz="UTC")
    daily = selected.groupby(selected["event_time_utc"].dt.floor("D")).size().reindex(calendar, fill_value=0)
    connected = selected.loc[selected["duration_seconds"].gt(0)]
    return {
        "call_count": float(len(selected)),
        "duration_minutes": float(connected["duration_seconds"].sum() / 60.0),
        "active_days": float(daily.gt(0).sum()),
    }


def build_calls_only_canonical(
    surveys: pd.DataFrame,
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per survey while excluding assessment day and future logs."""

    survey_ids = set(surveys["participant_id"])
    unknown_call_ids = sorted(set(calls["participant_id"]) - survey_ids)
    if unknown_call_ids:
        raise ValueError(f"call log ids missing from surveys: {unknown_call_ids[:3]}")
    rows: list[dict[str, Any]] = []
    by_id = {key: part for key, part in calls.groupby("participant_id", sort=False)}
    for survey in surveys.itertuples(index=False):
        events = by_id.get(survey.participant_id)
        if events is None:
            events = calls.iloc[0:0]
        assessment_day = survey.assessment_time_source_naive.date()
        row: dict[str, Any] = {
            "participant_id": survey.participant_id,
            "assessment_day": assessment_day.isoformat(),
            "age_group": survey.age_group,
            "phq9_score_target": int(survey.phq9_score_target),
            "phq9_ge5_target": int(survey.phq9_ge5_target),
            "phq9_ge10_target": int(survey.phq9_ge10_target),
            "social.valid_days": 14,
            "social.baseline_valid_days": 28,
            "social.no_call_history": bool(len(events) == 0),
        }
        summaries: dict[int, dict[str, float]] = {}
        for window in (*MAIN_WINDOWS, *SENSITIVITY_WINDOWS):
            summaries[window] = _window_summary(events, assessment_day, window)
            for metric, value in summaries[window].items():
                row[f"social.{metric}_{window}d"] = value
        prior = _prior_window_summary(events, assessment_day)
        recent = summaries[14]
        row["social.recent_vs_prior_call_count_delta"] = recent["call_count"] - prior["call_count"]
        row["social.recent_vs_prior_call_count_ratio"] = (recent["call_count"] + 1.0) / (prior["call_count"] + 1.0)
        row["social.recent_vs_prior_duration_delta_minutes"] = recent["call_duration_minutes"] - prior["duration_minutes"]
        row["social.recent_vs_prior_active_days_delta"] = recent["active_call_days"] - prior["active_days"]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("participant_id", kind="stable").reset_index(drop=True)
    if len(frame) != len(surveys) or frame["participant_id"].duplicated().any():
        raise ValueError("DepreST-CAT canonical join is not one-to-one")
    if frame[list(SOCIAL_MAIN_FEATURES)].isna().any().any():
        raise ValueError("DepreST-CAT main social features contain missing values")
    survey_times = surveys.set_index("participant_id")["assessment_time_source_naive"]
    merged = calls.join(survey_times, on="participant_id", how="left", validate="many_to_one")
    event_naive = merged["event_time_utc"].dt.tz_convert(None)
    assessment = merged["assessment_time_source_naive"]
    audit = {
        "canonical_rows": int(len(frame)),
        "canonical_participants": int(frame["participant_id"].nunique()),
        "id_join_one_to_one": True,
        "survey_ids_without_calls": int(frame["social.no_call_history"].sum()),
        "call_rows": int(len(calls)),
        "call_participants": int(calls["participant_id"].nunique()),
        "direction_counts": {
            str(int(key)): int(value)
            for key, value in calls["direction_code"].value_counts().sort_index().items()
        },
        "known_direction_rows": int(calls["direction_code"].isin(KNOWN_CALL_DIRECTIONS).sum()),
        "unknown_direction_rows": int((~calls["direction_code"].isin(KNOWN_CALL_DIRECTIONS)).sum()),
        "direction_features_enabled": False,
        "zero_duration_rows": int(calls["duration_seconds"].eq(0).sum()),
        "negative_duration_rows": int(calls["duration_seconds"].lt(0).sum()),
        "missing_contact_rows": int(calls["contact_hash"].isna().sum()),
        "epoch_or_pre_2000_rows": int(calls["event_time_utc"].dt.year.lt(2000).sum()),
        "same_calendar_day_rows": int(event_naive.dt.date.eq(assessment.dt.date).sum()),
        "same_day_after_assessment_rows_by_naive_comparison": int(
            (event_naive.dt.date.eq(assessment.dt.date) & event_naive.ge(assessment)).sum()
        ),
        "events_after_assessment_by_naive_comparison": int(event_naive.ge(assessment).sum()),
        "production_window_policy": "UTC event calendar day D-14..D-1 / D-28..D-1; day0 excluded",
        "timezone_status": "survey timestamp local-naive; source timezone undocumented; day0 excluded",
        "contact_policy": "hash used only for aggregate count/entropy/concentration and never persisted in canonical",
        "participant_id_policy": "source id replaced by salted one-way SHA-256 prefix before persistence",
    }
    return frame, audit


def assert_social_feature_contract(feature_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in feature_names)
    allowed = set(SOCIAL_MAIN_FEATURES) | set(SOCIAL_SENSITIVITY_FEATURES)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unsupported R7 social risk features: {unknown}")
    forbidden_tokens = (
        "phq", "gad", "appversion", "group", "covid", "remote", "treatment",
        "participant", "contact_hash", "assessment", "timestamp", "direction",
    )
    forbidden = sorted(name for name in names if any(token in name.lower() for token in forbidden_tokens))
    if forbidden:
        raise ValueError(f"forbidden R7 social risk features: {forbidden}")
    return names


__all__ = [
    "CALL_FILE", "DeprestPaths", "KNOWN_CALL_DIRECTIONS", "MAIN_WINDOWS",
    "SENSITIVITY_WINDOWS", "SOCIAL_MAIN_FEATURES", "SOCIAL_SENSITIVITY_FEATURES",
    "SOCIAL_PRODUCTION_FEATURES",
    "SURVEY_FILE", "TEXT_FILE", "assert_social_feature_contract",
    "build_calls_only_canonical", "load_call_events", "load_surveys", "locate_deprest_root",
    "pseudonymize_participant_id",
]
