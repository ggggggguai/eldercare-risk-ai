"""Adapter for the Conscious Wearable external auxiliary dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DATASET_ID = "conscious_wearable"
ADAPTER_VERSION = "conscious-wearable-v1"
COMMON_FEATURES = (
    "steps_awake_mean",
    "steps_awake_sum_iqr",
    "sleep_asleep_weekday_mean",
    "sleep_asleep_weekend_mean",
    "sleep_in_bed_weekday_mean",
    "sleep_in_bed_weekend_mean",
    "sleep_ratio_asleep_in_bed_weekday_mean",
    "sleep_ratio_asleep_in_bed_weekend_mean",
    "sleep_in_bed_iqr",
    "sleep_asleep_iqr",
    "sleep_ratio_asleep_in_bed_iqr",
    "sleep_main_start_hour_adj_median",
    "sleep_main_start_hour_adj_iqr",
    "sleep_main_start_hour_adj_range",
    "sleep__hypersomnia_count_",
    "sleep__hyposomnia_count_",
    "sleep_asleep_mean_recent",
    "sleep_in_bed_mean_recent",
    "sleep_ratio_asleep_in_bed_mean_recent",
)
SLEEP_FEATURES = tuple(name for name in COMMON_FEATURES if name.startswith("sleep_"))
ACTIVITY_FEATURES = tuple(name for name in COMMON_FEATURES if name.startswith("steps_"))


class ConsciousWearableAdapterError(ValueError):
    """Raised when the external dataset violates its frozen schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, Any]:
    required = ("survey.csv", "sleep_diary.csv", "sensor_hrv_filtered.csv")
    files = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise ConsciousWearableAdapterError(f"missing Conscious file: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "dataset_id": DATASET_ID,
        "adapter_version": ADAPTER_VERSION,
        "files": files,
    }


def _iqr(values: Sequence[float]) -> float:
    numeric = np.asarray(values, dtype="float64")
    numeric = numeric[np.isfinite(numeric)]
    return (
        float(np.quantile(numeric, 0.75) - np.quantile(numeric, 0.25))
        if numeric.size
        else np.nan
    )


def _clock_hour(value: Any) -> float:
    parsed = pd.to_timedelta(str(value), errors="coerce")
    if pd.isna(parsed):
        return np.nan
    hour = float(parsed.total_seconds() / 3600.0)
    return hour + 24.0 if hour < 12.0 else hour


def _mean(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> float:
    series = pd.to_numeric(
        frame.loc[mask, column] if mask is not None else frame[column], errors="coerce"
    )
    return float(series.mean()) if series.notna().any() else np.nan


def _window_features(
    diary: pd.DataFrame, sensor: pd.DataFrame
) -> dict[str, float | int]:
    weekday = diary["date"].dt.dayofweek.lt(5)
    weekend = ~weekday
    asleep = pd.to_numeric(diary["sleep_duration"], errors="coerce")
    in_bed = pd.to_numeric(diary["in_bed_duration"], errors="coerce")
    efficiency = pd.to_numeric(diary["sleep_efficiency"], errors="coerce")
    bedtime = diary["go2bed"].map(_clock_hour)
    recent = diary.sort_values("date", kind="mergesort").tail(7)
    daily_steps = (
        sensor.groupby("date", sort=True)["steps"].sum(min_count=1)
        if not sensor.empty
        else pd.Series(dtype="float64")
    )
    return {
        "steps_awake_mean": float(daily_steps.mean()) if len(daily_steps) else np.nan,
        "steps_awake_sum_iqr": _iqr(daily_steps),
        "sleep_asleep_weekday_mean": _mean(diary, "sleep_duration", weekday),
        "sleep_asleep_weekend_mean": _mean(diary, "sleep_duration", weekend),
        "sleep_in_bed_weekday_mean": _mean(diary, "in_bed_duration", weekday),
        "sleep_in_bed_weekend_mean": _mean(diary, "in_bed_duration", weekend),
        "sleep_ratio_asleep_in_bed_weekday_mean": _mean(
            diary, "sleep_efficiency", weekday
        ),
        "sleep_ratio_asleep_in_bed_weekend_mean": _mean(
            diary, "sleep_efficiency", weekend
        ),
        "sleep_in_bed_iqr": _iqr(in_bed),
        "sleep_asleep_iqr": _iqr(asleep),
        "sleep_ratio_asleep_in_bed_iqr": _iqr(efficiency),
        "sleep_main_start_hour_adj_median": float(bedtime.median()),
        "sleep_main_start_hour_adj_iqr": _iqr(bedtime),
        "sleep_main_start_hour_adj_range": float(bedtime.max() - bedtime.min()),
        "sleep__hypersomnia_count_": int(asleep.gt(9.0).sum()),
        "sleep__hyposomnia_count_": int(asleep.lt(6.0).sum()),
        "sleep_asleep_mean_recent": _mean(recent, "sleep_duration"),
        "sleep_in_bed_mean_recent": _mean(recent, "in_bed_duration"),
        "sleep_ratio_asleep_in_bed_mean_recent": _mean(recent, "sleep_efficiency"),
    }


def build_windows(root: Path) -> pd.DataFrame:
    """Build two design-aligned 14-day windows per participant."""

    source_manifest(root)
    survey = pd.read_csv(root / "survey.csv")
    diary = pd.read_csv(root / "sleep_diary.csv")
    sensor = pd.read_csv(root / "sensor_hrv_filtered.csv")
    if survey.shape != (49, 23) or len(diary) != 1372 or len(sensor) != 38913:
        raise ConsciousWearableAdapterError("frozen Conscious row counts changed")
    if survey["deviceId"].duplicated().any():
        raise ConsciousWearableAdapterError("survey participant IDs are duplicated")
    diary["date"] = pd.to_datetime(diary["date"], errors="raise")
    sensor["ts_start"] = pd.to_numeric(sensor["ts_start"], errors="coerce")
    sensor["ts_end"] = pd.to_numeric(sensor["ts_end"], errors="coerce")
    sensor["timestamp"] = pd.to_datetime(
        sensor["ts_start"], unit="ms", utc=True, errors="coerce"
    )
    sensor["date"] = sensor["timestamp"].dt.tz_convert(None).dt.normalize()
    sensor["duration_seconds"] = (sensor["ts_end"] - sensor["ts_start"]) / 1000.0
    sensor["time_anomaly"] = sensor["timestamp"].dt.year.ne(2021)
    sensor["negative_missingness"] = pd.to_numeric(
        sensor["missingness_score"], errors="coerce"
    ).lt(0)
    sensor = sensor.loc[
        ~sensor["time_anomaly"]
        & sensor["duration_seconds"].ge(240.0)
        & pd.to_numeric(sensor["missingness_score"], errors="coerce").le(0.35)
    ].copy()
    rows: list[dict[str, Any]] = []
    survey_by_id = survey.set_index("deviceId")
    for participant, participant_diary in diary.groupby("userId", sort=True):
        ordered = participant_diary.sort_values("date", kind="mergesort").reset_index(
            drop=True
        )
        if len(ordered) != 28 or participant not in survey_by_id.index:
            raise ConsciousWearableAdapterError(
                "participant does not have 28 aligned diary days"
            )
        participant_sensor = sensor.loc[sensor["deviceId"].eq(participant)]
        labels = survey_by_id.loc[participant]
        targets = (
            ("mid", "PHQ9_2", "PHQ9_1", "GAD7_2", "ISI_2"),
            ("final", "PHQ9_F", "PHQ9_2", "GAD7_F", "ISI_F"),
        )
        for window_index, (
            window_name,
            target_name,
            previous_name,
            gad_name,
            isi_name,
        ) in enumerate(targets):
            selected_diary = ordered.iloc[
                window_index * 14 : (window_index + 1) * 14
            ].copy()
            start = selected_diary["date"].min()
            end = selected_diary["date"].max()
            selected_sensor = participant_sensor.loc[
                participant_sensor["date"].between(start, end)
            ]
            row = {
                "dataset_id": DATASET_ID,
                "participant_id": str(participant),
                "global_participant_id": f"{DATASET_ID}::{participant}",
                "window_id": f"{DATASET_ID}::{participant}::{window_name}",
                "window_index": window_index,
                "window_name": window_name,
                "window_start": start,
                "window_end": end,
                "diary_day_count": len(selected_diary),
                "sensor_window_count": len(selected_sensor),
                "sensor_day_count": selected_sensor["date"].nunique(),
                "negative_missingness_window_count": int(
                    selected_sensor["negative_missingness"].sum()
                ),
                "phq9_previous": float(labels[previous_name]),
                "phq9_score": float(labels[target_name]),
                "phq9_change": float(labels[target_name] - labels[previous_name]),
                "gad7_score": float(labels[gad_name]),
                "isi_score": float(labels[isi_name]),
            }
            row.update(_window_features(selected_diary, selected_sensor))
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["participant_id", "window_index"], kind="mergesort"
    )
    if (
        len(result) != 98
        or result["participant_id"].nunique() != 49
        or result["window_id"].duplicated().any()
    ):
        raise ConsciousWearableAdapterError("Conscious window contract failed")
    return result.reset_index(drop=True)


__all__ = [
    "ACTIVITY_FEATURES",
    "ADAPTER_VERSION",
    "COMMON_FEATURES",
    "ConsciousWearableAdapterError",
    "DATASET_ID",
    "SLEEP_FEATURES",
    "build_windows",
    "source_manifest",
]
