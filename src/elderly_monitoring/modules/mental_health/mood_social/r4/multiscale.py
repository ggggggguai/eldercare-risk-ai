"""Causal 1/3/7/14/28-day feature generator for future runtime-parity data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


WINDOW_DAYS = (1, 3, 7, 14, 28)


def _robust_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    array = values[valid]
    if len(array) < 2:
        return float("nan")
    pairs: list[float] = []
    positions = np.flatnonzero(valid)
    for left in range(len(array) - 1):
        for right in range(left + 1, len(array)):
            pairs.append((array[right] - array[left]) / (positions[right] - positions[left]))
    return float(np.median(pairs))


def add_causal_daily_multiscale_features(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    activity_sleep_pairs: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Aggregate only observations in ``[D-window+1, D]`` for each person.

    Input must be one row per participant/calendar day.  Missing dates remain
    missing rather than being silently interpreted as zero activity or sleep.
    The individual baseline is strictly prior-to-D and therefore does not use
    another participant or a future row.
    """

    required = {"global_participant_id", "observation_date", *value_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"daily multiscale frame missing columns: {missing}")
    result = frame.copy()
    result["observation_date"] = pd.to_datetime(result["observation_date"], errors="raise")
    if result.duplicated(["global_participant_id", "observation_date"]).any():
        raise ValueError("daily multiscale input has duplicate participant/date rows")
    for left, right in activity_sleep_pairs:
        if left not in value_columns or right not in value_columns:
            raise ValueError("activity-sleep coupling columns must also be value columns")
    generated: list[str] = []
    ordered = result.sort_values(
        ["global_participant_id", "observation_date"], kind="stable"
    )
    for _person, part in ordered.groupby("global_participant_id", sort=False):
        for row_index, current_row in part.iterrows():
            current_date = current_row["observation_date"]
            prior = part.loc[part["observation_date"].lt(current_date)]
            for column in value_columns:
                prior_values = pd.to_numeric(prior[column], errors="coerce").to_numpy(float)
                prior_median = float(np.nanmedian(prior_values)) if np.isfinite(prior_values).any() else np.nan
                prior_q25 = float(np.nanpercentile(prior_values, 25)) if np.isfinite(prior_values).any() else np.nan
                prior_q75 = float(np.nanpercentile(prior_values, 75)) if np.isfinite(prior_values).any() else np.nan
                prior_iqr = prior_q75 - prior_q25
                current_value = float(pd.to_numeric(pd.Series([current_row[column]]), errors="coerce").iloc[0])
                baseline_name = f"multiscale.{column}.prior_baseline_deviation"
                result.loc[row_index, baseline_name] = (
                    (current_value - prior_median) / max(prior_iqr, 1.0e-6)
                    if np.isfinite(current_value) and np.isfinite(prior_median)
                    else np.nan
                )
                if baseline_name not in generated:
                    generated.append(baseline_name)
            for days in WINDOW_DAYS:
                lower = current_date - pd.Timedelta(days=days - 1)
                window = part.loc[
                    part["observation_date"].between(lower, current_date, inclusive="both")
                ]
                for column in value_columns:
                    values = pd.to_numeric(window[column], errors="coerce").to_numpy(float)
                    finite = values[np.isfinite(values)]
                    prefix = f"multiscale.{column}.d{days}"
                    summary = {
                        "current": float(values[-1]) if len(values) and np.isfinite(values[-1]) else np.nan,
                        "mean": float(np.mean(finite)) if len(finite) else np.nan,
                        "median": float(np.median(finite)) if len(finite) else np.nan,
                        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)) if len(finite) else np.nan,
                        "robust_slope": _robust_slope(values),
                        "volatility": float(np.std(finite)) if len(finite) else np.nan,
                        "observed_days": float(len(finite)),
                    }
                    prior_values = pd.to_numeric(
                        part.loc[part["observation_date"].lt(lower), column], errors="coerce"
                    ).to_numpy(float)
                    prior_finite = prior_values[np.isfinite(prior_values)]
                    if len(prior_finite):
                        median = float(np.median(prior_finite))
                        iqr = float(np.percentile(prior_finite, 75) - np.percentile(prior_finite, 25))
                        threshold = max(1.5 * iqr, 1.0e-6)
                        summary["abnormal_days"] = float(np.sum(np.abs(finite - median) > threshold))
                    else:
                        summary["abnormal_days"] = np.nan
                    for suffix, value in summary.items():
                        name = f"{prefix}.{suffix}"
                        result.loc[row_index, name] = value
                        if name not in generated:
                            generated.append(name)
                for activity, sleep in activity_sleep_pairs:
                    paired = window[[activity, sleep]].apply(pd.to_numeric, errors="coerce").dropna()
                    name = f"multiscale.coupling.{activity}__{sleep}.d{days}.spearman"
                    result.loc[row_index, name] = (
                        float(paired[activity].corr(paired[sleep], method="spearman"))
                        if len(paired) >= 3
                        else np.nan
                    )
                    if name not in generated:
                        generated.append(name)
    for column in value_columns:
        short = f"multiscale.{column}.d3.mean"
        long = f"multiscale.{column}.d28.mean"
        name = f"multiscale.{column}.recent_d3_minus_d28"
        result[name] = result[short] - result[long]
        generated.append(name)
    return result, tuple(generated)


__all__ = ["WINDOW_DAYS", "add_causal_daily_multiscale_features"]
