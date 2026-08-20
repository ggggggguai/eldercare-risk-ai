"""Causal PSYCHE personal-change features for OPT-V333-R5-001/002."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    assert_r5_research_feature_names,
)


def _prior_stat(values: np.ndarray, statistic: str) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(1, len(values)):
        prior = values[:index]
        prior = prior[np.isfinite(prior)]
        if len(prior) == 0:
            continue
        if statistic == "median":
            output[index] = float(np.median(prior))
        elif statistic == "mad":
            median = float(np.median(prior))
            output[index] = float(np.median(np.abs(prior - median)))
        elif statistic == "iqr":
            output[index] = float(np.percentile(prior, 75) - np.percentile(prior, 25))
        elif statistic == "std":
            if len(prior) >= 2:
                output[index] = float(np.std(prior, ddof=1))
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown prior statistic: {statistic}")
    return output


def _prior_slope(values: np.ndarray, months: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(1, len(values)):
        prior_values = values[:index]
        prior_months = months[:index]
        valid = np.isfinite(prior_values) & np.isfinite(prior_months)
        if valid.sum() >= 2 and np.ptp(prior_months[valid]) > 0:
            output[index] = float(
                np.polyfit(prior_months[valid], prior_values[valid], 1)[0]
            )
    return output


def _strict_prior_rolling(values: np.ndarray, *, window: int, statistic: str) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(1, len(values)):
        prior = values[max(0, index - int(window)) : index]
        prior = prior[np.isfinite(prior)]
        if len(prior) == 0:
            continue
        if statistic == "mean":
            output[index] = float(np.mean(prior))
        elif statistic == "std" and len(prior) >= 2:
            output[index] = float(np.std(prior, ddof=1))
        else:
            if statistic != "std":  # pragma: no cover - guarded by callers
                raise ValueError(f"unknown prior rolling statistic: {statistic}")
    return output


def _safe_log_ratio(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite_reference = np.abs(reference[np.isfinite(reference)])
    scale = float(np.median(finite_reference)) if finite_reference.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    numerator = np.maximum(current + scale, np.finfo(float).eps)
    denominator = np.maximum(reference + scale, np.finfo(float).eps)
    return np.clip(np.log(numerator / denominator), -5.0, 5.0)


def add_psyche_personal_change_features(
    frame: pd.DataFrame,
    *,
    source_features: Sequence[str] = R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    include_batch2: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Generate target-time and strictly-prior personal changes.

    The target row's current passive summary may be used. Every reference,
    slope and volatility feature is computed from earlier nominal months only.
    The collection month is used solely to order observations and measure the
    time gap; it is never returned as a risk feature.
    """

    assert_r5_research_feature_names(source_features)
    required = {"global_participant_id", "nominal_month", *source_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"r5 causal feature frame missing columns: {missing}")
    result = frame.copy()
    result["nominal_month"] = pd.to_numeric(result["nominal_month"], errors="coerce")
    if result["nominal_month"].isna().any():
        raise ValueError("PSYCHE nominal_month is required for causal ordering")
    if result.duplicated(["global_participant_id", "nominal_month"]).any():
        raise ValueError("duplicate PSYCHE participant/month rows")
    order = result.sort_values(
        ["global_participant_id", "nominal_month"], kind="stable"
    ).index
    ordered = result.loc[order].reset_index().rename(columns={"index": "_original_index"})
    group_positions = [
        np.asarray(positions, dtype=int)
        for positions in ordered.groupby("global_participant_id", sort=False).indices.values()
    ]
    generated_values: dict[str, np.ndarray] = {}
    batch1: list[str] = []
    ordered_months = ordered["nominal_month"].to_numpy(float)
    for feature in source_features:
        ordered_values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(float)
        definitions_by_suffix = {
            suffix: np.full(len(ordered), np.nan, dtype=float)
            for suffix in (
                "lag1",
                "delta_lag1",
                "prior_median",
                "prior_mean",
                "delta_prior_median",
                "prior_mad",
                "prior_iqr",
                "prior_std",
                "prior_ewma",
                "delta_prior_ewma",
                "recent2_mean",
                "recent2_std",
                "recent_minus_long_term",
                "prior_slope_per_month",
                "robust_deviation",
                "bounded_log_ratio",
                "months_since_lag1",
            )
        }
        for positions in group_positions:
            values = ordered_values[positions]
            months = ordered_months[positions]
            lag1 = np.roll(values, 1)
            lag1[0] = np.nan
            gap = np.roll(months, 1)
            gap = months - gap
            gap[0] = np.nan
            median = _prior_stat(values, "median")
            mean = np.full(len(values), np.nan, dtype=float)
            for index in range(1, len(values)):
                prior = values[:index]
                prior = prior[np.isfinite(prior)]
                if len(prior):
                    mean[index] = float(np.mean(prior))
            mad = _prior_stat(values, "mad")
            iqr = _prior_stat(values, "iqr")
            std = _prior_stat(values, "std")
            slope = _prior_slope(values, months)
            shifted = pd.Series(values).shift(1)
            ewma = shifted.ewm(alpha=0.5, adjust=False, min_periods=1).mean().to_numpy(float)
            recent2_mean = _strict_prior_rolling(values, window=2, statistic="mean")
            recent2_std = _strict_prior_rolling(values, window=2, statistic="std")
            robust_scale = np.where(np.isfinite(mad) & (mad > 1.0e-8), 1.4826 * mad, np.nan)
            definitions = {
                "lag1": lag1,
                "delta_lag1": values - lag1,
                "prior_median": median,
                "prior_mean": mean,
                "delta_prior_median": values - median,
                "prior_mad": mad,
                "prior_iqr": iqr,
                "prior_std": std,
                "prior_ewma": ewma,
                "delta_prior_ewma": values - ewma,
                "recent2_mean": recent2_mean,
                "recent2_std": recent2_std,
                "recent_minus_long_term": recent2_mean - mean,
                "prior_slope_per_month": slope,
                "robust_deviation": np.clip((values - median) / robust_scale, -8.0, 8.0),
                "bounded_log_ratio": _safe_log_ratio(values, median),
                "months_since_lag1": gap,
            }
            for suffix, feature_values in definitions.items():
                definitions_by_suffix[suffix][positions] = feature_values
        original_positions = ordered["_original_index"].to_numpy()
        for suffix, feature_values in definitions_by_suffix.items():
            name = f"r5_causal.{feature}_{suffix}"
            restored = pd.Series(feature_values, index=original_positions).reindex(result.index)
            generated_values[name] = restored.to_numpy(float)
            batch1.append(name)

    if generated_values:
        result = pd.concat(
            [result, pd.DataFrame(generated_values, index=result.index)], axis=1
        )

    batch2: list[str] = []
    if include_batch2:
        pairs = (
            (
                "source__steps_awake_mean",
                "source__sleep_ratio_asleep_in_bed_mean_recent",
                "activity_x_sleep_efficiency",
            ),
            (
                "source__steps__sedentary_day_count_",
                "source__sleep_asleep_mean_recent",
                "sedentary_x_sleep_duration",
            ),
            (
                "source__sleep_asleep_mean_recent",
                "source__sleep_main_start_hour_adj_iqr",
                "duration_x_onset_variability",
            ),
        )
        for activity, sleep, suffix in pairs:
            if activity not in result or sleep not in result:
                continue
            current_name = f"r5_causal.interaction_{suffix}"
            generated_values = {
                current_name: pd.to_numeric(result[activity], errors="coerce")
                * pd.to_numeric(result[sleep], errors="coerce")
            }
            activity_delta = f"r5_causal.{activity}_delta_prior_median"
            sleep_delta = f"r5_causal.{sleep}_delta_prior_median"
            trend_name = f"r5_causal.interaction_{suffix}_joint_change"
            generated_values[trend_name] = result[activity_delta] * result[sleep_delta]
            result = pd.concat(
                [result, pd.DataFrame(generated_values, index=result.index)], axis=1
            )
            batch2.extend([current_name, trend_name])
        weekday_activity = "source__steps_awake_mean"
        weekday_sleep = "source__sleep_asleep_weekday_mean"
        weekend_sleep = "source__sleep_asleep_weekend_mean"
        if {weekday_activity, weekday_sleep, weekend_sleep}.issubset(result.columns):
            name = "r5_causal.interaction_activity_x_weekend_sleep_difference"
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        {
                            name: pd.to_numeric(result[weekday_activity], errors="coerce")
                            * (
                                pd.to_numeric(result[weekend_sleep], errors="coerce")
                                - pd.to_numeric(result[weekday_sleep], errors="coerce")
                            )
                        },
                        index=result.index,
                    ),
                ],
                axis=1,
            )
            batch2.append(name)
        difference_pairs = (
            (
                "source__sleep_asleep_weekend_mean",
                "source__sleep_asleep_weekday_mean",
                "weekend_minus_weekday_sleep_duration",
            ),
            (
                "source__sleep_in_bed_weekend_mean",
                "source__sleep_in_bed_weekday_mean",
                "weekend_minus_weekday_time_in_bed",
            ),
            (
                "source__sleep_ratio_asleep_in_bed_weekend_mean",
                "source__sleep_ratio_asleep_in_bed_weekday_mean",
                "weekend_minus_weekday_sleep_efficiency",
            ),
        )
        for weekend, weekday, suffix in difference_pairs:
            if {weekend, weekday}.issubset(result.columns):
                name = f"r5_causal.{suffix}"
                result = pd.concat(
                    [
                        result,
                        pd.DataFrame(
                            {
                                name: pd.to_numeric(result[weekend], errors="coerce")
                                - pd.to_numeric(result[weekday], errors="coerce")
                            },
                            index=result.index,
                        ),
                    ],
                    axis=1,
                )
                batch2.append(name)
        short = "source__sleep__hyposomnia_count_"
        long = "source__sleep__hypersomnia_count_"
        if {short, long}.issubset(result.columns):
            name = "r5_causal.abnormal_sleep_day_count"
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        {
                            name: pd.to_numeric(result[short], errors="coerce")
                            + pd.to_numeric(result[long], errors="coerce")
                        },
                        index=result.index,
                    ),
                ],
                axis=1,
            )
            batch2.append(name)

    generated = (*batch1, *batch2)
    assert_r5_research_feature_names((*source_features, *generated))
    return result, tuple(batch1), tuple(batch2)


__all__ = ["add_psyche_personal_change_features"]
