"""Change-point detection support for trend-shift explanations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from statistics import fmean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FeatureChangePoint:
    feature: str
    change_index: int
    change_time: str | None
    before_mean: float
    after_mean: float
    delta: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangePointEvidence:
    """Auxiliary trend-shift evidence; it is explanatory, not diagnostic."""

    status: str
    max_shift_score: float
    has_change: bool
    changes: list[FeatureChangePoint] = field(default_factory=list)
    factors: list[str] = field(default_factory=list)
    reference_count: int = 0
    method: str = "sliding_mean_shift"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changes"] = [change.to_dict() for change in self.changes]
        return payload


def detect_trend_change(
    history_records: Iterable[Mapping[str, Any]],
    current_features: Mapping[str, float | None],
    *,
    feature_names: Iterable[str],
    min_points: int = 6,
    window: int = 3,
    min_delta: float = 0.25,
) -> ChangePointEvidence:
    """Detect recent upward shifts in normalized risk-score time series."""

    history = sorted(
        [record for record in history_records if isinstance(record, Mapping)],
        key=_record_sort_key,
    )
    if len(history) + 1 < min_points:
        return _insufficient("insufficient_time_series_points", len(history))

    changes: list[FeatureChangePoint] = []
    for feature in feature_names:
        current_value = _optional_unit(current_features.get(feature))
        if current_value is None:
            continue
        series_records = [
            record
            for record in history
            if _optional_unit(record.get(feature)) is not None
        ]
        values = [float(_optional_unit(record.get(feature)) or 0.0) for record in series_records]
        records_for_feature: list[Mapping[str, Any] | None] = [*series_records]
        values.append(current_value)
        records_for_feature.append(None)
        if len(values) < min_points:
            continue
        change = _strongest_positive_shift(
            feature,
            values,
            records_for_feature,
            window=window,
            min_delta=min_delta,
        )
        if change is not None:
            changes.append(change)

    if not changes:
        return ChangePointEvidence(
            status="available",
            max_shift_score=0.0,
            has_change=False,
            reference_count=len(history),
            reason="no_positive_shift_over_threshold",
        )

    changes.sort(key=lambda item: item.strength, reverse=True)
    return ChangePointEvidence(
        status="available",
        max_shift_score=round(max(change.strength for change in changes), 4),
        has_change=True,
        changes=changes[:4],
        factors=[change.feature for change in changes[:4]],
        reference_count=len(history),
    )


def _strongest_positive_shift(
    feature: str,
    values: list[float],
    records: list[Mapping[str, Any] | None],
    *,
    window: int,
    min_delta: float,
) -> FeatureChangePoint | None:
    best: FeatureChangePoint | None = None
    minimum_side = max(2, window)
    for split in range(minimum_side, len(values) - minimum_side + 1):
        before = values[max(0, split - window):split]
        after = values[split:min(len(values), split + window)]
        if len(before) < minimum_side or len(after) < minimum_side:
            continue
        before_mean = fmean(before)
        after_mean = fmean(after)
        delta = after_mean - before_mean
        if delta < min_delta:
            continue
        change = FeatureChangePoint(
            feature=feature,
            change_index=split,
            change_time=_record_time(records[split]),
            before_mean=round(before_mean, 4),
            after_mean=round(after_mean, 4),
            delta=round(delta, 4),
            strength=round(_clamp(delta / max(min_delta * 2.0, 0.01)), 4),
        )
        if best is None or change.strength > best.strength:
            best = change
    return best


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("date") or ""),
        str(record.get("timestamp") or ""),
        str(record.get("end_time") or record.get("start_time") or ""),
    )


def _record_time(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    value = record.get("date") or record.get("timestamp") or record.get("end_time") or record.get("start_time")
    return str(value) if value is not None else None


def _optional_unit(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _insufficient(reason: str, reference_count: int = 0) -> ChangePointEvidence:
    return ChangePointEvidence(
        status="insufficient_data",
        max_shift_score=0.0,
        has_change=False,
        reference_count=reference_count,
        reason=reason,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
