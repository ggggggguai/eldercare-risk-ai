"""Causal PHQ-history expert for target-day current-state nowcasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


HISTORY_FEATURES = (
    "history.last_score",
    "history.last_ge5",
    "history.last_ge10",
    "history.age_days",
    "history.count",
    "history.mean",
    "history.ewma",
    "history.mad",
    "history.delta_last_two",
    "history.slope",
    "history.ge5_fraction",
    "history.ge10_fraction",
    "history.ge5_streak",
    "history.ge10_streak",
)


@dataclass(frozen=True)
class HistoricalPHQ:
    assessment_date: date
    score: int
    known_at: date | None = None


def _freshness(age_days: int) -> tuple[str, bool, str]:
    if age_days <= 13:
        return "recent_screening_continuity", True, "strong"
    if age_days <= 30:
        return "14_30_days", True, "strong"
    if age_days <= 90:
        return "31_90_days", True, "medium"
    if age_days <= 180:
        return "91_180_days", False, "weak_background"
    return "over_180_days", False, "display_only"


def build_history_features(
    assessments: Sequence[HistoricalPHQ],
    *,
    target_date: date,
    inference_cutoff: date | None = None,
) -> tuple[dict[str, float] | None, dict[str, object]]:
    cutoff = inference_cutoff or target_date
    eligible = sorted(
        (
            record
            for record in assessments
            if record.assessment_date < target_date
            and (record.known_at or record.assessment_date) <= cutoff
        ),
        key=lambda record: record.assessment_date,
    )
    if not eligible:
        return None, {"available": False, "reason": "no_strictly_prior_known_phq"}
    scores = np.asarray([record.score for record in eligible], dtype=float)
    if (scores < 0).any() or (scores > 27).any():
        raise ValueError("historical PHQ score must be in 0..27")
    age_days = (target_date - eligible[-1].assessment_date).days
    freshness, participates, grade = _freshness(age_days)
    weights = np.power(0.5, np.arange(len(scores) - 1, -1, -1) / 2.0)
    ge5 = scores >= 5
    ge10 = scores >= 10
    def streak(values: np.ndarray) -> int:
        result = 0
        for value in values[::-1]:
            if not value:
                break
            result += 1
        return result
    slope = float(np.polyfit(np.arange(len(scores)), scores, 1)[0]) if len(scores) >= 2 else 0.0
    features = {
        "history.last_score": float(scores[-1]),
        "history.last_ge5": float(ge5[-1]),
        "history.last_ge10": float(ge10[-1]),
        "history.age_days": float(age_days),
        "history.count": float(len(scores)),
        "history.mean": float(scores.mean()),
        "history.ewma": float(np.average(scores, weights=weights)),
        "history.mad": float(np.median(np.abs(scores - np.median(scores)))),
        "history.delta_last_two": float(scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0,
        "history.slope": slope,
        "history.ge5_fraction": float(ge5.mean()),
        "history.ge10_fraction": float(ge10.mean()),
        "history.ge5_streak": float(streak(ge5)),
        "history.ge10_streak": float(streak(ge10)),
    }
    return features, {
        "available": True,
        "record_count": len(eligible),
        "last_age_days": age_days,
        "freshness": freshness,
        "participates_in_rule_vote": participates,
        "evidence_grade": grade,
    }


def build_psyche_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create strictly prior-wave features; first wave is unavailable."""

    required = {"global_participant_id", "nominal_month", "phq9_score_r3_target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PSYCHE history frame missing {missing}")
    work = frame.loc[frame["dataset_id"].eq("psyche_d")].copy()
    work = work.sort_values(["global_participant_id", "nominal_month"], kind="stable")
    rows: list[dict[str, object]] = []
    for participant, part in work.groupby("global_participant_id", sort=False):
        scores: list[float] = []
        months: list[float] = []
        for row in part.itertuples(index=False):
            if scores:
                array = np.asarray(scores, float)
                ge5 = array >= 5
                ge10 = array >= 10
                rows.append({
                    "r4_row_id": row.r4_row_id,
                    "global_participant_id": participant,
                    "dataset_id": "psyche_d",
                    "nominal_month": float(row.nominal_month),
                    "phq9_ge5_target": int(row.phq9_ge5_r3_target),
                    "phq9_ge10_target": int(row.phq9_ge10_r3_target),
                    "history.last_score": float(array[-1]),
                    "history.last_ge5": float(ge5[-1]),
                    "history.last_ge10": float(ge10[-1]),
                    "history.age_days": float(max(1.0, (float(row.nominal_month) - months[-1]) * 30.4375)),
                    "history.count": float(len(array)),
                    "history.mean": float(array.mean()),
                    "history.ewma": float(np.average(array, weights=np.power(0.5, np.arange(len(array) - 1, -1, -1) / 2.0))),
                    "history.mad": float(np.median(np.abs(array - np.median(array)))),
                    "history.delta_last_two": float(array[-1] - array[-2]) if len(array) >= 2 else 0.0,
                    "history.slope": float(np.polyfit(months, array, 1)[0]) if len(array) >= 2 else 0.0,
                    "history.ge5_fraction": float(ge5.mean()),
                    "history.ge10_fraction": float(ge10.mean()),
                    "history.ge5_streak": float(sum(1 for value in ge5[::-1] if value)),
                    "history.ge10_streak": float(sum(1 for value in ge10[::-1] if value)),
                })
            scores.append(float(row.phq9_score_r3_target))
            months.append(float(row.nominal_month))
    result = pd.DataFrame(rows)
    if result.empty or result[list(HISTORY_FEATURES)].isna().any().any():
        raise ValueError("PSYCHE history features are empty or incomplete")
    return result


__all__ = ["HISTORY_FEATURES", "HistoricalPHQ", "build_history_features", "build_psyche_history_frame"]
