"""Canonical r3 training-table loading without split or label ambiguity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    DEFAULT_OUTPUT_RELATIVE,
    PROCESSED_RELATIVE,
    _load_canonical_frame,
    _repository_root,
)


def _unified_phq9_score(frame: pd.DataFrame) -> pd.Series:
    """Return the publisher-confirmed score target for every canonical row."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    source_columns = {
        "psyche_d": "phq9_score_end",
        "resilient": "phq9_score",
        "nhanes": "phq9_total",
        "nhanes_ssq_2005_2008": "phq9_total",
        "shenzhen_elderly": "phq9_score",
        "hefei_elderly": "phq9_score",
    }
    for dataset_id, column in source_columns.items():
        mask = frame["dataset_id"].eq(dataset_id)
        if column not in frame.columns:
            raise ValueError(f"missing PHQ target column {column!r} for {dataset_id}")
        result.loc[mask] = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    if result.isna().any():
        missing = frame.loc[result.isna(), "dataset_id"].value_counts().to_dict()
        raise ValueError(f"missing unified PHQ-9 targets: {missing}")
    if ((result < 0) | (result > 27)).any():
        raise ValueError("unified PHQ-9 score outside 0..27")
    return result


def load_r3_training_frame(
    *,
    repository_root: Path | None = None,
    population_path: Path | None = None,
) -> pd.DataFrame:
    """Load six frozen canonicals and attach the pre-model population decision."""

    root = _repository_root(repository_root)
    canonical, _ = _load_canonical_frame(root)
    path = population_path or root / DEFAULT_OUTPUT_RELATIVE / "population_assignments.parquet"
    population = pd.read_parquet(path)
    contract_columns = [
        "r3_row_id",
        "route_c",
        "route_s",
        "route_p",
        "route_pattern",
        "r3_eligible",
        "common_support",
        "no_evidence",
        "eligibility_reason",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    ]
    missing = sorted(set(contract_columns).difference(population.columns))
    if missing:
        raise ValueError(f"population artifact missing columns: {missing}")
    frame = canonical.merge(
        population[contract_columns],
        on="r3_row_id",
        how="left",
        validate="one_to_one",
    )
    if frame["r3_eligible"].isna().any():
        raise ValueError("canonical rows are not fully covered by population artifact")
    frame["phq9_score_r3_target"] = _unified_phq9_score(frame)
    for threshold in (5, 10, 15, 20):
        frame[f"phq9_ge{threshold}_r3_target"] = (
            frame["phq9_score_r3_target"] >= threshold
        ).astype(int)
    if not frame["phq9_ge10_r3_target"].eq(frame["binary_target"].astype(int)).all():
        raise ValueError("canonical binary target is inconsistent with PHQ-9 >= 10")
    return frame.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)


def inner_fold_series(frame: pd.DataFrame, outer_fold: int) -> pd.Series:
    """Return frozen inner validation fold assignments for one outer split."""

    key = str(int(outer_fold))

    def extract(value: Any) -> float:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            # PyArrow can materialize struct fields as mapping-like objects.
            try:
                value = dict(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid inner-fold mapping") from exc
        result = value.get(key)
        return np.nan if result is None else float(result)

    result = frame["inner_validation_fold_by_outer_fold"].map(extract)
    is_outer_test = frame["outer_fold"].eq(int(outer_fold))
    if result[is_outer_test].notna().any():
        raise ValueError("outer-test participant received an inner fold")
    if result[~is_outer_test].isna().any():
        raise ValueError("outer-train participant is missing an inner fold")
    return result


def assert_participant_isolation(frame: pd.DataFrame) -> None:
    """Verify participant and row identities are compatible with the frozen split."""

    if frame["r3_row_id"].duplicated().any():
        raise ValueError("duplicate r3 row ids")
    folds_per_participant = frame.groupby("global_participant_id")["outer_fold"].nunique()
    if not folds_per_participant.eq(1).all():
        raise ValueError("participant appears in multiple outer folds")
    for outer_fold in range(5):
        inner = inner_fold_series(frame, outer_fold)
        work = frame.assign(_inner=inner)
        inner_per_participant = (
            work[work["outer_fold"].ne(outer_fold)]
            .groupby("global_participant_id")["_inner"]
            .nunique()
        )
        if not inner_per_participant.eq(1).all():
            raise ValueError(f"participant appears in multiple inner folds for outer {outer_fold}")


def eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["r3_eligible"].eq(1)].copy()


def route_rows(frame: pd.DataFrame, route_patterns: Iterable[str]) -> pd.DataFrame:
    patterns = {str(value) for value in route_patterns}
    return frame[frame["route_pattern"].isin(patterns)].copy()


__all__ = [
    "assert_participant_isolation",
    "eligible_rows",
    "inner_fold_series",
    "load_r3_training_frame",
    "route_rows",
]
