"""Load the r5 reused population on a pre-registered participant split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    assert_participant_isolation,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    DEFAULT_DATA_RELATIVE,
    R5_ALL_SEEDS,
    R5_LABEL_TASKS,
)


def _repository_root(repository_root: Path | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(__file__).resolve().parents[7]


def load_r5_development_frame(
    *, repository_root: Path | None = None, seed: int
) -> pd.DataFrame:
    """Load r5 rows with the exact frozen assignment for ``seed``."""

    if int(seed) not in R5_ALL_SEEDS:
        raise ValueError(f"unregistered r5 split seed: {seed}")
    root = _repository_root(repository_root)
    split_path = root / DEFAULT_DATA_RELATIVE / "splits" / f"seed-{int(seed)}.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"r5 split has not been prepared: {split_path}")
    base = load_r4_development_frame(repository_root=root).copy()
    split = pd.read_parquet(split_path)
    expected = set(base["r4_row_id"].astype(str))
    if set(split["r4_row_id"].astype(str)) != expected:
        raise ValueError("r5 split row keys differ from the r4 scoring population")
    base = base.drop(columns=["outer_fold", "inner_validation_fold_by_outer_fold"])
    result = base.merge(
        split[
            [
                "r4_row_id",
                "r5_row_id",
                "split_seed",
                "outer_fold",
                "inner_validation_fold_by_outer_fold",
            ]
        ],
        on="r4_row_id",
        how="left",
        validate="one_to_one",
    )
    if result[["r5_row_id", "outer_fold"]].isna().any().any():
        raise ValueError("r5 split merge is incomplete")
    # r3's isolation verifier accepts JSON inner maps and provides the same
    # outer/inner invariants used throughout the nested training code.
    assert_participant_isolation(result)
    if not result["split_seed"].eq(int(seed)).all():
        raise ValueError("r5 split seed column drifted")
    return result


def r5_inner_fold_series(frame: pd.DataFrame, outer_fold: int) -> pd.Series:
    """Return inner validation folds for one r5 outer train partition."""

    key = str(int(outer_fold))

    def extract(value: object) -> float:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            value = dict(value)  # type: ignore[arg-type]
        result = value.get(key)
        return np.nan if result is None else float(result)

    series = frame["inner_validation_fold_by_outer_fold"].map(extract)
    is_test = frame["outer_fold"].eq(int(outer_fold))
    if series[is_test].notna().any() or series[~is_test].isna().any():
        raise ValueError(f"invalid r5 inner folds for outer {outer_fold}")
    return series


def select_r5_label_task(frame: pd.DataFrame, task_id: str) -> pd.DataFrame:
    """Select one frozen current-state task without changing its rows."""

    if task_id not in R5_LABEL_TASKS:
        raise ValueError(f"unknown r5 task: {task_id!r}")
    result = frame.copy()
    result["binary_target"] = result[R5_LABEL_TASKS[task_id]].astype(int)
    result["r5_task_id"] = str(task_id)
    # r4 recipe helpers consume this compatibility field. It is metadata, not
    # a risk feature, and is never passed to a model vector.
    result["r4_task_id"] = str(task_id)
    return result


__all__ = [
    "load_r5_development_frame",
    "r5_inner_fold_series",
    "select_r5_label_task",
]
