"""Load r6 rows on preregistered participant-grouped nested splits."""

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
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    DEFAULT_DATA_RELATIVE,
    R6_ALL_SEEDS,
    R6_LABEL_TASKS,
    repository_root,
)


def load_r6_development_frame(*, repository_root_value: Path | None = None, seed: int) -> pd.DataFrame:
    if int(seed) not in R6_ALL_SEEDS:
        raise ValueError(f"unregistered r6 split seed: {seed}")
    root = repository_root(repository_root_value)
    split_path = root / DEFAULT_DATA_RELATIVE / "splits" / f"seed-{int(seed)}.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"r6 split has not been prepared: {split_path}")
    base = load_r4_development_frame(repository_root=root).copy()
    split = pd.read_parquet(split_path)
    if set(split["r4_row_id"].astype(str)) != set(base["r4_row_id"].astype(str)):
        raise ValueError("r6 split row keys differ from scoring population")
    base = base.drop(columns=["outer_fold", "inner_validation_fold_by_outer_fold"], errors="ignore")
    result = base.merge(
        split[[
            "r4_row_id", "r5_row_id", "r6_row_id", "split_seed", "outer_fold",
            "inner_validation_fold_by_outer_fold",
        ]],
        on="r4_row_id", how="left", validate="one_to_one",
    )
    if result[["r6_row_id", "outer_fold"]].isna().any().any():
        raise ValueError("r6 split merge is incomplete")
    assert_participant_isolation(result)
    if not result["split_seed"].eq(int(seed)).all():
        raise ValueError("r6 split seed column drifted")
    return result


def r6_inner_fold_series(frame: pd.DataFrame, outer_fold: int) -> pd.Series:
    key = str(int(outer_fold))

    def extract(value: object) -> float:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)  # type: ignore[arg-type]
        result = parsed.get(key)
        return np.nan if result is None else float(result)

    series = frame["inner_validation_fold_by_outer_fold"].map(extract)
    is_test = frame["outer_fold"].eq(int(outer_fold))
    if series[is_test].notna().any() or series[~is_test].isna().any():
        raise ValueError(f"invalid r6 inner folds for outer {outer_fold}")
    return series


def select_r6_label_task(frame: pd.DataFrame, task_id: str) -> pd.DataFrame:
    if task_id not in R6_LABEL_TASKS:
        raise ValueError(f"unknown r6 task: {task_id!r}")
    result = frame.copy()
    result["binary_target"] = result[R6_LABEL_TASKS[task_id]].astype(int)
    result["r6_task_id"] = str(task_id)
    return result


__all__ = ["load_r6_development_frame", "r6_inner_fold_series", "select_r6_label_task"]
