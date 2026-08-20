"""R4 adaptive-development population and dual current-state labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    assert_participant_isolation,
    eligible_rows,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    R4_LABEL_TASKS,
)


def load_r4_development_frame(*, repository_root: Path | None = None) -> pd.DataFrame:
    """Load the frozen scorable rows; this is not a new blind population."""

    frame = eligible_rows(
        load_r3_training_frame(repository_root=repository_root)
    ).copy()
    if frame["dataset_id"].eq("hefei_elderly").any():
        raise ValueError("Hefei has no approved passive features and must remain auxiliary-only")
    frame["r4_row_id"] = frame["r3_row_id"].astype(str)
    assert_participant_isolation(frame)
    return frame


def select_label_task(frame: pd.DataFrame, task_id: str) -> pd.DataFrame:
    """Select one frozen current-state head without changing its scoring rows."""

    if task_id not in R4_LABEL_TASKS:
        raise ValueError(f"unknown r4 label task: {task_id!r}")
    column = R4_LABEL_TASKS[task_id]
    if column not in frame.columns:
        raise ValueError(f"r4 frame missing label column: {column}")
    result = frame.copy()
    result["binary_target"] = result[column].astype(int)
    result["r4_task_id"] = task_id
    return result


__all__ = ["load_r4_development_frame", "select_label_task"]
