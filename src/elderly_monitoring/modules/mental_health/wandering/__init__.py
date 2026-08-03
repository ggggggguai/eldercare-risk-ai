"""Internal trajectory foundations for wandering-like behavior research.

This package is deliberately not re-exported from ``mental_health``. Callers
must opt into the experimental namespace explicitly until its evaluation gates
and integration contract are complete.
"""

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    TrajectoryDatasetError,
    iter_trajectory_jsonl,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import (
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
    TrajectorySchemaError,
)

__all__ = [
    "TRAJECTORY_SAMPLE_SCHEMA_VERSION",
    "CoordinateSystem",
    "PatternLabel",
    "TrajectoryDatasetError",
    "TrajectorySample",
    "TrajectorySchemaError",
    "iter_trajectory_jsonl",
    "load_trajectory_jsonl",
    "write_trajectory_jsonl",
]
