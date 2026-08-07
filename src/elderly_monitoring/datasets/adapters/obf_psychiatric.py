"""Adapter for OBF-Psychiatric participant-level activity rhythm features."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "obf_psychiatric"
ADAPTER_VERSION = "obf-psychiatric-v1"
MODEL_FEATURES = (
    "activity_mean",
    "activity_sd",
    "activity_zero_fraction",
    "activity_median",
    "activity_q25",
    "activity_q75",
    "relative_amplitude",
    "interdaily_stability",
    "intradaily_variability",
    "daily_total_mean",
    "daily_total_cv",
)
GROUPS = ("adhd", "clinical", "control", "depression", "schizophrenia")


class ObfPsychiatricAdapterError(ValueError):
    """Raised when OBF source or feature contracts are violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(
        [
            *(root.glob("*-info.csv")),
            *(path for group in GROUPS for path in (root / group).glob("*.csv")),
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if len(paths) != 167:
        raise ObfPsychiatricAdapterError(
            f"expected 167 OBF source files, found {len(paths)}"
        )
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        aggregate.update(f"{relative}\0{digest}\n".encode())
        total_bytes += path.stat().st_size
    return {
        "dataset_id": DATASET_ID,
        "adapter_version": ADAPTER_VERSION,
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(),
    }


def _rhythm_features(frame: pd.DataFrame) -> dict[str, float | int]:
    activity = pd.to_numeric(frame["activity"], errors="coerce").astype("float64")
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid = activity.notna() & timestamp.notna()
    activity = activity.loc[valid]
    timestamp = timestamp.loc[valid]
    if len(activity) < 1440:
        raise ObfPsychiatricAdapterError(
            "participant has less than one day of valid activity"
        )
    hourly = (
        pd.DataFrame({"timestamp": timestamp, "activity": activity})
        .set_index("timestamp")["activity"]
        .resample("1h")
        .mean()
        .dropna()
    )
    profile = hourly.groupby(hourly.index.hour).mean().reindex(range(24))
    if profile.notna().sum() < 20:
        relative_amplitude = np.nan
    else:
        filled = profile.interpolate(limit_direction="both").to_numpy(dtype="float64")
        circular = np.concatenate([filled, filled])
        m10 = max(np.mean(circular[start : start + 10]) for start in range(24))
        l5 = min(np.mean(circular[start : start + 5]) for start in range(24))
        relative_amplitude = float((m10 - l5) / (m10 + l5)) if m10 + l5 else 0.0
    hourly_values = hourly.to_numpy(dtype="float64")
    grand = float(np.mean(hourly_values))
    denominator = float(np.sum((hourly_values - grand) ** 2))
    hour_means = hourly.groupby(hourly.index.hour).mean()
    interdaily = (
        float(
            len(hourly_values)
            * np.sum((hour_means - grand) ** 2)
            / (24.0 * denominator)
        )
        if denominator > 0 and len(hour_means) == 24
        else np.nan
    )
    intradaily = (
        float(
            len(hourly_values)
            * np.sum(np.diff(hourly_values) ** 2)
            / ((len(hourly_values) - 1) * denominator)
        )
        if denominator > 0 and len(hourly_values) > 1
        else np.nan
    )
    daily = (
        pd.DataFrame({"date": timestamp.dt.normalize(), "activity": activity})
        .groupby("date")["activity"]
        .sum(min_count=1)
    )
    daily_mean = float(daily.mean())
    return {
        "activity_mean": float(activity.mean()),
        "activity_sd": float(activity.std(ddof=0)),
        "activity_zero_fraction": float(activity.eq(0).mean()),
        "activity_median": float(activity.median()),
        "activity_q25": float(activity.quantile(0.25)),
        "activity_q75": float(activity.quantile(0.75)),
        "relative_amplitude": relative_amplitude,
        "interdaily_stability": interdaily,
        "intradaily_variability": intradaily,
        "daily_total_mean": daily_mean,
        "daily_total_cv": float(daily.std(ddof=0) / daily_mean)
        if daily_mean
        else np.nan,
        "valid_minute_count": len(activity),
        "valid_day_count": int(daily.index.nunique()),
        "collection_start_year": int(timestamp.dt.year.min()),
        "collection_end_year": int(timestamp.dt.year.max()),
    }


def build_participant_features(root: Path) -> pd.DataFrame:
    source_manifest(root)
    info_by_group = {
        group: pd.read_csv(root / f"{group}-info.csv", na_values=["NA"])
        for group in GROUPS
    }
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        info = info_by_group[group]
        key = "number"
        if key not in info.columns:
            raise ObfPsychiatricAdapterError(f"missing participant key for {group}")
        info_index = info.assign(number=info[key].astype(str)).set_index("number")
        for path in sorted((root / group).glob("*.csv")):
            participant = path.stem
            frame = pd.read_csv(path, usecols=["timestamp", "date", "activity"])
            row: dict[str, Any] = {
                "dataset_id": DATASET_ID,
                "participant_id": participant,
                "global_participant_id": f"{DATASET_ID}::{participant}",
                "source_group": group,
            }
            row.update(_rhythm_features(frame))
            if participant in info_index.index:
                metadata = info_index.loc[participant]
                row["madrs1"] = pd.to_numeric(metadata.get("madrs1"), errors="coerce")
                row["madrs2"] = pd.to_numeric(metadata.get("madrs2"), errors="coerce")
            else:
                row["madrs1"] = np.nan
                row["madrs2"] = np.nan
            row["madrs_change"] = row["madrs2"] - row["madrs1"]
            rows.append(row)
    result = (
        pd.DataFrame(rows)
        .sort_values("global_participant_id", kind="mergesort")
        .reset_index(drop=True)
    )
    paired = result["madrs1"].notna() & result["madrs2"].notna()
    if (
        len(result) != 162
        or result["global_participant_id"].nunique() != 162
        or int(paired.sum()) != 23
    ):
        raise ObfPsychiatricAdapterError("OBF participant or MADRS counts changed")
    return result


__all__ = [
    "ADAPTER_VERSION",
    "DATASET_ID",
    "GROUPS",
    "MODEL_FEATURES",
    "ObfPsychiatricAdapterError",
    "build_participant_features",
    "source_manifest",
]
