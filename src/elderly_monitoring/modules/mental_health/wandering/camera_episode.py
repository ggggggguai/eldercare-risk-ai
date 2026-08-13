"""Development-only episode candidates for TopoWander camera windows."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


CAMERA_EPISODE_CANDIDATE_SCHEMA_VERSION = "wandering-camera-episode-candidate-v1"
CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION = "wandering-camera-primary-prediction-v1"
POLICY_STATUS = "development_unfrozen"
PREDICTION_STATUS = "development_candidate"
FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")
BINARY_CLASS_ORDER = ("direct_or_non_wandering", "wandering_like")
_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
)
_DEVELOPMENT_SCOPE_FIELDS = (
    "participant_id",
    "session_id",
    "camera_setup_id",
    "clock_domain_id",
)


class CameraEpisodeError(ValueError):
    """A primary-window prediction cannot enter episode aggregation safely."""


def aggregate_episode_candidates(
    predictions: Sequence[Mapping[str, Any]],
    *,
    merge_gap_seconds: float,
) -> list[dict[str, Any]]:
    """Merge only adjacent same-shape ready windows under an explicit dev policy."""

    if (
        isinstance(merge_gap_seconds, bool)
        or not isinstance(merge_gap_seconds, (int, float))
        or not math.isfinite(float(merge_gap_seconds))
        or float(merge_gap_seconds) < 0.0
    ):
        raise CameraEpisodeError("episode merge_gap_seconds must be explicit, finite, and non-negative")
    validated = [_validate_prediction(row) for row in predictions]
    ordered = sorted(validated, key=_prediction_sort_key)

    episodes: list[dict[str, Any]] = []
    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in ordered:
        scope = _scope_key(row)
        status = row["window_status"]
        if status != "ready":
            active.pop(scope, None)
            continue
        current = active.get(scope)
        can_merge = (
            current is not None
            and current["predicted_pattern"] == row["four_class"]["predicted_label"]
            and float(row["window_start_sec"])
            - float(current["episode_end_sec_exclusive"])
            <= float(merge_gap_seconds)
        )
        if can_merge:
            _extend_episode(current, row)
            continue
        episode = _new_episode(row, merge_gap_seconds=float(merge_gap_seconds))
        episodes.append(episode)
        active[scope] = episode

    for episode in episodes:
        _finalize_episode(episode)
    return sorted(episodes, key=_episode_sort_key)


def _validate_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise CameraEpisodeError("episode input must contain prediction objects")
    value = dict(row)
    value.setdefault("evidence_scope", "synthetic_contract_only")
    required = {
        "schema_version",
        "window_id",
        "window_status",
        "parent_tracklet_id",
        *_SCOPE_FIELDS,
        "window_start_sec",
        "window_end_sec",
        "reason_codes",
        "quality_flags",
        "probability_calibrated",
        "model_invocation_skipped",
        "binary",
        "subtype",
        "four_class",
        "evidence_scope",
    }
    if not required.issubset(value):
        raise CameraEpisodeError("episode input prediction fields are incomplete")
    if value["schema_version"] != CAMERA_PRIMARY_PREDICTION_SCHEMA_VERSION:
        raise CameraEpisodeError("episode input prediction schema has drifted")
    if value["window_status"] not in {"ready", "unavailable", "inference_error"}:
        raise CameraEpisodeError("episode input prediction status is invalid")
    if not isinstance(value["window_id"], str) or not value["window_id"]:
        raise CameraEpisodeError("episode input window_id is invalid")
    if not isinstance(value["parent_tracklet_id"], str) or not value["parent_tracklet_id"]:
        raise CameraEpisodeError("episode input parent_tracklet_id is invalid")
    if not isinstance(value["evidence_scope"], str) or not value["evidence_scope"]:
        raise CameraEpisodeError("episode input evidence_scope is invalid")
    for name in _SCOPE_FIELDS[:-1]:
        if not isinstance(value[name], str) or not value[name]:
            raise CameraEpisodeError(f"episode input {name} is invalid")
    if not isinstance(value["track_id"], int) or isinstance(value["track_id"], bool):
        raise CameraEpisodeError("episode input track_id is invalid")
    development_values = [value.get(name) for name in _DEVELOPMENT_SCOPE_FIELDS]
    if any(item is not None for item in development_values):
        if any(not isinstance(item, str) or not item for item in development_values):
            raise CameraEpisodeError("episode input development scope is incomplete")
    start = _finite_float(value["window_start_sec"], "window_start_sec")
    end = _finite_float(value["window_end_sec"], "window_end_sec")
    if end <= start:
        raise CameraEpisodeError("episode input window interval is invalid")
    value["window_start_sec"] = start
    value["window_end_sec"] = end
    if value["probability_calibrated"] is not False:
        raise CameraEpisodeError("episode inputs must remain uncalibrated")

    if value["window_status"] == "ready":
        binary = _validate_named_probability(value["binary"], BINARY_CLASS_ORDER, "binary")
        four = _validate_named_probability(value["four_class"], FOUR_CLASS_ORDER, "four_class")
        if value["model_invocation_skipped"] is not False:
            raise CameraEpisodeError("ready episode input must have invoked the model")
        value["binary"] = binary
        value["four_class"] = four
    else:
        if any(value[name] is not None for name in ("binary", "subtype", "four_class")):
            raise CameraEpisodeError("non-ready episode inputs must not retain probabilities")
    return value


def _validate_named_probability(value: Any, order: tuple[str, ...], role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "class_order",
        "probabilities",
        "predicted_label",
    }:
        raise CameraEpisodeError(f"episode {role} probability object is invalid")
    if value["class_order"] != list(order) or value["predicted_label"] not in order:
        raise CameraEpisodeError(f"episode {role} class order or label has drifted")
    try:
        probabilities = np.asarray(value["probabilities"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CameraEpisodeError(f"episode {role} probabilities must be numeric") from exc
    if (
        probabilities.shape != (len(order),)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise CameraEpisodeError(f"episode {role} probabilities are invalid")
    if role == "binary":
        predicted_index = 1 if float(probabilities[1]) >= 0.5 else 0
    else:
        predicted_index = int(np.argmax(probabilities))
    if order[predicted_index] != value["predicted_label"]:
        raise CameraEpisodeError(f"episode {role} predicted label is inconsistent")
    return {
        "class_order": list(order),
        "probabilities": probabilities.tolist(),
        "predicted_label": value["predicted_label"],
    }


def _scope_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    development_scope = tuple(row.get(name) for name in _DEVELOPMENT_SCOPE_FIELDS)
    return (
        tuple(row[name] for name in _SCOPE_FIELDS)
        + development_scope
        + (row["parent_tracklet_id"], row["evidence_scope"])
    )


def _prediction_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return _scope_key(row) + (
        float(row["window_start_sec"]),
        float(row["window_end_sec"]),
        str(row["window_id"]),
    )


def _episode_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in _SCOPE_FIELDS) + (
        row["parent_tracklet_id"],
        row["episode_start_sec"],
        row["episode_candidate_id"],
    )


def _new_episode(row: Mapping[str, Any], *, merge_gap_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_CANDIDATE_SCHEMA_VERSION,
        "episode_candidate_id": "",
        **{name: row[name] for name in _SCOPE_FIELDS},
        **{
            name: row[name]
            for name in _DEVELOPMENT_SCOPE_FIELDS
            if row.get(name) is not None
        },
        "parent_tracklet_id": row["parent_tracklet_id"],
        "episode_start_sec": row["window_start_sec"],
        "episode_end_sec_exclusive": row["window_end_sec"],
        "contributing_window_ids": [row["window_id"]],
        "predicted_pattern": row["four_class"]["predicted_label"],
        "binary_probability_summary": [row["binary"]["probabilities"]],
        "four_class_probability_summary": [row["four_class"]["probabilities"]],
        "prediction_status": PREDICTION_STATUS,
        "evidence_scope": row["evidence_scope"],
        "policy_status": POLICY_STATUS,
        "merge_gap_seconds": merge_gap_seconds,
        "probability_calibrated": False,
        "alert_decision": None,
    }


def _extend_episode(episode: dict[str, Any], row: Mapping[str, Any]) -> None:
    episode["episode_end_sec_exclusive"] = max(
        float(episode["episode_end_sec_exclusive"]),
        float(row["window_end_sec"]),
    )
    episode["contributing_window_ids"].append(row["window_id"])
    episode["binary_probability_summary"].append(row["binary"]["probabilities"])
    episode["four_class_probability_summary"].append(row["four_class"]["probabilities"])


def _finalize_episode(episode: dict[str, Any]) -> None:
    binary = np.mean(np.asarray(episode["binary_probability_summary"], dtype=np.float64), axis=0)
    four = np.mean(np.asarray(episode["four_class_probability_summary"], dtype=np.float64), axis=0)
    episode["binary_probability_summary"] = {
        "class_order": list(BINARY_CLASS_ORDER),
        "mean_probabilities": binary.tolist(),
    }
    episode["four_class_probability_summary"] = {
        "class_order": list(FOUR_CLASS_ORDER),
        "mean_probabilities": four.tolist(),
    }
    identity = {
        **{name: episode[name] for name in _SCOPE_FIELDS},
        **{
            name: episode[name]
            for name in _DEVELOPMENT_SCOPE_FIELDS
            if episode.get(name) is not None
        },
        "parent_tracklet_id": episode["parent_tracklet_id"],
        "contributing_window_ids": episode["contributing_window_ids"],
        "predicted_pattern": episode["predicted_pattern"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    episode["episode_candidate_id"] = f"episode-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CameraEpisodeError(f"episode input {name} must be finite")
    return float(value)


__all__ = [
    "CAMERA_EPISODE_CANDIDATE_SCHEMA_VERSION",
    "CameraEpisodeError",
    "aggregate_episode_candidates",
]
