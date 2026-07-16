from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid


PREDICTION_REQUIRED_FIELDS = {
    "video_id",
    "task_type",
    "event_type",
    "prediction_id",
    "score",
    "start_time",
    "end_time",
    "onset_time",
    "status",
    "quality_state",
    "model_version",
    "config_hash",
    "split_id",
}

ELIGIBLE_REVIEW_STATUSES = {"reviewed", "final"}
PROVISIONAL_PROTOCOL_STATUSES = {"development", "provisional", "development_provisional"}


@dataclass(frozen=True)
class EventEvaluationResult:
    metrics: dict[str, Any]
    matches: list[dict[str, Any]]
    false_positives: list[dict[str, Any]]
    false_negatives: list[dict[str, Any]]
    excluded_samples: list[dict[str, Any]]
    threshold_curve: list[dict[str, Any]]
    confidence_intervals: dict[str, Any]
    protocol_version: str
    protocol_status: str
    config_hash: str


def evaluate_event_predictions(
    ground_truth: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    manifest: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> EventEvaluationResult:
    """Evaluate event predictions under a deterministic one-to-one protocol."""
    normalized_config = _validate_config(config)
    config_hash = _canonical_hash(normalized_config)
    truth_rows = [dict(row) for row in ground_truth]
    prediction_rows = [dict(row) for row in predictions]
    manifest_rows = [dict(row) for row in manifest]

    _validate_unique_ids(truth_rows, "label_id", "ground truth")
    _validate_predictions(prediction_rows, normalized_config)

    eligible_truth, excluded_truth = _filter_ground_truth(truth_rows)
    eligible_predictions, excluded_predictions = _filter_predictions(prediction_rows)
    _validate_task_type(eligible_truth, eligible_predictions, normalized_config)
    _validate_split_ids(eligible_truth, eligible_predictions)

    score_threshold = float(normalized_config["score_threshold"])
    selected_predictions = [
        row for row in eligible_predictions if float(row["score"]) >= score_threshold
    ]
    selected_predictions = _mark_reset_gap_violations(
        selected_predictions, normalized_config
    )
    selected_predictions = _merge_predictions(selected_predictions, normalized_config)
    matched_pairs = _match_events(
        eligible_truth,
        selected_predictions,
        normalized_config,
    )
    matches, false_positives, false_negatives = _build_match_outputs(
        eligible_truth,
        selected_predictions,
        matched_pairs,
        manifest_rows,
        normalized_config,
    )

    threshold_curve = _build_threshold_curve(
        eligible_truth,
        eligible_predictions,
        normalized_config,
    )
    metrics = _build_metrics(
        eligible_truth,
        selected_predictions,
        matches,
        false_positives,
        false_negatives,
        threshold_curve,
        manifest_rows,
        normalized_config,
    )
    confidence_intervals = _cluster_bootstrap(
        eligible_truth,
        eligible_predictions,
        manifest_rows,
        normalized_config,
    )
    excluded = sorted(
        excluded_truth + excluded_predictions,
        key=lambda row: (
            str(row.get("video_id", "")),
            str(row.get("label_id", row.get("prediction_id", ""))),
        ),
    )

    return EventEvaluationResult(
        metrics=metrics,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        excluded_samples=excluded,
        threshold_curve=threshold_curve,
        confidence_intervals=confidence_intervals,
        protocol_version=str(normalized_config["protocol_version"]),
        protocol_status=str(normalized_config["protocol_status"]),
        config_hash=config_hash,
    )


def write_evaluation_bundle(
    result: EventEvaluationResult,
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    """Stage a complete evidence bundle and atomically commit the directory."""
    directory = Path(output_dir)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.staging.",
            dir=directory.parent,
        )
    )
    try:
        _write_bundle_contents(result, staging, metadata)
        _commit_staged_directory(staging, directory, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_bundle_contents(
    result: EventEvaluationResult,
    directory: Path,
    metadata: Mapping[str, Any],
) -> None:
    metrics_payload = {
        **result.metrics,
        "confidence_intervals_95": result.confidence_intervals,
        "protocol_version": result.protocol_version,
        "protocol_status": result.protocol_status,
        "evaluation_config_hash": result.config_hash,
        "reproducibility": dict(metadata),
    }
    _atomic_write_text(
        directory / "metrics.json",
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_jsonl(directory / "matches.jsonl", result.matches)
    _atomic_write_jsonl(
        directory / "false_positives.jsonl", result.false_positives
    )
    _atomic_write_jsonl(
        directory / "false_negatives.jsonl", result.false_negatives
    )
    _atomic_write_jsonl(
        directory / "excluded_samples.jsonl", result.excluded_samples
    )
    _atomic_write_text(
        directory / "threshold_curve.csv",
        _threshold_curve_csv(result.threshold_curve),
    )
    _atomic_write_text(
        directory / "report.md",
        _render_report(result, metadata),
    )


def _commit_staged_directory(
    staging: Path, directory: Path, *, overwrite: bool
) -> None:
    lock_path = directory.parent / f".{directory.name}.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"evaluation output commit is locked: {directory}"
        ) from exc

    backup: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
            lock_handle.flush()
            os.fsync(lock_handle.fileno())

        if os.path.lexists(directory):
            if not overwrite:
                raise FileExistsError(
                    f"evaluation output directory already exists: {directory}"
                )
            backup = directory.parent / (
                f".{directory.name}.backup.{os.getpid()}.{uuid.uuid4().hex}"
            )
            os.replace(directory, backup)

        try:
            os.replace(staging, directory)
            _fsync_directory(directory.parent)
        except BaseException:
            if backup is not None:
                if os.path.lexists(directory):
                    _remove_path(directory)
                os.replace(backup, directory)
                backup = None
                _fsync_directory(directory.parent)
            raise

        if backup is not None:
            _remove_path(backup)
            backup = None
    finally:
        if backup is not None and not os.path.lexists(directory):
            os.replace(backup, directory)
        lock_path.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "protocol_version",
        "protocol_status",
        "score_threshold",
        "iou_threshold",
        "onset_tolerance_sec",
        "search_window_before_sec",
        "search_window_after_sec",
        "matching_operator",
        "one_to_one_assignment",
        "event_merge_enabled",
        "event_merge_gap_sec",
        "event_reset_gap_sec",
        "duplicate_alert_policy",
        "ground_truth_event_types",
        "pr_auc_method",
        "reset_rule_application",
        "event_type_compatibility",
        "bootstrap_iterations",
        "bootstrap_seed",
        "minimum_clusters_for_inference",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"evaluation config is missing fields: {', '.join(missing)}")

    normalized = dict(config)
    if not str(normalized["protocol_version"]).strip():
        raise ValueError("config.protocol_version must not be empty")
    if str(normalized["protocol_status"]) not in {
        *PROVISIONAL_PROTOCOL_STATUSES,
        "frozen",
    }:
        raise ValueError("config.protocol_status must be development/provisional or frozen")
    for field in ("score_threshold", "iou_threshold"):
        value = _finite_number(normalized[field], f"config.{field}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"config.{field} must be between 0 and 1")
        normalized[field] = value
    for field in (
        "onset_tolerance_sec",
        "search_window_before_sec",
        "search_window_after_sec",
        "event_merge_gap_sec",
        "event_reset_gap_sec",
    ):
        value = _finite_number(normalized[field], f"config.{field}")
        if value < 0:
            raise ValueError(f"config.{field} must be non-negative")
        normalized[field] = value
    for field in ("bootstrap_iterations", "minimum_clusters_for_inference"):
        value = int(normalized[field])
        if value < 0:
            raise ValueError(f"config.{field} must be non-negative")
        normalized[field] = value
    normalized["bootstrap_seed"] = int(normalized["bootstrap_seed"])
    if normalized["minimum_clusters_for_inference"] < 1:
        raise ValueError("config.minimum_clusters_for_inference must be positive")
    if (
        str(normalized["protocol_status"]) == "frozen"
        and normalized["bootstrap_iterations"] < 10_000
    ):
        raise ValueError("frozen evaluation requires at least 10,000 bootstrap iterations")
    if normalized["matching_operator"] not in {"iou_or_onset", "iou_and_onset"}:
        raise ValueError("config.matching_operator must be iou_or_onset or iou_and_onset")
    if normalized["one_to_one_assignment"] != "maximum_cardinality_deterministic":
        raise ValueError(
            "config.one_to_one_assignment must be maximum_cardinality_deterministic"
        )
    if not isinstance(normalized["event_merge_enabled"], bool):
        raise ValueError("config.event_merge_enabled must be boolean")
    if normalized["duplicate_alert_policy"] != "unmatched_is_false_positive":
        raise ValueError(
            "config.duplicate_alert_policy must be unmatched_is_false_positive"
        )
    if normalized["pr_auc_method"] != "average_precision_step":
        raise ValueError("config.pr_auc_method must be average_precision_step")
    if normalized["reset_rule_application"] != "audit_only_postprocessed_predictions":
        raise ValueError(
            "config.reset_rule_application must be audit_only_postprocessed_predictions"
        )
    ground_truth_types = normalized["ground_truth_event_types"]
    if not isinstance(ground_truth_types, list) or not ground_truth_types or any(
        not isinstance(value, str) or not value for value in ground_truth_types
    ):
        raise ValueError("config.ground_truth_event_types must be a non-empty list")
    normalized["ground_truth_event_types"] = sorted(set(ground_truth_types))

    compatibility = normalized["event_type_compatibility"]
    if not isinstance(compatibility, Mapping) or not compatibility:
        raise ValueError("config.event_type_compatibility must be a non-empty mapping")
    normalized_compatibility: dict[str, list[str]] = {}
    for key, values in compatibility.items():
        if not isinstance(values, (list, tuple, set)) or not values:
            raise ValueError(
                "each config.event_type_compatibility value must be a non-empty list"
            )
        normalized_compatibility[str(key)] = sorted({str(value) for value in values})
    normalized["event_type_compatibility"] = normalized_compatibility
    return normalized


def _validate_unique_ids(
    rows: Sequence[Mapping[str, Any]], field: str, kind: str
) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        raw_value = row.get(field)
        if raw_value in (None, ""):
            raise ValueError(f"{kind} row {index} is missing {field}")
        value = str(raw_value)
        if value in seen:
            raise ValueError(f"duplicate {field} in {kind}: {value}")
        seen.add(value)


def _validate_predictions(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    _validate_unique_ids(rows, "prediction_id", "predictions")
    signatures: set[tuple[str, str, str]] = set()
    provisional = str(config["protocol_status"]) in PROVISIONAL_PROTOCOL_STATUSES
    for index, row in enumerate(rows, start=1):
        missing = sorted(PREDICTION_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"prediction row {index} is missing fields: {', '.join(missing)}"
            )
        start = _finite_number(row["start_time"], f"prediction {index}.start_time")
        end = _finite_number(row["end_time"], f"prediction {index}.end_time")
        onset = _finite_number(row["onset_time"], f"prediction {index}.onset_time")
        score = _finite_number(row["score"], f"prediction {index}.score")
        if start > end:
            raise ValueError(f"prediction row {index} has start_time after end_time")
        if start < 0 or end < 0 or not start <= onset <= end:
            raise ValueError(
                f"prediction row {index} onset_time must be inside a non-negative interval"
            )
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"prediction row {index} score must be between 0 and 1")
        if "alert_time" in row:
            alert_time = _finite_number(
                row["alert_time"], f"prediction {index}.alert_time"
            )
            if alert_time < 0:
                raise ValueError(
                    f"prediction row {index} alert_time must be non-negative"
                )
        elif not provisional:
            raise ValueError(
                "frozen evaluation requires prediction alert_time; onset_time is an event boundary"
            )
        if "risk_level" in row:
            risk_level = row["risk_level"]
            if (
                isinstance(risk_level, bool)
                or not isinstance(risk_level, int)
                or not 0 <= risk_level <= 4
            ):
                raise ValueError(
                    f"prediction row {index} risk_level must be an integer from 0 to 4"
                )
        if not str(row["video_id"]) or not str(row["task_type"]):
            raise ValueError(f"prediction row {index} has an empty video_id or task_type")
        config_hash = str(row["config_hash"])
        if len(config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_hash.lower()
        ):
            raise ValueError(f"prediction row {index} has an invalid config_hash")
        signatures.add(
            (
                str(row["model_version"]),
                str(row["config_hash"]),
                str(row["split_id"]),
            )
        )
        _ = onset
    if len(signatures) > 1:
        raise ValueError(
            "prediction file mixes model_version, config_hash, or split_id values"
        )


def _filter_ground_truth(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        reason: str | None = None
        if str(row.get("event_type")) == "uncertain":
            reason = "uncertain_ground_truth"
        elif str(row.get("review_status")) not in ELIGIBLE_REVIEW_STATUSES:
            reason = "ineligible_review_status"
        elif row.get("eligibility") is not True:
            reason = "ineligible_ground_truth"
        elif row.get("source_exists") is False:
            reason = "missing_source"
        if reason is not None:
            excluded.append({**row, "record_type": "ground_truth", "exclusion_reason": reason})
            continue
        _validate_truth_row(row)
        eligible.append(row)
    return sorted(eligible, key=_truth_sort_key), excluded


def _validate_truth_row(row: Mapping[str, Any]) -> None:
    required = {
        "label_id",
        "video_id",
        "task_type",
        "event_type",
        "start_time",
        "end_time",
        "review_status",
        "split_id",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"ground truth is missing fields: {', '.join(missing)}")
    start = _finite_number(row["start_time"], "ground_truth.start_time")
    end = _finite_number(row["end_time"], "ground_truth.end_time")
    if start > end:
        raise ValueError("ground truth has start_time after end_time")
    if "onset_time" in row:
        onset = _finite_number(row["onset_time"], "ground_truth.onset_time")
        if start < 0 or end < 0 or not start <= onset <= end:
            raise ValueError(
                "ground truth onset_time must be inside a non-negative interval"
            )


def _filter_predictions(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status")) != "emitted":
            excluded.append(
                {
                    **row,
                    "record_type": "prediction",
                    "exclusion_reason": "prediction_not_emitted",
                }
            )
        else:
            eligible.append(row)
    return sorted(eligible, key=_prediction_sort_key), excluded


def _validate_split_ids(
    truths: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> None:
    split_ids = {
        str(row["split_id"])
        for row in [*truths, *predictions]
        if row.get("split_id") not in (None, "")
    }
    if len(split_ids) > 1:
        raise ValueError(f"split_id mismatch: {', '.join(sorted(split_ids))}")


def _validate_task_type(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    expected = config.get("task_type")
    if not expected:
        return
    unexpected = sorted(
        {
            str(row.get("task_type"))
            for row in [*truths, *predictions]
            if str(row.get("task_type")) != str(expected)
        }
    )
    if unexpected:
        raise ValueError(
            f"evaluation config task_type {expected!r} does not allow: {unexpected}"
        )


def _match_events(
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[tuple[int, int, float, float]]:
    candidates_by_truth: dict[
        int,
        list[
            tuple[
                tuple[float, float, float, float, str],
                int,
                float,
                float,
            ]
        ],
    ] = {}
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            metrics = _candidate_metrics(truth, prediction, config)
            if metrics is None:
                continue
            iou, onset_delta = metrics
            rank = (
                float(prediction.get("alert_time", prediction["onset_time"])),
                -iou,
                onset_delta,
                -float(prediction["score"]),
                str(prediction["prediction_id"]),
            )
            candidates_by_truth.setdefault(truth_index, []).append(
                (rank, prediction_index, iou, onset_delta)
            )

    for candidates in candidates_by_truth.values():
        candidates.sort(key=lambda candidate: candidate[0])
    assigned_truth_by_prediction: dict[int, int] = {}

    def assign(truth_index: int, visited_predictions: set[int]) -> bool:
        for _, prediction_index, _, _ in candidates_by_truth.get(truth_index, []):
            if prediction_index in visited_predictions:
                continue
            visited_predictions.add(prediction_index)
            previous_truth = assigned_truth_by_prediction.get(prediction_index)
            if previous_truth is None or assign(previous_truth, visited_predictions):
                assigned_truth_by_prediction[prediction_index] = truth_index
                return True
        return False

    truth_order = sorted(
        candidates_by_truth,
        key=lambda index: (len(candidates_by_truth[index]), _truth_sort_key(truths[index])),
    )
    for truth_index in truth_order:
        assign(truth_index, set())

    pairs: list[tuple[int, int, float, float]] = []
    for prediction_index, truth_index in assigned_truth_by_prediction.items():
        detail = next(
            candidate
            for candidate in candidates_by_truth[truth_index]
            if candidate[1] == prediction_index
        )
        pairs.append((truth_index, prediction_index, detail[2], detail[3]))
    return sorted(pairs, key=lambda item: _truth_sort_key(truths[item[0]]))


def _candidate_metrics(
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[float, float] | None:
    if str(truth["video_id"]) != str(prediction["video_id"]):
        return None
    if str(truth["task_type"]) != str(prediction["task_type"]):
        return None
    allowed_types = set(
        config["event_type_compatibility"].get(str(truth["event_type"]), [])
    )
    if str(prediction["event_type"]) not in allowed_types:
        return None
    iou = _interval_iou(
        float(truth["start_time"]),
        float(truth["end_time"]),
        float(prediction["start_time"]),
        float(prediction["end_time"]),
    )
    truth_onset = float(truth.get("onset_time", truth["start_time"]))
    prediction_onset = float(prediction["onset_time"])
    onset_delta = abs(prediction_onset - truth_onset)
    in_search_window = (
        prediction_onset
        >= truth_onset - float(config["search_window_before_sec"]) - 1e-12
        and prediction_onset
        <= float(truth["end_time"])
        + float(config["search_window_after_sec"])
        + 1e-12
    )
    passes_iou = iou + 1e-12 >= float(config["iou_threshold"])
    passes_onset = onset_delta <= float(config["onset_tolerance_sec"]) + 1e-12
    if config["matching_operator"] == "iou_and_onset":
        passes = passes_iou and passes_onset
    else:
        passes = passes_iou or passes_onset
    return (iou, onset_delta) if in_search_window and passes else None


def _build_match_outputs(
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    pairs: Sequence[tuple[int, int, float, float]],
    manifest: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched_truth = {pair[0] for pair in pairs}
    matched_predictions = {pair[1] for pair in pairs}
    cluster_by_video = _cluster_by_video(manifest, truths, predictions)
    matches: list[dict[str, Any]] = []
    for truth_index, prediction_index, iou, onset_delta in pairs:
        truth = truths[truth_index]
        prediction = predictions[prediction_index]
        alert_time = float(prediction.get("alert_time", prediction["onset_time"]))
        truth_onset = float(truth.get("onset_time", truth["start_time"]))
        match = {
            "label_id": str(truth["label_id"]),
            "prediction_id": str(prediction["prediction_id"]),
            "video_id": str(truth["video_id"]),
            "task_type": str(truth["task_type"]),
            "ground_truth_event_type": str(truth["event_type"]),
            "prediction_event_type": str(prediction["event_type"]),
            "score": float(prediction["score"]),
            "boundary_iou": round(iou, 12),
            "onset_absolute_error_sec": round(onset_delta, 12),
            "onset_detection_latency_sec": round(alert_time - truth_onset, 12),
            "detection_latency_sec": round(alert_time - truth_onset, 12),
            "alert_time": alert_time,
            "reset_gap_violation": bool(prediction.get("reset_gap_violation", False)),
            "cluster_id": _cluster_id(truth, cluster_by_video),
        }
        first_level_3_alert = _first_level_3_alert_time(
            truth, predictions, config
        )
        if truth.get("reference_event_start") is not None and first_level_3_alert is not None:
            match["first_level_3_alert_time"] = first_level_3_alert
            match["lead_time_sec"] = round(
                float(truth["reference_event_start"]) - first_level_3_alert, 12
            )
        matches.append(match)

    false_positives = [
        {
            **predictions[index],
            "exclusion_reason": "unmatched_prediction",
            "cluster_id": _cluster_id(predictions[index], cluster_by_video),
        }
        for index in range(len(predictions))
        if index not in matched_predictions
    ]
    false_negatives = [
        {
            **truths[index],
            "exclusion_reason": "unmatched_ground_truth",
            "cluster_id": _cluster_id(truths[index], cluster_by_video),
        }
        for index in range(len(truths))
        if index not in matched_truth
    ]
    return (
        sorted(matches, key=lambda row: (row["video_id"], row["label_id"])),
        sorted(false_positives, key=_prediction_sort_key),
        sorted(false_negatives, key=_truth_sort_key),
    )


def _first_level_3_alert_time(
    truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float | None:
    alert_times = [
        float(prediction.get("alert_time", prediction["onset_time"]))
        for prediction in predictions
        if int(prediction.get("risk_level", 0)) >= 3
        and _candidate_metrics(truth, prediction, config) is not None
    ]
    return min(alert_times) if alert_times else None


def _build_threshold_curve(
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    thresholds = sorted({float(row["score"]) for row in predictions}, reverse=True)
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [row for row in predictions if float(row["score"]) >= threshold]
        selected = _mark_reset_gap_violations(selected, config)
        selected = _merge_predictions(selected, config)
        pairs = _match_events(truths, selected, config)
        tp = len(pairs)
        fp = len(selected) - tp
        fn = len(truths) - tp
        precision, recall, f1 = _rates(tp, fp, fn)
        curve.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return curve


def _merge_predictions(
    predictions: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not config["event_merge_enabled"]:
        return list(predictions)
    gap = float(config["event_merge_gap_sec"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (
            str(row["video_id"]),
            str(row["task_type"]),
            str(row["event_type"]),
        )
        grouped.setdefault(key, []).append(row)

    merged_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=_prediction_sort_key)
        current = dict(rows[0])
        constituent_ids = [str(current["prediction_id"])]
        for row in rows[1:]:
            if float(row["start_time"]) - float(current["end_time"]) > gap:
                merged_rows.append(_finalize_merged_prediction(current, constituent_ids))
                current = dict(row)
                constituent_ids = [str(current["prediction_id"])]
                continue
            constituent_ids.append(str(row["prediction_id"]))
            current["start_time"] = min(
                float(current["start_time"]), float(row["start_time"])
            )
            current["end_time"] = max(
                float(current["end_time"]), float(row["end_time"])
            )
            current["onset_time"] = min(
                float(current["onset_time"]), float(row["onset_time"])
            )
            if current.get("alert_time") is not None or row.get("alert_time") is not None:
                current["alert_time"] = min(
                    float(current.get("alert_time", current["onset_time"])),
                    float(row.get("alert_time", row["onset_time"])),
                )
            current["score"] = max(float(current["score"]), float(row["score"]))
            current["risk_level"] = max(
                int(current.get("risk_level", 0)), int(row.get("risk_level", 0))
            )
        merged_rows.append(_finalize_merged_prediction(current, constituent_ids))
    return sorted(merged_rows, key=_prediction_sort_key)


def _mark_reset_gap_violations(
    predictions: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    reset_gap = float(config["event_reset_gap_sec"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for source in predictions:
        row = dict(source)
        key = (
            str(row["video_id"]),
            str(row["task_type"]),
            str(row["event_type"]),
        )
        grouped.setdefault(key, []).append(row)

    marked: list[dict[str, Any]] = []
    for key in sorted(grouped):
        previous_end: float | None = None
        for row in sorted(grouped[key], key=_prediction_sort_key):
            start = float(row["start_time"])
            row["reset_gap_violation"] = bool(
                previous_end is not None and start - previous_end < reset_gap
            )
            previous_end = max(previous_end or float(row["end_time"]), float(row["end_time"]))
            marked.append(row)
    return sorted(marked, key=_prediction_sort_key)


def _finalize_merged_prediction(
    row: dict[str, Any], constituent_ids: Sequence[str]
) -> dict[str, Any]:
    if len(constituent_ids) == 1:
        return row
    ids = sorted(constituent_ids)
    row["constituent_prediction_ids"] = ids
    row["prediction_id"] = "merged_" + hashlib.sha256(
        "\n".join(ids).encode("utf-8")
    ).hexdigest()[:20]
    return row


def _build_metrics(
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    false_positives: Sequence[dict[str, Any]],
    false_negatives: Sequence[dict[str, Any]],
    threshold_curve: Sequence[dict[str, Any]],
    manifest: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision, recall, f1 = _rates(tp, fp, fn)
    camera_hours, continuous_videos = _continuous_camera_hours(manifest)
    continuous_fp = sum(
        1 for row in false_positives if str(row.get("video_id")) in continuous_videos
    )
    household_days, household_videos = _continuous_household_days(manifest)
    household_fp = sum(
        1 for row in false_positives if str(row.get("video_id")) in household_videos
    )
    recovery_truth = sum(1 for row in truths if row.get("event_type") == "recovery")
    recovery_matches = sum(
        1 for row in matches if row.get("ground_truth_event_type") == "recovery"
    )
    recovery_predictions = sum(
        1 for row in predictions if row.get("event_type") == "recovery"
    )
    recovery_fp = sum(
        1 for row in false_positives if row.get("event_type") == "recovery"
    )
    lead_times = [
        float(row["lead_time_sec"]) for row in matches if "lead_time_sec" in row
    ]
    boundary_ious = [float(row["boundary_iou"]) for row in matches]
    detection_latencies = [
        float(row["detection_latency_sec"])
        for row in matches
        if row.get("ground_truth_event_type") == "fall"
    ]
    onset_latencies = [
        float(row["onset_detection_latency_sec"])
        for row in matches
        if row.get("ground_truth_event_type") == "near_fall"
    ]

    metrics: dict[str, Any] = {
        "ground_truth_count": len(truths),
        "prediction_count": len(predictions),
        "score_threshold": float(config["score_threshold"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_pr_auc": _event_pr_auc(threshold_curve, len(truths)),
        "pr_auc_method": str(config["pr_auc_method"]),
        "miss_rate": _divide(fn, tp + fn),
        "false_discovery_rate": _divide(fp, tp + fp),
        "traditional_fpr": None,
        "traditional_fpr_unavailable_reason": (
            "registered_true_negative_observation_units_required"
        ),
        "negative_observation_count": None,
        "mean_boundary_iou": _mean(boundary_ious),
        "mean_onset_detection_latency_sec": _mean(onset_latencies),
        "mean_detection_latency_sec": _mean(detection_latencies),
        "lead_times_sec": sorted(lead_times),
        "mean_lead_time_sec": _mean(lead_times),
        "continuous_camera_hours": camera_hours,
        "continuous_false_positive_count": continuous_fp,
        "fp_per_camera_hour": _divide(continuous_fp, camera_hours),
        "continuous_household_days": household_days,
        "household_false_positive_count": household_fp,
        "fp_per_household_day": _divide(household_fp, household_days),
        "recovery_recall": _divide(recovery_matches, recovery_truth),
        "erroneous_recovery_rate": _divide(recovery_fp, recovery_predictions),
        "manual_review_alert_count": len(predictions),
        "alert_time_fallback_count": sum(
            1 for row in predictions if row.get("alert_time") is None
        ),
        "reset_gap_violation_count": sum(
            1 for row in predictions if row.get("reset_gap_violation") is True
        ),
        "manual_review_alerts_per_camera_hour": _divide(
            sum(
                1
                for row in predictions
                if str(row.get("video_id")) in continuous_videos
            ),
            camera_hours,
        ),
        "matching_protocol": {
            "iou_threshold": float(config["iou_threshold"]),
            "onset_tolerance_sec": float(config["onset_tolerance_sec"]),
            "search_window_before_sec": float(config["search_window_before_sec"]),
            "search_window_after_sec": float(config["search_window_after_sec"]),
            "matching_operator": str(config["matching_operator"]),
            "one_to_one_assignment": str(config["one_to_one_assignment"]),
            "pr_auc_method": str(config["pr_auc_method"]),
            "reset_rule_application": str(config["reset_rule_application"]),
            "event_merge_enabled": bool(config["event_merge_enabled"]),
            "event_merge_gap_sec": float(config["event_merge_gap_sec"]),
            "event_reset_gap_sec": float(config["event_reset_gap_sec"]),
            "duplicate_alert_policy": str(config["duplicate_alert_policy"]),
            "event_type_compatibility": dict(config["event_type_compatibility"]),
        },
    }
    return metrics


def _cluster_bootstrap(
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    manifest: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_by_video = _cluster_by_video(manifest, truths, predictions)
    clusters = sorted(
        {
            _cluster_id(row, cluster_by_video)
            for row in [*truths, *predictions, *manifest]
            if row.get("video_id") not in (None, "")
            or row.get("asset_id") not in (None, "")
        }
    )
    thresholds = sorted({float(row["score"]) for row in predictions}, reverse=True)
    profiles = {
        cluster: _cluster_evaluation_profile(
            cluster,
            truths,
            predictions,
            manifest,
            cluster_by_video,
            thresholds,
            config,
        )
        for cluster in clusters
    }
    iterations = int(config["bootstrap_iterations"])
    minimum = int(config["minimum_clusters_for_inference"])
    metric_names = (
        "precision",
        "recall",
        "f1",
        "event_pr_auc",
        "mean_boundary_iou",
        "mean_detection_latency_sec",
        "mean_onset_detection_latency_sec",
        "mean_lead_time_sec",
        "fp_per_camera_hour",
        "recovery_recall",
        "erroneous_recovery_rate",
    )
    payload: dict[str, Any] = {
        "method": "cluster_percentile_bootstrap",
        "pr_auc_method": str(config["pr_auc_method"]),
        "cluster_count": len(clusters),
        "iterations": iterations if clusters else 0,
        "seed": int(config["bootstrap_seed"]),
        "exploratory": len(clusters) < minimum or iterations <= 0,
        **{name: None for name in metric_names},
    }
    if not clusters or iterations <= 0:
        return payload

    rng = random.Random(int(config["bootstrap_seed"]))
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        selected = rng.choices(clusters, k=len(clusters))
        tp = sum(profiles[key]["tp"] for key in selected)
        fp = sum(profiles[key]["fp"] for key in selected)
        fn = sum(profiles[key]["fn"] for key in selected)
        precision, recall, f1 = _rates(tp, fp, fn)
        truth_count = sum(profiles[key]["truth_count"] for key in selected)
        curve: list[dict[str, Any]] = []
        for threshold_index, threshold in enumerate(thresholds):
            curve_tp = sum(
                profiles[key]["curve"][threshold_index]["tp"] for key in selected
            )
            curve_fp = sum(
                profiles[key]["curve"][threshold_index]["fp"] for key in selected
            )
            curve_fn = sum(
                profiles[key]["curve"][threshold_index]["fn"] for key in selected
            )
            curve_precision, curve_recall, curve_f1 = _rates(
                curve_tp, curve_fp, curve_fn
            )
            curve.append(
                {
                    "threshold": threshold,
                    "tp": curve_tp,
                    "fp": curve_fp,
                    "fn": curve_fn,
                    "precision": curve_precision,
                    "recall": curve_recall,
                    "f1": curve_f1,
                }
            )
        event_pr_auc = _event_pr_auc(curve, truth_count)
        boundary_iou = _mean(
            [value for key in selected for value in profiles[key]["boundary_iou"]]
        )
        detection_latency = _mean(
            [
                value
                for key in selected
                for value in profiles[key]["detection_latency"]
            ]
        )
        lead_time = _mean(
            [value for key in selected for value in profiles[key]["lead_time"]]
        )
        onset_latency = _mean(
            [value for key in selected for value in profiles[key]["onset_latency"]]
        )
        camera_hours = sum(profiles[key]["camera_hours"] for key in selected)
        continuous_fp = sum(profiles[key]["continuous_fp"] for key in selected)
        fp_per_camera_hour = _divide(continuous_fp, camera_hours)
        recovery_matches = sum(profiles[key]["recovery_matches"] for key in selected)
        recovery_truth = sum(profiles[key]["recovery_truth"] for key in selected)
        recovery_fp = sum(profiles[key]["recovery_fp"] for key in selected)
        recovery_predictions = sum(
            profiles[key]["recovery_predictions"] for key in selected
        )
        for name, value in (
            ("precision", precision),
            ("recall", recall),
            ("f1", f1),
            ("event_pr_auc", event_pr_auc),
            ("mean_boundary_iou", boundary_iou),
            ("mean_detection_latency_sec", detection_latency),
            ("mean_onset_detection_latency_sec", onset_latency),
            ("mean_lead_time_sec", lead_time),
            ("fp_per_camera_hour", fp_per_camera_hour),
            ("recovery_recall", _divide(recovery_matches, recovery_truth)),
            (
                "erroneous_recovery_rate",
                _divide(recovery_fp, recovery_predictions),
            ),
        ):
            if value is not None:
                samples[name].append(value)
    for name, values in samples.items():
        if values:
            payload[name] = {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
            }
    return payload


def _cluster_evaluation_profile(
    cluster: str,
    truths: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    manifest: Sequence[dict[str, Any]],
    cluster_by_video: Mapping[str, str],
    thresholds: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_truths = [
        row for row in truths if _cluster_id(row, cluster_by_video) == cluster
    ]
    cluster_predictions = [
        row for row in predictions if _cluster_id(row, cluster_by_video) == cluster
    ]
    cluster_manifest = [
        row for row in manifest if _cluster_id(row, cluster_by_video) == cluster
    ]
    selected = [
        row
        for row in cluster_predictions
        if float(row["score"]) >= float(config["score_threshold"])
    ]
    selected = _mark_reset_gap_violations(selected, config)
    selected = _merge_predictions(selected, config)
    pairs = _match_events(cluster_truths, selected, config)
    matches, false_positives, false_negatives = _build_match_outputs(
        cluster_truths,
        selected,
        pairs,
        cluster_manifest,
        config,
    )
    curve: list[dict[str, int]] = []
    for threshold in thresholds:
        threshold_predictions = [
            row for row in cluster_predictions if float(row["score"]) >= threshold
        ]
        threshold_predictions = _mark_reset_gap_violations(
            threshold_predictions, config
        )
        threshold_predictions = _merge_predictions(threshold_predictions, config)
        threshold_tp = len(
            _match_events(cluster_truths, threshold_predictions, config)
        )
        curve.append(
            {
                "tp": threshold_tp,
                "fp": len(threshold_predictions) - threshold_tp,
                "fn": len(cluster_truths) - threshold_tp,
            }
        )
    camera_hours, continuous_videos = _continuous_camera_hours(cluster_manifest)
    return {
        "truth_count": len(cluster_truths),
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "curve": curve,
        "boundary_iou": [float(row["boundary_iou"]) for row in matches],
        "detection_latency": [
            float(row["detection_latency_sec"])
            for row in matches
            if row.get("ground_truth_event_type") == "fall"
        ],
        "onset_latency": [
            float(row["onset_detection_latency_sec"])
            for row in matches
            if row.get("ground_truth_event_type") == "near_fall"
        ],
        "lead_time": [
            float(row["lead_time_sec"]) for row in matches if "lead_time_sec" in row
        ],
        "camera_hours": camera_hours or 0.0,
        "continuous_fp": sum(
            1
            for row in false_positives
            if str(row.get("video_id")) in continuous_videos
        ),
        "recovery_truth": sum(
            row.get("event_type") == "recovery" for row in cluster_truths
        ),
        "recovery_matches": sum(
            row.get("ground_truth_event_type") == "recovery" for row in matches
        ),
        "recovery_predictions": sum(
            row.get("event_type") == "recovery" for row in selected
        ),
        "recovery_fp": sum(
            row.get("event_type") == "recovery" for row in false_positives
        ),
    }


def _event_pr_auc(curve: Sequence[Mapping[str, Any]], truth_count: int) -> float | None:
    if truth_count == 0:
        return None
    if not curve:
        return 0.0
    area = 0.0
    previous_recall = 0.0
    for row in curve:
        recall = float(row["recall"] or 0.0)
        precision = float(row["precision"] or 0.0)
        if recall > previous_recall:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return round(area, 12)


def _rates(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union == 0:
        return 1.0 if a_start == b_start else 0.0
    return intersection / union


def _continuous_camera_hours(
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[float | None, set[str]]:
    seconds = 0.0
    videos: set[str] = set()
    for row in manifest:
        if row.get("eligibility") is not True:
            continue
        if row.get("continuous_monitoring_eligible") is not True:
            continue
        duration = row.get("continuous_duration_sec", row.get("duration_sec"))
        if duration is None:
            continue
        value = _finite_number(duration, "manifest.continuous_duration_sec")
        if value <= 0:
            continue
        seconds += value
        videos.add(str(row.get("video_id", row.get("asset_id"))))
    if seconds == 0:
        return None, videos
    return seconds / 3600.0, videos


def _continuous_household_days(
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[float | None, set[str]]:
    periods: dict[str, tuple[str, float]] = {}
    videos: set[str] = set()
    for row in manifest:
        if row.get("eligibility") is not True:
            continue
        if row.get("continuous_monitoring_eligible") is not True:
            continue
        household_id = str(row.get("household_id", "")).strip()
        period_id = str(row.get("household_monitoring_period_id", "")).strip()
        raw_days = row.get("continuous_household_days")
        if not household_id or not period_id or raw_days is None:
            continue
        days = _finite_number(raw_days, "manifest.continuous_household_days")
        if days <= 0:
            continue
        existing = periods.get(period_id)
        current = (household_id, days)
        if existing is not None and existing != current:
            raise ValueError(
                "household monitoring period has inconsistent household or duration"
            )
        periods[period_id] = current
        videos.add(str(row.get("video_id", row.get("asset_id"))))
    total_days = sum(days for _, days in periods.values())
    return (total_days or None), videos


def _cluster_by_video(
    manifest: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    rows_by_video: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for kind, rows in (
        ("truth", truths),
        ("manifest", manifest),
        ("prediction", predictions),
    ):
        for row in rows:
            video_id = str(row.get("video_id", row.get("asset_id", "")))
            if not video_id:
                continue
            rows_by_video.setdefault(
                video_id,
                {"truth": [], "manifest": [], "prediction": []},
            )[kind].append(row)

    mapping: dict[str, str] = {}
    for video_id, by_kind in sorted(rows_by_video.items()):
        trusted_rows = [*by_kind["truth"], *by_kind["manifest"]]
        subjects = _known_values(trusted_rows, "subject_id")
        if len(subjects) > 1:
            raise ValueError(f"conflicting subject_id values for video {video_id}")
        if not subjects:
            subjects = _known_values(by_kind["prediction"], "subject_id")
            if len(subjects) > 1:
                raise ValueError(f"conflicting prediction subject_id values for video {video_id}")
        if subjects:
            mapping[video_id] = f"subject:{next(iter(subjects))}"
            continue

        source_groups = _known_values(trusted_rows, "source_group_id")
        if len(source_groups) > 1:
            raise ValueError(f"conflicting source_group_id values for video {video_id}")
        if not source_groups:
            source_groups = _known_values(by_kind["prediction"], "source_group_id")
            if len(source_groups) > 1:
                raise ValueError(
                    f"conflicting prediction source_group_id values for video {video_id}"
                )
        mapping[video_id] = (
            f"source:{next(iter(source_groups))}"
            if source_groups
            else f"video:{video_id}"
        )
    return mapping


def _cluster_id(row: Mapping[str, Any], cluster_by_video: Mapping[str, str]) -> str:
    video_id = str(row.get("video_id", row.get("asset_id", "unknown")))
    if video_id in cluster_by_video:
        return cluster_by_video[video_id]
    subject = str(row.get("subject_id", "unknown"))
    if subject and subject != "unknown":
        return f"subject:{subject}"
    source_group = str(row.get("source_group_id", ""))
    if source_group and source_group != "unknown":
        return f"source:{source_group}"
    return f"video:{video_id}"


def _known_values(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    return {
        str(row[field])
        for row in rows
        if row.get(field) not in (None, "", "unknown")
    }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _divide(numerator: float, denominator: float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _truth_sort_key(row: Mapping[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row.get("video_id", "")),
        str(row.get("task_type", "")),
        float(row.get("start_time", 0.0)),
        str(row.get("label_id", "")),
    )


def _prediction_sort_key(row: Mapping[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row.get("video_id", "")),
        str(row.get("task_type", "")),
        float(row.get("start_time", 0.0)),
        str(row.get("prediction_id", "")),
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _threshold_curve_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fieldnames = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_report(
    result: EventEvaluationResult, metadata: Mapping[str, Any]
) -> str:
    metrics = result.metrics
    lines = [
        "# Fall-risk event evaluation",
        "",
        f"Protocol: `{result.protocol_version}` (`{result.protocol_status}`)",
        f"Evaluation config SHA-256: `{result.config_hash}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "precision",
        "recall",
        "f1",
        "event_pr_auc",
        "tp",
        "fp",
        "fn",
        "false_discovery_rate",
        "miss_rate",
        "fp_per_camera_hour",
        "mean_boundary_iou",
        "mean_detection_latency_sec",
        "mean_lead_time_sec",
    ):
        lines.append(f"| `{key}` | `{metrics.get(key)}` |")
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- `Precision = TP / (TP + FP)`; `Recall = TP / (TP + FN)`.",
            "- `false_discovery_rate = FP / (TP + FP)`; traditional FPR is only emitted when an explicit negative-observation denominator exists.",
            "- `lead_time = reference_event_start - alert_time`; positive values are early warnings.",
            "- Fall detection latency and near-fall onset detection latency use `alert_time - reference onset`; recovery is reported with recovery-specific metrics. A provisional input may fall back to prediction `onset_time`, and the fallback count is reported.",
            "- `fp_per_camera_hour` uses only manifest rows explicitly marked `continuous_monitoring_eligible=true`.",
            "- `event_pr_auc` uses step-wise average precision (`delta_recall * precision`), scans every distinct score threshold, and reruns the same one-to-one matcher.",
            "",
            "## Matching protocol",
            "",
            "```json",
            json.dumps(
                metrics["matching_protocol"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Interpretation",
            "",
            "This run uses a provisional development protocol unless the protocol status is explicitly frozen. Undefined denominators are reported as `null`; short event clips are not counted as continuous camera hours.",
            "",
            "## Reproducibility",
            "",
        ]
    )
    for key in sorted(metadata):
        lines.append(f"- `{key}`: `{metadata[key]}`")
    return "\n".join(lines) + "\n"
