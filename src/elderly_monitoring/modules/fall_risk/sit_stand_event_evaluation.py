from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIRECTIONS = ("sit_to_stand", "stand_to_sit")


def evaluate_sit_stand_events(
    ground_truth: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _validate_config(config)
    truth_rows = [dict(row) for row in ground_truth]
    prediction_rows = [dict(row) for row in predictions]
    _validate_unique(truth_rows, "label_id", "ground truth")
    _validate_unique(prediction_rows, "prediction_id", "prediction")
    events = [
        row
        for row in truth_rows
        if row.get("interval_type") == "event" and row.get("eligibility") != "ignore"
    ]
    backgrounds = [
        row
        for row in truth_rows
        if row.get("interval_type") == "explicit_background"
        and row.get("eligibility") != "ignore"
    ]
    ignores = [
        row
        for row in truth_rows
        if row.get("interval_type") == "ignore" or row.get("eligibility") == "ignore"
    ]
    for row in events:
        _validate_interval(row, "label_id")
        if row.get("transition_type") not in DIRECTIONS:
            raise ValueError(f"invalid ground-truth transition: {row.get('label_id')}")
    for row in backgrounds + ignores:
        _validate_interval(row, "label_id")

    eligible_predictions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    rejected_count = 0
    reviewed_intervals = events + backgrounds
    for row in prediction_rows:
        _validate_prediction(row)
        if row.get("quality_state") not in {"valid", "accepted"}:
            rejected_count += 1
            excluded.append({**row, "exclusion_reason": "quality_rejected"})
            continue
        if any(_overlap(row, ignored) > 0 for ignored in ignores):
            excluded.append({**row, "exclusion_reason": "overlaps_ignore_interval"})
            continue
        if not any(_overlap(row, reviewed) > 0 for reviewed in reviewed_intervals):
            excluded.append({**row, "exclusion_reason": "outside_reviewed_intervals"})
            continue
        eligible_predictions.append(row)

    threshold = float(protocol["score_threshold"])
    selected = [row for row in eligible_predictions if float(row["score"]) >= threshold]
    match_result = _match(events, selected, protocol, require_direction=True)
    matches = _match_rows(events, selected, match_result[0])
    matched_truth = {left for left, _ in match_result[0]}
    matched_predictions = {right for _, right in match_result[0]}
    false_negatives = [
        {**row, "failure_reason": "unmatched_ground_truth"}
        for index, row in enumerate(events)
        if index not in matched_truth
    ]
    false_positives = [
        {**row, "failure_reason": "unmatched_prediction"}
        for index, row in enumerate(selected)
        if index not in matched_predictions
    ]
    threshold_curve = _threshold_curve(events, eligible_predictions, protocol)
    direction_metric = _direction_macro_f1(events, selected, protocol)
    onset_errors = [float(row["onset_error_sec"]) for row in matches]
    offset_errors = [float(row["offset_error_sec"]) for row in matches]
    ious = [float(row["boundary_iou"]) for row in matches]
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    event_f1 = _f1(precision, recall)
    background_duration_sec = sum(_duration(row) for row in backgrounds)
    background_fp = [
        row
        for row in false_positives
        if any(_overlap(row, background) > 0 for background in backgrounds)
    ]
    fp_per_hour = (
        {
            "status": "available",
            "value": round(len(background_fp) / (background_duration_sec / 3600.0), 6),
            "false_positive_count": len(background_fp),
            "background_camera_hours": round(background_duration_sec / 3600.0, 6),
        }
        if background_duration_sec > 0
        else {
            "status": "unavailable",
            "value": None,
            "reason": "no_explicit_background_duration",
        }
    )
    metrics = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": event_f1,
        "pr_auc": _average_precision(threshold_curve),
        "direction_macro_f1": direction_metric,
        "onset_error_sec": _summary(onset_errors),
        "offset_error_sec": _summary(offset_errors),
        "boundary_iou": _summary(ious),
        "fp_per_hour": fp_per_hour,
        "hard_negative_false_positive_rate": _hard_negative_rates(
            backgrounds, background_fp
        ),
        "quality_rejection_rate": _ratio(rejected_count, len(prediction_rows)),
        "effective_prediction_coverage": _ratio(
            len(eligible_predictions), len(prediction_rows)
        ),
        "stratified": _stratified(events, selected, protocol),
    }
    return {
        "schema_version": "sit-stand-event-evaluation-v1",
        "protocol_version": protocol["protocol_version"],
        "protocol_status": protocol["protocol_status"],
        "metrics": metrics,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "excluded": excluded,
        "threshold_curve": threshold_curve,
        "confidence_intervals_95": _bootstrap(
            events, backgrounds, selected, protocol
        ),
        "test_access": {"test_evaluated": False},
    }


def write_sit_stand_evaluation_bundle(
    result: Mapping[str, Any], output_dir: Path
) -> None:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"sit-stand evaluation output exists: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "metrics.json", dict(result))
    for filename, key in (
        ("matches.jsonl", "matches"),
        ("false_positives.jsonl", "false_positives"),
        ("false_negatives.jsonl", "false_negatives"),
        ("excluded.jsonl", "excluded"),
    ):
        _write_jsonl(destination / filename, result.get(key, []))
    _write_jsonl(destination / "threshold_curve.jsonl", result.get("threshold_curve", []))


def _match(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    require_direction: bool,
) -> tuple[list[tuple[int, int]], list[tuple[float, float, str, str, int, int]]]:
    candidates: list[tuple[float, float, str, str, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            if str(truth.get("video_id")) != str(prediction.get("video_id")):
                continue
            if require_direction and truth.get("transition_type") != prediction.get(
                "transition_type"
            ):
                continue
            iou = _iou(truth, prediction)
            onset_error = abs(
                float(truth["onset_time"]) - float(prediction["onset_time"])
            )
            offset_error = abs(
                float(truth["offset_time"]) - float(prediction["offset_time"])
            )
            compatible = iou >= float(config["iou_threshold"]) or (
                onset_error <= float(config["onset_tolerance_sec"])
                and offset_error <= float(config["offset_tolerance_sec"])
            )
            if compatible:
                candidates.append(
                    (
                        -iou,
                        onset_error + offset_error,
                        str(truth.get("label_id")),
                        str(prediction.get("prediction_id")),
                        truth_index,
                        prediction_index,
                    )
                )
    candidates.sort()
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for candidate in candidates:
        truth_index, prediction_index = candidate[-2:]
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        pairs.append((truth_index, prediction_index))
    pairs.sort()
    return pairs, candidates


def _match_rows(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows = []
    for truth_index, prediction_index in pairs:
        truth = truths[truth_index]
        prediction = predictions[prediction_index]
        rows.append(
            {
                "label_id": truth["label_id"],
                "prediction_id": prediction["prediction_id"],
                "video_id": truth["video_id"],
                "transition_type": truth["transition_type"],
                "score": float(prediction["score"]),
                "onset_error_sec": round(
                    abs(float(truth["onset_time"]) - float(prediction["onset_time"])),
                    6,
                ),
                "offset_error_sec": round(
                    abs(float(truth["offset_time"]) - float(prediction["offset_time"])),
                    6,
                ),
                "boundary_iou": round(_iou(truth, prediction), 6),
            }
        )
    return rows


def _threshold_curve(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    thresholds = {
        round(float(config["score_threshold"]), 9),
        *(round(float(row["score"]), 9) for row in predictions),
    }
    rows = []
    for threshold in sorted(thresholds, reverse=True):
        selected = [row for row in predictions if float(row["score"]) >= threshold]
        pairs, _ = _match(truths, selected, config, require_direction=True)
        tp = len(pairs)
        fp = len(selected) - tp
        fn = len(truths) - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        rows.append(
            {
                "threshold": threshold,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )
    return rows


def _direction_macro_f1(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    pairs, _ = _match(truths, predictions, config, require_direction=False)
    matched_truth = {left for left, _ in pairs}
    matched_prediction = {right for _, right in pairs}
    active_directions = [
        direction
        for direction in DIRECTIONS
        if any(row.get("transition_type") == direction for row in truths)
        or any(row.get("transition_type") == direction for row in predictions)
    ]
    scores = []
    for direction in active_directions:
        tp = sum(
            truths[left].get("transition_type") == direction
            and predictions[right].get("transition_type") == direction
            for left, right in pairs
        )
        fp = sum(
            predictions[right].get("transition_type") == direction
            and truths[left].get("transition_type") != direction
            for left, right in pairs
        ) + sum(
            prediction.get("transition_type") == direction
            for index, prediction in enumerate(predictions)
            if index not in matched_prediction
        )
        fn = sum(
            truths[left].get("transition_type") == direction
            and predictions[right].get("transition_type") != direction
            for left, right in pairs
        ) + sum(
            truth.get("transition_type") == direction
            for index, truth in enumerate(truths)
            if index not in matched_truth
        )
        scores.append(_f1(_ratio(tp, tp + fp), _ratio(tp, tp + fn)))
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def _hard_negative_rates(
    backgrounds: Sequence[Mapping[str, Any]],
    false_positives: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    hit_ids: defaultdict[str, set[str]] = defaultdict(set)
    for row in backgrounds:
        category = row.get("hard_negative_type")
        if not isinstance(category, str) or not category:
            continue
        totals[category] += 1
        if any(_overlap(row, prediction) > 0 for prediction in false_positives):
            hit_ids[category].add(str(row.get("label_id")))
    return {
        category: round(len(hit_ids[category]) / total, 6)
        for category, total in sorted(totals.items())
    }


def _stratified(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        "dataset",
        "source_group_id",
        "scene_region",
        "transition_type",
    ):
        values = sorted({str(row.get(field, "unknown")) for row in truths})
        field_result: dict[str, Any] = {}
        for value in values:
            subset = [row for row in truths if str(row.get(field, "unknown")) == value]
            videos = {str(row.get("video_id")) for row in subset}
            subset_predictions = [
                row for row in predictions if str(row.get("video_id")) in videos
            ]
            pairs, _ = _match(subset, subset_predictions, config, require_direction=True)
            tp = len(pairs)
            precision = _ratio(tp, len(subset_predictions))
            recall = _ratio(tp, len(subset))
            field_result[value] = {
                "event_count": len(subset),
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        output[field] = field_result
    return output


def _bootstrap(
    events: Sequence[Mapping[str, Any]],
    backgrounds: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reviewed = [*events, *backgrounds]
    groups = sorted(
        {str(row.get("split_group_id", row.get("video_id"))) for row in reviewed}
    )
    iterations = int(config["bootstrap_iterations"])
    if len(groups) < 2 or iterations <= 0:
        return {
            "status": "unavailable",
            "reason": "fewer_than_two_independent_groups",
            "group_count": len(groups),
        }
    by_group = {
        group: [
            row
            for row in reviewed
            if str(row.get("split_group_id", row.get("video_id"))) == group
        ]
        for group in groups
    }
    rng = random.Random(int(config["bootstrap_seed"]))
    scores = []
    for _ in range(iterations):
        sampled_groups = [rng.choice(groups) for _ in groups]
        sampled_truths: list[dict[str, Any]] = []
        sampled_predictions: list[dict[str, Any]] = []
        for replicate, group in enumerate(sampled_groups):
            group_reviewed = by_group[group]
            videos = {str(row.get("video_id")) for row in group_reviewed}
            for row in group_reviewed:
                if row.get("interval_type") != "event":
                    continue
                sampled_truths.append(
                    {
                        **row,
                        "label_id": f"{replicate}:{row['label_id']}",
                        "video_id": f"{replicate}:{row['video_id']}",
                    }
                )
            for row in predictions:
                if str(row.get("video_id")) in videos:
                    sampled_predictions.append(
                        {
                            **row,
                            "prediction_id": f"{replicate}:{row['prediction_id']}",
                            "video_id": f"{replicate}:{row['video_id']}",
                        }
                    )
        pairs, _ = _match(
            sampled_truths, sampled_predictions, config, require_direction=True
        )
        tp = len(pairs)
        precision = _ratio(tp, len(sampled_predictions))
        recall = _ratio(tp, len(sampled_truths))
        scores.append(_f1(precision, recall))
    scores.sort()
    return {
        "status": "available",
        "event_f1": {
            "lower": _percentile(scores, 0.025),
            "upper": _percentile(scores, 0.975),
        },
        "group_count": len(groups),
        "iterations": iterations,
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "protocol_version",
        "protocol_status",
        "score_threshold",
        "iou_threshold",
        "onset_tolerance_sec",
        "offset_tolerance_sec",
        "bootstrap_iterations",
        "bootstrap_seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"sit-stand evaluation config missing: {', '.join(missing)}")
    result = dict(config)
    for field in ("score_threshold", "iou_threshold"):
        value = float(result[field])
        if not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"config.{field} must be within [0, 1]")
        result[field] = value
    for field in ("onset_tolerance_sec", "offset_tolerance_sec"):
        value = float(result[field])
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"config.{field} must be finite and non-negative")
        result[field] = value
    result["bootstrap_iterations"] = int(result["bootstrap_iterations"])
    result["bootstrap_seed"] = int(result["bootstrap_seed"])
    return result


def _validate_prediction(row: Mapping[str, Any]) -> None:
    _validate_interval(row, "prediction_id")
    if row.get("transition_type") not in DIRECTIONS:
        raise ValueError(f"invalid prediction transition: {row.get('prediction_id')}")
    score = float(row.get("score"))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"invalid prediction score: {row.get('prediction_id')}")


def _validate_interval(row: Mapping[str, Any], id_field: str) -> None:
    if not isinstance(row.get(id_field), str) or not row.get(id_field):
        raise ValueError(f"missing {id_field}")
    if not isinstance(row.get("video_id"), str) or not row.get("video_id"):
        raise ValueError(f"missing video_id for {row.get(id_field)}")
    onset = float(row.get("onset_time"))
    offset = float(row.get("offset_time"))
    if not math.isfinite(onset) or not math.isfinite(offset) or onset < 0 or offset <= onset:
        raise ValueError(f"invalid interval for {row.get(id_field)}")


def _validate_unique(rows: Sequence[Mapping[str, Any]], field: str, name: str) -> None:
    values = [row.get(field) for row in rows]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} requires non-empty {field}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name} {field}")


def _duration(row: Mapping[str, Any]) -> float:
    return float(row["offset_time"]) - float(row["onset_time"])


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if str(left.get("video_id")) != str(right.get("video_id")):
        return 0.0
    return max(
        0.0,
        min(float(left["offset_time"]), float(right["offset_time"]))
        - max(float(left["onset_time"]), float(right["onset_time"])),
    )


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = _overlap(left, right)
    union = _duration(left) + _duration(right) - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None}
    ordered = sorted(values)
    return {
        "median": round(float(statistics.median(ordered)), 6),
        "p90": _percentile(ordered, 0.9),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(float(values[lower]), 6)
    fraction = position - lower
    return round(float(values[lower] * (1 - fraction) + values[upper] * fraction), 6)


def _average_precision(curve: Sequence[Mapping[str, Any]]) -> float:
    points = sorted(
        {(float(row["recall"]), float(row["precision"])) for row in curve}
    )
    area = 0.0
    previous_recall = 0.0
    for recall, precision in points:
        if recall > previous_recall:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return round(area, 6)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
