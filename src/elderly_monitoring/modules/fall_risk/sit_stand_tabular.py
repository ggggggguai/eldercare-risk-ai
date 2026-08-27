from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


SCHEMA_VERSION = "sit-stand-tabular-checkpoint-v1"
MODEL_VERSION = "sit-stand-random-forest-v1"
SPLIT_SEED = "sitstand-selection-v1"


def split_validation_groups(
    group_ids: Sequence[Any], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids).astype(str)
    truth = np.asarray(labels, dtype=np.int64)
    if len(groups) != len(truth) or not len(groups):
        raise ValueError("sit-stand group IDs and labels must be non-empty and aligned")
    unique_groups = sorted(set(groups.tolist()))
    selection_groups = {
        group
        for group in unique_groups
        if int(
            hashlib.sha256(f"{SPLIT_SEED}|{group}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        % 2
        == 0
    }
    selection = np.asarray([group in selection_groups for group in groups])
    confirmation = ~selection
    for name, mask in (("selection", selection), ("confirmation", confirmation)):
        if set(truth[mask].tolist()) != {0, 1}:
            raise ValueError(f"sit-stand {name} partition lacks both classes")
    if set(groups[selection]) & set(groups[confirmation]):
        raise ValueError("sit-stand selection and confirmation groups overlap")
    return selection, confirmation


def aggregate_event_scores(
    indices: Sequence[int],
    scores: Sequence[float],
    *,
    labels: Sequence[int],
    event_ids: Sequence[Any],
    split_group_ids: Sequence[Any],
    datasets: Sequence[Any],
    action_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    selected_indices = np.asarray(indices, dtype=np.int64)
    selected_scores = np.asarray(scores, dtype=np.float64)
    if len(selected_indices) != len(selected_scores):
        raise ValueError("sit-stand indices and scores must have equal length")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, score in zip(selected_indices, selected_scores, strict=True):
        grouped[str(event_ids[index])].append((int(index), float(score)))
    rows: list[dict[str, Any]] = []
    for event_id, values in sorted(grouped.items()):
        source_indices = [item[0] for item in values]
        event_labels = {int(labels[index]) for index in source_indices}
        if len(event_labels) != 1:
            raise ValueError(f"sit-stand event {event_id} has inconsistent labels")
        first = source_indices[0]
        rows.append(
            {
                "event_id": event_id,
                "label": event_labels.pop(),
                "score": float(np.mean([item[1] for item in values])),
                "window_count": len(values),
                "split_group_id": str(split_group_ids[first]),
                "dataset": str(datasets[first]),
                "action_id": str(action_ids[first]),
            }
        )
    return rows


def select_f1_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    if not rows:
        raise ValueError("sit-stand threshold selection rows are empty")
    scores = sorted({float(row["score"]) for row in rows})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    ranked = []
    for threshold in candidates:
        metrics = binary_event_metrics(rows, threshold)
        ranked.append(
            (
                metrics["f1"],
                metrics["recall"],
                metrics["precision"],
                -abs(threshold - 0.5),
                -threshold,
                threshold,
                metrics,
            )
        )
    best = max(ranked)
    return float(best[5]), dict(best[6])


def binary_event_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if len(truth) == 0 or set(truth.tolist()) != {0, 1}:
        raise ValueError("sit-stand event metrics require both classes")
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0
    )
    return {
        "event_count": len(rows),
        "positive_count": int(np.sum(truth == 1)),
        "negative_count": int(np.sum(truth == 0)),
        "true_positive": int(np.sum((truth == 1) & (predictions == 1))),
        "false_positive": int(np.sum((truth == 0) & (predictions == 1))),
        "false_negative": int(np.sum((truth == 1) & (predictions == 0))),
        "true_negative": int(np.sum((truth == 0) & (predictions == 0))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "threshold": float(threshold),
    }
