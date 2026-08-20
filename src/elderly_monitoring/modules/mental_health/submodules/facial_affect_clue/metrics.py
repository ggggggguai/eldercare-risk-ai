from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


LABEL_NAMES = {0: "negative", 1: "positive", 2: "surprise"}
LABELS = (0, 1, 2)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if true.size == 0 or true.shape != predicted.shape:
        raise ValueError("Metrics require equally sized non-empty label arrays")
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=LABELS,
        zero_division=0,
    )
    macro_f1 = float(f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0))
    macro_recall = float(
        recall_score(true, predicted, labels=LABELS, average="macro", zero_division=0)
    )
    present_class_recall = recall[support > 0]
    balanced_accuracy = float(np.mean(present_class_recall))
    return {
        "sample_count": int(true.size),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": macro_f1,
        "uf1": macro_f1,
        "macro_recall": macro_recall,
        "uar": macro_recall,
        "balanced_accuracy": balanced_accuracy,
        "per_class": {
            LABEL_NAMES[label]: {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(true, predicted, labels=LABELS).tolist(),
    }


def misclassification_rows(
    *,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, subject_id, truth, prediction, probability in zip(
        sample_ids, subject_ids, y_true, y_pred, probabilities, strict=True
    ):
        if int(truth) == int(prediction):
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "true_label": int(truth),
                "true_label_name": LABEL_NAMES[int(truth)],
                "predicted_label": int(prediction),
                "predicted_label_name": LABEL_NAMES[int(prediction)],
                "confidence": float(max(probability)),
                "probabilities": {
                    LABEL_NAMES[label]: float(probability[label]) for label in LABELS
                },
            }
        )
    return rows
