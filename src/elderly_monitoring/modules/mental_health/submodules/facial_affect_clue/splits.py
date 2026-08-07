from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_SCHEMA_VERSION = "smic_subject_loso_v1"


def _label_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(record["label"]) for record in records)
    return {str(label): counts[label] for label in (0, 1, 2)}


def _select_validation_subjects(
    subject_records: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
    count: int,
) -> tuple[str, ...]:
    remaining_records = [
        record for subject in candidates for record in subject_records[subject]
    ]
    global_counts = Counter(int(record["label"]) for record in remaining_records)
    global_total = sum(global_counts.values())
    target_size = global_total * 0.20

    def score(subjects: tuple[str, ...]) -> tuple[float, float, float, tuple[str, ...]]:
        validation = [
            record for subject in subjects for record in subject_records[subject]
        ]
        validation_counts = Counter(int(record["label"]) for record in validation)
        validation_total = len(validation)
        training_counts = global_counts - validation_counts
        missing_penalty = 1000.0 * (
            sum(validation_counts[label] == 0 for label in (0, 1, 2))
            + sum(training_counts[label] == 0 for label in (0, 1, 2))
        )
        size_penalty = abs(validation_total - target_size) / max(target_size, 1.0)
        distribution_penalty = sum(
            abs(
                validation_counts[label] / max(validation_total, 1)
                - global_counts[label] / max(global_total, 1)
            )
            for label in (0, 1, 2)
        )
        return missing_penalty, distribution_penalty, size_penalty, subjects

    options = list(combinations(sorted(candidates), count))
    if not options:
        raise ValueError("No validation subject combinations available")
    return min(options, key=score)


def build_smic_loso_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_subject_count: int = 3,
) -> dict[str, Any]:
    subject_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("source_dataset") != "smic_hs":
            raise ValueError("SMIC LOSO split received a non-SMIC record")
        subject_records[str(record["subject_id"])].append(record)
    subjects = sorted(subject_records)
    if len(subjects) < validation_subject_count + 2:
        raise ValueError("Not enough subjects for train/validation/test isolation")

    folds: list[dict[str, Any]] = []
    for test_subject in subjects:
        candidates = [subject for subject in subjects if subject != test_subject]
        validation_subjects = _select_validation_subjects(
            subject_records, candidates, validation_subject_count
        )
        training_subjects = tuple(
            subject for subject in candidates if subject not in validation_subjects
        )

        def ids(selected_subjects: Sequence[str]) -> list[str]:
            return sorted(
                str(record["sample_id"])
                for subject in selected_subjects
                for record in subject_records[subject]
            )

        train_ids = ids(training_subjects)
        validation_ids = ids(validation_subjects)
        test_ids = ids((test_subject,))
        train_records = [
            record for subject in training_subjects for record in subject_records[subject]
        ]
        validation_records = [
            record for subject in validation_subjects for record in subject_records[subject]
        ]
        test_records = list(subject_records[test_subject])
        folds.append(
            {
                "fold_id": f"loso_{test_subject}",
                "test_subject": test_subject,
                "train_subjects": list(training_subjects),
                "validation_subjects": list(validation_subjects),
                "test_subjects": [test_subject],
                "train_sample_ids": train_ids,
                "validation_sample_ids": validation_ids,
                "test_sample_ids": test_ids,
                "counts": {
                    "train": len(train_ids),
                    "validation": len(validation_ids),
                    "test": len(test_ids),
                },
                "label_counts": {
                    "train": _label_counts(train_records),
                    "validation": _label_counts(validation_records),
                    "test": _label_counts(test_records),
                },
                "selection_authority": {
                    "checkpoint": "validation_macro_f1_then_validation_loss",
                    "test_used_for_selection": False,
                },
            }
        )
    result = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task_id": "MODEL-ME-002",
        "source_dataset": "smic_hs",
        "sample_count": len(records),
        "subject_count": len(subjects),
        "subjects": subjects,
        "validation_policy": {
            "type": "subject_level_greedy_distribution_match",
            "subject_count": validation_subject_count,
            "target_fraction": 0.20,
        },
        "folds": folds,
    }
    validate_smic_loso_splits(result)
    return result


def validate_smic_loso_splits(split_manifest: Mapping[str, Any]) -> None:
    folds = split_manifest["folds"]
    subjects = set(split_manifest["subjects"])
    test_subjects: list[str] = []
    for fold in folds:
        train_subjects = set(fold["train_subjects"])
        validation_subjects = set(fold["validation_subjects"])
        test = set(fold["test_subjects"])
        if train_subjects & validation_subjects or train_subjects & test or validation_subjects & test:
            raise ValueError(f"Subject leakage in {fold['fold_id']}")
        if train_subjects | validation_subjects | test != subjects:
            raise ValueError(f"Subject coverage mismatch in {fold['fold_id']}")
        train_ids = set(fold["train_sample_ids"])
        validation_ids = set(fold["validation_sample_ids"])
        test_ids = set(fold["test_sample_ids"])
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError(f"Sample leakage in {fold['fold_id']}")
        if fold["selection_authority"]["test_used_for_selection"]:
            raise ValueError(f"Test selection leakage in {fold['fold_id']}")
        test_subjects.extend(fold["test_subjects"])
    if sorted(test_subjects) != sorted(subjects):
        raise ValueError("Each subject must be the test subject exactly once")


def write_split_manifest(split_manifest: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return sha256(output_path.read_bytes()).hexdigest()
