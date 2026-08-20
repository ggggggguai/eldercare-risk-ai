from __future__ import annotations

from collections import Counter
from hashlib import sha256
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_evaluation import validate_subject_partitions


CASME2_CROSSWALK_SCHEMA_VERSION = "casme2_three_class_crosswalk_v1"
EVAL_ME_005_SPLIT_SCHEMA_VERSION = "eval_me_005_subject_protocols_v1"
LABEL_NAMES = {0: "negative", 1: "positive", 2: "surprise"}
EMOTION_TO_PROJECT_LABEL = {
    "disgust": 0,
    "repression": 0,
    "happiness": 1,
    "surprise": 2,
}


def label_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(record["label"]) for record in records)
    return {str(label): counts.get(label, 0) for label in range(3)}


def dataset_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["evaluation_dataset"]) for record in records)
    return dict(sorted(counts.items()))


def _subject_records(
    records: Sequence[Mapping[str, Any]], subject_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    selected = set(subject_ids)
    return [record for record in records if str(record["subject_id"]) in selected]


def _stable_tiebreak(salt: str, subjects: Sequence[str]) -> str:
    payload = (salt + "\0" + "\0".join(subjects)).encode()
    return sha256(payload).hexdigest()


def select_validation_subjects(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_subjects: Sequence[str],
    count: int,
    salt: str,
    require_dataset_coverage: bool = False,
) -> list[str]:
    candidates = sorted(set(candidate_subjects))
    if count < 1 or len(candidates) <= count:
        raise ValueError("Validation subject count leaves no training subject")
    training_side = _subject_records(records, candidates)
    all_datasets = {
        str(record["evaluation_dataset"]) for record in training_side
    }
    target_label = np.asarray(
        [int(label_counts(training_side)[str(label)]) for label in range(3)],
        dtype=np.float64,
    )
    target_label /= target_label.sum()
    target_size = len(training_side) * count / len(candidates)
    subject_stats: dict[str, tuple[np.ndarray, int, set[str]]] = {}
    for subject in candidates:
        subject_rows = _subject_records(records, [subject])
        subject_stats[subject] = (
            np.asarray(
                [
                    int(label_counts(subject_rows)[str(label)])
                    for label in range(3)
                ],
                dtype=np.float64,
            ),
            len(subject_rows),
            {str(record["evaluation_dataset"]) for record in subject_rows},
        )
    best: tuple[float, str, tuple[str, ...]] | None = None
    for combination in itertools.combinations(candidates, count):
        selected_label = sum(
            (subject_stats[subject][0] for subject in combination),
            start=np.zeros(3, dtype=np.float64),
        )
        if np.any(selected_label <= 0):
            continue
        selected_datasets = set().union(
            *(subject_stats[subject][2] for subject in combination)
        )
        if require_dataset_coverage and selected_datasets != all_datasets:
            continue
        selected_label /= selected_label.sum()
        selected_size = sum(subject_stats[subject][1] for subject in combination)
        distribution_error = float(np.abs(selected_label - target_label).sum())
        size_error = abs(selected_size - target_size) / max(target_size, 1.0)
        score = distribution_error + 0.15 * size_error
        candidate = (score, _stable_tiebreak(salt, combination), combination)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No training-side validation subjects satisfy the contract")
    return list(best[2])


def _fold(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> dict[str, Any]:
    partitions = {
        "train": _subject_records(records, train_subjects),
        "validation": _subject_records(records, validation_subjects),
        "test": _subject_records(records, test_subjects),
    }
    sample_ids = {
        name: sorted(str(record["sample_id"]) for record in rows)
        for name, rows in partitions.items()
    }
    contract = validate_subject_partitions(
        records,
        train_sample_ids=sample_ids["train"],
        validation_sample_ids=sample_ids["validation"],
        test_sample_ids=sample_ids["test"],
    )
    return {
        "fold_id": fold_id,
        "train_sample_ids": sample_ids["train"],
        "validation_sample_ids": sample_ids["validation"],
        "test_sample_ids": sample_ids["test"],
        "subjects": {
            "train": sorted(train_subjects),
            "validation": sorted(validation_subjects),
            "test": sorted(test_subjects),
        },
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "label_counts": {
            name: label_counts(rows) for name, rows in partitions.items()
        },
        "dataset_counts": {
            name: dataset_counts(rows) for name, rows in partitions.items()
        },
        "partition_contract": contract,
        "selection_authority": {
            "checkpoint": "training_side_validation_uf1_uar_harmonic_then_loss",
            "test_used_for_selection": False,
        },
    }


def build_loso_protocol(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
    validation_subject_count: int,
    require_validation_dataset_coverage: bool = False,
) -> dict[str, Any]:
    subjects = sorted({str(record["subject_id"]) for record in records})
    folds = []
    for test_subject in subjects:
        training_side = [subject for subject in subjects if subject != test_subject]
        validation = select_validation_subjects(
            records,
            candidate_subjects=training_side,
            count=validation_subject_count,
            salt=f"{protocol_id}:{test_subject}",
            require_dataset_coverage=require_validation_dataset_coverage,
        )
        train = [subject for subject in training_side if subject not in validation]
        folds.append(
            _fold(
                records,
                fold_id=f"loso_{test_subject}",
                train_subjects=train,
                validation_subjects=validation,
                test_subjects=[test_subject],
            )
        )
    return {
        "protocol_id": protocol_id,
        "kind": "strict_nested_loso",
        "fold_count": len(folds),
        "subject_count": len(subjects),
        "validation_subject_count": validation_subject_count,
        "folds": folds,
    }


def build_cross_dataset_protocol(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
    training_dataset: str,
    test_dataset: str,
    validation_subject_count: int = 3,
) -> dict[str, Any]:
    train_source = [
        record
        for record in records
        if record["evaluation_dataset"] == training_dataset
    ]
    train_subjects = sorted(
        {str(record["subject_id"]) for record in train_source}
    )
    test_subjects = sorted(
        {
            str(record["subject_id"])
            for record in records
            if record["evaluation_dataset"] == test_dataset
        }
    )
    validation = select_validation_subjects(
        train_source,
        candidate_subjects=train_subjects,
        count=validation_subject_count,
        salt=protocol_id,
    )
    train = [subject for subject in train_subjects if subject not in validation]
    fold = _fold(
        records,
        fold_id=protocol_id,
        train_subjects=train,
        validation_subjects=validation,
        test_subjects=test_subjects,
    )
    return {
        "protocol_id": protocol_id,
        "kind": "cross_dataset_holdout",
        "training_dataset": training_dataset,
        "test_dataset": test_dataset,
        "fold_count": 1,
        "validation_subject_count": validation_subject_count,
        "domain_gap_note": (
            "Camera, frame-rate, subject, induction, and annotation domains differ; "
            "cross-dataset metrics do not establish S10 transfer."
        ),
        "folds": [fold],
    }


def audit_protocols(
    records: Sequence[Mapping[str, Any]], protocols: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        issues.append({"code": "duplicate_sample_id"})
    expected_fold_counts = {
        "casme2_nested_loso": 24,
        "cross_smic_to_casme2": 1,
        "cross_casme2_to_smic": 1,
        "merged_nested_loso": 40,
    }
    for protocol_id, expected_count in expected_fold_counts.items():
        protocol = protocols.get(protocol_id)
        if protocol is None or protocol.get("fold_count") != expected_count:
            issues.append({"code": "protocol_fold_count", "protocol": protocol_id})
            continue
        test_coverage: list[str] = []
        for fold in protocol["folds"]:
            try:
                validate_subject_partitions(
                    records,
                    train_sample_ids=fold["train_sample_ids"],
                    validation_sample_ids=fold["validation_sample_ids"],
                    test_sample_ids=fold["test_sample_ids"],
                )
            except Exception as error:  # noqa: BLE001
                issues.append(
                    {
                        "code": "partition_contract",
                        "protocol": protocol_id,
                        "fold": fold.get("fold_id"),
                        "error": repr(error),
                    }
                )
            test_coverage.extend(fold["test_sample_ids"])
        expected_test = {
            sample_id
            for sample_id, record in by_id.items()
            if (
                protocol_id in {"merged_nested_loso"}
                or protocol_id == "casme2_nested_loso"
                and record["evaluation_dataset"] == "casme2"
                or protocol_id == "cross_smic_to_casme2"
                and record["evaluation_dataset"] == "casme2"
                or protocol_id == "cross_casme2_to_smic"
                and record["evaluation_dataset"] == "smic_hs"
            )
        }
        if set(test_coverage) != expected_test:
            issues.append({"code": "test_coverage", "protocol": protocol_id})
        if len(test_coverage) != len(expected_test):
            issues.append({"code": "test_multiplicity", "protocol": protocol_id})
    return {
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "record_count": len(records),
        "protocol_count": len(protocols),
    }
