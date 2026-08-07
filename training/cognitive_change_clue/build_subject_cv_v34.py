"""Build and audit the frozen CogPic official-Train-only V3.4 folds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from .common import (
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        workspace_relative,
    )


SEED = 20260805
VALIDATION_SEED_BASE = 20260905
LABEL_ORDER = ("HC", "MCI", "AD")
EXPECTED_SKLEARN_VERSION = "1.9.0"
DEFAULT_OUTPUT_ROOT = (
    ALGORITHM_ROOT / "data" / "processed" / "cognitive_change_clue" / "v3.4.0"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_ROOT / "cogpic_subject_cv_v34.json"
DEFAULT_AUDIT_PATH = DEFAULT_OUTPUT_ROOT / "cogpic_subject_cv_v34_audit.json"


class CognitiveV34SplitError(RuntimeError):
    pass


def _subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "diagnosis_label", "official_split"}
    missing = required.difference(frame.columns)
    if missing:
        raise CognitiveV34SplitError(f"manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise CognitiveV34SplitError("manifest contains duplicate sample_id values")
    subjects = frame[["subject_id", "diagnosis_label", "official_split"]].drop_duplicates()
    if subjects.groupby("subject_id")["diagnosis_label"].nunique().max() != 1:
        raise CognitiveV34SplitError("a subject has conflicting diagnosis labels")
    if subjects.groupby("subject_id")["official_split"].nunique().max() != 1:
        raise CognitiveV34SplitError("a subject crosses the official Train/Test boundary")
    return subjects.sort_values("subject_id", kind="stable").reset_index(drop=True)


def _labels_for(subject_ids: Sequence[str], labels: Mapping[str, str]) -> list[str]:
    try:
        return [labels[subject_id] for subject_id in subject_ids]
    except KeyError as exc:
        raise CognitiveV34SplitError(f"split subject is absent from manifest: {exc.args[0]}") from exc


def _counts(subject_ids: Sequence[str], labels: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(_labels_for(subject_ids, labels))
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}


def _split_once(
    subject_ids: Sequence[str],
    labels: Mapping[str, str],
    *,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    ordered = sorted(str(value) for value in subject_ids)
    y = _labels_for(ordered, labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=float(test_size), random_state=int(random_state)
    )
    keep_indices, held_indices = next(splitter.split(ordered, y))
    return (
        sorted(ordered[int(index)] for index in keep_indices),
        sorted(ordered[int(index)] for index in held_indices),
    )


def build_subject_cv(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    source_split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    strict_counts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sklearn_version != EXPECTED_SKLEARN_VERSION:
        raise CognitiveV34SplitError(
            f"scikit-learn {EXPECTED_SKLEARN_VERSION} is required, got {sklearn_version}"
        )
    manifest_path = Path(manifest_path)
    source_split_path = Path(source_split_path)
    output_path = Path(output_path)
    audit_path = Path(audit_path)
    frame = pd.read_parquet(manifest_path, engine="pyarrow")
    subjects = _subject_table(frame)
    source = json.loads(source_split_path.read_text(encoding="utf-8"))
    source_splits = source.get("splits", {})
    official_train = sorted(
        str(value)
        for value in list(source_splits.get("train", ()))
        + list(source_splits.get("validation", ()))
    )
    official_test = sorted(str(value) for value in source_splits.get("test", ()))
    train_set, test_set = set(official_train), set(official_test)
    if len(train_set) != len(official_train) or len(test_set) != len(official_test):
        raise CognitiveV34SplitError("source split contains duplicate subject IDs")
    if train_set & test_set:
        raise CognitiveV34SplitError("source split leaks official Test into official Train")
    if strict_counts and (len(official_train) != 459 or len(official_test) != 115):
        raise CognitiveV34SplitError(
            f"unexpected source counts train={len(official_train)} test={len(official_test)}"
        )

    labels = {
        str(row.subject_id): str(row.diagnosis_label)
        for row in subjects.itertuples(index=False)
    }
    official = {
        str(row.subject_id): str(row.official_split)
        for row in subjects.itertuples(index=False)
    }
    all_manifest_subjects = set(labels)
    if train_set | test_set != all_manifest_subjects:
        raise CognitiveV34SplitError("source split does not exactly cover manifest subjects")
    if any(official[subject_id] != "train" for subject_id in official_train):
        raise CognitiveV34SplitError("official Test subject reached the V3.4 development pool")
    if any(official[subject_id] != "test" for subject_id in official_test):
        raise CognitiveV34SplitError("source Test list disagrees with manifest metadata")
    if set(_labels_for(official_train, labels)) != set(LABEL_ORDER):
        raise CognitiveV34SplitError("official Train is missing one or more diagnosis classes")

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_labels = _labels_for(official_train, labels)
    folds: list[dict[str, Any]] = []
    for outer_fold, (development_indices, evaluation_indices) in enumerate(
        outer.split(official_train, outer_labels)
    ):
        development = sorted(official_train[int(index)] for index in development_indices)
        outer_evaluation = sorted(
            official_train[int(index)] for index in evaluation_indices
        )
        inner_selection, inner_calibration = _split_once(
            development,
            labels,
            test_size=0.15,
            random_state=SEED + outer_fold,
        )
        inner_train, inner_validation = _split_once(
            inner_selection,
            labels,
            test_size=0.15,
            random_state=VALIDATION_SEED_BASE + outer_fold,
        )
        roles = {
            "inner_train": inner_train,
            "inner_validation": inner_validation,
            "inner_calibration": inner_calibration,
            "outer_evaluation": outer_evaluation,
        }
        folds.append(
            {
                "outer_fold": outer_fold,
                "inner_calibration_seed": SEED + outer_fold,
                "inner_validation_seed": VALIDATION_SEED_BASE + outer_fold,
                **{f"{name}_subjects": values for name, values in roles.items()},
                "counts": {name: len(values) for name, values in roles.items()},
                "per_label_counts": {
                    name: _counts(values, labels) for name, values in roles.items()
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "cogpic_subject_cv_v34",
        "seed": SEED,
        "sklearn_version": sklearn_version,
        "source_split_path": workspace_relative(source_split_path),
        "source_split_sha256": sha256_file(source_split_path),
        "source_manifest_path": workspace_relative(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_scope": "official_train_only",
        "subject_sort": "unicode_codepoint_ascending",
        "label_order": list(LABEL_ORDER),
        "official_train_subject_count": len(official_train),
        "official_test_subject_count": len(official_test),
        "official_test_overlap_count": 0,
        "folds": folds,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_json(output_path, payload)
    audit = audit_subject_cv(
        payload,
        frame=frame,
        official_train_subjects=official_train,
        official_test_subjects=official_test,
        split_file_sha256=sha256_file(output_path),
    )
    atomic_write_json(audit_path, audit)
    return payload, audit


def audit_subject_cv(
    payload: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    official_train_subjects: Sequence[str],
    official_test_subjects: Sequence[str],
    split_file_sha256: str | None = None,
) -> dict[str, Any]:
    official_train = set(str(value) for value in official_train_subjects)
    official_test = set(str(value) for value in official_test_subjects)
    evaluation_counts: Counter[str] = Counter()
    fold_audits: list[dict[str, Any]] = []
    violations: list[str] = []
    for fold in payload.get("folds", ()):
        outer_fold = int(fold["outer_fold"])
        roles = {
            name: set(str(value) for value in fold[f"{name}_subjects"])
            for name in (
                "inner_train",
                "inner_validation",
                "inner_calibration",
                "outer_evaluation",
            )
        }
        role_names = list(roles)
        overlap: dict[str, int] = {}
        for index, left in enumerate(role_names):
            for right in role_names[index + 1 :]:
                overlap[f"{left}__{right}"] = len(roles[left] & roles[right])
        coverage = set().union(*roles.values())
        evaluation_counts.update(roles["outer_evaluation"])
        test_overlap = len(coverage & official_test)
        outside_train = len(coverage - official_train)
        if any(overlap.values()):
            violations.append(f"fold {outer_fold} contains role overlap")
        if coverage != official_train:
            violations.append(f"fold {outer_fold} does not cover official Train exactly")
        if test_overlap:
            violations.append(f"fold {outer_fold} contains official Test subjects")
        if outside_train:
            violations.append(f"fold {outer_fold} contains unknown subjects")
        fold_audits.append(
            {
                "outer_fold": outer_fold,
                "role_pair_overlap_counts": overlap,
                "official_train_coverage_count": len(coverage),
                "official_test_overlap_count": test_overlap,
                "outside_official_train_count": outside_train,
            }
        )

    missing_evaluation = sorted(official_train - set(evaluation_counts))
    duplicate_evaluation = sorted(
        subject_id for subject_id, count in evaluation_counts.items() if count != 1
    )
    if missing_evaluation:
        violations.append("one or more official Train subjects never enter outer evaluation")
    if duplicate_evaluation:
        violations.append("one or more subjects enter outer evaluation more than once")

    sample_subjects = frame[["sample_id", "subject_id"]].copy()
    task_counts = sample_subjects.groupby("subject_id")["sample_id"].nunique().to_dict()
    unexpected_task_counts = {
        str(subject_id): int(count)
        for subject_id, count in task_counts.items()
        if str(subject_id) in official_train and int(count) != 3
    }
    if unexpected_task_counts:
        violations.append("one or more official Train subjects do not have exactly three tasks")

    audit = {
        "schema_version": "cogpic_subject_cv_v34_audit",
        "status": "passed" if not violations else "failed",
        "split_file_sha256": split_file_sha256,
        "official_train_subject_count": len(official_train),
        "official_test_subject_count": len(official_test),
        "official_test_overlap_count": len(official_train & official_test),
        "outer_evaluation_unique_subject_count": len(evaluation_counts),
        "outer_evaluation_missing_subjects": missing_evaluation,
        "outer_evaluation_duplicate_subjects": duplicate_evaluation,
        "expected_tasks_per_subject": 3,
        "unexpected_task_counts": unexpected_task_counts,
        "task_cross_fold_count": 0 if not duplicate_evaluation else len(duplicate_evaluation),
        "folds": fold_audits,
        "violations": violations,
    }
    if violations:
        raise CognitiveV34SplitError("; ".join(violations))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, audit = build_subject_cv(
        manifest_path=args.manifest,
        source_split_path=args.source_split,
        output_path=args.output,
        audit_path=args.audit,
        strict_counts=not args.allow_count_mismatch,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "folds": len(payload["folds"]),
                "official_train_subjects": payload["official_train_subject_count"],
                "official_test_overlap": payload["official_test_overlap_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CognitiveV34SplitError",
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "LABEL_ORDER",
    "SEED",
    "audit_subject_cv",
    "build_subject_cv",
]

