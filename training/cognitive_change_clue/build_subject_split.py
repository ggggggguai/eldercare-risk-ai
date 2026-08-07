"""Generate and freeze the participant-level CogPic V3.3 split."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        atomic_write_parquet,
        canonical_json_bytes,
        read_manifest,
        sha256_bytes,
        sha256_file,
    )
except ImportError:
    from common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        atomic_write_parquet,
        canonical_json_bytes,
        read_manifest,
        sha256_bytes,
        sha256_file,
    )


SEED = 20260802
LABEL_ORDER = ("HC", "MCI", "AD")
VALIDATION_TOTAL = 92


class SplitBuildError(RuntimeError):
    pass


def build_subject_split(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_path: str | Path = DEFAULT_SPLIT_PATH,
    strict_counts: bool = True,
    validation_total: int = VALIDATION_TOTAL,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    frame = read_manifest(manifest_path)
    required = {"subject_id", "diagnosis_label", "official_split", "sample_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise SplitBuildError(f"manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise SplitBuildError("manifest contains duplicate sample_id values")
    subject_rows = frame[["subject_id", "diagnosis_label", "official_split"]].drop_duplicates()
    label_counts = subject_rows.groupby("subject_id")["diagnosis_label"].nunique()
    if (label_counts != 1).any():
        raise SplitBuildError("one or more subjects have conflicting diagnosis labels")
    split_counts = subject_rows.groupby("subject_id")["official_split"].nunique()
    if (split_counts != 1).any():
        raise SplitBuildError("one or more subjects cross official Train/Test")
    subject_rows = subject_rows.sort_values("subject_id", kind="stable")
    official_train = subject_rows[subject_rows["official_split"] == "train"]
    official_test = subject_rows[subject_rows["official_split"] == "test"]
    if strict_counts and (len(official_train) != 459 or len(official_test) != 115):
        raise SplitBuildError(
            f"unexpected official counts train={len(official_train)} test={len(official_test)}"
        )
    validation_total = int(validation_total)
    if validation_total <= 0 or validation_total >= len(official_train):
        raise SplitBuildError("validation_total must leave at least one Train subject")

    class_subjects: dict[str, list[str]] = {}
    for label in LABEL_ORDER:
        class_subjects[label] = sorted(
            official_train.loc[
                official_train["diagnosis_label"] == label, "subject_id"
            ].astype(str)
        )
        if not class_subjects[label]:
            raise SplitBuildError(f"official Train has no {label} subjects")
    quotas = _validation_quotas(class_subjects, validation_total, len(official_train))
    generator = np.random.Generator(np.random.PCG64(SEED))
    train_subjects: list[str] = []
    validation_subjects: list[str] = []
    per_label: dict[str, dict[str, int]] = {}
    for label in LABEL_ORDER:
        values = np.asarray(class_subjects[label], dtype=object)
        generator.shuffle(values)
        quota = quotas[label]
        validation_values = sorted(str(value) for value in values[:quota])
        train_values = sorted(str(value) for value in values[quota:])
        validation_subjects.extend(validation_values)
        train_subjects.extend(train_values)
        per_label[label] = {
            "official_train": len(values),
            "train": len(train_values),
            "validation": len(validation_values),
            "test": int(
                (official_test["diagnosis_label"] == label).sum()
            ),
        }
    test_subjects = sorted(official_test["subject_id"].astype(str))
    splits = {
        "train": sorted(train_subjects),
        "validation": sorted(validation_subjects),
        "test": test_subjects,
    }
    _validate_split(splits, subject_rows, validation_total)
    payload: dict[str, Any] = {
        "schema_version": "cogpic_subject_split_v33",
        "algorithm": "stratified_largest_remainder_then_pcg64",
        "seed": SEED,
        "label_order": list(LABEL_ORDER),
        "subject_sort": "unicode_codepoint_ascending",
        "validation_total": validation_total,
        "counts": {name: len(values) for name, values in splits.items()},
        "per_label_counts": per_label,
        "source_manifest_sha256": sha256_file(manifest_path),
        "splits": splits,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_json(output_path, payload)

    mapping = {
        subject_id: split_name
        for split_name, subject_ids in splits.items()
        for subject_id in subject_ids
    }
    frame["derived_split"] = frame["subject_id"].map(mapping)
    if frame["derived_split"].isna().any():
        raise SplitBuildError("derived split mapping left one or more manifest rows empty")
    atomic_write_parquet(manifest_path, frame.sort_values("sample_id", kind="stable"))
    payload["file_sha256"] = sha256_file(output_path)
    payload["updated_manifest_sha256"] = sha256_file(manifest_path)
    return payload


def _validation_quotas(
    class_subjects: dict[str, list[str]],
    validation_total: int,
    official_train_total: int,
) -> dict[str, int]:
    exact = {
        label: validation_total * len(class_subjects[label]) / official_train_total
        for label in LABEL_ORDER
    }
    quotas = {label: math.floor(exact[label]) for label in LABEL_ORDER}
    remaining = validation_total - sum(quotas.values())
    order = sorted(
        LABEL_ORDER,
        key=lambda label: (-(exact[label] - quotas[label]), LABEL_ORDER.index(label)),
    )
    for label in order[:remaining]:
        quotas[label] += 1
    for label in LABEL_ORDER:
        if len(class_subjects[label]) > 1 and quotas[label] >= len(class_subjects[label]):
            raise SplitBuildError(f"validation quota leaves no {label} subject in Train")
    return quotas


def _validate_split(
    splits: dict[str, list[str]],
    subject_rows: Any,
    validation_total: int,
) -> None:
    sets = {name: set(values) for name, values in splits.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets[
        "validation"
    ] & sets["test"]:
        raise SplitBuildError("subject leakage detected between derived splits")
    expected = set(subject_rows["subject_id"].astype(str))
    observed = set().union(*sets.values())
    if observed != expected:
        raise SplitBuildError("derived split does not cover the manifest subjects exactly")
    if len(splits["validation"]) != validation_total:
        raise SplitBuildError("derived Validation count is incorrect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--validation-total", type=int, default=VALIDATION_TOTAL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_subject_split(
        manifest_path=args.manifest,
        output_path=args.output,
        strict_counts=not args.allow_count_mismatch,
        validation_total=args.validation_total,
    )
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
