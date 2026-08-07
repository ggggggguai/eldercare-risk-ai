"""Official-Train-only cached feature views for cognitive V3.4 experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .build_subject_cv_v34 import DEFAULT_OUTPUT_PATH
    from .common import load_json, sha256_file
    from .dataset import (
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        CognitiveDatasetError,
        CognitiveFeatureDataset,
    )
except ImportError:
    from build_subject_cv_v34 import DEFAULT_OUTPUT_PATH  # type: ignore[no-redef]
    from common import load_json, sha256_file  # type: ignore[no-redef]
    from dataset import (  # type: ignore[no-redef]
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        CognitiveDatasetError,
        CognitiveFeatureDataset,
    )


ROLE_NAMES = (
    "inner_train",
    "inner_validation",
    "inner_calibration",
    "outer_evaluation",
)


class V34DataError(CognitiveDatasetError):
    pass


class SubjectFeatureDataset(Dataset[dict[str, Any]]):
    """A deterministic view of cached records selected by whole subjects."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        subject_ids: Iterable[str],
        *,
        complete_modalities: Sequence[str] = (),
        available_modalities: Sequence[str] = MODALITY_ORDER,
    ) -> None:
        requested = set(str(value) for value in subject_ids)
        if not requested:
            raise V34DataError("subject view cannot be empty")
        required_indices = {MODALITY_ORDER.index(name) for name in complete_modalities}
        unknown_modalities = set(complete_modalities).difference(MODALITY_ORDER)
        if unknown_modalities:
            raise V34DataError(f"unsupported required modalities: {sorted(unknown_modalities)}")
        unknown_available = set(available_modalities).difference(MODALITY_ORDER)
        if unknown_available or not available_modalities:
            raise V34DataError(
                f"unsupported available modalities: {sorted(unknown_available)}"
            )
        forced_missing = np.asarray(
            [name not in set(available_modalities) for name in MODALITY_ORDER],
            dtype=np.bool_,
        )
        selected = []
        observed: set[str] = set()
        for source in records:
            subject_id = str(source["subject_id"])
            if subject_id not in requested:
                continue
            observed.add(subject_id)
            missing = source["missing_mask"]
            if any(bool(missing[index]) for index in required_indices):
                continue
            row = copy.copy(dict(source))
            row["missing_mask"] = np.asarray(source["missing_mask"], dtype=np.bool_) | forced_missing
            selected.append(row)
        missing_subjects = requested - observed
        if missing_subjects:
            preview = sorted(missing_subjects)[:5]
            raise V34DataError(f"subjects are absent from official Train cache: {preview}")
        if not selected:
            raise V34DataError("subject view has no records after modality filtering")
        self.records = sorted(selected, key=lambda row: str(row["sample_id"]))
        self.subject_ids = tuple(sorted(requested))
        if any(str(row["subject_id"]) not in requested for row in self.records):
            raise V34DataError("subject view contains an out-of-scope record")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
            "manifest_index": int(record["manifest_index"]),
            "diagnosis_label": str(record["diagnosis_label"]),
            "features": {
                name: torch.from_numpy(record["features"][name].copy())
                for name in MODALITY_ORDER
            },
            "quality": torch.from_numpy(record["quality"].copy()),
            "missing_mask": torch.from_numpy(record["missing_mask"].copy()),
            "hc_vs_non_hc": torch.tensor(record["hc_vs_non_hc"], dtype=torch.float32),
            "mci_vs_hc": torch.tensor(record["mci_vs_hc"], dtype=torch.float32),
            "mci_vs_hc_mask": torch.tensor(record["mci_vs_hc_mask"], dtype=torch.bool),
            "ad_mci_hc": torch.tensor(record["ad_mci_hc"], dtype=torch.long),
            "moca": torch.tensor(record["moca"], dtype=torch.float32),
            "moca_mask": torch.tensor(record["moca_mask"], dtype=torch.bool),
        }


def load_official_train_pool(*, verify_hashes: bool = True) -> tuple[list[dict[str, Any]], dict[str, str]]:
    train = CognitiveFeatureDataset(split_name="train", verify_hashes=verify_hashes)
    validation = CognitiveFeatureDataset(split_name="validation", verify_hashes=verify_hashes)
    records = sorted(
        [*train.records, *validation.records], key=lambda row: str(row["sample_id"])
    )
    sample_ids = [str(row["sample_id"]) for row in records]
    subjects = {str(row["subject_id"]) for row in records}
    if len(records) != 1377 or len(subjects) != 459:
        raise V34DataError(
            f"official Train cache count mismatch tasks={len(records)} subjects={len(subjects)}"
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise V34DataError("official Train feature pool contains duplicate sample IDs")
    input_hashes = dict(train.input_hashes)
    if validation.input_hashes != input_hashes:
        raise V34DataError("V3.3 Train and Validation views disagree on frozen inputs")
    if verify_hashes and input_hashes != EXPECTED_INPUT_SHA256:
        raise V34DataError("official Train cache hashes are not the frozen V3.3 hashes")
    return records, input_hashes


def load_fold(
    outer_fold: int,
    *,
    split_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    split_path = Path(split_path)
    payload = load_json(split_path)
    if payload.get("schema_version") != "cogpic_subject_cv_v34":
        raise V34DataError("V3.4 split schema mismatch")
    if int(payload.get("official_test_overlap_count", -1)) != 0:
        raise V34DataError("V3.4 split is not isolated from official Test")
    matches = [
        fold for fold in payload.get("folds", ()) if int(fold["outer_fold"]) == int(outer_fold)
    ]
    if len(matches) != 1:
        raise V34DataError(f"V3.4 split has no unique fold {outer_fold}")
    fold = dict(matches[0])
    roles = {
        name: set(str(value) for value in fold[f"{name}_subjects"])
        for name in ROLE_NAMES
    }
    if len(set().union(*roles.values())) != 459:
        raise V34DataError(f"fold {outer_fold} does not cover official Train")
    if sum(len(value) for value in roles.values()) != 459:
        raise V34DataError(f"fold {outer_fold} contains subject leakage")
    fold["split_sha256"] = sha256_file(split_path)
    return fold


def build_fold_datasets(
    records: Sequence[Mapping[str, Any]],
    fold: Mapping[str, Any],
    *,
    selection_modalities: Sequence[str],
    available_modalities: Sequence[str] = MODALITY_ORDER,
) -> dict[str, SubjectFeatureDataset]:
    datasets = {
        "inner_train": SubjectFeatureDataset(
            records,
            fold["inner_train_subjects"],
            available_modalities=available_modalities,
        ),
        "inner_validation": SubjectFeatureDataset(
            records,
            fold["inner_validation_subjects"],
            complete_modalities=selection_modalities,
            available_modalities=available_modalities,
        ),
        "inner_calibration": SubjectFeatureDataset(
            records,
            fold["inner_calibration_subjects"],
            available_modalities=available_modalities,
        ),
        "outer_evaluation": SubjectFeatureDataset(
            records,
            fold["outer_evaluation_subjects"],
            available_modalities=available_modalities,
        ),
    }
    subject_roles = {
        name: set(dataset.subject_ids) for name, dataset in datasets.items()
    }
    if sum(len(value) for value in subject_roles.values()) != len(
        set().union(*subject_roles.values())
    ):
        raise V34DataError("fold datasets leak subjects between roles")
    return datasets


__all__ = [
    "ROLE_NAMES",
    "SubjectFeatureDataset",
    "V34DataError",
    "build_fold_datasets",
    "load_fold",
    "load_official_train_pool",
]
