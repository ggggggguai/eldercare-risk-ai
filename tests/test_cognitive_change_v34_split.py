import json
from pathlib import Path

import pandas as pd
import pytest

from training.cognitive_change_clue import build_subject_cv_v34 as cv_module


LABEL_COUNTS = {"HC": 142, "MCI": 205, "AD": 112}
TEST_LABEL_COUNTS = {"HC": 36, "MCI": 51, "AD": 28}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    train_subjects = []
    test_subjects = []
    for official_split, counts in (("train", LABEL_COUNTS), ("test", TEST_LABEL_COUNTS)):
        for diagnosis, count in counts.items():
            for index in range(count):
                subject_id = f"{official_split}-{diagnosis}-{index:03d}"
                (train_subjects if official_split == "train" else test_subjects).append(
                    subject_id
                )
                for task in range(1, 4):
                    rows.append(
                        {
                            "sample_id": f"{subject_id}-task-{task}",
                            "subject_id": subject_id,
                            "diagnosis_label": diagnosis,
                            "official_split": official_split,
                        }
                    )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, engine="pyarrow", index=False)
    source_split = tmp_path / "split.json"
    source_split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": sorted(train_subjects[:367]),
                    "validation": sorted(train_subjects[367:]),
                    "test": sorted(test_subjects),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest, source_split


def _build(tmp_path: Path):
    manifest, source_split = _fixture(tmp_path)
    output = tmp_path / "v34.json"
    audit = tmp_path / "audit.json"
    payload, audit_payload = cv_module.build_subject_cv(
        manifest_path=manifest,
        source_split_path=source_split,
        output_path=output,
        audit_path=audit,
    )
    return payload, audit_payload, output, audit


def test_v34_split_is_official_train_only_subject_safe_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(cv_module, "workspace_relative", lambda path: Path(path).name)
    payload, audit, output, audit_path = _build(tmp_path)
    assert output.is_file() and audit_path.is_file()
    assert payload["sklearn_version"] == "1.9.0"
    assert payload["official_train_subject_count"] == 459
    assert payload["official_test_subject_count"] == 115
    assert payload["official_test_overlap_count"] == 0
    assert audit["status"] == "passed"
    assert audit["outer_evaluation_unique_subject_count"] == 459
    assert audit["task_cross_fold_count"] == 0
    assert len(payload["folds"]) == 5

    evaluation_subjects = []
    for fold in payload["folds"]:
        roles = [
            set(fold[f"{name}_subjects"])
            for name in (
                "inner_train",
                "inner_validation",
                "inner_calibration",
                "outer_evaluation",
            )
        ]
        assert len(set().union(*roles)) == 459
        assert sum(len(role) for role in roles) == 459
        assert all(
            set(fold["per_label_counts"][name]) == {"HC", "MCI", "AD"}
            for name in (
                "inner_train",
                "inner_validation",
                "inner_calibration",
                "outer_evaluation",
            )
        )
        evaluation_subjects.extend(fold["outer_evaluation_subjects"])
    assert len(evaluation_subjects) == len(set(evaluation_subjects)) == 459


def test_v34_split_generation_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(cv_module, "workspace_relative", lambda path: Path(path).name)
    first, _, _, _ = _build(tmp_path / "first")
    second, _, _, _ = _build(tmp_path / "second")
    for key in ("source_split_path", "source_manifest_path"):
        first[key] = second[key]
    assert first == second


def test_v34_split_rejects_source_test_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(cv_module, "workspace_relative", lambda path: Path(path).name)
    manifest, source_split = _fixture(tmp_path)
    payload = json.loads(source_split.read_text(encoding="utf-8"))
    leaked = payload["splits"]["test"][0]
    payload["splits"]["train"].append(leaked)
    source_split.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(cv_module.CognitiveV34SplitError, match="leaks official Test"):
        cv_module.build_subject_cv(
            manifest_path=manifest,
            source_split_path=source_split,
            output_path=tmp_path / "v34.json",
            audit_path=tmp_path / "audit.json",
            strict_counts=False,
        )
