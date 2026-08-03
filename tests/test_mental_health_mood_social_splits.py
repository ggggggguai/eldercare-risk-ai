from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social import splits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = (
    REPOSITORY_ROOT / "data" / "processed" / "mental_health" / "mood_social" / "v3.3.3"
)
VALIDATOR = REPOSITORY_ROOT / "scripts" / "validate_mood_social_splits_v3_3_3.py"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _formal_participant_units() -> list[splits.ParticipantUnit]:
    units: list[splits.ParticipantUnit] = []
    for path in sorted(PROCESSED_ROOT.glob("*/canonical_*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=["dataset_id", "global_participant_id", "binary_target"],
        )
        grouped = (
            frame.groupby(
                ["dataset_id", "global_participant_id"],
                sort=True,
                observed=True,
            )["binary_target"]
            .agg(row_count="size", positive_row_count="sum")
            .reset_index()
        )
        units.extend(
            splits.ParticipantUnit(
                dataset_id=str(row.dataset_id),
                global_participant_id=str(row.global_participant_id),
                row_count=int(row.row_count),
                positive_row_count=int(row.positive_row_count),
            )
            for row in grouped.itertuples(index=False)
        )
    return units


def _run_validator(manifest_path: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hash_stratification_is_deterministic_balanced_and_seeded() -> None:
    units = [
        splits.ParticipantUnit(
            dataset_id="dataset_a",
            global_participant_id=(
                f"dataset_a::participant_{row_count}_{positive_row_count}_{index:02d}"
            ),
            row_count=row_count,
            positive_row_count=positive_row_count,
        )
        for row_count, positive_row_count in ((1, 0), (1, 1), (4, 2))
        for index in range(13)
    ]

    first = splits.assign_participant_folds(
        units,
        scope="outer",
        seed=20260728,
        fold_count=5,
    )
    second = splits.assign_participant_folds(
        list(reversed(units)),
        scope="outer",
        seed=20260728,
        fold_count=5,
    )
    changed_seed = splits.assign_participant_folds(
        units,
        scope="outer",
        seed=20260729,
        fold_count=5,
    )

    assert first == second
    assert first != changed_seed
    assert set(first) == {unit.global_participant_id for unit in units}
    for row_count, positive_row_count in ((1, 0), (1, 1), (4, 2)):
        stratum_keys = {
            unit.global_participant_id
            for unit in units
            if (
                unit.row_count == row_count
                and unit.positive_row_count == positive_row_count
            )
        }
        counts = [sum(first[key] == fold for key in stratum_keys) for fold in range(5)]
        assert max(counts) - min(counts) <= 1


def test_nested_assignments_exclude_outer_test_from_its_inner_split() -> None:
    units = [
        splits.ParticipantUnit(
            dataset_id="dataset_a",
            global_participant_id=f"dataset_a::participant_{target}_{index:02d}",
            row_count=1,
            positive_row_count=target,
        )
        for target in (0, 1)
        for index in range(10)
    ]

    assignments = splits.build_nested_assignments(
        units,
        seed=20260728,
        outer_fold_count=5,
        inner_fold_count=5,
    )

    assert len(assignments) == len(units)
    for assignment in assignments:
        outer_fold = assignment["outer_fold"]
        inner = assignment["inner_validation_fold_by_outer_fold"]
        assert set(inner) == {"0", "1", "2", "3", "4"}
        assert inner[str(outer_fold)] is None
        assert all(
            value in range(5) for key, value in inner.items() if key != str(outer_fold)
        )

    for outer_fold in range(5):
        outer_train = {
            row["global_participant_id"]
            for row in assignments
            if row["outer_fold"] != outer_fold
        }
        inner_validation = [
            {
                row["global_participant_id"]
                for row in assignments
                if row["inner_validation_fold_by_outer_fold"][str(outer_fold)]
                == inner_fold
            }
            for inner_fold in range(5)
        ]
        assert set().union(*inner_validation) == outer_train
        assert sum(map(len, inner_validation)) == len(outer_train)


def test_formal_manifest_binds_inputs_and_has_no_participant_leakage() -> None:
    manifest = splits.build_split_manifest(repository_root=REPOSITORY_ROOT)

    assert manifest["manifest_version"] == (
        "mood-social-nested-participant-split-manifest-v1"
    )
    assert manifest["status"] == "frozen"
    assert manifest["protocol"]["random_seed"] == 20260728
    assert manifest["protocol"]["outer_fold_count"] == 5
    assert manifest["protocol"]["inner_fold_count"] == 5
    assert manifest["protocol"]["stratification_fields"] == [
        "dataset_id",
        "participant_row_count",
        "participant_positive_row_count",
    ]
    assert manifest["feature_schema_binding"] == {
        "schema_version": "mood_social_feature_schema_v3_3_3",
        "snapshot_relative_path": (
            "data/processed/mental_health/mood_social/v3.3.3/"
            "manifests/feature_schema_manifest.json"
        ),
        "snapshot_sha256": (
            "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
        ),
    }
    assert [row["dataset_id"] for row in manifest["inputs"]] == [
        "psyche_d",
        "resilient",
        "nhanes",
        "shenzhen_elderly",
        "nhanes_ssq_2005_2008",
    ]
    assert manifest["totals"] == {
        "dataset_count": 5,
        "negative_row_count": 18112,
        "participant_count": 15361,
        "positive_row_count": 4079,
        "row_count": 22191,
    }

    assignments = manifest["participant_assignments"]
    assert len(assignments) == 15361
    assert len({row["global_participant_id"] for row in assignments}) == 15361
    assert all(
        set(row)
        == {
            "dataset_id",
            "global_participant_id",
            "inner_validation_fold_by_outer_fold",
            "outer_fold",
        }
        for row in assignments
    )
    assert all("binary_target" not in row for row in assignments)
    assert all("participant_id" not in row for row in assignments)

    assignment_by_key = {row["global_participant_id"]: row for row in assignments}
    for path in sorted(PROCESSED_ROOT.glob("*/canonical_*.parquet")):
        frame = pd.read_parquet(path, columns=["global_participant_id"])
        assert frame["global_participant_id"].map(assignment_by_key).notna().all()
        assert (
            frame.assign(
                outer_fold=frame["global_participant_id"].map(
                    {key: row["outer_fold"] for key, row in assignment_by_key.items()}
                )
            )
            .groupby("global_participant_id")["outer_fold"]
            .nunique()
            .eq(1)
            .all()
        )

    assert (
        hashlib.sha256(_canonical_json_bytes(assignments)).hexdigest()
        == (manifest["integrity"]["participant_assignments_sha256"])
    )
    core = {key: value for key, value in manifest.items() if key != "integrity"}
    assert (
        hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
        == (manifest["integrity"]["manifest_core_sha256"])
    )


def test_formal_strata_are_balanced_in_every_outer_and_inner_scope() -> None:
    units = _formal_participant_units()
    manifest = splits.build_split_manifest(repository_root=REPOSITORY_ROOT)
    assignments = {
        row["global_participant_id"]: row for row in manifest["participant_assignments"]
    }

    def assert_balanced(
        eligible: list[splits.ParticipantUnit],
        fold_by_key: dict[str, int],
    ) -> None:
        strata: dict[tuple[str, int, int], list[str]] = {}
        for unit in eligible:
            strata.setdefault(unit.stratum, []).append(unit.global_participant_id)
        for keys in strata.values():
            counts = [
                sum(fold_by_key[key] == fold for key in keys) for fold in range(5)
            ]
            assert max(counts) - min(counts) <= 1

    assert_balanced(
        units,
        {key: row["outer_fold"] for key, row in assignments.items()},
    )
    for outer_fold in range(5):
        eligible = [
            unit
            for unit in units
            if assignments[unit.global_participant_id]["outer_fold"] != outer_fold
        ]
        assert_balanced(
            eligible,
            {
                unit.global_participant_id: assignments[unit.global_participant_id][
                    "inner_validation_fold_by_outer_fold"
                ][str(outer_fold)]
                for unit in eligible
            },
        )


def test_build_is_byte_deterministic_and_writer_is_guarded(
    tmp_path: Path,
) -> None:
    first = splits.build_split_manifest(repository_root=REPOSITORY_ROOT)
    second = splits.build_split_manifest(repository_root=REPOSITORY_ROOT)
    assert _canonical_json_bytes(first) == _canonical_json_bytes(second)

    output = tmp_path / "splits" / "split_manifest.json"
    first_hash = splits.write_split_manifest(first, output)
    assert first_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        splits.write_split_manifest(second, output)
    assert output.read_bytes() == original

    assert splits.write_split_manifest(second, output, overwrite=True) == first_hash
    assert output.read_bytes() == original
    assert not (output.parent / ".split_manifest.write.lock").exists()


def test_input_hash_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sha256 = splits._sha256_file

    def drifted_sha256(path: Path) -> str:
        if path.name == "canonical_psyche_d.parquet":
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(splits, "_sha256_file", drifted_sha256)
    with pytest.raises(splits.SplitInputError, match="frozen input hash"):
        splits.build_split_manifest(repository_root=REPOSITORY_ROOT)


def test_independent_validator_accepts_formal_manifest_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    manifest = splits.build_split_manifest(repository_root=REPOSITORY_ROOT)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest))

    completed = _run_validator(manifest_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "pass"
    assert result["participant_count"] == 15361
    assert result["outer_fold_count"] == 5
    assert result["inner_fold_count"] == 5
    assert (
        result["split_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )

    manifest["participant_assignments"][0]["outer_fold"] = (
        manifest["participant_assignments"][0]["outer_fold"] + 1
    ) % 5
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    rejected = _run_validator(manifest_path)
    assert rejected.returncode != 0
    assert "participant_" not in rejected.stderr


def test_validator_does_not_import_split_builder() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "elderly_monitoring.modules.mental_health.mood_social.splits" not in source
