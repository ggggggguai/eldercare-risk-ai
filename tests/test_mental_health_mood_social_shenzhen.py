from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from elderly_monitoring.datasets.adapters import shenzhen
from elderly_monitoring.modules.mental_health.validation.public_datasets import (
    shenzhen as compatibility,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "mental_health"
    / "mood_social"
    / "v3.3.3"
    / "manifests"
)


def _row(code: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "code": code,
        "educationcat": 2,
        "marriagecat": 2,
        "chronicdiseases": 2,
        "Monthlypersonalincome": 3,
        "drinking": 1,
        "Smoking": 1,
        "Healthstatus": 3,
        "PHQ9score": 4,
        "GAD7score": 3,
        "ISIscore": 7,
        "Sleepduration": 7.5,
        "AD8score": 1,
        "CSIDscore": 8,
        "ULSscore": 9,
        "Depressivesymptoms": 0,
        "Anxietysymptoms": 0,
        "Mildcognitiveimpairment": 0,
        "Earlydementia": 0,
        "Insomnia": 1,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).loc[:, list(shenzhen.SOURCE_COLUMNS)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data001_compatibility_entrypoint_reexports_authoritative_adapter() -> None:
    assert compatibility.build_canonical_frame is shenzhen.build_canonical_frame
    assert compatibility.build_shenzhen_artifacts is shenzhen.build_shenzhen_artifacts


def test_dec046_excludes_every_row_in_each_conflicting_code_group() -> None:
    source = _frame(
        [
            _row("UNIQUE01", PHQ9score=12),
            _row("CONFL001", educationcat=1),
            _row("CONFL001", educationcat=5),
            _row("CONFL002", Healthstatus=1),
            _row("CONFL002", Healthstatus=5),
            _row("UNIQUE02", PHQ9score=2),
        ]
    )

    canonical = shenzhen.build_canonical_frame(source)

    assert canonical["participant_id"].tolist() == ["UNIQUE01", "UNIQUE02"]
    assert canonical["participant_id"].is_unique
    assert canonical["participant_key_policy"].eq(shenzhen.PARTICIPANT_KEY_POLICY).all()
    assert not canonical["participant_id"].str.contains("CONFL", regex=False).any()


def test_only_strict_profile_fields_are_mapped() -> None:
    source = _frame(
        [
            _row(
                "PROFILE1",
                marriagecat=1,
                Healthstatus=5,
                educationcat=1,
                Monthlypersonalincome=2,
                chronicdiseases=2,
                PHQ9score=10,
                Sleepduration=16,
                ULSscore=24,
            ),
            _row(
                "PROFILE2",
                marriagecat=2,
                Healthstatus=1,
                educationcat=5,
                Monthlypersonalincome=5,
                chronicdiseases=1,
                PHQ9score=0,
            ),
        ]
    )

    canonical = shenzhen.build_canonical_frame(source)

    assert canonical["social_context.marital_status"].tolist() == [
        "not_partnered",
        "partnered",
    ]
    assert canonical["social_context.self_rated_health"].tolist() == [5, 1]
    assert canonical["social_context.education_level"].tolist() == [
        "primary_or_less",
        "high_or_above",
    ]
    assert canonical["social_context.economic_status"].tolist() == ["low", "high"]
    for target in (
        "social_context.chronic_disease_count",
        "sleep.sleep_duration_norm",
        "social_context.social_participation_days_per_week",
    ):
        assert canonical[target].isna().all()
        assert canonical[f"feature_mask.{target}"].eq(0).all()
    assert canonical["source__Sleepduration"].tolist() == [16.0, 7.5]
    assert canonical["source__ULSscore"].tolist() == [24, 9]
    assert canonical["binary_target"].tolist() == [1, 0]


def test_mapping_covers_schema_and_prohibits_target_leakage() -> None:
    mapping = shenzhen.build_field_mapping()
    targets = mapping["target_fields"]

    assert len(targets) == 54
    assert set(mapping["base_supported_feature_set"]) == {
        "social_context.marital_status",
        "social_context.self_rated_health",
        "social_context.education_level",
        "social_context.economic_status",
    }
    assert mapping["fold_supported_feature_set"] == []
    assert (
        mapping["production_input_policy"][
            "source_label_and_neighbouring_scale_columns_are_inputs"
        ]
        is False
    )
    assert (
        mapping["production_input_policy"]["self_report_sleep_is_production_input"]
        is False
    )
    assert (
        mapping["production_input_policy"]["chronic_binary_is_mapped_to_count"] is False
    )
    allowed_sources = {
        source
        for target in targets
        if target["mapping_status"] == "mapped"
        for source in target["source_fields"]
    }
    assert allowed_sources == {
        "marriagecat",
        "Healthstatus",
        "educationcat",
        "Monthlypersonalincome",
    }


def test_modified_data001_manifest_is_rejected_before_source_build(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("dataset_file_manifest.jsonl", "feature_schema_manifest.json"):
        shutil.copy2(MANIFESTS_ROOT / name, manifests / name)
    with (manifests / "dataset_file_manifest.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")

    with pytest.raises(shenzhen.ShenzhenAdapterError, match="manifest SHA-256"):
        shenzhen.validate_frozen_inputs(WORKSPACE_ROOT, manifests)


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, shenzhen.BuildResult]:
    output = tmp_path_factory.mktemp("shenzhen-artifacts")
    first = shenzhen.build_shenzhen_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=output,
        overwrite=True,
    )
    second = shenzhen.build_shenzhen_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=output,
        overwrite=False,
    )
    assert first.artifact_sha256 == second.artifact_sha256
    return output, second


def test_real_build_has_frozen_aggregate_statistics(
    generated_artifacts: tuple[Path, shenzhen.BuildResult],
) -> None:
    output, result = generated_artifacts
    frame = pq.read_table(
        output / "shenzhen" / "canonical_shenzhen.parquet"
    ).to_pandas()

    assert result.row_count == 5_327
    assert result.participant_count == 5_327
    assert result.positive_row_count == 186
    assert frame["participant_id"].is_unique
    assert frame["phq9_severity"].value_counts().sort_index().to_dict() == {
        0: 4_773,
        1: 368,
        2: 117,
        3: 46,
        4: 23,
    }
    assert not any(
        "ecdf" in path.name.casefold() for path in (output / "shenzhen").iterdir()
    )


def test_real_artifacts_are_hashed_and_reports_are_aggregate_only(
    generated_artifacts: tuple[Path, shenzhen.BuildResult],
) -> None:
    output, result = generated_artifacts
    manifest = json.loads(
        (output / "shenzhen" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for relative, row in manifest["artifacts"].items():
        assert _sha256(output / Path(relative)) == row["sha256"]
    assert result.artifact_sha256["shenzhen/artifact_manifest.json"] == _sha256(
        output / "shenzhen" / "artifact_manifest.json"
    )
    combined = "".join(
        path.read_text(encoding="utf-8")
        for path in (
            output / "shenzhen" / "data_quality_report.json",
            output / "shenzhen" / "adapter_metadata.json",
            output / "mappings" / "shenzhen_v3_3_3_mapping.json",
        )
    )
    source = pd.read_csv(
        WORKSPACE_ROOT / shenzhen.SOURCE_RELATIVE / shenzhen.SOURCE_CSV_NAME,
        dtype="string",
        encoding="utf-8-sig",
    )
    assert not any(str(value) in combined for value in source["code"])
    assert '"participant_values_written_to_report": false' in combined
    assert '"arbitrary_rows_kept": 0' in combined


def test_independent_validator_reconstructs_real_outputs(
    generated_artifacts: tuple[Path, shenzhen.BuildResult],
) -> None:
    output, _ = generated_artifacts
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "validate_shenzhen_v3_3_3.py"),
            "--workspace-root",
            str(WORKSPACE_ROOT),
            "--output-root",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert result["status"] == "pass"
    assert result["participant_count"] == 5_327
    assert result["positive_count"] == 186
    assert result["check_count"] >= 150
    validator_source = (
        REPOSITORY_ROOT / "scripts" / "validate_shenzhen_v3_3_3.py"
    ).read_text(encoding="utf-8")
    assert "datasets.adapters.shenzhen" not in validator_source
