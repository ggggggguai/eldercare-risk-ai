from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from elderly_monitoring.datasets.adapters import nhanes_ssq
from elderly_monitoring.modules.mental_health.validation.public_datasets import (
    nhanes_ssq as compatibility,
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
SOURCE_ROOT = WORKSPACE_ROOT / nhanes_ssq.SOURCE_RELATIVE


def _demo(
    seqn: int,
    *,
    age: int = 65,
    sex: int = 1,
    marital: int = 1,
    education: int = 3,
    household_income: int = 5,
    poverty_ratio: float = 2.0,
    cycle: str = "D",
) -> dict[str, object]:
    return {
        "SEQN": seqn,
        "RIDAGEYR": age,
        "RIAGENDR": sex,
        "DMDMARTL": marital,
        "DMDEDUC2": education,
        "INDHHINC": household_income if cycle == "D" else np.nan,
        "INDHHIN2": household_income if cycle == "E" else np.nan,
        "INDFMPIR": poverty_ratio,
    }


def _dpq(
    seqn: int,
    *,
    item_values: tuple[object, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0),
    function_value: object = 0,
) -> dict[str, object]:
    return {
        "SEQN": seqn,
        **{
            field: value
            for field, value in zip(nhanes_ssq.DPQ_FIELDS, item_values, strict=True)
        },
        "DPQ100": function_value,
    }


def _ssq(seqn: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "SEQN": seqn,
        "SSQ011": 1,
        **{field: np.nan for field in nhanes_ssq.SSQ_SUPPORT_SOURCE_FIELDS},
        "SSQ031": 2,
        "SSQ041": np.nan,
        "SSD044": 0,
        "SSQ051": 1,
        "SSQ061": 3,
    }
    row.update(overrides)
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_private_key(value: Any) -> bool:
    if isinstance(value, dict):
        forbidden = {
            "participant_ids",
            "participant_values",
            "global_participant_ids",
            "seqn_values",
        }
        return bool(forbidden.intersection(value)) or any(
            _contains_private_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_private_key(item) for item in value)
    return False


def test_data001_compatibility_entrypoint_reexports_authoritative_adapter() -> None:
    assert compatibility.build_canonical_frame is nhanes_ssq.build_canonical_frame
    assert (
        compatibility.build_nhanes_ssq_artifacts
        is nhanes_ssq.build_nhanes_ssq_artifacts
    )


def test_cycle_key_dpq_scoring_and_strict_profile_mapping() -> None:
    zero = np.nextafter(0.0, 1.0)
    demographics = {
        "D": pd.DataFrame([_demo(101, age=69, cycle="D")]),
        "E": pd.DataFrame(
            [
                _demo(
                    101,
                    age=82,
                    sex=2,
                    marital=3,
                    education=5,
                    cycle="E",
                )
            ]
        ),
    }
    dpq = {
        "D": pd.DataFrame([_dpq(101, item_values=(zero,) * 9)]),
        "E": pd.DataFrame([_dpq(101, item_values=(3,) * 9)]),
    }
    ssq = {
        "D": pd.DataFrame([_ssq(101, SSQ021A=10, SSQ061=4)]),
        "E": pd.DataFrame([_ssq(101, SSQ021L=21, SSQ061=8)]),
    }

    canonical = nhanes_ssq.build_canonical_frame(demographics, dpq, ssq)

    assert canonical["participant_id"].tolist() == ["D:101", "E:101"]
    assert canonical["global_participant_id"].tolist() == [
        "nhanes_ssq_2005_2008::D:101",
        "nhanes_ssq_2005_2008::E:101",
    ]
    assert canonical["phq9_total"].tolist() == [0, 27]
    assert canonical["phq9_severity"].tolist() == [0, 4]
    assert canonical["binary_target"].tolist() == [0, 1]
    assert canonical["social_context.age_group"].tolist() == ["60_69", "80_plus"]
    assert canonical["social_context.sex"].tolist() == ["male", "female"]
    assert canonical["social_context.marital_status"].tolist() == [
        "partnered",
        "not_partnered",
    ]
    assert canonical["social_context.education_level"].tolist() == [
        "middle",
        "high_or_above",
    ]
    assert canonical["source__SSQ061"].tolist() == [4, 8]
    assert canonical["source__SSQ021A"].tolist()[0] == 10
    assert pd.isna(canonical["source__SSQ021A"].tolist()[1])


def test_incomplete_or_invalid_dpq_and_missing_ssq_are_excluded() -> None:
    demographics = {
        "D": pd.DataFrame(
            [
                _demo(1, cycle="D"),
                _demo(2, cycle="D"),
                _demo(3, cycle="D"),
                _demo(4, age=59, cycle="D"),
            ]
        ),
        "E": pd.DataFrame(columns=nhanes_ssq.DEMOGRAPHIC_SOURCE_COLUMNS),
    }
    dpq = {
        "D": pd.DataFrame(
            [
                _dpq(1),
                _dpq(2, item_values=(0, 0, 0, 0, 7, 0, 0, 0, 0)),
                _dpq(3),
                _dpq(4),
            ]
        ),
        "E": pd.DataFrame(columns=("SEQN", *nhanes_ssq.DPQ_FIELDS, "DPQ100")),
    }
    ssq = {
        "D": pd.DataFrame([_ssq(1), _ssq(2), _ssq(4)]),
        "E": pd.DataFrame(columns=("SEQN", *nhanes_ssq.SSQ_FIELDS)),
    }

    canonical = nhanes_ssq.build_canonical_frame(demographics, dpq, ssq)

    assert canonical["participant_id"].tolist() == ["D:1"]


def test_ssq_fields_never_map_to_s10_or_other_production_features() -> None:
    mapping = nhanes_ssq.build_field_mapping()
    mapped = {
        row["canonical_value_column"]
        for row in mapping["target_fields"]
        if row["mapping_status"] == "mapped"
    }

    assert mapped == {
        "social_context.age_group",
        "social_context.sex",
        "social_context.marital_status",
        "social_context.education_level",
    }
    assert mapping["fold_supported_feature_set"] == []
    assert mapping["x_source"]["applicable"] is False
    assert mapping["ecdf_contract"]["applicable"] is False
    assert mapping["ssq_boundary"]["s10_social_contact_mapping_allowed"] is False
    assert mapping["ssq_boundary"]["SSQ061_maps_to_active_contact_count"] is False
    assert all(
        not row["source_fields"]
        for row in mapping["target_fields"]
        if row["group"] == "social_contact"
    )


def test_all_unsupported_features_are_null_with_mask_zero() -> None:
    canonical = nhanes_ssq.build_canonical_frame(
        {
            "D": pd.DataFrame([_demo(1)]),
            "E": pd.DataFrame(columns=nhanes_ssq.DEMOGRAPHIC_SOURCE_COLUMNS),
        },
        {
            "D": pd.DataFrame([_dpq(1)]),
            "E": pd.DataFrame(columns=("SEQN", *nhanes_ssq.DPQ_FIELDS, "DPQ100")),
        },
        {
            "D": pd.DataFrame([_ssq(1)]),
            "E": pd.DataFrame(columns=("SEQN", *nhanes_ssq.SSQ_FIELDS)),
        },
    )
    supported = set(nhanes_ssq.BASE_SUPPORTED_FEATURE_SET)
    for target in nhanes_ssq.feature_columns():
        if target in supported:
            continue
        assert canonical[target].isna().all()
        assert canonical[f"feature_mask.{target}"].eq(0).all()
    assert not any("x_source" in column for column in canonical.columns)
    assert len(canonical.columns) == 156


def test_source_manifest_is_frozen_with_xpt_and_codebook_fields() -> None:
    manifest_path = SOURCE_ROOT / nhanes_ssq.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert _sha256(manifest_path) == nhanes_ssq.SOURCE_MANIFEST_SHA256
    assert manifest["source_collection_sha256"] == nhanes_ssq.SOURCE_COLLECTION_SHA256
    assert manifest["file_count"] == 12
    assert manifest["total_bytes"] == 9_762_095
    assert manifest["downloaded_on"] == "2026-07-31"
    dpq_d = {row["file_name"]: row for row in manifest["files"]}
    assert dpq_d["DPQ_D.xpt"]["field_count"] == 11
    assert dpq_d["DPQ_D.htm"]["field_count"] == 13
    assert {"DPQ001", "DPQ095"}.issubset(dpq_d["DPQ_D.htm"]["fields"])
    assert {"DPQ001", "DPQ095"}.isdisjoint(dpq_d["DPQ_D.xpt"]["fields"])


def test_modified_source_manifest_is_rejected_before_source_build(
    tmp_path: Path,
) -> None:
    fake_workspace = tmp_path / "workspace"
    fake_source = fake_workspace / nhanes_ssq.SOURCE_RELATIVE
    fake_algorithm = fake_workspace / "algorithm" / "eldercare-risk-ai-main"
    fake_manifests = (
        fake_algorithm
        / "data"
        / "processed"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "manifests"
    )
    fake_source.mkdir(parents=True)
    fake_manifests.mkdir(parents=True)
    shutil.copy2(
        SOURCE_ROOT / nhanes_ssq.SOURCE_MANIFEST_NAME,
        fake_source / nhanes_ssq.SOURCE_MANIFEST_NAME,
    )
    for name in ("dataset_file_manifest.jsonl", "feature_schema_manifest.json"):
        shutil.copy2(MANIFESTS_ROOT / name, fake_manifests / name)
    with (fake_source / nhanes_ssq.SOURCE_MANIFEST_NAME).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")

    with pytest.raises(nhanes_ssq.NhanesSsqAdapterError, match="manifest SHA-256"):
        nhanes_ssq.validate_frozen_inputs(fake_workspace, fake_manifests)


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, nhanes_ssq.BuildResult]:
    output = tmp_path_factory.mktemp("nhanes-ssq-artifacts")
    first = nhanes_ssq.build_nhanes_ssq_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=output,
        overwrite=True,
    )
    second = nhanes_ssq.build_nhanes_ssq_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=output,
        overwrite=False,
    )
    assert first.artifact_sha256 == second.artifact_sha256
    return output, second


def test_real_build_has_frozen_aggregate_statistics(
    generated_artifacts: tuple[Path, nhanes_ssq.BuildResult],
) -> None:
    output, result = generated_artifacts
    frame = pq.read_table(
        output / nhanes_ssq.DATASET_ID / "canonical_nhanes_ssq_2005_2008.parquet"
    ).to_pandas()

    assert result.row_count == 3_150
    assert result.participant_count == 3_150
    assert result.positive_row_count == 185
    assert frame["cycle"].value_counts().sort_index().to_dict() == {
        "D": 1_311,
        "E": 1_839,
    }
    assert frame["phq9_severity"].value_counts().sort_index().to_dict() == {
        0: 2_521,
        1: 444,
        2: 120,
        3: 50,
        4: 15,
    }
    assert frame["global_participant_id"].is_unique
    assert not any(
        "ecdf" in path.name.casefold()
        for path in (output / nhanes_ssq.DATASET_ID).iterdir()
    )


def test_reports_are_aggregate_only_and_artifacts_are_hashed(
    generated_artifacts: tuple[Path, nhanes_ssq.BuildResult],
) -> None:
    output, result = generated_artifacts
    dataset_root = output / nhanes_ssq.DATASET_ID
    manifest = json.loads(
        (dataset_root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for relative, row in manifest["artifacts"].items():
        assert _sha256(output / Path(relative)) == row["sha256"]
    assert result.artifact_sha256[
        f"{nhanes_ssq.DATASET_ID}/artifact_manifest.json"
    ] == _sha256(dataset_root / "artifact_manifest.json")
    for path in (
        dataset_root / "data_quality_report.json",
        dataset_root / "adapter_metadata.json",
        output / "mappings" / "nhanes_ssq_2005_2008_v3_3_3_mapping.json",
        dataset_root / "artifact_manifest.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not _contains_private_key(payload)
    combined = "".join(
        path.read_text(encoding="utf-8")
        for path in (
            dataset_root / "data_quality_report.json",
            dataset_root / "adapter_metadata.json",
            output / "mappings" / "nhanes_ssq_2005_2008_v3_3_3_mapping.json",
        )
    )
    assert '"participant_values_written_to_report": false' in combined
    assert '"s10_social_contact_mapping_allowed": false' in combined
    assert '"formal_ecdf_instance_created": false' in combined


def test_independent_validator_reconstructs_real_outputs(
    generated_artifacts: tuple[Path, nhanes_ssq.BuildResult],
) -> None:
    output, _ = generated_artifacts
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "validate_nhanes_ssq_v3_3_3.py"),
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
    assert result["participant_count"] == 3_150
    assert result["positive_count"] == 185
    assert result["check_count"] >= 200
    validator_source = (
        REPOSITORY_ROOT / "scripts" / "validate_nhanes_ssq_v3_3_3.py"
    ).read_text(encoding="utf-8")
    assert "datasets.adapters.nhanes_ssq" not in validator_source
