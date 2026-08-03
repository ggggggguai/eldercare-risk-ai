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

from elderly_monitoring.datasets.adapters import nhanes
from elderly_monitoring.modules.mental_health.validation.public_datasets import (
    nhanes as compatibility,
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


def _demo(
    seqns: list[int],
    *,
    ages: list[int] | None = None,
    sexes: list[int] | None = None,
    marital: list[int] | None = None,
    education: list[int] | None = None,
) -> pd.DataFrame:
    length = len(seqns)
    return pd.DataFrame(
        {
            "SEQN": seqns,
            "RIDAGEYR": ages or [65] * length,
            "RIAGENDR": sexes or [1] * length,
            "DMDMARTL": marital or [1] * length,
            "DMDEDUC2": education or [4] * length,
        }
    )


def _dpq(
    seqns: list[int],
    *,
    item_values: list[list[float]] | None = None,
    dpq100: list[float] | None = None,
) -> pd.DataFrame:
    values = item_values or [[0.0] * 9 for _ in seqns]
    data: dict[str, object] = {"SEQN": seqns}
    for index, field in enumerate(nhanes.DPQ_FIELDS):
        data[field] = [row[index] for row in values]
    data["DPQ100"] = dpq100 or [1.0] * len(seqns)
    return pd.DataFrame(data)


def _pam_row(
    seqn: int,
    window_number: int,
    *,
    calendar_date: str = "2000-01-02",
    excluded: int = 0,
    m10: float = 10.0,
    l5: float = 2.0,
    sleep_minutes: float = 400.0,
    period_minutes: float = 500.0,
    onset: float = 23.0,
    wake: float = 31.0,
    awake_minutes: float = 50.0,
) -> dict[str, object]:
    return {
        "SEQN": seqn,
        "calendar_date": calendar_date,
        "window_number": window_number,
        "excluded": excluded,
        "M10VALUE": m10,
        "L5VALUE": l5,
        "dur_spt_sleep_min": sleep_minutes,
        "dur_spt_min": period_minutes,
        "sleep_efficiency": sleep_minutes / period_minutes,
        "sleeponset": onset,
        "wakeup": wake,
        "dur_spt_wake_IN_min": awake_minutes,
    }


def _canonical(
    seqns: list[int],
    *,
    cycle: str = "G",
    m10_values: list[float] | None = None,
) -> pd.DataFrame:
    other = "H" if cycle == "G" else "G"
    values = m10_values or [float(index + 1) for index in range(len(seqns))]
    demos = {cycle: _demo(seqns), other: _demo([900_001], ages=[59])}
    dpqs = {cycle: _dpq(seqns), other: _dpq([900_001])}
    pam = pd.DataFrame(
        [
            _pam_row(seqn, 1, m10=value)
            for seqn, value in zip(seqns, values, strict=True)
        ]
        + [_pam_row(900_001, 1)]
    )
    return nhanes.build_canonical_frame(demos, dpqs, pam)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data001_compatibility_entrypoint_reexports_authoritative_adapter() -> None:
    assert compatibility.build_nhanes_artifacts is nhanes.build_nhanes_artifacts
    assert compatibility.TrainingFoldECDF is nhanes.TrainingFoldECDF


def test_cycle_scoped_keys_and_dpq100_is_not_scored() -> None:
    demos = {
        "G": _demo([101], ages=[65]),
        "H": _demo([201], ages=[75]),
    }
    dpqs = {
        "G": _dpq([101], item_values=[[1] * 9], dpq100=[3]),
        "H": _dpq([201], item_values=[[0] * 9], dpq100=[0]),
    }
    pam = pd.DataFrame([_pam_row(101, 1), _pam_row(201, 1)])

    frame = nhanes.build_canonical_frame(demos, dpqs, pam)

    assert frame["cycle"].tolist() == ["G", "H"]
    assert frame["participant_id"].tolist() == ["G:101", "H:201"]
    assert frame["global_participant_id"].tolist() == [
        "nhanes::G:101",
        "nhanes::H:201",
    ]
    assert frame["phq9_total"].tolist() == [9, 0]
    assert "DPQ100" not in frame.columns


def test_same_seqn_in_two_cycles_is_rejected_instead_of_cross_joined() -> None:
    demos = {"G": _demo([101]), "H": _demo([101])}
    dpqs = {"G": _dpq([101]), "H": _dpq([101])}
    pam = pd.DataFrame([_pam_row(101, 1)])

    with pytest.raises(nhanes.NhanesAdapterError, match="more than one survey cycle"):
        nhanes.build_canonical_frame(demos, dpqs, pam)


def test_age_filter_and_all_nine_valid_dpq_items_are_required() -> None:
    underflow_zero = 5.397605e-79
    demos = {
        "G": _demo([101, 102, 103], ages=[59, 60, 80]),
        "H": _demo([201]),
    }
    dpqs = {
        "G": _dpq(
            [101, 102, 103],
            item_values=[
                [0] * 9,
                [underflow_zero] * 9,
                [0, 0, 0, 0, 0, 0, 0, 0, 7],
            ],
        ),
        "H": _dpq([201], item_values=[[0, 0, 0, 0, 0, 0, 0, 0, 9]]),
    }
    pam = pd.DataFrame(
        [_pam_row(101, 1), _pam_row(102, 1), _pam_row(103, 1), _pam_row(201, 1)]
    )

    frame = nhanes.build_canonical_frame(demos, dpqs, pam)

    assert len(frame) == 1
    assert frame["participant_id"].iloc[0] == "G:102"
    assert frame["phq9_total"].iloc[0] == 0


def test_window_number_not_calendar_date_controls_limit_and_audit_storage() -> None:
    demos = {"G": _demo([101]), "H": _demo([201])}
    dpqs = {"G": _dpq([101]), "H": _dpq([201])}
    rows = [
        _pam_row(
            101,
            window,
            calendar_date="2000-01-02",
            m10=float(window),
        )
        for window in range(1, 9)
    ]
    rows.append(_pam_row(201, 1))

    frame = nhanes.build_canonical_frame(demos, dpqs, pd.DataFrame(rows))
    row = frame.loc[frame["cycle"].eq("G")].iloc[0]

    assert row["pam_valid_day_count"] == 7
    assert json.loads(row["window_number_values_json"]) == [1, 2, 3, 4, 5, 6, 7]
    assert row["x_source_value"] == pytest.approx(4.0)


def test_excluded_rows_are_not_silently_substituted_or_averaged() -> None:
    demos = {"G": _demo([101]), "H": _demo([201])}
    dpqs = {"G": _dpq([101]), "H": _dpq([201])}
    pam = pd.DataFrame(
        [
            _pam_row(101, 1, excluded=1, m10=1_000.0),
            _pam_row(101, 2, excluded=0, m10=20.0),
            _pam_row(201, 1),
        ]
    )

    frame = nhanes.build_canonical_frame(demos, dpqs, pam)
    row = frame.loc[frame["cycle"].eq("G")].iloc[0]

    assert json.loads(row["window_number_values_json"]) == [2]
    assert row["x_source_value"] == pytest.approx(20.0)
    assert row["pam_valid_day_count"] == 1


def test_strict_production_mappings_and_null_masks() -> None:
    demos = {
        "G": _demo([101], ages=[80], sexes=[2], marital=[2], education=[1]),
        "H": _demo([201]),
    }
    dpqs = {"G": _dpq([101]), "H": _dpq([201])}
    pam = pd.DataFrame(
        [
            _pam_row(101, 1, m10=10.0, l5=2.0, onset=23.0, wake=31.0),
            _pam_row(101, 2, m10=20.0, l5=4.0, onset=24.0, wake=32.0),
            _pam_row(201, 1),
        ]
    )

    frame = nhanes.build_canonical_frame(demos, dpqs, pam)
    row = frame.loc[frame["cycle"].eq("G")].iloc[0]

    assert row["x_source_value"] == pytest.approx(15.0)
    assert row["activity.relative_amplitude"] == pytest.approx(2.0 / 3.0)
    assert row["activity.activity_variability"] == pytest.approx(1.0 / 3.0)
    assert row["sleep.sleep_duration_norm"] == pytest.approx(400.0 / 1440.0)
    assert row["sleep.time_in_bed_norm"] == pytest.approx(500.0 / 1440.0)
    assert row["sleep.sleep_efficiency"] == pytest.approx(0.8)
    assert row["sleep.sleep_fragmentation"] == pytest.approx(0.1)
    assert row["social_context.age_group"] == "80_plus"
    assert row["social_context.sex"] == "female"
    assert row["social_context.marital_status"] == "not_partnered"
    assert row["social_context.education_level"] == "primary_or_less"
    assert pd.isna(row[nhanes.ACTIVITY_VOLUME_COLUMN])
    assert row[nhanes.ACTIVITY_VOLUME_MASK_COLUMN] == 0
    assert pd.isna(row["social_context.economic_status"])
    assert row["feature_mask.social_context.economic_status"] == 0
    assert pd.isna(row["physiology.heart_rate_mean_bpm"])
    assert row["feature_mask.physiology.heart_rate_mean_bpm"] == 0


def test_mapping_covers_schema_and_prohibits_leakage() -> None:
    mapping = nhanes.build_field_mapping()
    target_rows = mapping["target_fields"]

    assert len(target_rows) == 54
    assert {row["canonical_value_column"] for row in target_rows} == {
        column
        for column in nhanes.canonical_column_order()
        if column.startswith(
            ("activity.", "sleep.", "physiology.", "social_context.", "social_contact.")
        )
    }
    assert mapping["labels"]["valid_item_codes"] == [0, 1, 2, 3]
    assert mapping["labels"]["missing_item_codes"] == [7, 9]
    assert mapping["labels"]["excluded_from_total"] == "DPQ100"
    assert mapping["labels"]["input_feature_use"] == "prohibited"
    assert mapping["x_source"]["only_canonical_storage_of_participant_mean"] is True
    assert mapping["production_input_policy"]["cycle_is_input"] is False
    assert mapping["production_input_policy"]["survey_weights_are_inputs"] is False
    assert mapping["ecdf_contract"]["formal_instance_before_DATA_007"] is False


def test_training_fold_ecdf_is_right_continuous_and_excludes_held_out_values() -> None:
    frame = _canonical([101, 102, 103], m10_values=[10.0, 20.0, 1_000.0])
    artifact_hash = "a" * 64
    frame_hash = nhanes.canonical_frame_sha256(frame)
    training_ids = frame["global_participant_id"].iloc[:2].astype(str).tolist()

    ecdf = nhanes.TrainingFoldECDF.fit(
        frame,
        training_participant_ids=training_ids,
        split_id="outer-1",
        canonical_artifact_sha256=artifact_hash,
        expected_canonical_frame_sha256=frame_hash,
    )
    transformed = ecdf.transform(frame)

    assert len(transformed) == len(frame) == 3
    assert ecdf.sorted_training_values == (10.0, 20.0)
    assert transformed[nhanes.ACTIVITY_VOLUME_COLUMN].tolist() == [0.5, 1.0, 1.0]
    assert transformed[nhanes.ACTIVITY_VOLUME_MASK_COLUMN].tolist() == [1, 1, 1]
    assert frame[nhanes.ACTIVITY_VOLUME_COLUMN].isna().all()
    assert frame[nhanes.ACTIVITY_VOLUME_MASK_COLUMN].eq(0).all()


@pytest.mark.parametrize("split_id", ["", "   "])
def test_training_fold_ecdf_requires_nonempty_split_id(split_id: str) -> None:
    frame = _canonical([101, 102], m10_values=[10.0, 20.0])
    with pytest.raises(nhanes.NhanesAdapterError, match="non-empty split_id"):
        nhanes.TrainingFoldECDF.fit(
            frame,
            training_participant_ids=frame["global_participant_id"]
            .astype(str)
            .tolist(),
            split_id=split_id,
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256=nhanes.canonical_frame_sha256(frame),
        )


def test_training_fold_ecdf_serialization_is_bound_and_private() -> None:
    frame = _canonical([101, 102], m10_values=[10.0, 20.0])
    participant_ids = frame["global_participant_id"].astype(str).tolist()
    ecdf = nhanes.TrainingFoldECDF.fit(
        frame,
        training_participant_ids=participant_ids,
        split_id="outer-2",
        canonical_artifact_sha256="b" * 64,
        expected_canonical_frame_sha256=nhanes.canonical_frame_sha256(frame),
    )
    payload = ecdf.to_dict()
    restored = nhanes.TrainingFoldECDF.from_dict(payload)

    assert restored == ecdf
    assert restored.to_bytes() == ecdf.to_bytes()
    assert restored.sha256 == ecdf.sha256
    assert "training_participant_ids" not in payload
    assert not any(
        identifier in ecdf.to_bytes().decode("utf-8") for identifier in participant_ids
    )


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

    with pytest.raises(nhanes.NhanesAdapterError, match="manifest SHA-256"):
        nhanes.validate_frozen_inputs(WORKSPACE_ROOT, manifests)


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, nhanes.BuildResult]:
    output = tmp_path_factory.mktemp("nhanes-artifacts")
    result = nhanes.build_nhanes_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=output,
        overwrite=True,
    )
    return output, result


def test_real_build_has_frozen_aggregate_statistics_and_no_ecdf(
    generated_artifacts: tuple[Path, nhanes.BuildResult],
) -> None:
    output, result = generated_artifacts
    frame = pq.read_table(output / "nhanes" / "canonical_nhanes.parquet").to_pandas()

    assert result.row_count == 2_775
    assert result.participant_count == 2_775
    assert result.positive_row_count == 254
    assert frame["cycle"].value_counts().to_dict() == {"H": 1_402, "G": 1_373}
    assert frame["phq9_severity"].value_counts().sort_index().to_dict() == {
        0: 2_079,
        1: 442,
        2: 165,
        3: 65,
        4: 24,
    }
    assert frame[nhanes.ACTIVITY_VOLUME_COLUMN].isna().all()
    assert not any(
        "ecdf" in path.name.lower() for path in (output / "nhanes").iterdir()
    )


def test_real_artifact_manifest_hashes_and_reports_are_aggregate_only(
    generated_artifacts: tuple[Path, nhanes.BuildResult],
) -> None:
    output, result = generated_artifacts
    manifest = json.loads(
        (output / "nhanes" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for relative, row in manifest["artifacts"].items():
        assert _sha256(output / Path(relative)) == row["sha256"]
    assert result.artifact_sha256["nhanes/artifact_manifest.json"] == _sha256(
        output / "nhanes" / "artifact_manifest.json"
    )

    quality_text = (output / "nhanes" / "data_quality_report.json").read_text(
        encoding="utf-8"
    )
    metadata_text = (output / "nhanes" / "adapter_metadata.json").read_text(
        encoding="utf-8"
    )
    mapping_text = (output / "mappings" / "nhanes_v3_3_3_mapping.json").read_text(
        encoding="utf-8"
    )
    combined = quality_text + metadata_text + mapping_text
    assert "nhanes::G:" not in combined
    assert "nhanes::H:" not in combined
    assert '"participant_values_written_to_report": false' in quality_text
    assert '"same_calendar_date_records_silently_deduplicated": 0' in quality_text


def test_independent_validator_reconstructs_real_outputs(
    generated_artifacts: tuple[Path, nhanes.BuildResult],
) -> None:
    output, _ = generated_artifacts
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "validate_nhanes_v3_3_3.py"),
        "--workspace-root",
        str(WORKSPACE_ROOT),
        "--output-root",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert result["status"] == "pass"
    assert result["participant_count"] == 2_775
    assert result["positive_row_count"] == 254
    assert result["validation_check_count"] >= 200
    validator_source = (
        REPOSITORY_ROOT / "scripts" / "validate_nhanes_v3_3_3.py"
    ).read_text(encoding="utf-8")
    assert "datasets.adapters.nhanes" not in validator_source
