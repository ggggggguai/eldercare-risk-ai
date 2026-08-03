from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import elderly_monitoring.datasets.adapters.resilient as resilient
from elderly_monitoring.datasets.adapters.resilient import (
    ACTIVITY_VOLUME_COLUMN,
    ACTIVITY_VOLUME_MASK_COLUMN,
    AggregatedSensorWindow,
    EXPECTED_GAD_POSITIVES,
    EXPECTED_GDS_POSITIVES,
    EXPECTED_PARTICIPANTS,
    EXPECTED_PHQ_CATEGORY_COUNTS,
    EXPECTED_POSITIVES,
    RESILIENT_COLLECTION_SHA256,
    SOURCE_FIELDS,
    TrainingFoldECDF,
    aggregate_sensor_window,
    build_canonical_frame,
    build_field_mapping,
    build_resilient_artifacts,
    canonical_column_order,
    canonical_frame_sha256,
    validate_frozen_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)
from elderly_monitoring.modules.mental_health.validation.public_datasets.resilient import (
    build_resilient_artifacts as compatibility_build,
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
PROFILE_FIELDS = {
    "profile_sex",
    "profile_age_group",
    "profile_essential_hypertension",
    "profile_osteoarthritis",
}
SENSOR_SOURCE_FIELDS = tuple(
    name for name in SOURCE_FIELDS if name not in PROFILE_FIELDS
)


def _minimal_frames(
    anchor: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, pd.DataFrame]:
    timestamp = pd.Timestamp(anchor)
    sleep_start = timestamp + pd.Timedelta(hours=22)
    sleep_end = sleep_start + pd.Timedelta(hours=8)
    physio_time = sleep_start + pd.Timedelta(hours=1)
    return {
        "ScanWatch_HR.csv": pd.DataFrame(
            {"HR Timestamp": [timestamp.isoformat()], "Heart Rate": [70.0]}
        ),
        "ScanWatch_Steps.csv": pd.DataFrame(
            {"Steps Timestamp": [timestamp.isoformat()], "Steps": [10.0]}
        ),
        "Sleep_physio.csv": pd.DataFrame(
            {
                "Heart Rate": [65.0],
                "Timestamp": [physio_time.isoformat()],
                "Respiration Rate": [15.0],
                "Snoring": [0.0],
                "SDNN_1": [30.0],
            }
        ),
        "Sleep_state.csv": pd.DataFrame(
            {
                "Start time": [sleep_start.isoformat()],
                "End time": [sleep_end.isoformat()],
                "Sleep state": ["light"],
            }
        ),
    }


def _sensor_source_values(step_value: float | None) -> dict[str, object]:
    values: dict[str, object] = {name: None for name in SENSOR_SOURCE_FIELDS}
    values.update(
        {
            "scanwatch_steps_daily_mean": step_value,
            "scanwatch_steps_valid_days": int(step_value is not None),
            "sleep_valid_nights": 0,
            "scanwatch_hr_valid_days": 0,
            "sleep_physio_valid_nights": 0,
        }
    )
    return values


def _canonical_with_steps(values: list[float | None]) -> pd.DataFrame:
    demographics_rows = []
    windows: dict[str, AggregatedSensorWindow] = {}
    for position, value in enumerate(values, 1):
        participant_id = f"participant_{position:04d}"
        demographics_rows.append(
            {
                "user_id": participant_id,
                "Sex": "Female" if position % 2 else "Male",
                "Age group": "[72, 75]",
                "Essential hypertension": "False",
                "Osteoarthritis": "False",
                "phq_date": "01/01/2026",
                "phq_total": 2,
                "gds_date": "01/01/2026",
                "gds_total": 1,
                "gad_date": "01/01/2026",
                "gad_total": 1,
            }
        )
        windows[participant_id] = AggregatedSensorWindow(
            window_start_date=date(2026, 1, 1),
            window_end_date=date(2026, 1, 14),
            source_values=_sensor_source_values(value),
            quality={},
        )
    return build_canonical_frame(pd.DataFrame(demographics_rows), windows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independently_validate(output_root: Path) -> dict[str, object]:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "scripts" / "validate_resilient_v3_3_3.py")
    )
    return namespace["validate"](
        workspace_root=WORKSPACE_ROOT,
        manifests_root=MANIFESTS_ROOT,
        output_root=output_root,
    )


def test_data001_compatibility_facade_reexports_authoritative_adapter() -> None:
    assert compatibility_build is build_resilient_artifacts


def test_window_uses_london_calendar_across_dst_for_exactly_fourteen_days() -> None:
    frames = _minimal_frames("2026-03-29T00:30:00+00:00")

    window = aggregate_sensor_window(frames)

    assert window.window_start_date == date(2026, 3, 29)
    assert window.window_end_date == date(2026, 4, 11)
    assert (window.window_end_date - window.window_start_date).days + 1 == 14
    assert window.quality["window_calendar_timezone"] == "Europe/London"


def test_steps_are_hourly_increments_and_require_twenty_distinct_hours() -> None:
    frames = _minimal_frames()
    valid_day = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    invalid_day = pd.date_range("2026-01-02", periods=19, freq="h", tz="UTC")
    frames["ScanWatch_Steps.csv"] = pd.DataFrame(
        {
            "Steps Timestamp": [
                value.isoformat() for value in [*valid_day, *invalid_day]
            ],
            "Steps": [10.0] * len(valid_day) + [100.0] * len(invalid_day),
        }
    )

    window = aggregate_sensor_window(frames)

    assert window.source_values["scanwatch_steps_valid_days"] == 1
    assert window.source_values["scanwatch_steps_daily_mean"] == pytest.approx(200.0)
    assert window.quality["ScanWatch_Steps.csv"]["valid_days"] == 1
    assert (
        window.quality["ScanWatch_Steps.csv"]["minimum_distinct_hours_per_valid_day"]
        == 20
    )


def test_cross_midnight_main_sleep_clips_overlap_and_treats_wakeup_as_awake() -> None:
    frames = _minimal_frames()
    frames["Sleep_state.csv"] = pd.DataFrame(
        {
            "Start time": [
                "2026-01-01T21:30:00+00:00",
                "2026-01-01T22:00:00+00:00",
                "2026-01-01T22:30:00+00:00",
                "2026-01-01T23:30:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ],
            "End time": [
                "2026-01-01T22:00:00+00:00",
                "2026-01-01T23:00:00+00:00",
                "2026-01-01T23:30:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-02T06:00:00+00:00",
            ],
            "Sleep state": ["wakeup", "light", "deep", "wakeup", "REM"],
        }
    )
    frames["Sleep_physio.csv"]["Timestamp"] = ["2026-01-01T23:00:00+00:00"]

    window = aggregate_sensor_window(frames)

    assert window.source_values["sleep_valid_nights"] == 1
    assert window.source_values["sleep_duration_minutes_mean"] == pytest.approx(450.0)
    assert window.source_values["time_in_bed_minutes_mean"] == pytest.approx(510.0)
    assert window.source_values["sleep_efficiency"] == pytest.approx(450.0 / 510.0)
    assert window.source_values["sleep_awakening_count_mean"] == pytest.approx(1.0)
    assert window.source_values["sleep_fragmentation"] == pytest.approx(
        60.0 / 510.0 + 1.0 / 7.5
    )
    assert window.source_values["sleep_onset_sin"] == pytest.approx(-0.5)
    assert window.source_values["sleep_onset_cos"] == pytest.approx(3**0.5 / 2.0)


def test_sleep_physio_deduplicates_by_field_and_uses_only_main_sleep() -> None:
    frames = _minimal_frames()
    frames["Sleep_state.csv"] = pd.DataFrame(
        {
            "Start time": ["2026-01-01T22:00:00+00:00"],
            "End time": ["2026-01-02T06:00:00+00:00"],
            "Sleep state": ["light"],
        }
    )
    frames["Sleep_physio.csv"] = pd.DataFrame(
        {
            "Heart Rate": [60.0, 80.0, 90.0, 200.0],
            "Timestamp": [
                "2026-01-01T23:00:00+00:00",
                "2026-01-01T23:00:00+00:00",
                "2026-01-02T01:00:00+00:00",
                "2026-01-02T12:00:00+00:00",
            ],
            "Respiration Rate": [12.0, 16.0, 18.0, 30.0],
            "Snoring": [30.0, 90.0, 0.0, 600.0],
            "SDNN_1": [20.0, 40.0, 50.0, 999.0],
        }
    )

    window = aggregate_sensor_window(frames)

    assert window.source_values["sleep_physio_valid_nights"] == 1
    assert window.source_values["sleep_hr_mean_bpm"] == pytest.approx(80.0)
    assert window.source_values["sleep_hr_min_bpm"] == pytest.approx(70.0)
    assert window.source_values["sleep_hr_sd_bpm"] == pytest.approx(10.0)
    assert window.source_values["sleep_hrv_sdnn_ms"] == pytest.approx(40.0)
    assert window.source_values["sleep_respiration_rate_mean_bpm"] == pytest.approx(
        16.0
    )
    assert window.source_values["sleep_respiration_rate_sd_bpm"] == pytest.approx(2.0)
    assert window.source_values["sleep_snoring_minutes_mean"] == pytest.approx(1.0)
    quality = window.quality["Sleep_physio.csv"]
    assert quality["duplicate_timestamp_rows_removed"] == 1
    assert quality["aligned_rows"] == 2


def test_labels_map_without_leaking_questionnaire_or_ace_fields() -> None:
    participants = [f"participant_{position:04d}" for position in range(1, 4)]
    demographics = pd.DataFrame(
        {
            "user_id": participants,
            "Sex": ["Female", "Male", None],
            "Age group": ["[72, 75]", "[76, 87]", "[88, 99]"],
            "Essential hypertension": ["True", "False", None],
            "Osteoarthritis": ["False", "True", None],
            "phq_date": ["01/01/2026"] * 3,
            "phq_total": [9, 10, 20],
            "gds_date": ["01/01/2026"] * 3,
            "gds_total": [4, 5, 6],
            "gad_date": ["01/01/2026"] * 3,
            "gad_total": [9, 10, 11],
            "phq1": [3, 3, 3],
            "ACE_total": [4, 4, 4],
        }
    )
    windows = {
        participant_id: AggregatedSensorWindow(
            window_start_date=date(2026, 1, 1),
            window_end_date=date(2026, 1, 14),
            source_values=_sensor_source_values(float(position * 100)),
            quality={},
        )
        for position, participant_id in enumerate(participants, 1)
    }

    canonical = build_canonical_frame(demographics, windows)

    assert canonical["phq9_category"].tolist() == [1, 2, 4]
    assert canonical["binary_target"].tolist() == [0, 1, 1]
    assert canonical["gds15_binary_target"].tolist() == [0, 1, 1]
    assert canonical["gad7_binary_target"].tolist() == [0, 1, 1]
    assert canonical["social_context.age_group"].tolist() == [
        "70_79",
        pd.NA,
        "80_plus",
    ]
    assert "phq1" not in canonical.columns
    assert "ACE_total" not in canonical.columns
    assert canonical[ACTIVITY_VOLUME_COLUMN].isna().all()
    assert canonical[ACTIVITY_VOLUME_MASK_COLUMN].tolist() == [0, 0, 0]
    assert canonical["social_context.chronic_disease_count"].isna().all()
    assert canonical["feature_mask.social_context.chronic_disease_count"].tolist() == [
        0,
        0,
        0,
    ]


def test_mapping_is_complete_strict_and_defers_formal_ecdf() -> None:
    mapping = build_field_mapping()
    schema = feature_schema_manifest()
    target_count = sum(len(schema["groups"][group]) for group in schema["group_order"])

    assert len(mapping["source_fields"]) == len(SOURCE_FIELDS) == 35
    assert len(mapping["target_fields"]) == target_count == 54
    assert len(mapping["base_supported_feature_set"]) == 24
    assert mapping["fold_supported_feature_set"] == [ACTIVITY_VOLUME_COLUMN]
    assert mapping["ecdf_contract"]["formal_instances_before_data007"] is False
    assert mapping["labels"]["input_feature_use"] == "prohibited"
    assert mapping["sensor_contract"]["os_metadata_parsed"] is False
    assert mapping["sensor_contract"]["Sleep_state.csv"]["wakeup_semantics"] == (
        "awake state"
    )
    chronic = next(
        row
        for row in mapping["target_fields"]
        if row["canonical_value_column"] == "social_context.chronic_disease_count"
    )
    assert chronic["mapping_status"] == "unsupported"
    scan_hr = next(
        row
        for row in mapping["source_fields"]
        if row["source_field"] == "scanwatch_hr_mean_bpm"
    )
    assert scan_hr["mapping_status"] == "audit_only"


def test_training_fold_ecdf_requires_explicit_split_and_handles_ties() -> None:
    frame = _canonical_with_steps([1.0, 2.0, 2.0, 4.0, 5.0, None])
    before = frame.copy(deep=True)
    training_ids = [
        f"resilient::participant_{position:04d}" for position in range(1, 5)
    ]
    fitted = TrainingFoldECDF.fit(
        frame,
        training_participant_ids=training_ids,
        split_id="outer-1",
        canonical_artifact_sha256="a" * 64,
        expected_canonical_frame_sha256=canonical_frame_sha256(frame),
    )
    transformed = fitted.transform(frame)

    assert fitted.sorted_training_values == (1.0, 2.0, 2.0, 4.0)
    assert transformed[ACTIVITY_VOLUME_COLUMN].tolist()[:5] == [
        0.25,
        0.75,
        0.75,
        1.0,
        1.0,
    ]
    assert pd.isna(transformed[ACTIVITY_VOLUME_COLUMN].iloc[5])
    assert transformed[ACTIVITY_VOLUME_MASK_COLUMN].tolist() == [1, 1, 1, 1, 1, 0]
    pd.testing.assert_frame_equal(frame, before)

    restored = TrainingFoldECDF.from_dict(fitted.to_dict())
    assert restored == fitted
    assert restored.to_bytes() == fitted.to_bytes()
    assert b"participant_0001" not in fitted.to_bytes()
    assert b"participant_0004" not in fitted.to_bytes()

    with pytest.raises(resilient.ResilientAdapterError, match="split_id"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=training_ids,
            split_id="",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256=canonical_frame_sha256(frame),
        )
    with pytest.raises(resilient.ResilientAdapterError, match="absent"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=["resilient::participant_9999"],
            split_id="outer-2",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256=canonical_frame_sha256(frame),
        )


def test_frozen_input_bindings_match_authorized_hashes() -> None:
    binding = validate_frozen_inputs(WORKSPACE_ROOT, MANIFESTS_ROOT)

    assert binding.file_manifest_sha256 == (
        "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
    )
    assert binding.feature_schema_sha256 == (
        "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
    )
    assert binding.source_collection_sha256 == RESILIENT_COLLECTION_SHA256


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    first = tmp_path_factory.mktemp("resilient-first")
    second = tmp_path_factory.mktemp("resilient-second")
    first_result = build_resilient_artifacts(
        workspace_root=WORKSPACE_ROOT,
        manifests_root=MANIFESTS_ROOT,
        output_root=first,
    )
    second_result = build_resilient_artifacts(
        workspace_root=WORKSPACE_ROOT,
        manifests_root=MANIFESTS_ROOT,
        output_root=second,
    )
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    return first, second


def test_real_build_has_frozen_statistics_and_no_formal_ecdf(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    canonical = pd.read_parquet(first / "resilient" / "canonical_resilient.parquet")

    assert len(canonical) == EXPECTED_PARTICIPANTS == 73
    assert canonical["participant_id"].nunique() == EXPECTED_PARTICIPANTS
    assert int(canonical["binary_target"].sum()) == EXPECTED_POSITIVES
    assert int(canonical["gds15_binary_target"].sum()) == EXPECTED_GDS_POSITIVES
    assert int(canonical["gad7_binary_target"].sum()) == EXPECTED_GAD_POSITIVES
    assert {
        int(key): int(value)
        for key, value in canonical["phq9_category"].value_counts().items()
    } == EXPECTED_PHQ_CATEGORY_COUNTS
    assert list(canonical.columns) == list(canonical_column_order())
    assert canonical[ACTIVITY_VOLUME_COLUMN].isna().all()
    assert int(canonical[ACTIVITY_VOLUME_MASK_COLUMN].sum()) == 0
    assert not list((first / "resilient").glob("*ecdf*"))


def test_two_real_builds_are_byte_identical_and_manifest_is_correct(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, second = generated_artifacts
    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()

    manifest = json.loads(
        (first / "resilient" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["artifacts"].items():
        path = first / Path(relative)
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_reports_are_aggregate_only_without_participant_values(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    report_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(first.rglob("*.json"))
    )

    assert "resilient::participant_" not in report_text
    assert '"source_participant_values_written_to_report": false' in report_text
    assert '"formal_ecdf_instance_created": false' in report_text


def test_independent_validator_recomputes_source_and_all_artifacts(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts

    result = _independently_validate(first)

    assert result["status"] == "pass"
    assert result["check_count"] >= 1400
    assert result["canonical"] == {
        "row_count": 73,
        "participant_count": 73,
        "phq9_positive_count": 10,
        "gds15_positive_count": 23,
        "gad7_positive_count": 6,
        "phq9_category_counts": {"0": 37, "1": 26, "2": 5, "3": 4, "4": 1},
    }


def test_independent_validator_rejects_rehashed_canonical_tampering(
    generated_artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    first, _ = generated_artifacts
    tampered = tmp_path / "tampered"
    shutil.copytree(first, tampered)
    canonical_path = tampered / "resilient" / "canonical_resilient.parquet"
    canonical = pd.read_parquet(canonical_path)
    canonical.loc[0, "sleep.sleep_duration_norm"] += 0.01
    canonical.to_parquet(canonical_path, index=False)

    manifest_path = tampered / "resilient" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = "resilient/canonical_resilient.parquet"
    manifest["artifacts"][relative] = {
        "bytes": canonical_path.stat().st_size,
        "sha256": _sha256(canonical_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="independent DATA-003 check failed"):
        _independently_validate(tampered)
