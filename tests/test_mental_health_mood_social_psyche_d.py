from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import elderly_monitoring.datasets.adapters.psyche_d as psyche_d
from elderly_monitoring.datasets.adapters.psyche_d import (
    ACTIVITY_VOLUME_COLUMN,
    ACTIVITY_VOLUME_MASK_COLUMN,
    EXPECTED_CANONICAL_PARTICIPANTS,
    EXPECTED_CANONICAL_ROWS,
    EXPECTED_END_CATEGORY_COUNTS,
    EXPECTED_POSITIVE_ROWS,
    LABEL_FIELDS,
    PSYCHE_D_SOURCE_FIELDS,
    PsycheDAdapterError,
    TrainingFoldECDF,
    build_canonical_frame,
    build_field_mapping,
    build_psyche_d_artifacts,
    canonical_column_order,
    parse_sample_index,
    validate_frozen_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)
from elderly_monitoring.modules.mental_health.validation.public_datasets.psyche_d import (
    build_psyche_d_artifacts as compatibility_build,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D"
MANIFESTS_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "mental_health"
    / "mood_social"
    / "v3.3.3"
    / "manifests"
)


def _synthetic_source() -> pd.DataFrame:
    index = pd.Index(
        ["family_alpha_2", "family_alpha_10", "person_beta_1", "person_gamma_3"]
    )
    data: dict[str, list[float | None]] = {
        name: [None, None, None, None] for name in PSYCHE_D_SOURCE_FIELDS
    }
    data["steps_awake_mean"] = [100.0, 200.0, 300.0, None]
    data["sleep_asleep_mean_recent"] = [720.0, 360.0, None, 480.0]
    data["sleep_in_bed_mean_recent"] = [800.0, 400.0, None, 600.0]
    data["sleep_ratio_asleep_in_bed_mean_recent"] = [0.9, 0.8, None, 0.8]
    data["sleep_main_start_hour_adj_median"] = [24.0, 30.0, None, 35.0]
    data.update(
        {
            "phq9_score_start": [1.0, 2.0, 3.0, 4.0],
            "phq9_score_end": [2.0, 12.0, 5.0, None],
            "phq9_cat_start": [0.0, 0.0, 1.0, 1.0],
            "phq9_cat_end": [0.0, 2.0, 1.0, None],
        }
    )
    return pd.DataFrame(data, index=index)


def _ecdf_frame() -> pd.DataFrame:
    index = pd.Index(
        [
            "a_train_1",
            "a_train_2",
            "b_train_1",
            "b_train_2",
            "z_held_out_1",
            "z_held_out_2",
            "z_held_out_3",
            "z_held_out_4",
        ]
    )
    data: dict[str, list[float | None]] = {
        name: [None] * 8 for name in PSYCHE_D_SOURCE_FIELDS
    }
    data["steps_awake_mean"] = [1.0, 2.0, 2.0, 4.0, 0.0, 2.0, 5.0, None]
    data.update(
        {
            "phq9_score_start": [1.0] * 8,
            "phq9_score_end": [2.0] * 8,
            "phq9_cat_start": [0.0] * 8,
            "phq9_cat_end": [0.0] * 8,
        }
    )
    return build_canonical_frame(pd.DataFrame(data, index=index))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independently_validate(*, output_root: Path) -> dict[str, object]:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "scripts" / "validate_psyche_d_v3_3_3.py")
    )
    validate = namespace["validate"]
    return validate(workspace_root=WORKSPACE_ROOT, output_root=output_root)


def test_data001_compatibility_entrypoint_reexports_authoritative_adapter() -> None:
    assert compatibility_build is build_psyche_d_artifacts


def test_sample_index_splits_only_the_final_underscore() -> None:
    parsed = parse_sample_index(pd.Index(["participant_with_gap_12", "simple_2"]))

    assert parsed["participant_id"].tolist() == ["participant_with_gap", "simple"]
    assert parsed["nominal_month"].tolist() == [12, 2]
    assert str(parsed["nominal_month"].dtype) == "Int32"


def test_sample_index_rejects_malformed_key_without_exposing_it() -> None:
    private_value = "private-value-without-month"

    with pytest.raises(PsycheDAdapterError) as exc_info:
        parse_sample_index(pd.Index([private_value]))

    assert private_value not in str(exc_info.value)


def test_synthetic_canonical_filters_labels_and_maps_only_strict_targets() -> None:
    source = _synthetic_source()
    canonical = build_canonical_frame(source)

    assert canonical["participant_id"].tolist() == [
        "family_alpha",
        "family_alpha",
        "person_beta",
    ]
    assert canonical["nominal_month"].tolist() == [2, 10, 1]
    assert canonical["binary_target"].tolist() == [0, 1, 0]
    assert canonical["timescale_semantics"].unique().tolist() == [
        "timescale_proxy_transfer"
    ]
    assert canonical["sleep.sleep_duration_norm"].tolist()[:2] == [0.5, 0.25]
    assert canonical["sleep.sleep_efficiency"].tolist()[:2] == [0.9, 0.8]
    assert canonical["sleep.sleep_onset_sin"].iloc[0] == pytest.approx(0.0)
    assert canonical["sleep.sleep_onset_cos"].iloc[0] == pytest.approx(1.0)
    assert canonical["sleep.sleep_onset_sin"].iloc[1] == pytest.approx(1.0)
    assert canonical["feature_mask.sleep.sleep_onset_sin"].tolist() == [1, 1, 0]
    assert canonical[ACTIVITY_VOLUME_COLUMN].isna().all()
    assert canonical[ACTIVITY_VOLUME_MASK_COLUMN].tolist() == [0, 0, 0]
    assert canonical["source__steps_awake_mean"].tolist() == [100.0, 200.0, 300.0]
    assert list(canonical.columns) == list(canonical_column_order())
    assert not any("date" in name for name in canonical.columns)


def test_labels_are_not_source_features_and_invalid_labels_are_rejected() -> None:
    source = _synthetic_source()
    source.loc[source.index[0], "phq9_score_end"] = 1.5

    with pytest.raises(PsycheDAdapterError, match="finite integers"):
        build_canonical_frame(source)

    assert not set(LABEL_FIELDS).intersection(PSYCHE_D_SOURCE_FIELDS)


def test_mapping_covers_every_source_and_mh003_target() -> None:
    mapping = build_field_mapping(SOURCE_ROOT)
    schema = feature_schema_manifest()
    expected_targets = sum(
        len(schema["groups"][group]) for group in schema["group_order"]
    )

    assert [row["source_field"] for row in mapping["source_fields"]] == list(
        PSYCHE_D_SOURCE_FIELDS
    )
    assert len(mapping["target_fields"]) == expected_targets == 54
    assert len(mapping["base_supported_feature_set"]) == 5
    assert mapping["fold_supported_feature_set"] == [ACTIVITY_VOLUME_COLUMN]
    assert mapping["labels"]["input_feature_use"] == "prohibited"
    activity = next(
        row
        for row in mapping["target_fields"]
        if row["canonical_value_column"] == ACTIVITY_VOLUME_COLUMN
    )
    assert activity["mapping_status"] == "fold_derived"
    assert activity["base_canonical_policy"].startswith("null_with_mask_0")


def test_training_fold_ecdf_excludes_held_out_values_and_handles_ties() -> None:
    source = _ecdf_frame()
    before = source.copy(deep=True)
    preprocessor = TrainingFoldECDF.fit(
        source,
        training_participant_ids=["psyche_d::b_train", "psyche_d::a_train"],
        split_id="outer-1",
        canonical_artifact_sha256="a" * 64,
        expected_canonical_frame_sha256=psyche_d.canonical_frame_sha256(source),
    )
    transformed = preprocessor.transform(source)

    assert preprocessor.sorted_training_values == (1.0, 2.0, 2.0, 4.0)
    assert transformed[ACTIVITY_VOLUME_COLUMN].tolist()[:7] == [
        0.25,
        0.75,
        0.75,
        1.0,
        0.0,
        0.75,
        1.0,
    ]
    assert pd.isna(transformed[ACTIVITY_VOLUME_COLUMN].iloc[7])
    assert transformed[ACTIVITY_VOLUME_MASK_COLUMN].tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    pd.testing.assert_frame_equal(source, before)
    assert source[ACTIVITY_VOLUME_COLUMN].isna().all()


def test_training_fold_ecdf_round_trip_is_deterministic_and_private() -> None:
    frame = _ecdf_frame()
    preprocessor = TrainingFoldECDF.fit(
        frame,
        training_participant_ids=["psyche_d::a_train", "psyche_d::b_train"],
        split_id="outer-2",
        canonical_artifact_sha256="a" * 64,
        expected_canonical_frame_sha256=psyche_d.canonical_frame_sha256(frame),
    )
    restored = TrainingFoldECDF.from_dict(preprocessor.to_dict())

    assert restored == preprocessor
    assert restored.to_bytes() == preprocessor.to_bytes()
    assert restored.sha256 == preprocessor.sha256
    assert restored.canonical_artifact_sha256 == "a" * 64
    assert len(restored.canonical_frame_sha256) == 64
    assert b"a_train" not in preprocessor.to_bytes()
    assert b"b_train" not in preprocessor.to_bytes()


@pytest.mark.parametrize("split_id", ["", "   "])
def test_training_fold_ecdf_requires_nonempty_split(split_id: str) -> None:
    frame = _ecdf_frame()
    with pytest.raises(PsycheDAdapterError, match="split_id"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=["psyche_d::a_train"],
            split_id=split_id,
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256=psyche_d.canonical_frame_sha256(frame),
        )


def test_training_fold_ecdf_rejects_participants_absent_from_frame() -> None:
    frame = _ecdf_frame()
    with pytest.raises(PsycheDAdapterError, match="absent"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=["psyche_d::not_present"],
            split_id="outer-3",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256=psyche_d.canonical_frame_sha256(frame),
        )


def test_training_fold_ecdf_rejects_broken_dataset_or_source_binding() -> None:
    missing_dataset = _ecdf_frame()
    missing_dataset.loc[0, "dataset_id"] = pd.NA
    with pytest.raises(PsycheDAdapterError, match="dataset ID"):
        TrainingFoldECDF.fit(
            missing_dataset,
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-4",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256="b" * 64,
        )

    mismatched_source = _ecdf_frame()
    mismatched_source.loc[0, "x_source_value"] = 999.0
    with pytest.raises(PsycheDAdapterError, match="do not match"):
        TrainingFoldECDF.fit(
            mismatched_source,
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-4",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256="b" * 64,
        )


def test_training_fold_ecdf_rejects_incomplete_duplicate_or_unbound_canonical() -> None:
    frame = _ecdf_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(PsycheDAdapterError, match="unique"):
        TrainingFoldECDF.fit(
            duplicate,
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-5",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256="b" * 64,
        )

    with pytest.raises(PsycheDAdapterError, match="complete DATA-002"):
        TrainingFoldECDF.fit(
            frame[["dataset_id", "x_source_value"]],
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-5",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256="b" * 64,
        )

    with pytest.raises(PsycheDAdapterError, match="artifact SHA-256"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-5",
            canonical_artifact_sha256="invalid",
            expected_canonical_frame_sha256=psyche_d.canonical_frame_sha256(frame),
        )

    with pytest.raises(PsycheDAdapterError, match="frame SHA-256"):
        TrainingFoldECDF.fit(
            frame,
            training_participant_ids=["psyche_d::a_train"],
            split_id="outer-5",
            canonical_artifact_sha256="a" * 64,
            expected_canonical_frame_sha256="b" * 64,
        )


def test_frozen_input_validation_rejects_schema_and_collection_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(psyche_d, "MH003_FEATURE_SCHEMA_SHA256", "0" * 64)
    with pytest.raises(PsycheDAdapterError, match="live MH-003"):
        validate_frozen_inputs(WORKSPACE_ROOT, MANIFESTS_ROOT)

    monkeypatch.undo()
    monkeypatch.setattr(psyche_d, "PSYCHE_D_COLLECTION_SHA256", "0" * 64)
    with pytest.raises(PsycheDAdapterError, match="collection"):
        validate_frozen_inputs(WORKSPACE_ROOT, MANIFESTS_ROOT)


def test_frozen_input_validation_rejects_modified_file_manifest(tmp_path: Path) -> None:
    shutil.copy2(MANIFESTS_ROOT / "feature_schema_manifest.json", tmp_path)
    manifest_copy = tmp_path / "dataset_file_manifest.jsonl"
    manifest_copy.write_bytes(
        (MANIFESTS_ROOT / "dataset_file_manifest.jsonl").read_bytes() + b"\n"
    )

    with pytest.raises(PsycheDAdapterError, match="manifest SHA-256"):
        validate_frozen_inputs(WORKSPACE_ROOT, tmp_path)


def test_output_root_must_not_overlap_source_data_tree() -> None:
    binding = validate_frozen_inputs(WORKSPACE_ROOT, MANIFESTS_ROOT)

    with pytest.raises(PsycheDAdapterError, match="must not overlap"):
        psyche_d._validated_output_root(SOURCE_ROOT, binding)


def test_build_revalidates_source_binding_after_all_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = validate_frozen_inputs(WORKSPACE_ROOT, MANIFESTS_ROOT)
    changed = replace(binding, source_collection_sha256="0" * 64)
    observations = iter((binding, changed))
    monkeypatch.setattr(
        psyche_d,
        "validate_frozen_inputs",
        lambda *_args, **_kwargs: next(observations),
    )

    with pytest.raises(PsycheDAdapterError, match="changed while being read"):
        build_psyche_d_artifacts(output_root=tmp_path)

    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_artifact_publication_rolls_back_and_commits_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_paths = (
        "mappings/psyche_d_v3_3_3_mapping.json",
        "psyche_d/adapter_metadata.json",
        "psyche_d/canonical_psyche_d.parquet",
        "psyche_d/data_quality_report.json",
        "psyche_d/artifact_manifest.json",
    )
    old_payloads = {name: f"old:{name}".encode() for name in relative_paths}
    new_payloads = {name: f"new:{name}".encode() for name in relative_paths}
    psyche_d._publish_artifacts(tmp_path, old_payloads, overwrite=False)
    old_snapshot = {
        name: (tmp_path / Path(name)).read_bytes() for name in relative_paths
    }

    real_replace = psyche_d.os.replace

    def fail_during_publish(source: str | Path, target: str | Path) -> None:
        if (
            str(source).endswith(".stage")
            and Path(target).name == "canonical_psyche_d.parquet"
        ):
            raise OSError("injected publish failure")
        real_replace(source, target)

    monkeypatch.setattr(psyche_d.os, "replace", fail_during_publish)
    with pytest.raises(OSError, match="injected"):
        psyche_d._publish_artifacts(tmp_path, new_payloads, overwrite=True)
    assert {
        name: (tmp_path / Path(name)).read_bytes() for name in relative_paths
    } == old_snapshot
    assert not list(tmp_path.rglob("*.stage"))
    assert not list(tmp_path.rglob("*.backup"))

    published: list[str] = []

    def record_publish(source: str | Path, target: str | Path) -> None:
        if str(source).endswith(".stage"):
            published.append(Path(target).as_posix())
        real_replace(source, target)

    monkeypatch.setattr(psyche_d.os, "replace", record_publish)
    psyche_d._publish_artifacts(tmp_path, new_payloads, overwrite=True)
    assert published[-1].endswith("psyche_d/artifact_manifest.json")


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    first = tmp_path_factory.mktemp("psyche-d-first")
    second = tmp_path_factory.mktemp("psyche-d-second")
    first_result = build_psyche_d_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=first,
    )
    second_result = build_psyche_d_artifacts(
        workspace_root=WORKSPACE_ROOT,
        output_root=second,
    )
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    return first, second


def test_real_build_has_frozen_statistics_and_no_prefitted_ecdf(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    canonical_path = first / "psyche_d" / "canonical_psyche_d.parquet"
    canonical = pd.read_parquet(canonical_path)

    assert len(canonical) == EXPECTED_CANONICAL_ROWS
    assert canonical["participant_id"].nunique() == EXPECTED_CANONICAL_PARTICIPANTS
    assert int(canonical["binary_target"].sum()) == EXPECTED_POSITIVE_ROWS
    assert {
        int(key): int(value)
        for key, value in canonical["phq9_cat_end"].value_counts().items()
    } == EXPECTED_END_CATEGORY_COUNTS
    assert canonical[ACTIVITY_VOLUME_COLUMN].isna().all()
    assert int(canonical[ACTIVITY_VOLUME_MASK_COLUMN].sum()) == 0
    assert canonical["timescale_semantics"].unique().tolist() == [
        "timescale_proxy_transfer"
    ]
    assert not any("date" in name for name in canonical.columns)


def test_real_build_column_order_types_and_mapping_are_frozen(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    canonical_path = first / "psyche_d" / "canonical_psyche_d.parquet"
    schema = pq.read_schema(canonical_path)
    mapping = json.loads(
        (first / "mappings" / "psyche_d_v3_3_3_mapping.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema.names == list(canonical_column_order())
    assert str(schema.field("nominal_month").type) == "int32"
    assert str(schema.field("phq9_score_end").type) == "int16"
    assert str(schema.field("phq9_cat_end").type) == "int8"
    assert str(schema.field(ACTIVITY_VOLUME_COLUMN).type) == "double"
    assert str(schema.field(ACTIVITY_VOLUME_MASK_COLUMN).type) == "int8"
    assert len(mapping["source_fields"]) == 27
    assert len(mapping["target_fields"]) == 54


def test_canonical_frame_hash_survives_parquet_round_trip(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    canonical = pd.read_parquet(first / "psyche_d" / "canonical_psyche_d.parquet")
    metadata = json.loads(
        (first / "psyche_d" / "adapter_metadata.json").read_text(encoding="utf-8")
    )

    assert (
        psyche_d.canonical_frame_sha256(canonical) == metadata["canonical_frame_sha256"]
    )


def test_two_real_builds_are_byte_identical_and_hash_manifest_is_correct(
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

    manifest_path = first / "psyche_d" / "artifact_manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in artifact_manifest["artifacts"].items():
        path = first / Path(relative)
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_reports_contain_aggregates_without_participant_values(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts
    report_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(first.rglob("*.json"))
    )

    assert "psyche_d::" not in report_text
    assert '"participant_values_written_to_report": false' in report_text
    assert '"complete_four_label_rows": 10866' in report_text

    quality = json.loads(
        (first / "psyche_d" / "data_quality_report.json").read_text(encoding="utf-8")
    )
    steps_missingness = quality["source_field_missingness"]["steps_awake_mean"]
    assert steps_missingness["full_source_population"]["missing_count"] == 20
    assert steps_missingness["complete_label_population"]["missing_count"] == 9


def test_independent_validator_recomputes_inputs_outputs_and_statistics(
    generated_artifacts: tuple[Path, Path],
) -> None:
    first, _ = generated_artifacts

    result = _independently_validate(output_root=first)

    assert result["status"] == "pass"
    assert result["check_count"] >= 75
    assert result["canonical"]["row_count"] == EXPECTED_CANONICAL_ROWS


def test_independent_validator_rejects_rehashed_canonical_tampering(
    generated_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _ = generated_artifacts
    tampered = tmp_path / "tampered"
    shutil.copytree(first, tampered)
    canonical_path = tampered / "psyche_d" / "canonical_psyche_d.parquet"
    canonical = pd.read_parquet(canonical_path)
    canonical.loc[0, "sleep.sleep_duration_norm"] += 0.01
    canonical.to_parquet(canonical_path, index=False)

    manifest_path = tampered / "psyche_d" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = "psyche_d/canonical_psyche_d.parquet"
    manifest["artifacts"][relative] = {
        "bytes": canonical_path.stat().st_size,
        "sha256": _sha256(canonical_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="independent DATA-002 check failed"):
        _independently_validate(output_root=tampered)
