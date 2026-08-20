"""Deterministic participant-level nested splits for mood-social V3.3.3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    feature_schema_manifest,
)


MANIFEST_VERSION = "mood-social-nested-participant-split-manifest-v1"
ASSIGNMENT_ALGORITHM_VERSION = "mood-social-participant-nested-stratified-hash-v1"
SPLIT_ID = "mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1"
RANDOM_SEED = 20260728
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
PROCESSED_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
FEATURE_SCHEMA_RELATIVE = (
    PROCESSED_RELATIVE / "manifests" / "feature_schema_manifest.json"
)
DEFAULT_SPLIT_RELATIVE = PROCESSED_RELATIVE / "splits" / "split_manifest.json"
_WRITE_LOCK_NAME = ".split_manifest.write.lock"


class SplitInputError(ValueError):
    """Raised when a frozen input or participant contract is invalid."""


class SplitWriteError(RuntimeError):
    """Raised when the split artifact cannot be written safely."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    directory_name: str
    canonical_name: str
    mapping_name: str
    artifact_manifest_sha256: str
    canonical_sha256: str
    mapping_sha256: str
    frame_sha256: str
    row_count: int
    participant_count: int
    positive_row_count: int
    processed_relative: Path = PROCESSED_RELATIVE

    @property
    def canonical_relative_path(self) -> Path:
        return self.processed_relative / self.directory_name / self.canonical_name

    @property
    def artifact_manifest_relative_path(self) -> Path:
        return self.processed_relative / self.directory_name / "artifact_manifest.json"

    @property
    def adapter_metadata_relative_path(self) -> Path:
        return self.processed_relative / self.directory_name / "adapter_metadata.json"

    @property
    def mapping_relative_path(self) -> Path:
        return self.processed_relative / "mappings" / self.mapping_name


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="psyche_d",
        directory_name="psyche_d",
        canonical_name="canonical_psyche_d.parquet",
        mapping_name="psyche_d_v3_3_3_mapping.json",
        artifact_manifest_sha256=(
            "bf4c7e871b48eb85bd5be0cc9a6df4af6d4c638eddea520babcfe454927b9e24"
        ),
        canonical_sha256=(
            "c913768ea03dd509f1036e85bd66aa7560066023847306b5abddad950cc30b33"
        ),
        mapping_sha256=(
            "99c3a684c9dd902d0ef2a4f2a198abec3708acbf087029f2d68e26dff7082a74"
        ),
        frame_sha256=(
            "0ec9d9e42f2a5e068053ae69f4a66d1064819e4e7c914d04f74981cfd822cfa8"
        ),
        row_count=10_866,
        participant_count=4_036,
        positive_row_count=3_444,
    ),
    DatasetSpec(
        dataset_id="resilient",
        directory_name="resilient",
        canonical_name="canonical_resilient.parquet",
        mapping_name="resilient_v3_3_3_mapping.json",
        artifact_manifest_sha256=(
            "08f6bbdd6e0e09446b8a0e69e20bc90b864600acbcb4aa8cbf25fb37cce27156"
        ),
        canonical_sha256=(
            "fa2ddd3b79ccd3a7d3cb9073512f9a3813e26b48abd5621cd70cae53d1039229"
        ),
        mapping_sha256=(
            "8c878c33ee719f600cb08628f023d2a897e030e0eebe4633286999df19519b91"
        ),
        frame_sha256=(
            "5df4077cdab59f1e3a51c468cca32c1def88fb4c490bfa295eb1d5f5564bca7d"
        ),
        row_count=73,
        participant_count=73,
        positive_row_count=10,
    ),
    DatasetSpec(
        dataset_id="nhanes",
        directory_name="nhanes",
        canonical_name="canonical_nhanes.parquet",
        mapping_name="nhanes_v3_3_3_mapping.json",
        artifact_manifest_sha256=(
            "477a47965068f27766dbb25d470abbf5db0cb5acd63a339be9a949797907963e"
        ),
        canonical_sha256=(
            "e6438d99f99b12f3607e1ee6482a48c6cc9235b04028f0184b5b96bd4320654f"
        ),
        mapping_sha256=(
            "78bafb92230949efa5ea09378f6133965b75cfac73da32f99dd9382dbfcd8802"
        ),
        frame_sha256=(
            "68cc1d90d3c028dd5c09f749d9d0cea432706c2cd71bd3173c72345e670bee63"
        ),
        row_count=2_775,
        participant_count=2_775,
        positive_row_count=254,
    ),
    DatasetSpec(
        dataset_id="shenzhen_elderly",
        directory_name="shenzhen",
        canonical_name="canonical_shenzhen.parquet",
        mapping_name="shenzhen_v3_3_3_mapping.json",
        artifact_manifest_sha256=(
            "da809d71393a6993b0c723a2ee27d8b76fb0fe605750e31aa8e356b397122e35"
        ),
        canonical_sha256=(
            "85ea64ba940e671427a3a9e9ea7a515a90fa57c88520251b3b9964759ebfa054"
        ),
        mapping_sha256=(
            "9e7c087b2906465abdb0d179838efef42abcb353b617e7261570601a14d7df90"
        ),
        frame_sha256=(
            "d5cdc2f585feb1e3573c09e092a56246439fa930d245d07191a02d3f42e7e1ac"
        ),
        row_count=5_327,
        participant_count=5_327,
        positive_row_count=186,
    ),
    DatasetSpec(
        dataset_id="nhanes_ssq_2005_2008",
        directory_name="nhanes_ssq_2005_2008",
        canonical_name="canonical_nhanes_ssq_2005_2008.parquet",
        mapping_name="nhanes_ssq_2005_2008_v3_3_3_mapping.json",
        artifact_manifest_sha256=(
            "e610fea41e25b70b656aadfcbef21a9b8a9445ba419b267e4d7cb10c431e3f9e"
        ),
        canonical_sha256=(
            "b0da21d9677d3cf59bdec347eedb3a9c34324374003e27c014b1b46f91db52ce"
        ),
        mapping_sha256=(
            "945bbd504cb4d2578c40b87ec1e039770ce45f1d02154bbc7bc6903cde486aea"
        ),
        frame_sha256=(
            "0fdb9c743f43d6fa2b5e8397ad1aa54d3f2c6e6e7ab09c4640055ca5923e2610"
        ),
        row_count=3_150,
        participant_count=3_150,
        positive_row_count=185,
    ),
)


@dataclass(frozen=True, slots=True)
class ParticipantUnit:
    dataset_id: str
    global_participant_id: str
    row_count: int
    positive_row_count: int

    def __post_init__(self) -> None:
        if (
            not self.dataset_id
            or "\0" in self.dataset_id
            or not self.global_participant_id
            or "\0" in self.global_participant_id
        ):
            raise SplitInputError("participant key contains an invalid value")
        if not self.global_participant_id.startswith(f"{self.dataset_id}::"):
            raise SplitInputError("global participant key is not namespaced")
        if self.global_participant_id == f"{self.dataset_id}::":
            raise SplitInputError("global participant key has an empty local key")
        if self.row_count <= 0:
            raise SplitInputError("participant row count must be positive")
        if not 0 <= self.positive_row_count <= self.row_count:
            raise SplitInputError("participant target counts are invalid")

    @property
    def stratum(self) -> tuple[str, int, int]:
        return (
            self.dataset_id,
            self.row_count,
            self.positive_row_count,
        )


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SplitInputError("a frozen input file is not readable") from exc
    return digest.hexdigest()


def _digest_parts(*parts: object) -> bytes:
    encoded: list[bytes] = []
    for part in parts:
        value = str(part)
        if "\0" in value:
            raise SplitInputError("split hash input contains a NUL character")
        encoded.append(value.encode("utf-8"))
    return hashlib.sha256(b"\0".join(encoded)).digest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitInputError("a frozen JSON input is not readable") from exc


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise SplitInputError("artifact manifest contains an unsafe path")
    candidate = (root / candidate_relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SplitInputError("artifact manifest path escapes the processed root")
    return candidate


def assign_participant_folds(
    participants: Iterable[ParticipantUnit],
    *,
    scope: str,
    seed: int,
    fold_count: int,
) -> dict[str, int]:
    """Assign exact participant-label strata by seeded hash round-robin."""

    if not scope or "\0" in scope:
        raise SplitInputError("split scope must be a non-empty safe string")
    if fold_count < 2:
        raise SplitInputError("fold count must be at least two")
    materialized = list(participants)
    by_key: dict[str, ParticipantUnit] = {}
    strata: dict[tuple[str, int, int], list[ParticipantUnit]] = defaultdict(list)
    for participant in materialized:
        if participant.global_participant_id in by_key:
            raise SplitInputError("participant keys are not unique")
        by_key[participant.global_participant_id] = participant
        strata[participant.stratum].append(participant)
    if not materialized:
        raise SplitInputError("no participants are available for splitting")

    assignments: dict[str, int] = {}
    for stratum in sorted(
        strata,
        key=lambda value: (
            value[0].encode("utf-8"),
            value[1],
            value[2],
        ),
    ):
        members = strata[stratum]
        if len(members) < fold_count:
            raise SplitInputError(
                "a stratification cell has fewer participants than folds"
            )
        ordered = sorted(
            members,
            key=lambda participant: (
                _digest_parts(
                    ASSIGNMENT_ALGORITHM_VERSION,
                    scope,
                    seed,
                    "participant",
                    participant.global_participant_id,
                ),
                participant.global_participant_id.encode("utf-8"),
            ),
        )
        offset_digest = _digest_parts(
            ASSIGNMENT_ALGORITHM_VERSION,
            scope,
            seed,
            "stratum",
            *stratum,
        )
        offset = int.from_bytes(offset_digest[:8], "big") % fold_count
        for rank, participant in enumerate(ordered):
            assignments[participant.global_participant_id] = (
                offset + rank
            ) % fold_count

    if set(assignments) != set(by_key):
        raise SplitInputError("participant assignment coverage is incomplete")
    return assignments


def build_nested_assignments(
    participants: Iterable[ParticipantUnit],
    *,
    seed: int = RANDOM_SEED,
    outer_fold_count: int = OUTER_FOLD_COUNT,
    inner_fold_count: int = INNER_FOLD_COUNT,
) -> list[dict[str, Any]]:
    """Build one outer fold and four applicable inner-fold assignments per person."""

    materialized = sorted(
        list(participants),
        key=lambda value: value.global_participant_id.encode("utf-8"),
    )
    outer = assign_participant_folds(
        materialized,
        scope="outer",
        seed=seed,
        fold_count=outer_fold_count,
    )
    inner_by_outer: dict[int, dict[str, int]] = {}
    for outer_fold in range(outer_fold_count):
        outer_train = [
            participant
            for participant in materialized
            if outer[participant.global_participant_id] != outer_fold
        ]
        inner_by_outer[outer_fold] = assign_participant_folds(
            outer_train,
            scope=f"inner:outer={outer_fold}",
            seed=seed,
            fold_count=inner_fold_count,
        )

    output: list[dict[str, Any]] = []
    for participant in materialized:
        key = participant.global_participant_id
        outer_fold = outer[key]
        inner = {
            str(candidate_outer): (
                None
                if candidate_outer == outer_fold
                else inner_by_outer[candidate_outer][key]
            )
            for candidate_outer in range(outer_fold_count)
        }
        output.append(
            {
                "dataset_id": participant.dataset_id,
                "global_participant_id": key,
                "inner_validation_fold_by_outer_fold": inner,
                "outer_fold": outer_fold,
            }
        )
    return output


def _validate_schema_binding(
    repository_root: Path,
    feature_schema_relative: Path = FEATURE_SCHEMA_RELATIVE,
) -> dict[str, str]:
    snapshot = repository_root / feature_schema_relative
    if _sha256_file(snapshot) != FEATURE_SCHEMA_SHA256:
        raise SplitInputError("frozen feature schema hash changed")
    stored = _load_json(snapshot)
    live = feature_schema_manifest()
    if stored != live:
        raise SplitInputError("stored and live feature schemas differ")
    if str(live.get("schema_version")) != FEATURE_SCHEMA_VERSION:
        raise SplitInputError("live feature schema version changed")
    if _sha256_bytes(_canonical_json_bytes(live)) != FEATURE_SCHEMA_SHA256:
        raise SplitInputError("live feature schema hash changed")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "snapshot_relative_path": feature_schema_relative.as_posix(),
        "snapshot_sha256": FEATURE_SCHEMA_SHA256,
    }


def _validate_artifact_manifest(
    repository_root: Path,
    processed_root: Path,
    spec: DatasetSpec,
) -> tuple[list[ParticipantUnit], dict[str, Any]]:
    artifact_manifest_path = repository_root / spec.artifact_manifest_relative_path
    if _sha256_file(artifact_manifest_path) != spec.artifact_manifest_sha256:
        raise SplitInputError("frozen input hash changed: artifact manifest")
    artifact_manifest = _load_json(artifact_manifest_path)
    if (
        artifact_manifest.get("dataset_id") != spec.dataset_id
        or artifact_manifest.get("complete") is not True
        or artifact_manifest.get("artifact_count") != 4
    ):
        raise SplitInputError("artifact manifest contract changed")

    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 4:
        raise SplitInputError("artifact manifest file set changed")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise SplitInputError("artifact manifest entry is malformed")
        path = _safe_artifact_path(processed_root, relative)
        try:
            byte_count = path.stat().st_size
        except OSError as exc:
            raise SplitInputError("a bound artifact is missing") from exc
        if byte_count != expected.get("bytes") or _sha256_file(path) != expected.get(
            "sha256"
        ):
            raise SplitInputError("frozen input hash or size changed: bound artifact")

    canonical_path = repository_root / spec.canonical_relative_path
    mapping_path = repository_root / spec.mapping_relative_path
    metadata_path = repository_root / spec.adapter_metadata_relative_path
    if _sha256_file(canonical_path) != spec.canonical_sha256:
        raise SplitInputError("frozen input hash changed: canonical")
    if _sha256_file(mapping_path) != spec.mapping_sha256:
        raise SplitInputError("frozen input hash changed: mapping")

    canonical_artifact_key = f"{spec.directory_name}/{spec.canonical_name}"
    mapping_artifact_key = f"mappings/{spec.mapping_name}"
    metadata_artifact_key = f"{spec.directory_name}/adapter_metadata.json"
    if (
        artifacts.get(canonical_artifact_key, {}).get("sha256") != spec.canonical_sha256
        or artifacts.get(mapping_artifact_key, {}).get("sha256") != spec.mapping_sha256
        or metadata_artifact_key not in artifacts
    ):
        raise SplitInputError("artifact manifest does not bind required inputs")

    metadata = _load_json(metadata_path)
    if (
        metadata.get("dataset_id") != spec.dataset_id
        or metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or metadata.get("canonical_frame_sha256") != spec.frame_sha256
        or metadata.get("canonical_row_count") != spec.row_count
        or metadata.get("canonical_participant_count") != spec.participant_count
    ):
        raise SplitInputError("adapter metadata binding changed")
    input_binding = artifact_manifest.get("input_binding")
    if (
        not isinstance(input_binding, dict)
        or input_binding.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
        or input_binding.get("all_bindings_validated") is not True
    ):
        raise SplitInputError("source input binding is incomplete")

    try:
        frame = pd.read_parquet(
            canonical_path,
            columns=[
                "dataset_id",
                "global_participant_id",
                "participant_id",
                "binary_target",
            ],
        )
    except (OSError, ValueError, TypeError) as exc:
        raise SplitInputError("canonical participant columns are not readable") from exc
    if len(frame) != spec.row_count:
        raise SplitInputError("canonical row count changed")
    if (
        frame[
            [
                "dataset_id",
                "global_participant_id",
                "participant_id",
                "binary_target",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise SplitInputError("canonical split columns contain missing values")
    if set(frame["dataset_id"].astype(str)) != {spec.dataset_id}:
        raise SplitInputError("canonical dataset namespace changed")

    participant_ids = frame["participant_id"].astype(str)
    global_ids = frame["global_participant_id"].astype(str)
    expected_global_ids = spec.dataset_id + "::" + participant_ids
    if not global_ids.equals(expected_global_ids):
        raise SplitInputError("canonical global participant keys changed")
    try:
        targets = pd.to_numeric(frame["binary_target"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SplitInputError("canonical binary targets are not numeric") from exc
    if not targets.isin([0, 1]).all():
        raise SplitInputError("canonical binary targets are not binary")

    split_frame = pd.DataFrame(
        {
            "dataset_id": spec.dataset_id,
            "global_participant_id": global_ids,
            "binary_target": targets.astype("int64"),
        }
    )
    grouped = (
        split_frame.groupby(
            ["dataset_id", "global_participant_id"],
            sort=True,
            observed=True,
        )["binary_target"]
        .agg(row_count="size", positive_row_count="sum")
        .reset_index()
    )
    if (
        len(grouped) != spec.participant_count
        or int(targets.sum()) != spec.positive_row_count
    ):
        raise SplitInputError("canonical participant or target counts changed")
    participants = [
        ParticipantUnit(
            dataset_id=str(row.dataset_id),
            global_participant_id=str(row.global_participant_id),
            row_count=int(row.row_count),
            positive_row_count=int(row.positive_row_count),
        )
        for row in grouped.itertuples(index=False)
    ]

    metadata_sha256 = str(artifacts[metadata_artifact_key]["sha256"])
    binding = {
        "adapter_metadata_relative_path": (
            spec.adapter_metadata_relative_path.as_posix()
        ),
        "adapter_metadata_sha256": metadata_sha256,
        "artifact_manifest_relative_path": (
            spec.artifact_manifest_relative_path.as_posix()
        ),
        "artifact_manifest_sha256": spec.artifact_manifest_sha256,
        "canonical_column_order_version": metadata.get(
            "canonical_column_order_version"
        ),
        "canonical_frame_sha256": spec.frame_sha256,
        "canonical_relative_path": spec.canonical_relative_path.as_posix(),
        "canonical_sha256": spec.canonical_sha256,
        "dataset_id": spec.dataset_id,
        "mapping_relative_path": spec.mapping_relative_path.as_posix(),
        "mapping_sha256": spec.mapping_sha256,
        "negative_row_count": spec.row_count - spec.positive_row_count,
        "participant_count": spec.participant_count,
        "positive_row_count": spec.positive_row_count,
        "row_count": spec.row_count,
        "source_input_binding": input_binding,
    }
    return participants, binding


def _count_partition(
    participants: Sequence[ParticipantUnit],
    dataset_specs: Sequence[DatasetSpec] = DATASET_SPECS,
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, int]] = {}
    for spec in dataset_specs:
        selected = [
            participant
            for participant in participants
            if participant.dataset_id == spec.dataset_id
        ]
        if not selected:
            continue
        row_count = sum(value.row_count for value in selected)
        positive = sum(value.positive_row_count for value in selected)
        by_dataset[spec.dataset_id] = {
            "negative_row_count": row_count - positive,
            "participant_count": len(selected),
            "positive_row_count": positive,
            "row_count": row_count,
        }
    row_count = sum(value.row_count for value in participants)
    positive = sum(value.positive_row_count for value in participants)
    return {
        "by_dataset": by_dataset,
        "negative_row_count": row_count - positive,
        "participant_count": len(participants),
        "positive_row_count": positive,
        "row_count": row_count,
    }


def _build_fold_summaries(
    participants: Sequence[ParticipantUnit],
    assignments: Sequence[Mapping[str, Any]],
    dataset_specs: Sequence[DatasetSpec] = DATASET_SPECS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_by_key = {str(row["global_participant_id"]): row for row in assignments}
    outer_summaries: list[dict[str, Any]] = []
    inner_summaries: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        outer_test = [
            participant
            for participant in participants
            if assignment_by_key[participant.global_participant_id]["outer_fold"]
            == outer_fold
        ]
        outer_train = [
            participant
            for participant in participants
            if assignment_by_key[participant.global_participant_id]["outer_fold"]
            != outer_fold
        ]
        outer_summaries.append(
            {
                "outer_fold": outer_fold,
                "test": _count_partition(outer_test, dataset_specs),
                "train": _count_partition(outer_train, dataset_specs),
            }
        )
        inner_folds: list[dict[str, Any]] = []
        for inner_fold in range(INNER_FOLD_COUNT):
            validation = [
                participant
                for participant in outer_train
                if assignment_by_key[participant.global_participant_id][
                    "inner_validation_fold_by_outer_fold"
                ][str(outer_fold)]
                == inner_fold
            ]
            validation_keys = {
                participant.global_participant_id for participant in validation
            }
            inner_train = [
                participant
                for participant in outer_train
                if participant.global_participant_id not in validation_keys
            ]
            inner_folds.append(
                {
                    "inner_fold": inner_fold,
                    "train": _count_partition(inner_train, dataset_specs),
                    "validation": _count_partition(validation, dataset_specs),
                }
            )
        inner_summaries.append(
            {
                "folds": inner_folds,
                "outer_fold": outer_fold,
                "outer_train": _count_partition(outer_train, dataset_specs),
            }
        )
    return outer_summaries, inner_summaries


def _validate_assignment_balance(
    participants: Sequence[ParticipantUnit],
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    assignment_by_key = {str(row["global_participant_id"]): row for row in assignments}

    def validate_scope(
        eligible: Sequence[ParticipantUnit],
        fold_by_key: Mapping[str, int],
        fold_count: int,
    ) -> None:
        strata: dict[tuple[str, int, int], list[str]] = defaultdict(list)
        for participant in eligible:
            strata[participant.stratum].append(participant.global_participant_id)
        for keys in strata.values():
            counts = [
                sum(fold_by_key[key] == fold for key in keys)
                for fold in range(fold_count)
            ]
            if max(counts) - min(counts) > 1:
                raise SplitInputError("stratified fold balance guarantee failed")

    validate_scope(
        participants,
        {key: int(row["outer_fold"]) for key, row in assignment_by_key.items()},
        OUTER_FOLD_COUNT,
    )
    for outer_fold in range(OUTER_FOLD_COUNT):
        eligible = [
            participant
            for participant in participants
            if int(assignment_by_key[participant.global_participant_id]["outer_fold"])
            != outer_fold
        ]
        validate_scope(
            eligible,
            {
                participant.global_participant_id: int(
                    assignment_by_key[participant.global_participant_id][
                        "inner_validation_fold_by_outer_fold"
                    ][str(outer_fold)]
                )
                for participant in eligible
            },
            INNER_FOLD_COUNT,
        )


def build_split_manifest(
    *,
    repository_root: Path | None = None,
    dataset_specs: Sequence[DatasetSpec] = DATASET_SPECS,
    processed_relative: Path = PROCESSED_RELATIVE,
    feature_schema_relative: Path = FEATURE_SCHEMA_RELATIVE,
    split_id: str = SPLIT_ID,
    frozen_on: str = "2026-07-31",
    task_id: str = "DATA-007",
) -> dict[str, Any]:
    """Read and bind the five frozen canonicals, then build nested assignments."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[5]
    )
    processed_root = root / processed_relative
    if not (root / "pyproject.toml").is_file() or not processed_root.is_dir():
        raise SplitInputError("algorithm repository root is invalid")

    if any(spec.processed_relative != processed_relative for spec in dataset_specs):
        raise SplitInputError("dataset processed roots do not match split scope")
    schema_binding = _validate_schema_binding(root, feature_schema_relative)
    participants: list[ParticipantUnit] = []
    input_bindings: list[dict[str, Any]] = []
    for spec in dataset_specs:
        dataset_participants, binding = _validate_artifact_manifest(
            root,
            processed_root,
            spec,
        )
        participants.extend(dataset_participants)
        input_bindings.append(binding)
    keys = [participant.global_participant_id for participant in participants]
    if len(keys) != len(set(keys)):
        raise SplitInputError("global participant keys overlap across datasets")

    assignments = build_nested_assignments(participants)
    _validate_assignment_balance(participants, assignments)
    outer_summaries, inner_summaries = _build_fold_summaries(
        participants,
        assignments,
        dataset_specs,
    )
    row_count = sum(participant.row_count for participant in participants)
    positive = sum(participant.positive_row_count for participant in participants)
    totals = {
        "dataset_count": len(dataset_specs),
        "negative_row_count": row_count - positive,
        "participant_count": len(participants),
        "positive_row_count": positive,
        "row_count": row_count,
    }
    protocol = {
        "assignment_algorithm_version": ASSIGNMENT_ALGORITHM_VERSION,
        "balance_guarantee": (
            "participant counts differ by at most one within each "
            "stratification cell and split scope"
        ),
        "fold_indexing": "zero_based",
        "global_participant_key_format": "dataset_id::participant_id",
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_role": (
            "cross_fitting_validation_assignment_within_each_outer_training_set"
        ),
        "inner_scope_template": "inner:outer={outer_fold}",
        "label_usage": (
            "binary_target is used only for participant-level stratification "
            "and aggregate fold statistics"
        ),
        "outer_fold_count": OUTER_FOLD_COUNT,
        "outer_test_role": "evaluation_only",
        "participant_key_field": "global_participant_id",
        "participant_ordering": (
            "seeded SHA-256 digest with UTF-8 global key tie-break"
        ),
        "preprocessing_fit_scope": {
            "categorical_encoding": "corresponding_training_fold_only",
            "ecdf": "corresponding_training_fold_only",
            "feature_selection": "corresponding_training_fold_only",
            "fusion": "corresponding_training_fold_only",
            "imputation": "corresponding_training_fold_only",
            "probability_calibration": "corresponding_training_fold_only",
            "standardization": "corresponding_training_fold_only",
        },
        "random_seed": RANDOM_SEED,
        "stratification_fields": [
            "dataset_id",
            "participant_row_count",
            "participant_positive_row_count",
        ],
        "stratum_offset": (
            "first eight SHA-256 digest bytes as unsigned big-endian "
            "integer modulo fold count"
        ),
    }
    core: dict[str, Any] = {
        "feature_schema_binding": schema_binding,
        "frozen_on": frozen_on,
        "inner_cross_fitting_folds": inner_summaries,
        "inputs": input_bindings,
        "manifest_version": MANIFEST_VERSION,
        "outer_folds": outer_summaries,
        "participant_assignments": assignments,
        "protocol": protocol,
        "split_id": split_id,
        "status": "frozen",
        "task_id": task_id,
        "totals": totals,
    }
    integrity = {
        "input_bindings_sha256": _sha256_bytes(_canonical_json_bytes(input_bindings)),
        "manifest_core_sha256": _sha256_bytes(_canonical_json_bytes(core)),
        "participant_assignments_sha256": _sha256_bytes(
            _canonical_json_bytes(assignments)
        ),
        "protocol_sha256": _sha256_bytes(_canonical_json_bytes(protocol)),
    }
    return {**core, "integrity": integrity}


def write_split_manifest(
    manifest: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write a deterministic manifest under a cooperative lock."""

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / _WRITE_LOCK_NAME
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SplitWriteError("split manifest write lock already exists") from exc
    temporary_path: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock:
            lock.write(f"{os.getpid()}\n")
            lock.flush()
            os.fsync(lock.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        payload = _canonical_json_bytes(dict(manifest))
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        return _sha256_bytes(payload)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def build_and_write_split_manifest(
    *,
    repository_root: Path | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], str]:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[5]
    )
    manifest = build_split_manifest(repository_root=root)
    destination = (
        Path(output_path) if output_path is not None else root / DEFAULT_SPLIT_RELATIVE
    )
    digest = write_split_manifest(
        manifest,
        destination,
        overwrite=overwrite,
    )
    return manifest, digest


__all__ = [
    "ASSIGNMENT_ALGORITHM_VERSION",
    "DATASET_SPECS",
    "DEFAULT_SPLIT_RELATIVE",
    "FEATURE_SCHEMA_SHA256",
    "INNER_FOLD_COUNT",
    "MANIFEST_VERSION",
    "OUTER_FOLD_COUNT",
    "ParticipantUnit",
    "RANDOM_SEED",
    "SPLIT_ID",
    "SplitInputError",
    "SplitWriteError",
    "assign_participant_folds",
    "build_and_write_split_manifest",
    "build_nested_assignments",
    "build_split_manifest",
    "write_split_manifest",
]
