"""V3.3.3-r2 six-source participant split binding.

This module reuses the audited DATA-007 assignment algorithm while binding the
new r2 artifacts and the publisher-corrected Hefei canonical.  The historical
``v3.3.3`` split remains byte-for-byte frozen in :mod:`splits`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.mood_social.splits import (
    DatasetSpec,
    build_split_manifest,
    write_split_manifest,
)


PROCESSED_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r2")
FEATURE_SCHEMA_RELATIVE = PROCESSED_RELATIVE / "manifests" / "feature_schema_manifest.json"
DEFAULT_SPLIT_RELATIVE = PROCESSED_RELATIVE / "splits" / "split_manifest.json"
SPLIT_ID = "mood-social-v3.3.3-r2-participant-nested-5x5-seed-20260728-v1"
FROZEN_ON = "2026-08-11"
TASK_ID = "OPT-V333-001-DATA-007-R2"


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="psyche_d",
        directory_name="psyche_d",
        canonical_name="canonical_psyche_d.parquet",
        mapping_name="psyche_d_v3_3_3_mapping.json",
        artifact_manifest_sha256="ef680d34cabd95a30bab223e20c3e913525850a8d9365fdbfb78582944375c50",
        canonical_sha256="c913768ea03dd509f1036e85bd66aa7560066023847306b5abddad950cc30b33",
        mapping_sha256="2d3af06eca579b1d063981ad4060bb86ae8b97a75cbec1b83b9ae6fa1c4430c8",
        frame_sha256="0ec9d9e42f2a5e068053ae69f4a66d1064819e4e7c914d04f74981cfd822cfa8",
        row_count=10_866,
        participant_count=4_036,
        positive_row_count=3_444,
        processed_relative=PROCESSED_RELATIVE,
    ),
    DatasetSpec(
        dataset_id="resilient",
        directory_name="resilient",
        canonical_name="canonical_resilient.parquet",
        mapping_name="resilient_v3_3_3_mapping.json",
        artifact_manifest_sha256="6fc7f488d800511b2ea30dbab8d970187608d0ef6cbcde2e5900cab439699a4f",
        canonical_sha256="fa2ddd3b79ccd3a7d3cb9073512f9a3813e26b48abd5621cd70cae53d1039229",
        mapping_sha256="ff8576d02b83cd264eb4334e6f7c398991067c6a84f287796f857db0adb52bcf",
        frame_sha256="5df4077cdab59f1e3a51c468cca32c1def88fb4c490bfa295eb1d5f5564bca7d",
        row_count=73,
        participant_count=73,
        positive_row_count=10,
        processed_relative=PROCESSED_RELATIVE,
    ),
    DatasetSpec(
        dataset_id="nhanes",
        directory_name="nhanes",
        canonical_name="canonical_nhanes.parquet",
        mapping_name="nhanes_v3_3_3_mapping.json",
        artifact_manifest_sha256="f524cc2a93a29f427261fb4e9b1a6523d33263e86cc8cdebeb89a256f63989b9",
        canonical_sha256="e6438d99f99b12f3607e1ee6482a48c6cc9235b04028f0184b5b96bd4320654f",
        mapping_sha256="e23748a507dc9c011a9e515bb79f2e0a7eb85647923b2f74f5f7e9b27320af82",
        frame_sha256="68cc1d90d3c028dd5c09f749d9d0cea432706c2cd71bd3173c72345e670bee63",
        row_count=2_775,
        participant_count=2_775,
        positive_row_count=254,
        processed_relative=PROCESSED_RELATIVE,
    ),
    DatasetSpec(
        dataset_id="shenzhen_elderly",
        directory_name="shenzhen",
        canonical_name="canonical_shenzhen.parquet",
        mapping_name="shenzhen_v3_3_3_mapping.json",
        artifact_manifest_sha256="ff32b173322cc067318b0734f73789e02902f4c913c7a016ff541fcbefd0e6da",
        canonical_sha256="85ea64ba940e671427a3a9e9ea7a515a90fa57c88520251b3b9964759ebfa054",
        mapping_sha256="f9aadda11b73c87e9ce4e2d42d37cccc04653b031bd9c8fedd284faca84e9fc4",
        frame_sha256="d5cdc2f585feb1e3573c09e092a56246439fa930d245d07191a02d3f42e7e1ac",
        row_count=5_327,
        participant_count=5_327,
        positive_row_count=186,
        processed_relative=PROCESSED_RELATIVE,
    ),
    DatasetSpec(
        dataset_id="nhanes_ssq_2005_2008",
        directory_name="nhanes_ssq_2005_2008",
        canonical_name="canonical_nhanes_ssq_2005_2008.parquet",
        mapping_name="nhanes_ssq_2005_2008_v3_3_3_mapping.json",
        artifact_manifest_sha256="b6c16479ccaf6a10506e182cebe0a378ba2c9327cf65431c718020e4cfecd06b",
        canonical_sha256="b0da21d9677d3cf59bdec347eedb3a9c34324374003e27c014b1b46f91db52ce",
        mapping_sha256="945bbd504cb4d2578c40b87ec1e039770ce45f1d02154bbc7bc6903cde486aea",
        frame_sha256="0fdb9c743f43d6fa2b5e8397ad1aa54d3f2c6e6e7ab09c4640055ca5923e2610",
        row_count=3_150,
        participant_count=3_150,
        positive_row_count=185,
        processed_relative=PROCESSED_RELATIVE,
    ),
    DatasetSpec(
        dataset_id="hefei_elderly",
        directory_name="hefei",
        canonical_name="canonical_hefei.parquet",
        mapping_name="hefei_v3_3_3_mapping.json",
        artifact_manifest_sha256="9f0c9d003d4068058cf67e478c03eeaa8fff058cc40a0c1891e0f8e19a992510",
        canonical_sha256="b7e25585e98d7749729dad59ceaf377bf09e6937d8aee741e1b321ff18a22bda",
        mapping_sha256="541d9d153f658326d92fdc6aca5d11119caed7740baac324914f0b4e7550d80d",
        frame_sha256="e26ca012514c6e8bc28c82da5cb229d970af29701d17c8e57e714867da2b746c",
        row_count=510,
        participant_count=510,
        positive_row_count=31,
        processed_relative=PROCESSED_RELATIVE,
    ),
)


def build_r2_split_manifest(*, repository_root: Path | None = None) -> dict[str, Any]:
    """Bind all six r2 canonicals and build deterministic nested assignments."""

    return build_split_manifest(
        repository_root=repository_root,
        dataset_specs=DATASET_SPECS,
        processed_relative=PROCESSED_RELATIVE,
        feature_schema_relative=FEATURE_SCHEMA_RELATIVE,
        split_id=SPLIT_ID,
        frozen_on=FROZEN_ON,
        task_id=TASK_ID,
    )


def build_and_write_r2_split_manifest(
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
    manifest = build_r2_split_manifest(repository_root=root)
    destination = Path(output_path) if output_path is not None else root / DEFAULT_SPLIT_RELATIVE
    digest = write_split_manifest(manifest, destination, overwrite=overwrite)
    return manifest, digest


__all__ = [
    "DATASET_SPECS",
    "DEFAULT_SPLIT_RELATIVE",
    "FEATURE_SCHEMA_RELATIVE",
    "FROZEN_ON",
    "PROCESSED_RELATIVE",
    "SPLIT_ID",
    "TASK_ID",
    "build_and_write_r2_split_manifest",
    "build_r2_split_manifest",
]
