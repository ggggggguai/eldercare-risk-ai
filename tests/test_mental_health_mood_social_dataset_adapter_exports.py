"""Integration checks for the frozen public-dataset adapter namespaces."""

from elderly_monitoring.datasets import adapters
from elderly_monitoring.modules.mental_health.validation import public_datasets


def test_authoritative_adapter_exports_are_namespaced_without_legacy_collisions() -> (
    None
):
    assert adapters.ADAPTER_VERSION == adapters.psyche_d.ADAPTER_VERSION
    assert adapters.TrainingFoldECDF is adapters.psyche_d.TrainingFoldECDF
    assert adapters.build_canonical_frame is adapters.psyche_d.build_canonical_frame

    assert adapters.RESILIENT_DATASET_ID == "resilient"
    assert adapters.ResilientTrainingFoldECDF is adapters.resilient.TrainingFoldECDF
    assert (
        adapters.build_resilient_canonical_frame
        is adapters.resilient.build_canonical_frame
    )
    assert (
        adapters.build_resilient_artifacts
        is adapters.resilient.build_resilient_artifacts
    )

    assert adapters.NHANES_DATASET_ID == "nhanes"
    assert adapters.NhanesTrainingFoldECDF is adapters.nhanes.TrainingFoldECDF
    assert (
        adapters.build_nhanes_canonical_frame is adapters.nhanes.build_canonical_frame
    )
    assert adapters.build_nhanes_artifacts is adapters.nhanes.build_nhanes_artifacts

    assert adapters.NHANES_SSQ_DATASET_ID == "nhanes_ssq_2005_2008"
    assert (
        adapters.build_nhanes_ssq_canonical_frame
        is adapters.nhanes_ssq.build_canonical_frame
    )
    assert (
        adapters.build_nhanes_ssq_artifacts
        is adapters.nhanes_ssq.build_nhanes_ssq_artifacts
    )

    assert adapters.SHENZHEN_DATASET_ID == "shenzhen_elderly"
    assert (
        adapters.build_shenzhen_canonical_frame
        is adapters.shenzhen.build_canonical_frame
    )
    assert (
        adapters.build_shenzhen_artifacts is adapters.shenzhen.build_shenzhen_artifacts
    )


def test_data001_compatibility_exports_point_to_authoritative_implementations() -> None:
    assert public_datasets.TrainingFoldECDF is adapters.psyche_d.TrainingFoldECDF
    assert (
        public_datasets.ResilientTrainingFoldECDF is adapters.resilient.TrainingFoldECDF
    )
    assert (
        public_datasets.build_resilient_artifacts
        is adapters.resilient.build_resilient_artifacts
    )
    assert public_datasets.NhanesTrainingFoldECDF is adapters.nhanes.TrainingFoldECDF
    assert (
        public_datasets.build_nhanes_artifacts is adapters.nhanes.build_nhanes_artifacts
    )
    assert (
        public_datasets.build_nhanes_ssq_canonical_frame
        is adapters.nhanes_ssq.build_canonical_frame
    )
    assert (
        public_datasets.build_nhanes_ssq_artifacts
        is adapters.nhanes_ssq.build_nhanes_ssq_artifacts
    )
    assert (
        public_datasets.build_shenzhen_canonical_frame
        is adapters.shenzhen.build_canonical_frame
    )
    assert (
        public_datasets.build_shenzhen_artifacts
        is adapters.shenzhen.build_shenzhen_artifacts
    )
