"""Step-8 synthetic camera corruption contract tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    AUGMENTATION_CONFIG_SCHEMA_VERSION,
    AUGMENTATION_CONFIG_SHA256,
    CameraCorruptionError,
    apply_projective,
    build_camera_adapter_input,
    compose_projective_matrix,
    derive_child_identity,
    derive_versioned_child_identity,
    derive_versioned_fault_identity,
    fit_virtual_canvas,
    generate_corrupted_camera_view,
    interpolate_local_offsets,
    load_augmentation_config,
    quantize_half_up,
    sample_ar1_noise,
    verify_augmentation_trust_roots,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import load_camera_config
from elderly_monitoring.modules.mental_health.wandering.camera_qc import run_camera_qc
from elderly_monitoring.modules.mental_health.wandering.preprocessing import load_preprocessing_config
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_augmentation_v1.yaml"
V2_CONFIG = ROOT / "configs/modules/wandering_augmentation_v2.yaml"
V3_CONFIG = ROOT / "configs/modules/wandering_augmentation_v3.yaml"


def test_frozen_config_and_identity_contract() -> None:
    config = load_augmentation_config(CONFIG)
    assert config["schema_version"] == AUGMENTATION_CONFIG_SCHEMA_VERSION
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == AUGMENTATION_CONFIG_SHA256
    child_id, child_seed = derive_child_identity(
        parent_sample_id="parent-a",
        split="train",
        severity="medium",
        view_role="train_pair",
        attempt_index=2,
        global_seed=config["global_seed"],
    )
    expected = hashlib.sha256(
        b"wandering-augmentation-v1|parent-a|train|medium|train_pair|2"
    ).hexdigest()
    assert child_id == expected
    assert child_seed == int.from_bytes(
        hashlib.sha256(f"20260805|{child_id}".encode()).digest()[:8], "big"
    )


def test_v2_uses_new_artifact_id_but_preserves_v1_replay_seed() -> None:
    config = load_augmentation_config(V2_CONFIG)
    child_id, replay_key, child_seed = derive_versioned_child_identity(
        parent_sample_id="parent-a",
        split="train",
        severity="medium",
        view_role="train_pair",
        attempt_index=2,
        config_schema_version=config["schema_version"],
        global_seed=config["global_seed"],
    )
    expected_v2 = hashlib.sha256(
        b"wandering-augmentation-v2|parent-a|train|medium|train_pair|2"
    ).hexdigest()
    expected_replay = hashlib.sha256(
        b"wandering-augmentation-v1|parent-a|train|medium|train_pair|2"
    ).hexdigest()
    assert child_id == expected_v2
    assert replay_key == expected_replay
    assert child_seed == int.from_bytes(
        hashlib.sha256(f"20260805|{expected_replay}".encode()).digest()[:8], "big"
    )
    assert config["diagnostic_predecessor"]["output_reuse_allowed"] is False
    assert config["output_schemas"]["manifest"] == "wandering-augmentation-manifest-v2"


def test_v3_only_changes_the_frozen_allowlisted_config_paths() -> None:
    v2 = load_augmentation_config(V2_CONFIG)
    v3 = load_augmentation_config(V3_CONFIG)
    unchanged = (
        "purpose",
        "global_seed",
        "trust_roots",
        "split_policy",
        "profiles",
        "semantic_check",
        "fault_suite",
        "compatibility",
        "visual_review",
    )
    for field in unchanged:
        assert v3[field] == v2[field]
    v2_generation = dict(v2["generation"])
    v3_generation = dict(v3["generation"])
    assert v2_generation.pop("max_train_attempts_per_severity") == 4
    assert v3_generation.pop("max_train_attempts_per_severity") == {
        "low": 4,
        "medium": 16,
    }
    assert v3_generation == v2_generation
    assert v3["schema_version"] == "wandering-augmentation-config-v3"
    assert set(v3["diagnostic_predecessors"]) == {"v1", "v2"}
    assert v3["quota_gate"]["wp_min_selected_per_class_per_severity"] == 50
    assert v3["quota_gate"]["smartcare_min_selected_per_binary_class_per_severity"] == 10
    assert set(v3["output_schemas"].values()) == {
        "wandering-augmentation-attempt-v3",
        "wandering-augmented-pair-v3",
        "wandering-corruption-pressure-v3",
        "wandering-qc-fault-v3",
        "wandering-augmentation-report-v3",
        "wandering-augmentation-manifest-v3",
        "wandering-anchorless-compatibility-v3",
    }


def test_v3_artifact_namespace_preserves_v1_replay_key_and_seed() -> None:
    config = load_augmentation_config(V3_CONFIG)
    child_id, replay_key, child_seed = derive_versioned_child_identity(
        parent_sample_id="parent-a",
        split="train",
        severity="medium",
        view_role="train_pair",
        attempt_index=7,
        config_schema_version=config["schema_version"],
        global_seed=config["global_seed"],
    )
    assert child_id == hashlib.sha256(
        b"wandering-augmentation-v3|parent-a|train|medium|train_pair|7"
    ).hexdigest()
    assert replay_key == hashlib.sha256(
        b"wandering-augmentation-v1|parent-a|train|medium|train_pair|7"
    ).hexdigest()
    assert child_seed == int.from_bytes(
        hashlib.sha256(f"20260805|{replay_key}".encode()).digest()[:8], "big"
    )
    fault_id, fault_replay, fault_seed = derive_versioned_fault_identity(
        parent_sample_id="parent-a",
        fault_or_control_type="long_gap_4_buckets",
        control=False,
        config_schema_version=config["schema_version"],
        global_seed=config["global_seed"],
    )
    assert fault_id == hashlib.sha256(
        b"wandering-qc-fault-v3|parent-a|long_gap_4_buckets"
    ).hexdigest()
    assert fault_replay == hashlib.sha256(
        b"wandering-qc-fault-v1|parent-a|long_gap_4_buckets"
    ).hexdigest()
    assert fault_seed == int.from_bytes(
        hashlib.sha256(f"20260805|{fault_replay}".encode()).digest()[:8], "big"
    )


def test_config_rejects_any_byte_drift(tmp_path: Path) -> None:
    drifted = tmp_path / "config.yaml"
    drifted.write_bytes(CONFIG.read_bytes().replace(b"global_seed: 20260805", b"global_seed: 1"))
    with pytest.raises(CameraCorruptionError, match="frozen values"):
        load_augmentation_config(drifted)


def test_isotropic_virtual_canvas_fit_and_degenerate_rejection() -> None:
    points = np.asarray([[0.0, 0.0], [4.0, 2.0], [2.0, 1.0]])
    fitted = fit_virtual_canvas(points, np.ones(3), extent=0.70, epsilon=1e-6)
    np.testing.assert_allclose(fitted, [[0.15, 0.325], [0.85, 0.675], [0.5, 0.5]])
    with pytest.raises(CameraCorruptionError, match="degenerate_parent_span"):
        fit_virtual_canvas(np.ones((3, 2)), np.ones(3), extent=0.70, epsilon=1e-6)


def test_column_vector_projective_matrix_order_and_denominator_gate() -> None:
    identity = compose_projective_matrix(
        theta_degrees=0.0,
        uniform_scale=1.0,
        anisotropic_x=1.0,
        anisotropic_y=1.0,
        shear_x=0.0,
        shear_y=0.0,
        projective_x=0.0,
        projective_y=0.0,
        translation_x=0.0,
        translation_y=0.0,
    )
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-15)
    rotation = compose_projective_matrix(
        theta_degrees=90.0,
        uniform_scale=1.0,
        anisotropic_x=1.0,
        anisotropic_y=1.0,
        shear_x=0.0,
        shear_y=0.0,
        projective_x=0.0,
        projective_y=0.0,
        translation_x=0.0,
        translation_y=0.0,
    )
    projected, denominator = apply_projective(np.asarray([[0.75, 0.5]]), rotation)
    np.testing.assert_allclose(projected, [[0.5, 0.75]], atol=1e-15)
    np.testing.assert_allclose(denominator, [1.0])
    invalid_h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, -1.0]])
    with pytest.raises(CameraCorruptionError, match="projection_denominator_invalid"):
        apply_projective(np.asarray([[0.0, 0.0], [1.0, 0.0]]), invalid_h)


def test_local_offset_interpolation_smoothing_and_protected_zero() -> None:
    control_indices = [0, 4, 9]
    control_offsets = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    result = interpolate_local_offsets(
        point_count=10,
        control_indices=control_indices,
        control_offsets=control_offsets,
        protected_indices={0, 4, 9},
    )
    assert result.shape == (10, 2)
    np.testing.assert_array_equal(result[[0, 4, 9]], np.zeros((3, 2)))
    assert np.all(result[:, 1] == 0.0)
    assert result[2, 0] > result[1, 0]


def test_ar1_uses_stationary_sigma_zero_state_and_single_clipped_draws() -> None:
    class StubRng:
        def normal(self, _mean: float, _sigma: float, *, size: tuple[int, ...]) -> np.ndarray:
            assert size == (3, 1)
            return np.asarray([[100.0], [0.0], [-100.0]])

    values = sample_ar1_noise(StubRng(), count=3, rho=0.6, stationary_sigma=0.5, dimensions=1)
    innovation_sigma = 0.5 * math.sqrt(1.0 - 0.6**2)
    expected = np.asarray([3 * innovation_sigma, 0.6 * 3 * innovation_sigma, 0.6**2 * 3 * innovation_sigma - 3 * innovation_sigma])
    np.testing.assert_allclose(values[:, 0], expected)
    np.testing.assert_array_equal(quantize_half_up(np.asarray([0.49, 0.5, 1.5, 2.51])), [0, 1, 2, 3])


def test_three_missing_buckets_interpolate_but_four_are_rejected() -> None:
    camera = load_camera_config(ROOT / "configs/modules/wandering_camera_v1.yaml")
    points = np.column_stack((np.linspace(0.30, 0.70, 80), np.full(80, 0.70)))
    heights = np.full(80, 0.10)
    observed_three = np.ones(80, dtype=bool)
    observed_three[38:41] = False
    ready = run_camera_qc(
        build_camera_adapter_input(
            points,
            heights,
            record_id="three-gap",
            camera_config=camera,
            observed_mask=observed_three,
        ),
        camera,
    )
    assert len(ready.ready_inputs) == 1
    assert ready.window_records[0]["interpolated_bucket_count"] == 3
    observed_four = np.ones(80, dtype=bool)
    observed_four[38:42] = False
    rejected = run_camera_qc(
        build_camera_adapter_input(
            points,
            heights,
            record_id="four-gap",
            camera_config=camera,
            observed_mask=observed_four,
        ),
        camera,
    )
    assert len(rejected.ready_inputs) == 0
    assert "long_internal_gap" in {
        reason for row in rejected.tracklet_records for reason in row["quality_flags"]
    }


def test_real_child_is_order_independent_and_reuses_step7_step4() -> None:
    config = load_augmentation_config(CONFIG)
    trusted = verify_augmentation_trust_roots(config, ROOT)
    bundle = load_preprocessing_bundle(
        rf_config_path=trusted["rf_config"],
        project_root=ROOT,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    parent = sorted(bundle.records_for_split("train"), key=lambda row: row["sample_id"])[0]
    camera = load_camera_config(trusted["camera_config"])
    preprocessing = load_preprocessing_config(trusted["preprocessing_config"])
    stats = json.loads(trusted["preprocessing_feature_stats"].read_text(encoding="utf-8"))
    first = generate_corrupted_camera_view(
        parent,
        severity="low",
        view_role="train_pair",
        attempt_index=0,
        config=config,
        camera_config=camera,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    second = generate_corrupted_camera_view(
        parent,
        severity="low",
        view_role="train_pair",
        attempt_index=0,
        config=config,
        camera_config=camera,
        preprocessing_config=preprocessing,
        feature_stats=stats,
    )
    assert first.attempt_record == second.attempt_record
    assert first.adapter_input == second.adapter_input
    parameters = first.attempt_record["projection_parameters"]
    profile = config["profiles"]["low"]
    assert profile["anisotropic_scale"][0] <= parameters["anisotropic_x"] <= profile["anisotropic_scale"][1]
    assert profile["anisotropic_scale"][0] <= parameters["anisotropic_y"] <= profile["anisotropic_scale"][1]
    if first.prepared_window is not None:
        raw = np.asarray(first.prepared_window["raw_features"])
        np.testing.assert_array_equal(raw[:, 10:12], np.zeros((80, 2)))


def test_v2_v3_replay_v1_corruption_and_qc_bytes_with_only_artifact_identity_changed() -> None:
    v1 = load_augmentation_config(CONFIG)
    v2 = load_augmentation_config(V2_CONFIG)
    v3 = load_augmentation_config(V3_CONFIG)
    trusted = verify_augmentation_trust_roots(v3, ROOT)
    bundle = load_preprocessing_bundle(
        rf_config_path=trusted["rf_config"],
        project_root=ROOT,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    parent = sorted(bundle.records_for_split("train"), key=lambda row: row["sample_id"])[0]
    camera = load_camera_config(trusted["camera_config"])
    preprocessing = load_preprocessing_config(trusted["preprocessing_config"])
    stats = json.loads(trusted["preprocessing_feature_stats"].read_text(encoding="utf-8"))

    views = [
        generate_corrupted_camera_view(
            parent,
            severity="medium",
            view_role="train_pair",
            attempt_index=2,
            config=config,
            camera_config=camera,
            preprocessing_config=preprocessing,
            feature_stats=stats,
        )
        for config in (v1, v2, v3)
    ]
    v1_view, v2_view, v3_view = views
    assert len({view.attempt_record["child_id"] for view in views}) == 3
    assert len({view.attempt_record["corruption_replay_key"] for view in views}) == 1
    assert len({view.attempt_record["child_seed"] for view in views}) == 1
    for field in (
        "projection_matrix",
        "projection_parameters",
        "corruption_parameters",
        "realized_missing_indices",
        "contiguous_gap",
        "crop_edge",
        "frame_ids",
        "qc_status",
        "qc_reason_codes",
    ):
        assert len({json.dumps(view.attempt_record[field], sort_keys=True) for view in views}) == 1
    assert v1_view.adapter_input == v2_view.adapter_input == v3_view.adapter_input
    assert v1_view.qc_result == v2_view.qc_result == v3_view.qc_result
    assert v1_view.prepared_window == v2_view.prepared_window == v3_view.prepared_window
