"""Step-8 semantic-preserving augmentation contract tests."""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.wandering.augmentation import (
    AUGMENTED_PAIR_SCHEMA_VERSION,
    AugmentationBuildError,
    build_augmentation_bundle,
    build_qc_fault_suite,
    coverage_entropy_pair,
    endpoint_repeat_rate,
    pacing_reversal_reason_codes,
    semantic_check,
    train_attempt_budget,
    validate_train_selection_quotas,
    wp_train_pacing_reversal_histogram,
)
from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    load_augmentation_config,
    verify_augmentation_trust_roots,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import load_camera_config
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.topology import compute_topology


ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_CONFIG = ROOT / "configs/modules/wandering_augmentation_v1.yaml"
V2_CONFIG = ROOT / "configs/modules/wandering_augmentation_v2.yaml"
V3_CONFIG = ROOT / "configs/modules/wandering_augmentation_v3.yaml"
DIAGNOSTIC_BUNDLE = ROOT / "data/processed/wandering/augmentation/v1"
BLOCKED_V2_BUNDLE = ROOT / "data/processed/wandering/augmentation/v2"
PREPROCESSING_CONFIG = ROOT / "configs/data/wandering_preprocessing_v1.yaml"


def _line() -> np.ndarray:
    return np.column_stack((np.linspace(-0.5, 0.5, 80), np.zeros(80)))


def _parent(points: np.ndarray, label: str, source: str = "wandering_patterns") -> dict:
    preprocessing = load_preprocessing_config(PREPROCESSING_CONFIG)
    mask = np.ones(80, dtype=np.int8)
    return {
        "sample_id": "parent",
        "source_dataset": source,
        "split": "train",
        "pattern_label": label if source == "wandering_patterns" else "unknown",
        "pattern_supervision_eligible": source == "wandering_patterns",
        "binary_label": 0 if label == "direct" else 1,
        "binary_supervision_eligible": True,
        "shape_normalized_points": points.tolist(),
        "point_mask": mask.tolist(),
        "topology": compute_topology(
            points,
            mask,
            step_epsilon=float(preprocessing["step_epsilon"]),
            **preprocessing["topology"],
        ),
    }


def test_joint_coverage_entropy_max_edge_and_zero_span() -> None:
    clean = np.asarray([[0.0, 2.0], [1.0, 2.0], [0.0, 2.0], [1.0, 2.0]])
    corrupted = clean.copy()
    clean_entropy, corrupted_entropy = coverage_entropy_pair(
        clean, np.ones(4), corrupted, np.ones(4), grid_size=8, epsilon=1e-6
    )
    expected = math.log(2.0) / math.log(64.0)
    assert clean_entropy == pytest.approx(expected)
    assert corrupted_entropy == pytest.approx(expected)


def test_endpoint_repeat_uses_last_ten_against_first_ten_inclusive_radius() -> None:
    points = np.zeros((80, 2), dtype=np.float64)
    points[10:70, 0] = 1.0
    points[70:, 0] = 0.10
    assert endpoint_repeat_rate(points, np.ones(80), reference_points=10, query_points=10, radius=0.10) == 1.0


def test_direct_identity_passes_and_added_reversal_has_structured_reason() -> None:
    config = load_augmentation_config(AUGMENTATION_CONFIG)
    preprocessing = load_preprocessing_config(PREPROCESSING_CONFIG)
    clean = _line()
    accepted = semantic_check(
        _parent(clean, "direct"),
        clean,
        np.ones(80),
        config=config,
        preprocessing_config=preprocessing,
        qc_ready=True,
    )
    assert accepted.status == "accepted"
    reversal = clean.copy()
    reversal[40:, 0] = np.linspace(0.0, -0.5, 40)
    rejected = semantic_check(
        _parent(clean, "direct"),
        reversal,
        np.ones(80),
        config=config,
        preprocessing_config=preprocessing,
        qc_ready=True,
    )
    assert rejected.status == "rejected"
    assert any(reason.startswith("direct_") for reason in rejected.reason_codes)


def test_smartcare_executes_only_common_checks_without_pseudo_subtype() -> None:
    config = load_augmentation_config(AUGMENTATION_CONFIG)
    preprocessing = load_preprocessing_config(PREPROCESSING_CONFIG)
    clean = _line()
    result = semantic_check(
        _parent(clean, "unknown", source="smartcare"),
        clean,
        np.ones(80),
        config=config,
        preprocessing_config=preprocessing,
        qc_ready=True,
    )
    assert result.status == "accepted"
    assert result.assigned_subtype is None


def test_qc_unavailable_never_becomes_semantically_accepted() -> None:
    config = load_augmentation_config(AUGMENTATION_CONFIG)
    preprocessing = load_preprocessing_config(PREPROCESSING_CONFIG)
    clean = _line()
    result = semantic_check(
        _parent(clean, "direct"),
        clean,
        np.ones(80),
        config=config,
        preprocessing_config=preprocessing,
        qc_ready=False,
    )
    assert result.status == "not_run_qc_unavailable"
    assert result.reason_codes == ("camera_qc_unavailable",)


def test_output_directory_refusal_is_fail_closed(tmp_path: Path) -> None:
    from elderly_monitoring.modules.mental_health.wandering.augmentation import (
        commit_new_output_directory,
    )

    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(AugmentationBuildError, match="already exists"):
        commit_new_output_directory(output, {"x.json": b"{}\n"})
    nested = tmp_path / "new" / "nested" / "bundle"
    commit_new_output_directory(nested, {"x.json": b"{}\n"})
    assert (nested / "x.json").read_bytes() == b"{}\n"


def test_diagnostic_v1_hashes_are_frozen_and_builder_refuses_reuse() -> None:
    expected = {
        AUGMENTATION_CONFIG: "db0834ed96977511c292158886e25231461505991a83684c0c2e137e58cb2e1e",
        DIAGNOSTIC_BUNDLE / "manifest.json": "e733a99c4d17eb4f6d3295c2dea19196fece8b26026a140768bf2b155995474a",
        DIAGNOSTIC_BUNDLE / "augmentation_report.json": "ee1e0b1afadc08693b9017802fda207e2b8edcd5c269816ae17c25e7b94005d7",
    }
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected} == expected
    with pytest.raises(AugmentationBuildError, match="already exists"):
        build_augmentation_bundle(
            config_path=AUGMENTATION_CONFIG,
            project_root=ROOT,
            output_dir=DIAGNOSTIC_BUNDLE,
        )


def test_v2_config_is_frozen_and_no_formal_v2_bundle_exists() -> None:
    assert hashlib.sha256(V2_CONFIG.read_bytes()).hexdigest() == (
        "11e184e450eea8bd82966979bc32765863cbe843c5ef0cdea934b6e3e26905ff"
    )
    assert not BLOCKED_V2_BUNDLE.exists()


def test_read_only_v2_replay_fixes_all_train_selection_strata() -> None:
    rows = [
        json.loads(line)
        for line in (DIAGNOSTIC_BUNDLE / "augmentation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    labels: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        parent_id = str(row["parent_sample_id"])
        grouped.setdefault((parent_id, str(row["severity"])), []).append(row)
        labels[parent_id] = (
            str(row["source_dataset"]),
            str(row["labels"]["four_class"])
            if row["source_dataset"] == "wandering_patterns"
            else f"binary_{int(row['labels']['binary'])}",
        )
    selected: Counter[tuple[str, str, str]] = Counter()
    pacing_reasons = {
        "pacing_too_few_reversals",
        "pacing_reversal_lost",
        "pacing_excess_reversals",
    }
    for (parent_id, severity), attempts in grouped.items():
        source, label = labels[parent_id]
        for attempt in sorted(attempts, key=lambda value: int(value["attempt_index"])):
            reasons = list(attempt["semantic_reason_codes"])
            if (
                attempt["qc_status"] == "ready"
                and source == "wandering_patterns"
                and label == "pacing"
            ):
                reasons = [reason for reason in reasons if reason not in pacing_reasons]
                reasons.extend(
                    pacing_reversal_reason_codes(
                        int(attempt["clean_metrics"]["reversal_count"]),
                        int(attempt["corrupted_metrics"]["reversal_count"]),
                        max_lost_from_clean=1,
                        max_added_to_clean=1,
                    )
                )
            if attempt["qc_status"] == "ready" and not reasons:
                selected[(source, label, severity)] += 1
                break
    assert selected == Counter(
        {
            ("wandering_patterns", "direct", "low"): 137,
            ("wandering_patterns", "direct", "medium"): 34,
            ("wandering_patterns", "pacing", "low"): 124,
            ("wandering_patterns", "pacing", "medium"): 26,
            ("wandering_patterns", "lapping", "low"): 176,
            ("wandering_patterns", "lapping", "medium"): 109,
            ("wandering_patterns", "random", "low"): 233,
            ("wandering_patterns", "random", "medium"): 159,
            ("smartcare", "binary_0", "low"): 47,
            ("smartcare", "binary_0", "medium"): 30,
            ("smartcare", "binary_1", "low"): 68,
            ("smartcare", "binary_1", "medium"): 57,
        }
    )


def test_full_quota_gate_reports_every_failing_stratum() -> None:
    config = {
        "quota_gate": {
            "wp_min_selected_per_class_per_severity": 50,
            "smartcare_min_selected_per_binary_class_per_severity": 10,
            "report_all_fields": [
                "parent_count",
                "at_least_once_qc_ready_count",
                "semantic_accepted_count",
                "selected_count",
                "qc_reason_counts",
                "semantic_rejection_reason_counts",
            ],
        }
    }
    strata = [
        {
            "source_dataset": "wandering_patterns",
            "label": "direct",
            "severity": "medium",
            "parent_count": 280,
            "at_least_once_qc_ready_count": 180,
            "semantic_accepted_count": 34,
            "selected_count": 34,
            "qc_reason_counts": {"long_internal_gap": 10},
            "semantic_rejection_reason_counts": {"curvature_ratio_out_of_range": 40},
        },
        {
            "source_dataset": "wandering_patterns",
            "label": "pacing",
            "severity": "medium",
            "parent_count": 280,
            "at_least_once_qc_ready_count": 188,
            "semantic_accepted_count": 26,
            "selected_count": 26,
            "qc_reason_counts": {"long_internal_gap": 12},
            "semantic_rejection_reason_counts": {"curvature_ratio_out_of_range": 259},
        },
    ]
    with pytest.raises(AugmentationBuildError) as caught:
        validate_train_selection_quotas(strata, config=config)
    message = str(caught.value)
    assert '"label":"direct"' in message
    assert '"label":"pacing"' in message


def test_v3_train_attempt_budget_is_severity_specific_and_frozen() -> None:
    config = load_augmentation_config(V3_CONFIG)
    assert train_attempt_budget(config, "low") == 4
    assert train_attempt_budget(config, "medium") == 16
    with pytest.raises(AugmentationBuildError, match="invalid train attempt budget"):
        train_attempt_budget(config, "high")


def test_frozen_wp_train_pacing_reversal_histogram_is_exact() -> None:
    config = load_augmentation_config(AUGMENTATION_CONFIG)
    trusted = verify_augmentation_trust_roots(config, ROOT)
    bundle = load_preprocessing_bundle(
        rf_config_path=trusted["rf_config"],
        project_root=ROOT,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    pacing = [
        row
        for row in bundle.records_for_split("train")
        if row["source_dataset"] == "wandering_patterns" and row["pattern_label"] == "pacing"
    ]
    assert wp_train_pacing_reversal_histogram(pacing) == {"0": 233, "1": 47}


@pytest.mark.parametrize(
    ("clean", "corrupted"),
    [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
)
def test_v2_parent_relative_pacing_truths_accept(clean: int, corrupted: int) -> None:
    assert pacing_reversal_reason_codes(
        clean,
        corrupted,
        max_lost_from_clean=1,
        max_added_to_clean=1,
    ) == ()


@pytest.mark.parametrize(("clean", "corrupted"), [(0, 2), (1, 3)])
def test_v2_parent_relative_pacing_excess_is_rejected(clean: int, corrupted: int) -> None:
    assert pacing_reversal_reason_codes(
        clean,
        corrupted,
        max_lost_from_clean=1,
        max_added_to_clean=1,
    ) == ("pacing_excess_reversals",)


def test_frozen_fault_suite_has_240_gated_and_48_non_gating_controls() -> None:
    config = load_augmentation_config(AUGMENTATION_CONFIG)
    trusted = verify_augmentation_trust_roots(config, ROOT)
    bundle = load_preprocessing_bundle(
        rf_config_path=trusted["rf_config"],
        project_root=ROOT,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    records = build_qc_fault_suite(
        bundle.records_for_split("validation"),
        config=config,
        camera_config=load_camera_config(trusted["camera_config"]),
    )
    assert Counter(row["record_role"] for row in records) == Counter(
        gated_fault=240, limitation_control=48
    )
    controls = [row for row in records if row["record_role"] == "limitation_control"]
    assert all(row["training_allowed"] is False for row in controls)
    assert all(row["expected_qc_reason"] is None for row in controls)
    for row in controls:
        evidence = row["boundary_evidence"]
        np.testing.assert_allclose(evidence["mapped_d40"], evidence["expected_d40"], atol=1e-12)
        np.testing.assert_allclose(evidence["boundary_step"], evidence["parent_last_step"], atol=1e-12)
        np.testing.assert_allclose(evidence["donor_first_step"], evidence["parent_last_step"], atol=1e-12)
