"""Step-8 frozen compatibility-report contract tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import canonical_json_bytes
from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    load_augmentation_config,
)
from elderly_monitoring.modules.mental_health.wandering.compatibility_report import (
    COMPATIBILITY_SCHEMA_VERSION,
    CompatibilityReportError,
    LoadedAugmentationBundle,
    build_compatibility_report,
    compatibility_output_schemas,
    jensen_shannon_base2,
    load_visual_review_evidence,
    summarize_prediction_groups,
)


EXTERNAL_A = "a" * 64
ROOT = Path(__file__).resolve().parents[1]


def _review_fixture(root: Path, *, status: str = "human_review_passed") -> dict[str, str]:
    root.mkdir()
    (root / "figures").mkdir()
    files: dict[str, bytes] = {
        "README.md": b"# review\n",
    }
    records: list[dict] = []
    index = 0
    for source, labels, per_stratum in (
        ("wandering_patterns", ("direct", "pacing", "lapping", "random"), 50),
        ("smartcare", (0, 1), 10),
    ):
        for label in labels:
            for severity in ("low", "medium"):
                child_ids = sorted(
                    (
                        f"{source}-{label}-{severity}-{value:03d}"
                        for value in range(per_stratum)
                    ),
                    key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
                )
                for child_id in child_ids:
                    index += 1
                    relative = f"figures/pairs/{index:03d}.png"
                    files[relative] = f"png-{index}".encode()
                    records.append(
                        {
                            "source_dataset": source,
                            "label": label,
                            "severity": severity,
                            "child_id": child_id,
                            "parent_sample_id": f"parent-{index:03d}",
                            "figure": relative,
                        }
                    )
    selection = canonical_json_bytes(
        {
            "schema_version": "wandering-augmentation-visual-selection-v3",
            "selection": "sha256_child_id_ascending",
            "augmentation_manifest_sha256": EXTERNAL_A,
            "records": records,
        }
    )
    files["figures/selection_index.json"] = selection
    for relative, payload in files.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    descriptors = {
        relative: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
        for relative, payload in files.items()
    }
    visual = canonical_json_bytes(
        {
            "schema_version": "wandering-augmentation-visual-manifest-v3",
            "augmentation_manifest_sha256": EXTERNAL_A,
            "artifacts": descriptors,
            "human_review_contract": {
                "path": "HUMAN_REVIEW.md",
                "initial_status": "pending_human_review",
                "required_final_status": "human_review_passed",
                "final_sha256_source": "external_cli",
            },
        }
    )
    visual_path = root / "figures/visual_manifest.json"
    visual_path.write_bytes(visual)
    visual_sha = hashlib.sha256(visual).hexdigest()
    header = {
        "schema_version": "wandering-augmentation-human-review-v3",
        "status": status,
        "augmentation_manifest_sha256": EXTERNAL_A,
        "visual_manifest_sha256": visual_sha,
        "selection_index_sha256": hashlib.sha256(selection).hexdigest(),
        "reviewed_pair_count": 440,
        "reviewer": "project-owner" if status == "human_review_passed" else "pending",
        "review_date": "2026-08-07" if status == "human_review_passed" else "pending",
        "rejected_child_ids": [],
        "final_decision": status,
    }
    human = (
        "<!-- wandering-augmentation-human-review-v3\n"
        + json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n-->\n# Human review\n"
    ).encode()
    (root / "HUMAN_REVIEW.md").write_bytes(human)
    return {
        "visual_sha": visual_sha,
        "human_sha": hashlib.sha256(human).hexdigest(),
        "selection_sha": hashlib.sha256(selection).hexdigest(),
    }


def _row(
    parent: str,
    severity: str,
    *,
    true_label: str,
    predicted_label: str | None,
    probabilities: list[float] | None,
) -> dict:
    return {
        "parent_sample_id": parent,
        "source_dataset": "wandering_patterns",
        "task": "binary",
        "model": "rf_binary",
        "severity": severity,
        "view_status": "ready" if probabilities is not None else "unavailable",
        "reason_codes": [] if probabilities is not None else ["long_internal_gap"],
        "true_label": true_label,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
    }


def test_jensen_shannon_base2_is_symmetric_bounded_and_exact_at_extremes() -> None:
    assert jensen_shannon_base2([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert jensen_shannon_base2([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    left = jensen_shannon_base2([0.75, 0.25], [0.25, 0.75])
    right = jensen_shannon_base2([0.25, 0.75], [0.75, 0.25])
    assert 0.0 <= left <= 1.0
    assert left == right
    with pytest.raises(CompatibilityReportError):
        jensen_shannon_base2([0.5, 0.5], [1.0])


def test_jensen_shannon_accepts_camera_float32_rounding_but_rejects_material_drift() -> None:
    camera_probabilities = [
        0.7810612320899963,
        0.09971078485250473,
        0.00016621312533970922,
        0.11906180530786514,
    ]
    assert sum(camera_probabilities) == pytest.approx(1.000000035375706)
    assert jensen_shannon_base2(camera_probabilities, camera_probabilities) == 0.0
    with pytest.raises(CompatibilityReportError):
        jensen_shannon_base2([0.99, 0.0], [0.5, 0.5])


def test_coverage_denominator_includes_unavailable_and_conditionals_do_not() -> None:
    rows = [
        _row("a", "clean", true_label="wandering_like", predicted_label="wandering_like", probabilities=[0.1, 0.9]),
        _row("b", "clean", true_label="direct_or_non_wandering", predicted_label="direct_or_non_wandering", probabilities=[0.8, 0.2]),
        _row("c", "clean", true_label="wandering_like", predicted_label="wandering_like", probabilities=[0.2, 0.8]),
        _row("a", "low", true_label="wandering_like", predicted_label="wandering_like", probabilities=[0.2, 0.8]),
        _row("b", "low", true_label="direct_or_non_wandering", predicted_label=None, probabilities=None),
        _row("c", "low", true_label="wandering_like", predicted_label="direct_or_non_wandering", probabilities=[0.6, 0.4]),
    ]
    summary = summarize_prediction_groups(rows)
    low = next(item for item in summary if item["severity"] == "low")
    assert low["coverage_denominator"] == 3
    assert low["ready_count"] == 2
    assert low["conditional_denominator"] == 2
    assert low["ready_coverage"] == pytest.approx(2 / 3)
    assert low["clean_corrupted_label_agreement"] == pytest.approx(0.5)
    assert math.isfinite(low["mean_jensen_shannon_base2"])


def test_tasks_and_models_remain_separate_without_fusion() -> None:
    rows = [
        _row("a", "clean", true_label="wandering_like", predicted_label="wandering_like", probabilities=[0.1, 0.9]),
    ]
    second = dict(rows[0], model="tcn_binary")
    summary = summarize_prediction_groups([*rows, second])
    assert {(item["model"], item["task"]) for item in summary} == {
        ("rf_binary", "binary"),
        ("tcn_binary", "binary"),
    }
    assert COMPATIBILITY_SCHEMA_VERSION == "wandering-anchorless-compatibility-v1"


def test_v3_compatibility_child_schemas_do_not_fall_back_to_v1() -> None:
    config = load_augmentation_config(ROOT / "configs/modules/wandering_augmentation_v3.yaml")
    assert compatibility_output_schemas(config) == (
        "wandering-compatibility-prediction-v3",
        "wandering-compatibility-manifest-v3",
    )


def test_augmentation_artifact_order_matches_canonical_json_key_order() -> None:
    import elderly_monitoring.modules.mental_health.wandering.compatibility_report as target

    assert target._AUGMENTATION_ARTIFACTS == tuple(
        sorted(target._AUGMENTATION_ARTIFACTS)
    )


def test_v3_visual_and_signed_human_review_chain_is_exact(tmp_path: Path) -> None:
    root = tmp_path / "review"
    expected = _review_fixture(root)
    evidence = load_visual_review_evidence(
        visual_review_dir=root,
        expected_visual_manifest_sha256=expected["visual_sha"],
        expected_human_review_sha256=expected["human_sha"],
        expected_augmentation_manifest_sha256=EXTERNAL_A,
    )
    assert evidence.visual_manifest_sha256 == expected["visual_sha"]
    assert evidence.human_review_sha256 == expected["human_sha"]
    assert evidence.selection_index_sha256 == expected["selection_sha"]
    assert evidence.reviewed_pair_count == 440
    assert evidence.human_review_status == "human_review_passed"


def test_v3_visual_review_rejects_pending_human_status(tmp_path: Path) -> None:
    root = tmp_path / "review"
    expected = _review_fixture(root, status="pending_human_review")
    with pytest.raises(CompatibilityReportError, match="human_review_passed"):
        load_visual_review_evidence(
            visual_review_dir=root,
            expected_visual_manifest_sha256=expected["visual_sha"],
            expected_human_review_sha256=expected["human_sha"],
            expected_augmentation_manifest_sha256=EXTERNAL_A,
        )


def test_v3_visual_review_rejects_artifact_tamper_and_path_escape(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    expected = _review_fixture(tampered)
    (tampered / "README.md").write_bytes(b"changed\n")
    with pytest.raises(CompatibilityReportError, match="artifact binding"):
        load_visual_review_evidence(
            visual_review_dir=tampered,
            expected_visual_manifest_sha256=expected["visual_sha"],
            expected_human_review_sha256=expected["human_sha"],
            expected_augmentation_manifest_sha256=EXTERNAL_A,
        )

    escaped = tmp_path / "escaped"
    expected = _review_fixture(escaped)
    manifest_path = escaped / "figures/visual_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["../escape.png"] = {"sha256": "0" * 64, "byte_count": 0}
    payload = canonical_json_bytes(manifest)
    manifest_path.write_bytes(payload)
    with pytest.raises(CompatibilityReportError, match="path"):
        load_visual_review_evidence(
            visual_review_dir=escaped,
            expected_visual_manifest_sha256=hashlib.sha256(payload).hexdigest(),
            expected_human_review_sha256=expected["human_sha"],
            expected_augmentation_manifest_sha256=EXTERNAL_A,
        )


def test_visual_failure_precedes_trust_roots_and_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import elderly_monitoring.modules.mental_health.wandering.compatibility_report as target

    monkeypatch.setattr(
        target,
        "load_augmentation_bundle",
        lambda **_kwargs: LoadedAugmentationBundle(
            tmp_path, {}, EXTERNAL_A, (), (), ()
        ),
    )
    calls = {"trust": 0, "models": 0}

    def forbidden_trust(*_args: object, **_kwargs: object) -> None:
        calls["trust"] += 1
        raise AssertionError("trust roots must not load before visual evidence")

    def forbidden_models(*_args: object, **_kwargs: object) -> None:
        calls["models"] += 1
        raise AssertionError("models must not load before visual evidence")

    monkeypatch.setattr(target, "verify_augmentation_trust_roots", forbidden_trust)
    monkeypatch.setattr(target, "load_trusted_camera_models", forbidden_models)
    output = tmp_path / "compatibility"
    with pytest.raises(CompatibilityReportError, match="visual review directory"):
        build_compatibility_report(
            config_path=Path(__file__).resolve().parents[1]
            / "configs/modules/wandering_augmentation_v3.yaml",
            project_root=Path(__file__).resolve().parents[1],
            augmentation_bundle=tmp_path / "augmentation",
            expected_augmentation_manifest_sha256=EXTERNAL_A,
            visual_review_dir=tmp_path / "missing-review",
            expected_visual_manifest_sha256="b" * 64,
            expected_human_review_sha256="c" * 64,
            rf_development_dir=tmp_path / "rf",
            expected_rf_development_manifest_sha256=(
                "fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2"
            ),
            tcn_development_dir=tmp_path / "tcn",
            expected_tcn_development_manifest_sha256=(
                "0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e"
            ),
            output_dir=output,
        )
    assert calls == {"trust": 0, "models": 0}
    assert not output.exists()
