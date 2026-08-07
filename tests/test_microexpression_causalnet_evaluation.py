from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (
    CausalNetConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_dataset import (
    CausalNetArtifactDataset,
    EXPECTED_ROUTE_ORDER,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_evaluation import (
    CausalNetTrainingConfig,
    aggregate_seed_predictions,
    release_decision,
    run_nested_loso_fold,
    validate_subject_partitions,
)


def _record(tmp_path: Path, subject: str, label: int) -> dict[str, object]:
    sample_id = f"smic_hs__{subject}__label_{label}"
    path = tmp_path / f"{sample_id}.npz"
    inputs = np.random.default_rng(label + int(subject[1:])).normal(
        size=(4, 3, 28, 28)
    ).astype(np.float32)
    metadata = {
        "sample_id": sample_id,
        "subject_id": subject,
        "label": label,
        "route_order": list(EXPECTED_ROUTE_ORDER),
        "input_shape": [4, 3, 28, 28],
        "onset_index": 0,
        "apex_index": 1,
        "offset_index": 2,
        "quality_status": "pass",
        "spatial_variant": "source_compatible_roi",
    }
    np.savez_compressed(
        path,
        inputs=inputs,
        label=np.asarray(label, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return {
        "sample_id": sample_id,
        "subject_id": subject,
        "label": label,
        "artifact_path": path.as_posix(),
    }


def _records(tmp_path: Path) -> list[dict[str, object]]:
    return [
        _record(tmp_path, subject, label)
        for subject in ("s01", "s02", "s03")
        for label in (0, 1, 2)
    ]


def test_preloaded_dataset_survives_source_removal(tmp_path: Path) -> None:
    record = _record(tmp_path, "s01", 0)
    dataset = CausalNetArtifactDataset([record], preload=True)
    Path(str(record["artifact_path"])).unlink()
    assert dataset[0]["inputs"].shape == (4, 3, 28, 28)


def test_partition_validation_rejects_subject_leakage(tmp_path: Path) -> None:
    records = _records(tmp_path)
    by_subject = {
        subject: [str(row["sample_id"]) for row in records if row["subject_id"] == subject]
        for subject in ("s01", "s02", "s03")
    }
    with pytest.raises(ValueError, match="subject leakage"):
        validate_subject_partitions(
            records,
            train_sample_ids=[by_subject["s01"][0]],
            validation_sample_ids=by_subject["s01"][1:],
            test_sample_ids=by_subject["s03"],
        )


def test_small_nested_loso_fold_uses_validation_checkpoint_only(tmp_path: Path) -> None:
    records = _records(tmp_path)
    sample_ids = {
        subject: [str(row["sample_id"]) for row in records if row["subject_id"] == subject]
        for subject in ("s01", "s02", "s03")
    }
    result = run_nested_loso_fold(
        records=records,
        train_sample_ids=sample_ids["s01"],
        validation_sample_ids=sample_ids["s02"],
        test_sample_ids=sample_ids["s03"],
        model_config=CausalNetConfig(
            dim=32,
            heads=2,
            block_repeats=(1, 1, 1),
            cross_heads=4,
            cross_depth=1,
            mlp_mult=2,
        ),
        training_config=CausalNetTrainingConfig(
            max_epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=8,
            loss="cross_entropy",
        ),
        seed=17,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        fold_id="loso_s03",
        input_hashes={
            "artifact_manifest_sha256": "a" * 64,
            "preprocessing_config_sha256": "b" * 64,
        },
        evaluation_bundle_hash="c" * 64,
    )
    assert result["status"] == "completed"
    assert result["test_used_for_selection"] is False
    assert result["test"]["evaluation_count"] == 1
    assert result["checkpoint"]["strict_reload_max_logit_difference"] == 0.0
    assert {row["subject_id"] for row in result["test"]["predictions"]} == {"s03"}


def test_seed_aggregation_reports_all_required_views() -> None:
    rows = [
        {
            "sample_id": f"s{subject}_{label}",
            "subject_id": f"s{subject}",
            "label": label,
            "prediction": label,
        }
        for subject in range(1, 5)
        for label in (0, 1, 2)
    ]
    summary = aggregate_seed_predictions(
        rows,
        seed=5,
        bootstrap_iterations=20,
    )
    assert summary["pooled"]["uf1"] == 1.0
    assert summary["pooled"]["uar"] == 1.0
    assert len(summary["subject_metrics"]) == 4
    assert set(summary["pooled"]["per_class"]) == {"negative", "positive", "surprise"}
    assert summary["pooled"]["confusion_matrix"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    assert set(summary["bootstrap_95_ci"]) == {"uf1", "uar"}


def test_release_decision_uses_uf1_uar_and_zero_fold_gate() -> None:
    def summaries(score: float, zero: bool = False) -> list[dict[str, object]]:
        return [
            {
                "pooled": {"uf1": score, "uar": score},
                "zero_score_folds": ["s01"] if zero else [],
            }
            for _ in range(3)
        ]

    assert release_decision(summaries(0.71)) == "candidate"
    assert release_decision(summaries(0.61)) == "research_baseline"
    assert release_decision(summaries(0.61, zero=True)) == "rejected"
    assert release_decision(summaries(0.54)) == "rejected"
