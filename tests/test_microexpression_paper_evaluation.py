from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (
    ManifoldGraphConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_dataset import (
    PAPER_ORDERED_LANDMARK_INDICES,
    PAPER_REGION_COUNTS,
    PAPER_REGION_OFFSETS,
    PaperV2ArtifactDataset,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_evaluation import (
    EvaluationCandidate,
    FocalLoss,
    _loader,
    aggregate_seed_predictions,
    build_loss,
    choose_candidate,
    class_weights,
    run_candidate_training,
    validate_subject_partitions,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_mhssa_tgcn import (
    PaperMHSSATGCNConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_training import (
    PaperTrainingConfig,
)
from scripts.run_microexpression_paper_nested_loso import aggregate_final_results


def _record(
    root: Path,
    *,
    sample_id: str,
    subject_id: str,
    label: int,
) -> dict[str, object]:
    generator = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    path = root / f"{sample_id}.npz"
    np.savez_compressed(
        path,
        patches=generator.normal(size=(43, 75)).astype(np.float32),
        region_ids=np.repeat(np.arange(6), PAPER_REGION_COUNTS).astype(np.int64),
        region_offsets=np.asarray(PAPER_REGION_OFFSETS, dtype=np.int64),
        keypoints=generator.uniform(1.0, 30.0, size=(43, 2)).astype(np.float32),
        ordered_landmark_indices=np.asarray(
            PAPER_ORDERED_LANDMARK_INDICES, dtype=np.int64
        ),
        label=np.asarray(label, dtype=np.int64),
    )
    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "artifact_path": path.as_posix(),
        "label": label,
        "quality_status": "pass",
        "apex_boundary_status": "interior",
    }


def _records(root: Path) -> list[dict[str, object]]:
    return [
        _record(root, sample_id=f"train_{label}", subject_id="s_train", label=label)
        for label in range(3)
    ] + [
        _record(root, sample_id=f"val_{label}", subject_id="s_val", label=label)
        for label in range(3)
    ] + [
        _record(root, sample_id=f"test_{label}", subject_id="s_test", label=label)
        for label in range(3)
    ]


def _candidate(*, loss_mode: str = "cross_entropy") -> EvaluationCandidate:
    return EvaluationCandidate(
        graph_mode="cosine",
        fusion_formula="direct_rbf_stable",
        attention_variant="stable_variant",
        loss_mode=loss_mode,
    )


def _model_config() -> PaperMHSSATGCNConfig:
    return PaperMHSSATGCNConfig(
        hidden_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        gcn_layers=1,
        dropout=0.0,
        attention_variant="stable_variant",
        graph=ManifoldGraphConfig(
            mode="cosine",
            fusion_formula="direct_rbf_stable",
        ),
    )


def _training_config() -> PaperTrainingConfig:
    return PaperTrainingConfig(
        max_epochs=1,
        patience=1,
        batch_size=3,
        num_workers=0,
    )


def test_focal_weighted_and_balanced_sampler_modes(tmp_path: Path) -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
    targets = torch.tensor([0, 1])
    assert torch.allclose(
        FocalLoss(gamma=0.0)(logits, targets),
        torch.nn.functional.cross_entropy(logits, targets),
    )

    records = _records(tmp_path)
    weights = class_weights(records[:6])
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])
    weighted = build_loss("weighted_cross_entropy", records[:6], torch.device("cpu"))
    assert isinstance(weighted, torch.nn.CrossEntropyLoss)
    assert weighted.weight is not None

    dataset = PaperV2ArtifactDataset(records[:3])
    loader = _loader(
        dataset,
        candidate=_candidate(loss_mode="balanced_sampler"),
        training_config=_training_config(),
        shuffle=True,
        seed=17,
    )
    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_candidate_selection_uses_validation_score_then_tiebreakers() -> None:
    first = {
        "status": "completed",
        "candidate_id": "a",
        "validation": {
            "selection_score": 0.7,
            "metrics": {"uf1": 0.8, "uar": 0.6},
            "loss": 0.9,
        },
    }
    second = {
        "status": "completed",
        "candidate_id": "b",
        "validation": {
            "selection_score": 0.71,
            "metrics": {"uf1": 0.71, "uar": 0.71},
            "loss": 1.2,
        },
    }
    assert choose_candidate([first, second]) is second


def test_partition_contract_rejects_subject_leakage(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records[3]["subject_id"] = "s_train"
    with pytest.raises(ValueError, match="subject leakage"):
        validate_subject_partitions(
            records,
            train_sample_ids=["train_0", "train_1", "train_2"],
            validation_sample_ids=["val_0", "val_1", "val_2"],
            test_sample_ids=["test_0", "test_1", "test_2"],
        )


def test_validation_only_final_test_once_and_resume_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    common = {
        "records": records,
        "train_sample_ids": ["train_0", "train_1", "train_2"],
        "validation_sample_ids": ["val_0", "val_1", "val_2"],
        "candidate": _candidate(),
        "base_model_config": _model_config(),
        "training_config": _training_config(),
        "au_adjacency": torch.eye(6),
        "seed": 23,
        "device": torch.device("cpu"),
        "fold_id": "loso_fixture",
        "source_manifest": {"path": "fixture.jsonl", "sha256": "fixture"},
    }
    selection_dir = tmp_path / "selection"
    selection = run_candidate_training(
        **common,
        test_sample_ids=None,
        output_dir=selection_dir,
        stage="selection",
    )
    assert selection["test"] is None
    assert selection["run_contract"]["partitions"]["test"]["sample_count"] == 0
    assert not (selection_dir / "test_predictions.jsonl").exists()

    from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue import (
        paper_evaluation,
    )

    original_evaluate = paper_evaluation.evaluate_loader
    test_evaluations = 0

    def tracking_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal test_evaluations
        loader = args[1]
        subjects = {record["subject_id"] for record in loader.dataset.records}
        if subjects == {"s_test"}:
            test_evaluations += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(paper_evaluation, "evaluate_loader", tracking_evaluate)
    final_dir = tmp_path / "final"
    final = run_candidate_training(
        **common,
        test_sample_ids=["test_0", "test_1", "test_2"],
        output_dir=final_dir,
        stage="final_outer_test",
    )
    assert final["test"]["evaluation_count"] == 1
    assert test_evaluations == 1

    resumed = run_candidate_training(
        **common,
        test_sample_ids=["test_0", "test_1", "test_2"],
        output_dir=final_dir,
        stage="final_outer_test",
    )
    assert resumed["result_sha256"]
    assert test_evaluations == 1
    with pytest.raises(RuntimeError, match="contract mismatch"):
        run_candidate_training(
            **{**common, "candidate": _candidate(loss_mode="focal")},
            test_sample_ids=["test_0", "test_1", "test_2"],
            output_dir=final_dir,
            stage="final_outer_test",
        )


def _prediction(sample_id: str, subject_id: str, label: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "label": label,
        "prediction": label,
        "probabilities": [1.0 if index == label else 0.0 for index in range(3)],
        "quality_status": "pass",
        "apex_boundary_status": "interior",
    }


def test_bootstrap_and_multi_fold_seed_aggregation(tmp_path: Path) -> None:
    rows = [
        _prediction("a", "s01", 0),
        _prediction("b", "s02", 1),
        _prediction("c", "s03", 2),
    ]
    first = aggregate_seed_predictions(rows, seed=5, bootstrap_iterations=20)
    second = aggregate_seed_predictions(rows, seed=5, bootstrap_iterations=20)
    assert first["bootstrap_95_ci"] == second["bootstrap_95_ci"]

    final_results = []
    for seed in (5, 7):
        for fold_id, row in zip(("f1", "f2", "f3"), rows, strict=True):
            final_results.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "test": {"predictions": [row]},
                }
            )
    aggregate = aggregate_final_results(
        final_results=final_results,
        expected_sample_ids={"a", "b", "c"},
        seeds=[5, 7],
        bootstrap_iterations=20,
        output_root=tmp_path / "aggregate",
    )
    assert aggregate["seed_count"] == 2
    assert aggregate["bootstrap_iterations"] == 20
    assert [summary["fold_count"] for summary in aggregate["seed_summaries"]] == [
        3,
        3,
    ]
