from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (
    ManifoldGraphConfig,
    PaperGraphBuilder,
    build_au_cooccurrence_prior,
    classical_mds,
    floyd_warshall,
    parse_action_units,
    rbf_adjacency,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.evaluation_variants import (
    evaluation_variant_id,
    normalize_evaluation_flow,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_dataset import (
    PAPER_REGION_COUNTS,
    PAPER_REGION_OFFSETS,
    PAPER_ORDERED_LANDMARK_INDICES,
    PaperV2Collator,
    PaperV2ArtifactDataset,
    collate_paper_v2,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_mhssa_tgcn import (
    PAPER_MODEL_SCHEMA_VERSION,
    PaperMHSSATGCN,
    PaperMHSSATGCNConfig,
    load_paper_model_checkpoint,
    reciprocal_distance_weights,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_training import (
    PaperTrainingConfig,
    build_paper_training_components,
    save_paper_model_checkpoint,
)


def _small_config(
    *,
    attention_variant: str = "paper_exact",
    graph_mode: str = "fused",
) -> PaperMHSSATGCNConfig:
    return PaperMHSSATGCNConfig(
        hidden_dim=32,
        transformer_layers=2,
        transformer_heads=4,
        gcn_layers=3,
        dropout=0.0,
        attention_variant=attention_variant,
        graph=ManifoldGraphConfig(
            mode=graph_mode,
            mds_dimensions=2,
            rbf_sigma=1.0,
            fusion_formula="paper_equation_3_17",
        ),
    )


def _write_artifact(path: Path, *, label: int = 1) -> dict[str, object]:
    generator = np.random.default_rng(7)
    region_ids = np.repeat(np.arange(6), PAPER_REGION_COUNTS)
    keypoints = generator.uniform(1.0, 30.0, size=(43, 2)).astype(np.float32)
    np.savez_compressed(
        path,
        patches=generator.normal(size=(43, 75)).astype(np.float32),
        region_ids=region_ids.astype(np.int64),
        region_offsets=np.asarray(PAPER_REGION_OFFSETS, dtype=np.int64),
        keypoints=keypoints,
        ordered_landmark_indices=np.asarray(
            PAPER_ORDERED_LANDMARK_INDICES, dtype=np.int64
        ),
        label=np.asarray(label, dtype=np.int64),
    )
    return {
        "sample_id": "smic_hs__s01__fixture",
        "subject_id": "s01",
        "artifact_path": path.as_posix(),
        "label": label,
        "quality_status": "pass",
        "apex_boundary_status": "interior",
    }


def _batch(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    patches = torch.randn(batch_size, 6, 12, 75, generator=generator)
    keypoints = torch.rand(batch_size, 6, 12, 2, generator=generator) * 31.0
    masks = torch.zeros(batch_size, 6, 12, dtype=torch.bool)
    for region, count in enumerate(PAPER_REGION_COUNTS):
        masks[:, region, :count] = True
    patches[~masks] = 0.0
    keypoints[~masks] = 0.0
    return patches, masks, keypoints


def test_paper_dataset_and_collate_preserve_six_real_regions(tmp_path: Path) -> None:
    record = _write_artifact(tmp_path / "sample.npz")
    dataset = PaperV2ArtifactDataset([record])
    batch = collate_paper_v2([dataset[0]])

    assert batch["patches"].shape == (1, 6, 12, 75)
    assert batch["masks"].sum(dim=-1).tolist() == [list(PAPER_REGION_COUNTS)]
    assert batch["labels"].tolist() == [1]
    assert torch.all(batch["ordered_landmark_indices"][~batch["masks"]] == -1)


def test_paper_dataset_rejects_non_finite_patch(tmp_path: Path) -> None:
    record = _write_artifact(tmp_path / "sample.npz")
    with np.load(record["artifact_path"], allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    arrays["patches"][0, 0] = np.nan
    np.savez_compressed(record["artifact_path"], **arrays)
    with pytest.raises(ValueError, match="non-finite"):
        PaperV2ArtifactDataset([record])


def test_order_ablation_is_deterministic_and_region_local(tmp_path: Path) -> None:
    record = _write_artifact(tmp_path / "sample.npz")
    sample = PaperV2ArtifactDataset([record])[0]
    paper = PaperV2Collator(order_mode="paper")([sample])
    standard = PaperV2Collator(order_mode="standard_index")([sample])
    random_one = PaperV2Collator(order_mode="random_fixed", order_seed=9)([sample])
    random_two = PaperV2Collator(order_mode="random_fixed", order_seed=9)([sample])

    assert torch.equal(random_one["patches"], random_two["patches"])
    assert not torch.equal(paper["patches"], standard["patches"])
    assert not torch.equal(paper["patches"], random_one["patches"])
    for batch in (paper, standard, random_one):
        assert batch["masks"].sum(dim=-1).tolist() == [list(PAPER_REGION_COUNTS)]


def test_evaluation_flow_normalization_and_variant_ids() -> None:
    flow = np.zeros((3, 4, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    flow[..., 1] = -1.0
    magnitude = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
    normalized, scales = normalize_evaluation_flow(
        flow, magnitude, quantile=0.995
    )
    assert normalized.shape == (3, 4, 3)
    assert np.isfinite(normalized).all()
    assert normalized[..., 2].min() >= 0.0
    assert scales["vector_scale"] == pytest.approx(2.0)
    assert evaluation_variant_id("P3", "tvl1", "magnitude") == (
        "p3_tvl1_magnitude"
    )


def test_reciprocal_distance_exact_epsilon_diagonal_and_mask() -> None:
    coordinates = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    weights = reciprocal_distance_weights(
        coordinates,
        mask,
        epsilon=1e-3,
        variant="paper_exact",
    )
    assert weights[0, 0, 0].item() == pytest.approx(1000.0)
    assert weights[0, 0, 1].item() == pytest.approx(1.0 / 5.001)
    assert torch.count_nonzero(weights[0, 2]).item() == 0
    assert torch.count_nonzero(weights[0, :, 2]).item() == 0
    assert torch.isfinite(weights).all()


def test_stable_reciprocal_distance_has_unit_diagonal_and_finite_rows() -> None:
    coordinates = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    weights = reciprocal_distance_weights(
        coordinates,
        mask,
        epsilon=1e-6,
        variant="stable_variant",
    )
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.full((1, 3), 3.0))


def test_floyd_warshall_uses_shorter_intermediate_path() -> None:
    costs = torch.tensor(
        [[[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]]]
    )
    distances = floyd_warshall(costs)
    assert distances[0, 0, 2].item() == pytest.approx(2.0)
    assert torch.equal(distances, distances.transpose(1, 2))


def test_classical_mds_and_rbf_preserve_euclidean_triangle() -> None:
    points = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]])
    distances = torch.cdist(points, points)
    embedded = classical_mds(distances, dimensions=2)
    assert torch.allclose(torch.cdist(embedded, embedded), distances, atol=1e-4)
    adjacency = rbf_adjacency(embedded, sigma=2.0)
    assert adjacency[0, 0, 1].item() == pytest.approx(
        np.exp(-(3.0**2) / (2.0 * 2.0**2)), abs=1e-5
    )
    assert torch.allclose(
        torch.diagonal(adjacency, dim1=-2, dim2=-1), torch.ones(1, 3)
    )


def test_graph_fusion_matches_paper_equation_and_is_batch_independent() -> None:
    prior = torch.eye(6)
    builder = PaperGraphBuilder(
        prior,
        ManifoldGraphConfig(
            mode="fused",
            fusion_formula="paper_equation_3_17",
            rbf_sigma=0.8,
        ),
    )
    features = torch.stack(
        [torch.eye(6), torch.arange(36, dtype=torch.float32).reshape(6, 6)]
    )
    adjacency, details = builder(features)
    expected = 0.5 * details["au_adjacency"] + 0.5 * details[
        "geometry_fusion_term"
    ]
    assert torch.allclose(adjacency, expected)
    assert not torch.equal(adjacency[0], adjacency[1])
    assert torch.isfinite(adjacency).all()


def test_au_prior_parser_and_class_balanced_audit(tmp_path: Path) -> None:
    assert parse_action_units("4+7+L10+15A") == (4, 7, 10, 15)
    paths = []
    for source in ("casme2", "samm"):
        path = tmp_path / f"{source}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["dataset", "label", "Action Units"]
            )
            writer.writeheader()
            writer.writerow(
                {"dataset": source, "label": 0, "Action Units": "4+7"}
            )
            writer.writerow(
                {"dataset": source, "label": 1, "Action Units": "12+14"}
            )
            writer.writerow(
                {"dataset": source, "label": 2, "Action Units": "1+6+9"}
            )
        paths.append(path)
    adjacency, audit = build_au_cooccurrence_prior(paths)
    assert adjacency.shape == (6, 6)
    assert np.allclose(adjacency, adjacency.T)
    assert float(adjacency.max()) == pytest.approx(1.0)
    assert np.all(np.diag(adjacency) > 0)
    assert audit["parsed_rows"] == 6
    assert audit["class_support"] == {"0": 2, "1": 2, "2": 2}
    assert audit["source"] == "reconstructed_from_student_auxiliary_csv"
    assert audit["method"].endswith("equation_3_12")


def test_paper_model_forward_backward_and_mask_invariance() -> None:
    torch.manual_seed(31)
    model = PaperMHSSATGCN(torch.eye(6), _small_config()).eval()
    patches, masks, keypoints = _batch()
    logits, aux = model(patches, masks, keypoints, return_aux=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
    loss.backward()
    assert logits.shape == (2, 3)
    assert aux["adjacency"].shape == (2, 6, 6)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    changed_patches = patches.clone()
    changed_keypoints = keypoints.clone()
    changed_patches[~masks] = 1e6
    changed_keypoints[~masks] = 1e6
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        changed = model(changed_patches, masks, changed_keypoints)
    assert torch.allclose(logits, changed, atol=1e-6, rtol=1e-6)


def test_paper_training_configuration_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    model = PaperMHSSATGCN(torch.eye(6), _small_config(graph_mode="au")).eval()
    training = PaperTrainingConfig()
    optimizer, scheduler, loss_function = build_paper_training_components(
        model, training
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert scheduler.step_size == 10
    assert scheduler.gamma == pytest.approx(0.95)
    assert isinstance(loss_function, torch.nn.CrossEntropyLoss)
    assert training.learning_rate == pytest.approx(5e-5)
    assert training.max_epochs == 200
    assert training.patience == 15

    checkpoint_path = tmp_path / "paper.pt"
    checkpoint_hash = save_paper_model_checkpoint(
        checkpoint_path,
        model=model,
        training_config=training,
        metadata={"task_id": "MODEL-ME-005", "fold_id": "smoke"},
    )
    loaded, checkpoint = load_paper_model_checkpoint(str(checkpoint_path))
    patches, masks, keypoints = _batch(batch_size=1)
    with torch.no_grad():
        expected = model(patches, masks, keypoints)
        actual = loaded.eval()(patches, masks, keypoints)
    assert checkpoint["model_schema_version"] == PAPER_MODEL_SCHEMA_VERSION
    assert len(checkpoint_hash) == 64
    assert torch.equal(expected, actual)
