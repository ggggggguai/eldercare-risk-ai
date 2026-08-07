from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm import (
    DSTM,
    DSTMConfig,
    load_dstm_checkpoint,
    save_dstm_checkpoint,
    symmetric_info_nce,
    temporal_neighbor_bias,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm_dataset import (
    DSTMArtifactCollator,
    DSTMArtifactDataset,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm_preprocess import (
    DSTMTemporalConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.fold_reducers import (
    FoldOnlyReducer,
    FoldReducerConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_dataset import (
    PAPER_ORDERED_LANDMARK_INDICES,
    PAPER_REGION_COUNTS,
    PAPER_REGION_OFFSETS,
)


def _spatial_record(path: Path, sample_id: str = "sample", label: int = 1) -> dict[str, object]:
    generator = np.random.default_rng(11)
    np.savez_compressed(
        path,
        patches=generator.normal(size=(43, 75)).astype(np.float32),
        region_ids=np.repeat(np.arange(6), PAPER_REGION_COUNTS).astype(np.int64),
        region_offsets=np.asarray(PAPER_REGION_OFFSETS, dtype=np.int64),
        keypoints=generator.uniform(1.0, 30.0, size=(43, 2)).astype(np.float32),
        ordered_landmark_indices=np.asarray(PAPER_ORDERED_LANDMARK_INDICES, dtype=np.int64),
        label=np.asarray(label, dtype=np.int64),
    )
    return {
        "sample_id": sample_id,
        "subject_id": "s01",
        "sequence_id": sample_id,
        "source_dataset": "smic_hs",
        "artifact_path": path.as_posix(),
        "label": label,
        "quality_status": "pass",
        "apex_boundary_status": "interior",
    }


def _temporal_record(path: Path, sample_id: str = "sample", label: int = 1) -> dict[str, object]:
    generator = np.random.default_rng(12)
    sequence = generator.normal(size=(5, 3, 32, 32)).astype(np.float32)
    np.savez_compressed(
        path,
        flow_sequence=sequence,
        frame_pairs=np.column_stack((np.arange(5), np.arange(1, 6))).astype(np.int64),
        label=np.asarray(label, dtype=np.int64),
    )
    return {
        "sample_id": sample_id,
        "subject_id": "s01",
        "sequence_id": sample_id,
        "source_dataset": "smic_hs",
        "artifact_path": path.as_posix(),
        "frame_count": 6,
        "onset_index": 0,
        "apex_index": 5,
        "sequence_start": "onset",
        "sequence_end": "apex",
        "label": label,
    }


def _small_model() -> DSTM:
    return DSTM(
        torch.eye(6),
        DSTMConfig(
            hidden_dim=32,
            temporal_cnn_feature_dim=16,
            temporal_reducer_dim=8,
            transformer_layers=3,
            transformer_heads=4,
            gcn_layers=2,
            dropout=0.0,
        ),
    )


def test_temporal_neighbor_bias_has_local_guidance_and_padding_mask() -> None:
    features = torch.tensor(
        [[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [0.1, 0.0], [0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, True, True, False]])
    bias = temporal_neighbor_bias(features, mask, neighbor_k=1, bias_value=0.3)
    assert bias.shape == (1, 5, 5)
    assert bias[0, 0, 3].item() == pytest.approx(0.3)
    assert bias[0, 0, 1].item() == pytest.approx(0.0)
    assert bias[0, 0, 4].item() < -1e20
    assert torch.all(bias[0, 4] == 0)


def test_dstm_temporal_config_is_frozen_to_thesis_onset_apex() -> None:
    assert DSTMTemporalConfig().sequence_end == "apex"
    with pytest.raises(ValueError, match="onset to apex"):
        DSTMTemporalConfig(sequence_end="offset")


def test_info_nce_is_finite_and_has_gradients() -> None:
    spatial = torch.randn(4, 8, requires_grad=True)
    temporal = torch.randn(4, 8, requires_grad=True)
    loss = symmetric_info_nce(spatial, temporal, temperature=0.07)
    loss.backward()
    assert torch.isfinite(loss)
    assert spatial.grad is not None and torch.isfinite(spatial.grad).all()
    assert temporal.grad is not None and torch.isfinite(temporal.grad).all()


def test_fold_only_reducer_rejects_forbidden_subjects_and_transforms() -> None:
    features = np.random.default_rng(3).normal(size=(30, 10)).astype(np.float32)
    reducer = FoldOnlyReducer(FoldReducerConfig(kind="pca", n_components=4))
    with pytest.raises(ValueError, match="forbidden test"):
        reducer.fit(
            features,
            sample_ids=[f"sample-{i}" for i in range(30)],
            subject_ids=["s01"] * 30,
            fold_id="fold",
            forbidden_subject_ids=["s01"],
        )
    reducer.fit(
        features,
        sample_ids=[f"sample-{i}" for i in range(30)],
        subject_ids=["s01"] * 30,
        fold_id="fold",
    )
    transformed = reducer.transform(features[:3])
    assert transformed.shape == (3, 4)
    assert reducer.metadata()["fit_transform_called_in_forward"] is False


def test_dstm_dataset_pads_variable_sequences_and_preserves_mask(tmp_path: Path) -> None:
    spatial_path = tmp_path / "spatial.npz"
    temporal_path = tmp_path / "temporal.npz"
    spatial = _spatial_record(spatial_path)
    temporal = _temporal_record(temporal_path)
    dataset = DSTMArtifactDataset([spatial], [temporal])
    batch = DSTMArtifactCollator()([dataset[0]])
    assert batch["flow_sequences"].shape == (1, 5, 3, 32, 32)
    assert batch["temporal_mask"].tolist() == [[True, True, True, True, True]]
    assert batch["temporal_lengths"].tolist() == [5]


def test_dstm_forward_backward_gate_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = _small_model().train()
    batch_size, time = 3, 7
    patches = torch.randn(batch_size, 6, 12, 75)
    masks = torch.zeros(batch_size, 6, 12, dtype=torch.bool)
    for region, count in enumerate(PAPER_REGION_COUNTS):
        masks[:, region, :count] = True
    patches[~masks] = 0.0
    temporal_features = torch.randn(batch_size, time, 8)
    temporal_mask = torch.tensor(
        [[True] * 7, [True] * 5 + [False] * 2, [True] * 6 + [False]],
        dtype=torch.bool,
    )
    logits, aux = model(
        patches,
        masks,
        temporal_features=temporal_features,
        temporal_mask=temporal_mask,
        return_aux=True,
    )
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss = loss + model.config.alignment_lambda * aux["alignment_loss"]
    loss.backward()
    assert logits.shape == (batch_size, 3)
    assert torch.isfinite(loss)
    assert torch.all((aux["gate"] >= 0) & (aux["gate"] <= 1))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint = tmp_path / "dstm.pt"
    digest = save_dstm_checkpoint(
        checkpoint,
        model=model,
        training_config={"learning_rate": 1e-4},
        metadata={"task_id": "MODEL-ME-006", "fold_id": "smoke"},
    )
    restored, metadata = load_dstm_checkpoint(checkpoint)
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(
            patches,
            masks,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
        )
        actual = restored(
            patches,
            masks,
            temporal_features=temporal_features,
            temporal_mask=temporal_mask,
        )
    assert len(digest) == 64
    assert metadata["model_schema_version"] == "dstm_paper_chapter4_v2"
    assert torch.equal(expected, actual)


def test_dstm_temporal_path_uses_transform_only_reducer() -> None:
    class TransformOnly:
        fitted = True
        output_dim = 8

        def __init__(self) -> None:
            self.calls = 0

        def transform(self, values: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.zeros((len(values), self.output_dim), dtype=np.float32)

        def fit_transform(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
            raise AssertionError("DSTM must never call fit_transform in forward")

    reducer = TransformOnly()
    model = _small_model().eval()
    patches = torch.randn(1, 6, 12, 75)
    masks = torch.zeros(1, 6, 12, dtype=torch.bool)
    for region, count in enumerate(PAPER_REGION_COUNTS):
        masks[:, region, :count] = True
    flow = torch.randn(1, 5, 3, 32, 32)
    with torch.no_grad():
        logits = model(
            patches,
            masks,
            flow_sequences=flow,
            temporal_mask=torch.ones(1, 5, dtype=torch.bool),
            temporal_reducer=reducer,
        )
    assert logits.shape == (1, 3)
    assert reducer.calls == 1
