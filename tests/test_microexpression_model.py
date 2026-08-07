from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dataset import (
    MicroexpressionArtifactDataset,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.metrics import (
    classification_metrics,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.model import (
    MHSSATGCN,
    MHSSATGCNConfig,
    load_model_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.splits import (
    build_smic_loso_splits,
    validate_smic_loso_splits,
)


def _config() -> MHSSATGCNConfig:
    return MHSSATGCNConfig(
        hidden_dim=32,
        region_embedding_dim=8,
        intra_depth=1,
        intra_heads=4,
        inter_depth=1,
        inter_heads=4,
        gcn_layers=1,
        dropout=0.0,
    )


def _batch(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    patches = torch.randn(batch_size, 72, 147, generator=generator)
    region_ids = torch.arange(6).repeat_interleave(12).repeat(batch_size, 1)
    masks = torch.zeros(batch_size, 72)
    valid_counts = (5, 5, 6, 6, 9, 12)
    for region, count in enumerate(valid_counts):
        start = region * 12
        masks[:, start : start + count] = 1
    keypoints = torch.rand(batch_size, 72, 2, generator=generator) * 31.0
    keypoints[masks == 0] = 0
    return patches, region_ids, masks, keypoints


def test_model_forward_backward_and_batch_adjacency() -> None:
    model = MHSSATGCN(_config())
    patches, region_ids, masks, keypoints = _batch()
    logits, aux = model(
        patches, region_ids, masks, keypoints, return_aux=True
    )
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
    loss.backward()

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert aux["adjacency"].shape == (2, 6, 6)
    assert torch.isfinite(aux["adjacency"]).all()
    assert not torch.equal(aux["adjacency"][0], aux["adjacency"][1])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_masked_padding_cannot_change_logits() -> None:
    torch.manual_seed(11)
    model = MHSSATGCN(_config()).eval()
    patches, region_ids, masks, keypoints = _batch(batch_size=1)
    changed_patches = patches.clone()
    changed_keypoints = keypoints.clone()
    changed_patches[masks == 0] = 1000.0
    changed_keypoints[masks == 0] = 1000.0

    with torch.no_grad():
        original = model(patches, region_ids, masks, keypoints)
        changed = model(changed_patches, region_ids, masks, changed_keypoints)
    assert torch.allclose(original, changed, atol=1e-6, rtol=1e-6)


def test_checkpoint_round_trip_is_strict(tmp_path: Path) -> None:
    model = MHSSATGCN(_config()).eval()
    checkpoint_path = tmp_path / "fold.pt"
    torch.save(
        {
            "model_schema_version": "mhssa_tgcn_v1",
            "model_config": model.config.as_dict(),
            "model_state_dict": model.state_dict(),
            "fold_id": "smoke",
        },
        checkpoint_path,
    )
    loaded, checkpoint = load_model_checkpoint(str(checkpoint_path))
    patches, region_ids, masks, keypoints = _batch(batch_size=1)
    with torch.no_grad():
        expected = model(patches, region_ids, masks, keypoints)
        actual = loaded.eval()(patches, region_ids, masks, keypoints)
    assert checkpoint["fold_id"] == "smoke"
    assert torch.equal(expected, actual)


def test_artifact_dataset_loads_fixed_arrays(tmp_path: Path) -> None:
    artifact_path = tmp_path / "sample.npz"
    patches, region_ids, masks, keypoints = _batch(batch_size=1)
    np.savez_compressed(
        artifact_path,
        patches=patches[0].numpy().astype(np.float32),
        region_ids=region_ids[0].numpy().astype(np.int64),
        masks=masks[0].numpy().astype(np.float32),
        keypoints=keypoints[0].numpy().astype(np.float32),
        label=np.asarray(1, dtype=np.int64),
    )
    record = {
        "sample_id": "smic_hs__s01__sample",
        "subject_id": "s01",
        "artifact_path": artifact_path.as_posix(),
        "label": 1,
    }
    dataset = MicroexpressionArtifactDataset([record])
    item = dataset[0]
    assert item["patches"].shape == (72, 147)
    assert item["label"].item() == 1
    assert item["sample_id"] == record["sample_id"]


def test_subject_loso_split_has_no_subject_or_sample_leakage() -> None:
    records = []
    for subject_number in range(1, 9):
        subject = f"s{subject_number:02d}"
        for label in (0, 1, 2):
            records.append(
                {
                    "sample_id": f"smic_hs__{subject}__sample_{label}",
                    "source_dataset": "smic_hs",
                    "subject_id": subject,
                    "label": label,
                }
            )
    split_manifest = build_smic_loso_splits(
        records, validation_subject_count=2
    )
    validate_smic_loso_splits(split_manifest)

    assert len(split_manifest["folds"]) == 8
    assert all(
        fold["selection_authority"]["test_used_for_selection"] is False
        for fold in split_manifest["folds"]
    )


def test_metrics_include_required_macro_and_per_class_results() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2]
    )
    assert metrics["uf1"] == metrics["macro_f1"]
    assert metrics["uar"] == metrics["macro_recall"]
    assert set(metrics["per_class"]) == {"negative", "positive", "surprise"}
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
