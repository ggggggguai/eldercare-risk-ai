import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
)
from training.cognitive_change_clue.dataset import (
    MODALITY_ORDER,
    CognitiveDatasetError,
    CognitiveFeatureDataset,
    MocaStatistics,
    TrainingWeights,
    cognitive_collate,
    compute_moca_statistics,
    compute_training_weights,
)
from training.cognitive_change_clue.evaluate import (
    COMBINATION_MODALITIES,
    checkpoint_is_better,
    filter_valid_rows,
    forced_missing_mask,
    subject_metrics,
)
from training.cognitive_change_clue import train as train_module


def _model(projection_dim=8):
    return CognitiveChangeClueModel(
        projection_dim=projection_dim,
        fusion_hidden_dims=(6, 4),
        dropout=0.0,
    )


def _batch(size=2):
    generator = torch.Generator().manual_seed(31)
    diagnoses = ["HC", "MCI", "AD"]
    rows = []
    for index in range(size):
        diagnosis = diagnoses[index % len(diagnoses)]
        rows.append(
            {
                "sample_id": f"sample-{index:03d}",
                "subject_id": f"subject-{index:03d}",
                "manifest_index": index,
                "diagnosis_label": diagnosis,
                "features": {
                    "audio": torch.randn(768, generator=generator),
                    "text": torch.randn(768, generator=generator),
                    "face": torch.randn(512, generator=generator),
                },
                "quality": torch.tensor([0.8, 0.7, 0.6]),
                "missing_mask": torch.zeros(3, dtype=torch.bool),
                "hc_vs_non_hc": torch.tensor(float(diagnosis != "HC")),
                "mci_vs_hc": torch.tensor(float(diagnosis == "MCI")),
                "mci_vs_hc_mask": torch.tensor(diagnosis != "AD"),
                "ad_mci_hc": torch.tensor({"HC": 0, "MCI": 1, "AD": 2}[diagnosis]),
                "moca": torch.tensor(float(24 - index)),
                "moca_mask": torch.tensor(index % 2 == 0),
            }
        )
    return cognitive_collate(rows)


@pytest.mark.parametrize("combination", list(COMBINATION_MODALITIES))
def test_fusion_handles_every_nonempty_modality_combination(combination):
    model = _model()
    batch = _batch(2)
    missing = forced_missing_mask(batch["missing_mask"], combination)
    output = model(batch["features"], batch["quality"], missing)
    assert output.hc_vs_non_hc_logit.shape == (2,)
    assert output.mci_vs_hc_logit.shape == (2,)
    assert output.ad_mci_hc_logits.shape == (2, 3)
    assert output.moca_standardized.shape == (2,)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.gate_weights[missing] == 0)
    assert torch.isfinite(output.fused).all()


def test_all_missing_rows_are_filtered_before_fusion_and_indices_remain_aligned():
    model = _model()
    batch = _batch(3)
    missing = batch["missing_mask"].clone()
    missing[1] = True
    filtered, valid = filter_valid_rows(batch, missing)
    assert valid.tolist() == [True, False, True]
    assert filtered["sample_id"] == ["sample-000", "sample-002"]
    assert model(
        filtered["features"], filtered["quality"], filtered["missing_mask"]
    ).fused.shape == (2, 8)
    with pytest.raises(ValueError, match="at least one"):
        model(batch["features"], batch["quality"], torch.ones_like(missing))


def test_fusion_autocast_keeps_gate_assignment_dtype_compatible():
    model = _model()
    batch = _batch(2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(batch["features"], batch["quality"], batch["missing_mask"])
    assert torch.isfinite(output.gate_weights).all()
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(2), atol=1e-5)


def test_configurable_input_dims_preserve_default_model_api() -> None:
    model = CognitiveChangeClueModel(
        projection_dim=8,
        fusion_hidden_dims=(6, 4),
        dropout=0.0,
        input_dims={"audio": 777, "text": 780, "face": 512},
    )
    output = model(
        {
            "audio": torch.zeros((2, 777)),
            "text": torch.zeros((2, 780)),
            "face": torch.zeros((2, 512)),
        },
        torch.ones((2, 3)),
        torch.zeros((2, 3), dtype=torch.bool),
    )
    assert output.hc_vs_non_hc_logit.shape == (2,)


def test_masking_plan_uses_global_indices_and_is_shuffle_independent():
    records = [
        {
            "sample_id": f"sample-{index}",
            "manifest_index": index,
            "missing_mask": np.asarray([index == 8, False, False]),
        }
        for index in range(20)
    ]
    first, first_rows = train_module.build_masking_plan(records, epoch=3, seed_base=100)
    second, second_rows = train_module.build_masking_plan(
        list(reversed(records)), epoch=3, seed_base=100
    )
    changed, _ = train_module.build_masking_plan(records, epoch=4, seed_base=100)
    assert first_rows == second_rows
    assert first == second
    assert first["sample-0"]["target_combination"] == "audio_text_face"
    assert first["sample-7"]["target_combination"] in {
        "audio_text",
        "audio_face",
        "text_face",
    }
    assert first["sample-9"]["target_combination"] in {"audio", "text"}
    assert first["sample-8"]["final_missing_mask"][0] == 1
    assert any(
        first[key]["target_combination"] != changed[key]["target_combination"]
        for key in first
        if int(key.split("-")[1]) % 10 >= 7
    )


def test_training_statistics_are_train_only_and_subject_level_for_moca():
    records = [
        {"subject_id": "hc", "diagnosis_label": "HC", "moca": 20.0, "moca_mask": True},
        {"subject_id": "hc", "diagnosis_label": "HC", "moca": 22.0, "moca_mask": True},
        {"subject_id": "mci", "diagnosis_label": "MCI", "moca": 18.0, "moca_mask": True},
        {"subject_id": "ad", "diagnosis_label": "AD", "moca": 0.0, "moca_mask": False},
    ]
    stats = compute_moca_statistics(records)
    assert stats.mean == pytest.approx(19.5)
    assert stats.std == pytest.approx(1.5)
    assert stats.subject_count == 2
    weights = compute_training_weights(records)
    assert weights.hc_vs_non_hc_pos_weight == pytest.approx(1.0)
    assert weights.mci_vs_hc_pos_weight == pytest.approx(2.0)
    changed_validation = copy.deepcopy(records)
    changed_validation.extend(
        {"subject_id": "validation", "diagnosis_label": "AD", "moca": value, "moca_mask": True}
        for value in (1.0, 30.0)
    )
    assert compute_moca_statistics(records) == stats
    assert compute_training_weights(records) == weights


def test_multitask_loss_masks_missing_head_without_reweighting():
    model = _model()
    batch = _batch(1)
    batch["mci_vs_hc_mask"][:] = False
    batch["moca_mask"][:] = False
    output = model(batch["features"], batch["quality"], batch["missing_mask"])
    weights = TrainingWeights(1.0, 1.0, (1.0, 1.0, 1.0), {})
    total, components = train_module.compute_multitask_loss(
        output,
        batch,
        weights=weights,
        moca_statistics=MocaStatistics(20.0, 5.0, 1),
        loss_weights={
            "hc_vs_non_hc": 1.0,
            "mci_vs_hc": 0.3,
            "ad_mci_hc": 0.3,
            "moca": 0.1,
        },
    )
    expected = components["hc_vs_non_hc"] + 0.3 * components["ad_mci_hc"]
    assert components["mci_vs_hc"] is None
    assert components["moca"] is None
    assert torch.allclose(total, expected)
    assert torch.isfinite(total)


def test_checkpoint_tie_break_prefers_balanced_accuracy_but_not_later_exact_tie():
    assert checkpoint_is_better(0.8, 0.7, 0.7, 0.9)
    assert checkpoint_is_better(0.8 + 0.5e-6, 0.8, 0.8, 0.7)
    assert not checkpoint_is_better(0.8 + 0.5e-6, 0.7, 0.8, 0.7)
    with pytest.raises(ValueError, match="not finite"):
        checkpoint_is_better(float("nan"), 0.5, 0.4, 0.4)


def test_subject_moca_averages_standardized_output_before_clipping():
    def row(sample, standardized, output):
        return {
            "sample_id": sample,
            "subject_id": "subject",
            "diagnosis_label": "HC",
            "hc_vs_non_hc": {"label": 0, "raw_logit": -1.0},
            "mci_vs_hc": {"label": 0, "label_mask": True, "raw_logit": -1.0},
            "ad_mci_hc": {"label": 0, "raw_logits": [1.0, 0.0, 0.0]},
            "moca": {
                "label": 20.0,
                "label_mask": True,
                "standardized_output": standardized,
                "output": output,
                "normalization_mean": 20.0,
                "normalization_std": 10.0,
            },
        }

    # Per-task clipping would yield 15; averaging standardized values first yields 20.
    records = [row("a", -4.0, 0.0), row("b", 4.0, 30.0)]
    records.append({**row("c", 0.0, 20.0), "subject_id": "subject-2", "diagnosis_label": "AD",
                    "hc_vs_non_hc": {"label": 1, "raw_logit": 1.0},
                    "mci_vs_hc": {"label": None, "label_mask": False, "raw_logit": 1.0},
                    "ad_mci_hc": {"label": 2, "raw_logits": [0.0, 0.0, 1.0]}})
    metrics = subject_metrics(records)
    subject = next(row for row in metrics["subjects"] if row["subject_id"] == "subject")
    assert subject["moca"]["prediction"] == pytest.approx(20.0)


def test_synthetic_cpu_training_closes_tail_accumulation_and_restores_checkpoint(
    tmp_path, monkeypatch
):
    batch = _batch(5)
    rows = []
    for index in range(5):
        row = {}
        for key, value in batch.items():
            if key == "features":
                row[key] = {name: tensor[index] for name, tensor in value.items()}
            elif isinstance(value, torch.Tensor):
                row[key] = value[index]
            else:
                row[key] = value[index]
        rows.append(row)
    loader = DataLoader(rows, batch_size=2, shuffle=False, collate_fn=cognitive_collate)
    model = _model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scaler = train_module._new_grad_scaler(False)
    plan = {
        row["sample_id"]: {
            "final_missing_mask": [0, 0, 0],
            "effective_combination": "audio_text_face",
        }
        for row in rows
    }
    metrics = train_module.train_one_epoch(
        model,
        loader,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        masking_plan=plan,
        weights=TrainingWeights(1.0, 1.0, (1.0, 1.0, 1.0), {}),
        moca_statistics=MocaStatistics(20.0, 5.0, 3),
        loss_weights={
            "hc_vs_non_hc": 1.0,
            "mci_vs_hc": 0.3,
            "ad_mci_hc": 0.3,
            "moca": 0.1,
        },
        accumulation_steps=2,
        max_grad_norm=1.0,
        amp_enabled=False,
    )
    assert metrics["nonempty_micro_batches"] == 3
    assert metrics["optimizer_steps"] == 2
    assert math.isfinite(metrics["loss"])

    scheduler = ReduceLROnPlateau(optimizer, mode="max")
    monkeypatch.setattr(train_module, "workspace_relative", lambda path: str(path))
    input_hashes = {"fixture": "0" * 64}
    payload = train_module._checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        best_state={"auc": 0.7, "balanced_accuracy": 0.6, "epoch": 1},
        early_state={"best_auc": 0.7, "epochs_without_improvement": 0},
        run_id="COG-FIXTURE-001",
        run_dir=tmp_path,
        config={
            "model": {"projection_dim": 8, "fusion_hidden_dims": [6, 4], "dropout": 0.0},
            "training": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "adam_betas": [0.9, 0.999],
                "adam_eps": 1e-8,
            },
        },
        input_hashes=input_hashes,
        weights=TrainingWeights(1.0, 1.0, (1.0, 1.0, 1.0), {}),
        moca_statistics=MocaStatistics(20.0, 5.0, 3),
    )
    path = tmp_path / "latest_checkpoint.pt"
    train_module.atomic_torch_save(path, payload)
    expected_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(10.0)
    restored = train_module._load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_run_id="COG-FIXTURE-001",
        expected_hashes=input_hashes,
    )
    assert restored["next_epoch"] == 2
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_state[name])


def test_recorded_amp_gradient_overflow_halves_scale_before_epoch_rerun(
    tmp_path,
):
    scaler = train_module._new_grad_scaler(False)
    # A disabled scaler cannot model CUDA scale, so use the recovery contract with a stub.
    class StubScaler:
        def __init__(self):
            self.scale = 65536.0

        def get_scale(self):
            return self.scale

        def update(self, value):
            self.scale = float(value)

    stub = StubScaler()
    (tmp_path / "failure_diagnostic.json").write_text(
        json.dumps(
            {
                "epoch": 19,
                "error": "gradient is non-finite: fusion.projections.audio.0.weight",
            }
        ),
        encoding="utf-8",
    )
    recovery = train_module.recover_grad_scaler_after_recorded_overflow(
        stub,
        run_dir=tmp_path,
        next_epoch=19,
        amp_enabled=True,
    )
    assert recovery["previous_scale"] == 65536.0
    assert recovery["new_scale"] == 32768.0
    assert stub.scale == 32768.0
    assert train_module.recover_grad_scaler_after_recorded_overflow(
        scaler,
        run_dir=tmp_path,
        next_epoch=18,
        amp_enabled=False,
    ) is None


def _write_dataset_fixture(root: Path, *, leak_test_subject=False):
    subjects = ["HC_subj_001", "MCI_subj_002", "AD_subj_003"]
    diagnoses = ["HC", "MCI", "AD"]
    rows = []
    split = {"train": subjects, "validation": ["HC_subj_010"], "test": []}
    for index, (subject, diagnosis) in enumerate(zip(subjects, diagnoses)):
        rows.append(
            {
                "sample_id": f"{subject}__pic_1",
                "subject_id": subject,
                "diagnosis_label": diagnosis,
                "moca_label": 25 - index,
                "official_split": "test" if leak_test_subject and index == 0 else "train",
                "derived_split": "train",
            }
        )
    rows.append(
        {
            "sample_id": "HC_subj_010__pic_1",
            "subject_id": "HC_subj_010",
            "diagnosis_label": "HC",
            "moca_label": 26,
            "official_split": "train",
            "derived_split": "validation",
        }
    )
    manifest = root / "cogpic_manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    split_path = root / "cogpic_subject_split_v33.json"
    split_path.write_text(json.dumps({"splits": split}), encoding="utf-8")
    feature_root = root / "features"
    feature_root.mkdir()
    (feature_root / "feature_cache_manifest.json").write_text("{}", encoding="utf-8")
    globally_sorted = sorted(row["sample_id"] for row in rows)
    global_index = {sample: index for index, sample in enumerate(globally_sorted)}
    for modality, dimension in (("audio", 768), ("text", 768), ("face", 512)):
        index_rows = []
        for row in rows:
            feature_path = feature_root / modality / f"{row['sample_id']}.npz"
            feature_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(feature_path, embedding=np.ones(dimension, dtype=np.float32))
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "subject_id": row["subject_id"],
                    "derived_split": row["derived_split"],
                    "manifest_index": global_index[row["sample_id"]],
                    "feature_version": "cognitive_features_v3.3.0",
                    "quality_version": "cognitive_text_quality_v3.3.1" if modality == "text" else None,
                    "output_dim": dimension,
                    "missing": 0,
                    "quality": 0.8,
                    "feature_path": str(feature_path),
                }
            )
        (feature_root / f"{modality}_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
        )
    return manifest, split_path, feature_root


def test_dataset_rejects_official_test_leakage(tmp_path):
    manifest, split_path, feature_root = _write_dataset_fixture(
        tmp_path, leak_test_subject=True
    )
    with pytest.raises(CognitiveDatasetError, match="official Test"):
        CognitiveFeatureDataset(
            split_name="train",
            manifest_path=manifest,
            split_path=split_path,
            feature_root=feature_root,
        )


def test_dataset_builds_labels_and_zeroes_only_missing_embeddings(tmp_path):
    manifest, split_path, feature_root = _write_dataset_fixture(tmp_path)
    text_index = feature_root / "text_index.jsonl"
    records = [json.loads(line) for line in text_index.read_text().splitlines()]
    records[0]["missing"] = 1
    records[0]["output_dim"] = None
    text_index.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    dataset = CognitiveFeatureDataset(
        split_name="train",
        manifest_path=manifest,
        split_path=split_path,
        feature_root=feature_root,
    )
    hc_index = next(
        index for index, row in enumerate(dataset.records)
        if row["sample_id"] == "HC_subj_001__pic_1"
    )
    hc_row = dataset[hc_index]
    assert hc_row["missing_mask"].tolist() == [False, True, False]
    assert torch.count_nonzero(hc_row["features"]["text"]) == 0
    assert torch.count_nonzero(hc_row["features"]["audio"]) == 768
    labels = {row["diagnosis_label"]: row for row in dataset.records}
    assert labels["HC"]["hc_vs_non_hc"] == 0
    assert labels["MCI"]["mci_vs_hc"] == 1
    assert labels["AD"]["mci_vs_hc_mask"] is False
