import json
from pathlib import Path

import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
)
from training.cognitive_change_clue import train_v34
from training.cognitive_change_clue import run_v34_resilient
from training.cognitive_change_clue.v34_data import SubjectFeatureDataset


def _record(sample_id, subject_id, diagnosis, missing=(False, False, False)):
    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "manifest_index": int(sample_id.rsplit("-", 1)[-1]),
        "diagnosis_label": diagnosis,
        "features": {
            "audio": np.ones(768, dtype=np.float32),
            "text": np.ones(768, dtype=np.float32) * 2,
            "face": np.ones(512, dtype=np.float32) * 3,
        },
        "quality": np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
        "missing_mask": np.asarray(missing, dtype=np.bool_),
        "hc_vs_non_hc": float(diagnosis != "HC"),
        "mci_vs_hc": float(diagnosis == "MCI"),
        "mci_vs_hc_mask": diagnosis != "AD",
        "ad_mci_hc": {"HC": 0, "MCI": 1, "AD": 2}[diagnosis],
        "moca": 25.0,
        "moca_mask": True,
    }


def _config():
    return {
        "training": {
            "seed": 1,
            "micro_batch_size": 2,
            "num_workers": 0,
            "train_shuffle": False,
            "validation_shuffle": False,
            "train_drop_last": False,
            "validation_drop_last": False,
        },
        "outputs": {"prediction_schema_version": "cognitive_v34_raw_prediction_v1"},
    }


def test_predict_raw_preserves_all_rows_and_uses_null_for_missing_branch():
    records = [
        _record("sample-1", "subject-hc", "HC"),
        _record("sample-2", "subject-mci", "MCI", missing=(False, True, False)),
    ]
    dataset = SubjectFeatureDataset(records, {"subject-hc", "subject-mci"})
    torch.manual_seed(1)
    model = CognitiveChangeClueModel(projection_dim=8, fusion_hidden_dims=(6, 4), dropout=0.0)
    rows = train_v34.predict_raw(
        model,
        dataset,
        config=_config(),
        device=torch.device("cpu"),
        candidate_id="V34-A0",
        outer_fold=0,
        split_role="outer_evaluation",
    )
    assert len(rows) == 2
    assert set(rows[0]) == train_v34.PREDICTION_FIELDS
    assert rows[1]["raw_logit"] is not None
    assert rows[1]["text_logit"] is None
    assert rows[1]["audio_logit"] is not None


def test_prediction_bundle_detects_hash_tampering(tmp_path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    row = {
        "schema_version": "cognitive_v34_raw_prediction_v1",
        "candidate_id": "V34-A0",
        "outer_fold": 0,
        "split_role": "inner_validation",
        "sample_id": "sample-1",
        "subject_id": "subject-1",
        "diagnosis": "HC",
        "label_hc_vs_non_hc": 0,
        "raw_logit": 0.0,
        "audio_logit": 0.0,
        "text_logit": 0.0,
        "face_logit": 0.0,
        "q_audio": 1.0,
        "q_text": 1.0,
        "q_face": 1.0,
        "audio_missing_mask": 0,
        "text_missing_mask": 0,
        "face_missing_mask": 0,
    }
    files = {}
    for role, file_name in train_v34.PREDICTION_FILES.items():
        role_row = {**row, "split_role": role}
        path = candidate_dir / file_name
        path.write_text(json.dumps(role_row) + "\n", encoding="utf-8")
        files[role] = {
            "path": file_name,
            "row_count": 1,
            "scored_row_count": 1,
            "sha256": train_v34.sha256_file(path),
        }
    (candidate_dir / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "cognitive_v34_prediction_manifest_v1",
                "candidate_id": "V34-A0",
                "outer_fold": 0,
                "split_sha256": "split",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    train_v34.verify_prediction_bundle(candidate_dir, expected_split_sha256="split")
    with (candidate_dir / train_v34.PREDICTION_FILES["outer_evaluation"]).open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write("{}\n")
    with pytest.raises(train_v34.V34TrainingError, match="hash mismatch"):
        train_v34.verify_prediction_bundle(candidate_dir, expected_split_sha256="split")


def test_completed_fold_path_extracts_fold_not_candidate_name(tmp_path):
    path = tmp_path / "fold_3" / "V34-A0" / "fold_metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    completed = sorted(
        int(item.parent.parent.name.split("_")[-1])
        for item in tmp_path.glob("fold_*/V34-A0/fold_metrics.json")
    )
    assert completed == [3]


def test_candidate_records_and_masking_never_expose_excluded_modalities():
    records = [_record("sample-1", "subject-hc", "HC")]
    candidate_records = train_v34._candidate_records(records, ("audio", "text"))
    assert candidate_records[0]["missing_mask"].tolist() == [False, False, True]
    for epoch in range(1, 20):
        plan, _ = train_v34._candidate_masking_plan(
            candidate_records,
            modalities=("audio", "text"),
            epoch=epoch,
            seed_base=20260805,
        )
        assert plan["sample-1"]["final_missing_mask"][2] == 1
        assert not all(plan["sample-1"]["final_missing_mask"])


def test_dataset_filters_on_natural_completeness_before_candidate_masking():
    records = [_record("sample-1", "subject-hc", "HC")]
    dataset = SubjectFeatureDataset(
        records,
        {"subject-hc"},
        complete_modalities=("audio", "text", "face"),
        available_modalities=("audio", "text"),
    )
    assert len(dataset) == 1
    assert dataset[0]["missing_mask"].tolist() == [False, False, True]


def test_main_head_policy_zeroes_all_auxiliary_losses():
    weights = train_v34._candidate_loss_weights(
        {"hc_vs_non_hc": 1.0, "mci_vs_hc": 0.3, "ad_mci_hc": 0.3, "moca": 0.1},
        "main_head",
    )
    assert weights == {
        "hc_vs_non_hc": 1.0,
        "mci_vs_hc": 0.0,
        "ad_mci_hc": 0.0,
        "moca": 0.0,
    }


def test_candidate_registry_freezes_opt_cog_002_semantics():
    assert train_v34.CANDIDATES["V34-A1"]["modalities"] == ("audio", "text")
    assert train_v34.CANDIDATES["V34-A1"]["heads"] == "four_head"
    assert train_v34.CANDIDATES["V34-A2"]["heads"] == "main_head"
    assert train_v34.CANDIDATES["V34-A3-Audio"]["modalities"] == ("audio",)
    assert train_v34.CANDIDATES["V34-A3-Text"]["modalities"] == ("text",)
    assert train_v34.CANDIDATES["V34-Face"]["modalities"] == ("face",)


def test_resilient_runner_retries_only_recorded_amp_overflow(tmp_path):
    candidate_dir = tmp_path / "fold_0" / "V34-A1"
    candidate_dir.mkdir(parents=True)
    diagnostic = candidate_dir / "failure_diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "error": "gradient is non-finite: fusion.weight",
                "latest_checkpoint": "checkpoint.pt",
            }
        ),
        encoding="utf-8",
    )
    assert run_v34_resilient._recoverable_failure(tmp_path, "V34-A1") is not None
    diagnostic.write_text(
        json.dumps({"error": "prediction hash mismatch", "latest_checkpoint": "checkpoint.pt"}),
        encoding="utf-8",
    )
    assert run_v34_resilient._recoverable_failure(tmp_path, "V34-A1") is None
