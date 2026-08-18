from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.fall_event_continuous import (
    FALL_EVENT_CONTINUOUS_JOINTS,
)
from elderly_monitoring.modules.fall_risk.fall_event_continuous_tcn import (
    ContinuousFallTCN,
    ContinuousFallTCNConfig,
    ContinuousFallTCNRuntimePredictor,
    ContinuousFallTCNShadowPredictor,
    train_continuous_fall_tcn,
)


def test_model_is_prefix_causal() -> None:
    torch.manual_seed(1)
    model = ContinuousFallTCN(
        input_channels=340,
        hidden_channels=8,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
    ).eval()
    features = torch.randn(1, 340, 16)
    features[:, 14, :] = 1.0
    changed = features.clone()
    changed[:, :, -1] += 10.0

    with torch.no_grad():
        _, onset = model(features)
        _, changed_onset = model(changed)

    assert onset.shape == (1, 16)
    torch.testing.assert_close(onset[:, :-1], changed_onset[:, :-1])


def test_short_training_writes_provisional_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rng = np.random.default_rng(7)
        features = rng.normal(size=(20, 16, 17, 20)).astype(np.float32)
        targets = np.asarray([0, 1] * 10, dtype=np.int64)
        features[targets == 1, :, 11:13, 8] += 2.0
        partitions = np.asarray(["train"] * 16 + ["validation"] * 4)
        dataset = root / "dataset.npz"
        np.savez_compressed(
            dataset,
            features=features,
            presence_targets=targets,
            onset_targets=np.full(20, 8, dtype=np.int64),
            onset_masks=np.asarray([1 if target else 0 for target in targets], dtype=np.float32),
            sample_weights=np.ones(20, dtype=np.float32),
            presence_loss_weights=np.ones(20, dtype=np.float32),
            onset_loss_weights=np.ones(20, dtype=np.float32),
            partitions=partitions,
        )
        metadata = root / "metadata.json"
        metadata.write_text(json.dumps({"schema_version": "fall-event-continuous-dataset-v1"}), encoding="utf-8")

        result = train_continuous_fall_tcn(
            dataset,
            root / "training",
            metadata_path=metadata,
            config=ContinuousFallTCNConfig(
                hidden_channels=8,
                dilations=(1, 2),
                dropout=0.0,
                batch_size=4,
                epochs=2,
                patience=2,
                seed=7,
            ),
        )
        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
        metrics = json.loads((root / "training" / "metrics.json").read_text(encoding="utf-8"))
        assert "validation_probabilities" not in metrics["validation"]
        assert "validation_targets" not in metrics["validation"]

    assert checkpoint["schema_version"] == "fall-event-continuous-tcn-v1"
    assert checkpoint["test_evaluated"] is False
    assert checkpoint["main_path_replacement"] is False
    assert "validation_probabilities" not in checkpoint["validation_metrics"]
    assert "validation_targets" not in checkpoint["validation_metrics"]


def test_shadow_predictor_adapts_causal_pose_to_flattened_tcn_input() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model = ContinuousFallTCN(
            input_channels=340,
            hidden_channels=4,
            kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        )
        checkpoint = root / "best_model.pt"
        torch.save(
            {
                "schema_version": "fall-event-continuous-tcn-v1",
                "status": "development_provisional",
                "task": "fall_event_continuous_presence_onset",
                "model_config": {
                    "input_channels": 340,
                    "hidden_channels": 4,
                    "kernel_size": 3,
                    "dilations": [1],
                    "dropout": 0.0,
                },
                "normalization": {
                    "mean": np.zeros((17, 20), dtype=np.float32).tolist(),
                    "std": np.ones((17, 20), dtype=np.float32).tolist(),
                },
                "state_dict": model.state_dict(),
                "test_evaluated": False,
                "main_path_replacement": False,
            },
            checkpoint,
        )
        records = []
        for frame_id in range(40):
            records.append(
                {
                    "person_id": "elder-1",
                    "track_id": 1,
                    "timestamp_sec": frame_id / 8.0,
                    "coordinate_system": "image_normalized_0_1",
                    "bbox": [0.2, 0.1, 0.8, 0.9],
                    "keypoints": [
                        {
                            "name": name,
                            "x": 0.4,
                            "y": 0.3 + index * 0.01,
                            "score": 0.9,
                            "valid": True,
                        }
                        for index, name in enumerate(FALL_EVENT_CONTINUOUS_JOINTS)
                    ],
                }
            )
        predictor = ContinuousFallTCNShadowPredictor([checkpoint], device="cpu")
        result = predictor.predict_records(records)
        runtime_predictor = ContinuousFallTCNRuntimePredictor(
            [checkpoint], device="cpu"
        )
        runtime_result = runtime_predictor.predict_records(records)

    assert result["fall_event_tcn_shadow_status"] == "provisional_shadow"
    assert 0.0 <= result["fall_event_tcn_shadow_score"] <= 1.0
    assert result["fall_event_tcn_shadow_observed_frame_count"] == 32
    assert runtime_result["fall_event_tcn_status"] == "valid"
    assert runtime_result["fall_event_tcn_score"] == result["fall_event_tcn_shadow_score"]
    assert runtime_result["fall_event_tcn_observed_frame_count"] == 32
