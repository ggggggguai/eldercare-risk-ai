from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.sit_stand_continuous_tcn import (
    ContinuousSitStandTCN,
    ContinuousSitStandTCNConfig,
    SitStandStreamDecoderConfig,
    _loss,
    decode_sit_stand_stream,
    train_continuous_sit_stand_tcn,
)


class ContinuousSitStandTCNTest(unittest.TestCase):
    def test_boundary_positive_weight_penalizes_missed_boundaries(self) -> None:
        outputs = {
            "frame_logits": torch.zeros(1, 4, 3),
            "boundary_logits": torch.full((1, 4, 2), -2.0),
            "presence_logits": torch.zeros(1, 2),
            "direction_logits": torch.zeros(1, 2),
        }
        targets = torch.tensor([1])
        frame_targets = torch.ones(1, 4, dtype=torch.long)
        boundary_targets = torch.zeros(1, 4, 2)
        boundary_targets[0, 0, 0] = 1.0
        boundary_targets[0, -1, 1] = 1.0
        masks = torch.ones(1, 4)
        weights = torch.ones(1)

        unweighted = _loss(
            outputs,
            targets,
            frame_targets,
            boundary_targets,
            masks,
            weights,
            ContinuousSitStandTCNConfig(boundary_positive_weight=1.0),
        )
        weighted = _loss(
            outputs,
            targets,
            frame_targets,
            boundary_targets,
            masks,
            weights,
            ContinuousSitStandTCNConfig(boundary_positive_weight=8.0),
        )

        self.assertGreater(weighted["boundary"], unweighted["boundary"])

    def test_stream_decoder_confirms_and_closes_event_at_boundary(self) -> None:
        rows = [
            self._stream_row(0.00, presence=0.1),
            self._stream_row(0.25, presence=0.8, onset=0.9),
            self._stream_row(0.50, presence=0.9),
            self._stream_row(0.75, presence=0.9, offset=0.9),
            self._stream_row(1.00, presence=0.1),
        ]

        events = decode_sit_stand_stream(
            rows,
            video_id="video_1",
            stream_id="person_1:track_1",
            config=SitStandStreamDecoderConfig(
                confirmation_frames=2,
                state_threshold=0.5,
                presence_threshold=0.5,
                boundary_threshold=0.5,
            ),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["transition_type"], "sit_to_stand")
        self.assertEqual(events[0]["onset_time"], 0.25)
        self.assertEqual(events[0]["offset_time"], 0.75)
        self.assertEqual(events[0]["quality_state"], "valid")

    def test_stream_decoder_does_not_merge_opposite_directions(self) -> None:
        rows = [
            self._stream_row(0.00, presence=0.9, onset=0.9),
            self._stream_row(0.25, presence=0.9, offset=0.9),
            self._stream_row(
                0.50,
                presence=0.9,
                onset=0.9,
                direction="stand_to_sit",
            ),
            self._stream_row(
                0.75,
                presence=0.9,
                offset=0.9,
                direction="stand_to_sit",
            ),
        ]

        events = decode_sit_stand_stream(
            rows,
            video_id="video_1",
            stream_id="person_1:track_1",
            config=SitStandStreamDecoderConfig(
                confirmation_frames=1,
                event_merge_gap_sec=0.5,
            ),
        )

        self.assertEqual(
            [event["transition_type"] for event in events],
            ["sit_to_stand", "stand_to_sit"],
        )

    def test_stream_decoder_does_not_confirm_across_long_gap(self) -> None:
        events = decode_sit_stand_stream(
            [
                self._stream_row(0.0, presence=0.9, onset=0.9),
                self._stream_row(2.0, presence=0.9, offset=0.9),
            ],
            video_id="video_1",
            stream_id="person_1:track_1",
            config=SitStandStreamDecoderConfig(
                confirmation_frames=2,
                max_sequence_gap_sec=0.5,
            ),
        )

        self.assertEqual(events, [])

    def test_frame_outputs_are_causal(self) -> None:
        torch.manual_seed(1)
        model = ContinuousSitStandTCN(
            joint_count=14,
            input_channels=9,
            hidden_channels=8,
            dilations=(1, 2),
            dropout=0.0,
        ).eval()
        baseline = torch.randn(2, 8, 14, 9)
        changed = baseline.clone()
        changed[:, 5:] += 100.0

        with torch.no_grad():
            first = model(baseline)["frame_logits"]
            second = model(changed)["frame_logits"]

        torch.testing.assert_close(first[:, :5], second[:, :5])

    def test_training_requires_materialized_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, metadata = self._dataset(root, gate=False)
            with self.assertRaisesRegex(ValueError, "materialized development gate"):
                train_continuous_sit_stand_tcn(
                    data,
                    root / "output",
                    metadata_path=metadata,
                    config=ContinuousSitStandTCNConfig(epochs=1),
                    allow_provisional=True,
                )

    def test_one_epoch_smoke_writes_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, metadata = self._dataset(root, gate=True)
            output = root / "output"
            summary = train_continuous_sit_stand_tcn(
                data,
                output,
                metadata_path=metadata,
                config=ContinuousSitStandTCNConfig(
                    epochs=1,
                    batch_size=4,
                    hidden_channels=8,
                    dilations=(1, 2),
                    dropout=0.0,
                    seed=7,
                    device="cpu",
                ),
                allow_provisional=True,
            )

            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertFalse(summary["test_access"]["test_evaluated"])

    @staticmethod
    def _stream_row(
        timestamp: float,
        *,
        presence: float,
        onset: float = 0.0,
        offset: float = 0.0,
        direction: str = "sit_to_stand",
    ) -> dict:
        if direction == "sit_to_stand":
            frame = [0.05, 0.9, 0.05]
            direction_probabilities = [0.9, 0.1]
        else:
            frame = [0.05, 0.05, 0.9]
            direction_probabilities = [0.1, 0.9]
        return {
            "timestamp_sec": timestamp,
            "presence_score": presence,
            "frame_probabilities": frame,
            "boundary_probabilities": [onset, offset],
            "direction_probabilities": direction_probabilities,
        }

    @staticmethod
    def _dataset(root: Path, *, gate: bool) -> tuple[Path, Path]:
        sample_count, frames, joints, channels = 12, 8, 14, 9
        rng = np.random.default_rng(3)
        features = rng.normal(size=(sample_count, frames, joints, channels)).astype(
            np.float32
        )
        features[..., -1] = 1.0
        targets = np.asarray([0, 1, 2, 0, 1, 2] * 2, dtype=np.int64)
        frame_targets = np.repeat(targets[:, None], frames, axis=1)
        boundary_targets = np.zeros((sample_count, frames, 2), dtype=np.float32)
        boundary_targets[targets > 0, 0, 0] = 1.0
        boundary_targets[targets > 0, -1, 1] = 1.0
        supervision_masks = np.ones((sample_count, frames), dtype=np.float32)
        partitions = np.asarray(["train"] * 6 + ["validation"] * 6)
        data = root / "dataset.npz"
        np.savez(
            data,
            features=features,
            targets=targets,
            frame_targets=frame_targets,
            boundary_targets=boundary_targets,
            supervision_masks=supervision_masks,
            sample_weights=np.ones(sample_count, dtype=np.float32),
            sample_ids=np.asarray([f"sample_{index}" for index in range(sample_count)]),
            partitions=partitions,
        )
        metadata = root / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": "sit-stand-continuous-dataset-v1",
                    "task": "sit_stand_event_localization_v1",
                    "status": "development_provisional",
                    "continuous_model_training_allowed": gate,
                    "materialized_development_gate": {"passed": gate},
                    "semantic_arrays_sha256": "fixture",
                    "test_pose_read": False,
                    "test_features_generated": False,
                    "test_evaluated": False,
                }
            ),
            encoding="utf-8",
        )
        return data, metadata


if __name__ == "__main__":
    unittest.main()
