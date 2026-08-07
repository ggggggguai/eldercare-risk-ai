from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.gait_tcn import (
    GaitTCNPredictor,
    GaitTCNTrainingConfig,
    LightweightGaitTCN,
    aggregate_participant_predictions,
    predict_gait_risk_scores,
    train_gait_tcn,
)
from elderly_monitoring.modules.fall_risk.kinecal_gait import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
    KinecalGaitPreparationConfig,
    build_canonical_sequence,
    make_windows,
    parse_kinecal_frame,
    prepare_kinecal_gait_dataset,
    stratified_participant_split,
)


SOURCE_JOINTS = {
    "ShoulderLeft": (-0.30, 0.80),
    "ShoulderRight": (0.30, 0.80),
    "ElbowLeft": (-0.45, 0.45),
    "ElbowRight": (0.45, 0.45),
    "WristLeft": (-0.50, 0.10),
    "WristRight": (0.50, 0.10),
    "HipLeft": (-0.20, 0.00),
    "HipRight": (0.20, 0.00),
    "KneeLeft": (-0.20, -0.45),
    "KneeRight": (0.20, -0.45),
    "AnkleLeft": (-0.25, -0.90),
    "AnkleRight": (0.25, -0.90),
}


def skeleton_frame_text(*, x_offset: float = 0.0, inferred: str | None = None) -> str:
    lines = []
    for index, (joint, (x, y)) in enumerate(SOURCE_JOINTS.items()):
        state = "Inferred" if joint == inferred else "Tracked"
        pixel_x = 960.0 + (x * 400.0) + x_offset
        pixel_y = 540.0 - (y * 400.0)
        lines.append(
            f"{joint} {state} {x:.4f} {y:.4f} 3.0 "
            f"{pixel_x:.4f} {pixel_y:.4f}"
        )
    lines.append("Head Tracked 0.0 1.1 3.0 960.0 100.0")
    return "\n".join(lines) + "\n"


def write_recording(root: Path, participant_number: str) -> None:
    skeleton_dir = (
        root
        / "kinecal"
        / participant_number
        / f"{participant_number}_3m-walk-Front-View"
        / "skel"
    )
    skeleton_dir.mkdir(parents=True)
    for index, tick in enumerate((1000, 1033, 1066, 1099)):
        (skeleton_dir / f"{tick}.txt").write_text(
            skeleton_frame_text(x_offset=float(index * 5)),
            encoding="utf-8",
        )


class KinecalGaitPreparationTest(unittest.TestCase):
    def test_extracts_twelve_common_joints_and_two_derived_centers(self) -> None:
        xy, quality = parse_kinecal_frame(
            skeleton_frame_text(inferred="AnkleLeft")
        )

        self.assertEqual(xy.shape, (14, 2))
        self.assertEqual(quality.shape, (14,))
        self.assertEqual(CANONICAL_GAIT_JOINTS[-2:], ("pelvis_center", "shoulder_center"))
        np.testing.assert_allclose(xy[12], [960.0, 540.0])
        np.testing.assert_allclose(xy[13], [960.0, 220.0])
        self.assertEqual(quality[10], 0.5)
        self.assertEqual(quality[12], 1.0)

    def test_marks_unprojectable_infinite_pixel_joint_as_missing(self) -> None:
        text = skeleton_frame_text().replace(
            "AnkleLeft Tracked -0.2500 -0.9000 3.0 860.0000 900.0000",
            "AnkleLeft Inferred -0.2500 -0.9000 3.0 -∞ -∞",
        )

        xy, quality = parse_kinecal_frame(text)

        self.assertTrue(np.isnan(xy[10]).all())
        self.assertEqual(quality[10], 0.0)

    def test_builds_root_centered_resampled_five_channel_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recording = Path(tempdir)
            for index, tick in enumerate((1000, 1033, 1066, 1099)):
                (recording / f"{tick}.txt").write_text(
                    skeleton_frame_text(x_offset=float(index * 4)),
                    encoding="utf-8",
                )

            sequence = build_canonical_sequence(
                recording,
                target_fps=30.0,
                max_gap_sec=0.10,
            )

        self.assertEqual(sequence.shape[1:], (14, 5))
        self.assertGreaterEqual(sequence.shape[0], 3)
        np.testing.assert_allclose(sequence[:, 12, :2], 0.0, atol=1e-6)
        self.assertTrue(np.isfinite(sequence).all())
        self.assertGreater(float(sequence[:, :, 4].mean()), 0.9)
        self.assertEqual(GAIT_TCN_CHANNELS, ("x", "y", "dx", "dy", "quality"))

    def test_uses_suffix_timestamp_when_frame_name_has_numeric_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recording = Path(tempdir)
            for index, tick in enumerate((1000, 1033, 1066)):
                (recording / f"72057594037946549_{tick}.txt").write_text(
                    skeleton_frame_text(x_offset=float(index)),
                    encoding="utf-8",
                )

            sequence = build_canonical_sequence(recording, target_fps=30.0)

        self.assertEqual(sequence.shape[0], 2)

    def test_windowing_pads_short_sequence_and_keeps_final_window(self) -> None:
        sequence = np.ones((10, 14, 5), dtype=np.float32)

        short_windows = make_windows(sequence, window_frames=16, stride_frames=8)
        long_windows = make_windows(
            np.ones((25, 14, 5), dtype=np.float32),
            window_frames=16,
            stride_frames=8,
        )

        self.assertEqual(len(short_windows), 1)
        self.assertEqual(short_windows[0][0].shape, (16, 14, 5))
        self.assertEqual(short_windows[0][1:], (10, 0, 10))
        self.assertTrue(np.all(short_windows[0][0][10:] == 0.0))
        self.assertEqual([item[3] for item in long_windows], [16, 24, 25])

    def test_participant_split_is_stratified_deterministic_and_leak_free(self) -> None:
        labels = {
            **{f"NF{index}": 0 for index in range(8)},
            **{f"FH{index}": 1 for index in range(8)},
        }

        first = stratified_participant_split(
            labels,
            seed=42,
            train_fraction=0.5,
            validation_fraction=0.25,
        )
        second = stratified_participant_split(
            labels,
            seed=42,
            train_fraction=0.5,
            validation_fraction=0.25,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(labels))
        for partition in ("train", "validation", "test"):
            partition_labels = {
                labels[participant]
                for participant, assigned in first.items()
                if assigned == partition
            }
            self.assertEqual(partition_labels, {0, 1})

    def test_prepares_npz_with_participant_level_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            raw = root / "raw"
            output = root / "prepared"
            raw.mkdir()
            rows = [
                ("SPPB1", "NF"),
                ("SPPB2", "NF"),
                ("SPPB3", "NF"),
                ("SPPB4", "FHs"),
                ("SPPB5", "FHm"),
                ("SPPB6", "FHs"),
            ]
            register_lines = ["part_id,group,age"]
            for participant, group in rows:
                register_lines.append(f"{participant},{group},70")
                write_recording(raw, participant.removeprefix("SPPB"))
            (raw / "register.csv").write_text(
                "\n".join(register_lines) + "\n",
                encoding="utf-8",
            )

            summary = prepare_kinecal_gait_dataset(
                raw,
                output,
                config=KinecalGaitPreparationConfig(
                    target_fps=30.0,
                    window_frames=8,
                    stride_frames=4,
                    seed=7,
                    train_fraction=1 / 3,
                    validation_fraction=1 / 3,
                ),
            )

            with np.load(output / "dataset.npz") as dataset:
                features = dataset["features"]
                participant_ids = dataset["participant_ids"]
                partitions = dataset["partitions"]
                labels = dataset["labels"]

            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(features.shape[1:], (8, 14, 5))
        self.assertEqual(summary["participant_count"], 6)
        self.assertEqual(metadata["label_mapping"], {"FHm": 1, "FHs": 1, "NF": 0})
        self.assertEqual(set(labels.tolist()), {0, 1})
        partition_by_participant: dict[str, set[str]] = {}
        for participant, partition in zip(participant_ids, partitions, strict=True):
            partition_by_participant.setdefault(str(participant), set()).add(str(partition))
        self.assertTrue(
            all(len(values) == 1 for values in partition_by_participant.values())
        )


class LightweightGaitTCNTest(unittest.TestCase):
    def test_model_outputs_binary_logits_and_stays_lightweight(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=32,
            dilations=(1, 2, 4),
        )

        logits = model(torch.randn(4, 128, 14, 5))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertLess(parameter_count, 100_000)

    def test_mask_only_quality_cannot_change_encoded_output(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
            use_quality_as_feature=False,
        ).eval()
        low_quality = torch.randn(2, 8, 14, 5)
        low_quality[..., 4] = 0.2
        high_quality = low_quality.clone()
        high_quality[..., 4] = 0.9

        with torch.inference_mode():
            low_logits = model(low_quality)
            high_logits = model(high_quality)

        torch.testing.assert_close(low_logits, high_logits)

    def test_mask_only_quality_keeps_binary_joint_visibility(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
            use_quality_as_feature=False,
        ).eval()
        captured: list[torch.Tensor] = []
        handle = model.input_projection[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        features = torch.randn(1, 8, 14, 5)
        features[..., 4] = 0.7
        features[:, :, 3, :4] = 0.0
        features[:, :, 3, 4] = 0.0

        try:
            with torch.inference_mode():
                model(features)
        finally:
            handle.remove()

        encoded = captured[0].transpose(1, 2).reshape(1, 8, 14, 5)
        self.assertTrue(torch.all(encoded[:, :, :3, 4] == 1.0))
        self.assertTrue(torch.all(encoded[:, :, 3, 4] == 0.0))

    def test_hierarchical_probability_is_gated_by_walking_probability(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
            use_quality_as_feature=False,
            hierarchical_walking_gate=True,
        ).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            model.walking_classifier[-1].bias.copy_(torch.tensor([2.0, -2.0]))
        features = torch.zeros(1, 8, 14, 5)
        features[..., 4] = 1.0

        with torch.inference_mode():
            probabilities = model.predict_probabilities(features)

        conditional = float(probabilities["conditional_gait_probability"][0])
        walking = float(probabilities["walking_probability"][0])
        risk = float(probabilities["gait_risk_probability"][0])
        self.assertAlmostEqual(risk, conditional * walking, places=6)
        self.assertLess(risk, 0.05)

    def test_aggregates_window_probabilities_by_participant(self) -> None:
        rows = aggregate_participant_predictions(
            participant_ids=["SPPB1", "SPPB1", "SPPB2"],
            labels=[0, 0, 1],
            probabilities=[0.1, 0.3, 0.8],
        )

        self.assertEqual(rows[0]["participant_id"], "SPPB1")
        self.assertAlmostEqual(rows[0]["probability"], 0.2)
        self.assertEqual(rows[0]["window_count"], 2)
        self.assertEqual(rows[1]["label"], 1)

    def test_checkpoint_inference_returns_window_gait_risk_scores(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "model.pt"
            torch.save(
                {
                    "schema_version": "gait-tcn-checkpoint-v1",
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "joint_count": 14,
                        "input_channels": 5,
                        "hidden_channels": 8,
                        "kernel_size": 5,
                        "dilations": [1],
                        "dropout": 0.0,
                        "class_count": 2,
                    },
                    "task": "gait_instability_vs_normal_activity",
                    "label_mapping": {"normal_activity": 0, "gait_instability": 1},
                },
                checkpoint_path,
            )

            scores = predict_gait_risk_scores(
                checkpoint_path,
                np.zeros((3, 8, 14, 5), dtype=np.float32),
                device="cpu",
                batch_size=2,
            )
            predictor = GaitTCNPredictor(
                checkpoint_path,
                device="cpu",
                window_frames=8,
                expected_task="gait_instability_vs_normal_activity",
            )
            coordinates = {
                "left_shoulder": (0.40, 0.25),
                "right_shoulder": (0.60, 0.25),
                "left_elbow": (0.35, 0.38),
                "right_elbow": (0.65, 0.38),
                "left_wrist": (0.32, 0.50),
                "right_wrist": (0.68, 0.50),
                "left_hip": (0.43, 0.52),
                "right_hip": (0.57, 0.52),
                "left_knee": (0.43, 0.70),
                "right_knee": (0.57, 0.70),
                "left_ankle": (0.42, 0.90),
                "right_ankle": (0.58, 0.90),
            }
            records = [
                {
                    "frame_id": frame_id,
                    "keypoints": [
                        {
                            "name": name,
                            "x_smooth": x + (frame_id * 0.002),
                            "y_smooth": y,
                            "valid": True,
                            "quality_weight": 0.9,
                            "is_jump_outlier": False,
                        }
                        for name, (x, y) in coordinates.items()
                    ],
                }
                for frame_id in range(8)
            ]
            record_score = predictor.predict_records(records)

        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all((0.0 <= scores) & (scores <= 1.0)))
        self.assertGreaterEqual(record_score, 0.0)
        self.assertLessEqual(record_score, 1.0)

    def test_runtime_predictor_rejects_checkpoint_for_the_wrong_task(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "model.pt"
            torch.save(
                {
                    "schema_version": "gait-tcn-checkpoint-v1",
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "joint_count": 14,
                        "input_channels": 5,
                        "hidden_channels": 8,
                        "kernel_size": 5,
                        "dilations": [1],
                        "dropout": 0.0,
                        "class_count": 2,
                    },
                    "task": "retrospective_fall_history_proxy_binary_classification",
                    "label_mapping": {"NF": 0, "FHs": 1, "FHm": 1},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint task"):
                GaitTCNPredictor(
                    checkpoint_path,
                    device="cpu",
                    expected_task="gait_instability_vs_normal_activity",
                )

    def test_runtime_predictor_enforces_checkpoint_observation_minimum(self) -> None:
        model = LightweightGaitTCN(
            joint_count=14,
            input_channels=5,
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "model.pt"
            torch.save(
                {
                    "schema_version": "gait-tcn-checkpoint-v1",
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "joint_count": 14,
                        "input_channels": 5,
                        "hidden_channels": 8,
                        "kernel_size": 5,
                        "dilations": [1],
                        "dropout": 0.0,
                        "class_count": 2,
                    },
                    "task": "gait_instability_vs_normal_activity",
                    "label_mapping": {"normal_activity": 0, "gait_instability": 1},
                    "input_contract": {
                        "window_frames": 8,
                        "target_fps": 4.0,
                        "max_gap_sec": 0.5,
                        "min_observed_frames": 4,
                    },
                },
                checkpoint_path,
            )
            predictor = GaitTCNPredictor(
                checkpoint_path,
                device="cpu",
                expected_task="gait_instability_vs_normal_activity",
            )
            sparse_record = {
                "frame_id": 100,
                "timestamp_sec": 1.75,
                "keypoints": [
                    {
                        "name": name,
                        "x_smooth": 0.5 + (x * 0.1),
                        "y_smooth": 0.5 - (y * 0.1),
                        "valid": True,
                        "quality_weight": 0.9,
                        "is_jump_outlier": False,
                    }
                    for name, (x, y) in SOURCE_JOINTS.items()
                ],
            }

            with self.assertRaisesRegex(ValueError, "observed pose frames"):
                predictor.predict_records([sparse_record])

    def test_trains_one_epoch_and_writes_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            raw = root / "raw"
            prepared = root / "prepared"
            training_output = root / "training"
            raw.mkdir()
            rows = [
                ("SPPB1", "NF"),
                ("SPPB2", "NF"),
                ("SPPB3", "NF"),
                ("SPPB4", "FHs"),
                ("SPPB5", "FHm"),
                ("SPPB6", "FHs"),
            ]
            (raw / "register.csv").write_text(
                "part_id,group,age\n"
                + "".join(f"{participant},{group},70\n" for participant, group in rows),
                encoding="utf-8",
            )
            for participant, _ in rows:
                write_recording(raw, participant.removeprefix("SPPB"))
            prepare_kinecal_gait_dataset(
                raw,
                prepared,
                config=KinecalGaitPreparationConfig(
                    window_frames=8,
                    stride_frames=4,
                    train_fraction=1 / 3,
                    validation_fraction=1 / 3,
                ),
            )

            summary = train_gait_tcn(
                prepared / "dataset.npz",
                training_output,
                config=GaitTCNTrainingConfig(
                    epochs=1,
                    batch_size=2,
                    hidden_channels=8,
                    dilations=(1,),
                    patience=1,
                    device="cpu",
                ),
            )
            metrics = json.loads(
                (training_output / "metrics.json").read_text(encoding="utf-8")
            )
            checkpoint_exists = (training_output / "best_model.pt").is_file()
            window_predictions_exist = (
                training_output / "test_window_predictions.jsonl"
            ).is_file()

        self.assertEqual(summary["epochs_completed"], 1)
        self.assertGreater(summary["parameter_count"], 0)
        self.assertTrue(checkpoint_exists)
        self.assertTrue(window_predictions_exist)
        self.assertEqual(
            metrics["task"], "retrospective_fall_history_proxy_binary_classification"
        )

    def test_cli_help_exposes_prepare_and_training_options(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        commands = (
            ("scripts/prepare/prepare_kinecal_gait_tcn.py", "--window-frames"),
            ("scripts/train/train_gait_tcn.py", "--epochs"),
        )
        for script, option in commands:
            completed = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
