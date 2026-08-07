from __future__ import annotations

import unittest

import torch

from elderly_monitoring.modules.fall_risk.gait_action_pretraining import (
    ACTION_PRETRAINING_TARGETS,
    build_action_pretraining_rows,
    transfer_action_pretrained_encoder,
)
from elderly_monitoring.modules.fall_risk.gait_tcn import (
    GaitTCNTrainingConfig,
    LightweightGaitTCN,
)


def _label(
    action_id: str,
    *,
    label_id: str,
    training_tier: str = "primary",
    action_type_training_tier: str = "primary",
) -> dict[str, object]:
    return {
        "schema_version": "fall-risk-action-label-v3",
        "label_id": label_id,
        "asset_id": f"asset_{label_id}",
        "video_id": f"video_{label_id}",
        "source_group_id": f"source_{label_id}",
        "sample_group_id": f"sample_{label_id}",
        "subject_id": f"subject_{label_id}",
        "action_id": action_id,
        "action_family": "walk",
        "action_type": "test_action",
        "training_tier": training_tier,
        "action_type_training_tier": action_type_training_tier,
        "start_frame": 0,
        "end_frame_exclusive": 16,
        "start_time": 0.0,
        "end_time_exclusive": 4.0,
    }


def _assignment(label: dict[str, object], partition: str) -> dict[str, object]:
    return {
        "label_id": label["label_id"],
        "label_kind": "action",
        "asset_id": label["asset_id"],
        "video_id": label["video_id"],
        "source_group_id": label["source_group_id"],
        "sample_group_id": label["sample_group_id"],
        "training_tier": label["training_tier"],
        "split_group_id": f"split_{label['label_id']}",
        "partition": partition,
    }


def _manifest(label: dict[str, object], dataset: str) -> dict[str, object]:
    return {
        "asset_id": label["asset_id"],
        "video_id": label["video_id"],
        "dataset": dataset,
        "scene_region": "test_room",
        "eligibility": True,
    }


class ActionPretrainingSelectionTest(unittest.TestCase):
    def test_uses_all_supported_non_test_action_groups(self) -> None:
        labels = [
            _label("A01", label_id="normal"),
            _label("C03", label_id="stumble", training_tier="auxiliary"),
            _label("D02", label_id="fall"),
            _label("D04", label_id="post_fall"),
        ]
        rows = build_action_pretraining_rows(
            labels,
            assignment_rows=[
                _assignment(labels[0], "train"),
                _assignment(labels[1], "train"),
                _assignment(labels[2], "validation"),
                _assignment(labels[3], "train"),
            ],
            manifest_rows=[
                _manifest(labels[0], "ntu_rgbd"),
                _manifest(labels[1], "ntu_rgbd"),
                _manifest(labels[2], "le2i_imvia"),
                _manifest(labels[3], "le2i_imvia"),
            ],
        )

        targets = {row["label_id"]: row["pretraining_target"] for row in rows}
        self.assertEqual(
            targets,
            {
                "normal": "normal_locomotion",
                "stumble": "balance_loss",
                "fall": "fall_or_post_fall",
                "post_fall": "fall_or_post_fall",
            },
        )
        self.assertEqual(
            {
                row["label_id"]: row["pretraining_label"]
                for row in rows
            },
            {
                label_id: ACTION_PRETRAINING_TARGETS.index(target)
                for label_id, target in targets.items()
            },
        )

    def test_excludes_locked_test_unknown_and_invalid_action_type_labels(self) -> None:
        labels = [
            _label("A01", label_id="test"),
            _label("U01", label_id="unknown", training_tier="ignore", action_type_training_tier="ignore"),
            _label("C02", label_id="invalid_type", action_type_training_tier="ignore"),
            _label("B03", label_id="usable", action_type_training_tier="auxiliary"),
        ]
        rows = build_action_pretraining_rows(
            labels,
            assignment_rows=[
                _assignment(labels[0], "test"),
                _assignment(labels[1], "train"),
                _assignment(labels[2], "train"),
                _assignment(labels[3], "train"),
            ],
            manifest_rows=[_manifest(label, "le2i_imvia") for label in labels],
        )

        self.assertEqual([row["label_id"] for row in rows], ["usable"])

    def test_auxiliary_labels_cannot_enter_validation(self) -> None:
        label = _label(
            "B03",
            label_id="auxiliary",
            training_tier="auxiliary",
            action_type_training_tier="auxiliary",
        )

        [row] = build_action_pretraining_rows(
            [label],
            assignment_rows=[_assignment(label, "validation")],
            manifest_rows=[_manifest(label, "le2i_imvia")],
        )

        self.assertEqual(row["partition"], "excluded")


class ActionPretrainingTransferTest(unittest.TestCase):
    def test_rejects_freeze_period_outside_training_schedule(self) -> None:
        with self.assertRaisesRegex(ValueError, "freeze_encoder_epochs"):
            GaitTCNTrainingConfig(epochs=5, freeze_encoder_epochs=5)

    def test_transfers_encoder_without_overwriting_binary_classifier(self) -> None:
        source = LightweightGaitTCN(
            hidden_channels=8, class_count=len(ACTION_PRETRAINING_TARGETS)
        )
        target = LightweightGaitTCN(hidden_channels=8, class_count=2)
        with torch.no_grad():
            for parameter in source.input_projection.parameters():
                parameter.fill_(0.25)
            for parameter in source.temporal_blocks.parameters():
                parameter.fill_(0.5)
        classifier_before = {
            name: value.detach().clone()
            for name, value in target.classifier.state_dict().items()
        }
        checkpoint = {
            "schema_version": "fall-risk-action-tcn-pretraining-v1",
            "encoder_state_dict": {
                **{
                    f"input_projection.{name}": value
                    for name, value in source.input_projection.state_dict().items()
                },
                **{
                    f"temporal_blocks.{name}": value
                    for name, value in source.temporal_blocks.state_dict().items()
                },
            },
            "model_config": {
                "joint_count": 14,
                "input_channels": 5,
                "hidden_channels": 8,
                "kernel_size": 5,
                "dilations": [1, 2, 4, 8],
                "dropout": 0.2,
                "class_count": len(ACTION_PRETRAINING_TARGETS),
            },
        }

        transfer_action_pretrained_encoder(target, checkpoint)

        self.assertTrue(
            all(
                torch.allclose(value, torch.full_like(value, 0.25))
                for value in target.input_projection.state_dict().values()
                if value.is_floating_point()
            )
        )
        self.assertTrue(
            all(
                torch.equal(value, classifier_before[name])
                for name, value in target.classifier.state_dict().items()
            )
        )

    def test_rejects_incompatible_encoder_shape(self) -> None:
        source = LightweightGaitTCN(
            hidden_channels=16, class_count=len(ACTION_PRETRAINING_TARGETS)
        )
        target = LightweightGaitTCN(hidden_channels=8, class_count=2)
        checkpoint = {
            "schema_version": "fall-risk-action-tcn-pretraining-v1",
            "encoder_state_dict": {
                **{
                    f"input_projection.{name}": value
                    for name, value in source.input_projection.state_dict().items()
                },
                **{
                    f"temporal_blocks.{name}": value
                    for name, value in source.temporal_blocks.state_dict().items()
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "incompatible pretrained encoder"):
            transfer_action_pretrained_encoder(target, checkpoint)


if __name__ == "__main__":
    unittest.main()
