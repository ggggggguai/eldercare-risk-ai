from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from elderly_monitoring.modules.mental_health.wandering.model import (
    TOPOWANDER_PARAMETER_COUNT,
    create_topowander_model,
    create_topowander_training_model,
    load_topowander_config,
)
from elderly_monitoring.modules.mental_health.wandering.performance import (
    CheckpointError,
    M0S_PRIMARY_SEED,
    M0S_SEEDS,
    PerformanceDataError,
    WanderingPerformanceDataset,
    collate_performance_batch,
    compute_training_weights,
    create_training_optimizer,
    evaluate_joint_predictions,
    joint_supervised_loss,
    load_development_records,
    load_performance_config,
    load_training_checkpoint,
    paired_error_comparison,
    recover_training_run,
    save_training_checkpoint,
    selection_rank,
)


ROOT = Path(__file__).resolve().parents[1]
FORWARD_CONFIG = ROOT / "configs/modules/wandering_topowander_mpt_v1.yaml"
PERFORMANCE_CONFIG = ROOT / "configs/modules/wandering_performance_v1.yaml"


def _record(
    sample_id: str,
    *,
    source: str,
    binary_label: int,
    pattern_label: str,
    split: str = "train",
) -> dict[str, object]:
    time = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    points = np.stack((time, np.zeros_like(time)), axis=1)
    features = np.zeros((80, 14), dtype=np.float32)
    features[:, 0:2] = points
    features[:, 12] = 1.0
    features[:, 13] = 1.0
    eligible = source == "wandering_patterns"
    return {
        "sample_id": sample_id,
        "source_dataset": source,
        "split": split,
        "preprocess_status": "ready",
        "binary_label": binary_label,
        "binary_supervision_eligible": True,
        "pattern_label": pattern_label,
        "pattern_supervision_eligible": eligible,
        "model_features": features.tolist(),
        "shape_normalized_points": points.tolist(),
        "point_mask": np.ones(80, dtype=np.float32).tolist(),
    }


def _training_records() -> list[dict[str, object]]:
    return [
        _record("wp-direct", source="wandering_patterns", binary_label=0, pattern_label="direct"),
        _record("wp-pacing", source="wandering_patterns", binary_label=1, pattern_label="pacing"),
        _record("wp-lapping", source="wandering_patterns", binary_label=1, pattern_label="lapping"),
        _record("wp-random", source="wandering_patterns", binary_label=1, pattern_label="random"),
        _record("sc-direct", source="smartcare", binary_label=0, pattern_label="unknown"),
        _record("sc-wandering", source="smartcare", binary_label=1, pattern_label="unknown"),
    ]


class WanderingPerformanceConfigAndDataTest(unittest.TestCase):
    def test_performance_config_is_separate_and_constructs_exact_forward_model(self) -> None:
        config = load_performance_config(PERFORMANCE_CONFIG, project_root=ROOT)
        self.assertEqual(config["model"]["forward_config_path"], "configs/modules/wandering_topowander_mpt_v1.yaml")
        self.assertNotIn("tcn", config)
        self.assertNotIn("transformer", config)
        self.assertEqual(tuple(config["training"]["seeds"]), M0S_SEEDS)
        self.assertEqual(config["training"]["primary_seed"], M0S_PRIMARY_SEED)
        self.assertNotIn("seed", config["training"])
        self.assertEqual(config["runtime"]["device"], "cpu")
        self.assertLessEqual(config["training"]["max_epochs"], 50)

        forward = load_topowander_config(ROOT / config["model"]["forward_config_path"])
        model = create_topowander_model(forward)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), TOPOWANDER_PARAMETER_COUNT)

    def test_training_initialization_is_repeatable_per_seed_and_changes_across_seeds(self) -> None:
        forward = load_topowander_config(FORWARD_CONFIG)
        first = create_topowander_training_model(forward, seed=M0S_SEEDS[0])
        repeated = create_topowander_training_model(forward, seed=M0S_SEEDS[0])
        different = create_topowander_training_model(forward, seed=M0S_SEEDS[1])

        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, repeated.state_dict()[name]), name)
        self.assertTrue(
            any(
                not torch.equal(value, different.state_dict()[name])
                for name, value in first.state_dict().items()
            )
        )

    def test_dataset_and_collate_assemble_three_inputs_and_mask_subtypes(self) -> None:
        dataset = WanderingPerformanceDataset(_training_records())
        batch = collate_performance_batch([dataset[index] for index in range(len(dataset))])

        self.assertEqual(tuple(batch["model_features"].shape), (6, 80, 14))
        self.assertEqual(tuple(batch["shape_normalized_points"].shape), (6, 80, 2))
        self.assertEqual(tuple(batch["point_mask"].shape), (6, 80))
        self.assertEqual(batch["binary_label"].tolist(), [0, 1, 1, 1, 0, 1])
        self.assertEqual(batch["subtype_label"].tolist(), [-1, 0, 1, 2, -1, -1])
        self.assertEqual(batch["pattern_label"].tolist(), [0, 1, 2, 3, -1, -1])

    def test_dataset_rejects_test_rows_and_label_or_mask_drift(self) -> None:
        test_row = _record(
            "forbidden-test",
            source="wandering_patterns",
            binary_label=0,
            pattern_label="direct",
            split="test",
        )
        with self.assertRaisesRegex(PerformanceDataError, "train/validation"):
            WanderingPerformanceDataset([test_row])

        bad_label = _record(
            "bad-label",
            source="wandering_patterns",
            binary_label=0,
            pattern_label="pacing",
        )
        with self.assertRaisesRegex(PerformanceDataError, "label"):
            WanderingPerformanceDataset([bad_label])

        bad_mask = _record(
            "bad-mask",
            source="wandering_patterns",
            binary_label=0,
            pattern_label="direct",
        )
        bad_mask["point_mask"][0] = 0.0  # type: ignore[index]
        with self.assertRaisesRegex(PerformanceDataError, "mask channel"):
            WanderingPerformanceDataset([bad_mask])

    def test_materialized_development_bundle_exposes_only_fixed_train_validation(self) -> None:
        config = load_performance_config(PERFORMANCE_CONFIG, project_root=ROOT)
        train, validation, identity = load_development_records(config, project_root=ROOT)
        self.assertEqual((len(train), len(validation)), (1257, 278))
        self.assertEqual({row["split"] for row in train}, {"train"})
        self.assertEqual({row["split"] for row in validation}, {"validation"})
        self.assertTrue(identity["shared_container_wp_test_integrity_parsed"])
        self.assertFalse(identity["wp_test_exposed_to_development_accessor"])
        self.assertFalse(identity["wp_test_materialized"])
        self.assertFalse(identity["wp_test_used_for_inference"])
        self.assertFalse(identity["wp_test_used_for_scoring"])
        self.assertFalse(identity["wp_test_used_for_training"])
        self.assertFalse(identity["wp_test_used_for_selection"])
        self.assertFalse(identity["smartcare_official_opened"])
        self.assertFalse(identity["sealed_camera_opened"])


class WanderingPerformanceLossAndEvaluatorTest(unittest.TestCase):
    def test_joint_loss_uses_all_binary_labels_and_only_legal_subtypes(self) -> None:
        binary_logits = torch.tensor([[0.2], [-0.1], [0.7]], dtype=torch.float32, requires_grad=True)
        subtype_logits = torch.tensor(
            [[20.0, -20.0, 5.0], [0.4, -0.2, 0.1], [-30.0, 30.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        binary_labels = torch.tensor([0, 1, 1], dtype=torch.long)
        subtype_labels = torch.tensor([-1, 0, -1], dtype=torch.long)
        loss = joint_supervised_loss(
            binary_logits,
            subtype_logits,
            binary_labels,
            subtype_labels,
            binary_sample_weights=torch.ones(3),
            subtype_class_weights=torch.ones(3),
            subtype_loss_weight=1.0,
        )
        changed = subtype_logits.detach().clone()
        changed[0] = torch.tensor([-50.0, 50.0, 0.0])
        changed[2] = torch.tensor([50.0, -50.0, 0.0])
        changed.requires_grad_(True)
        changed_loss = joint_supervised_loss(
            binary_logits.detach(),
            changed,
            binary_labels,
            subtype_labels,
            binary_sample_weights=torch.ones(3),
            subtype_class_weights=torch.ones(3),
            subtype_loss_weight=1.0,
        )
        self.assertAlmostEqual(float(loss.detach()), float(changed_loss.detach()), places=6)

        loss.backward()
        self.assertTrue(torch.isfinite(binary_logits.grad).all())
        self.assertTrue(torch.isfinite(subtype_logits.grad).all())
        self.assertTrue(torch.equal(subtype_logits.grad[[0, 2]], torch.zeros_like(subtype_logits.grad[[0, 2]])))
        self.assertGreater(float(subtype_logits.grad[1].abs().sum()), 0.0)

    def test_joint_evaluator_reports_perfect_four_class_and_source_equal_binary(self) -> None:
        rows = _training_records()
        for row in rows:
            row["split"] = "validation"
        binary_logits = np.asarray([-12.0, 12.0, 12.0, 12.0, -12.0, 12.0], dtype=np.float64)
        subtype_logits = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [0.0, 12.0, 0.0],
                [0.0, 0.0, 12.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = evaluate_joint_predictions(rows, binary_logits, subtype_logits)

        self.assertEqual(result.metrics["wp_four_class"]["sample_count"], 4)
        self.assertEqual(result.metrics["binary"]["wp"]["sample_count"], 4)
        self.assertEqual(result.metrics["binary"]["smartcare"]["sample_count"], 2)
        self.assertEqual(result.metrics["wp_four_class"]["macro_f1"], 1.0)
        self.assertEqual(result.metrics["binary"]["source_equal_macro_f1"], 1.0)
        self.assertEqual(result.metrics["selection_score"], 1.0)
        self.assertEqual(len(result.predictions), 6)
        self.assertEqual(
            [row["predicted_pattern_label"] for row in result.predictions[:4]],
            ["direct", "pacing", "lapping", "random"],
        )

    def test_evaluator_rejects_nonfinite_shape_and_label_errors(self) -> None:
        rows = _training_records()
        with self.assertRaisesRegex(PerformanceDataError, "shape"):
            evaluate_joint_predictions(rows, np.zeros(5), np.zeros((6, 3)))
        bad = np.zeros(6)
        bad[2] = np.nan
        with self.assertRaisesRegex(PerformanceDataError, "finite"):
            evaluate_joint_predictions(rows, bad, np.zeros((6, 3)))
        drift = deepcopy(rows)
        drift[1]["binary_label"] = 0
        with self.assertRaisesRegex(PerformanceDataError, "label"):
            evaluate_joint_predictions(drift, np.zeros(6), np.zeros((6, 3)))

    def test_selection_rank_uses_fixed_tie_break_order(self) -> None:
        earlier = selection_rank(0.91, 0.96, 0.91, 0.4, 3)
        later = selection_rank(0.91, 0.96, 0.91, 0.4, 4)
        lower_loss = selection_rank(0.91, 0.96, 0.91, 0.3, 9)
        higher_four = selection_rank(0.91, 0.97, 0.91, 0.8, 9)
        self.assertGreater(earlier, later)
        self.assertGreater(lower_loss, earlier)
        self.assertGreater(higher_four, lower_loss)

    def test_paired_error_comparison_reports_discordant_and_shared_counts(self) -> None:
        result = paired_error_comparison(
            {"a", "b", "c", "d"},
            {"a", "b"},
            {"b", "c"},
            baseline_name="rf",
            baseline_predictions_sha256="1" * 64,
            baseline_predictions_path="saved.jsonl",
        )
        self.assertEqual(result["both_wrong_count"], 1)
        self.assertEqual(result["either_wrong_count"], 3)
        self.assertEqual(result["both_correct_count"], 1)
        self.assertEqual(result["m0_wrong_baseline_correct_count"], 1)
        self.assertEqual(result["m0_correct_baseline_wrong_count"], 1)
        self.assertFalse(result["baseline_retrained"])


class WanderingPerformanceCheckpointTest(unittest.TestCase):
    def test_numeric_checkpoint_round_trip_restores_model_optimizer_and_rng(self) -> None:
        forward = load_topowander_config(FORWARD_CONFIG)
        model = create_topowander_model(forward)
        optimizer = create_training_optimizer(model, learning_rate=1.0e-3, weight_decay=1.0e-4)
        dataset = WanderingPerformanceDataset(_training_records()[:4])
        batch = collate_performance_batch([dataset[index] for index in range(len(dataset))])
        weights = compute_training_weights(_training_records())

        outputs = model(batch["model_features"], batch["shape_normalized_points"], batch["point_mask"])
        loss = joint_supervised_loss(
            outputs["binary_logit"],
            outputs["subtype_logits"],
            batch["binary_label"],
            batch["subtype_label"],
            binary_sample_weights=torch.ones(4),
            subtype_class_weights=torch.tensor(weights.subtype_class_weights, dtype=torch.float32),
            subtype_loss_weight=1.0,
        )
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_training_checkpoint(
                Path(directory),
                epoch=1,
                seed=M0S_PRIMARY_SEED,
                model=model,
                optimizer=optimizer,
                forward_config_path=FORWARD_CONFIG,
                performance_config_sha256="1" * 64,
                metrics={"selection_score": 0.5},
            )
            restored = load_training_checkpoint(
                checkpoint,
                forward_config_path=FORWARD_CONFIG,
                expected_performance_config_sha256="1" * 64,
                expected_seed=M0S_PRIMARY_SEED,
                learning_rate=1.0e-3,
                weight_decay=1.0e-4,
            )

            self.assertEqual(restored.epoch, 1)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)
            original_by_name = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
            restored_by_name = {
                name: parameter for name, parameter in restored.model.named_parameters() if parameter.requires_grad
            }
            for name in original_by_name:
                original_state = optimizer.state[original_by_name[name]]
                restored_state = restored.optimizer.state[restored_by_name[name]]
                self.assertEqual(set(original_state), set(restored_state))
                for key in original_state:
                    self.assertTrue(
                        torch.equal(original_state[key], restored_state[key]),
                        f"{name}:{key} original={original_state[key]!r} restored={restored_state[key]!r}",
                    )

            with self.assertRaises(CheckpointError):
                save_training_checkpoint(
                    Path(directory),
                    epoch=1,
                    seed=M0S_PRIMARY_SEED,
                    model=model,
                    optimizer=optimizer,
                    forward_config_path=FORWARD_CONFIG,
                    performance_config_sha256="1" * 64,
                    metrics={"selection_score": 0.5},
                )

    def test_recovery_repairs_checkpoint_history_latest_and_best_after_fault_window(self) -> None:
        forward = load_topowander_config(FORWARD_CONFIG)
        model = create_topowander_training_model(forward, seed=M0S_PRIMARY_SEED)
        optimizer = create_training_optimizer(model, learning_rate=1.0e-3, weight_decay=1.0e-4)
        dataset = WanderingPerformanceDataset(_training_records()[:4])
        batch = collate_performance_batch([dataset[index] for index in range(len(dataset))])
        weights = compute_training_weights(_training_records())
        outputs = model(batch["model_features"], batch["shape_normalized_points"], batch["point_mask"])
        loss = joint_supervised_loss(
            outputs["binary_logit"],
            outputs["subtype_logits"],
            batch["binary_label"],
            batch["subtype_label"],
            binary_sample_weights=torch.ones(4),
            subtype_class_weights=torch.tensor(weights.subtype_class_weights, dtype=torch.float32),
            subtype_loss_weight=1.0,
        )
        loss.backward()
        optimizer.step()
        performance_sha256 = __import__("hashlib").sha256(PERFORMANCE_CONFIG.read_bytes()).hexdigest()
        metrics_one = {
            "epoch": 1,
            "validation_loss": 0.4,
            "joint_metrics": {
                "selection_score": 0.7,
                "wp_four_class": {"macro_f1": 0.7},
                "binary": {"source_equal_macro_f1": 0.8},
            },
            "no_improvement_epochs": 0,
        }
        metrics_two = {
            "epoch": 2,
            "validation_loss": 0.3,
            "joint_metrics": {
                "selection_score": 0.8,
                "wp_four_class": {"macro_f1": 0.8},
                "binary": {"source_equal_macro_f1": 0.9},
            },
            "no_improvement_epochs": 0,
        }

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoints = run / "checkpoints"
            save_training_checkpoint(
                checkpoints,
                epoch=1,
                seed=M0S_PRIMARY_SEED,
                model=model,
                optimizer=optimizer,
                forward_config_path=FORWARD_CONFIG,
                performance_config_sha256=performance_sha256,
                metrics=metrics_one,
            )
            save_training_checkpoint(
                checkpoints,
                epoch=2,
                seed=M0S_PRIMARY_SEED,
                model=model,
                optimizer=optimizer,
                forward_config_path=FORWARD_CONFIG,
                performance_config_sha256=performance_sha256,
                metrics=metrics_two,
            )
            # Fault window: epoch 2 is fully committed, while history/latest/best still point to epoch 1.
            (run / "training_history.jsonl").write_text(
                __import__("json").dumps(metrics_one, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (checkpoints / "latest.json").write_text(
                '{"checkpoint":"epoch-0001","epoch":1}\n', encoding="utf-8"
            )
            (checkpoints / "best.json").write_text(
                '{"checkpoint":"epoch-0001","epoch":1,"selection_rank":[0.7,0.7,0.8,-0.4,-1]}\n',
                encoding="utf-8",
            )

            recovered = recover_training_run(
                PERFORMANCE_CONFIG,
                project_root=ROOT,
                run_dir=run,
                seed=M0S_PRIMARY_SEED,
            )

            self.assertEqual(recovered.latest_checkpoint.name, "epoch-0002")
            self.assertEqual(recovered.best_checkpoint.name, "epoch-0002")
            self.assertEqual([row["epoch"] for row in recovered.history], [1, 2])
            self.assertIn("training_history", recovered.repaired)
            self.assertIn("latest_pointer", recovered.repaired)
            self.assertIn("best_pointer", recovered.repaired)
            restored = load_training_checkpoint(
                recovered.latest_checkpoint,
                forward_config_path=FORWARD_CONFIG,
                expected_performance_config_sha256=performance_sha256,
                expected_seed=M0S_PRIMARY_SEED,
                learning_rate=1.0e-3,
                weight_decay=1.0e-4,
            )
            self.assertEqual(restored.epoch, 2)


if __name__ == "__main__":
    unittest.main()
