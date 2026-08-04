from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

import elderly_monitoring.modules.mental_health.wandering.tcn_baseline as tcn_module
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.tcn_baseline import (
    BINARY_CLASS_NAMES,
    FOUR_CLASS_NAMES,
    PRIMARY_SEED,
    TASK_BINARY,
    TASK_FOUR_CLASS,
    TCN_BINARY_PARAMETER_COUNT,
    TCN_FOUR_CLASS_PARAMETER_COUNT,
    TCN_SEEDS,
    ModelTrustError,
    TCNBaselineError,
    build_binary_sample_weights,
    build_tcn_development_artifacts,
    build_tensor_cohort,
    create_tcn_model,
    deterministic_npz_bytes,
    four_class_hierarchical_loss,
    four_class_probabilities,
    load_deterministic_npz,
    load_tcn_config,
    model_semantic_fingerprint,
    safe_load_tcn_model,
    select_early_stopping_epoch,
    validate_model_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_tcn_v1.yaml"


class WanderingTCNContractTest(unittest.TestCase):
    def test_config_freezes_trust_roots_protocol_and_has_no_official_input(self) -> None:
        config = load_tcn_config(CONFIG)
        self.assertEqual(config["schema_version"], "wandering-tcn-config-v1")
        self.assertEqual(config["purpose"], "comparison_only")
        self.assertEqual(
            config["trust_roots"]["rf_config"]["sha256"],
            "d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35",
        )
        self.assertEqual(
            config["trust_roots"]["rf_development_manifest"]["sha256"],
            "fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2",
        )
        self.assertEqual(
            config["trust_roots"]["rf_public_manifest"]["sha256"],
            "8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c",
        )
        self.assertNotIn("official_validation", CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["preprocessing"]["input_shape"], [80, 14])
        self.assertEqual(config["preprocessing"]["time_channel_indices"], [10, 11])
        self.assertEqual(config["preprocessing"]["mask_channel_index"], 12)
        self.assertFalse(config["preprocessing"]["temporal_features_enabled"])
        self.assertEqual(config["training"]["device"], "cpu")
        self.assertEqual(config["training"]["augmentation"], "none")
        self.assertEqual(config["checkpoint"]["format"], "deterministic_npz_v1")
        self.assertFalse(config["checkpoint"]["allow_pickle"])

    def test_config_rejects_extensions_path_drift_and_hyperparameter_drift(self) -> None:
        base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        mutations = []
        extended = copy.deepcopy(base)
        extended["unexpected"] = True
        mutations.append(extended)
        escaped = copy.deepcopy(base)
        escaped["trust_roots"]["rf_config"]["path"] = "../wandering_rf_v1.yaml"
        mutations.append(escaped)
        tuned = copy.deepcopy(base)
        tuned["training"]["optimizer"]["learning_rate"] = 0.001
        mutations.append(tuned)
        official = copy.deepcopy(base)
        official["trust_roots"]["official_validation"] = {"path": "sealed.jsonl", "sha256": "0" * 64}
        mutations.append(official)
        with tempfile.TemporaryDirectory() as temporary:
            for index, mutation in enumerate(mutations):
                path = Path(temporary) / f"bad_{index}.yaml"
                path.write_text(yaml.safe_dump(mutation, sort_keys=False), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(TCNBaselineError):
                    load_tcn_config(path)

    def test_models_are_independent_have_fixed_counts_and_no_target_components(self) -> None:
        four = create_tcn_model(TASK_FOUR_CLASS, seed=PRIMARY_SEED)
        binary = create_tcn_model(TASK_BINARY, seed=PRIMARY_SEED)
        self.assertEqual(tuple(FOUR_CLASS_NAMES), ("direct", "pacing", "lapping", "random"))
        self.assertEqual(tuple(BINARY_CLASS_NAMES), ("direct_or_non_wandering", "wandering_like"))
        self.assertEqual(tuple(TCN_SEEDS), (20260731, 20260801, 20260802, 20260803, 20260804))
        self.assertEqual(sum(parameter.numel() for parameter in four.parameters()), TCN_FOUR_CLASS_PARAMETER_COUNT)
        self.assertEqual(sum(parameter.numel() for parameter in binary.parameters()), TCN_BINARY_PARAMETER_COUNT)
        self.assertEqual(TCN_FOUR_CLASS_PARAMETER_COUNT, 40836)
        self.assertEqual(TCN_BINARY_PARAMETER_COUNT, 34433)
        four_ids = {id(parameter) for parameter in four.parameters()}
        binary_ids = {id(parameter) for parameter in binary.parameters()}
        self.assertFalse(four_ids & binary_ids)
        names = " ".join(name.lower() for name, _ in four.named_modules())
        for forbidden in ("attention", "transformer", "patch", "prototype", "energy"):
            self.assertNotIn(forbidden, names)

    def test_real_strict_bundle_has_exact_task_counts_and_smartcare_never_enters_four_class(self) -> None:
        bundle = load_preprocessing_bundle(
            rf_config_path=ROOT / "configs/modules/wandering_rf_v1.yaml",
            project_root=ROOT,
            mode=BUNDLE_MODE_DEVELOPMENT,
        )
        self.assertEqual(bundle.total_record_count, 1790)
        self.assertEqual(bundle.ready_count, 1775)
        self.assertEqual(bundle.unavailable_count, 15)
        self.assertEqual(len(bundle.unavailable_records()), 15)
        for task, expected in ((TASK_FOUR_CLASS, (1120, 240)), (TASK_BINARY, (1257, 278))):
            records = bundle.task_records(task)
            train = build_tensor_cohort(records, task=task, split="train")
            validation = build_tensor_cohort(records, task=task, split="validation")
            self.assertEqual((len(train.records), len(validation.records)), expected)
            self.assertEqual(tuple(train.features.shape[1:]), (80, 14))
            self.assertEqual(tuple(train.masks.shape[1:]), (80,))
            if task == TASK_FOUR_CLASS:
                self.assertEqual(Counter(row["source_dataset"] for row in train.records), Counter(wandering_patterns=1120))
                self.assertEqual(Counter(train.labels.tolist()), Counter({0: 280, 1: 280, 2: 280, 3: 280}))
            else:
                self.assertEqual(Counter(row["source_dataset"] for row in train.records), Counter(wandering_patterns=1120, smartcare=137))
                self.assertAlmostEqual(float(train.weights.sum()), 1257.0, places=3)

    def test_input_validation_and_masked_slots_cannot_change_output(self) -> None:
        model = create_tcn_model(TASK_FOUR_CLASS, seed=PRIMARY_SEED).eval()
        generator = torch.Generator().manual_seed(7)
        features = torch.randn((2, 80, 14), generator=generator)
        mask = torch.ones((2, 80), dtype=torch.float32)
        mask[:, 17:29] = 0.0
        features[:, :, 10:12] = 0.0
        features[:, :, 12] = mask
        changed = features.clone()
        changed[:, 17:29, :] = torch.randn((2, 12, 14), generator=generator) * 1000.0
        changed[:, 17:29, 12] = 0.0
        with torch.no_grad():
            first = model(features, mask)
            second = model(changed, mask)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
        validate_model_inputs(features, mask)
        invalid_cases = [
            (features[:, :-1], mask),
            (features, mask[:, :-1]),
            (features, torch.full_like(mask, 0.5)),
            (features, torch.zeros_like(mask)),
            (features.clone(), mask),
        ]
        invalid_cases[-1][0][0, 0, 0] = torch.nan
        for invalid_features, invalid_mask in invalid_cases:
            with self.subTest(shape=(tuple(invalid_features.shape), tuple(invalid_mask.shape))):
                with self.assertRaises(TCNBaselineError):
                    validate_model_inputs(invalid_features, invalid_mask)
        mismatch = features.clone()
        mismatch[:, :, 12] = 1.0 - mask
        with self.assertRaises(TCNBaselineError):
            validate_model_inputs(mismatch, mask)

    def test_hierarchical_probabilities_and_losses_match_hand_calculation(self) -> None:
        gate = torch.tensor([[0.0], [np.log(3.0)]], dtype=torch.float64)
        subtype = torch.tensor([[0.0, 0.0, 0.0], [np.log(2.0), 0.0, 0.0]], dtype=torch.float64)
        probabilities = four_class_probabilities(gate, subtype)
        expected = torch.tensor(
            [[0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], [0.25, 0.375, 0.1875, 0.1875]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(probabilities, expected)
        labels = torch.tensor([0, 2], dtype=torch.int64)
        loss = four_class_hierarchical_loss(gate, subtype, labels)
        expected_loss = (-np.log(0.5) - np.log(0.75) - np.log(0.25)) / 2.0
        self.assertAlmostEqual(float(loss), expected_loss, places=12)

    def test_binary_source_class_weights_have_equal_group_totals(self) -> None:
        rows = []
        for source, label, count in (
            ("wandering_patterns", 0, 280),
            ("wandering_patterns", 1, 840),
            ("smartcare", 0, 63),
            ("smartcare", 1, 74),
        ):
            rows.extend({"source_dataset": source, "binary_label": label, "split": "train"} for _ in range(count))
        weights = build_binary_sample_weights(rows)
        totals: dict[tuple[str, int], float] = defaultdict(float)
        for row, weight in zip(rows, weights, strict=True):
            totals[(row["source_dataset"], row["binary_label"])] += float(weight)
        self.assertEqual(Counter(round(value, 9) for value in totals.values()), Counter({314.25: 4}))
        self.assertAlmostEqual(float(weights[0]), 1.1223214285714285)

    def test_early_stopping_uses_strict_delta_keeps_earlier_tie_and_minimum_epochs(self) -> None:
        metrics = [0.5] * 29 + [0.6] + [0.6001] * 20
        result = select_early_stopping_epoch(
            metrics,
            minimum_epochs=30,
            patience=20,
            minimum_improvement=1e-4,
        )
        self.assertEqual(result["best_epoch"], 30)
        self.assertEqual(result["stopped_epoch"], 50)
        self.assertEqual(result["stop_reason"], "early_stopping_patience")
        self.assertTrue(result["validation_used_for_checkpoint_selection"])
        self.assertFalse(result["test_used_for_checkpoint_selection"])

    def test_deterministic_npz_round_trip_and_semantic_fingerprint(self) -> None:
        model = create_tcn_model(TASK_BINARY, seed=PRIMARY_SEED).eval()
        payload_a = deterministic_npz_bytes(model)
        payload_b = deterministic_npz_bytes(model)
        self.assertEqual(payload_a, payload_b)
        self.assertEqual(hashlib.sha256(payload_a).hexdigest(), hashlib.sha256(payload_b).hexdigest())
        loaded = create_tcn_model(TASK_BINARY, seed=PRIMARY_SEED).eval()
        load_deterministic_npz(loaded, payload_a)
        self.assertEqual(model_semantic_fingerprint(model, task=TASK_BINARY), model_semantic_fingerprint(loaded, task=TASK_BINARY))
        features = torch.zeros((1, 80, 14), dtype=torch.float32)
        mask = torch.ones((1, 80), dtype=torch.float32)
        features[:, :, 12] = mask
        with torch.no_grad():
            torch.testing.assert_close(model(features, mask)["binary_logit"], loaded(features, mask)["binary_logit"], rtol=0.0, atol=0.0)

        archive = np.load(io.BytesIO(payload_a), allow_pickle=False)
        arrays = {name: archive[name] for name in archive.files}
        arrays[next(iter(arrays))] = arrays[next(iter(arrays))].copy()
        arrays[next(iter(arrays))].flat[0] = np.nan
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        with self.assertRaises(ModelTrustError):
            load_deterministic_npz(loaded, buffer.getvalue())

    def test_development_bundle_is_atomic_has_ten_checkpoints_and_safe_loader_fails_before_npz(self) -> None:
        def fake_train(*, task: str, seed: int, train: object, validation: object, config: object) -> dict[str, object]:
            model = create_tcn_model(task, seed=seed).eval()
            labels = validation.labels.numpy()
            width = 4 if task == TASK_FOUR_CLASS else 2
            probabilities = np.full((len(labels), width), 0.1 if width == 4 else 0.2, dtype=np.float64)
            if width == 4:
                probabilities[np.arange(len(labels)), labels] = 0.7
            else:
                probabilities[np.arange(len(labels)), labels] = 0.8
            selection = {
                "best_epoch": 30,
                "best_metric": 1.0,
                "stopped_epoch": 50,
                "stop_reason": "early_stopping_patience",
                "minimum_epochs": 30,
                "patience": 20,
                "minimum_improvement": 0.0001,
                "strictly_greater_improvement": True,
                "tie_policy": "keep_earlier",
                "validation_used_for_checkpoint_selection": True,
                "test_used_for_checkpoint_selection": False,
            }
            history = {
                "schema_version": "wandering-tcn-training-history-v1",
                "task": task,
                "seed": seed,
                "fit_partition": "train",
                "validation_partition": "validation",
                "test_seen": False,
                "official_source_opened": False,
                "epochs": [],
                "selection": selection,
            }
            return {"model": model, "history": history, "validation_probabilities": probabilities}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            with mock.patch.object(tcn_module, "_configure_deterministic_runtime"), mock.patch.object(
                tcn_module, "_train_one_model", side_effect=fake_train
            ) as trainer:
                result = build_tcn_development_artifacts(
                    tcn_config_path=CONFIG,
                    project_root=ROOT,
                    output_dir=output,
                )
            self.assertEqual(trainer.call_count, 10)
            self.assertEqual(len(list((output / "models/four_class").glob("*.npz"))), 5)
            self.assertEqual(len(list((output / "models/binary").glob("*.npz"))), 5)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(), result.manifest_sha256)
            self.assertFalse(manifest["test_artifacts_present"])
            self.assertFalse(manifest["official_source_opened"])
            self.assertTrue(manifest["validation_used_for_checkpoint_selection"])
            self.assertEqual(len(manifest["models"]), 10)
            index = json.loads((output / "data_index.json").read_text(encoding="utf-8"))
            self.assertFalse(index["development_constraints"]["test_tensor_generated"])
            self.assertFalse(index["development_constraints"]["rf_public_manifest_opened"])
            loaded = safe_load_tcn_model(
                output,
                task=TASK_FOUR_CLASS,
                seed=PRIMARY_SEED,
                expected_manifest_sha256=result.manifest_sha256,
                tcn_config_path=CONFIG,
                project_root=ROOT,
            )
            self.assertEqual(model_semantic_fingerprint(loaded, task=TASK_FOUR_CLASS), manifest["models"][f"four_class/{PRIMARY_SEED}"]["semantic_fingerprint"])

            checkpoint = output / f"models/four_class/seed_{PRIMARY_SEED}.npz"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with mock.patch.object(tcn_module, "load_deterministic_npz") as npz_loader:
                with self.assertRaises(ModelTrustError):
                    safe_load_tcn_model(
                        output,
                        task=TASK_FOUR_CLASS,
                        seed=PRIMARY_SEED,
                        expected_manifest_sha256=result.manifest_sha256,
                        tcn_config_path=CONFIG,
                        project_root=ROOT,
                    )
                npz_loader.assert_not_called()

    def test_existing_development_output_is_rejected_before_any_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already_exists"
            output.mkdir()
            with mock.patch.object(tcn_module, "_train_one_model") as trainer:
                with self.assertRaises(FileExistsError):
                    build_tcn_development_artifacts(
                        tcn_config_path=CONFIG,
                        project_root=ROOT,
                        output_dir=output,
                    )
                trainer.assert_not_called()

    def test_cli_phase_boundaries_require_external_hashes_and_evaluator_has_no_fit(self) -> None:
        train_source = (ROOT / "scripts/wandering/train_tcn_baseline.py").read_text(encoding="utf-8")
        evaluate_source = (ROOT / "scripts/wandering/evaluate_tcn_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("rf_public", train_source)
        self.assertNotIn("test", train_source.lower())
        self.assertNotIn(".fit(", evaluate_source)
        self.assertIn("--expected-development-manifest-sha256", evaluate_source)
        self.assertIn("--expected-rf-public-manifest-sha256", evaluate_source)

    def test_batch_one_and_multiple_batches_have_finite_gradients_in_all_stages(self) -> None:
        for task, labels in ((TASK_FOUR_CLASS, torch.tensor([0, 1, 2, 3])), (TASK_BINARY, torch.tensor([0, 1, 0, 1]))):
            model = create_tcn_model(task, seed=PRIMARY_SEED).train()
            features = torch.randn((4, 80, 14), generator=torch.Generator().manual_seed(11))
            mask = torch.ones((4, 80), dtype=torch.float32)
            features[:, :, 10:12] = 0.0
            features[:, :, 12] = mask
            outputs = model(features, mask)
            if task == TASK_FOUR_CLASS:
                loss = four_class_hierarchical_loss(outputs["gate_logit"], outputs["subtype_logits"], labels)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["binary_logit"].squeeze(1), labels.float())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            groups = {"input_projection": False, "blocks.0": False, "blocks.1": False, "blocks.2": False, "head": False}
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for prefix in tuple(groups):
                    if name.startswith(prefix):
                        groups[prefix] = True
            self.assertTrue(all(groups.values()), groups)


if __name__ == "__main__":
    unittest.main()
