from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch

from elderly_monitoring.modules.mental_health.wandering import release as release_module
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_FROZEN_WP_TEST,
)
from elderly_monitoring.modules.mental_health.wandering.release import (
    CandidateManifestError,
    CandidateRuntime,
    RELEASE_IMPLEMENTATION_SOURCE_FILES,
    ReleaseArtifactError,
    WPReleaseDataError,
    build_candidate_bundle,
    canonical_wp_cohort_identity,
    compare_primary_validation_parity,
    configure_release_cpu_runtime,
    evaluate_wp_predictions,
    freeze_release_implementation_identity,
    load_candidate_from_manifest,
    run_authorized_frozen_wp_score,
    run_wp_records_evaluation,
    validate_wp_release_cohort,
    validate_wp_release_record,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_IDENTITY = (
    ROOT
    / "reports/mental_health/wandering_performance/m0r_release_prep_v1"
    / "identity/training_candidate_identity.json"
)
TRAINING_IDENTITY_SHA256 = "c4e8576fce46a69bca3cebf81306197c9abfa439ff720bb3ced95ea74f580228"
CLI = ROOT / "scripts/wandering/release_wandering_candidate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(sample_id: str, pattern: str, *, split: str = "validation") -> dict[str, object]:
    time = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    points = np.stack((time, np.zeros_like(time)), axis=1)
    features = np.zeros((80, 14), dtype=np.float32)
    features[:, 0:2] = points
    features[:, 12] = 1.0
    features[:, 13] = 1.0
    return {
        "sample_id": sample_id,
        "source_dataset": "wandering_patterns",
        "split": split,
        "preprocess_status": "ready",
        "binary_label": 0 if pattern == "direct" else 1,
        "binary_supervision_eligible": True,
        "pattern_label": pattern,
        "pattern_supervision_eligible": True,
        "model_features": features.tolist(),
        "shape_normalized_points": points.tolist(),
        "point_mask": np.ones(80, dtype=np.float32).tolist(),
    }


def _fixed_cohort(split: str = "validation") -> list[dict[str, object]]:
    return [
        _record(f"{split}-{pattern}-{index:02d}", pattern, split=split)
        for pattern in ("direct", "pacing", "lapping", "random")
        for index in range(60)
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _make_score_fixture(root: Path) -> tuple[Path, str]:
    archive = root / "release-source-v3.zip"
    release_identity = root / "release-identity-v3.json"
    frozen = freeze_release_implementation_identity(
        project_root=ROOT,
        source_files=RELEASE_IMPLEMENTATION_SOURCE_FILES,
        archive_path=archive,
        identity_path=release_identity,
        git_head="test-head",
    )
    built = build_candidate_bundle(
        project_root=ROOT,
        training_identity_path=TRAINING_IDENTITY,
        expected_training_identity_sha256=TRAINING_IDENTITY_SHA256,
        release_identity_path=release_identity,
        expected_release_identity_sha256=frozen.identity_sha256,
        output_dir=root / "candidate-v3",
    )
    return built.manifest_path, built.manifest_sha256


class _FakeFrozenBundle:
    def __init__(self, manifest: dict[str, object], records: list[dict[str, object]]) -> None:
        accessor = manifest["frozen_wp_accessor"]
        assert isinstance(accessor, dict)
        upstream = accessor["upstream_artifacts"]
        assert isinstance(upstream, dict)
        self.mode = BUNDLE_MODE_FROZEN_WP_TEST
        self.input_hashes = {
            role: descriptor["sha256"]
            for role, descriptor in upstream.items()
        }
        self.split_sha256 = accessor["split_sha256"]
        self.preprocessing_config_sha256 = accessor["preprocessing_config_sha256"]
        self.integrity_report = {
            "all_frozen_hashes_verified": True,
            "manifest_binding_verified": True,
            "human_review_status": "human_review_passed",
            "near_neighbor_cross_partition_lt_0_05": 4054,
            "official_source_path_present": False,
            "total_records": 1790,
        }
        self._records = tuple(records)

    def records_for_split(self, split: str) -> tuple[dict[str, object], ...]:
        if split != "test":
            raise AssertionError("formal controller requested a non-test split")
        return self._records


class WPReleasePhaseAndMetricsTest(unittest.TestCase):
    def test_record_validator_is_phase_aware_and_never_relabels(self) -> None:
        validation = _record("validation-direct", "direct", split="validation")
        validated = validate_wp_release_record(validation, expected_split="validation")
        self.assertEqual(validated.split, "validation")
        with self.assertRaisesRegex(WPReleaseDataError, "expected split"):
            validate_wp_release_record(validation, expected_split="test")

        test = _record("test-direct", "direct", split="test")
        validated_test = validate_wp_release_record(test, expected_split="test")
        self.assertEqual(validated_test.split, "test")
        smartcare = dict(test)
        smartcare["source_dataset"] = "smartcare"
        with self.assertRaisesRegex(WPReleaseDataError, "WanderingPatterns"):
            validate_wp_release_record(smartcare, expected_split="test")

    def test_fixed_validation_and_test_cohorts_require_exact_ids_labels_and_counts(self) -> None:
        validation = validate_wp_release_cohort(_fixed_cohort("validation"), expected_split="validation")
        test = validate_wp_release_cohort(_fixed_cohort("test"), expected_split="test")
        self.assertEqual(len(validation), 240)
        self.assertEqual(len(test), 240)

        missing = _fixed_cohort("test")[:-1]
        with self.assertRaisesRegex(WPReleaseDataError, "240"):
            validate_wp_release_cohort(missing, expected_split="test")
        duplicate = _fixed_cohort("validation")
        duplicate[-1]["sample_id"] = duplicate[0]["sample_id"]
        with self.assertRaisesRegex(WPReleaseDataError, "unique"):
            validate_wp_release_cohort(duplicate, expected_split="validation")

    def test_wp_evaluator_fixes_threshold_and_hierarchical_class_order(self) -> None:
        rows = [_record(f"fixture-{name}", name) for name in ("direct", "pacing", "lapping", "random")]
        binary_logits = np.asarray([-12.0, 0.0, 12.0, 12.0], dtype=np.float64)
        subtype_logits = np.asarray(
            [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]],
            dtype=np.float64,
        )
        result = evaluate_wp_predictions(
            rows,
            binary_logits,
            subtype_logits,
            expected_split="validation",
            require_fixed_cohort=False,
        )
        self.assertEqual(result.predictions[1]["binary_probability"], 0.5)
        self.assertEqual(result.predictions[1]["predicted_binary_label"], 1)
        self.assertEqual(
            [row["predicted_pattern_label"] for row in result.predictions],
            ["direct", "direct", "lapping", "random"],
        )
        self.assertGreater(
            result.predictions[1]["subtype_probabilities"]["pacing"],
            result.predictions[1]["subtype_probabilities"]["lapping"],
        )
        self.assertEqual(result.metrics["wp_four_class"]["labels"], ["direct", "pacing", "lapping", "random"])
        self.assertEqual(result.metrics["wp_binary"]["labels"], ["direct_or_non_wandering", "wandering_like"])

    def test_parity_comparison_requires_exact_identity_labels_probabilities_and_metrics(self) -> None:
        rows = [_record(f"fixture-{name}", name) for name in ("direct", "pacing", "lapping", "random")]
        logits = np.asarray([-12.0, 12.0, 12.0, 12.0], dtype=np.float64)
        subtype = np.eye(3, dtype=np.float64)[[0, 0, 1, 2]] * 12.0
        result = evaluate_wp_predictions(
            rows, logits, subtype, expected_split="validation", require_fixed_cohort=False
        )
        report = compare_primary_validation_parity(
            result,
            reference_predictions=result.predictions,
            reference_wp_four_metrics=result.metrics["wp_four_class"],
            reference_wp_binary_metrics=result.metrics["wp_binary"],
            tolerance=1.0e-7,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_probability_difference"], 0.0)

        drift = [dict(row) for row in result.predictions]
        drift[0]["true_pattern_label"] = "random"
        with self.assertRaisesRegex(WPReleaseDataError, "label"):
            compare_primary_validation_parity(
                result,
                reference_predictions=drift,
                reference_wp_four_metrics=result.metrics["wp_four_class"],
                reference_wp_binary_metrics=result.metrics["wp_binary"],
                tolerance=1.0e-7,
            )


class CandidateManifestAndBundleTest(unittest.TestCase):
    def test_compact_bundle_and_external_manifest_bound_loader(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-bundle-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, manifest_sha = _make_score_fixture(root)
            bundle = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(path.name for path in bundle.iterdir()),
                [
                    "candidate_manifest.json",
                    "forward_config.yaml",
                    "frozen_wp_rf_config.yaml",
                    "model_state.npz",
                    "performance_config.yaml",
                ],
            )
            self.assertFalse(any("optimizer" in path.name or "rng" in path.name for path in bundle.iterdir()))
            with self.assertRaisesRegex(CandidateManifestError, "external manifest SHA"):
                load_candidate_from_manifest(
                    manifest_path, expected_manifest_sha256="0" * 64
                )

            runtime = load_candidate_from_manifest(
                manifest_path,
                expected_manifest_sha256=manifest_sha,
            )
            self.assertFalse(runtime.model.training)
            self.assertEqual({parameter.device.type for parameter in runtime.model.parameters()}, {"cpu"})
            original = runtime.model.forward

            def guarded(*args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
                self.assertTrue(torch.is_inference_mode_enabled())
                self.assertFalse(torch.is_grad_enabled())
                return original(*args, **kwargs)

            runtime.model.forward = guarded  # type: ignore[method-assign]
            binary, subtype = runtime.predict_logits(
                [_record(f"fixture-{name}", name) for name in ("direct", "pacing", "lapping", "random")],
                expected_split="validation",
                batch_size=4,
                require_fixed_cohort=False,
            )
            self.assertEqual(binary.shape, (4,))
            self.assertEqual(subtype.shape, (4, 3))

            observed_batch_sizes: list[int] = []

            def recording(
                model_features: torch.Tensor,
                shape_normalized_points: torch.Tensor,
                point_mask: torch.Tensor,
            ) -> dict[str, torch.Tensor]:
                self.assertTrue(torch.is_inference_mode_enabled())
                batch_size = int(model_features.shape[0])
                observed_batch_sizes.append(batch_size)
                return {
                    "binary_logit": torch.zeros((batch_size, 1), dtype=torch.float32),
                    "subtype_logits": torch.zeros((batch_size, 3), dtype=torch.float32),
                }

            runtime.model.forward = recording  # type: ignore[method-assign]
            fixed_binary, fixed_subtype = runtime.predict_logits(
                _fixed_cohort("validation"),
                expected_split="validation",
                batch_size=64,
                require_fixed_cohort=True,
            )
            self.assertEqual(observed_batch_sizes, [64, 64, 64, 64, 22])
            self.assertEqual(fixed_binary.shape, (240,))
            self.assertEqual(fixed_subtype.shape, (240, 3))
            self.assertEqual(
                manifest["inference_contract"]["fixed_cohort_wp_prefix_padding"],
                38,
            )
            self.assertEqual(manifest["inference_contract"]["runtime"]["intra_op_threads"], 8)
            self.assertEqual(manifest["inference_contract"]["runtime"]["inter_op_threads"], 1)
            self.assertFalse(manifest["formal_score_entry"]["caller_records_allowed"])

            with self.assertRaises(FileExistsError):
                build_candidate_bundle(
                    project_root=ROOT,
                    training_identity_path=ROOT
                    / manifest["identity_files"]["training_candidate_identity"]["path"],
                    expected_training_identity_sha256=manifest["training_candidate_identity"][
                        "original_bytes_sha256"
                    ],
                    release_identity_path=ROOT
                    / manifest["identity_files"]["release_implementation_identity"]["path"],
                    expected_release_identity_sha256=manifest["release_implementation_identity"][
                        "original_bytes_sha256"
                    ],
                    output_dir=bundle,
                )

    def test_loader_rejects_manifest_path_escape_even_with_new_external_hash(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-escape-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, _manifest_sha = _make_score_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["model_state"]["path"] = "../model_state.npz"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(CandidateManifestError, "relative path"):
                load_candidate_from_manifest(
                    manifest_path,
                    expected_manifest_sha256=_sha256(manifest_path),
                )

    def test_release_source_archive_preserves_bytes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "source/a.py"
            second = root / "source/b.py"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"A = 1\n")
            second.write_bytes(b"B = 2\n")
            identity = root / "identity/release.json"
            archive = root / "identity/release.zip"
            frozen = freeze_release_implementation_identity(
                project_root=root,
                source_files=(Path("source/a.py"), Path("source/b.py")),
                archive_path=archive,
                identity_path=identity,
                git_head="abc123",
            )
            self.assertEqual(frozen.identity["files"]["source/a.py"]["sha256"], _sha256(first))
            self.assertEqual(frozen.identity["files"]["source/b.py"]["sha256"], _sha256(second))
            self.assertEqual(frozen.identity["source_archive"]["sha256"], _sha256(archive))
            with self.assertRaises(FileExistsError):
                freeze_release_implementation_identity(
                    project_root=root,
                    source_files=(Path("source/a.py"),),
                    archive_path=archive,
                    identity_path=identity,
                    git_head="abc123",
                )


class M0RHFormalScoreEntryContractTest(unittest.TestCase):
    def test_score_cli_exposes_only_the_four_frozen_entry_arguments(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(CLI), "score-frozen-wp", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for allowed in (
            "--project-root",
            "--manifest",
            "--expected-manifest-sha256",
            "--output-dir",
        ):
            self.assertIn(allowed, completed.stdout)
        for forbidden in (
            "--records",
            "--expected-split",
            "--batch-size",
            "--threshold",
            "--seed",
            "--threads",
        ):
            self.assertNotIn(forbidden, completed.stdout)

    def test_evaluate_records_rejects_test_before_loading_a_manifest(self) -> None:
        with self.assertRaisesRegex(WPReleaseDataError, "validation-only"):
            run_wp_records_evaluation(
                manifest_path=Path("does-not-exist.json"),
                expected_manifest_sha256="0" * 64,
                records_path=Path("does-not-exist.jsonl"),
                expected_split="test",
                output_dir=Path("does-not-exist-output"),
                batch_size=64,
            )

    def test_canonical_record_and_order_hashes_are_order_sensitive(self) -> None:
        rows = _fixed_cohort("test")
        first = canonical_wp_cohort_identity(rows, expected_split="test")
        reordered = list(rows)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        second = canonical_wp_cohort_identity(reordered, expected_split="test")
        self.assertNotEqual(first["canonical_records_sha256"], second["canonical_records_sha256"])
        self.assertNotEqual(first["ordered_sample_ids_sha256"], second["ordered_sample_ids_sha256"])
        self.assertEqual(first["sample_count"], 240)
        self.assertEqual(first["pattern_counts"], {name: 60 for name in ("direct", "pacing", "lapping", "random")})
        self.assertEqual(first["binary_counts"], {"0": 60, "1": 180})

    def test_runtime_contract_is_set_and_read_back_in_a_fresh_process(self) -> None:
        code = (
            "import json; "
            "from elderly_monitoring.modules.mental_health.wandering.release import "
            "configure_release_cpu_runtime; "
            "print(json.dumps(configure_release_cpu_runtime({"
            "'device':'cpu','intra_op_threads':8,'inter_op_threads':1,"
            "'batch_size':64,'num_workers':0,'pin_memory':False})), flush=True)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(completed.stdout.strip())
        self.assertEqual(
            observed,
            {
                "device": "cpu",
                "intra_op_threads": 8,
                "inter_op_threads": 1,
                "batch_size": 64,
                "num_workers": 0,
                "pin_memory": False,
            },
        )

    def test_formal_controller_uses_only_internal_frozen_accessor_and_atomic_output(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-score-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, manifest_sha = _make_score_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = _fixed_cohort("test")
            fake_bundle = _FakeFrozenBundle(manifest, records)
            output = root / "formal-score"
            logits = np.zeros(240, dtype=np.float64)
            subtype = np.zeros((240, 3), dtype=np.float64)
            with (
                mock.patch.object(
                    release_module,
                    "_verify_frozen_wp_upstream_files",
                    return_value={"all_upstream_hashes_verified": True},
                ) as upstream,
                mock.patch.object(
                    release_module,
                    "load_preprocessing_bundle",
                    return_value=fake_bundle,
                ) as accessor,
                mock.patch.object(
                    CandidateRuntime,
                    "predict_logits",
                    return_value=(logits, subtype),
                ) as inference,
            ):
                committed = run_authorized_frozen_wp_score(
                    project_root=ROOT,
                    manifest_path=manifest_path,
                    expected_manifest_sha256=manifest_sha,
                    output_dir=output,
                    active_cli_path=CLI,
                )
            self.assertEqual(committed, output.resolve())
            upstream.assert_called_once()
            accessor.assert_called_once()
            self.assertEqual(accessor.call_args.kwargs["mode"], BUNDLE_MODE_FROZEN_WP_TEST)
            self.assertEqual(accessor.call_args.kwargs["project_root"], ROOT.resolve())
            inference.assert_called_once()
            self.assertEqual(inference.call_args.kwargs["expected_split"], "test")
            self.assertEqual(inference.call_args.kwargs["batch_size"], 64)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "artifact_manifest.json",
                    "confusion.json",
                    "errors.jsonl",
                    "execution.json",
                    "metrics.json",
                    "predictions.jsonl",
                ],
            )
            execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual(execution["phase"], "test")
            self.assertEqual(execution["accessor_mode"], BUNDLE_MODE_FROZEN_WP_TEST)
            self.assertEqual(execution["runtime"]["intra_op_threads"], 8)
            self.assertEqual(execution["runtime"]["inter_op_threads"], 1)
            self.assertEqual(execution["runtime"]["batch_size"], 64)
            self.assertTrue(execution["preflight_completed_before_accessor"])
            self.assertFalse(execution["fit_performed"])
            self.assertFalse(execution["optimizer_loaded"])
            self.assertFalse(execution["threshold_search_performed"])
            self.assertIn("canonical_records_sha256", execution["cohort_identity"])
            self.assertIn("ordered_sample_ids_sha256", execution["cohort_identity"])

    def test_output_exists_and_preflight_failures_never_call_accessor_or_inference(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-preflight-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, manifest_sha = _make_score_fixture(root)
            output = root / "already-present"
            output.mkdir()
            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaises(FileExistsError):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=manifest_path,
                        expected_manifest_sha256="0" * 64,
                        output_dir=output,
                        active_cli_path=CLI,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()

            copied_cli = root / "copied-cli.py"
            shutil.copyfile(CLI, copied_cli)
            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(ReleaseArtifactError, "active release source path"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        output_dir=root / "active-cli-failure",
                        active_cli_path=copied_cli,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()

            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(CandidateManifestError, "external manifest SHA"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=manifest_path,
                        expected_manifest_sha256="0" * 64,
                        output_dir=root / "manifest-failure",
                        active_cli_path=CLI,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()

    def test_identity_cross_binding_and_invalid_accessor_cohort_fail_before_inference(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-identity-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, manifest_sha = _make_score_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            altered_identity = root / "altered-release-identity.json"
            altered = dict(manifest["release_implementation_identity"]["identity"])
            altered["git_head_context"] = "different-context"
            _write_json(altered_identity, altered)
            crossed = json.loads(manifest_path.read_text(encoding="utf-8"))
            crossed["identity_files"]["release_implementation_identity"] = {
                "path": altered_identity.relative_to(ROOT).as_posix(),
                "sha256": _sha256(altered_identity),
                "size_bytes": altered_identity.stat().st_size,
            }
            crossed_manifest = root / "crossed-identity/candidate_manifest.json"
            _write_json(crossed_manifest, crossed)
            for source in manifest_path.parent.iterdir():
                if source.name != "candidate_manifest.json":
                    shutil.copyfile(source, crossed_manifest.parent / source.name)
            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(ReleaseArtifactError, "cross-binding|embedded identity"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=crossed_manifest,
                        expected_manifest_sha256=_sha256(crossed_manifest),
                        output_dir=root / "crossed-output",
                        active_cli_path=CLI,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()

            fake_bundle = _FakeFrozenBundle(manifest, _fixed_cohort("test")[:-1])
            with (
                mock.patch.object(
                    release_module,
                    "_verify_frozen_wp_upstream_files",
                    return_value={"all_upstream_hashes_verified": True},
                ),
                mock.patch.object(
                    release_module,
                    "load_preprocessing_bundle",
                    return_value=fake_bundle,
                ) as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(WPReleaseDataError, "240"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        output_dir=root / "invalid-cohort-output",
                        active_cli_path=CLI,
                    )
                accessor.assert_called_once()
                inference.assert_not_called()

    def test_model_config_active_source_and_archive_drift_fail_closed_before_accessor(self) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="m0rh-drift-", dir=ROOT / "tmp") as directory:
            root = Path(directory)
            manifest_path, manifest_sha = _make_score_fixture(root)

            for artifact in ("model_state.npz", "forward_config.yaml", "performance_config.yaml", "frozen_wp_rf_config.yaml"):
                with self.subTest(artifact=artifact):
                    copy = root / f"candidate-{artifact.replace('.', '-')}"
                    shutil.copytree(manifest_path.parent, copy)
                    target = copy / artifact
                    target.write_bytes(target.read_bytes() + b"drift")
                    copied_manifest = copy / "candidate_manifest.json"
                    with (
                        mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                        mock.patch.object(CandidateRuntime, "predict_logits") as inference,
                    ):
                        with self.assertRaises(CandidateManifestError):
                            run_authorized_frozen_wp_score(
                                project_root=ROOT,
                                manifest_path=copied_manifest,
                                expected_manifest_sha256=manifest_sha,
                                output_dir=root / f"out-{artifact.replace('.', '-')}",
                                active_cli_path=CLI,
                            )
                        accessor.assert_not_called()
                        inference.assert_not_called()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            release_identity = manifest["release_implementation_identity"]["identity"]
            release_path = "src/elderly_monitoring/modules/mental_health/wandering/release.py"
            release_identity["files"][release_path]["sha256"] = "0" * 64
            source_drift_manifest = root / "source-drift/candidate_manifest.json"
            _write_json(source_drift_manifest, manifest)
            for source in manifest_path.parent.iterdir():
                if source.name != "candidate_manifest.json":
                    shutil.copyfile(source, source_drift_manifest.parent / source.name)
            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(ReleaseArtifactError, "active release source"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=source_drift_manifest,
                        expected_manifest_sha256=_sha256(source_drift_manifest),
                        output_dir=root / "out-source-drift",
                        active_cli_path=CLI,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()

            clean = json.loads(manifest_path.read_text(encoding="utf-8"))
            archive_descriptor = clean["release_implementation_identity"]["identity"]["source_archive"]
            original_archive = ROOT / archive_descriptor["path"]
            altered_archive = root / "release-source-extra-entry.zip"
            shutil.copyfile(original_archive, altered_archive)
            with zipfile.ZipFile(altered_archive, mode="a", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("unexpected.txt", b"unexpected")
            archive_descriptor["path"] = altered_archive.relative_to(ROOT).as_posix()
            archive_descriptor["sha256"] = _sha256(altered_archive)
            archive_descriptor["size_bytes"] = altered_archive.stat().st_size
            archive_drift_manifest = root / "archive-drift/candidate_manifest.json"
            _write_json(archive_drift_manifest, clean)
            for source in manifest_path.parent.iterdir():
                if source.name != "candidate_manifest.json":
                    shutil.copyfile(source, archive_drift_manifest.parent / source.name)
            with (
                mock.patch.object(release_module, "load_preprocessing_bundle") as accessor,
                mock.patch.object(CandidateRuntime, "predict_logits") as inference,
            ):
                with self.assertRaisesRegex(ReleaseArtifactError, "archive entry set"):
                    run_authorized_frozen_wp_score(
                        project_root=ROOT,
                        manifest_path=archive_drift_manifest,
                        expected_manifest_sha256=_sha256(archive_drift_manifest),
                        output_dir=root / "out-archive-drift",
                        active_cli_path=CLI,
                    )
                accessor.assert_not_called()
                inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
