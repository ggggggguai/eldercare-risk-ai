from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch
import yaml

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

from elderly_monitoring.modules.mental_health.wandering import model as model_module
from elderly_monitoring.modules.mental_health.wandering.model import (
    TOPOWANDER_PARAMETER_COUNT,
    ExplicitRelationAttention,
    TopoWanderContractError,
    TopoWanderMPT,
    TopoWanderStateError,
    build_patch_geometry,
    build_symmetric_relation_features,
    create_topowander_model,
    deterministic_npz_bytes,
    hierarchical_four_class_probabilities,
    load_topowander_config,
    safe_load_topowander_model,
    signed_offset_indices,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "modules" / "wandering_topowander_mpt_v1.yaml"
MODEL_PATH = ROOT / "src" / "elderly_monitoring" / "modules" / "mental_health" / "wandering" / "model.py"


class _StatefulMapping(Mapping[str, object]):
    """Expose exact values once, then a different view on later reads."""

    def __init__(self, first: dict[str, object], later: dict[str, object]) -> None:
        self._first = first
        self._later = later
        self._read_counts: dict[str, int] = {}

    def __getitem__(self, key: str) -> object:
        count = self._read_counts.get(key, 0)
        self._read_counts[key] = count + 1
        return (self._first if count == 0 else self._later)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._first)

    def __len__(self) -> int:
        return len(self._first)


def _valid_inputs(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.linspace(-1.0, 1.0, 80, dtype=torch.float32)
    points = torch.stack((t, 0.15 * torch.sin(t * 5.0)), dim=1)
    points = points.unsqueeze(0).repeat(batch_size, 1, 1)
    features = torch.zeros((batch_size, 80, 14), dtype=torch.float32)
    features[:, :, 0:2] = points
    features[:, :, 2] = 0.1
    features[:, :, 5] = 0.5
    mask = torch.ones((batch_size, 80), dtype=torch.float32)
    features[:, :, 12] = mask
    features[:, :, 13] = 1.0
    return features, points, mask


def _forward_fixture_v1() -> dict[str, torch.Tensor]:
    t = torch.linspace(-1.0, 1.0, 80, dtype=torch.float32, device="cpu")
    points = torch.stack((t, 0.15 * torch.sin(t * 5.0)), dim=1).unsqueeze(0)
    features = torch.zeros((1, 80, 14), dtype=torch.float32, device="cpu")
    features[:, :, 0:2] = points
    features[:, :, 2] = 0.1
    features[:, :, 5] = 0.5
    mask = torch.ones((1, 80), dtype=torch.float32, device="cpu")
    features[:, :, 12] = mask
    features[:, :, 13] = 1.0
    return {
        "model_features": features,
        "shape_normalized_points": points,
        "point_mask": mask,
    }


def _deterministic_tensor_npz_v1(tensors: dict[str, torch.Tensor]) -> bytes:
    keys = list(tensors)
    assert len(keys) == len(set(keys))
    assert all(isinstance(key, str) and key.isascii() for key in keys)
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for key in sorted(tensors, key=lambda value: value.encode("ascii")):
            tensor = tensors[key]
            assert isinstance(tensor, torch.Tensor)
            array = np.ascontiguousarray(
                tensor.detach().cpu().numpy(),
                dtype=np.dtype("<f4"),
            )
            assert array.dtype.str == "<f4" and np.isfinite(array).all()
            array_bytes = io.BytesIO()
            np.lib.format.write_array(
                array_bytes,
                array,
                version=(1, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, array_bytes.getvalue())
    return output.getvalue()


def _partial_inputs(valid_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, points, mask = _valid_inputs()
    mask.zero_()
    mask[:, valid_indices] = 1.0
    features[:, :, 12] = mask
    features[:, :, 13] = mask
    return features, points, mask


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _npz_payload(
    arrays: dict[str, np.ndarray],
    *,
    timestamp: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for key in sorted(arrays):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(array_bytes, arrays[key], allow_pickle=True)
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=timestamp)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_bytes.getvalue())
    return output.getvalue()


class TopoWanderConfigAndInputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_topowander_config(CONFIG_PATH)

    def test_exact_config_and_parameter_count_are_frozen(self) -> None:
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7",
        )
        self.assertEqual(self.config["schema_version"], "wandering-topowander-mpt-forward-config-v1")
        self.assertEqual(self.config["purpose"], "architecture_forward_contract_only")
        self.assertEqual(self.config["patch"]["count"], 19)
        self.assertEqual(self.config["patch"]["semantic_projection"]["input_channels"], 11)
        self.assertEqual(self.config["relation"]["signed_offset"]["bucket_count"], 37)
        self.assertEqual(self.config["state"]["parameter_count"], 204466)
        self.assertEqual(
            self.config["verification"],
            {
                "device": "cpu",
                "amp": False,
                "deterministic_algorithms": True,
                "intra_op_threads": 1,
                "inter_op_threads": 1,
            },
        )
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(TOPOWANDER_PARAMETER_COUNT, 204466)

        model = create_topowander_model(self.config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 204466)

    def test_exact_config_rejects_missing_extra_enum_and_numeric_drift(self) -> None:
        mutations = []
        missing = copy.deepcopy(self.config)
        del missing["input"]["sequence_length"]
        mutations.append(missing)
        extra = copy.deepcopy(self.config)
        extra["unexpected"] = True
        mutations.append(extra)
        wrong_enum = copy.deepcopy(self.config)
        wrong_enum["tcn"]["convolution"] = "causal"
        mutations.append(wrong_enum)
        numeric_drift = copy.deepcopy(self.config)
        numeric_drift["patch"]["minimum_valid_points"] = 5
        mutations.append(numeric_drift)

        with tempfile.TemporaryDirectory() as directory:
            for index, mutation in enumerate(mutations):
                with self.subTest(index=index):
                    path = Path(directory) / f"mutation-{index}.yaml"
                    _write_yaml(path, mutation)
                    with self.assertRaises(TopoWanderContractError):
                        load_topowander_config(path)

    def test_public_config_loader_reads_one_raw_byte_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "stable-config.yaml"
            config_path.write_bytes(CONFIG_PATH.read_bytes())
            original_open = Path.open
            config_read_modes: list[str] = []

            def guarded_open(path: Path, *args: object, **kwargs: object):
                if path == config_path:
                    mode = kwargs.get("mode", args[0] if args else "r")
                    config_read_modes.append(str(mode))
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", autospec=True, side_effect=guarded_open):
                loaded = load_topowander_config(config_path)

        self.assertEqual(loaded, self.config)
        self.assertEqual(config_read_modes, ["rb"])

    def test_stateful_mappings_are_rejected_before_model_construction(self) -> None:
        exact_top_level = copy.deepcopy(self.config)
        drifted_top_level = copy.deepcopy(self.config)
        drifted_top_level["relation"]["center_distance_scale"] = 0.20

        nested_config = copy.deepcopy(self.config)
        exact_relation = copy.deepcopy(self.config["relation"])
        drifted_relation = copy.deepcopy(exact_relation)
        drifted_relation["center_distance_scale"] = 0.20
        nested_config["relation"] = _StatefulMapping(exact_relation, drifted_relation)

        cases = {
            "top_level": _StatefulMapping(exact_top_level, drifted_top_level),
            "nested": nested_config,
        }
        original_build_modules = TopoWanderMPT._build_modules
        original_initialize_parameters = TopoWanderMPT._initialize_parameters
        for name, candidate in cases.items():
            with self.subTest(name=name):
                with (
                    mock.patch.object(
                        TopoWanderMPT,
                        "_build_modules",
                        autospec=True,
                        side_effect=original_build_modules,
                    ) as build_modules,
                    mock.patch.object(
                        TopoWanderMPT,
                        "_initialize_parameters",
                        autospec=True,
                        side_effect=original_initialize_parameters,
                    ) as initialize_parameters,
                ):
                    with self.assertRaisesRegex(
                        TopoWanderContractError,
                        "model construction requires the exact TopoWander v1 config",
                    ):
                        TopoWanderMPT(candidate)
                    build_modules.assert_not_called()
                    initialize_parameters.assert_not_called()

        self.assertEqual(TopoWanderMPT.__init__.__annotations__["config"], "dict[str, Any]")
        self.assertEqual(create_topowander_model.__annotations__["config"], "dict[str, Any]")

    def test_exact_config_rejects_duplicate_yaml_keys_at_every_mapping_depth(self) -> None:
        source = CONFIG_PATH.read_text(encoding="utf-8")
        schema_line = "schema_version: wandering-topowander-mpt-forward-config-v1"
        sequence_line = "  sequence_length: 80"
        duplicate_cases = {
            "top_same": f"{source.rstrip()}\n{schema_line}\n",
            "top_different": (
                source.replace(schema_line, "schema_version: rejected-duplicate-first-value", 1).rstrip()
                + f"\n{schema_line}\n"
            ),
            "nested_same": source.replace(sequence_line, f"{sequence_line}\n{sequence_line}", 1),
            "nested_different": source.replace(
                sequence_line,
                f"  sequence_length: 79\n{sequence_line}",
                1,
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            for name, payload in duplicate_cases.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.yaml"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(TopoWanderContractError):
                        load_topowander_config(path)

    def test_value_equal_type_drift_is_rejected_by_loader_and_constructor_before_parameters(self) -> None:
        mutations: dict[str, dict[str, object]] = {}

        bool_false_to_int = copy.deepcopy(self.config)
        bool_false_to_int["input"]["temporal_features_enabled"] = 0
        mutations["bool_false_to_int"] = bool_false_to_int

        bool_true_to_int = copy.deepcopy(self.config)
        bool_true_to_int["input"]["require_same_device"] = 1
        mutations["bool_true_to_int"] = bool_true_to_int

        int_to_bool = copy.deepcopy(self.config)
        int_to_bool["tcn"]["input_projection"]["kernel_size"] = True
        mutations["int_to_bool"] = int_to_bool

        int_to_float = copy.deepcopy(self.config)
        int_to_float["state"]["parameter_count"] = 204466.0
        mutations["int_to_float"] = int_to_float

        float_to_int = copy.deepcopy(self.config)
        float_to_int["relation"]["center_distance_max"] = 4
        mutations["float_to_int"] = float_to_int

        list_element_type_drift = copy.deepcopy(self.config)
        list_element_type_drift["state"]["zip_timestamp"][0] = 1980.0
        mutations["list_element_int_to_float"] = list_element_type_drift

        with tempfile.TemporaryDirectory() as directory:
            for name, mutation in mutations.items():
                with self.subTest(loader=name):
                    path = Path(directory) / f"{name}.yaml"
                    _write_yaml(path, mutation)
                    with self.assertRaises(TopoWanderContractError):
                        load_topowander_config(path)

                with self.subTest(constructor=name):
                    with (
                        mock.patch.object(TopoWanderMPT, "_build_modules", autospec=True) as build_modules,
                        mock.patch.object(
                            TopoWanderMPT,
                            "_initialize_parameters",
                            autospec=True,
                        ) as initialize_parameters,
                    ):
                        with self.assertRaises(TopoWanderContractError):
                            TopoWanderMPT(mutation)
                        build_modules.assert_not_called()
                        initialize_parameters.assert_not_called()

    def test_root_non_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "root-list.yaml"
            _write_yaml(path, [])
            with self.assertRaises(TopoWanderContractError):
                load_topowander_config(path)
        with self.assertRaises(TopoWanderContractError):
            TopoWanderMPT([])

    def test_invalid_exact_config_prevents_model_creation_state_read_and_assignment(self) -> None:
        invalid_config = copy.deepcopy(self.config)
        invalid_config["state"]["parameter_count"] = 204466.0

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "type-drift.yaml"
            state_path = Path(directory) / "must-not-be-read.npz"
            _write_yaml(config_path, invalid_config)
            expected_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
            original_read_bytes = Path.read_bytes
            state_read_count = 0

            def guarded_read_bytes(path: Path) -> bytes:
                nonlocal state_read_count
                if path == state_path:
                    state_read_count += 1
                    raise AssertionError("numeric state was read before exact config rejection")
                return original_read_bytes(path)

            with (
                mock.patch.object(Path, "read_bytes", autospec=True, side_effect=guarded_read_bytes),
                mock.patch.object(model_module, "create_topowander_model", autospec=True) as create_model,
                mock.patch.object(TopoWanderMPT, "load_state_dict", autospec=True) as assign,
            ):
                with self.assertRaisesRegex(
                    TopoWanderStateError,
                    "trusted config does not match the exact forward contract",
                ):
                    safe_load_topowander_model(
                        state_path,
                        config_path=config_path,
                        expected_config_sha256=expected_sha256,
                    )
                create_model.assert_not_called()
                assign.assert_not_called()
            self.assertEqual(state_read_count, 0)

    def test_safe_loader_hashes_and_parses_one_config_byte_snapshot(self) -> None:
        valid_config_bytes = CONFIG_PATH.read_bytes()
        invalid_config_bytes = valid_config_bytes.replace(
            b"parameter_count: 204466",
            b"parameter_count: 204466.0",
            1,
        )
        self.assertNotEqual(invalid_config_bytes, valid_config_bytes)
        invalid_config_sha256 = hashlib.sha256(invalid_config_bytes).hexdigest()
        state_payload = deterministic_npz_bytes(create_topowander_model(self.config))

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "changing-config.yaml"
            state_path = Path(directory) / "state-must-not-be-read.npz"
            config_path.write_bytes(valid_config_bytes)
            state_path.write_bytes(state_payload)
            original_open = Path.open
            config_read_count = 0
            state_read_count = 0

            def guarded_open(path: Path, *args: object, **kwargs: object):
                nonlocal config_read_count, state_read_count
                if path == config_path:
                    config_read_count += 1
                    if config_read_count == 1:
                        return io.BytesIO(invalid_config_bytes)
                if path == state_path:
                    state_read_count += 1
                return original_open(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "open", autospec=True, side_effect=guarded_open),
                mock.patch.object(
                    model_module,
                    "create_topowander_model",
                    autospec=True,
                    wraps=create_topowander_model,
                ) as create_model,
                mock.patch.object(TopoWanderMPT, "load_state_dict", autospec=True) as assign,
            ):
                with self.assertRaisesRegex(
                    TopoWanderStateError,
                    "trusted config does not match the exact forward contract",
                ):
                    safe_load_topowander_model(
                        state_path,
                        config_path=config_path,
                        expected_config_sha256=invalid_config_sha256,
                    )
                self.assertEqual(config_read_count, 1)
                self.assertEqual(state_read_count, 0)
                create_model.assert_not_called()
                assign.assert_not_called()

    def test_model_config_is_recursive_immutable_snapshot(self) -> None:
        with self.subTest("caller_mapping_mutation_does_not_change_forward"):
            caller_config = load_topowander_config(CONFIG_PATH)
            model = create_topowander_model(caller_config).eval()
            inputs = _valid_inputs()
            baseline = model(*inputs)
            caller_config["relation"]["center_distance_scale"] = 0.20
            changed = model(*inputs)
            for key in baseline:
                torch.testing.assert_close(baseline[key], changed[key], rtol=0.0, atol=0.0)

        with self.subTest("caller_mapping_is_not_aliased"):
            caller_config = load_topowander_config(CONFIG_PATH)
            model = create_topowander_model(caller_config)
            caller_config["relation"]["center_distance_scale"] = 0.20
            self.assertEqual(model.config["relation"]["center_distance_scale"], 0.10)

        with self.subTest("caller_list_is_not_aliased"):
            caller_config = load_topowander_config(CONFIG_PATH)
            expected_order = tuple(caller_config["heads"]["forward_outputs"])
            model = create_topowander_model(caller_config)
            caller_config["heads"]["forward_outputs"][0] = "mutated"
            self.assertEqual(tuple(model.config["heads"]["forward_outputs"]), expected_order)

        with self.subTest("exposed_mapping_is_read_only"):
            model = create_topowander_model(load_topowander_config(CONFIG_PATH))
            with self.assertRaises(TypeError):
                model.config["relation"]["center_distance_scale"] = 0.20

        with self.subTest("exposed_sequence_is_read_only"):
            model = create_topowander_model(load_topowander_config(CONFIG_PATH))
            with self.assertRaises(TypeError):
                model.config["heads"]["forward_outputs"][0] = "mutated"

        with self.subTest("config_attribute_cannot_be_rebound"):
            model = create_topowander_model(load_topowander_config(CONFIG_PATH))
            with self.assertRaises(AttributeError):
                model.config = load_topowander_config(CONFIG_PATH)

    def test_valid_full_and_partial_masks_run(self) -> None:
        model = create_topowander_model(self.config).eval()
        for inputs in (
            _valid_inputs(),
            _partial_inputs(list(range(0, 16)) + list(range(24, 40))),
        ):
            with self.subTest(valid_points=int(inputs[2].sum().item())):
                outputs = model(*inputs)
                self.assertEqual(tuple(outputs["binary_logit"].shape), (1, 1))
                self.assertEqual(tuple(outputs["subtype_logits"].shape), (1, 3))
                self.assertEqual(tuple(outputs["projection_embedding"].shape), (1, 64))

    def test_input_shape_dtype_device_and_value_contracts_fail_closed(self) -> None:
        features, points, mask = _valid_inputs()

        invalid_cases: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = [
            ("empty_batch", features[:0], points[:0], mask[:0]),
            ("wrong_sequence", features[:, :-1], points[:, :-1], mask[:, :-1]),
            ("wrong_feature_channels", features[:, :, :-1], points, mask),
            ("wrong_shape_channels", features, points[:, :, :1], mask),
            ("wrong_mask_shape", features, points, mask[:, :-1]),
            ("feature_dtype", features.to(torch.float64), points, mask),
            ("shape_dtype", features, points.to(torch.float64), mask),
            ("mask_dtype", features, points, mask.to(torch.bool)),
        ]
        for name, candidate_features, candidate_points, candidate_mask in invalid_cases:
            with self.subTest(name=name), self.assertRaises(TopoWanderContractError):
                create_topowander_model(self.config)(candidate_features, candidate_points, candidate_mask)

        meta_mask = torch.empty(mask.shape, dtype=torch.float32, device="meta")
        with self.assertRaisesRegex(TopoWanderContractError, "same device"):
            create_topowander_model(self.config)(features, points, meta_mask)

        for name, tensor_index, index in (
            ("feature_nan", 0, (0, 0, 0)),
            ("shape_inf", 1, (0, 0, 0)),
            ("mask_nan", 2, (0, 0)),
        ):
            candidates = [value.clone() for value in (features, points, mask)]
            candidates[tensor_index][index] = float("nan") if name != "shape_inf" else float("inf")
            with self.subTest(name=name), self.assertRaises(TopoWanderContractError):
                create_topowander_model(self.config)(*candidates)

    def test_mask_quality_time_and_minimum_validity_contracts_fail_closed(self) -> None:
        features, points, mask = _valid_inputs()
        cases = []

        nonbinary = mask.clone()
        nonbinary[0, 0] = 0.5
        cases.append((features, points, nonbinary))

        wrong_channel = features.clone()
        wrong_channel[0, 0, 12] = 0.0
        cases.append((wrong_channel, points, mask))

        wrong_quality = features.clone()
        wrong_quality[0, 0, 13] = 1.01
        cases.append((wrong_quality, points, mask))

        wrong_time = features.clone()
        wrong_time[0, 0, 10] = 1.0
        cases.append((wrong_time, points, mask))

        all_zero = _partial_inputs([])
        too_few = _partial_inputs(list(range(7)))
        no_valid_patch = _partial_inputs(list(range(0, 4)) + list(range(20, 24)))
        cases.extend((all_zero, too_few, no_valid_patch))

        model = create_topowander_model(self.config)
        for index, candidate in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(TopoWanderContractError):
                model(*candidate)


class TopoWanderPatchAndRelationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_topowander_config(CONFIG_PATH)

    def _geometry_from_first_patch(self, first_patch: list[tuple[float, float]], quality: float = 1.0):
        points = torch.zeros((1, 80, 2), dtype=torch.float32)
        points[0, :8] = torch.tensor(first_patch, dtype=torch.float32)
        mask = torch.zeros((1, 80), dtype=torch.float32)
        mask[0, :8] = 1.0
        quality_values = torch.zeros((1, 80), dtype=torch.float32)
        quality_values[0, :8] = quality
        return build_patch_geometry(points, mask, quality_values, self.config["patch"])

    def test_patch_count_threshold_and_no_gap_bridging(self) -> None:
        _, points, _ = _valid_inputs()
        quality = torch.ones((1, 80), dtype=torch.float32)
        full_mask = torch.ones((1, 80), dtype=torch.float32)
        geometry = build_patch_geometry(points, full_mask, quality, self.config["patch"])
        self.assertEqual(tuple(geometry.semantics.shape), (1, 19, 11))
        self.assertEqual(int(geometry.patch_mask.sum().item()), 19)

        mask = torch.zeros((1, 80), dtype=torch.float32)
        mask[0, [0, 1, 2, 4, 5, 6]] = 1.0
        gap_points = torch.zeros((1, 80, 2), dtype=torch.float32)
        gap_points[0, 0:3, 0] = torch.tensor([0.0, 1.0, 2.0])
        gap_points[0, 4:7, 0] = torch.tensor([10.0, 11.0, 12.0])
        gap_geometry = build_patch_geometry(gap_points, mask, quality, self.config["patch"])
        self.assertEqual(float(gap_geometry.patch_mask[0, 0]), 1.0)
        self.assertAlmostEqual(float(gap_geometry.semantics[0, 0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(gap_geometry.semantics[0, 0, 1]), 4.0, places=6)

        mask[0, 6] = 0.0
        invalid_geometry = build_patch_geometry(gap_points, mask, quality, self.config["patch"])
        self.assertEqual(float(invalid_geometry.patch_mask[0, 0]), 0.0)
        self.assertTrue(torch.equal(invalid_geometry.semantics[0, 0], torch.zeros(11)))

    def test_straight_reversal_and_closed_loop_semantics_have_manual_truth(self) -> None:
        straight = [(float(index), 0.0) for index in range(8)]
        straight_geometry = self._geometry_from_first_patch(straight, quality=0.75)
        expected_straight = torch.tensor(
            [7.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.75],
            dtype=torch.float32,
        )
        torch.testing.assert_close(straight_geometry.semantics[0, 0], expected_straight, rtol=0.0, atol=1e-6)

        reversal = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
        reversal_geometry = self._geometry_from_first_patch(reversal)
        expected_reversal = torch.tensor(
            [7.0, 1.0, 1.0 / 7.0, torch.pi / 6.0, torch.pi, torch.pi, 1.0, 0.1, 0.0, 1.0, 1.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(reversal_geometry.semantics[0, 0], expected_reversal, rtol=0.0, atol=1e-6)

        loop = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        loop_geometry = self._geometry_from_first_patch(loop)
        expected_loop = torch.tensor(
            [7.0, 1.0, 1.0 / 7.0, torch.pi / 2.0, torch.pi / 2.0, 3.0 * torch.pi, 1.0, 0.1, 1.0, 1.0, 1.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(loop_geometry.semantics[0, 0], expected_loop, rtol=0.0, atol=1e-6)

    def test_prior_patch_revisit_distance_has_manual_truth(self) -> None:
        points = torch.zeros((1, 80, 2), dtype=torch.float32)
        base = torch.stack((torch.linspace(-0.2, 0.2, 8), torch.zeros(8)), dim=1)
        points[0, 0:8] = base
        points[0, 8:16] = base
        mask = torch.zeros((1, 80), dtype=torch.float32)
        mask[0, 0:16] = 1.0
        quality = mask.clone()
        geometry = build_patch_geometry(points, mask, quality, self.config["patch"])
        self.assertEqual(float(geometry.patch_mask[0, 2]), 1.0)
        self.assertAlmostEqual(float(geometry.semantics[0, 2, 7]), 0.0, places=6)

    def test_relation_features_are_symmetric_and_signed_offsets_are_directional(self) -> None:
        _, points, _ = _valid_inputs()
        mask = torch.ones((1, 80), dtype=torch.float32)
        quality = torch.linspace(0.5, 1.0, 80).unsqueeze(0)
        geometry = build_patch_geometry(points, mask, quality, self.config["patch"])
        relation, pair_mask = build_symmetric_relation_features(geometry, self.config["relation"])
        torch.testing.assert_close(relation, relation.transpose(1, 2), rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(pair_mask, pair_mask.transpose(1, 2)))

        indices = signed_offset_indices(19, self.config["relation"]["signed_offset"])
        self.assertEqual(tuple(indices.shape), (19, 19))
        self.assertEqual(int(indices[0, 0]), 18)
        self.assertEqual(int(indices[0, 1]), 19)
        self.assertEqual(int(indices[1, 0]), 17)
        self.assertEqual(int(indices[0, 18]), 36)
        self.assertEqual(int(indices[18, 0]), 0)

    def test_cls_bias_is_zero_and_invalid_keys_receive_zero_attention_weight(self) -> None:
        model = create_topowander_model(self.config).eval()
        features, points, mask = _partial_inputs(list(range(0, 16)))
        encoding = model.encode(features, points, mask, return_attention=True)
        relation_bias = encoding.relation_bias
        self.assertTrue(torch.equal(relation_bias[:, :, 0, :], torch.zeros_like(relation_bias[:, :, 0, :])))
        self.assertTrue(torch.equal(relation_bias[:, :, :, 0], torch.zeros_like(relation_bias[:, :, :, 0])))

        token_mask = torch.cat((torch.ones((1, 1), dtype=torch.bool), encoding.patch_mask.to(torch.bool)), dim=1)
        invalid_keys = ~token_mask[0]
        self.assertTrue(bool(invalid_keys.any()))
        for weights in encoding.attention_weights:
            self.assertTrue(torch.equal(weights[0, :, :, invalid_keys], torch.zeros_like(weights[0, :, :, invalid_keys])))
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), rtol=0.0, atol=1e-6)

    def test_explicit_attention_requires_additive_four_head_bias(self) -> None:
        attention = ExplicitRelationAttention(d_model=96, heads=4, dropout=0.0)
        values = torch.zeros((1, 20, 96), dtype=torch.float32)
        key_mask = torch.ones((1, 20), dtype=torch.bool)
        with self.assertRaises(TopoWanderContractError):
            attention(values, key_mask, torch.zeros((1, 20, 20)))


class TopoWanderForwardGradientAndStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_topowander_config(CONFIG_PATH)
        cls.config_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

    def test_forward_signature_v1_baseline(self) -> None:
        fixture = _forward_fixture_v1()
        self.assertEqual(
            set(fixture),
            {"model_features", "shape_normalized_points", "point_mask"},
        )
        self.assertEqual(tuple(fixture["model_features"].shape), (1, 80, 14))
        self.assertEqual(tuple(fixture["shape_normalized_points"].shape), (1, 80, 2))
        self.assertEqual(tuple(fixture["point_mask"].shape), (1, 80))
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                and tensor.dtype == torch.float32
                and bool(torch.isfinite(tensor).all())
                for tensor in fixture.values()
            )
        )

        fixture_payload = _deterministic_tensor_npz_v1(fixture)
        self.assertEqual(len(fixture_payload), 6192)
        self.assertEqual(
            hashlib.sha256(fixture_payload).hexdigest(),
            "584d22203c5df5860e19eca9f8d758d7d33eb40a46096eb2025c7ee58b457c72",
        )

        model = create_topowander_model(self.config).cpu().eval()
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        with torch.no_grad():
            outputs = model(
                fixture["model_features"],
                fixture["shape_normalized_points"],
                fixture["point_mask"],
            )
        expected_shapes = {
            "binary_logit": (1, 1),
            "subtype_logits": (1, 3),
            "projection_embedding": (1, 64),
        }
        expected_output_order = (
            "binary_logit",
            "subtype_logits",
            "projection_embedding",
        )
        self.assertEqual(tuple(outputs), tuple(self.config["heads"]["forward_outputs"]))
        self.assertEqual(tuple(outputs), expected_output_order)
        self.assertEqual(set(outputs), set(expected_shapes))
        for key, shape in expected_shapes.items():
            self.assertEqual(tuple(outputs[key].shape), shape)
            self.assertEqual(outputs[key].dtype, torch.float32)
            self.assertEqual(outputs[key].device.type, "cpu")
            self.assertTrue(bool(torch.isfinite(outputs[key]).all()))

        output_payload = _deterministic_tensor_npz_v1(outputs)
        self.assertEqual(len(output_payload), 1022)
        self.assertEqual(
            hashlib.sha256(output_payload).hexdigest(),
            "01415e02b01c71ad1d96fc2d5bc7b1fd8a0e400c45cd480ea8a62092293ef61c",
        )
        with zipfile.ZipFile(io.BytesIO(output_payload), mode="r") as archive:
            self.assertEqual(archive.comment, b"")
            self.assertEqual(
                archive.namelist(),
                ["binary_logit.npy", "projection_embedding.npy", "subtype_logits.npy"],
            )
            self.assertEqual([info.file_size for info in archive.infolist()], [132, 384, 140])
            for info in archive.infolist():
                self.assertFalse(info.is_dir())
                self.assertEqual(info.extra, b"")
                self.assertEqual(info.comment, b"")

    def test_eval_batch_permutation_and_single_batch_equivalence(self) -> None:
        model = create_topowander_model(self.config).eval()
        full = _valid_inputs()
        partial_a = _partial_inputs(list(range(0, 24)))
        partial_b = _partial_inputs(list(range(8, 40)))
        batch = tuple(torch.cat(items, dim=0) for items in zip(full, partial_a, partial_b))
        outputs = model(*batch)

        for index, sample in enumerate(zip(*[tensor.split(1, dim=0) for tensor in batch])):
            single = model(*sample)
            for key in outputs:
                torch.testing.assert_close(outputs[key][index : index + 1], single[key], rtol=1e-5, atol=1e-6)

        permutation = torch.tensor([2, 0, 1])
        permuted = model(*(tensor[permutation] for tensor in batch))
        for key in outputs:
            torch.testing.assert_close(permuted[key], outputs[key][permutation], rtol=1e-5, atol=1e-6)

    def test_masked_values_are_output_invariant_and_receive_zero_input_gradient(self) -> None:
        model = create_topowander_model(self.config).eval()
        features, points, mask = _partial_inputs(list(range(0, 16)) + list(range(24, 40)))
        changed_features = features.clone()
        changed_points = points.clone()
        invalid = mask == 0.0
        changed_features[invalid] = torch.linspace(-50.0, 50.0, int(invalid.sum()) * 14).reshape(-1, 14)
        changed_features[:, :, 12] = mask
        changed_features[:, :, 13][invalid] = 0.25
        changed_points[invalid] = torch.linspace(-100.0, 100.0, int(invalid.sum()) * 2).reshape(-1, 2)

        baseline = model(features, points, mask)
        changed = model(changed_features, changed_points, mask)
        for key in baseline:
            torch.testing.assert_close(baseline[key], changed[key], rtol=0.0, atol=0.0)

        grad_features = changed_features.clone().requires_grad_(True)
        grad_points = changed_points.clone().requires_grad_(True)
        outputs = model(grad_features, grad_points, mask)
        scalar = sum(value.sum() for value in outputs.values())
        scalar.backward()
        self.assertTrue(torch.equal(grad_features.grad[invalid], torch.zeros_like(grad_features.grad[invalid])))
        self.assertTrue(torch.equal(grad_points.grad[invalid], torch.zeros_like(grad_points.grad[invalid])))

    def test_outputs_probability_helper_and_all_parameter_gradients(self) -> None:
        model = create_topowander_model(self.config)
        features, points, mask = _valid_inputs(batch_size=2)
        outputs = model(features, points, mask)
        self.assertEqual(tuple(outputs["binary_logit"].shape), (2, 1))
        self.assertEqual(tuple(outputs["subtype_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["projection_embedding"].shape), (2, 64))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in outputs.values()))

        probabilities = hierarchical_four_class_probabilities(outputs["binary_logit"], outputs["subtype_logits"])
        self.assertEqual(tuple(probabilities.shape), (2, 4))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2), rtol=0.0, atol=1e-6)

        scalar = sum(value.square().mean() for value in outputs.values())
        scalar.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        nonfinite = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        self.assertEqual(missing, [])
        self.assertEqual(nonfinite, [])

    def test_eval_is_repeatable_and_constructor_preserves_caller_rng(self) -> None:
        torch.manual_seed(123456)
        before = torch.random.get_rng_state().clone()
        model = create_topowander_model(self.config).eval()
        after = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))

        inputs = _valid_inputs()
        first = model(*inputs)
        second = model(*inputs)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)

    def test_deterministic_npz_round_trip_and_metadata(self) -> None:
        model = create_topowander_model(self.config).eval()
        payload_a = deterministic_npz_bytes(model)
        payload_b = deterministic_npz_bytes(model)
        self.assertEqual(payload_a, payload_b)
        self.assertEqual(len(model.state_dict()), 77)
        self.assertEqual(len(payload_a), 838934)
        self.assertEqual(
            hashlib.sha256(payload_a).hexdigest(),
            "6bd7094845c749a2b73502e9480ec9e6fb3df24d313cb4674b41f3903e96cd40",
        )

        with zipfile.ZipFile(io.BytesIO(payload_a), mode="r") as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertEqual(len(names), len(set(names)))
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(info.create_system, 3)
                self.assertEqual((info.external_attr >> 16) & 0o777, 0o600)

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.npz"
            state_path.write_bytes(payload_a)
            loaded = safe_load_topowander_model(
                state_path,
                config_path=CONFIG_PATH,
                expected_config_sha256=self.config_sha256,
            ).eval()
        inputs = _valid_inputs()
        expected = model(*inputs)
        actual = loaded(*inputs)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0.0, atol=0.0)

    def test_safe_loader_rejects_config_and_state_before_any_assignment(self) -> None:
        model = create_topowander_model(self.config)
        good_payload = deterministic_npz_bytes(model)
        with np.load(io.BytesIO(good_payload), allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}

        first_key = sorted(arrays)[0]
        mutations: list[tuple[str, bytes]] = []
        missing = dict(arrays)
        missing.pop(first_key)
        mutations.append(("missing", _npz_payload(missing)))
        extra = dict(arrays)
        extra["unexpected"] = np.zeros((1,), dtype="<f4")
        mutations.append(("extra", _npz_payload(extra)))
        wrong_dtype = dict(arrays)
        wrong_dtype[first_key] = arrays[first_key].astype("<f8")
        mutations.append(("dtype", _npz_payload(wrong_dtype)))
        wrong_endian = dict(arrays)
        wrong_endian[first_key] = arrays[first_key].astype(">f4")
        mutations.append(("endian", _npz_payload(wrong_endian)))
        wrong_shape = dict(arrays)
        wrong_shape[first_key] = np.expand_dims(arrays[first_key], axis=0)
        mutations.append(("shape", _npz_payload(wrong_shape)))
        nonfinite = dict(arrays)
        nonfinite[first_key] = arrays[first_key].copy()
        nonfinite[first_key].reshape(-1)[0] = np.nan
        mutations.append(("nonfinite", _npz_payload(nonfinite)))
        object_array = dict(arrays)
        object_array[first_key] = np.array([object()], dtype=object)
        mutations.append(("object", _npz_payload(object_array)))
        mutations.append(("metadata", _npz_payload(arrays, timestamp=(1980, 1, 2, 0, 0, 0))))

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.npz"
            state_path.write_bytes(good_payload)
            with mock.patch.object(TopoWanderMPT, "load_state_dict", autospec=True) as assign:
                with self.assertRaises(TopoWanderStateError):
                    safe_load_topowander_model(
                        state_path,
                        config_path=CONFIG_PATH,
                        expected_config_sha256="0" * 64,
                    )
                assign.assert_not_called()

            for name, payload in mutations:
                with self.subTest(name=name):
                    state_path.write_bytes(payload)
                    with mock.patch.object(TopoWanderMPT, "load_state_dict", autospec=True) as assign:
                        with self.assertRaises(TopoWanderStateError):
                            safe_load_topowander_model(
                                state_path,
                                config_path=CONFIG_PATH,
                                expected_config_sha256=self.config_sha256,
                            )
                        assign.assert_not_called()

    def test_source_contract_has_no_data_training_or_unsafe_loader_path(self) -> None:
        source = MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_text = (
            "data/processed",
            "data\\processed",
            "sealed_external_test",
            "official_validation",
            "DataLoader",
            "torch.load(",
            "torch.save(",
        )
        for token in forbidden_text:
            self.assertNotIn(token, source)

        forbidden_function_names = {"train", "fit", "evaluate", "calibrate", "loss"}
        defined_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue(defined_names.isdisjoint(forbidden_function_names))

        internal_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("elderly_monitoring"):
                internal_imports.append(node.module)
            if isinstance(node, ast.Import):
                internal_imports.extend(alias.name for alias in node.names if alias.name.startswith("elderly_monitoring"))
        self.assertEqual(internal_imports, [])

        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("joblib", imported_roots)
        self.assertNotIn("pickle", imported_roots)


if __name__ == "__main__":
    unittest.main()
