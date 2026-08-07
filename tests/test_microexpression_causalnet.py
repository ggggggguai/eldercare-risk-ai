from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue import causalnet as causalnet_module
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue import causalnet_preprocess as preprocess_module
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (
    ARCHITECTURE_CORRECTION,
    CausalNet,
    CausalNetConfig,
    load_causalnet_checkpoint,
    save_causalnet_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_dataset import (
    CausalNetArtifactDataset,
    EXPECTED_ROUTE_ORDER,
    collate_causalnet,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_preprocess import (
    CausalNetPreprocessConfig,
    build_four_route_input,
    direction_map,
    directory_fingerprint,
    spatial_projection,
    write_artifact,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def _record(tmp_path: Path) -> dict[str, object]:
    frame_dir = tmp_path / "frames"
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(base, (80, 80), (176, 176), (180, 180, 180), -1)
    for index, shift in enumerate((0, 3, -2)):
        frame = np.roll(base, shift, axis=1)
        _write_image(frame_dir / f"frame{index}.bmp", frame)
    return {
        "sample_id": "smic_hs__s01__fixture",
        "source_dataset": "smic_hs",
        "sample_role": "classification",
        "subject_id": "s01",
        "sequence_id": "fixture",
        "frame_dir": frame_dir.as_posix(),
        "label": 0,
        "label_name": "negative",
        "onset_index": 0,
        "apex_index": 1,
        "offset_index": 2,
        "onset_frame": "frame0.bmp",
        "apex_frame": "frame1.bmp",
        "offset_frame": "frame2.bmp",
        "frame_annotation_source": "estimated_dc_rois",
        "apex_method": "dc_rois",
        "alignment_transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }


def _small_config() -> CausalNetConfig:
    return CausalNetConfig(
        dim=32,
        heads=2,
        block_repeats=(1, 1, 1),
        cross_heads=4,
        cross_depth=1,
        mlp_mult=2,
    )


def _artifact(tmp_path: Path, *, metadata_override: dict[str, object] | None = None) -> dict[str, object]:
    inputs = np.random.default_rng(8).normal(size=(4, 3, 28, 28)).astype(np.float32)
    metadata: dict[str, object] = {
        "sample_id": "smic_hs__s01__fixture",
        "subject_id": "s01",
        "label": 1,
        "route_order": list(EXPECTED_ROUTE_ORDER),
        "input_shape": [4, 3, 28, 28],
        "onset_index": 0,
        "apex_index": 1,
        "offset_index": 2,
        "spatial_variant": "source_compatible_roi",
    }
    metadata.update(metadata_override or {})
    return write_artifact(inputs, metadata, tmp_path / "artifact.npz")


def test_four_route_preprocess_shape_label_and_frozen_metadata(tmp_path: Path) -> None:
    inputs, metadata = build_four_route_input(_record(tmp_path))
    assert inputs.shape == (4, 3, 28, 28)
    assert inputs.dtype == np.float32
    assert np.isfinite(inputs).all()
    assert metadata["label"] == 0
    assert metadata["route_order"] == list(EXPECTED_ROUTE_ORDER)
    assert metadata["frame_annotation_source"] == "estimated_dc_rois"
    assert metadata["alignment_transform"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_preprocess_rejects_non_finite_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def invalid_flow(*args, **kwargs):
        return np.full((32, 32, 2), np.nan, dtype=np.float32)

    monkeypatch.setattr(preprocess_module, "estimate_flow", invalid_flow)
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_four_route_input(_record(tmp_path))


def test_preprocess_rejects_invalid_keyframe_order(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record["apex_index"] = 2
    record["offset_index"] = 1
    with pytest.raises(ValueError, match="key-frame order"):
        build_four_route_input(record)


def test_forward_and_reverse_direction_maps_have_distinct_hue() -> None:
    forward = np.zeros((16, 16, 2), dtype=np.float32)
    forward[..., 0] = 1.0
    reverse = -forward
    forward_rgb = direction_map(forward, 1.0)
    reverse_rgb = direction_map(reverse, 1.0)
    assert forward_rgb.shape == reverse_rgb.shape == (16, 16, 3)
    assert not np.array_equal(forward_rgb, reverse_rgb)
    assert float(np.abs(forward_rgb - reverse_rgb).mean()) > 0.2


def test_alignment_transform_is_reused_for_all_three_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = np.asarray(((0.9, 0.1, 4.0), (-0.1, 0.9, 5.0)), dtype=np.float32)
    calls: list[np.ndarray] = []
    monkeypatch.setattr(preprocess_module, "list_sequence_frames", lambda path: [Path("a"), Path("b"), Path("c")])

    def aligned(path: Path, transform: np.ndarray) -> np.ndarray:
        calls.append(transform.copy())
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[:, 80:180] = len(calls) * 20
        return image

    monkeypatch.setattr(preprocess_module, "_aligned_frame", aligned)
    record = {
        "sample_id": "smic_hs__s01__fixture",
        "source_dataset": "smic_hs",
        "sample_role": "classification",
        "subject_id": "s01",
        "sequence_id": "fixture",
        "frame_dir": "unused",
        "label": 0,
        "label_name": "negative",
        "onset_index": 0,
        "apex_index": 1,
        "offset_index": 2,
        "frame_annotation_source": "estimated_dc_rois",
        "apex_method": "dc_rois",
        "alignment_transform": expected.tolist(),
    }
    build_four_route_input(record)
    assert len(calls) == 3
    assert all(np.array_equal(call, expected) for call in calls)


def test_full_face_is_an_explicit_spatial_candidate() -> None:
    image = np.random.default_rng(4).normal(size=(32, 32, 3)).astype(np.float32)
    projected = spatial_projection(image, CausalNetPreprocessConfig(spatial_variant="full_face"))
    assert projected.shape == (28, 28, 3)
    assert np.isfinite(projected).all()


def test_dataset_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    record = _artifact(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        CausalNetArtifactDataset([record, dict(record)])


def test_directory_fingerprint_proves_unchanged_history(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    (history / "artifact.bin").write_bytes(b"frozen")
    before = directory_fingerprint(history)
    (tmp_path / "new-output.bin").write_bytes(b"new")
    after = directory_fingerprint(history)
    assert before == after


def test_small_model_forward_backward_and_finite_gradients() -> None:
    model = CausalNet(_small_config())
    inputs = torch.randn(2, 4, 3, 28, 28)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor((0, 2)))
    loss.backward()
    assert logits.shape == (2, 3)
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_real_npz_dataset_batch_forward(tmp_path: Path) -> None:
    record = _artifact(tmp_path)
    dataset = CausalNetArtifactDataset([record])
    batch = collate_causalnet([dataset[0]])
    logits = CausalNet(_small_config())(batch["inputs"])
    assert logits.shape == (1, 3)
    assert batch["labels"].tolist() == [1]


def test_model_rejects_shape_and_non_finite_inputs() -> None:
    model = CausalNet(_small_config())
    with pytest.raises(ValueError, match="expects"):
        model(torch.randn(1, 3, 28, 28))
    invalid = torch.randn(1, 4, 3, 28, 28)
    invalid[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(invalid)


def test_dataset_rejects_wrong_route_metadata(tmp_path: Path) -> None:
    record = _artifact(tmp_path, metadata_override={"route_order": list(reversed(EXPECTED_ROUTE_ORDER))})
    with pytest.raises(ValueError, match="route order"):
        CausalNetArtifactDataset([record])[0]


def test_checkpoint_round_trip_and_strict_metadata(tmp_path: Path) -> None:
    model = CausalNet(_small_config()).eval()
    inputs = torch.randn(1, 4, 3, 28, 28)
    path = tmp_path / "causalnet.pt"
    metadata = save_causalnet_checkpoint(
        path,
        model,
        training_config={"smoke_only": True},
        source_manifest_hash="a" * 64,
        preprocessing_config_hash="b" * 64,
        source_code_hash="c" * 64,
    )
    restored, payload = load_causalnet_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(model(inputs), restored.eval()(inputs))
    assert metadata["architecture_correction"] == ARCHITECTURE_CORRECTION
    assert payload["upstream_bit_exact"] is False
    assert payload["paper_reproduction_claim"] is False

    payload["architecture_correction"] = "invalid"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture_correction"):
        load_causalnet_checkpoint(path)


def test_runtime_model_does_not_import_third_party_or_accept_test_fold() -> None:
    source = inspect.getsource(causalnet_module)
    assert "third_party" not in source
    assert "test_fold" not in source
    assert tuple(inspect.signature(CausalNet.forward).parameters) == ("self", "inputs")
