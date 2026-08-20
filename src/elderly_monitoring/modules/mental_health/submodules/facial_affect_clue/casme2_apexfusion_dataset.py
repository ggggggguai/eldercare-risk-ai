from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .casme2_apexfusion_manifest import read_jsonl, sha256_file


REQUIRED_ARRAYS = {
    "appearance": (7, 3, 112, 112),
    "motion_short": (6, 4, 56, 56),
    "motion_long": (2, 4, 56, 56),
    "local_roi": (7, 5, 1, 32, 32),
    "landmark_dynamics": (7, 68, 6),
    "traditional": (6496,),
}
ROUTE_ORDER = ("appearance", "motion", "local_roi", "landmark")


class ApexFusionArtifactError(ValueError):
    """Raised when a FLOW-ME2 artifact violates the frozen model schema."""


def load_artifact(path: Path, *, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ApexFusionArtifactError(f"Artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, shape in REQUIRED_ARRAYS.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ApexFusionArtifactError(f"{key} shape mismatch at {path}: {arrays.get(key, np.empty(0)).shape} != {shape}")
    for key, value in arrays.items():
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ApexFusionArtifactError(f"Invalid numeric array {key} at {path}")
    return arrays


class ApexFusionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        artifact_manifest_path: Path,
        *,
        sample_ids: Sequence[str] | None = None,
        enabled_routes: Sequence[str] = ROUTE_ORDER,
        verify_hashes: bool = True,
    ) -> None:
        rows = read_jsonl(artifact_manifest_path)
        selected = set(sample_ids) if sample_ids is not None else None
        self.rows = [row for row in rows if selected is None or row["sample_id"] in selected]
        if selected is not None and {row["sample_id"] for row in self.rows} != selected:
            raise ApexFusionArtifactError("Requested sample ids are missing from artifact manifest")
        unknown = set(enabled_routes) - set(ROUTE_ORDER)
        if unknown:
            raise ApexFusionArtifactError(f"Unknown routes: {sorted(unknown)}")
        if not enabled_routes:
            raise ApexFusionArtifactError("At least one route must be enabled")
        self.route_mask = torch.tensor([route in enabled_routes for route in ROUTE_ORDER], dtype=torch.bool)
        self.verify_hashes = verify_hashes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays = load_artifact(Path(row["artifact_path"]), expected_sha256=row["artifact_sha256"] if self.verify_hashes else None)
        return {
            "appearance": torch.from_numpy(arrays["appearance"]),
            "motion": torch.from_numpy(np.concatenate((arrays["motion_short"], arrays["motion_long"]), axis=0)),
            "local_roi": torch.from_numpy(arrays["local_roi"]),
            "landmark": torch.from_numpy(arrays["landmark_dynamics"]),
            "traditional": torch.from_numpy(arrays["traditional"]),
            "route_mask": self.route_mask.clone(),
            "label": torch.tensor(int(row["three_class_id"]), dtype=torch.long),
            "aux_label": torch.tensor(int(row["aux_emotion_id"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
        }


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
