from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


AU_PRIOR_SCHEMA_VERSION = "microexpression_au_prior_reconstructed_v2"
REGION_NAMES = (
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose",
    "outer_lip",
)
REGION_AUS: Mapping[str, tuple[int, ...]] = {
    "left_eyebrow": (1, 2, 4),
    "right_eyebrow": (1, 2, 4),
    "left_eye": (5, 6, 7),
    "right_eye": (5, 6, 7),
    "nose": (9,),
    "outer_lip": (10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 27, 28),
}


def sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_action_units(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(token) for token in re.findall(r"\d+", value)}))


def active_regions(action_units: Iterable[int]) -> np.ndarray:
    action_unit_set = set(action_units)
    return np.asarray(
        [bool(action_unit_set.intersection(REGION_AUS[name])) for name in REGION_NAMES],
        dtype=np.float64,
    )


def build_au_cooccurrence_prior(
    csv_paths: Sequence[Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a class-balanced symmetric region co-occurrence prior."""
    if not csv_paths:
        raise ValueError("At least one AU CSV is required")
    rows_by_label: dict[int, list[tuple[int, ...]]] = {0: [], 1: [], 2: []}
    source_rows: dict[str, int] = {}
    parsed_rows = 0
    rows_without_mapped_au = 0
    observed_action_units: set[int] = set()

    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"label", "Action Units"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path} does not contain {sorted(required)}")
            for row in reader:
                row_count += 1
                try:
                    label = int(str(row["label"]).strip())
                except ValueError as error:
                    raise ValueError(f"Invalid label in {path} row {row_count + 1}") from error
                if label not in rows_by_label:
                    continue
                action_units = parse_action_units(str(row["Action Units"]))
                observed_action_units.update(action_units)
                active = active_regions(action_units)
                if not active.any():
                    rows_without_mapped_au += 1
                rows_by_label[label].append(action_units)
                parsed_rows += 1
        source_rows[path.resolve().as_posix()] = row_count

    matrices: list[np.ndarray] = []
    class_support: dict[str, int] = {}
    for label, action_unit_rows in rows_by_label.items():
        class_support[str(label)] = len(action_unit_rows)
        if not action_unit_rows:
            continue
        row_sets = [set(row) for row in action_unit_rows]
        cooccurrence = np.zeros((len(REGION_NAMES), len(REGION_NAMES)))
        for left_index, left_name in enumerate(REGION_NAMES):
            for right_index, right_name in enumerate(REGION_NAMES):
                for left_au in REGION_AUS[left_name]:
                    for right_au in REGION_AUS[right_name]:
                        cooccurrence[left_index, right_index] += sum(
                            left_au in row and right_au in row for row in row_sets
                        ) / len(row_sets)
        matrices.append(cooccurrence)
    if not matrices:
        raise ValueError("No supported labels were found in the AU CSV files")
    adjacency = np.mean(matrices, axis=0)
    adjacency = 0.5 * (adjacency + adjacency.T)
    maximum = float(adjacency.max())
    if maximum > 0:
        adjacency /= maximum
    adjacency = adjacency.astype(np.float32)

    audit = {
        "schema_version": AU_PRIOR_SCHEMA_VERSION,
        "source": "reconstructed_from_student_auxiliary_csv",
        "paper_reproduction_claim": False,
        "method": "class_balanced_pairwise_au_probability_sum_equation_3_12",
        "region_names": list(REGION_NAMES),
        "region_action_units": {
            name: list(action_units) for name, action_units in REGION_AUS.items()
        },
        "source_files": [
            {
                "path": path.resolve().as_posix(),
                "sha256": sha256_path(path),
                "row_count": source_rows[path.resolve().as_posix()],
            }
            for path in csv_paths
        ],
        "parsed_rows": parsed_rows,
        "rows_without_mapped_au": rows_without_mapped_au,
        "class_support": class_support,
        "observed_action_units": sorted(observed_action_units),
        "adjacency": adjacency.tolist(),
    }
    audit_bytes = json.dumps(
        audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    audit["content_sha256"] = sha256(audit_bytes).hexdigest()
    return adjacency, audit


def cosine_edge_cost(region_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if region_features.ndim != 3:
        raise ValueError("region_features must have shape [batch, regions, features]")
    normalized = F.normalize(region_features, p=2, dim=-1, eps=eps)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2)).clamp(-1.0, 1.0)
    costs = (1.0 - similarity).clamp_min(0.0)
    diagonal = torch.arange(costs.shape[-1], device=costs.device)
    costs[:, diagonal, diagonal] = 0.0
    return costs


def floyd_warshall(edge_costs: torch.Tensor) -> torch.Tensor:
    if edge_costs.ndim != 3 or edge_costs.shape[-1] != edge_costs.shape[-2]:
        raise ValueError("edge_costs must have shape [batch, nodes, nodes]")
    distances = edge_costs.clone()
    for intermediate in range(distances.shape[-1]):
        through = (
            distances[:, :, intermediate : intermediate + 1]
            + distances[:, intermediate : intermediate + 1, :]
        )
        distances = torch.minimum(distances, through)
    return 0.5 * (distances + distances.transpose(1, 2))


def classical_mds(
    distances: torch.Tensor,
    *,
    dimensions: int = 2,
) -> torch.Tensor:
    if distances.ndim != 3 or distances.shape[-1] != distances.shape[-2]:
        raise ValueError("distances must have shape [batch, nodes, nodes]")
    nodes = distances.shape[-1]
    if not 1 <= dimensions < nodes:
        raise ValueError("MDS dimensions must be between 1 and nodes - 1")
    identity = torch.eye(nodes, dtype=distances.dtype, device=distances.device)
    centering = identity - torch.full_like(identity, 1.0 / nodes)
    gram = -0.5 * torch.matmul(
        torch.matmul(centering, distances.square()), centering
    )
    gram = 0.5 * (gram + gram.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    values = eigenvalues[:, -dimensions:].flip(-1).clamp_min(0.0)
    vectors = eigenvectors[:, :, -dimensions:].flip(-1)
    return vectors * values.sqrt().unsqueeze(1)


def rbf_adjacency(
    embeddings: torch.Tensor,
    *,
    sigma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distances = torch.cdist(embeddings, embeddings, p=2)
    adjacency = torch.exp(-distances.square() / (2.0 * max(sigma, eps) ** 2))
    return 0.5 * (adjacency + adjacency.transpose(1, 2))


@dataclass(frozen=True)
class ManifoldGraphConfig:
    mode: str = "fused"
    mds_dimensions: int = 2
    rbf_sigma: float = 1.0
    fusion_formula: str = "paper_equation_3_17"
    eps: float = 1e-6

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperGraphBuilder(nn.Module):
    def __init__(
        self,
        au_adjacency: torch.Tensor,
        config: ManifoldGraphConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ManifoldGraphConfig()
        if au_adjacency.shape != (6, 6):
            raise ValueError("AU adjacency must have shape [6, 6]")
        if not torch.isfinite(au_adjacency).all():
            raise ValueError("AU adjacency contains non-finite values")
        self.register_buffer("au_adjacency", au_adjacency.float())
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def _manifold(self, region_features: torch.Tensor) -> dict[str, torch.Tensor]:
        edge_costs = cosine_edge_cost(region_features, self.config.eps)
        geodesic = floyd_warshall(edge_costs)
        embedding = classical_mds(
            geodesic, dimensions=self.config.mds_dimensions
        )
        manifold = rbf_adjacency(
            embedding, sigma=self.config.rbf_sigma, eps=self.config.eps
        )
        return {
            "cosine_edge_cost": edge_costs,
            "geodesic_distance": geodesic,
            "mds_embedding": embedding,
            "manifold_adjacency": manifold,
        }

    def forward(
        self,
        region_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mode = self.config.mode
        prior = self.au_adjacency.to(
            device=region_features.device, dtype=region_features.dtype
        ).unsqueeze(0).expand(region_features.shape[0], -1, -1)
        details: dict[str, torch.Tensor] = {"au_adjacency": prior}
        if mode == "au":
            adjacency = prior
        elif mode == "cosine":
            normalized = F.normalize(region_features, p=2, dim=-1, eps=self.config.eps)
            adjacency = torch.bmm(normalized, normalized.transpose(1, 2))
            adjacency = (adjacency + 1.0) * 0.5
            details["cosine_adjacency"] = adjacency
        elif mode in {"manifold", "fused"}:
            details.update(self._manifold(region_features))
            manifold = details["manifold_adjacency"]
            if mode == "manifold":
                adjacency = manifold
            elif self.config.fusion_formula == "paper_equation_3_17":
                maximum = manifold.amax(dim=(-2, -1), keepdim=True).clamp_min(
                    self.config.eps
                )
                geometry = 1.0 - manifold / maximum
                adjacency = self.alpha * prior + (1.0 - self.alpha) * geometry
                details["geometry_fusion_term"] = geometry
            elif self.config.fusion_formula == "direct_rbf_stable":
                adjacency = self.alpha * prior + (1.0 - self.alpha) * manifold
                details["geometry_fusion_term"] = manifold
            else:
                raise ValueError(
                    f"Unsupported fusion formula: {self.config.fusion_formula}"
                )
        else:
            raise ValueError(f"Unsupported graph mode: {mode}")
        if not torch.isfinite(adjacency).all():
            raise RuntimeError("Graph construction produced non-finite values")
        details["adjacency"] = adjacency
        details["alpha"] = self.alpha
        return adjacency, details
