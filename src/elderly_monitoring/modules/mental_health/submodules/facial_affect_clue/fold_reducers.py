from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np


FOLD_REDUCER_SCHEMA_VERSION = "dstm_fold_only_reducer_v1"


@dataclass(frozen=True)
class FoldReducerConfig:
    kind: str = "umap"
    n_components: int = 64
    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 20260806
    max_fit_frames: int = 4000

    def __post_init__(self) -> None:
        if self.kind not in {"umap", "pca"}:
            raise ValueError("Reducer kind must be umap or pca")
        if self.n_components < 2:
            raise ValueError("Reducer needs at least two components")
        if self.n_neighbors < 2:
            raise ValueError("UMAP n_neighbors must be at least two")
        if self.max_fit_frames < 4:
            raise ValueError("max_fit_frames must be at least four")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FoldOnlyReducer:
    """A fitted reducer whose fit boundary is explicit and auditable.

    `fit` is deliberately separate from `transform`; no caller in the model
    forward path can accidentally invoke fit_transform.  The stored metadata
    records exactly which subjects were allowed to fit the reducer.
    """

    def __init__(self, config: FoldReducerConfig | None = None) -> None:
        self.config = config or FoldReducerConfig()
        self._reducer: Any | None = None
        self.input_dim: int | None = None
        self.fit_sample_ids: tuple[str, ...] = ()
        self.fit_subject_ids: tuple[str, ...] = ()
        self.fold_id: str | None = None
        self.fit_feature_sha256: str | None = None

    @property
    def fitted(self) -> bool:
        return self._reducer is not None

    @property
    def output_dim(self) -> int:
        return self.config.n_components

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Reducer features must be [samples, dimensions], got {values.shape}")
        if values.shape[0] < 2:
            raise ValueError("Reducer needs at least two feature rows")
        if not np.isfinite(values).all():
            raise ValueError("Reducer features contain non-finite values")
        return values

    def fit(
        self,
        features: np.ndarray,
        *,
        sample_ids: Sequence[str],
        subject_ids: Sequence[str],
        fold_id: str,
        forbidden_sample_ids: Sequence[str] = (),
        forbidden_subject_ids: Sequence[str] = (),
    ) -> "FoldOnlyReducer":
        values = self._validate_features(features)
        if len(sample_ids) != len(subject_ids):
            raise ValueError("sample_ids and subject_ids must have equal length")
        if set(sample_ids) & set(forbidden_sample_ids):
            raise ValueError("Reducer fit sample ids intersect forbidden test ids")
        if set(subject_ids) & set(forbidden_subject_ids):
            raise ValueError("Reducer fit subject ids intersect forbidden test subjects")
        if len(sample_ids) == 0:
            raise ValueError("Reducer fit provenance cannot be empty")

        if values.shape[0] > self.config.max_fit_frames:
            generator = np.random.default_rng(self.config.random_state)
            selected = np.sort(
                generator.choice(
                    values.shape[0], size=self.config.max_fit_frames, replace=False
                )
            )
            values = values[selected]
            # The caller should pass one id per frame.  Sampling provenance is
            # represented by the selected rows rather than an untrue full list.
            sample_ids = [str(sample_ids[index]) for index in selected]
            subject_ids = [str(subject_ids[index]) for index in selected]
        self.input_dim = int(values.shape[1])
        self.fit_sample_ids = tuple(sorted({str(value) for value in sample_ids}))
        self.fit_subject_ids = tuple(sorted({str(value) for value in subject_ids}))
        self.fold_id = str(fold_id)
        self.fit_feature_sha256 = sha256(values.tobytes()).hexdigest()

        if self.config.kind == "umap":
            try:
                import umap
            except ImportError as error:  # pragma: no cover - environment-specific
                raise RuntimeError("umap-learn is required for the frozen UMAP configuration") from error
            neighbors = min(self.config.n_neighbors, max(2, values.shape[0] - 1))
            self._reducer = umap.UMAP(
                n_components=self.config.n_components,
                n_neighbors=neighbors,
                min_dist=self.config.min_dist,
                metric=self.config.metric,
                random_state=self.config.random_state,
                transform_seed=self.config.random_state,
                n_jobs=1,
            )
            self._reducer.fit(values)
        else:
            from sklearn.decomposition import PCA

            components = min(self.config.n_components, values.shape[0], values.shape[1])
            self._reducer = PCA(
                n_components=components,
                svd_solver="full",
                random_state=self.config.random_state,
            )
            self._reducer.fit(values)
            if components != self.config.n_components:
                raise ValueError(
                    f"PCA output dimension {components} cannot satisfy {self.config.n_components}"
                )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._reducer is None or self.input_dim is None:
            raise RuntimeError("Reducer must be fit on the training fold before transform")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Reducer transform expected [samples, {self.input_dim}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Reducer transform features contain non-finite values")
        transformed = np.asarray(self._reducer.transform(values), dtype=np.float32)
        if transformed.shape != (values.shape[0], self.config.n_components):
            raise RuntimeError(f"Unexpected reducer output shape: {transformed.shape}")
        if not np.isfinite(transformed).all():
            raise RuntimeError("Reducer produced non-finite values")
        return transformed

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": FOLD_REDUCER_SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "fold_id": self.fold_id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "fit_sample_count": len(self.fit_sample_ids),
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_subject_ids": list(self.fit_subject_ids),
            "fit_feature_sha256": self.fit_feature_sha256,
            "fit_only": True,
            "transform_only_for_validation_test": True,
            "fit_transform_called_in_forward": False,
        }

    def save(self, path: Path) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted reducer")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(path)
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata = self.metadata()
        metadata["reducer_path"] = path.resolve().as_posix()
        metadata["reducer_sha256"] = sha256_file(path)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata["metadata_path"] = metadata_path.resolve().as_posix()
        metadata["metadata_sha256"] = sha256_file(metadata_path)
        return metadata

    @classmethod
    def load(cls, path: Path) -> "FoldOnlyReducer":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls) or not loaded.fitted:
            raise ValueError("The reducer artifact is not a fitted FoldOnlyReducer")
        return loaded


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

