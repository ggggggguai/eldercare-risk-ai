from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .casme2_apexfusion_dataset import load_artifact


FEATURE_GROUPS = ("hoof", "lbp_top", "hog", "landmark_features", "motion_stats", "dense_motion")


@dataclass(frozen=True)
class TraditionalConfig:
    feature_groups: tuple[str, ...] = FEATURE_GROUPS
    pca_components: int = 32
    kernel: str = "linear"
    c: float = 1.0
    class_weight: str | None = "balanced"
    random_state: int = 20260808


def feature_vector(arrays: Mapping[str, np.ndarray], groups: Sequence[str] = FEATURE_GROUPS) -> np.ndarray:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown traditional groups: {sorted(unknown)}")
    parts = []
    for group in groups:
        if group == "dense_motion":
            motion = np.concatenate((arrays["motion_short"], arrays["motion_long"]), axis=0)
            pooled = motion.reshape(8, 4, 14, 4, 14, 4).mean(axis=(3, 5))
            parts.append(pooled.reshape(-1))
        else:
            parts.append(np.asarray(arrays[group], dtype=np.float32).reshape(-1))
    value = np.concatenate(parts).astype(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("Traditional feature vector contains non-finite values")
    return value


def feature_matrix(rows: Sequence[Mapping[str, Any]], groups: Sequence[str] = FEATURE_GROUPS) -> np.ndarray:
    return np.stack([feature_vector(load_artifact(Path(row["artifact_path"]), expected_sha256=row["artifact_sha256"]), groups) for row in rows]).astype(np.float32)


def build_pipeline(config: TraditionalConfig, *, max_samples: int | None = None) -> Pipeline:
    components = config.pca_components
    if max_samples is not None:
        components = max(1, min(components, max_samples - 1))
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", RobustScaler()),
            ("pca", PCA(n_components=components, whiten=True, random_state=config.random_state)),
            ("classifier", SVC(C=config.c, kernel=config.kernel, class_weight=config.class_weight, probability=True, random_state=config.random_state)),
        ]
    )


def save_traditional(path: Path, pipeline: Pipeline, config: TraditionalConfig, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": "casme2_traditional_model_v1", "config": config, "pipeline": pipeline, "metadata": dict(metadata)}, path)


def load_traditional(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if payload.get("schema_version") != "casme2_traditional_model_v1":
        raise ValueError("Unsupported traditional model schema")
    return payload
