"""Finite R6 candidate registry and model implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
    fit_r5_model,
    predict_r5_model,
)


Track = Literal["psyche_d_single_source_research", "multisource_deployable", "joint_route_recovery"]
Family = Literal["elasticnet_spline", "lightgbm", "catboost", "hist_gradient_boosting", "extra_trees", "causal_deepsets", "shrinkage_route_experts"]


@dataclass(frozen=True)
class R6CandidateSpec:
    candidate_id: str
    track: Track
    family: Family
    params: dict[str, Any]
    seed: int


def r6_candidate_registry(seed: int) -> tuple[R6CandidateSpec, ...]:
    """Return exactly 46 first-round candidates within every frozen family cap."""

    specs: list[R6CandidateSpec] = []

    def add(track: Track, family: Family, name: str, **params: Any) -> None:
        specs.append(R6CandidateSpec(name, track, family, params, int(seed)))

    # PSYCHE-D: 22 candidates.
    for leaves, minimum, regularization in (
        (5, 60, 8.0), (7, 40, 8.0), (7, 80, 12.0), (9, 60, 12.0),
        (11, 80, 16.0), (15, 60, 12.0), (15, 100, 18.0), (23, 100, 20.0),
    ):
        add("psyche_d_single_source_research", "lightgbm", f"psy_lgb_l{leaves}_m{minimum}_r{int(regularization)}", n_estimators=450, learning_rate=0.02, num_leaves=leaves, min_child_samples=minimum, reg_lambda=regularization, reg_alpha=regularization / 10, subsample=0.85, colsample_bytree=0.80)
    for depth, lr in ((3, 0.025), (4, 0.025), (5, 0.02), (6, 0.02)):
        add("psyche_d_single_source_research", "catboost", f"psy_cat_d{depth}_lr{str(lr).replace('.', '')}", iterations=400, depth=depth, learning_rate=lr, l2_leaf_reg=12.0, random_strength=0.5, rsm=0.85)
    for leaf, features in ((3, 0.5), (6, 0.7)):
        add("psyche_d_single_source_research", "extra_trees", f"psy_extra_leaf{leaf}_f{int(features*10)}", n_estimators=500, min_samples_leaf=leaf, max_features=features, class_weight="balanced_subsample")
    for leaves, l2 in ((15, 5.0), (31, 10.0)):
        add("psyche_d_single_source_research", "hist_gradient_boosting", f"psy_hist_l{leaves}_r{int(l2)}", max_leaf_nodes=leaves, l2_regularization=l2, learning_rate=0.05, max_iter=250, min_samples_leaf=40)
    for width, dropout, epochs in ((8, 0.0, 35), (12, 0.0, 40), (16, 0.1, 40), (24, 0.1, 45), (32, 0.2, 45), (48, 0.2, 50)):
        add("psyche_d_single_source_research", "causal_deepsets", f"psy_deepsets_w{width}_d{int(dropout*10)}", width=width, dropout=dropout, epochs=epochs, learning_rate=0.003, weight_decay=0.001)

    # Multi-source deployable: 20 candidates.
    for c, ratio, spline in ((0.03, 0.1, False), (0.05, 0.2, False), (0.1, 0.2, False), (0.3, 0.3, False), (0.05, 0.1, True), (0.1, 0.2, True), (0.3, 0.3, True), (0.6, 0.5, True)):
        add("multisource_deployable", "elasticnet_spline", f"multi_en_c{str(c).replace('.', '')}_l{int(ratio*10)}{'s' if spline else ''}", C=c, l1_ratio=ratio, spline=spline)
    for leaves, minimum in ((7, 40), (15, 80)):
        add("multisource_deployable", "lightgbm", f"multi_lgb_l{leaves}_m{minimum}", n_estimators=350, learning_rate=0.025, num_leaves=leaves, min_child_samples=minimum, reg_lambda=10.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.9)
    for depth in (3, 5):
        add("multisource_deployable", "catboost", f"multi_cat_d{depth}", iterations=300, depth=depth, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=0.5, rsm=0.9)
    for leaves, l2 in ((15, 5.0), (31, 10.0)):
        add("multisource_deployable", "hist_gradient_boosting", f"multi_hist_l{leaves}", max_leaf_nodes=leaves, l2_regularization=l2, learning_rate=0.05, max_iter=220, min_samples_leaf=40)
    for leaf, features in ((3, 0.7), (8, 1.0)):
        add("multisource_deployable", "extra_trees", f"multi_extra_leaf{leaf}", n_estimators=400, min_samples_leaf=leaf, max_features=features, class_weight="balanced_subsample")
    for alpha in (0.70, 0.80, 0.90, 0.95):
        add("multisource_deployable", "shrinkage_route_experts", f"multi_shrink_a{int(alpha*100)}", alpha=alpha, C=0.1, l1_ratio=0.2, spline=True)

    # Joint 101/111 recovery: 4 candidates, reusing remaining family capacity.
    for leaves, minimum in ((7, 40), (15, 80)):
        add("joint_route_recovery", "lightgbm", f"joint_lgb_l{leaves}_m{minimum}", n_estimators=400, learning_rate=0.025, num_leaves=leaves, min_child_samples=minimum, reg_lambda=10.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.9)
    for depth in (3, 5):
        add("joint_route_recovery", "catboost", f"joint_cat_d{depth}", iterations=350, depth=depth, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=0.5, rsm=0.9)

    observed: dict[str, int] = {}
    for spec in specs:
        observed[spec.family] = observed.get(spec.family, 0) + 1
    maximum = {"elasticnet_spline": 8, "lightgbm": 12, "catboost": 8, "hist_gradient_boosting": 4, "extra_trees": 4, "causal_deepsets": 6, "shrinkage_route_experts": 8}
    if len(specs) != 46 or any(observed.get(family, 0) > cap for family, cap in maximum.items()):
        raise AssertionError(f"r6 candidate budget drift: total={len(specs)}, observed={observed}")
    return tuple(specs)


def _as_r5_spec(spec: R6CandidateSpec) -> R5CandidateSpec:
    family = spec.family
    if family == "shrinkage_route_experts":
        family = "elasticnet_spline"
    if family not in {"elasticnet_spline", "lightgbm", "catboost", "extra_trees"}:
        raise ValueError(f"{spec.family} is not an r5-compatible family")
    params = dict(spec.params)
    params.pop("alpha", None)
    return R5CandidateSpec(spec.candidate_id, family, params, spec.seed)  # type: ignore[arg-type]


def fit_regular_model(frame: pd.DataFrame, features: Sequence[str], spec: R6CandidateSpec, *, sample_weight: np.ndarray) -> Any:
    if spec.family != "hist_gradient_boosting":
        return fit_r5_model(frame, features, _as_r5_spec(spec), sample_weight=sample_weight)
    preprocessor = ColumnTransformer([("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(features))], remainder="drop")
    model = HistGradientBoostingClassifier(random_state=spec.seed, **spec.params)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(frame[list(features)], frame["binary_target"].astype(int), model__sample_weight=np.asarray(sample_weight, float))
    return pipeline


def predict_regular_model(model: Any, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return np.clip(np.asarray(model.predict_proba(frame[list(features)])[:, 1], float), 1.0e-6, 1.0 - 1.0e-6)


class CausalDeepSetBinary:
    """Small local-only feature-set encoder with fold-local preprocessing."""

    def __init__(self, *, width: int, dropout: float, epochs: int, learning_rate: float, weight_decay: float, seed: int) -> None:
        self.params = {"width": width, "dropout": dropout, "epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay, "seed": seed}
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.model: Any = None

    def fit(self, frame: pd.DataFrame, features: Sequence[str], *, target: str, sample_weight: np.ndarray) -> "CausalDeepSetBinary":
        import torch
        from torch import nn

        torch.manual_seed(int(self.params["seed"]))
        raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.median = np.nanmedian(raw, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        filled = np.where(np.isfinite(raw), raw, self.median)
        self.scale = np.nanstd(filled, axis=0)
        self.scale[~np.isfinite(self.scale) | (self.scale < 1.0e-6)] = 1.0
        x = np.clip((filled - self.median) / self.scale, -8, 8).astype(np.float32)
        mask = np.isfinite(raw).astype(np.float32)
        y = frame[target].to_numpy(np.float32)
        w = np.asarray(sample_weight, np.float32)
        width = int(self.params["width"])

        class Net(nn.Module):
            def __init__(self, feature_count: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(feature_count, width)
                self.value = nn.Linear(1, width, bias=False)
                self.rho = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(float(self_dropout)), nn.Linear(width, 1))

            def forward(self, values: Any, observed: Any) -> Any:
                indices = torch.arange(values.shape[1], device=values.device)
                token = self.embedding(indices)[None, :, :] + self.value(values[:, :, None])
                token = torch.relu(token) * observed[:, :, None]
                pooled = token.sum(dim=1) / observed.sum(dim=1, keepdim=True).clamp_min(1.0)
                return self.rho(pooled).squeeze(1)

        self_dropout = float(self.params["dropout"])
        model = Net(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.params["learning_rate"]), weight_decay=float(self.params["weight_decay"]))
        tensor_x = torch.from_numpy(x); tensor_mask = torch.from_numpy(mask); tensor_y = torch.from_numpy(y); tensor_w = torch.from_numpy(w)
        generator = torch.Generator().manual_seed(int(self.params["seed"]))
        batch = min(256, len(frame))
        for _ in range(int(self.params["epochs"])):
            for indices in torch.randperm(len(frame), generator=generator).split(batch):
                optimizer.zero_grad()
                logits = model(tensor_x[indices], tensor_mask[indices])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tensor_y[indices], weight=tensor_w[indices])
                loss.backward(); optimizer.step()
        model.eval(); self.model = model
        return self

    def predict_proba(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        import torch

        if self.model is None or self.median is None or self.scale is None:
            raise ValueError("causal DeepSets model is not fitted")
        raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        mask = np.isfinite(raw).astype(np.float32)
        x = np.clip((np.where(np.isfinite(raw), raw, self.median) - self.median) / self.scale, -8, 8).astype(np.float32)
        with torch.no_grad():
            probability = torch.sigmoid(self.model(torch.from_numpy(x), torch.from_numpy(mask))).numpy()
        return np.clip(np.asarray(probability, float), 1.0e-6, 1.0 - 1.0e-6)


def fit_deepset_model(frame: pd.DataFrame, features: Sequence[str], spec: R6CandidateSpec, *, target: str, sample_weight: np.ndarray) -> CausalDeepSetBinary:
    return CausalDeepSetBinary(seed=spec.seed, **spec.params).fit(frame, features, target=target, sample_weight=sample_weight)


def serialize_spec(spec: R6CandidateSpec) -> dict[str, Any]:
    return asdict(spec)


__all__ = ["CausalDeepSetBinary", "R6CandidateSpec", "fit_deepset_model", "fit_regular_model", "predict_regular_model", "r6_candidate_registry", "serialize_spec"]
