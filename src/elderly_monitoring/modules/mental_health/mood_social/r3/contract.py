"""OPT-V333-002 scoring-population and deployable-feature contract.

The population decision is deliberately independent of labels, fitted models,
prediction success and metrics.  Only frozen canonical feature masks and the
runtime-observable C/S/P route contract decide eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_FEATURE_SPECS,
    SLEEP_FEATURE_SPECS,
    SOCIAL_CONTEXT_FEATURE_SPECS,
    FeatureSpec,
)


R3_PROTOCOL_VERSION = "mood-social-v3.3.3-r3"
R3_SPLIT_ID = "mood-social-v3.3.3-r2-participant-nested-5x5-seed-20260728-v1"
R3_SPLIT_SHA256 = "4132a303062e6ad6ed09039ce32ed7ec1aac9e356ba5da55b3f91f6386aeb337"
R3_FROZEN_DOCUMENT_SHA256 = (
    "010B4EDD8FE0E9F22824E8111CD863C6A004205F9DF3053B8365D9B6DCB52512"
)

PROCESSED_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r2")
DEFAULT_OUTPUT_RELATIVE = PROCESSED_RELATIVE / "r3"
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-002"
)
DEFAULT_OLD_OOF_RELATIVE = Path(
    "reports/mental_health/mood_social/MH-20260802-012/oof_predictions.parquet"
)


@dataclass(frozen=True)
class CanonicalDataset:
    dataset_id: str
    directory_name: str
    file_name: str


CANONICAL_DATASETS: tuple[CanonicalDataset, ...] = (
    CanonicalDataset("psyche_d", "psyche_d", "canonical_psyche_d.parquet"),
    CanonicalDataset("resilient", "resilient", "canonical_resilient.parquet"),
    CanonicalDataset("nhanes", "nhanes", "canonical_nhanes.parquet"),
    CanonicalDataset(
        "nhanes_ssq_2005_2008",
        "nhanes_ssq_2005_2008",
        "canonical_nhanes_ssq_2005_2008.parquet",
    ),
    CanonicalDataset(
        "shenzhen_elderly", "shenzhen", "canonical_shenzhen.parquet"
    ),
    CanonicalDataset("hefei_elderly", "hefei", "canonical_hefei.parquet"),
)


def _risk_spec_names(specs: Sequence[FeatureSpec]) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in specs
        if spec.name not in {"feature_coverage", "valid_days", "valid_nights"}
    )


ACTIVITY_RISK_FEATURES = tuple(
    f"activity.{name}" for name in _risk_spec_names(ACTIVITY_FEATURE_SPECS)
)
SLEEP_RISK_FEATURES = tuple(
    f"sleep.{name}" for name in _risk_spec_names(SLEEP_FEATURE_SPECS)
)
PROFILE_RISK_FEATURES = tuple(
    f"social_context.{spec.name}" for spec in SOCIAL_CONTEXT_FEATURE_SPECS
)
ACTIVITY_MODEL_FEATURES = tuple(
    f"activity.{spec.name}"
    for spec in ACTIVITY_FEATURE_SPECS
    if spec.name != "feature_coverage"
)
SLEEP_MODEL_FEATURES = tuple(
    f"sleep.{spec.name}"
    for spec in SLEEP_FEATURE_SPECS
    if spec.name != "feature_coverage"
)
PROFILE_MODEL_FEATURES = PROFILE_RISK_FEATURES
JOINT_MODEL_FEATURES = (*ACTIVITY_MODEL_FEATURES, *SLEEP_MODEL_FEATURES)

DEPLOYABLE_FEATURES = frozenset(
    (*ACTIVITY_MODEL_FEATURES, *SLEEP_MODEL_FEATURES, *PROFILE_MODEL_FEATURES)
)

FORBIDDEN_EXACT = frozenset(
    {
        "dataset_id",
        "source_id",
        "participant_id",
        "global_participant_id",
        "subject_id",
        "row_id",
        "r3_row_id",
        "binary_target",
        "feature_mask",
        "mask_pattern",
        "activity.feature_coverage",
        "sleep.feature_coverage",
        "physiology.feature_coverage",
    }
)
FORBIDDEN_PREFIXES = (
    "source__",
    "feature_mask.",
    "physiology.",
)
FORBIDDEN_NAME_TOKENS = (
    "phq",
    "dpq",
    "target",
    "label",
    "severity",
    "filename",
    "file_path",
    "filepath",
    "dataset",
    "participant",
    "subject",
)


def forbidden_feature_reason(name: str) -> str | None:
    """Return the first r3 denylist reason for a model input name."""

    normalized = str(name).strip().lower()
    if normalized in FORBIDDEN_EXACT:
        return "forbidden_exact"
    if normalized.startswith(FORBIDDEN_PREFIXES):
        return "forbidden_prefix"
    if normalized.startswith("i") and normalized[1:].isdigit():
        return "hefei_unmapped_item"
    if normalized in {"a18", "sfi", "sfi_score", "sfi_category"}:
        return "hefei_auxiliary_only"
    if any(token in normalized for token in FORBIDDEN_NAME_TOKENS):
        return "forbidden_semantic_token"
    if normalized.endswith("feature_coverage"):
        return "coverage_only"
    return None


def assert_deployable_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Validate risk inputs and return their stable tuple representation."""

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("a deployable model must have at least one risk feature")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate model features: {duplicates}")
    violations = {
        name: reason
        for name in names
        if (reason := forbidden_feature_reason(name)) is not None
    }
    unknown = sorted(set(names).difference(DEPLOYABLE_FEATURES))
    if violations or unknown:
        raise ValueError(
            "r3 deployable feature contract violation: "
            + json.dumps(
                {"forbidden": violations, "unknown": unknown},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return names


def _repository_root(repository_root: Path | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(__file__).resolve().parents[7]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    payload = b"".join(_canonical_json_bytes(dict(row)) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _load_canonical_frame(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for spec in CANONICAL_DATASETS:
        path = root / PROCESSED_RELATIVE / spec.directory_name / spec.file_name
        frame = pd.read_parquet(path).reset_index(names="canonical_row_index")
        if not frame["dataset_id"].eq(spec.dataset_id).all():
            raise ValueError(f"dataset_id mismatch in {path}")
        frame["r3_row_id"] = (
            frame["dataset_id"].astype(str)
            + "::row="
            + frame["canonical_row_index"].astype(str)
        )
        frames.append(frame)
        hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["r3_row_id"].duplicated().any():
        raise ValueError("r3 row ids are not unique")
    return combined, hashes


def _load_split_assignments(root: Path) -> tuple[pd.DataFrame, str]:
    path = root / PROCESSED_RELATIVE / "splits" / "split_manifest.json"
    digest = _sha256_file(path)
    if digest != R3_SPLIT_SHA256:
        raise ValueError(f"r2 split hash drift: expected {R3_SPLIT_SHA256}, got {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_id") != R3_SPLIT_ID:
        raise ValueError("unexpected r2 split id")
    assignments = pd.DataFrame(payload["participant_assignments"])
    return assignments, digest


def _observed_route(frame: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    columns = [f"feature_mask.{name}" for name in features]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"canonical frame missing route masks: {missing}")
    return frame[columns].fillna(0).astype(int).eq(1).any(axis=1)


def _population_frame(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    camera = _observed_route(frame, ACTIVITY_RISK_FEATURES)
    sleep = _observed_route(frame, SLEEP_RISK_FEATURES)
    profile = _observed_route(frame, PROFILE_RISK_FEATURES)
    route = (
        camera.astype(int).astype(str)
        + sleep.astype(int).astype(str)
        + profile.astype(int).astype(str)
    )
    eligible = camera | sleep | profile
    population = frame[
        [
            "r3_row_id",
            "dataset_id",
            "canonical_row_index",
            "global_participant_id",
            "binary_target",
        ]
    ].copy()
    population["route_c"] = camera.astype(int)
    population["route_s"] = sleep.astype(int)
    population["route_p"] = profile.astype(int)
    population["route_pattern"] = route
    population["r3_eligible"] = eligible.astype(int)
    # The replayed legacy route and the r3 route are both required to score
    # every eligible row.  This rule is fixed before either model is fitted.
    population["common_support"] = eligible.astype(int)
    population["no_evidence"] = (~eligible).astype(int)
    population["eligibility_reason"] = np.where(
        eligible, "production_route_evidence", "no_production_csp_evidence"
    )
    join_columns = [
        "global_participant_id",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    ]
    population = population.merge(
        assignments[join_columns],
        on="global_participant_id",
        how="left",
        validate="many_to_one",
    )
    if population["outer_fold"].isna().any():
        raise ValueError("population contains participants absent from frozen split")
    population["outer_fold"] = population["outer_fold"].astype(int)
    return population.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)


def _group_summary(frame: pd.DataFrame, group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(group, sort=True, dropna=False):
        rows.append(
            {
                group: str(key),
                "row_count": int(len(part)),
                "participant_count": int(part["global_participant_id"].nunique()),
                "positive_row_count": int(part["binary_target"].sum()),
                "prevalence": float(part["binary_target"].mean()),
            }
        )
    return rows


def _population_summary(population: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "selection_independence": (
            "feature masks and C/S/P route contract only; labels, predictions and metrics "
            "are not eligibility inputs"
        ),
        "totals": {
            "row_count": int(len(population)),
            "participant_count": int(population["global_participant_id"].nunique()),
            "positive_row_count": int(population["binary_target"].sum()),
        },
    }
    for name in ("common_support", "r3_eligible", "no_evidence"):
        selected = population[population[name].eq(1)]
        result[name] = {
            "row_count": int(len(selected)),
            "participant_count": int(selected["global_participant_id"].nunique()),
            "positive_row_count": int(selected["binary_target"].sum()),
            "prevalence": float(selected["binary_target"].mean()),
            "by_source": _group_summary(selected, "dataset_id"),
            "by_route": _group_summary(selected, "route_pattern"),
        }
    return result


def _spec_map() -> dict[str, FeatureSpec]:
    result: dict[str, FeatureSpec] = {}
    for prefix, specs in (
        ("activity", ACTIVITY_FEATURE_SPECS),
        ("sleep", SLEEP_FEATURE_SPECS),
        ("social_context", SOCIAL_CONTEXT_FEATURE_SPECS),
    ):
        for spec in specs:
            name = f"{prefix}.{spec.name}"
            if name in DEPLOYABLE_FEATURES:
                result[name] = spec
    return result


def _feature_audit(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "risk_feature_count": len(DEPLOYABLE_FEATURES),
        "features": {},
    }
    passed = True
    for feature, spec in sorted(_spec_map().items()):
        mask_name = f"feature_mask.{feature}"
        feature_rows: list[dict[str, Any]] = []
        for dataset_id, part in frame.groupby("dataset_id", sort=True):
            mask = pd.to_numeric(part[mask_name], errors="coerce").fillna(0).astype(int)
            values = part.loc[mask.eq(1), feature]
            masked_null = int(values.isna().sum())
            observed = values.dropna()
            out_of_range = 0
            invalid_category = 0
            quantiles: dict[str, float] | None = None
            if spec.categories:
                invalid_category = int((~observed.astype(str).isin(spec.categories)).sum())
            else:
                numeric = pd.to_numeric(observed, errors="coerce")
                out_of_range += int(numeric.isna().sum())
                numeric = numeric.dropna()
                if spec.minimum is not None:
                    if spec.exclusive_minimum:
                        out_of_range += int((numeric <= spec.minimum).sum())
                    else:
                        out_of_range += int((numeric < spec.minimum).sum())
                if spec.maximum is not None:
                    if spec.exclusive_maximum:
                        out_of_range += int((numeric >= spec.maximum).sum())
                    else:
                        out_of_range += int((numeric > spec.maximum).sum())
                if len(numeric):
                    q = numeric.quantile([0.01, 0.5, 0.99])
                    quantiles = {
                        "p01": float(q.loc[0.01]),
                        "p50": float(q.loc[0.5]),
                        "p99": float(q.loc[0.99]),
                    }
            row_pass = masked_null == 0 and out_of_range == 0 and invalid_category == 0
            passed = passed and row_pass
            feature_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "observed_count": int(mask.sum()),
                    "masked_null_count": masked_null,
                    "out_of_range_count": out_of_range,
                    "invalid_category_count": invalid_category,
                    "quantiles": quantiles,
                    "pass": row_pass,
                }
            )
        report["features"][feature] = {
            "schema": spec.to_dict(),
            "runtime_generator": "map_mood_social_features",
            "source_distribution_audit": feature_rows,
        }
    report["pass"] = passed
    return report


def _old_oof_alignment(root: Path, population: pd.DataFrame) -> dict[str, Any]:
    path = root / DEFAULT_OLD_OOF_RELATIVE
    old = pd.read_parquet(path)
    old_keys = set(
        zip(
            old["dataset_id"].astype(str),
            old["canonical_row_index"].astype(int),
            strict=True,
        )
    )
    common = population[population["common_support"].eq(1)]
    common_keys = set(
        zip(
            common["dataset_id"].astype(str),
            common["canonical_row_index"].astype(int),
            strict=True,
        )
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "historical_row_count": int(len(old)),
        "common_support_row_count": int(len(common)),
        "common_support_missing_from_historical_oof": int(len(common_keys - old_keys)),
        "historical_no_evidence_row_count": int(
            len(
                population[
                    population["no_evidence"].eq(1)
                    & population.apply(
                        lambda row: (
                            str(row["dataset_id"]), int(row["canonical_row_index"])
                        )
                        in old_keys,
                        axis=1,
                    )
                ]
            )
        ),
        "role": "alignment evidence only; probabilities are not the r3 replay baseline",
    }


def build_population_artifacts(
    *,
    repository_root: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze r3 populations and parity evidence before model search."""

    root = _repository_root(repository_root)
    output = Path(output_directory) if output_directory else root / DEFAULT_OUTPUT_RELATIVE
    report = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    frame, canonical_hashes = _load_canonical_frame(root)
    assignments, split_hash = _load_split_assignments(root)
    population = _population_frame(frame, assignments)
    summary = _population_summary(population)
    parity = _feature_audit(frame)
    alignment = _old_oof_alignment(root, population)

    expected = {
        "total": 22_701,
        "common_support": 22_070,
        "r3_eligible": 22_070,
        "no_evidence": 631,
    }
    actual = {
        "total": int(len(population)),
        "common_support": int(population["common_support"].sum()),
        "r3_eligible": int(population["r3_eligible"].sum()),
        "no_evidence": int(population["no_evidence"].sum()),
    }
    if actual != expected:
        raise ValueError(f"r3 population drift: expected {expected}, got {actual}")
    if not parity["pass"]:
        raise ValueError("r3 schema/distribution range audit failed")
    if alignment["common_support_missing_from_historical_oof"] != 0:
        raise ValueError("common-support is not aligned with historical row keys")

    population_path = output / "population_assignments.parquet"
    if population_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {population_path}")
    output.mkdir(parents=True, exist_ok=True)
    population.to_parquet(population_path, index=False)

    key_paths: dict[str, Path] = {}
    for name in ("common_support", "r3_eligible", "no_evidence"):
        key_path = output / f"{name}_row_keys.jsonl"
        key_rows = population.loc[
            population[name].eq(1),
            ["r3_row_id", "dataset_id", "canonical_row_index"],
        ].to_dict(orient="records")
        _write_jsonl(key_path, key_rows, overwrite=overwrite)
        key_paths[name] = key_path

    feature_contract = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "activity_model_features": list(
            assert_deployable_feature_names(ACTIVITY_MODEL_FEATURES)
        ),
        "sleep_model_features": list(assert_deployable_feature_names(SLEEP_MODEL_FEATURES)),
        "joint_model_features": list(assert_deployable_feature_names(JOINT_MODEL_FEATURES)),
        "profile_model_features": list(
            assert_deployable_feature_names(PROFILE_MODEL_FEATURES)
        ),
        "forbidden_exact": sorted(FORBIDDEN_EXACT),
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "route_only_inputs": [
            "feature_mask.*",
            "activity.feature_coverage",
            "sleep.feature_coverage",
            "physiology.feature_coverage",
        ],
    }
    summary.update(
        {
            "frozen_document_sha256": R3_FROZEN_DOCUMENT_SHA256,
            "split_id": R3_SPLIT_ID,
            "split_sha256": split_hash,
            "canonical_sha256": canonical_hashes,
            "historical_alignment": alignment,
        }
    )
    summary_path = output / "population_summary.json"
    contract_path = output / "feature_contract.json"
    parity_path = report / "schema_distribution_parity.json"
    _write_json(summary_path, summary, overwrite=overwrite)
    _write_json(contract_path, feature_contract, overwrite=overwrite)
    _write_json(parity_path, parity, overwrite=overwrite)

    artifact_paths = {
        "population_assignments": population_path,
        "population_summary": summary_path,
        "feature_contract": contract_path,
        "schema_distribution_parity": parity_path,
        **{f"{name}_row_keys": path for name, path in key_paths.items()},
    }
    artifact_hashes = {
        name: {
            "path": _display_path(path, root),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "task_id": "OPT-V333-002",
        "status": "pass",
        "population_counts": actual,
        "split_sha256": split_hash,
        "artifacts": artifact_hashes,
    }
    manifest_path = output / "artifact_manifest.json"
    _write_json(manifest_path, manifest, overwrite=overwrite)
    manifest["manifest_path"] = _display_path(manifest_path, root)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


__all__ = [
    "ACTIVITY_MODEL_FEATURES",
    "CANONICAL_DATASETS",
    "DEPLOYABLE_FEATURES",
    "JOINT_MODEL_FEATURES",
    "PROFILE_MODEL_FEATURES",
    "R3_PROTOCOL_VERSION",
    "SLEEP_MODEL_FEATURES",
    "assert_deployable_feature_names",
    "build_population_artifacts",
    "forbidden_feature_reason",
]
