"""Frozen protocol, label and feature contract for OPT-V333-R4-000."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    DEPLOYABLE_FEATURES,
    R3_SPLIT_ID,
    R3_SPLIT_SHA256,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    assert_participant_isolation,
    eligible_rows,
    load_r3_training_frame,
)


R4_PROTOCOL_VERSION = "mood-social-v3.3.3-r4"
R4_FROZEN_DOCUMENT_SHA256 = (
    "9D24C2B8B9CBBBF263ABFD7C1DDF74F987E80C86C09A6C85BCDB91152E410859"
)
R4_REPEAT_SEEDS = (20260812, 20260813, 20260814)
R4_LABEL_TASKS = {
    "phq_ge5_current": "phq9_ge5_r3_target",
    "phq_ge10_current": "phq9_ge10_r3_target",
}

DEFAULT_OUTPUT_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r4/protocol"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-000"
)
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r4.yaml")

R4_ALLOWED_FEATURES = frozenset(
    name
    for name in DEPLOYABLE_FEATURES
    if not name.endswith(("feature_coverage", "valid_days", "valid_nights"))
)
PSYCHE_RESEARCH_SENSOR_FEATURES = (
    "source__steps_awake_mean",
    "source__steps_mvpa_iqr",
    "source__steps_lpa_iqr",
    "source__steps_awake_sum_iqr",
    "source__steps__active_day_count_",
    "source__steps__sedentary_day_count_",
    "source__steps_mvpa_sum_recent",
    "source__steps_lpa_sum_recent",
    "source__steps_rolling_6_median_recent",
    "source__steps_rolling_6_max_recent",
    "source__sleep_asleep_weekday_mean",
    "source__sleep_asleep_weekend_mean",
    "source__sleep_in_bed_weekday_mean",
    "source__sleep_in_bed_weekend_mean",
    "source__sleep_ratio_asleep_in_bed_weekday_mean",
    "source__sleep_ratio_asleep_in_bed_weekend_mean",
    "source__sleep_in_bed_iqr",
    "source__sleep_asleep_iqr",
    "source__sleep_ratio_asleep_in_bed_iqr",
    "source__sleep_main_start_hour_adj_median",
    "source__sleep_main_start_hour_adj_iqr",
    "source__sleep_main_start_hour_adj_range",
    "source__sleep__hypersomnia_count_",
    "source__sleep__hyposomnia_count_",
    "source__sleep_asleep_mean_recent",
    "source__sleep_in_bed_mean_recent",
    "source__sleep_ratio_asleep_in_bed_mean_recent",
)
R4_FORBIDDEN_EXACT = frozenset(
    {
        "dataset_id",
        "source_id",
        "participant_id",
        "global_participant_id",
        "subject_id",
        "row_id",
        "r3_row_id",
        "r4_row_id",
        "binary_target",
        "route_pattern",
        "route_c",
        "route_s",
        "route_p",
        "feature_mask",
        "mask_pattern",
        "history_attention_index",
        "attention_index",
    }
)
R4_FORBIDDEN_PREFIXES = (
    "source__",
    "feature_mask.",
    "history_phq",
    "current_phq",
    "physiology.",
)
R4_FORBIDDEN_TOKENS = (
    "phq",
    "dpq",
    "cesd",
    "gds",
    "bdi",
    "madrs",
    "gad",
    "isi",
    "uls",
    "sfi",
    "attention_index",
    "screening",
    "depressive",
    "depression",
    "target",
    "label",
    "severity",
    "dataset",
    "source_id",
    "participant",
    "subject",
    "filename",
    "filepath",
)


def _repository_root(repository_root: Path | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(__file__).resolve().parents[7]


def _project_root(repository_root: Path) -> Path:
    candidate = repository_root.parents[1]
    if not (candidate / "项目文档").is_dir():
        raise FileNotFoundError("cannot locate root project documentation directory")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise FileExistsError(f"refusing to overwrite r4 protocol artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def forbidden_r4_feature_reason(name: str) -> str | None:
    """Return why a field cannot enter either r4 current-state risk head."""

    normalized = str(name).strip().lower()
    if normalized in R4_FORBIDDEN_EXACT:
        return "forbidden_exact"
    if normalized.startswith(R4_FORBIDDEN_PREFIXES):
        return "forbidden_prefix"
    if normalized.endswith("feature_coverage") or normalized.endswith("valid_days"):
        return "routing_or_coverage_only"
    if normalized.endswith("valid_nights"):
        return "routing_or_coverage_only"
    if normalized.startswith("i") and normalized[1:].isdigit():
        return "questionnaire_item"
    if normalized in {"a18", "sfi", "sfi_score", "sfi_category"}:
        return "hefei_auxiliary_only"
    if any(token in normalized for token in R4_FORBIDDEN_TOKENS):
        return "forbidden_semantic_token"
    return None


def assert_r4_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Enforce the frozen runtime-equivalent allowlist and semantic denylist."""

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("an r4 risk model must have at least one input feature")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    forbidden = {
        name: reason
        for name in names
        if (reason := forbidden_r4_feature_reason(name)) is not None
    }
    unknown = sorted(set(names).difference(R4_ALLOWED_FEATURES))
    if duplicates or forbidden or unknown:
        raise ValueError(
            "r4 feature contract violation: "
            + json.dumps(
                {
                    "duplicates": duplicates,
                    "forbidden": forbidden,
                    "not_runtime_approved": unknown,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return names


def assert_psyche_research_feature_names(
    feature_names: Iterable[str],
) -> tuple[str, ...]:
    """Allow audited local sensor summaries only on the research-only track."""

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("the PSYCHE research model needs at least one feature")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    approved = set(R4_ALLOWED_FEATURES) | set(PSYCHE_RESEARCH_SENSOR_FEATURES)
    unknown = sorted(set(names).difference(approved))
    # source__ is allowed here only through the exact audited list; retain all
    # other semantic leakage checks explicitly.
    forbidden = {
        name: reason
        for name in names
        if name not in PSYCHE_RESEARCH_SENSOR_FEATURES
        and (reason := forbidden_r4_feature_reason(name)) is not None
    }
    if duplicates or forbidden or unknown:
        raise ValueError(
            "r4 research feature contract violation: "
            + json.dumps(
                {"duplicates": duplicates, "forbidden": forbidden, "unknown": unknown},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return names


def _label_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    target = frame[column].astype(int)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "prevalence": float(target.mean()),
    }


def build_r4_protocol_artifacts(
    *,
    repository_root: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze r4's reused-development boundary without touching r3 artifacts."""

    root = _repository_root(repository_root)
    project_root = _project_root(root)
    frozen_document = (
        project_root
        / "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
        "00-V3.3.3-r4优化方案冻结说明.md"
    )
    frozen_hash = _sha256_file(frozen_document)
    if frozen_hash.upper() != R4_FROZEN_DOCUMENT_SHA256:
        raise ValueError(
            "r4 frozen document hash drift: "
            f"expected {R4_FROZEN_DOCUMENT_SHA256}, got {frozen_hash}"
        )
    config_path = root / DEFAULT_CONFIG_RELATIVE
    if not config_path.is_file():
        raise FileNotFoundError(f"missing frozen r4 config: {config_path}")

    frame = eligible_rows(load_r3_training_frame(repository_root=root)).copy()
    assert_participant_isolation(frame)
    if frame["dataset_id"].eq("hefei_elderly").any():
        raise ValueError("Hefei label/auxiliary-only rows entered r4 passive scoring")
    frame["r4_row_id"] = frame["r3_row_id"].astype(str)
    population = frame[
        [
            "r4_row_id",
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "outer_fold",
            "inner_validation_fold_by_outer_fold",
            "route_pattern",
            "phq9_score_r3_target",
            "phq9_ge5_r3_target",
            "phq9_ge10_r3_target",
        ]
    ].copy()

    labels = {
        task: _label_summary(frame, column)
        for task, column in R4_LABEL_TASKS.items()
    }
    by_source: dict[str, Any] = {}
    for source, part in frame.groupby("dataset_id", sort=True):
        by_source[str(source)] = {
            "rows": int(len(part)),
            "participants": int(part["global_participant_id"].nunique()),
            "route_patterns": sorted(part["route_pattern"].astype(str).unique()),
            "labels": {
                task: _label_summary(part, column)
                for task, column in R4_LABEL_TASKS.items()
            },
        }

    output = Path(output_directory) if output_directory else root / DEFAULT_OUTPUT_RELATIVE
    report = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    population_path = output / "r4_dual_label_population.parquet"
    manifest_path = output / "r4_protocol_manifest.json"
    exposure_path = report / "data_exposure_ledger.json"
    for path in (population_path, manifest_path, exposure_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r4 protocol artifact: {path}")
    population_path.parent.mkdir(parents=True, exist_ok=True)
    population.to_parquet(population_path, index=False)

    manifest = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development/reused-benchmark",
        "frozen_document": frozen_document.relative_to(project_root).as_posix(),
        "frozen_document_sha256": frozen_hash,
        "config": DEFAULT_CONFIG_RELATIVE.as_posix(),
        "config_sha256": _sha256_file(config_path),
        "inherited_split_id": R3_SPLIT_ID,
        "inherited_split_sha256": R3_SPLIT_SHA256,
        "repeat_seeds": list(R4_REPEAT_SEEDS),
        "scorable_rows": int(len(frame)),
        "participant_count": int(frame["global_participant_id"].nunique()),
        "labels": labels,
        "by_source": by_source,
        "hefei_policy": "label/auxiliary-only; excluded from passive-risk AUPRC",
        "historical_phq_policy": "V3.4-only; forbidden from V3.3.3-r4 inputs",
        "network_dataset_download": False,
        "population_path": population_path.relative_to(root).as_posix()
        if population_path.is_relative_to(root)
        else population_path.as_posix(),
    }
    _write_json(manifest_path, manifest, overwrite=overwrite)
    _write_json(
        exposure_path,
        {
            "protocol_version": R4_PROTOCOL_VERSION,
            "r3_outer_population_exposed": True,
            "current_population_use": "adaptive-development/reused-benchmark",
            "independent_blind_confirmation": False,
            "blind_confirmation_requirement": (
                "new participants, a new collection batch/time wave, or a source whose "
                "labels and predictions have never been inspected"
            ),
            "local_data_only": True,
            "r2_r3_artifacts_mutated": False,
        },
        overwrite=overwrite,
    )
    return manifest


__all__ = [
    "R4_ALLOWED_FEATURES",
    "R4_FROZEN_DOCUMENT_SHA256",
    "R4_LABEL_TASKS",
    "R4_PROTOCOL_VERSION",
    "R4_REPEAT_SEEDS",
    "PSYCHE_RESEARCH_SENSOR_FEATURES",
    "assert_psyche_research_feature_names",
    "assert_r4_feature_names",
    "build_r4_protocol_artifacts",
    "forbidden_r4_feature_reason",
]
