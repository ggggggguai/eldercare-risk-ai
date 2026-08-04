#!/usr/bin/env python3
"""Render the builder-fixed wandering step-4 diagnostic review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    TrajectoryDatasetError,
    load_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    InputHashMismatchError,
    PREPROCESSED_SAMPLE_SCHEMA_VERSION,
    PREPROCESSING_REPORT_SCHEMA_VERSION,
    PreprocessingDataError,
    load_preprocessing_config,
)


_READY_GROUPS = (
    ("wandering_patterns", "direct"),
    ("wandering_patterns", "pacing"),
    ("wandering_patterns", "lapping"),
    ("wandering_patterns", "random"),
    ("smartcare", "normal"),
    ("smartcare", "wandering_like"),
)


def visualize_preprocessing_from_files(
    *,
    config_path: str | Path,
    project_root: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Reverify sources and render only IDs fixed by the preprocessing report."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"wandering preprocessing diagnostics already exist: {output}")
    config_file = Path(config_path)
    config = load_preprocessing_config(config_file)
    root = Path(project_root).resolve(strict=True)
    bundle = Path(bundle_dir)
    source_paths = _reverify_public_sources(config, root)
    config_hash = _sha256_file(config_file)
    report = _load_json(bundle / "preprocessing_report.json")
    if report.get("schema_version") != PREPROCESSING_REPORT_SCHEMA_VERSION:
        raise PreprocessingDataError("invalid preprocessing report schema")
    if report.get("preprocessing_config_sha256") != config_hash:
        raise InputHashMismatchError("preprocessing report config SHA-256 drift")
    if report.get("split_sha256") != config["split_sha256"]:
        raise InputHashMismatchError("preprocessing report split SHA-256 drift")
    rows = _load_preprocessed_rows(bundle / "samples.jsonl")
    row_index = {row["sample_id"]: row for row in rows}
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise PreprocessingDataError("preprocessing report has no diagnostics selection")
    if diagnostics.get("seed") != config["diagnostics"]["seed"] or diagnostics.get(
        "samples_per_class"
    ) != config["diagnostics"]["samples_per_class"]:
        raise PreprocessingDataError("diagnostic selection parameters drifted")

    wp_samples = load_trajectory_jsonl(source_paths["wandering_patterns_samples"])
    smartcare_samples = load_trajectory_jsonl(source_paths["smartcare_train_pool"])
    source_index = {sample.sample_id: sample for sample in (*wp_samples, *smartcare_samples)}
    if len(source_index) != 1790:
        raise PreprocessingDataError("visualizer public source index must contain 1790 IDs")

    selected = diagnostics.get("ready_train_sample_ids")
    if not isinstance(selected, dict):
        raise PreprocessingDataError("report ready_train_sample_ids is invalid")
    selected_ids: list[str] = []
    for source, label in _READY_GROUPS:
        ids = selected.get(source, {}).get(label)
        if not isinstance(ids, list) or len(ids) != config["diagnostics"]["samples_per_class"]:
            raise PreprocessingDataError(f"fixed diagnostic group {source}/{label} is invalid")
        selected_ids.extend(ids)
        for sample_id in ids:
            row = row_index.get(sample_id)
            if row is None or row.get("preprocess_status") != "ready" or row.get("split") != "train":
                raise PreprocessingDataError(f"diagnostic ID is not ready train: {sample_id}")
            if row.get("source_dataset") != source:
                raise PreprocessingDataError(f"diagnostic source mismatch: {sample_id}")
            if source == "wandering_patterns" and row.get("pattern_label") != label:
                raise PreprocessingDataError(f"diagnostic pattern label mismatch: {sample_id}")
            expected_binary = 0 if label == "normal" else 1
            if source == "smartcare" and row.get("binary_label") != expected_binary:
                raise PreprocessingDataError(f"diagnostic binary label mismatch: {sample_id}")
            if sample_id not in source_index:
                raise PreprocessingDataError(f"diagnostic ID missing from public source: {sample_id}")
    if len(selected_ids) != 48 or len(set(selected_ids)) != 48:
        raise PreprocessingDataError("fixed ready diagnostics must contain 48 unique IDs")

    unavailable_ids = diagnostics.get("unavailable_sample_ids")
    if not isinstance(unavailable_ids, list) or len(unavailable_ids) != 15:
        raise PreprocessingDataError("fixed unavailable diagnostics must contain 15 IDs")
    for sample_id in unavailable_ids:
        row = row_index.get(sample_id)
        source = source_index.get(sample_id)
        if (
            row is None
            or source is None
            or row.get("preprocess_status") != "unavailable"
            or row.get("reason_codes") != ["too_few_valid_points"]
            or row.get("source_dataset") != "smartcare"
            or row.get("split") != "train"
        ):
            raise PreprocessingDataError(f"invalid fixed unavailable diagnostic ID: {sample_id}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    committed = False
    try:
        diagnostics_dir = temp_dir / "diagnostics"
        diagnostics_dir.mkdir()
        for source, label in _READY_GROUPS:
            ids = selected[source][label]
            _render_ready_group(
                diagnostics_dir / f"{source}_{label}.png",
                title=f"{source} / {label} / ready train",
                sample_ids=ids,
                source_index=source_index,
                row_index=row_index,
            )
        _render_unavailable_group(
            diagnostics_dir / "smartcare_unavailable_too_few_valid_points.png",
            sample_ids=unavailable_ids,
            source_index=source_index,
            row_index=row_index,
        )
        (temp_dir / "HUMAN_REVIEW.md").write_text(
            _human_review_markdown(selected, unavailable_ids),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp_dir, output)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return output


def _render_ready_group(
    output_path: Path,
    *,
    title: str,
    sample_ids: Sequence[str],
    source_index: Mapping[str, Any],
    row_index: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(len(sample_ids), 4, figsize=(15, 2.7 * len(sample_ids)))
    figure.suptitle(title, fontsize=14)
    for row_number, sample_id in enumerate(sample_ids):
        source = source_index[sample_id]
        record = row_index[sample_id]
        raw_points = np.asarray(source.points, dtype=np.float64)
        resampled = np.asarray(record["resampled_source_points"], dtype=np.float64)
        shape = np.asarray(record["shape_normalized_points"], dtype=np.float64)
        topology = record["topology"]
        _plot_path(axes[row_number, 0], raw_points, "source")
        _plot_path(axes[row_number, 1], resampled, "T=80 source")
        _plot_path(axes[row_number, 2], shape, "shape")
        _plot_topology(axes[row_number, 3], shape, topology)
        axes[row_number, 0].set_ylabel(sample_id, fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _render_unavailable_group(
    output_path: Path,
    *,
    sample_ids: Sequence[str],
    source_index: Mapping[str, Any],
    row_index: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(15, 18))
    figure.suptitle("SmartCare train unavailable: too_few_valid_points (all 15)", fontsize=14)
    for axis, sample_id in zip(axes.flat, sample_ids):
        source = source_index[sample_id]
        record = row_index[sample_id]
        points = np.asarray(source.points, dtype=np.float64)
        _plot_path(axis, points, "raw source path")
        label = "normal" if record["binary_label"] == 0 else "wandering-like"
        axis.set_title(
            f"{sample_id}\n{label}; N={record['input_point_count']}; "
            f"reason={record['reason_codes'][0]}",
            fontsize=7,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _plot_path(axis: Any, points: np.ndarray, title: str) -> None:
    axis.plot(points[:, 0], points[:, 1], "-o", linewidth=1.0, markersize=2.0)
    axis.scatter(points[0, 0], points[0, 1], c="green", marker="o", s=25, label="start")
    axis.scatter(points[-1, 0], points[-1, 1], c="black", marker="x", s=25, label="end")
    axis.set_title(title, fontsize=8)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)


def _plot_topology(axis: Any, points: np.ndarray, topology: Mapping[str, Any]) -> None:
    curvature = np.asarray(topology["abs_curvature"], dtype=np.float64)
    axis.plot(points[:, 0], points[:, 1], color="0.65", linewidth=0.8)
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=curvature,
        cmap="magma",
        s=9,
    )
    for event in topology["reversal_events"]:
        index = event["index"]
        axis.scatter(points[index, 0], points[index, 1], c="cyan", marker="x", s=35)
    for link in topology["selected_revisit_links"]:
        left = points[link["i"]]
        right = points[link["j"]]
        axis.plot([left[0], right[0]], [left[1], right[1]], "--", color="royalblue", linewidth=0.8)
    axis.set_title(
        f"curvature/reversal/revisit\nwind={topology['absolute_winding']:.3f}; "
        f"rev={len(topology['reversal_events'])}; pairs={topology['revisit_pair_count']}",
        fontsize=7,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    scatter.set_clim(vmin=0.0)


def _human_review_markdown(
    selected: Mapping[str, Mapping[str, Sequence[str]]],
    unavailable_ids: Sequence[str],
) -> str:
    lines = [
        "# 徘徊步骤 4 预处理人工图审",
        "",
        "状态：`pending_human_review`",
        "",
        "本文件只记录人工复核状态。自动测试和生成诊断图不能替代人工签字；状态更新为通过前，不允许开始步骤 5。",
        "",
        "## 待人工确认",
        "",
        "- [ ] 48 条 ready train 固定样本均同时核对来源轨迹、T=80 来源视图、shape 视图、曲率、反向点和回访连线。",
        "- [ ] 15 条 SmartCare train 短轨迹均保持原路径与 `too_few_valid_points`，未补点训练、删除或重分组。",
        "- [ ] WanderingPatterns 没有伪造 image-normalized 画布，图中没有 bbox 高度补偿内容。",
        "- [ ] 记录复核人、日期、异常 sample ID 与最终结论。",
        "",
        "复核人：",
        "",
        "复核日期：",
        "",
        "异常 sample ID：",
        "",
        "结论：",
        "",
        "## 固定 ready train ID",
        "",
    ]
    for source, groups in selected.items():
        for label, ids in groups.items():
            lines.append(f"- `{source}/{label}`: " + ", ".join(f"`{item}`" for item in ids))
    lines.extend(
        [
            "",
            "## 全部 unavailable 短轨迹 ID",
            "",
            "- " + ", ".join(f"`{item}`" for item in unavailable_ids),
            "",
        ]
    )
    return "\n".join(lines)


def _reverify_public_sources(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role in ("wandering_patterns_samples", "smartcare_train_pool"):
        descriptor = config["inputs"][role]
        path = (root / PurePosixPath(descriptor["path"])).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PreprocessingDataError(f"source {role} escapes project_root") from exc
        if not path.is_file():
            raise InputHashMismatchError(f"source {role} is missing")
        actual = _sha256_file(path)
        if actual != descriptor["sha256"]:
            raise InputHashMismatchError(
                f"source {role} SHA-256 drift: expected {descriptor['sha256']}, got {actual}"
            )
        paths[role] = path
    return paths


def _load_preprocessed_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                row = json.loads(raw_line)
                if not isinstance(row, dict) or row.get("schema_version") != PREPROCESSED_SAMPLE_SCHEMA_VERSION:
                    raise PreprocessingDataError(f"invalid preprocessed row {line_number}")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in ids:
                    raise PreprocessingDataError(f"invalid duplicate preprocessed ID at line {line_number}")
                ids.add(sample_id)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessingDataError(f"cannot load preprocessed samples: {path}") from exc
    if len(rows) != 1790:
        raise PreprocessingDataError("preprocessed samples must contain 1790 rows")
    return tuple(rows)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessingDataError(f"cannot load JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PreprocessingDataError(f"JSON artifact must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/wandering_preprocessing_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("data/processed/wandering/preprocessing/v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step4"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = visualize_preprocessing_from_files(
            config_path=args.config,
            project_root=args.project_root,
            bundle_dir=args.bundle,
            output_dir=args.output,
        )
    except (FileExistsError, OSError, PreprocessingDataError, TrajectoryDatasetError) as exc:
        print(f"Wandering preprocessing visualization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
