"""Seeded contact sheets for manual trajectory-label quality control."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from elderly_monitoring.modules.mental_health.wandering.schemas import (
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
)


VISUAL_REVIEW_SCHEMA_VERSION = "wandering-visual-review-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class WanderingVisualizationError(ValueError):
    """A deterministic visual-review selection or rendering error."""


@dataclass(frozen=True)
class VisualReviewBundle:
    output_dir: Path
    selection_path: Path
    plot_paths: tuple[Path, ...]


def write_visual_review(
    samples: Iterable[TrajectorySample],
    *,
    source_name: str,
    input_role: str,
    input_sha256: str,
    output_dir: str | Path,
    samples_per_class: int = 20,
    seed: int = 20260801,
) -> VisualReviewBundle:
    """Select and render a reproducible random contact sheet for every class."""

    if not isinstance(source_name, str) or _SLUG.fullmatch(source_name) is None:
        raise WanderingVisualizationError("source_name must be a lowercase slug")
    if not isinstance(input_role, str) or _SLUG.fullmatch(input_role) is None:
        raise WanderingVisualizationError("input_role must be a lowercase slug")
    if not isinstance(input_sha256, str) or _SHA256.fullmatch(input_sha256) is None:
        raise WanderingVisualizationError("input_sha256 must be lowercase SHA-256")
    if (
        isinstance(samples_per_class, bool)
        or not isinstance(samples_per_class, int)
        or samples_per_class < 20
    ):
        raise WanderingVisualizationError("samples_per_class must be at least 20")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise WanderingVisualizationError("seed must be a non-negative integer")

    materialized = tuple(samples)
    if not materialized or any(
        not isinstance(sample, TrajectorySample) for sample in materialized
    ):
        raise WanderingVisualizationError(
            "visual review requires TrajectorySample objects"
        )
    sample_ids = [sample.sample_id for sample in materialized]
    if len(set(sample_ids)) != len(sample_ids):
        raise WanderingVisualizationError("visual review sample_ids must be unique")

    by_class: dict[str, list[TrajectorySample]] = defaultdict(list)
    for sample in materialized:
        by_class[_visual_class(sample)].append(sample)
    selected: dict[str, tuple[TrajectorySample, ...]] = {}
    for label, candidates in sorted(by_class.items()):
        if len(candidates) < samples_per_class:
            raise WanderingVisualizationError(
                f"class {label!r} requires at least {samples_per_class} samples, "
                f"found {len(candidates)}"
            )
        ordered = sorted(candidates, key=lambda sample: sample.sample_id)
        class_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{source_name}:{label}".encode("utf-8")).digest()[:8],
            "big",
        )
        selected[label] = tuple(
            random.Random(class_seed).sample(ordered, samples_per_class)
        )

    final_output = Path(output_dir).resolve(strict=False)
    if final_output.exists():
        raise WanderingVisualizationError(
            f"visual review output already exists: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            dir=final_output.parent,
            prefix=f".{final_output.name}.",
            suffix=".tmp",
        )
    )
    try:
        plot_records: dict[str, dict[str, str]] = {}
        for label, class_samples in selected.items():
            plot_path = stage / f"{label}.png"
            _render_contact_sheet(
                plot_path,
                source_name=source_name,
                label=label,
                samples=class_samples,
                seed=seed,
            )
            plot_records[label] = {
                "path": plot_path.name,
                "sha256": _sha256_file(plot_path),
            }
        selection = {
            "schema_version": VISUAL_REVIEW_SCHEMA_VERSION,
            "source_name": source_name,
            "input_role": input_role,
            "input_sha256": input_sha256,
            "seed": seed,
            "samples_per_class": samples_per_class,
            "classes": {
                label: [sample.sample_id for sample in class_samples]
                for label, class_samples in selected.items()
            },
            "plots": plot_records,
            "review_status": "pending_human_review",
            "review_checks": [
                "label_matches_visible_path_shape",
                "point_order_forms_continuous_path",
                "no_obvious_coordinate_or_axis_error",
                "start_and_end_markers_are_plausible",
            ],
        }
        (stage / "selection.json").write_bytes(_canonical_json_bytes(selection))
        os.replace(stage, final_output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return VisualReviewBundle(
        output_dir=final_output,
        selection_path=final_output / "selection.json",
        plot_paths=tuple(
            final_output / record["path"]
            for _, record in sorted(plot_records.items())
        ),
    )


def _visual_class(sample: TrajectorySample) -> str:
    if sample.pattern_label is not PatternLabel.UNKNOWN:
        return sample.pattern_label.value
    if sample.binary_label == 0:
        return "normal"
    if sample.binary_label == 1:
        return "wandering_like"
    raise WanderingVisualizationError(
        f"sample {sample.sample_id!r} has neither subtype nor binary label"
    )


def _render_contact_sheet(
    path: Path,
    *,
    source_name: str,
    label: str,
    samples: Sequence[TrajectorySample],
    seed: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise WanderingVisualizationError(
            "matplotlib is required in the eldercare-ai environment for visual review"
        ) from exc

    column_count = 5
    row_count = (len(samples) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, 3.2 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        f"{source_name} | {label} | n={len(samples)} | seed={seed}",
        fontsize=15,
    )
    for axis, sample in zip(axes.flat, samples):
        points = [
            point for point, mask in zip(sample.points, sample.point_mask) if mask == 1
        ]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        axis.plot(x_values, y_values, color="#2563eb", linewidth=1.5, alpha=0.9)
        axis.scatter(
            [x_values[0]], [y_values[0]], color="#16a34a", s=28, zorder=3
        )
        axis.scatter(
            [x_values[-1]], [y_values[-1]], color="#dc2626", s=28, zorder=3
        )
        axis.set_title(f"{sample.sample_id[-10:]} | T={len(points)}", fontsize=8)
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.12)
        if sample.coordinate_system is CoordinateSystem.IMAGE_NORMALIZED:
            axis.invert_yaxis()
        axis.grid(True, linewidth=0.35, alpha=0.25)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(samples) :]:
        axis.axis("off")
    figure.savefig(
        path,
        dpi=150,
        facecolor="white",
        metadata={"Software": "elderly-monitoring-algorithms"},
    )
    plt.close(figure)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
