from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from elderly_monitoring.modules.fall_risk.fall_tiktok import (
    FallTiktokCollectionDecision,
    load_fall_tiktok_collection_decision,
    load_fall_tiktok_source_map,
)
from elderly_monitoring.modules.fall_risk.ntu_rgbd_cvat import (
    NTU_RGBD_A043_BATCH_ID,
    load_ntu_rgbd_a043_decision,
)


VIDEO_METADATA_FIELDS = (
    "fps_num",
    "fps_den",
    "fps",
    "frame_count",
    "duration_sec",
    "width",
    "height",
)


@dataclass(frozen=True)
class VideoMetadata:
    fps_num: int
    fps_den: int
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass(frozen=True)
class ManifestBuildResult:
    rows: list[dict[str, Any]]
    content: bytes
    manifest_sha256: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class PreVFallpMediaInventoryResult:
    rows: list[dict[str, Any]]
    content: bytes
    sha256: str
    summary: dict[str, Any]


class MediaProbeError(ValueError):
    pass


@dataclass(frozen=True)
class _Provenance:
    source_uri: str | None


_PROVENANCE = {
    "le2i_imvia": _Provenance(
        source_uri=(
            "https://search-data.ubfc.fr/imvia/"
            "FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html"
        ),
    ),
    "fall_detection_2017": _Provenance(
        source_uri="https://doi.org/10.6084/m9.figshare.28596332.v2",
    ),
    "ur_fall": _Provenance(
        source_uri="https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
    ),
    "toaga": _Provenance(
        source_uri=(
            "https://springernature.figshare.com/collections/"
            "The_Toronto_Older_Adults_Gait_Archive_Video_and_3D_"
            "Inertial_Motion_Capture_Data_of_Older_Adults_Walking/5515953"
        ),
    ),
    "gstride": _Provenance(
        source_uri="https://doi.org/10.5281/zenodo.17052815",
    ),
    "ltmm": _Provenance(
        source_uri="https://physionet.org/content/ltmm/1.0.0/",
    ),
    "pre_vfallp": _Provenance(source_uri=None),
    "caucafall": _Provenance(
        source_uri="https://doi.org/10.17632/7w7fccy7ky.4",
    ),
    "fall_tiktok": _Provenance(source_uri=None),
    "ntu_rgbd": _Provenance(
        source_uri="https://rose1.ntu.edu.sg/dataset/actionRecognition/",
    ),
}

_INTERNAL_AUTHORIZATION_CONFIG = Path(
    "configs/data/fall_risk_internal_authorizations.yaml"
)
_INTERNAL_AUTHORIZATION_SCHEMA = "fall-risk-internal-authorizations-v2"
_INTERNAL_AUTHORIZATION_URI_PREFIX = "internal://authorization/"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRE_VFALLP_MEDIA_INVENTORY_SCHEMA = "pre-vfallp-internal-media-inventory-v1"
_NTU_RGBD_EXTERNAL_MANIFEST = Path("data/manifests/ntu_rgbd_clip_manifest.jsonl")
_NTU_RGBD_LABEL_MAP = Path("configs/data/ntu_rgbd_clip_label_map_v2.json")
_NTU_RGBD_A043_DECISION = Path("configs/data/ntu_rgbd_a043_cvat_decision_v1.json")
_NTU_RGBD_A043_BATCH = Path(
    "data/annotations/fall_risk/generated/v2/ntu_rgbd_a043_cvat_review"
)
_NTU_RGBD_A043_ACTION_LABELS = _NTU_RGBD_A043_BATCH / "action_labels.jsonl"
_NTU_RGBD_A043_SOURCE_ANNOTATIONS = _NTU_RGBD_A043_BATCH / "source_annotations.zip"
_FALL_TIKTOK_SOURCE_MAP = Path("configs/data/fall_tiktok_source_map_v1.json")
_FALL_TIKTOK_COLLECTION_DECISION = Path(
    "configs/data/fall_tiktok_collection_decision_v1.json"
)
_FALL_TIKTOK_REDACTED_CVAT = Path(
    "data/annotations/fall_risk/cvat_exports/raw/fall_tiktok/"
    "fall_tiktok_cvat_redacted.zip"
)
_INTERNAL_AUTHORIZATION_EVIDENCE_TYPES = {
    "cvat_export_archive",
    "local_media_inventory",
}


@dataclass(frozen=True)
class _PreVFallpInventoryRecord:
    video_id: str
    path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class _InternalAuthorizationEvidence:
    kind: str
    name: str
    sha256: str
    path: str | None = None
    inventory_id: str | None = None
    inventory_records: Mapping[str, _PreVFallpInventoryRecord] | None = None


@dataclass(frozen=True)
class _InternalAuthorization:
    authorization_id: str
    approval_reference: str
    approved_at: str
    source_uri: str
    evidence: _InternalAuthorizationEvidence
    video_ids: frozenset[str]


_VideoProbe = Callable[[Path], VideoMetadata]
_NTU_RGBD_VIDEO_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})"
    r"R(?P<repetition>\d{3})A(?P<action>\d{3})_rgb\.avi",
    re.IGNORECASE,
)
_CAUCAFALL_VIDEO_PATTERN = re.compile(
    r"(?P<action>Walk|Hop|Pickupobject|SitDown|Kneel|FallForward|"
    r"FallBackwards|FallLeft|FallRight|FallSitting)S(?P<subject>\d+)\.avi",
    re.IGNORECASE,
)


def probe_video_metadata(
    path: Path | str,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[..., Any] | None = None,
) -> VideoMetadata:
    run = runner or subprocess.run
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,"
            "width,height,duration:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise MediaProbeError("ffprobe failed")

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc

    rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    if rate is None:
        rate = _parse_frame_rate(stream.get("r_frame_rate"))
    if rate is None:
        raise MediaProbeError("ffprobe did not return a valid frame rate")

    frame_count = _positive_int(stream.get("nb_frames"))
    if frame_count is None:
        frame_count = _positive_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _count_video_frames(path, ffprobe_bin=ffprobe_bin, runner=run)

    duration = _positive_float(stream.get("duration"))
    if duration is None:
        duration = _positive_float(payload.get("format", {}).get("duration"))
    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    if duration is None or width is None or height is None:
        raise MediaProbeError("ffprobe returned incomplete video metadata")

    return VideoMetadata(
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        fps=float(rate),
        frame_count=frame_count,
        duration_sec=duration,
        width=width,
        height=height,
    )


def build_fall_risk_manifest(
    repo_root: Path | str,
    *,
    probe_video: _VideoProbe | None = None,
    ffprobe_bin: str = "ffprobe",
) -> ManifestBuildResult:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")

    probe = probe_video or (
        lambda path: probe_video_metadata(path, ffprobe_bin=ffprobe_bin)
    )
    internal_authorizations = _load_internal_authorizations(root)
    rows: list[dict[str, Any]] = []
    adapters = (
        _adapt_le2i,
        _adapt_fall_detection_2017,
        _adapt_ur_fall,
        _adapt_toaga,
        _adapt_gstride,
        _adapt_ltmm,
        _adapt_pre_vfallp,
        _adapt_caucafall,
        _adapt_fall_tiktok,
    )
    for adapter in adapters:
        rows.extend(adapter(root, probe))
    rows.extend(_load_reviewed_ntu_rgbd_rows(root))

    _mark_duplicate_content(rows)
    _apply_internal_authorizations(rows, internal_authorizations)
    rows.sort(key=lambda row: str(row["path"]))
    content = _encode_jsonl(rows)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    summary = _build_summary(rows, manifest_sha256)
    return ManifestBuildResult(
        rows=rows,
        content=content,
        manifest_sha256=manifest_sha256,
        summary=summary,
    )


def _load_reviewed_ntu_rgbd_rows(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / _NTU_RGBD_EXTERNAL_MANIFEST
    label_map_path = root / _NTU_RGBD_LABEL_MAP
    if not manifest_path.is_file() and not label_map_path.is_file():
        return []
    if not manifest_path.is_file() or not label_map_path.is_file():
        missing = manifest_path if not manifest_path.is_file() else label_map_path
        raise FileNotFoundError(missing)

    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "ntu-rgbd-clip-label-map-v2":
        raise ValueError(f"unsupported NTU RGB+D label map: {label_map_path}")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError(f"invalid NTU RGB+D label mappings: {label_map_path}")
    reviewed_codes = {
        str(entry.get("source_action_code"))
        for entry in mappings
        if isinstance(entry, dict) and entry.get("mode") == "manual_exact"
    }
    accepted_a043_video_ids, a043_decision = _load_accepted_ntu_a043_video_ids(root)

    rows: list[dict[str, Any]] = []
    matched_a043_video_ids: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}:{line_number}: row must be an object")
        if row.get("dataset") != "ntu_rgbd":
            raise ValueError(f"{manifest_path}:{line_number}: unexpected dataset")
        source_action_code = row.get("source_action_code")
        video_id = str(row.get("video_id") or "")
        is_manual_a043 = (
            source_action_code == "A043" and video_id in accepted_a043_video_ids
        )
        if source_action_code not in reviewed_codes and not is_manual_a043:
            continue
        reviewed_row = dict(row)
        if is_manual_a043:
            matched_a043_video_ids.add(video_id)
            assert a043_decision is not None
            reviewed_row["label_source"] = "cvat_manual"
            reviewed_row["annotation_path"] = (
                _NTU_RGBD_A043_SOURCE_ANNOTATIONS.as_posix()
            )
            reviewed_row["label_batch_id"] = NTU_RGBD_A043_BATCH_ID
            reviewed_row["label_decision_id"] = a043_decision["decision_id"]
            reviewed_row["label_decision_path"] = _NTU_RGBD_A043_DECISION.as_posix()
        else:
            reviewed_row["label_source"] = "manual_exact_clip_boundary"
            reviewed_row["annotation_path"] = _NTU_RGBD_LABEL_MAP.as_posix()
        rows.append(reviewed_row)
    missing_a043 = sorted(accepted_a043_video_ids - matched_a043_video_ids)
    if missing_a043:
        raise ValueError(
            "accepted NTU RGB+D A043 video_id missing from external manifest: "
            + ", ".join(missing_a043[:5])
        )
    return rows


def _load_accepted_ntu_a043_video_ids(
    root: Path,
) -> tuple[set[str], dict[str, Any] | None]:
    decision_path = root / _NTU_RGBD_A043_DECISION
    action_path = root / _NTU_RGBD_A043_ACTION_LABELS
    annotation_path = root / _NTU_RGBD_A043_SOURCE_ANNOTATIONS
    artifacts = (decision_path, action_path, annotation_path)
    if not any(path.exists() for path in artifacts):
        return set(), None
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    decision = load_ntu_rgbd_a043_decision(decision_path)
    video_ids: set[str] = set()
    for line_number, line in enumerate(
        action_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"{action_path}:{line_number}: blank JSONL line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{action_path}:{line_number}: row must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{action_path}:{line_number}: missing video_id")
        if row.get("source") != "cvat":
            raise ValueError(f"{action_path}:{line_number}: source must be cvat")
        video_ids.add(video_id)
    if not video_ids:
        raise ValueError(f"accepted NTU RGB+D A043 batch is empty: {action_path}")
    return video_ids, decision


def write_fall_risk_manifest(
    repo_root: Path | str,
    output_path: Path | str = "data/manifests/fall_risk_video_manifest.jsonl",
    *,
    overwrite: bool = False,
    probe_video: _VideoProbe | None = None,
    ffprobe_bin: str = "ffprobe",
) -> ManifestBuildResult:
    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    if output.exists() and not overwrite:
        raise FileExistsError(f"manifest output already exists: {output}")
    result = build_fall_risk_manifest(
        root,
        probe_video=probe_video,
        ffprobe_bin=ffprobe_bin,
    )
    _atomic_write(output, result.content, overwrite=overwrite)
    return result


def build_pre_vfallp_media_inventory(
    repo_root: Path | str,
    *,
    inventory_id: str,
    subsets: Iterable[str],
) -> PreVFallpMediaInventoryResult:
    """Freeze a content-addressed inventory for an internally authorized subset.

    This artifact records local media identity only. It is not evidence of a public
    source or a label export, so callers must pair it with an explicit internal
    authorization before enabling the listed assets.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")
    normalized_inventory_id = _required_internal_authorization_string(
        inventory_id, "inventory_id"
    )
    normalized_subsets = frozenset(
        _required_internal_authorization_string(subset, "subset")
        for subset in subsets
    )
    if not normalized_subsets:
        raise ValueError("Pre_VFallp media inventory requires at least one subset")

    dataset_root = root / "data/external/Pre_VFallp"
    rows: list[dict[str, Any]] = []
    for path in _files(dataset_root, "*.mp4"):
        relative = path.relative_to(dataset_root)
        if len(relative.parts) < 2:
            raise ValueError(f"Pre_VFallp video is not inside a subset directory: {path}")
        subset = relative.parts[0]
        if subset not in normalized_subsets:
            continue
        repo_relative = _repo_relative(path, root)
        rows.append(
            {
                "schema_version": _PRE_VFALLP_MEDIA_INVENTORY_SCHEMA,
                "inventory_id": normalized_inventory_id,
                "dataset": "pre_vfallp",
                "subset": subset,
                "video_id": _path_identity("pre_vfallp_video", relative.with_suffix("")),
                "path": repo_relative,
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
    if not rows:
        raise FileNotFoundError(
            "no Pre_VFallp media found for requested subsets: "
            + ", ".join(sorted(normalized_subsets))
        )
    rows.sort(key=lambda row: str(row["video_id"]))
    content = _encode_jsonl(rows)
    inventory_sha256 = hashlib.sha256(content).hexdigest()
    return PreVFallpMediaInventoryResult(
        rows=rows,
        content=content,
        sha256=inventory_sha256,
        summary={
            "schema_version": _PRE_VFALLP_MEDIA_INVENTORY_SCHEMA,
            "inventory_id": normalized_inventory_id,
            "asset_count": len(rows),
            "subsets": sorted(normalized_subsets),
            "sha256": inventory_sha256,
        },
    )


def write_pre_vfallp_media_inventory(
    repo_root: Path | str,
    output_path: Path | str,
    *,
    inventory_id: str,
    subsets: Iterable[str],
    overwrite: bool = False,
) -> PreVFallpMediaInventoryResult:
    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    if output.exists() and not overwrite:
        raise FileExistsError(f"media inventory output already exists: {output}")
    result = build_pre_vfallp_media_inventory(
        root,
        inventory_id=inventory_id,
        subsets=subsets,
    )
    _atomic_write(output, result.content, overwrite=overwrite)
    return result


def build_ntu_rgbd_clip_manifest(
    source_root: Path | str,
    *,
    probe_video: _VideoProbe | None = None,
    ffprobe_bin: str = "ffprobe",
    workers: int = 1,
) -> ManifestBuildResult:
    """Build a source-isolated manifest for externally stored NTU RGB+D RGB clips.

    NTU media is intentionally allowed to remain outside the repository. Its manifest
    records absolute local paths while retaining a root-relative path only for stable
    asset identities.
    """
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NTU RGB+D source root does not exist: {root}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")

    paths = _files(root, "*_rgb.avi")
    if not paths:
        raise FileNotFoundError(f"no NTU RGB+D RGB AVI files found under: {root}")
    for path in paths:
        if _NTU_RGBD_VIDEO_PATTERN.fullmatch(path.name) is None:
            raise ValueError(f"invalid NTU RGB+D RGB filename: {path.name}")

    probe = probe_video or (
        lambda path: probe_video_metadata(path, ffprobe_bin=ffprobe_bin)
    )

    def build_row(path: Path) -> dict[str, Any]:
        return _ntu_rgbd_clip_asset_row(root, path, probe)

    if workers == 1:
        rows = [build_row(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(build_row, paths))

    video_ids = [str(row["video_id"]) for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("duplicate NTU RGB+D video_id derived from filenames")
    _mark_duplicate_content(rows)
    rows.sort(key=lambda row: str(row["video_id"]))
    content = _encode_jsonl(rows)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    summary = _build_summary(rows, manifest_sha256)
    summary.update(
        {
            "version": "ntu-rgbd-clip-manifest-v1",
            "source_root": root.as_posix(),
            "source_action_code_counts": dict(
                sorted(Counter(str(row["source_action_code"]) for row in rows).items())
            ),
        }
    )
    return ManifestBuildResult(
        rows=rows,
        content=content,
        manifest_sha256=manifest_sha256,
        summary=summary,
    )


def write_ntu_rgbd_clip_manifest(
    source_root: Path | str,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    probe_video: _VideoProbe | None = None,
    ffprobe_bin: str = "ffprobe",
    workers: int = 1,
) -> ManifestBuildResult:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"manifest output already exists: {output}")
    result = build_ntu_rgbd_clip_manifest(
        source_root,
        probe_video=probe_video,
        ffprobe_bin=ffprobe_bin,
        workers=workers,
    )
    _atomic_write(output, result.content, overwrite=overwrite)
    return result


def _adapt_le2i(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/le2i_imvia/raw/FallDataset"
    rows = []
    for path in _files(dataset_root, "*.avi"):
        relative = path.relative_to(dataset_root)
        if len(relative.parts) == 3 and relative.parts[1] == "Videos":
            subset = relative.parts[0]
        elif len(relative.parts) == 2:
            subset = relative.parts[0]
        else:
            continue
        match = re.fullmatch(r"video \((\d+)\)\.avi", path.name, re.IGNORECASE)
        if match is None:
            continue
        video_number = int(match.group(1))
        subset_id = _slug(subset)
        video_id = f"le2i_{subset_id}_video_{video_number}"
        annotation = _first_existing(
            dataset_root / subset / "Annotation_files" / f"video ({video_number}).txt",
            dataset_root / subset / "Annotations_files" / f"video ({video_number}).txt",
        )
        scene = {
            "home_01": "home",
            "home_02": "home",
            "coffee_room_01": "coffee_room",
            "coffee_room_02": "coffee_room",
            "lecture_room": "lecture_room",
            "office": "office",
        }.get(subset_id, "unknown")
        rows.append(
            _asset_row(
                root,
                path,
                dataset="le2i_imvia",
                subset=subset,
                media_type="video",
                modality="rgb_video",
                video_id=video_id,
                subject_id="unknown",
                source_group_id=f"le2i_{subset_id}_unknown_subject_pool",
                original_event_id=video_id,
                scene_region=scene,
                view="fixed_camera",
                label_source="le2i_txt" if annotation else "unlabeled",
                annotation_path=annotation,
                probe=probe,
            )
        )
    return rows


def _adapt_fall_detection_2017(
    root: Path, probe: _VideoProbe
) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/fall_detection_2017/raw/VideoDataset"
    rows = []
    for path in _files(dataset_root, "*.mp4"):
        if path.name.lower().startswith("timelapse"):
            continue
        if {"all_csvs", "all_plots"}.intersection(path.parts):
            continue

        relative = path.relative_to(dataset_root)
        subset = relative.parts[0] if relative.parts else "unknown"
        metadata_path = path.parent / "metadata.json"
        metadata, metadata_reasons = _load_fall_2017_metadata(metadata_path)
        if metadata is None:
            subject_id = "unknown"
            source_group_id = "fall_detection_2017_unresolved"
            original_event_id = _path_identity("fall_detection_2017_event", relative.parent)
            scene = "unknown"
        else:
            subject_number = int(metadata["subjectId"])
            location = int(metadata["locationId"])
            action = int(metadata["actionId"])
            side = _slug(str(metadata["side"]))
            attempt = int(metadata["attempt"])
            subject_id = f"fall_detection_2017_sbj_{subject_number:02d}"
            source_group_id = subject_id
            original_event_id = (
                f"{subject_id}_loc_{location}_act_{action}_side_{side}_attempt_{attempt}"
            )
            scene = f"location_{location}"
            metadata_reasons.extend(
                _fall_2017_path_mismatch_reasons(relative, metadata)
            )

        video_id = _unique_video_id(original_event_id, relative)
        rows.append(
            _asset_row(
                root,
                path,
                dataset="fall_detection_2017",
                subset=subset,
                media_type="video",
                modality="rgb_video",
                video_id=video_id,
                subject_id=subject_id,
                source_group_id=source_group_id,
                original_event_id=original_event_id,
                scene_region=scene,
                view="unknown",
                label_source="official_metadata_clip_label",
                annotation_path=metadata_path if metadata_path.is_file() else None,
                probe=probe,
                extra_exclusion_reasons=metadata_reasons,
            )
        )
    return rows


def _adapt_ur_fall(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/ur_fall/raw"
    rows = []
    video_pattern = re.compile(r"(fall|adl)-(\d+)-(cam[01])\.mp4", re.IGNORECASE)
    for path in _files(dataset_root, "*.mp4"):
        match = video_pattern.fullmatch(path.name)
        if match is None:
            continue
        event_type = match.group(1).lower()
        event_number = int(match.group(2))
        view = match.group(3).lower()
        event_id = f"ur_fall_{event_type}_{event_number:02d}"
        annotation = dataset_root / f"{event_type}-{event_number:02d}-data.csv"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="ur_fall",
                subset=event_type,
                media_type="video",
                modality="rgb_video",
                video_id=f"{event_id}_{view}",
                subject_id="unknown",
                source_group_id=event_id,
                original_event_id=event_id,
                scene_region="lab",
                view=view,
                label_source="official_filename_and_sync_data",
                annotation_path=annotation if annotation.is_file() else None,
                probe=probe,
            )
        )

    event_data_pattern = re.compile(
        r"(fall|adl)-(\d+)-(acc|data)\.csv", re.IGNORECASE
    )
    index_pattern = re.compile(
        r"urfall-(cam\d+)-(falls|adls)\.csv", re.IGNORECASE
    )
    for path in _files(dataset_root, "*.csv"):
        event_match = event_data_pattern.fullmatch(path.name)
        if event_match is not None:
            event_type = event_match.group(1).lower()
            event_number = int(event_match.group(2))
            kind = event_match.group(3).lower()
            event_id = f"ur_fall_{event_type}_{event_number:02d}"
            rows.append(
                _asset_row(
                    root,
                    path,
                    dataset="ur_fall",
                    subset=event_type,
                    media_type="timeseries",
                    modality=(
                        "wearable_accelerometer"
                        if kind == "acc"
                        else "event_sync_data"
                    ),
                    video_id=None,
                    subject_id="unknown",
                    source_group_id=event_id,
                    original_event_id=event_id,
                    scene_region="lab",
                    view=None,
                    label_source="official_filename_and_sensor_data",
                    annotation_path=None,
                    probe=probe,
                )
            )
            continue

        index_match = index_pattern.fullmatch(path.name)
        if index_match is None:
            continue
        camera = index_match.group(1).lower()
        event_type = "fall" if index_match.group(2).lower() == "falls" else "adl"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="ur_fall",
                subset="index",
                media_type="tabular",
                modality="camera_event_index",
                video_id=None,
                subject_id="unknown",
                source_group_id="ur_fall_dataset_index",
                original_event_id=f"ur_fall_{event_type}_{camera}_index",
                scene_region="lab",
                view=camera,
                label_source="official_camera_event_index",
                annotation_path=None,
                probe=probe,
            )
        )
    return rows


def _adapt_toaga(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/toaga/raw"
    videos_root = dataset_root / "Videos"
    pose_root = dataset_root / "Pose Tracking"
    participant_table = dataset_root / "Table_1.xlsx"
    rows = []
    pattern = re.compile(r"OAW(\d+)-(top|bottom)\.mp4", re.IGNORECASE)
    for path in _files(videos_root, "*.mp4"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        subject_number = int(match.group(1))
        view = match.group(2).lower()
        subject_id = f"toaga_oaw{subject_number:02d}"
        event_id = f"{subject_id}_walking"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="toaga",
                subset="walking",
                media_type="video",
                modality="rgb_video",
                video_id=f"{subject_id}_{view}",
                subject_id=subject_id,
                source_group_id=subject_id,
                original_event_id=event_id,
                scene_region="walking_lab",
                view=view,
                label_source="official_participant_table",
                annotation_path=(
                    participant_table if participant_table.is_file() else None
                ),
                probe=probe,
            )
        )

    pose_pattern = re.compile(
        r"OAW(\d+)-(OpenPose|Alphapose|Detectron)-"
        r"(top|bottom)-(front|back)-(\d+)\.csv",
        re.IGNORECASE,
    )
    for path in _files(pose_root, "*.csv"):
        match = pose_pattern.fullmatch(path.name)
        if match is None:
            continue
        subject_number = int(match.group(1))
        method = match.group(2).lower()
        camera = match.group(3).lower()
        direction = match.group(4).lower()
        trial = int(match.group(5))
        subject_id = f"toaga_oaw{subject_number:02d}"
        event_id = f"{subject_id}_walking"
        relative = path.relative_to(pose_root)
        path_reasons = []
        if (
            len(relative.parts) != 3
            or relative.parts[0].lower() != method
            or not relative.parts[1].isdigit()
            or int(relative.parts[1]) != subject_number
        ):
            path_reasons.append("metadata_path_mismatch")
        rows.append(
            _asset_row(
                root,
                path,
                dataset="toaga",
                subset=f"pose_tracking_{method}",
                media_type="timeseries",
                modality="pose_keypoints",
                video_id=None,
                subject_id=subject_id,
                source_group_id=subject_id,
                original_event_id=event_id,
                scene_region="walking_lab",
                view=f"{camera}_{direction}_trial_{trial}",
                label_source="official_pose_tracking",
                annotation_path=(
                    participant_table if participant_table.is_file() else None
                ),
                probe=probe,
                extra_exclusion_reasons=path_reasons,
            )
        )

    if participant_table.is_file():
        rows.append(
            _asset_row(
                root,
                participant_table,
                dataset="toaga",
                subset="participant_metadata",
                media_type="tabular",
                modality="participant_metadata",
                video_id=None,
                subject_id="unknown",
                source_group_id="toaga_participant_table",
                original_event_id="toaga_participant_table",
                scene_region="walking_lab",
                view=None,
                label_source="official_participant_table",
                annotation_path=None,
                probe=probe,
            )
        )
    return rows


def _adapt_gstride(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/gstride/raw"
    database_root = dataset_root / "GSTRIDE_database"
    participant_table = dataset_root / "GSTRIDE_DDBB.xlsx"
    rows = []
    pattern = re.compile(
        r"V(\d+)(?:_GAIT_(SEGMENTATION|PARAMETERS))?\.csv", re.IGNORECASE
    )
    for path in _files(database_root, "*.csv"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        subject_number = int(match.group(1))
        kind = (match.group(2) or "IMU").lower()
        subject_id = f"gstride_v{subject_number:03d}"
        if kind == "imu":
            media_type = "timeseries"
            modality = "foot_imu"
        elif kind == "segmentation":
            media_type = "tabular"
            modality = "gait_segmentation"
        else:
            media_type = "tabular"
            modality = "gait_parameters"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="gstride",
                subset=kind,
                media_type=media_type,
                modality=modality,
                video_id=None,
                subject_id=subject_id,
                source_group_id=subject_id,
                original_event_id=f"{subject_id}_gait_assessment",
                scene_region="walking_lab",
                view=None,
                label_source="official_table_proxy",
                annotation_path=(
                    participant_table if participant_table.is_file() else None
                ),
                probe=probe,
            )
        )
    if participant_table.is_file():
        rows.append(
            _asset_row(
                root,
                participant_table,
                dataset="gstride",
                subset="participant_metadata",
                media_type="tabular",
                modality="participant_metadata",
                video_id=None,
                subject_id="unknown",
                source_group_id="gstride_participant_table",
                original_event_id="gstride_participant_table",
                scene_region="walking_lab",
                view=None,
                label_source="official_participant_table",
                annotation_path=None,
                probe=probe,
            )
        )
    return rows


def _adapt_ltmm(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/ltmm/raw"
    rows = []
    table_modalities = {
        "ClinicalDemogData_COFL.xlsx": "clinical_demographic_table",
        "ReportHome75h.xlsx": "home_monitoring_report",
    }
    for name, modality in table_modalities.items():
        path = dataset_root / name
        if not path.is_file():
            continue
        identity = f"ltmm_{_slug(path.stem)}"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="ltmm",
                subset="clinical_tables",
                media_type="tabular",
                modality=modality,
                video_id=None,
                subject_id="unknown",
                source_group_id="ltmm_clinical_tables",
                original_event_id=identity,
                scene_region="longitudinal_monitoring",
                view=None,
                label_source="official_clinical_table",
                annotation_path=None,
                probe=probe,
            )
        )

    for path in _files(dataset_root, "*.dat"):
        relative = path.relative_to(dataset_root)
        subject_id = _ltmm_subject_id(path.stem)
        source_group_id = (
            subject_id if subject_id != "unknown" else "ltmm_unresolved_records"
        )
        header = path.with_suffix(".hea")
        subset = "lab_walk" if relative.parts[0] == "LabWalks" else "home_monitoring"
        rows.append(
            _asset_row(
                root,
                path,
                dataset="ltmm",
                subset=subset,
                media_type="timeseries",
                modality="waist_accelerometer",
                video_id=None,
                subject_id=subject_id,
                source_group_id=source_group_id,
                original_event_id=_path_identity("ltmm_record", relative.with_suffix("")),
                scene_region="longitudinal_monitoring",
                view=None,
                label_source="official_wfdb_record",
                annotation_path=header if header.is_file() else None,
                probe=probe,
                extra_exclusion_reasons=([] if header.is_file() else ["header_missing"]),
            )
        )
    return rows


def _adapt_pre_vfallp(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/Pre_VFallp"
    rows = []
    for path in _files(dataset_root, "*.mp4"):
        relative = path.relative_to(dataset_root)
        subset = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        video_id = _path_identity("pre_vfallp_video", relative.with_suffix(""))
        rows.append(
            _asset_row(
                root,
                path,
                dataset="pre_vfallp",
                subset=subset,
                media_type="video",
                modality="rgb_video",
                video_id=video_id,
                subject_id="unknown",
                source_group_id="pre_vfallp_unresolved",
                original_event_id=video_id,
                scene_region="unknown",
                view="unknown",
                label_source="unverified_directory_semantics",
                annotation_path=None,
                probe=probe,
                extra_exclusion_reasons=["dataset_quarantined"],
            )
        )
    return rows


def _adapt_fall_tiktok(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/抖音b站跌倒视频整理"
    if not dataset_root.is_dir():
        return []

    source_map_path = root / _FALL_TIKTOK_SOURCE_MAP
    entries = load_fall_tiktok_source_map(source_map_path)
    decision_path = root / _FALL_TIKTOK_COLLECTION_DECISION
    decision = load_fall_tiktok_collection_decision(decision_path)
    redacted_cvat_path = root / _FALL_TIKTOK_REDACTED_CVAT
    clip_annotation_path = (
        redacted_cvat_path if redacted_cvat_path.is_file() else source_map_path
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        sequence = int(entry["sequence"])
        filename = str(entry["filename"])
        original_event_id = f"fall_tiktok_sample_{sequence:03d}"
        source_group_id = decision.source_group_id
        clip_video_id = f"fall_tiktok_clip_{sequence:03d}"
        clip_path = dataset_root / "annotated_clips" / filename
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)

        clip_row = _asset_row(
            root,
            clip_path,
            dataset="fall_tiktok",
            subset="annotated_clips",
            media_type="video",
            modality="rgb_video",
            video_id=clip_video_id,
            subject_id="unknown",
            source_group_id=source_group_id,
            original_event_id=original_event_id,
            scene_region="unknown",
            view="unknown",
            label_source="cvat_manual",
            annotation_path=clip_annotation_path,
            probe=probe,
        )
        _apply_fall_tiktok_collection_decision(
            clip_row, decision, decision_path=decision_path, root=root
        )
        clip_row.update(
            {
                "source_sequence": sequence,
                "original_filename": entry["original_filename"],
            }
        )
        rows.append(clip_row)
    return rows


def _apply_fall_tiktok_collection_decision(
    row: dict[str, Any],
    decision: FallTiktokCollectionDecision,
    *,
    decision_path: Path,
    root: Path,
) -> None:
    reasons = set(row["exclusion_reasons"])
    reasons.discard("source_unknown")
    row.update(
        {
            "source_uri": decision.source_uri,
            "eligibility": not reasons,
            "exclusion_reasons": sorted(reasons),
            "provenance_status": decision.provenance_status,
            "collection_status": decision.collection_status,
            "training_use": decision.training_use,
            "redistribution_use": decision.redistribution_use,
            "consent_status": decision.consent_status,
            "subject_grouping_status": decision.subject_grouping_status,
            "collection_decision_id": decision.decision_id,
            "collection_decision_path": _repo_relative(decision_path, root),
            "collection_decision_sha256": _sha256_file(decision_path),
            "collection_decided_at": decision.decided_at,
            "collection_decided_by": decision.decided_by,
        }
    )


def _adapt_caucafall(root: Path, probe: _VideoProbe) -> list[dict[str, Any]]:
    dataset_root = root / "data/external/caucafall/raw"
    rows: list[dict[str, Any]] = []
    canonical_actions = {
        "walk": "Walk",
        "hop": "Hop",
        "pickupobject": "Pickupobject",
        "sitdown": "SitDown",
        "kneel": "Kneel",
        "fallforward": "FallForward",
        "fallbackwards": "FallBackwards",
        "fallleft": "FallLeft",
        "fallright": "FallRight",
        "fallsitting": "FallSitting",
    }
    subset_by_action = {
        "Walk": "walk",
        "Hop": "hop",
        "Pickupobject": "pickup_object",
        "SitDown": "sit_down",
        "Kneel": "kneel",
        "FallForward": "fall_forward",
        "FallBackwards": "fall_backward",
        "FallLeft": "fall_left",
        "FallRight": "fall_right",
        "FallSitting": "fall_sitting",
    }
    for path in _files(dataset_root, "*.avi"):
        relative = path.relative_to(dataset_root)
        if len(relative.parts) != 2:
            raise ValueError(f"invalid CaucaFall video path: {relative.as_posix()}")
        subject_dir = re.fullmatch(r"Subject\.(\d+)", relative.parts[0], re.IGNORECASE)
        action_match = _CAUCAFALL_VIDEO_PATTERN.fullmatch(path.name)
        if subject_dir is None or action_match is None:
            raise ValueError(f"invalid CaucaFall video path: {relative.as_posix()}")
        directory_subject = int(subject_dir.group(1))
        filename_subject = int(action_match.group("subject"))
        if directory_subject != filename_subject:
            raise ValueError(
                "CaucaFall subject directory and filename disagree: "
                f"{relative.as_posix()}"
            )
        source_action = canonical_actions[action_match.group("action").lower()]
        subject_id = f"caucafall_s{directory_subject:02d}"
        video_id = f"{subject_id}_{subset_by_action[source_action]}"
        annotation = (
            root
            / "data/annotations/fall_risk/cvat_exports/raw/caucafall_manual"
            / f"CAUCAFalls{directory_subject}.zip"
        )
        row = _asset_row(
            root,
            path,
            dataset="caucafall",
            subset=subset_by_action[source_action],
            media_type="video",
            modality="rgb_video",
            video_id=video_id,
            subject_id=subject_id,
            source_group_id=subject_id,
            original_event_id=video_id,
            scene_region="uncontrolled_home",
            view="fixed_camera",
            label_source="cvat_manual" if annotation.is_file() else "unlabeled",
            annotation_path=annotation if annotation.is_file() else None,
            probe=probe,
        )
        row["source_action_code"] = source_action
        rows.append(row)
    return rows


def _load_internal_authorizations(
    root: Path,
) -> Mapping[str, _InternalAuthorization]:
    config_path = root / _INTERNAL_AUTHORIZATION_CONFIG
    if not config_path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid internal authorization config: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("internal authorization config must be an object")
    if loaded.get("schema_version") != _INTERNAL_AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported internal authorization config schema_version")
    entries = loaded.get("authorizations")
    if not isinstance(entries, list):
        raise ValueError("internal authorization config authorizations must be a list")

    authorizations: dict[str, _InternalAuthorization] = {}
    video_ids: set[str] = set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"internal authorization {index} must be an object")
        authorization_id = _required_internal_authorization_string(
            entry.get("authorization_id"), "authorization_id"
        )
        if authorization_id in authorizations:
            raise ValueError(f"duplicate internal authorization_id: {authorization_id}")
        approval_reference = _required_internal_authorization_string(
            entry.get("approval_reference"), "approval_reference"
        )
        approved_at = _required_internal_authorization_string(
            entry.get("approved_at"), "approved_at"
        )
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approved_at):
            raise ValueError("internal authorization approved_at must be YYYY-MM-DD")
        source_uri = _required_internal_authorization_string(
            entry.get("source_uri"), "source_uri"
        )
        if source_uri != f"{_INTERNAL_AUTHORIZATION_URI_PREFIX}{authorization_id}":
            raise ValueError("internal authorization source_uri does not match authorization_id")
        evidence = _load_internal_authorization_evidence(root, entry.get("evidence"))
        if evidence.kind == "cvat_export_archive":
            raw_video_ids = entry.get("video_ids")
            if not isinstance(raw_video_ids, list) or not raw_video_ids:
                raise ValueError(
                    "CVAT archive internal authorization video_ids must be a "
                    "non-empty list"
                )
            current_video_ids = frozenset(
                _required_internal_authorization_string(value, "video_id")
                for value in raw_video_ids
            )
            if len(current_video_ids) != len(raw_video_ids):
                raise ValueError("internal authorization video_ids must be unique")
        else:
            if "video_ids" in entry:
                raise ValueError(
                    "local media inventory authorization scope must be derived "
                    "from the inventory, not duplicated in video_ids"
                )
            assert evidence.inventory_records is not None
            current_video_ids = frozenset(evidence.inventory_records)
            if not current_video_ids:
                raise ValueError("local media inventory authorization is empty")
        duplicated = current_video_ids.intersection(video_ids)
        if duplicated:
            raise ValueError(
                "video_id is covered by multiple internal authorizations: "
                + ", ".join(sorted(duplicated))
            )
        video_ids.update(current_video_ids)
        authorizations[authorization_id] = _InternalAuthorization(
            authorization_id=authorization_id,
            approval_reference=approval_reference,
            approved_at=approved_at,
            source_uri=source_uri,
            evidence=evidence,
            video_ids=current_video_ids,
        )
    return authorizations


def _load_internal_authorization_evidence(
    root: Path,
    value: Any,
) -> _InternalAuthorizationEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("internal authorization evidence must be an object")
    kind = _required_internal_authorization_string(value.get("kind"), "evidence.kind")
    if kind not in _INTERNAL_AUTHORIZATION_EVIDENCE_TYPES:
        raise ValueError(f"unsupported internal authorization evidence kind: {kind}")
    name = _required_internal_authorization_string(value.get("name"), "evidence.name")
    sha256 = _required_internal_authorization_string(
        value.get("sha256"), "evidence.sha256"
    ).lower()
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("internal authorization evidence.sha256 is invalid")
    if kind == "cvat_export_archive":
        if any(key in value for key in ("path", "inventory_id")):
            raise ValueError("CVAT archive evidence must not declare local inventory fields")
        return _InternalAuthorizationEvidence(kind=kind, name=name, sha256=sha256)

    inventory_id = _required_internal_authorization_string(
        value.get("inventory_id"), "evidence.inventory_id"
    )
    inventory_path = _internal_authorization_relative_path(
        value.get("path"), "evidence.path"
    )
    absolute_inventory_path = root / inventory_path
    if not absolute_inventory_path.is_file():
        raise FileNotFoundError(
            "internal authorization media inventory does not exist: "
            + absolute_inventory_path.as_posix()
        )
    if _sha256_file(absolute_inventory_path) != sha256:
        raise ValueError("internal authorization media inventory checksum does not match")
    inventory_records = _load_pre_vfallp_media_inventory(
        root,
        absolute_inventory_path,
        inventory_id=inventory_id,
    )
    return _InternalAuthorizationEvidence(
        kind=kind,
        name=name,
        sha256=sha256,
        path=inventory_path,
        inventory_id=inventory_id,
        inventory_records=inventory_records,
    )


def _internal_authorization_relative_path(value: Any, field: str) -> str:
    path = Path(_required_internal_authorization_string(value, field))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"internal authorization {field} must be a safe relative path")
    return path.as_posix()


def _load_pre_vfallp_media_inventory(
    root: Path,
    path: Path,
    *,
    inventory_id: str,
) -> Mapping[str, _PreVFallpInventoryRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Pre_VFallp media inventory is not UTF-8: {path}") from exc
    if not lines:
        raise ValueError("Pre_VFallp media inventory must not be empty")

    records: dict[str, _PreVFallpInventoryRecord] = {}
    dataset_root = root / "data/external/Pre_VFallp"
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(
                f"Pre_VFallp media inventory contains a blank line at {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid Pre_VFallp media inventory JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"Pre_VFallp media inventory line {line_number} must be an object"
            )
        required = {
            "schema_version",
            "inventory_id",
            "dataset",
            "subset",
            "video_id",
            "path",
            "sha256",
            "byte_size",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                "Pre_VFallp media inventory is missing fields at line "
                f"{line_number}: {', '.join(missing)}"
            )
        if row["schema_version"] != _PRE_VFALLP_MEDIA_INVENTORY_SCHEMA:
            raise ValueError("unsupported Pre_VFallp media inventory schema_version")
        if row["inventory_id"] != inventory_id:
            raise ValueError("Pre_VFallp media inventory_id does not match authorization")
        if row["dataset"] != "pre_vfallp":
            raise ValueError("Pre_VFallp media inventory dataset must be pre_vfallp")
        video_id = _required_internal_authorization_string(row["video_id"], "video_id")
        if video_id in records:
            raise ValueError(f"duplicate Pre_VFallp media inventory video_id: {video_id}")
        relative_path = _internal_authorization_relative_path(row["path"], "path")
        media_path = root / relative_path
        try:
            media_relative = media_path.resolve().relative_to(dataset_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "Pre_VFallp media inventory path is outside data/external/Pre_VFallp: "
                + relative_path
            ) from exc
        if len(media_relative.parts) < 2 or not media_path.is_file():
            raise ValueError(
                "Pre_VFallp media inventory path does not identify a media file: "
                + relative_path
            )
        subset = _required_internal_authorization_string(row["subset"], "subset")
        if subset != media_relative.parts[0]:
            raise ValueError("Pre_VFallp media inventory subset does not match path")
        expected_video_id = _path_identity(
            "pre_vfallp_video", media_relative.with_suffix("")
        )
        if video_id != expected_video_id:
            raise ValueError("Pre_VFallp media inventory video_id does not match path")
        sha256 = _required_internal_authorization_string(row["sha256"], "sha256").lower()
        if not _SHA256_PATTERN.fullmatch(sha256):
            raise ValueError("Pre_VFallp media inventory sha256 is invalid")
        if _sha256_file(media_path) != sha256:
            raise ValueError("Pre_VFallp media inventory media checksum does not match")
        byte_size = row["byte_size"]
        if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 1:
            raise ValueError("Pre_VFallp media inventory byte_size must be positive")
        if media_path.stat().st_size != byte_size:
            raise ValueError("Pre_VFallp media inventory byte_size does not match")
        records[video_id] = _PreVFallpInventoryRecord(
            video_id=video_id,
            path=relative_path,
            sha256=sha256,
            byte_size=byte_size,
        )
    return records


def _required_internal_authorization_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"internal authorization {field} must be a non-empty string")
    return value.strip()


def _apply_internal_authorizations(
    rows: list[dict[str, Any]],
    authorizations: Mapping[str, _InternalAuthorization],
) -> None:
    if not authorizations:
        return
    by_video_id = {
        str(row["video_id"]): row
        for row in rows
        if isinstance(row.get("video_id"), str) and row["video_id"]
    }
    for authorization in authorizations.values():
        missing = authorization.video_ids.difference(by_video_id)
        if missing:
            raise ValueError(
                "internal authorization references missing manifest video_id(s): "
                + ", ".join(sorted(missing))
            )
        for video_id in sorted(authorization.video_ids):
            row = by_video_id[video_id]
            if row.get("dataset") != "pre_vfallp":
                raise ValueError(
                    "internal authorization can only override pre_vfallp assets: "
                    + video_id
                )
            remaining_reasons = set(row["exclusion_reasons"]).difference(
                {"dataset_quarantined", "source_unknown"}
            )
            if remaining_reasons:
                raise ValueError(
                    "internal authorization cannot override non-provenance exclusions "
                    f"for {video_id}: {', '.join(sorted(remaining_reasons))}"
                )
            if authorization.evidence.inventory_records is not None:
                inventory_record = authorization.evidence.inventory_records[video_id]
                if (
                    row["path"] != inventory_record.path
                    or row["sha256"] != inventory_record.sha256
                ):
                    raise ValueError(
                        "internal authorization media inventory does not match "
                        f"manifest asset: {video_id}"
                    )
            row["source_uri"] = authorization.source_uri
            row["eligibility"] = True
            row["exclusion_reasons"] = []
            row["provenance_status"] = "internal_authorized_source_unverified"
            authorization_record: dict[str, Any] = {
                "authorization_id": authorization.authorization_id,
                "approval_reference": authorization.approval_reference,
                "approved_at": authorization.approved_at,
                "evidence": {
                    "kind": authorization.evidence.kind,
                    "name": authorization.evidence.name,
                    "sha256": authorization.evidence.sha256,
                },
                "authorized_video_count": len(authorization.video_ids),
            }
            if authorization.evidence.path is not None:
                authorization_record["evidence"]["path"] = authorization.evidence.path
            if authorization.evidence.inventory_id is not None:
                authorization_record["evidence"]["inventory_id"] = (
                    authorization.evidence.inventory_id
                )
            row["internal_authorization"] = authorization_record


def _asset_row(
    root: Path,
    path: Path,
    *,
    dataset: str,
    subset: str,
    media_type: str,
    modality: str,
    video_id: str | None,
    subject_id: str,
    source_group_id: str,
    original_event_id: str,
    scene_region: str,
    view: str | None,
    label_source: str,
    annotation_path: Path | None,
    probe: _VideoProbe,
    extra_exclusion_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    relative_path = _repo_relative(path, root)
    provenance = _PROVENANCE[dataset]
    reasons = set(extra_exclusion_reasons)
    if provenance.source_uri is None:
        reasons.add("source_unknown")

    metadata: VideoMetadata | None = None
    if media_type == "video":
        try:
            metadata = probe(path)
            _validate_video_metadata(metadata)
        except (MediaProbeError, OSError, ValueError, TypeError):
            reasons.add("media_probe_failed")

    row: dict[str, Any] = {
        "asset_id": _asset_id(dataset, relative_path),
        "dataset": dataset,
        "subset": subset,
        "path": relative_path,
        "sha256": _sha256_file(path),
        "media_type": media_type,
        "modality": modality,
        "video_id": video_id,
        "fps_num": metadata.fps_num if metadata else None,
        "fps_den": metadata.fps_den if metadata else None,
        "fps": metadata.fps if metadata else None,
        "frame_count": metadata.frame_count if metadata else None,
        "duration_sec": metadata.duration_sec if metadata else None,
        "width": metadata.width if metadata else None,
        "height": metadata.height if metadata else None,
        "subject_id": subject_id,
        "source_group_id": source_group_id,
        "original_event_id": original_event_id,
        "scene_region": scene_region,
        "view": view,
        "label_source": label_source,
        "annotation_path": (
            _repo_relative(annotation_path, root) if annotation_path else None
        ),
        "source_uri": provenance.source_uri,
        "consent_id": None,
        "eligibility": not reasons,
        "exclusion_reasons": sorted(reasons),
        "duplicate_group_id": None,
        "duplicate_of_asset_id": None,
    }
    return row


def _ntu_rgbd_clip_asset_row(
    source_root: Path,
    path: Path,
    probe: _VideoProbe,
) -> dict[str, Any]:
    match = _NTU_RGBD_VIDEO_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid NTU RGB+D RGB filename: {path.name}")
    groups = match.groupdict()
    setup = int(groups["setup"])
    camera = int(groups["camera"])
    person = int(groups["person"])
    repetition = int(groups["repetition"])
    source_action_code = f"A{int(groups['action']):03d}"
    subject_id = f"ntu_rgbd_p{person:03d}"
    original_event_id = (
        f"ntu_rgbd_s{setup:03d}_p{person:03d}_r{repetition:03d}_"
        f"a{int(groups['action']):03d}"
    )
    video_id = f"{original_event_id}_c{camera:03d}"
    try:
        relative_path = path.resolve().relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"NTU RGB+D media path is outside source root: {path}") from exc

    reasons: set[str] = set()
    try:
        metadata = probe(path)
        _validate_video_metadata(metadata)
    except (MediaProbeError, OSError, ValueError, TypeError):
        metadata = None
        reasons.add("media_probe_failed")

    return {
        "asset_id": _asset_id("ntu_rgbd", relative_path),
        "dataset": "ntu_rgbd",
        "subset": f"setup_s{setup:03d}",
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "media_type": "video",
        "modality": "rgb_video",
        "video_id": video_id,
        "fps_num": metadata.fps_num if metadata else None,
        "fps_den": metadata.fps_den if metadata else None,
        "fps": metadata.fps if metadata else None,
        "frame_count": metadata.frame_count if metadata else None,
        "duration_sec": metadata.duration_sec if metadata else None,
        "width": metadata.width if metadata else None,
        "height": metadata.height if metadata else None,
        "subject_id": subject_id,
        "source_group_id": f"ntu_rgbd_subject_p{person:03d}",
        "original_event_id": original_event_id,
        "scene_region": "unknown",
        "view": f"c{camera:03d}",
        "label_source": "official_filename_action_code",
        "annotation_path": None,
        "source_uri": _PROVENANCE["ntu_rgbd"].source_uri,
        "consent_id": None,
        "eligibility": not reasons,
        "exclusion_reasons": sorted(reasons),
        "duplicate_group_id": None,
        "duplicate_of_asset_id": None,
        "source_action_code": source_action_code,
    }


def _mark_duplicate_content(rows: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["sha256"])].append(row)
    for checksum, duplicates in by_hash.items():
        if len(duplicates) < 2:
            continue
        duplicates.sort(key=lambda row: str(row["asset_id"]))
        group_id = f"duplicate_sha256_{checksum[:24]}"
        canonical_id = str(duplicates[0]["asset_id"])
        for index, row in enumerate(duplicates):
            reasons = set(row["exclusion_reasons"])
            reasons.add("duplicate_content")
            row["exclusion_reasons"] = sorted(reasons)
            row["eligibility"] = False
            row["duplicate_group_id"] = group_id
            row["duplicate_of_asset_id"] = canonical_id if index else None


def _load_fall_2017_metadata(
    path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, ["metadata_missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("subjectId", "locationId", "actionId", "side", "attempt"):
            if key not in payload:
                raise KeyError(key)
        int(payload["subjectId"])
        int(payload["locationId"])
        int(payload["actionId"])
        int(payload["attempt"])
        if not str(payload["side"]).strip():
            raise ValueError("empty side")
        return payload, []
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, ["metadata_invalid"]


def _fall_2017_path_mismatch_reasons(
    relative: Path, metadata: dict[str, Any]
) -> list[str]:
    if len(relative.parts) < 4:
        return ["metadata_path_mismatch"]
    subject_match = re.fullmatch(r"SBJ_(\d+)_LOC(\d+)", relative.parts[-3])
    action_match = re.fullmatch(r"ACT(\d+)_([BFLR])_(\d+)", relative.parts[-2])
    if subject_match is None or action_match is None:
        return ["metadata_path_mismatch"]
    expected = (
        int(subject_match.group(1)),
        int(subject_match.group(2)),
        int(action_match.group(1)),
        action_match.group(2),
        int(action_match.group(3)),
    )
    actual = (
        int(metadata["subjectId"]),
        int(metadata["locationId"]),
        int(metadata["actionId"]),
        str(metadata["side"]),
        int(metadata["attempt"]),
    )
    return [] if expected == actual else ["metadata_path_mismatch"]


def _count_video_frames(
    path: Path | str,
    *,
    ffprobe_bin: str,
    runner: Callable[..., Any],
) -> int:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise MediaProbeError("ffprobe frame count failed")
    try:
        payload = json.loads(result.stdout)
        frame_count = _positive_int(payload["streams"][0].get("nb_read_frames"))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise MediaProbeError("ffprobe returned an invalid frame count") from exc
    if frame_count is None:
        raise MediaProbeError("ffprobe did not return a frame count")
    return frame_count


def _parse_frame_rate(value: Any) -> Fraction | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return float(parsed)


def _validate_video_metadata(metadata: VideoMetadata) -> None:
    if metadata.fps_num <= 0 or metadata.fps_den <= 0 or metadata.fps <= 0:
        raise ValueError("invalid frame rate")
    if metadata.frame_count <= 0 or metadata.duration_sec <= 0:
        raise ValueError("invalid frame count or duration")
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError("invalid dimensions")


def _files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob(pattern) if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _repo_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
        relative = path.absolute().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("manifest paths must remain inside the repository") from exc
    return relative.as_posix()


def _asset_id(dataset: str, relative_path: str) -> str:
    digest = hashlib.sha256(
        f"{dataset}\0{relative_path}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{dataset}_{digest}"


def _unique_video_id(event_id: str, relative: Path) -> str:
    path_digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{event_id}_{_slug(relative.stem)}_{path_digest}"


def _path_identity(prefix: str, path: Path) -> str:
    normalized = path.as_posix()
    readable = _slug(normalized)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{readable}_{digest}"


def _ltmm_subject_id(stem: str) -> str:
    match = re.match(r"([A-Za-z]+)(\d+)", stem)
    if match is None:
        return "unknown"
    return f"ltmm_{match.group(1).lower()}_{match.group(2)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _build_summary(
    rows: list[dict[str, Any]], manifest_sha256: str
) -> dict[str, Any]:
    dataset_counts = Counter(str(row["dataset"]) for row in rows)
    exclusion_counts = Counter(
        reason for row in rows for reason in row["exclusion_reasons"]
    )
    return {
        "version": "fall-risk-data-v2-candidate",
        "manifest_sha256": manifest_sha256,
        "asset_count": len(rows),
        "video_count": sum(row["media_type"] == "video" for row in rows),
        "eligible_count": sum(bool(row["eligibility"]) for row in rows),
        "ineligible_count": sum(not bool(row["eligibility"]) for row in rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
    }


def _atomic_write(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
            temporary_path = None
        else:
            os.link(temporary_path, path)
            temporary_path.unlink()
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
