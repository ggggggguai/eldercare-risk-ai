from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable


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


class MediaProbeError(ValueError):
    pass


@dataclass(frozen=True)
class _Provenance:
    source_uri: str | None
    license_id: str | None


_PROVENANCE = {
    "le2i_imvia": _Provenance(
        source_uri=(
            "https://search-data.ubfc.fr/imvia/"
            "FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html"
        ),
        license_id=None,
    ),
    "fall_detection_2017": _Provenance(
        source_uri="https://doi.org/10.6084/m9.figshare.28596332.v2",
        license_id="CC-BY-4.0",
    ),
    "ur_fall": _Provenance(
        source_uri="https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        license_id="CC-BY-NC-SA-4.0",
    ),
    "toaga": _Provenance(
        source_uri=(
            "https://springernature.figshare.com/collections/"
            "The_Toronto_Older_Adults_Gait_Archive_Video_and_3D_"
            "Inertial_Motion_Capture_Data_of_Older_Adults_Walking/5515953"
        ),
        license_id=None,
    ),
    "gstride": _Provenance(
        source_uri="https://doi.org/10.5281/zenodo.17052815",
        license_id="CC-BY-4.0",
    ),
    "ltmm": _Provenance(
        source_uri="https://physionet.org/content/ltmm/1.0.0/",
        license_id="ODC-BY-1.0",
    ),
    "pre_vfallp": _Provenance(source_uri=None, license_id=None),
}


_VideoProbe = Callable[[Path], VideoMetadata]


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
    rows: list[dict[str, Any]] = []
    adapters = (
        _adapt_le2i,
        _adapt_fall_detection_2017,
        _adapt_ur_fall,
        _adapt_toaga,
        _adapt_gstride,
        _adapt_ltmm,
        _adapt_pre_vfallp,
    )
    for adapter in adapters:
        rows.extend(adapter(root, probe))

    _mark_duplicate_content(rows)
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
    if provenance.license_id is None:
        reasons.add("license_unknown")

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
        "license_id": provenance.license_id,
        "consent_id": None,
        "review_status": "pending",
        "eligibility": not reasons,
        "exclusion_reasons": sorted(reasons),
        "duplicate_group_id": None,
        "duplicate_of_asset_id": None,
    }
    return row


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
        "version": "fall-risk-data-v1-candidate",
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
