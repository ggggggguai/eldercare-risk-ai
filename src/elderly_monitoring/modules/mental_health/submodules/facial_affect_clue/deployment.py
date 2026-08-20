from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .causalnet_opt_me_006_training import load_me6_checkpoint


B0_DEPLOYMENT_MANIFEST_SCHEMA = "facial_affect_b0_deployment_manifest_v1"
B0_DEPLOYMENT_MODEL_VERSION = "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
B0_EXPECTED_RUN_LOCK_SHA256 = (
    "e1266924e54631a05ddc90a4d77d5457c8d9acacfce348ccbb49e3f00c536bb2"
)
B0_CHECKPOINT_SCHEMA = "causalnet_opt_me_006_checkpoint_v1"
B0_CLASS_ORDER = ("negative", "positive", "surprise")
B0_ENSEMBLE_RULE = "mean_of_fold_temperature_calibrated_probabilities"
B0_REQUIRED_MEMBERS = {
    (seed, repeat_index, fold_index)
    for seed in (28081101, 28081119)
    for repeat_index in (0, 1)
    for fold_index in range(5)
}


class B0DeploymentError(RuntimeError):
    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_sha256(
    record: Mapping[str, Any], *, self_hash_field: str
) -> str:
    payload = dict(record)
    payload.pop(self_hash_field, None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_record_sha256(manifest, self_hash_field="manifest_sha256")


def canonical_routes(routes: Any) -> np.ndarray:
    try:
        array = np.asarray(routes, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise B0DeploymentError(
            "routes must contain numeric float32-compatible values",
            code="INVALID_INPUT",
            status_code=422,
        ) from exc
    if array.shape != (4, 3, 28, 28):
        raise B0DeploymentError(
            f"routes shape must be [4,3,28,28], got {list(array.shape)}",
            code="INVALID_INPUT",
            status_code=422,
        )
    if not np.isfinite(array).all():
        raise B0DeploymentError(
            "routes contain NaN or Inf",
            code="INVALID_INPUT",
            status_code=422,
        )
    if np.any(array[:2, :2] < -1.0) or np.any(array[:2, :2] > 1.0):
        raise B0DeploymentError(
            "flow u/v channels must be within [-1,1]",
            code="INVALID_INPUT",
            status_code=422,
        )
    if np.any(array[:2, 2] < 0.0) or np.any(array[:2, 2] > 1.0):
        raise B0DeploymentError(
            "flow strain channels must be within [0,1]",
            code="INVALID_INPUT",
            status_code=422,
        )
    if np.any(array[2:] < 0.0) or np.any(array[2:] > 1.0):
        raise B0DeploymentError(
            "direction routes must be within [0,1]",
            code="INVALID_INPUT",
            status_code=422,
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def routes_sha256(routes: np.ndarray) -> str:
    return hashlib.sha256(routes.tobytes(order="C")).hexdigest()


ModelLoader = Callable[..., tuple[torch.nn.Module, dict[str, Any]]]


class B0DeploymentRuntime:
    """Strict, lazy runtime for the frozen 20-member B0 deployment package."""

    def __init__(
        self,
        package_path: Path | str,
        *,
        device: str = "cpu",
        model_loader: ModelLoader = load_me6_checkpoint,
    ) -> None:
        self.package_path = Path(package_path)
        if device not in {"auto", "cpu", "cuda:0"}:
            raise ValueError("device must be auto, cpu, or cuda:0")
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu"
            if device == "auto"
            else device
        )
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise B0DeploymentError(
                "CUDA inference was requested but CUDA is unavailable",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        self.model_loader = model_loader
        self._manifest: dict[str, Any] | None = None
        self._members: list[tuple[torch.nn.Module, float, float]] | None = None
        self._deterministic_probe: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def verify_package(self) -> dict[str, Any]:
        with self._lock:
            if self._manifest is not None:
                return dict(self._manifest)
            manifest_path = self.package_path / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise B0DeploymentError(
                    f"frozen B0 manifest is unavailable: {manifest_path}",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                ) from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise B0DeploymentError(
                    "frozen B0 manifest cannot be read",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                ) from exc
            if not isinstance(manifest, dict):
                raise B0DeploymentError(
                    "frozen B0 manifest must be an object",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            self._validate_manifest(manifest)
            self._validate_freeze_audit(manifest)
            self._manifest = manifest
            return dict(manifest)

    def verify_deterministic_probe(self) -> dict[str, Any]:
        """Load the frozen ensemble once and prove repeatable canonical inference."""
        with self._lock:
            manifest = self._manifest or self.verify_package()
            if (
                self._deterministic_probe is not None
                and self._deterministic_probe.get("package_sha256")
                == manifest["manifest_sha256"]
            ):
                return dict(self._deterministic_probe)
            routes = np.zeros((4, 3, 28, 28), dtype=np.float32)
            first = self.predict(routes)
            second = self.predict(routes)
            first_probabilities = np.asarray(
                [first["probabilities"][label] for label in B0_CLASS_ORDER],
                dtype=np.float64,
            )
            second_probabilities = np.asarray(
                [second["probabilities"][label] for label in B0_CLASS_ORDER],
                dtype=np.float64,
            )
            if (
                first["prediction"] != second["prediction"]
                or first["input_sha256"] != second["input_sha256"]
                or not np.array_equal(first_probabilities, second_probabilities)
            ):
                raise B0DeploymentError(
                    "frozen B0 deterministic probe mismatch",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            self._deterministic_probe = {
                "status": "passed",
                "package_sha256": manifest["manifest_sha256"],
                "input_sha256": first["input_sha256"],
                "prediction": first["prediction"],
                "probabilities": first["probabilities"],
            }
            return dict(self._deterministic_probe)

    def _validate_freeze_audit(self, manifest: Mapping[str, Any]) -> None:
        audit_path = self.package_path / "freeze_audit.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise B0DeploymentError(
                "frozen B0 freeze audit is unavailable or invalid",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            ) from exc
        if (
            not isinstance(audit, dict)
            or audit.get("schema_version") != "facial_affect_b0_freeze_audit_v1"
            or audit.get("status") != "passed"
            or audit.get("error_count") != 0
            or audit.get("manifest_sha256") != manifest.get("manifest_sha256")
            or audit.get("checkpoint_file_count") != 20
            or audit.get("checkpoint_payload_audit_count") != 20
            or audit.get("run_lock_sha256") != B0_EXPECTED_RUN_LOCK_SHA256
            or audit.get("outer_test_metrics_accessed") is not False
            or audit.get("best_seed_or_fold_selected") is not False
        ):
            raise B0DeploymentError(
                "frozen B0 freeze audit contract drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        if audit.get("audit_sha256") != canonical_record_sha256(
            audit, self_hash_field="audit_sha256"
        ):
            raise B0DeploymentError(
                "frozen B0 freeze audit self hash mismatch",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": B0_DEPLOYMENT_MANIFEST_SCHEMA,
            "model_version": B0_DEPLOYMENT_MODEL_VERSION,
            "status": "frozen",
            "candidate_id": "B0",
            "checkpoint_schema_version": B0_CHECKPOINT_SCHEMA,
            "checkpoint_count": 20,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise B0DeploymentError(
                    f"frozen B0 manifest field drift: {key}",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
        if tuple(manifest.get("class_order", ())) != B0_CLASS_ORDER:
            raise B0DeploymentError(
                "frozen B0 class order drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        ensemble = manifest.get("ensemble")
        if not isinstance(ensemble, dict) or ensemble.get("rule") != B0_ENSEMBLE_RULE:
            raise B0DeploymentError(
                "frozen B0 ensemble rule drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        if ensemble.get("weighting") != "equal" or ensemble.get("member_count") != 20:
            raise B0DeploymentError(
                "frozen B0 ensemble member contract drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        lineage = manifest.get("lineage")
        if not isinstance(lineage, dict):
            raise B0DeploymentError(
                "frozen B0 lineage is missing",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        if lineage.get("run_lock_sha256") != B0_EXPECTED_RUN_LOCK_SHA256:
            raise B0DeploymentError(
                "frozen B0 run-lock hash drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        if (
            lineage.get("completion_audit_status") != "passed"
            or lineage.get("completion_audit_error_count") != 0
        ):
            raise B0DeploymentError(
                "frozen B0 completion audit did not pass with zero errors",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        recorded_hash = manifest.get("manifest_sha256")
        if recorded_hash != canonical_manifest_sha256(manifest):
            raise B0DeploymentError(
                "frozen B0 manifest self hash mismatch",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 20:
            raise B0DeploymentError(
                "frozen B0 package must contain exactly 20 checkpoint records",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )
        identities: set[tuple[int, int, int]] = set()
        package_root = self.package_path.resolve()
        for member in checkpoints:
            if not isinstance(member, dict):
                raise B0DeploymentError(
                    "frozen B0 checkpoint record is invalid",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            identity = (
                int(member.get("seed", -1)),
                int(member.get("repeat_index", -1)),
                int(member.get("fold_index", -1)),
            )
            identities.add(identity)
            expected_fold_id = (
                f"development_repeat_{identity[1]:02d}_fold_{identity[2]:02d}"
            )
            if member.get("fold_id") != expected_fold_id:
                raise B0DeploymentError(
                    "frozen B0 fold identity drift",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            temperature = float(member.get("temperature", 0.0))
            weight = float(member.get("weight", 0.0))
            if not np.isfinite(temperature) or temperature <= 0.0:
                raise B0DeploymentError(
                    "frozen B0 checkpoint temperature must be positive",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            if abs(weight - 0.05) > 1e-12:
                raise B0DeploymentError(
                    "frozen B0 checkpoint weights must be exactly equal",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            path = (self.package_path / str(member.get("path", ""))).resolve()
            if not path.is_relative_to(package_root):
                raise B0DeploymentError(
                    "frozen B0 checkpoint path escapes package",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            if not path.is_file() or path.stat().st_size != int(member.get("size_bytes", -1)):
                raise B0DeploymentError(
                    f"frozen B0 checkpoint size mismatch: {path.name}",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
            if sha256_file(path) != member.get("sha256"):
                raise B0DeploymentError(
                    f"frozen B0 checkpoint hash mismatch: {path.name}",
                    code="MODEL_UNAVAILABLE",
                    status_code=503,
                )
        if identities != B0_REQUIRED_MEMBERS:
            raise B0DeploymentError(
                "frozen B0 package member matrix drift",
                code="MODEL_UNAVAILABLE",
                status_code=503,
            )

    def _load_members(self) -> list[tuple[torch.nn.Module, float, float]]:
        with self._lock:
            if self._members is not None:
                return self._members
            manifest = self._manifest or self.verify_package()
            members: list[tuple[torch.nn.Module, float, float]] = []
            for record in manifest["checkpoints"]:
                path = self.package_path / record["path"]
                try:
                    model, payload = self.model_loader(path, map_location=self.device)
                except Exception as exc:
                    raise B0DeploymentError(
                        f"frozen B0 checkpoint cannot be loaded: {path.name}",
                        code="MODEL_UNAVAILABLE",
                        status_code=503,
                    ) from exc
                if (
                    payload.get("schema_version") != B0_CHECKPOINT_SCHEMA
                    or payload.get("candidate_id") != "B0"
                    or int(payload.get("parameter_count", -1)) != 623963
                    or payload.get("ema_state_dict") is not None
                    or payload.get("evidence_hashes", {}).get("run_lock.json")
                    != B0_EXPECTED_RUN_LOCK_SHA256
                ):
                    raise B0DeploymentError(
                        f"frozen B0 checkpoint metadata drift: {path.name}",
                        code="MODEL_UNAVAILABLE",
                        status_code=503,
                    )
                model = model.to(self.device)
                model.eval()
                members.append(
                    (model, float(record["temperature"]), float(record["weight"]))
                )
            self._members = members
            return members

    def predict(self, routes: Any, *, input_sha256: str | None = None) -> dict[str, Any]:
        array = canonical_routes(routes)
        actual_input_sha256 = routes_sha256(array)
        if input_sha256 is not None and input_sha256.lower() != actual_input_sha256:
            raise B0DeploymentError(
                "input_sha256 does not match canonical float32 routes",
                code="INVALID_INPUT",
                status_code=422,
            )
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        ensemble = torch.zeros(3, dtype=torch.float64, device=self.device)
        try:
            with self._lock, torch.inference_mode():
                for model, temperature, weight in self._load_members():
                    logits = model(tensor)
                    if logits.shape != (1, 3) or not torch.isfinite(logits).all():
                        raise RuntimeError("invalid logits")
                    ensemble += torch.softmax(logits[0] / temperature, dim=0).to(
                        dtype=torch.float64
                    ) * weight
        except B0DeploymentError:
            raise
        except Exception as exc:
            raise B0DeploymentError(
                "frozen B0 inference failed",
                code="INFERENCE_FAILED",
                status_code=500,
            ) from exc
        probabilities = ensemble.detach().cpu().numpy()
        probabilities = probabilities / probabilities.sum()
        prediction_index = int(np.argmax(probabilities))
        manifest = self._manifest or self.verify_package()
        return {
            "model_version": B0_DEPLOYMENT_MODEL_VERSION,
            "prediction": B0_CLASS_ORDER[prediction_index],
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(B0_CLASS_ORDER)
            },
            "confidence": float(probabilities[prediction_index]),
            "input_sha256": actual_input_sha256,
            "package_sha256": manifest["manifest_sha256"],
            "checkpoint_count": 20,
            "ensemble_rule": B0_ENSEMBLE_RULE,
        }

    def close(self) -> None:
        with self._lock:
            self._members = None
            self._deterministic_probe = None
            if self.device == "cuda:0" and torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = [
    "B0_CHECKPOINT_SCHEMA",
    "B0_CLASS_ORDER",
    "B0_DEPLOYMENT_MANIFEST_SCHEMA",
    "B0_DEPLOYMENT_MODEL_VERSION",
    "B0_ENSEMBLE_RULE",
    "B0_EXPECTED_RUN_LOCK_SHA256",
    "B0DeploymentError",
    "B0DeploymentRuntime",
    "canonical_manifest_sha256",
    "canonical_record_sha256",
    "canonical_routes",
    "routes_sha256",
    "sha256_file",
]
