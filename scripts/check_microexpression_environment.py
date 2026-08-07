from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PACKAGE_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "opencv": "cv2",
    "pillow": "PIL",
    "einops": "einops",
    "networkx": "networkx",
    "dlib": "dlib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the micro-expression Python and CUDA environment."
    )
    parser.add_argument("--github-source", type=Path)
    parser.add_argument("--chapter3-source", type=Path)
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, module_name in PACKAGE_MODULES.items():
        module = importlib.import_module(module_name)
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return versions


def run_numeric_checks() -> dict[str, Any]:
    import cv2
    import dlib
    import importlib.util
    import numpy as np
    import torch
    from einops import rearrange
    from sklearn.manifold import MDS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    tensor = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    product = (tensor @ tensor.T).cpu()

    frame = np.zeros((32, 32), dtype=np.uint8)
    shifted = np.roll(frame, 1, axis=1)
    flow = cv2.calcOpticalFlowFarneback(
        frame, shifted, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    embedded = MDS(
        n_components=2, random_state=0, n_init=1, max_iter=2, init="random"
    ).fit_transform(np.arange(15, dtype=np.float32).reshape(5, 3))
    rearranged = rearrange(
        np.arange(24).reshape(6, 4), "(h w) c -> h w c", h=2, w=3
    )
    models_spec = importlib.util.find_spec("face_recognition_models")
    if not models_spec or not models_spec.submodule_search_locations:
        raise RuntimeError("face-recognition-models is not installed")
    predictor = (
        Path(next(iter(models_spec.submodule_search_locations)))
        / "models/shape_predictor_68_face_landmarks.dat"
    )
    if not predictor.is_file():
        raise RuntimeError(f"Missing dlib predictor asset: {predictor}")

    return {
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_tensor_shape": list(product.shape),
        "gpu_tensor_sum": float(product.sum()),
        "opencv_flow_shape": list(flow.shape),
        "dlib_detector": type(dlib.get_frontal_face_detector()).__name__,
        "dlib_68_predictor": str(predictor.resolve()),
        "mds_shape": list(embedded.shape),
        "einops_shape": list(rearranged.shape),
    }


def import_source_modules(source_root: Path, modules: list[str]) -> list[str]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    original_cwd = Path.cwd()
    sys.path.insert(0, str(source_root))
    try:
        os.chdir(source_root)
        imported = []
        for module_name in modules:
            importlib.import_module(module_name)
            imported.append(module_name)
        return imported
    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(source_root))


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "pass",
        "python": sys.version.split()[0],
        "packages": {},
        "checks": {},
        "source_imports": {},
        "warnings": [],
        "errors": [],
    }

    try:
        result["packages"] = package_versions()
        result["checks"] = run_numeric_checks()

        if args.github_source:
            result["source_imports"]["github"] = import_source_modules(
                args.github_source, ["model"]
            )
        if args.chapter3_source:
            result["source_imports"]["chapter3"] = import_source_modules(
                args.chapter3_source,
                [
                    "config",
                    "models.model",
                    "models.layers",
                    "models.gcn",
                    "data.dataset",
                    "utils.adjacency",
                    "utils.metrics",
                    "train",
                ],
            )

        for source_root in (args.github_source, args.chapter3_source):
            if source_root:
                missing_assets = [
                    name
                    for name in (
                        "mmod_human_face_detector.dat",
                        "shape_predictor_68_face_landmarks.dat",
                    )
                    if not (source_root / name).is_file()
                ]
                if missing_assets:
                    result["warnings"].append(
                        {
                            "source": str(source_root.resolve()),
                            "missing_preprocess_assets": missing_assets,
                        }
                    )
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        result["status"] = "fail"
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
