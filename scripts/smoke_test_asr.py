"""Run a local FunASR pipeline smoke test against one audio file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from funasr import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "asr" / "asr-paraformer-zh-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local Paraformer, FSMN-VAD, and CT-Punc inference."
    )
    parser.add_argument("audio", type=Path, help="WAV or other supported audio file")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Directory containing paraformer/, fsmn-vad/, and ct-punc/",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda:0"),
        help="Inference device; auto selects CUDA when available",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument("--batch-size-s", type=int, default=60)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> int:
    args = parse_args()
    audio = args.audio.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    device = (
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    require_file(audio, "Audio file")
    model_paths = {
        "paraformer": model_root / "paraformer",
        "fsmn_vad": model_root / "fsmn-vad",
        "ct_punc": model_root / "ct-punc",
    }
    for name, directory in model_paths.items():
        require_file(directory / "config.yaml", f"{name} config")
        require_file(directory / "model.pt", f"{name} weights")

    print(f"device={device}")
    if device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"audio={audio}")
    print(f"model_root={model_root}")

    started = time.perf_counter()
    model = AutoModel(
        model=str(model_paths["paraformer"]),
        vad_model=str(model_paths["fsmn_vad"]),
        punc_model=str(model_paths["ct_punc"]),
        device=device,
        disable_update=True,
    )
    loaded_seconds = time.perf_counter() - started

    inference_started = time.perf_counter()
    result = model.generate(
        input=str(audio),
        cache={},
        batch_size_s=args.batch_size_s,
        merge_vad=True,
        merge_length_s=15,
        pred_timestamp=True,
        sentence_timestamp=True,
    )
    inference_seconds = time.perf_counter() - inference_started

    payload = {
        "runtime": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "model_load_seconds": round(loaded_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
        },
        "audio": str(audio),
        "models": {name: str(path) for name, path in model_paths.items()},
        "result": json_safe(result),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"output={output}")
    else:
        # ASCII console output avoids Windows/Conda failures on GBK terminals.
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ASR smoke test failed: {exc}", file=sys.stderr)
        raise
