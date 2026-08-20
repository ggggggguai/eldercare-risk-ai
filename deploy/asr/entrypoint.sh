#!/bin/sh
set -eu

MODEL_ROOT="${ASR_MODEL_ROOT:-/app/models/asr/asr-paraformer-zh-v1.0}"
SERVER_PORT="${PORT:-8081}"

echo "[entrypoint] uid=$(id -u)"
echo "[entrypoint] PORT=${SERVER_PORT}"
echo "[entrypoint] ASR_DEVICE=${ASR_DEVICE:-cpu}"
echo "[entrypoint] ASR_MODEL_ROOT=${MODEL_ROOT}"
echo "[entrypoint] checking model mount..."

if [ ! -d "${MODEL_ROOT}" ]; then
  echo "[entrypoint] ERROR: model directory does not exist: ${MODEL_ROOT}" >&2
  echo "[entrypoint] directories visible under /app/models:" >&2
  find /app/models -maxdepth 4 -type d -print 2>/dev/null || true
  exit 20
fi

if [ ! -r "${MODEL_ROOT}" ]; then
  echo "[entrypoint] ERROR: model directory is not readable by uid $(id -u)" >&2
  exit 21
fi

echo "[entrypoint] verifying native assets..."

python /app/scripts/deploy/verify_asr_assets.py \
  --project-root /app \
  --manifest /app/deploy/asr/asset_manifest.json \
  --verify-native

echo "[entrypoint] native asset verification passed"
echo "[entrypoint] starting uvicorn on 0.0.0.0:${SERVER_PORT}"

exec python -m uvicorn elderly_monitoring.modules.asr.api:app \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --workers 1 \
  --log-level debug
