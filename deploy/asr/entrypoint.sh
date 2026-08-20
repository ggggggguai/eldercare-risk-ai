#!/bin/sh
set -eu

python /app/scripts/deploy/verify_asr_assets.py \
  --project-root /app \
  --manifest /app/deploy/asr/asset_manifest.json \
  --verify-native

exec uvicorn elderly_monitoring.modules.asr.api:app \
  --host 0.0.0.0 \
  --port "${PORT:-8081}" \
  --workers 1
