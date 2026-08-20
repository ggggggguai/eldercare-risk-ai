#!/bin/sh
set -eu

if [ -z "${ALGORITHM_API_TOKEN:-}" ] || [ "${ALGORITHM_API_TOKEN}" = "change-me" ]; then
  echo "ALGORITHM_API_TOKEN must be supplied by the deployment environment" >&2
  exit 2
fi

python /app/scripts/deploy/verify_asr_assets.py \
  --project-root /app \
  --manifest /app/deploy/asr/asset_manifest.json \
  --verify-native

exec uvicorn elderly_monitoring.modules.asr.api:app \
  --host 0.0.0.0 \
  --port "${PORT:-8011}" \
  --workers 1
