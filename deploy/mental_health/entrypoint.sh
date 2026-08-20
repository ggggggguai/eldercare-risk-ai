#!/bin/sh
set -eu

if [ -z "${ALGORITHM_API_TOKEN:-}" ] || [ "${ALGORITHM_API_TOKEN}" = "change-me" ]; then
  echo "ALGORITHM_API_TOKEN must be supplied by the deployment environment" >&2
  exit 2
fi

if [ -z "${COGNITIVE_ASR_URL:-}" ]; then
  echo "COGNITIVE_ASR_URL must point to the private ASR service" >&2
  exit 2
fi

python /app/scripts/deploy/verify_mental_health_assets.py \
  --project-root /app \
  --manifest /app/deploy/mental_health/asset_manifest.json

exec uvicorn elderly_monitoring.service.mental_health_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1
