FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MENTAL_HEALTH_ASSET_MANIFEST=/app/deploy/mental_health/asset_manifest.json \
    COGNITIVE_MODEL_PACKAGE_PATH=/app/models/mental_health/cognitive_change_clue/v3.3.0 \
    COGNITIVE_MODEL_PACKAGE_V35_PATH=/app/models/mental_health/cognitive_change_clue/v3.5.0 \
    FACIAL_AFFECT_B0_PACKAGE_PATH=/app/models/mental_health/facial_affect/b0_opt_me_008_deployment_v1 \
    FACIAL_AFFECT_SPOTTING_CONFIG_PATH=/app/configs/modules/facial_affect_spotting_v1.json \
    MOOD_SOCIAL_FORECAST_PACKAGE_DIR=/app/models/mental_health/mood_social/v3.4.0/packages/MH-20260810-FDEP-001 \
    FACIAL_AFFECT_VIDEO_TEMP_ROOT=/tmp/eldercare-facial-affect

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mental-health-deploy.lock ./
RUN python -m pip install --no-cache-dir --retries 8 -r requirements-mental-health-deploy.lock \
    && python -m pip install --no-cache-dir --no-deps ultralytics==8.4.78

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY deploy ./deploy
COPY third_party ./third_party
COPY THIRD_PARTY_NOTICES_MICROEXPRESSION.md ./

RUN python -m pip install --no-cache-dir --no-deps . \
    && python scripts/deploy/verify_mental_health_assets.py --manifest-only \
    && useradd --create-home --uid 10001 algorithm \
    && chmod +x /app/deploy/mental_health/entrypoint.sh \
    && mkdir -p /tmp/eldercare-facial-affect \
    && chown -R algorithm:algorithm /tmp/eldercare-facial-affect

USER algorithm
EXPOSE 8080

ENTRYPOINT ["/app/deploy/mental_health/entrypoint.sh"]
