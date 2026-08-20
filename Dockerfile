FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PORT=8081 \
    ASR_PORT=8081 \
    ASR_MODEL_ROOT=/app/models/asr/asr-paraformer-zh-v1.0 \
    ASR_DEVICE=cuda:0 \
    ASR_NATIVE_ASSETS_MANIFEST=/app/configs/runtime/asr-native-assets.container.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

COPY requirements-asr-deploy.lock ./

RUN python -m pip install \
      --no-cache-dir \
      --no-compile \
      --retries 10 \
      --timeout 120 \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      -r requirements-asr-deploy.lock \
    && rm -rf /root/.cache /tmp/*

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY deploy ./deploy

RUN python -m pip install \
      --no-cache-dir \
      --no-compile \
      --no-deps \
      --retries 10 \
      --timeout 120 \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      . \
    && python scripts/freeze_asr_native_assets.py \
       --output /app/configs/runtime/asr-native-assets.container.json \
    && python scripts/deploy/verify_asr_assets.py \
       --manifest /app/deploy/asr/asset_manifest.json \
       --manifest-only \
    && useradd --create-home --uid 10001 asr \
    && chmod +x /app/deploy/asr/entrypoint.sh \
    && rm -rf /root/.cache /tmp/*

USER asr

EXPOSE 8081

ENTRYPOINT ["/app/deploy/asr/entrypoint.sh"]
