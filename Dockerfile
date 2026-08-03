FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/yolov8n-pose.pt \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

RUN python -m pip install \
      --no-cache-dir \
      --no-compile \
      --retries 10 \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
      "torch==2.6.0+cpu" \
      "torchvision==0.21.0+cpu" \
    && rm -rf \
      /usr/local/lib/python3.11/site-packages/torch/include \
      /usr/local/lib/python3.11/site-packages/torch/share/cmake \
      /usr/local/lib/python3.11/site-packages/torch/test \
      /root/.cache \
      /tmp/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install \
      --no-cache-dir \
      --no-compile \
      --retries 10 \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      "PyYAML==6.0.3" \
      "opencv-python-headless==4.13.0.92" \
      "matplotlib==3.11.1" \
      "requests==2.34.2" \
      "psutil==7.2.2" \
      "polars==1.43.2" \
      "nvidia-ml-py==13.610.43" \
      "ultralytics-thop==2.1.6" \
      "fastapi==0.116.1" \
      "uvicorn==0.35.0" \
      "httpx==0.28.1" \
    && python -m pip install \
      --no-cache-dir \
      --no-compile \
      --no-deps \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      "ultralytics==8.4.78" \
    && python -m pip install \
      --no-cache-dir \
      --no-compile \
      --no-deps \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
      . \
    && rm -rf /root/.cache /tmp/* \
    && useradd --create-home --uid 10001 algorithm

USER algorithm

EXPOSE 8080

CMD ["uvicorn", "elderly_monitoring.service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
