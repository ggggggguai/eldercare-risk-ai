FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_PROGRESS_BAR=on \
    MODEL_PATH=/models/yolov8n-pose.pt

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install \
      --no-cache-dir \
      --retries 10 \
      --timeout 600 \
      --progress-bar on \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchvision==0.21.0

RUN python -m pip install \
      --no-cache-dir \
      --retries 10 \
      --timeout 600 \
      --progress-bar on \
      ".[vision,service]"

RUN useradd --create-home --uid 10001 algorithm \
    && mkdir -p /models \
    && chown -R algorithm:algorithm /app /models

USER algorithm

EXPOSE 8080

CMD ["uvicorn", "elderly_monitoring.service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
