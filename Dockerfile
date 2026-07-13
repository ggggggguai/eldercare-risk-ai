FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/yolov8n-pose.pt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0 torchvision==0.21.0 \
    && python -m pip install --no-cache-dir ".[vision,service]" \
    && python -m pip uninstall -y opencv-python \
    && python -m pip install --no-cache-dir opencv-python-headless==4.13.0.92 \
    && useradd --create-home --uid 10001 algorithm

USER algorithm
EXPOSE 8080

CMD ["uvicorn", "elderly_monitoring.service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
