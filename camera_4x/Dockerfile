# Dockerfile
FROM python:3.10-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

FROM python:3.10-slim

LABEL maintainer="your-email@example.com"
LABEL description="Camera microservice with non-root user"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    libstdc++6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /root/.local /home/appuser/.local

RUN groupadd -r appuser -g 1000 && \
    useradd -r -u 1000 -g appuser -m -s /bin/bash appuser && \
    mkdir -p /app /data/photos /app/logs && \
    chown -R appuser:appuser /app /data/photos /app/logs

ENV PATH=/home/appuser/.local/bin:$PATH
ENV MVS_SDK_PATH=/opt/MVS
ENV MVCAM_COMMON_RUNENV=/opt/MVS/lib
ENV LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:$PYTHONPATH

# Runtime API settings (defaults)
ENV HOST=0.0.0.0
ENV PORT=8003
ENV API_PREFIX=

COPY --chown=appuser:appuser ./app /app/app
COPY --chown=appuser:appuser ./scripts /app/scripts

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python /app/scripts/healthcheck.py || exit 1

EXPOSE 8003

CMD ["sh", "-c", "python -m uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8003}"]
