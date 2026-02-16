# ============================================
# Stage 1: Builder - устанавливаем зависимости
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Устанавливаем build зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем ТОЛЬКО requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# ============================================
# Stage 2: Production - финальный образ
# ============================================
FROM python:3.10-slim

LABEL maintainer="jazzy@company.com"
LABEL description="QC Service - YOLO-based quality control"
LABEL version="1.0"

WORKDIR /app

# Устанавливаем runtime зависимости (для OpenCV, YOLO, libvips)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    # libvips для быстрой обработки изображений
    libvips42 \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные Python пакеты из builder
COPY --from=builder /root/.local /root/.local

# Обновляем PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Копируем код приложения (venv НЕ копируется!)
COPY app/ ./app/

# Healthcheck script
COPY app/scripts/ ./app/scripts/

# Создаем директории для данных
RUN mkdir -p /app/models \
    /app/data/quarantine \
    /app/data/cad \
    /app/logs

# Открываем порт
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python /app/scripts/healthcheck.py || exit 1

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
