# Camera Microservice

Production-ready микросервис для работы с промышленной камерой MVS (Hikvision).

## Архитектура хранения

1. **SHM (Shared Memory)** - основной канал для real-time доступа
   - Ring buffer с 16 слотами по 10MB
   - Zero-copy доступ для QC-сервиса
   - Всегда содержит последние 16 кадров

2. **RAM Buffer** - метаданные последних ~100MB кадров
   - Быстрый доступ через REST API
   - Хранит только metadata, не сами кадры

3. **Disk Archive** - автоматическое сохранение при переполнении RAM
   - Структура: `/data/frames/YYYY-MM-DD/HH/frame_*.bin`
   - Автоматическая ротация при достижении лимита

## Запуск через Docker

```bash
# 1. Создать .env файл
cp .env.example .env
# Отредактировать параметры камеры

# 2. Запустить сервис
docker-compose up -d --build

# 3. Проверить статус
curl http://localhost:8003/api/status

# 4. Логи
docker-compose logs -f camera-service
