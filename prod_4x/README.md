# QC Service - Quality Control Microservice

Микросервис компьютерного зрения для автоматической проверки качества деталей.

## Технологии

- FastAPI
- YOLO v8 (Ultralytics)
- OpenCV
- PyTorch
- Shared Memory (для работы с Camera API)

## Установка и запуск

### Вариант 1: Локальная разработка (без Docker)

```bash
# Клонировать репозиторий
git clone <repo-url>
cd qc-service

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Скопировать .env
cp .env.example .env

# Отредактировать .env (указать локальные пути)
nano .env

# Запустить
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
