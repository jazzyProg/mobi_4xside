# db_4x (Product Microservice)

Микросервис для работы с базой данных продуктов и деталей.

## Возможности

- Триграммный поиск продуктов по имени
- Выбор активной детали (singleton)
- Получение JSON данных активной детали
- Получение SVG изображений

## Технологии

- FastAPI
- PostgreSQL + asyncpg
- SQLAlchemy 2.0
- Docker

## Конфигурация

Скопируйте `.env.example` в `.env` и укажите реальные параметры.

> В репозитории **не хранить** `.env` (включая пароли).

## API

- `GET /health`
- `GET /products/search?name=...`
- `POST /products/select` body: `{ "id": 123 }`
- `GET /products/active_detail`
- `GET /products/active_detail/info`

## Установка и запуск

### Локально (разработка)

```bash
# Создать venv
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Создать .env файл
cp .env.example .env
# Отредактировать .env с реальными данными

# Запустить
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Тесты
pytest -q
```
