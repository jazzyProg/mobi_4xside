# gateway/Makefile

.PHONY: help up down restart logs build status clean

help:
	@echo "Production System Management"
	@echo "============================"
	@echo "make up          - Запустить всю систему"
	@echo "make down        - Остановить систему"
	@echo "make restart     - Перезапустить систему"
	@echo "make logs        - Показать логи всех сервисов"
	@echo "make build       - Пересобрать все образы"
	@echo "make status      - Статус сервисов"
	@echo "make clean       - Очистить все (volumes + images)"
	@echo ""
	@echo "Отдельные сервисы:"
	@echo "make logs-qc     - Логи QC Service"
	@echo "make logs-camera - Логи Camera API"
	@echo "make logs-signals- Логи Signals API"
	@echo "make logs-db     - Логи Products API"
	@echo ""
	@echo "make restart-qc  - Перезапустить QC Service"
	@echo "make update-qc   - Обновить QC Service без даунтайма"

# Запуск всей системы
up:
	@echo "🚀 Starting production system..."
	docker-compose up -d
	@sleep 5
	@echo "✅ System started!"
	@make status

# Остановка
down:
	@echo "🛑 Stopping production system..."
	docker-compose down
	@echo "✅ System stopped!"

# Полная перезагрузка
restart:
	@echo "🔄 Restarting production system..."
	docker-compose restart
	@echo "✅ System restarted!"

# Пересборка всех образов
build:
	@echo "🔨 Building all services..."
	docker-compose build --no-cache --parallel
	@echo "✅ Build complete!"

# Логи
logs:
	docker-compose logs -f --tail=100

logs-qc:
	docker-compose logs -f --tail=100 qc-service

logs-camera:
	docker-compose logs -f --tail=100 camera-api

logs-signals:
	docker-compose logs -f --tail=100 signals-api

logs-db:
	docker-compose logs -f --tail=100 db-api

# Перезапуск отдельных сервисов
restart-qc:
	docker-compose restart qc-service

restart-camera:
	docker-compose restart camera-api

restart-signals:
	docker-compose restart signals-api

restart-db:
	docker-compose restart db-api

# Обновление одного сервиса без даунтайма
update-qc:
	@echo "📦 Updating QC Service..."
	cd ../prod_4x && git pull
	docker-compose build qc-service
	docker-compose up -d --no-deps qc-service
	@echo "✅ QC Service updated!"

update-camera:
	@echo "📦 Updating Camera API..."
	cd ../camera_4x && git pull
	docker-compose build camera-api
	docker-compose up -d --no-deps camera-api
	@echo "✅ Camera API updated!"

# Статус
status:
	@echo "📊 System Status:"
	@echo "================"
	@docker-compose ps
	@echo ""
	@echo "🏥 Health Checks:"
	@echo "================"
	@curl -s http://localhost:8080/health | jq . || echo "Gateway: ❌"
	@curl -s http://localhost:8080/api/qc/health | jq . || echo "QC Service: ❌"
	@curl -s http://localhost:8080/api/camera/health | jq . || echo "Camera API: ❌"
	@curl -s http://localhost:8080/api/signals/health | jq . || echo "Signals API: ❌"
	@curl -s http://localhost:8080/api/products/health | jq . || echo "Products API: ❌"

# Очистка
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete!"

# Полная очистка с образами
clean-all:
	@echo "🧹 Deep cleaning..."
	docker-compose down -v --rmi all
	docker system prune -af
	@echo "✅ Deep cleanup complete!"
