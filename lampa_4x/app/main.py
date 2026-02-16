from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI

from app.api.routes import health_router, signals_router
from app.config import get_settings
from app.modbus.controller import ModbusController


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_app(
    controller_factory: Optional[Callable[..., ModbusController]] = None,
) -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)

    if controller_factory is None:
        def controller_factory() -> ModbusController:
            return ModbusController(
                host=settings.MODBUS_IP,
                port=settings.MODBUS_PORT,
                slave_id=settings.MODBUS_SLAVE_ID,
                timeout_sec=settings.MODBUS_TIMEOUT_SEC,
                connect_timeout_sec=settings.MODBUS_CONNECT_TIMEOUT_SEC,
                retries=settings.MODBUS_RETRIES,
                retry_delay_sec=settings.MODBUS_RETRY_DELAY_SEC,
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.controller = controller_factory()
        ok = await app.state.controller.connect()
        if not ok:
            logging.getLogger("signals-api").error("Modbus connect failed (STRICT_STARTUP=%s)", settings.STRICT_STARTUP)
            if settings.STRICT_STARTUP:
                raise RuntimeError("Modbus connect failed")
        yield
        # Graceful shutdown: cancel tasks + clear bits
        if settings.SHUTDOWN_CLEAR_BITS:
            bits = [settings.BIT_SUCCESS, settings.BIT_FAIL, settings.BIT_ALARM, settings.BIT_HEARTBEAT]
            try:
                await app.state.controller.shutdown(bits)
            except Exception:
                logging.getLogger("signals-api").exception("Shutdown cleanup failed")
        app.state.controller.disconnect()

    app = FastAPI(
        title="Signaling Service",
        description="Microservice for controlling signal lamps via Modbus TCP.",
        version="1.0.0",
        lifespan=lifespan,
    )

    prefix = settings.API_PREFIX.rstrip("/")
    app.include_router(health_router, prefix=prefix)
    app.include_router(signals_router, prefix=prefix)
    return app


app = create_app()
