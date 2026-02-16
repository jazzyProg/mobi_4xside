"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import router as camera_router
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.dependencies import cleanup_camera_manager

log = logging.getLogger("camera-service")


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.validate()
        settings.setup_sdk_paths()
        log.info("Service starting (host=%s port=%s prefix=%s)", settings.host, settings.port, settings.api_prefix or "")
        try:
            yield
        finally:
            cleanup_camera_manager()
            log.info("Service stopped")

    app = FastAPI(
        title="Camera Microservice",
        version="1.0.0",
        description="Industrial camera service (SHM + Disk storage)",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = (settings.api_prefix or "").rstrip("/")
    app.include_router(camera_router, prefix=prefix)
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=(settings.log_level or "INFO").lower(),
        reload=False,
    )
