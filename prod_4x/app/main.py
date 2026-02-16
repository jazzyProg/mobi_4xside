"""app.main

FastAPI entrypoint for QC service.

Key goals:
- Keep API responses and paths backward compatible.
- Keep background QC loop behavior equivalent, but with better structure.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.core.api_client import get_api_client
from app.core.state import state
from app.core.worker import qc_loop
from app.services import inference

# Logging is configured at import time to preserve existing container behavior.
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("QC-API")


def start_background_loop() -> None:
    """Start background QC loop (and start camera capture best-effort)."""
    if state.is_running:
        logger.warning("Background loop already running")
        return

    logger.info("=" * 60)
    logger.info("Starting QC System...")
    logger.info("=" * 60)

    # 1) Start camera capture (best-effort, preserve old behavior).
    try:
        api_client = get_api_client()
        session_id = f"qc_auto_{int(time.time())}"
        camera_response = api_client.start_capture(session_id)
        logger.info("Camera started: %s", camera_response)
    except Exception as e:
        logger.error("Failed to start camera: %s", e)
        logger.warning("QC loop will start anyway (camera might be already running)")

    # 2) Start loop thread.
    state.stop_event.clear()
    state.thread = threading.Thread(target=qc_loop, kwargs={"settings": settings}, daemon=True)
    state.thread.start()
    state.is_running = True

    logger.info("=" * 60)
    logger.info("QC System RUNNING")
    logger.info("=" * 60)


def stop_background_loop() -> None:
    """Stop background QC loop (and stop camera capture best-effort)."""
    if not state.is_running:
        logger.warning("Background loop not running")
        return

    logger.info("=" * 60)
    logger.info("Stopping QC System...")
    logger.info("=" * 60)

    # 1) Stop loop.
    state.stop_event.set()
    if state.thread:
        state.thread.join(timeout=5.0)
    state.is_running = False
    logger.info("QC loop stopped")

    # 2) Stop camera capture (best-effort).
    try:
        api_client = get_api_client()
        camera_response = api_client.stop_capture()
        logger.info("Camera stopped: %s", camera_response)
    except Exception as e:
        logger.warning("Failed to stop camera: %s", e)

    logger.info("=" * 60)
    logger.info("QC System STOPPED")
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    logger.info("=" * 60)
    logger.info("QC Service starting up...")
    logger.info("=" * 60)
    logger.info("YOLO model will be loaded on first inference call")

    yield

    logger.info("=" * 60)
    logger.info("QC Service shutting down...")
    logger.info("=" * 60)

    # Stop background loop (if any).
    state.stop_event.set()
    if state.thread:
        state.thread.join(timeout=5.0)
    state.is_running = False

    # Close API client.
    try:
        api_client = get_api_client()
        api_client.close()
        logger.info("API client closed")
    except Exception as e:
        logger.warning("Failed to close API client: %s", e)

    # Release GPU memory.
    try:
        inference.release_gpu()
        logger.info("GPU memory released")
    except Exception as e:
        logger.warning("Failed to release GPU: %s", e)

    logger.info("QC Service stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description="Quality Control Service with YOLO-based defect detection",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
