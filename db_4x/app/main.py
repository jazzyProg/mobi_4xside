from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.routes import health_router, products_router
from app.config import get_settings
from app.repositories.products import ProductsRepository, SqlProductsRepository


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_app(
    repo_factory: Optional[Callable[[], ProductsRepository]] = None,
) -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if repo_factory is not None:
            app.state.products_repo = repo_factory()
            yield
            return

        engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
        )
        sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        app.state.products_repo = SqlProductsRepository(sessionmaker=sessionmaker)

        yield

        await engine.dispose()

    app = FastAPI(
        title="Product Microservice",
        description="Microservice for searching/selecting product details via PostgreSQL + files.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.API_PREFIX.rstrip("/")
    app.include_router(health_router, prefix=prefix)
    app.include_router(products_router, prefix=prefix)

    @app.get("/")
    async def root():
        return {"service": "db_4x", "version": "1.0.0", "status": "running"}

    return app


app = create_app()
