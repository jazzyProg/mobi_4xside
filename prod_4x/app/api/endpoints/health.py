from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="qc-service", version=settings.API_VERSION)
