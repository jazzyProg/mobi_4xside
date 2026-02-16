from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import StatusResponse
from app.core.state import state

router = APIRouter(tags=["Status"])


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return current service status."""
    return StatusResponse(running=state.is_running, current_frame_id=state.last_processed_id, stats=state.stats)
