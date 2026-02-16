from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ControlResponse
from app.core.state import state

router = APIRouter(tags=["Control"])


@router.post("/control/start", response_model=ControlResponse)
async def start_loop() -> ControlResponse:
    """Start background processing loop."""
    if state.is_running:
        return ControlResponse(msg="Already running", status="running")

    # Import locally to avoid circular imports.
    from app import main

    main.start_background_loop()
    return ControlResponse(msg="Started", status="running")


@router.post("/control/stop", response_model=ControlResponse)
async def stop_loop() -> ControlResponse:
    """Stop background processing loop."""
    if not state.is_running:
        return ControlResponse(msg="Not running", status="idle")

    from app import main

    main.stop_background_loop()
    return ControlResponse(msg="Stopped", status="idle")
