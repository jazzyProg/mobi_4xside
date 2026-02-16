from __future__ import annotations

import logging

from fastapi import APIRouter

router = APIRouter(tags=["Accumulator"])
logger = logging.getLogger(__name__)


@router.get("/accumulator/status")
async def get_accumulator_status():
    """Return accumulator state."""
    from app.core.accumulator import get_accumulator_stats

    return get_accumulator_stats()


@router.post("/accumulator/reset")
async def reset_accumulator():
    """Reset all counters (manual)."""
    from app.core.accumulator import accumulator

    with accumulator.lock:
        for key in accumulator.counters:
            accumulator.counters[key] = 0
            accumulator.last_increment[key] = False
        accumulator.alarm_active = False

    logger.info("Accumulator manually reset via API")
    return {"msg": "Accumulator reset", "counters": accumulator.counters}
