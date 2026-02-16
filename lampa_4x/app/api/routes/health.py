from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.deps import get_controller
from app.modbus.controller import ModbusController

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    response: Response,
    controller: ModbusController = Depends(get_controller),
):
    ok = await controller.ping()
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok",
        "modbus_connected": ok,
        "last_error": controller.last_error,
    }
