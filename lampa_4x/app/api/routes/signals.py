from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import Settings, get_settings
from app.deps import get_controller
from app.modbus.controller import ModbusController

router = APIRouter(tags=["Signals"])


async def _require_modbus(controller: ModbusController) -> None:
    # Use real ping to avoid false positives when socket is open but device is dead.
    ok = await controller.ping()
    if not ok:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modbus is not connected")


@router.post("/signal/success", status_code=status.HTTP_202_ACCEPTED)
async def signal_success(
    duration: float = Query(default=3.0, ge=0.1, le=60.0, description="Длительность импульса в секундах"),
    settings: Settings = Depends(get_settings),  # <-- тот самый микро-фикс вместо Depends(lambda: Settings())
    controller: ModbusController = Depends(get_controller),
):
    await _require_modbus(controller)
    await controller.schedule_pulse(settings.BIT_SUCCESS, duration)
    return {"message": f"Success signal (DO{settings.BIT_SUCCESS + 1}) initiated with duration {duration}s."}


@router.post("/signal/fail", status_code=status.HTTP_202_ACCEPTED)
async def signal_fail(
    settings: Settings = Depends(get_settings),
    controller: ModbusController = Depends(get_controller),
):
    await _require_modbus(controller)
    await controller.schedule_pulse(settings.BIT_FAIL, settings.PULSE_DURATION_SEC)
    return {"message": f"Fail signal (DO{settings.BIT_FAIL + 1}) initiated."}


@router.post("/signal/alarm", status_code=status.HTTP_202_ACCEPTED)
async def signal_alarm(
    settings: Settings = Depends(get_settings),
    controller: ModbusController = Depends(get_controller),
):
    await _require_modbus(controller)
    await controller.schedule_pulse(settings.BIT_ALARM, settings.PULSE_DURATION_SEC)
    return {"message": f"Alarm signal (DO{settings.BIT_ALARM + 1}) initiated."}


@router.post("/signal/heartbeat", status_code=status.HTTP_202_ACCEPTED)
async def signal_heartbeat(
    duration: float = Query(default=1.0, ge=0.1, le=60.0, description="Длительность импульса в секундах"),
    settings: Settings = Depends(get_settings),
    controller: ModbusController = Depends(get_controller),
):
    await _require_modbus(controller)
    await controller.schedule_pulse(settings.BIT_HEARTBEAT, duration)
    return {"message": f"Manual heartbeat signal (DO{settings.BIT_HEARTBEAT + 1}) initiated with duration {duration}s."}
