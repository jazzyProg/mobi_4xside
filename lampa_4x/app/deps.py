from __future__ import annotations

from fastapi import Request

from app.modbus.controller import ModbusController


def get_controller(request: Request) -> ModbusController:
    """
    Dependency: get controller from app.state.
    This makes testing easy (tests can inject a fake controller).
    """
    return request.app.state.controller
