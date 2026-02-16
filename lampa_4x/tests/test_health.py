from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


class FakeController:
    def __init__(self, ok: bool):
        self._ok = ok
        self.last_error = None if ok else "connect_failed"

    async def connect(self) -> bool:
        return self._ok

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return self._ok

    async def ping(self) -> bool:
        return self._ok

    async def shutdown(self, bits_to_clear: list[int]) -> None:
        return None

    async def pulse_do(self, bit: int, duration: float) -> None:
        return None


def test_health_200_when_modbus_ok():
    app = create_app(controller_factory=lambda: FakeController(ok=True))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["modbus_connected"] is True


def test_health_503_when_modbus_down():
    app = create_app(controller_factory=lambda: FakeController(ok=False))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503
