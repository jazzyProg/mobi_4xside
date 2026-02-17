from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


class FakeController:
    def __init__(self):
        self.last_error = None
        self.calls = []
        self._connected = True

    async def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return self._connected

    async def ping(self) -> bool:
        return True

    async def shutdown(self, bits_to_clear: list[int]) -> None:
        return None

    async def schedule_pulse(self, bit: int, duration: float) -> None:
        self.calls.append((bit, duration))


def test_success_signal_enqueues_pulse():
    ctrl = FakeController()
    app = create_app(controller_factory=lambda: ctrl)
    settings = get_settings()
    with TestClient(app) as c:
        r = c.post("/signal/success?duration=0.5")
        assert r.status_code == 202
        assert ctrl.calls == [(settings.BIT_SUCCESS, 0.5)]


def test_camera_trigger_signal_enqueues_pulse_with_custom_duration():
    ctrl = FakeController()
    app = create_app(controller_factory=lambda: ctrl)
    settings = get_settings()
    with TestClient(app) as c:
        r = c.post("/signal/camera-trigger?duration=0.7")
        assert r.status_code == 202
        assert ctrl.calls == [(settings.BIT_CAMERA_TRIGGER, 0.7)]
