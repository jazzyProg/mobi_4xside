from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


class FakeRepo:
    def __init__(self, ok: bool):
        self._ok = ok

    async def ping(self) -> bool:
        return self._ok


def test_health_200_when_db_ok():
    app = create_app(repo_factory=lambda: FakeRepo(ok=True))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["db_connected"] is True


def test_health_503_when_db_down():
    app = create_app(repo_factory=lambda: FakeRepo(ok=False))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503
