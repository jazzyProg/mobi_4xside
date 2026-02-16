from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.products import ActiveDetailInfoRow


@dataclass
class FakeRepo:
    async def ping(self) -> bool:
        return True

    async def search_products(self, name: str, limit: int):
        # Return empty -> should be 200 with []
        return []

    async def activate_product(self, product_id: int) -> None:
        return None

    async def get_active_detail_json_path(self) -> str:
        raise Exception("not used in this test")

    async def get_active_detail_info(self) -> ActiveDetailInfoRow:
        return ActiveDetailInfoRow(product_name="X", position=1)


def test_search_empty():
    app = create_app(repo_factory=lambda: FakeRepo())
    with TestClient(app) as c:
        r = c.get("/products/search?name=abc")
        assert r.status_code == 200
        assert r.json() == []


def test_active_info_ok():
    app = create_app(repo_factory=lambda: FakeRepo())
    with TestClient(app) as c:
        r = c.get("/products/active_detail/info")
        assert r.status_code == 200
        assert r.json()["product_name"] == "X"
