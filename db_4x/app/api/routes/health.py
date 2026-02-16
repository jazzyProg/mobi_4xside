from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.deps import get_products_repo
from app.repositories.products import ProductsRepository

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    response: Response,
    repo: ProductsRepository = Depends(get_products_repo),
):
    """
    Health check:
    - returns 200 only when DB is reachable
    - returns 503 otherwise
    """
    ok = await repo.ping()
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok", "db_connected": ok}
