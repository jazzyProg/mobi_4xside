from __future__ import annotations

from fastapi import Request

from app.repositories.products import ProductsRepository


def get_products_repo(request: Request) -> ProductsRepository:
    """
    Dependency: repository stored in app.state (easy to replace in tests).
    """
    return request.app.state.products_repo
