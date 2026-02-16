from __future__ import annotations

import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import Settings, get_settings
from app.deps import get_products_repo
from app.models import (
    ActiveDetailInfo,
    ProductSearchResult,
    SelectProductRequest,
    SelectProductResponse,
)
from app.repositories.products import (
    ActiveDetailNotSelectedError,
    ProductNotFoundError,
    ProductsRepository,
)
from app.utils.files import UnsafePathError, read_text_file, resolve_file_path

log = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["Products"])


@router.get("/search", response_model=List[ProductSearchResult])
async def search_products(
    name: str = Query(..., min_length=1, description="Product name to search"),
    settings: Settings = Depends(get_settings),
    repo: ProductsRepository = Depends(get_products_repo),
):
    """
    Trigram search by product_name (PostgreSQL pg_trgm).
    Returns up to SEARCH_LIMIT results with embedded SVG content.
    """
    rows = await repo.search_products(name=name, limit=settings.SEARCH_LIMIT)
    base_dir = settings.files_root

    out: list[ProductSearchResult] = []
    for r in rows:
        try:
            svg_path = resolve_file_path(r.svg_path, base_dir=base_dir, allowed_suffixes={".svg"})
            svg = await read_text_file(svg_path)
        except UnsafePathError as e:
            log.warning("Skipping unsafe svg_path for product_id=%s: %s", r.id, e)
            continue

        if svg is None:
            # Keep old behavior: silently skip missing files
            continue

        out.append(
            ProductSearchResult(
                id=r.id,
                product_name=r.product_name,
                hash=r.hash,
                position=r.position,
                svg=svg,
            )
        )
    return out


@router.post("/select", response_model=SelectProductResponse)
async def select_product(
    request: SelectProductRequest,
    repo: ProductsRepository = Depends(get_products_repo),
):
    """
    Activate product by ID (singleton_detail id=1).
    """
    try:
        await repo.activate_product(product_id=request.id)
    except ProductNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Деталь не найдена в базе данных. Обратитесь к администратору.",
        )
    return SelectProductResponse(status="success", message="Деталь успешно выбрана")


@router.get("/active_detail")
async def get_active_detail(
    settings: Settings = Depends(get_settings),
    repo: ProductsRepository = Depends(get_products_repo),
):
    """
    Return JSON content of active detail from singleton_detail.
    """
    try:
        raw_json_path = await repo.get_active_detail_json_path()
    except ActiveDetailNotSelectedError:
        raise HTTPException(status_code=404, detail="Деталь еще не выбрана")

    try:
        json_path = resolve_file_path(raw_json_path, base_dir=settings.files_root, allowed_suffixes={".json"})
        json_text = await read_text_file(json_path)
    except UnsafePathError as e:
        log.error("Unsafe json_path from DB: %s", e)
        raise HTTPException(status_code=500, detail="JSON путь некорректен. Обратитесь к администратору.")

    if json_text is None:
        raise HTTPException(status_code=500, detail="JSON файл не найден. Обратитесь к администратору.")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Ошибка чтения JSON файла")


@router.get("/active_detail/info", response_model=ActiveDetailInfo)
async def get_active_detail_info(
    repo: ProductsRepository = Depends(get_products_repo),
):
    """
    Return product_name and position of active detail.
    """
    try:
        info = await repo.get_active_detail_info()
    except ActiveDetailNotSelectedError:
        raise HTTPException(status_code=404, detail="Деталь еще не выбрана")
    return ActiveDetailInfo(product_name=info.product_name, position=info.position)
