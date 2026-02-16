from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class ProductNotFoundError(Exception):
    pass


class ActiveDetailNotSelectedError(Exception):
    pass


@dataclass(frozen=True)
class ProductSearchRow:
    id: int
    product_name: str
    hash: str
    position: int
    svg_path: str


@dataclass(frozen=True)
class ActiveDetailInfoRow:
    product_name: str
    position: int


class ProductsRepository(Protocol):
    async def ping(self) -> bool: ...
    async def search_products(self, name: str, limit: int) -> List[ProductSearchRow]: ...
    async def activate_product(self, product_id: int) -> None: ...
    async def get_active_detail_json_path(self) -> str: ...
    async def get_active_detail_info(self) -> ActiveDetailInfoRow: ...


class SqlProductsRepository:
    """
    Thin repository over raw SQL (keeps current schema untouched).
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sm = sessionmaker

    async def ping(self) -> bool:
        try:
            async with self._sm() as s:
                await s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def search_products(self, name: str, limit: int) -> List[ProductSearchRow]:
        q = text(
            """
            SELECT id, product_name, hash, svg_path, position
            FROM product_files
            WHERE product_name % :search_name
            ORDER BY similarity(product_name, :search_name) DESC
            LIMIT :limit
            """
        )
        async with self._sm() as s:
            r = await s.execute(q, {"search_name": name, "limit": limit})
            rows = r.fetchall()

        out: list[ProductSearchRow] = []
        for row in rows:
            out.append(
                ProductSearchRow(
                    id=row.id,
                    product_name=row.product_name,
                    hash=row.hash,
                    position=row.position or 0,
                    svg_path=row.svg_path,
                )
            )
        return out

    async def activate_product(self, product_id: int) -> None:
        check_q = text(
            """
            SELECT id, product_name, position, hash
            FROM product_files
            WHERE id = :product_id
            """
        )
        upsert_q = text(
            """
            INSERT INTO singleton_detail (id, product_id, product_name, position, hash, selected_at)
            VALUES (1, :product_id, :product_name, :position, :hash, NOW())
            ON CONFLICT (id)
            DO UPDATE SET
                product_id = :product_id,
                product_name = :product_name,
                position = :position,
                hash = :hash,
                selected_at = NOW()
            """
        )

        async with self._sm() as s:
            r = await s.execute(check_q, {"product_id": product_id})
            product = r.fetchone()
            if not product:
                raise ProductNotFoundError()

            await s.execute(
                upsert_q,
                {
                    "product_id": product.id,
                    "product_name": product.product_name,
                    "position": product.position or 0,
                    "hash": product.hash,
                },
            )
            await s.commit()

    async def get_active_detail_json_path(self) -> str:
        q = text(
            """
            SELECT pf.json_path
            FROM singleton_detail sd
            JOIN product_files pf ON sd.product_id = pf.id
            WHERE sd.id = 1
            """
        )
        async with self._sm() as s:
            r = await s.execute(q)
            row = r.fetchone()
        if not row:
            raise ActiveDetailNotSelectedError()
        return row.json_path

    async def get_active_detail_info(self) -> ActiveDetailInfoRow:
        q = text(
            """
            SELECT product_name, position
            FROM singleton_detail
            WHERE id = 1
            """
        )
        async with self._sm() as s:
            r = await s.execute(q)
            row = r.fetchone()
        if not row:
            raise ActiveDetailNotSelectedError()
        return ActiveDetailInfoRow(product_name=row.product_name, position=row.position)
