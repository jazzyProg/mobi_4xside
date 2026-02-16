from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
# from datetime import datetime

# Response модели
class ProductSearchResult(BaseModel):
    product_name: str
    hash: str
    position: int
    id: int
    svg: str = Field(..., description="SVG file content")

class SelectProductRequest(BaseModel):
    id: int = Field(..., description="Product ID to select", gt=0)

class SelectProductResponse(BaseModel):
    status: str
    message: str

class ActiveDetailInfo(BaseModel):
    product_name: str
    position: int

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
