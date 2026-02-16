"""app.api.endpoints

API routers for QC service. Kept in a package to keep modules small and testable.
"""

from __future__ import annotations

from fastapi import APIRouter

from .accumulator import router as accumulator_router
from .check import router as check_router
from .control import router as control_router
from .health import router as health_router
from .proxy import router as proxy_router
from .status import router as status_router

router = APIRouter()
router.include_router(health_router)
router.include_router(status_router)
router.include_router(control_router)
router.include_router(check_router)
router.include_router(proxy_router)
router.include_router(accumulator_router)
