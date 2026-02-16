from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.api_client import ExternalServicesClient, get_api_client

router = APIRouter(tags=["Proxy"])


@router.get("/product/active")
async def get_active_product(api_client: ExternalServicesClient = Depends(get_api_client)):
    """Proxy to Products API."""
    try:
        return api_client.get_active_product()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Products API unavailable: {e}")


@router.post("/camera/capture/start")
async def start_camera_capture(session_id: str, api_client: ExternalServicesClient = Depends(get_api_client)):
    """Start camera capture."""
    try:
        return api_client.start_capture(session_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Camera API unavailable: {e}")


@router.post("/camera/capture/stop")
async def stop_camera_capture(api_client: ExternalServicesClient = Depends(get_api_client)):
    """Stop camera capture."""
    try:
        return api_client.stop_capture()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Camera API unavailable: {e}")
