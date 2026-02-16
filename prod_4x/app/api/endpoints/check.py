from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.schemas import CheckRequest, CheckResponse
from app.config import settings
from app.core.api_client import ExternalServicesClient, get_api_client
from app.core.state import state
from app.services.shm_reader import CameraSHMReader
from app.core.pipeline import process_single

logger = logging.getLogger(__name__)
router = APIRouter(tags=["QC"])


def _ensure_shm_client(api_client: ExternalServicesClient) -> None:
    """Initialize SHM reader once (best-effort)."""
    if state.shm_client:
        return

    shm_name = settings.CAMERA_SHM_NAME
    try:
        camera_status_url = f"{settings.CAMERA_API_URL}/status"
        resp = api_client._client.get(camera_status_url, timeout=5.0)  # type: ignore[attr-defined]
        if resp.status_code == 200:
            status_data = resp.json()
            shm_info = status_data.get("storage", {}).get("shm", {})
            shm_name = shm_info.get("name") or shm_name
    except Exception as e:
        logger.warning("Failed to get SHM from Camera API: %s, using fallback: %s", e, shm_name)

    logger.info("Initializing SHM reader: %s", shm_name)
    state.shm_client = CameraSHMReader(name=shm_name, slot_size=settings.SHM_SLOT_SIZE, slot_count=settings.SHM_NUM_SLOTS)


@router.post("/check/run", response_model=CheckResponse)
async def run_quality_check(
    request: CheckRequest,
    background_tasks: BackgroundTasks,
    api_client: ExternalServicesClient = Depends(get_api_client),
) -> CheckResponse:
    """Run QC check for a single session using latest frame."""
    start_time = time.time()

    try:
        logger.info("Starting QC check for session %s", request.session_id)

        if request.use_latest_frame:
            frames = api_client.get_latest_frames_metadata(count=1)
            if not frames:
                raise HTTPException(status_code=404, detail="No frames available")
            frame_meta = frames[0]
        else:
            raise HTTPException(status_code=501, detail="Specific frame_id not implemented")

        _ensure_shm_client(api_client)

        shm_slot = frame_meta.get("shm_slot")
        if shm_slot is None:
            raise HTTPException(status_code=500, detail="Frame metadata does not include shm_slot")

        frame_data, _slot_metadata = state.shm_client.read_slot(int(shm_slot))  # type: ignore[union-attr]
        if frame_data is None:
            raise HTTPException(status_code=500, detail=f"Failed to read frame from slot {shm_slot}")

        qc_ok, _vision_json, report = process_single(
            frame_data,
            model_path=settings.MODEL_PATH,
            quarantine_dir=settings.QUARANTINE_DIR,
            stem_name=f"manual_{request.session_id}",
            session_id=request.session_id,
        )

        if qc_ok:
            background_tasks.add_task(api_client.signal_success, 3.0)
        else:
            background_tasks.add_task(api_client.signal_fail)

        processing_time_ms = (time.time() - start_time) * 1000
        return CheckResponse(ok=qc_ok, session_id=request.session_id, report=report, processing_time_ms=round(processing_time_ms, 2))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Check failed for session %s: %s", request.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
