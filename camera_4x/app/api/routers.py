from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.schemas import (
    StartCaptureRequest,
    StopCaptureResponse,
    StatusResponse,
    FrameMetadataResponse,
    HealthResponse
)
from app.dependencies import get_camera_manager

router = APIRouter(tags=["camera"])

@router.post("/capture/start")
async def start_capture(
    request: StartCaptureRequest,
    manager = Depends(get_camera_manager)
):
    try:
        result = manager.start_capture(request.session_id)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capture/stop", response_model=StopCaptureResponse)
async def stop_capture(manager = Depends(get_camera_manager)):
    result = manager.stop_capture()
    return result

@router.get("/status", response_model=StatusResponse)
async def get_status(manager = Depends(get_camera_manager)):
    status = manager.get_status()
    return status

@router.get("/frames/latest/meta", response_model=List[FrameMetadataResponse])
async def get_latest_frames_metadata(
    count: int = 1,
    manager = Depends(get_camera_manager)
):
    if count < 1 or count > 100:
        raise HTTPException(status_code=400, detail="Count must be between 1 and 100")

    frames = manager.get_latest_frames(count)

    return [
        FrameMetadataResponse(
            frame_id=f.frame_id,
            session_id=f.session_id,
            timestamp=f.timestamp,
            width=f.width,
            height=f.height,
            pixel_format=f.pixel_format,
            size_bytes=f.size_bytes,
            storage_location=f.storage_location.value,
            shm_slot=f.shm_slot,
            disk_path=f.disk_path,
            camera_timestamp=f.camera_timestamp,
            exposure_time=f.exposure_time,
            gain=f.gain
        )
        for f in frames
    ]

@router.get("/frames/latest")
async def get_latest_frame_data(manager = Depends(get_camera_manager)):
    frames = manager.get_latest_frames(1)
    if not frames:
        raise HTTPException(status_code=404, detail="No frames available")

    latest = frames[0]
    data = manager.load_frame_data(latest)

    if not data:
        raise HTTPException(status_code=410, detail="Frame data no longer available")

    media_type = "image/jpeg" if latest.pixel_format.lower() == "jpeg" else "application/octet-stream"

    return Response(content=data, media_type=media_type)

@router.get("/frames/oldest/meta", response_model=FrameMetadataResponse)
async def get_oldest_frame_metadata(manager = Depends(get_camera_manager)):
    """
    Return metadata of the oldest frame in the queue (FIFO processing).

    Intended for QC service usage.
    """
    frame = manager.get_oldest_frame()
    if not frame:
        raise HTTPException(status_code=404, detail="No frames available")

    return FrameMetadataResponse(
        frame_id=frame.frame_id,
        session_id=frame.session_id,
        timestamp=frame.timestamp,
        width=frame.width,
        height=frame.height,
        pixel_format=frame.pixel_format,
        size_bytes=frame.size_bytes,
        storage_location=frame.storage_location.value,
        shm_slot=frame.shm_slot,
        disk_path=frame.disk_path,
        camera_timestamp=frame.camera_timestamp,
        exposure_time=frame.exposure_time,
        gain=frame.gain
    )

@router.get("/frames/{frame_id}/meta", response_model=FrameMetadataResponse)
async def get_frame_metadata(frame_id: int, manager = Depends(get_camera_manager)):
    frame = manager.get_frame_by_id(frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    return FrameMetadataResponse(
        frame_id=frame.frame_id,
        session_id=frame.session_id,
        timestamp=frame.timestamp,
        width=frame.width,
        height=frame.height,
        pixel_format=frame.pixel_format,
        size_bytes=frame.size_bytes,
        storage_location=frame.storage_location.value,
        shm_slot=frame.shm_slot,
        disk_path=frame.disk_path,
        camera_timestamp=frame.camera_timestamp,
        exposure_time=frame.exposure_time,
        gain=frame.gain
    )

@router.get("/frames/{frame_id}/data")
async def get_frame_data(frame_id: int, manager = Depends(get_camera_manager)):
    metadata = manager.get_frame_by_id(frame_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Frame not found")

    data = manager.load_frame_data(metadata)
    if not data:
        raise HTTPException(status_code=410, detail="Frame data no longer available")

    media_type = "image/jpeg" if metadata.pixel_format.lower() == "jpeg" else "application/octet-stream"

    return Response(content=data, media_type=media_type)

@router.delete("/frames/{frame_id}")
async def delete_frame(
    frame_id: int,
    manager = Depends(get_camera_manager)
):
    """
    Delete a processed frame from memory and disk.

    Intended to be called by QC service after successful processing.
    """
    try:
        deleted = manager.delete_frame(frame_id)
        if deleted:
            return {
                "status": "deleted",
                "frame_id": frame_id,
                "message": "Frame deleted successfully"
            }
        else:
            return {
                "status": "not_found",
                "frame_id": frame_id,
                "message": "Frame not found or already deleted"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(manager = Depends(get_camera_manager)):
    status = manager.get_status()

    if status["state"] == "error":
        raise HTTPException(status_code=503, detail="Service unhealthy")

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        state=status["state"]
    )
