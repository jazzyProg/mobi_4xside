from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class StartCaptureRequest(BaseModel):
    session_id: Optional[str] = None

class StopCaptureResponse(BaseModel):
    status: str
    message: str
    frames_captured: int

class FrameMetadataResponse(BaseModel):
    frame_id: int
    session_id: str
    timestamp: float
    width: int
    height: int
    pixel_format: str
    size_bytes: int
    storage_location: str
    shm_slot: Optional[int] = None
    disk_path: Optional[str] = None
    camera_timestamp: Optional[int] = None
    exposure_time: Optional[float] = None
    gain: Optional[float] = None

class SessionResponse(BaseModel):
    session_id: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    frames_total: int
    frames_in_memory: int
    frames_on_disk: int

class StorageStats(BaseModel):
    frames_in_memory: int
    memory_usage_mb: float
    memory_limit_mb: float
    memory_usage_percent: float

class DiskStats(BaseModel):
    total_size_gb: float
    max_size_gb: float
    usage_percent: float
    base_path: str

class ShmStats(BaseModel):
    name: str
    slot_count: int
    slot_size_mb: float
    total_size_mb: float
    write_index: int

class StatusResponse(BaseModel):
    state: str
    session: SessionResponse
    storage: dict
    error: Optional[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    state: str
