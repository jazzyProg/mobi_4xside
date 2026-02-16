from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class CameraState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class StorageLocation(str, Enum):
    MEMORY = "memory"
    DISK = "disk"
    SHM = "shm"

@dataclass
class SessionInfo:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    frames_count: int = 0
    frames_in_memory: int = 0
    frames_on_disk: int = 0

@dataclass
class FrameMetadata:
    frame_id: int
    session_id: str
    timestamp: float
    width: int
    height: int
    pixel_format: str
    size_bytes: int
    storage_location: StorageLocation
    shm_slot: Optional[int] = None
    disk_path: Optional[str] = None
    camera_timestamp: Optional[int] = None
    exposure_time: Optional[float] = None
    gain: Optional[float] = None

@dataclass
class CapturedFrame:
    metadata: FrameMetadata
    data: bytes
