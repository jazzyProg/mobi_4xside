"""
Application configuration (environment + .env).

Notes:
    - Avoid side effects at import time (do not mutate sys.path / env here).
    - SDK paths are configured during application startup (lifespan).
"""

from __future__ import annotations

import os
import socket
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # SDK / camera
    mvs_sdk_path: str = Field(default="/opt/MVS")
    camera_ip: str = Field(default="10.50.50.10")
    net_ip: str = Field(default="10.50.50.1")

    # Storage
    frame_buffer_limit_mb: int = Field(default=100, ge=10, le=2000)
    disk_storage_path: str = Field(default="/mnt/hdd/photos")
    max_disk_storage_gb: int = Field(default=5, ge=1, le=10000)
    enable_auto_cleanup: bool = Field(default=True)

    # Signals integration
    signals_api_url: str = Field(default="http://localhost:8002")
    quality_check_enabled: bool = Field(default=True)
    camera_trigger_pulse_sec: float = Field(default=1.0, ge=0.1, le=60.0)
    signal_timeout_sec: float = Field(default=0.2, ge=0.05, le=5.0)

    # SHM
    shm_name: str = Field(default="/qc_camera_ringbuffer")
    shm_slot_size: int = Field(default=20 * 1024 * 1024, ge=1024)  # 10MB
    shm_slot_count: int = Field(default=16, ge=2, le=1024)
    shm_create_mode: bool = Field(default=True)

    # Service
    service_name: str = Field(default="camera-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8003, ge=1, le=65535)
    log_level: str = Field(default="INFO")

    # API prefix (optional). Keep empty to avoid versioning.
    api_prefix: str = Field(default="")

    @property
    def mvs_sdk_py_path(self) -> str:
        return f"{self.mvs_sdk_path}/Samples/64/Python/MvImport"

    @property
    def mvs_sdk_lib_path(self) -> str:
        return f"{self.mvs_sdk_path}/lib/64"

    @field_validator("disk_storage_path")
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # Not fatal during validation; container/user might not have permissions.
                pass
        return v

    def validate(self) -> None:
        """
        Validate configuration. Raises ValueError on invalid configuration.
        """
        errors: list[str] = []

        def _validate_ip(value: str, name: str) -> None:
            try:
                socket.inet_aton(value)
            except OSError:
                errors.append(f"{name} has invalid IP format: {value}")

        _validate_ip(self.camera_ip, "camera_ip")
        _validate_ip(self.net_ip, "net_ip")

        if not (1 <= self.port <= 65535):
            errors.append(f"port must be 1-65535, got {self.port}")

        if self.shm_slot_count < 2:
            errors.append("shm_slot_count must be >= 2")

        if self.shm_slot_size < 1024:
            errors.append("shm_slot_size must be >= 1024 bytes")

        # Prefix normalization (allow '/api', '/api/v1', etc but no trailing slash)
        if self.api_prefix:
            if not self.api_prefix.startswith("/"):
                errors.append("api_prefix must start with '/' or be empty")
            if self.api_prefix != "/" and self.api_prefix.endswith("/"):
                errors.append("api_prefix must not end with '/' (use '/api', not '/api/')")

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    def setup_sdk_paths(self):
        """
        Configure SDK-related sys.path and environment variables.

        This should be called during application startup.
        """
        if os.path.isdir(self.mvs_sdk_py_path) and self.mvs_sdk_py_path not in sys.path:
            sys.path.insert(0, self.mvs_sdk_py_path)

        # MVS runtime
        os.environ["MVCAM_COMMON_RUNENV"] = f"{self.mvs_sdk_path}/lib"

        # Ensure LD_LIBRARY_PATH contains the correct directory.
        current_ld_path = os.getenv("LD_LIBRARY_PATH", "")
        if self.mvs_sdk_lib_path not in current_ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{self.mvs_sdk_lib_path}:{current_ld_path}".rstrip(":")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings accessor.
    """
    return Settings()
