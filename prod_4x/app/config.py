"""app.config

Application configuration loaded from environment variables (and optional `.env` file).

This module keeps backward compatibility with the previous `settings = Settings()` global.
Prefer `get_settings()` for dependency injection and testability.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """QC service settings.

    Note:
        Values can be overridden via environment variables.
    """

    # ===== External services =====
    PRODUCTS_API_URL: str = Field(default="http://localhost:8000")
    SIGNALS_API_URL: str = Field(default="http://localhost:8002")
    # Camera service has its own API prefix; keep default to preserve current deployments.
    CAMERA_API_URL: str = Field(default="http://localhost:8003")

    # ===== Paths =====
    MODEL_PATH: Path = Field(default=Path("/app/models/best.pt"))
    QUARANTINE_DIR: Path = Field(default=Path("/app/data/quarantine"))
    CAD_DIR: Path = Field(default=Path("/app/data/cad"))

    # ===== Processing =====
    DPI: int = Field(default=150, ge=1, le=1200)
    THICKNESS_MM: float = Field(default=3.0, ge=0.0, le=100.0)

    # ===== API =====
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8001, ge=1, le=65535)
    API_TITLE: str = Field(default="QC API Service")
    API_VERSION: str = Field(default="2.0.0")
    LOG_LEVEL: str = Field(default="INFO")

    # ===== Timeouts =====
    API_TIMEOUT: float = Field(default=30.0, ge=0.1, le=300.0)
    API_RETRY_ATTEMPTS: int = Field(default=3, ge=0, le=20)
    SIGNAL_TIMEOUT: float = Field(default=5.0, ge=0.1, le=60.0)

    # ===== Signal behavior =====
    QUALITY_CHECK_ENABLED: bool = Field(default=True)
    CAMERA_TRIGGER_PULSE_SEC: float = Field(default=1.0, ge=0.1, le=60.0)

    # ===== Camera / SHM =====
    # Keep as explicit URL for FIFO endpoint to preserve old code paths.
    CAMERA_METADATA_URL: str = Field(default="http://localhost:8003/frames/oldest/meta")
    CAMERA_SHM_NAME: str = Field(default="/qc_camera_ringbuffer")
    SHM_SLOT_SIZE: int = Field(default=20 * 1024 * 1024, ge=1024)  # bytes
    SHM_NUM_SLOTS: int = Field(default=16, ge=1, le=1024)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


# Backward compatibility: old code imports `settings`.
settings = get_settings()
