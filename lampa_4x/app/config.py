from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings loaded from environment (and optional .env file).

    IMPORTANT:
    BIT_* are 0-based bit indexes.
    Example: BIT_SUCCESS=3 -> DO4.
    """

    # Modbus
    MODBUS_IP: str = Field(default="10.50.50.11")
    MODBUS_PORT: int = Field(default=502, ge=1, le=65535)
    MODBUS_SLAVE_ID: int = Field(default=1, ge=0, le=247)
    MODBUS_TIMEOUT_SEC: float = Field(default=1.0, ge=0.1, le=10.0)
    # Connect timeout (async wrapper timeout). Socket timeout is MODBUS_TIMEOUT_SEC.
    MODBUS_CONNECT_TIMEOUT_SEC: float = Field(default=3.0, ge=0.1, le=30.0)

    # Simple retry policy for modbus operations
    MODBUS_RETRIES: int = Field(default=2, ge=0, le=10)
    MODBUS_RETRY_DELAY_SEC: float = Field(default=0.15, ge=0.0, le=5.0)

    # Output bits (0-based)
    BIT_SUCCESS: int = Field(default=3, ge=0, le=15)
    BIT_FAIL: int = Field(default=2, ge=0, le=15)
    BIT_ALARM: int = Field(default=4, ge=0, le=15)
    BIT_HEARTBEAT: int = Field(default=5, ge=0, le=15)
    BIT_CAMERA_TRIGGER: int = Field(default=6, ge=0, le=15)

    # Pulse timing
    PULSE_DURATION_SEC: float = Field(default=3.0, ge=0.1, le=60.0)
    CAMERA_TRIGGER_DURATION_SEC: float = Field(default=1.0, ge=0.1, le=60.0)

    # API
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8002, ge=1, le=65535)
    LOG_LEVEL: str = Field(default="INFO")

    # Behavior
    # If True: fail container startup when Modbus is unreachable.
    # If False: start API anyway; /health will be 503 and signal endpoints will be 503 until Modbus is back.
    STRICT_STARTUP: bool = Field(default=False)

    # Optional API prefix (e.g. "/api/v1"). Keep empty if you don't need it.
    API_PREFIX: str = Field(default="")

    # On graceful shutdown: cancel pending pulses and try to turn off all configured bits.
    SHUTDOWN_CLEAR_BITS: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings instance.

    Это и есть тот самый "микро-diff", который заменяет Depends(lambda: Settings())
    на нормальную переиспользуемую dependency.
    """
    return Settings()


# Optional: keep a module-level alias for convenience
settings = get_settings()
