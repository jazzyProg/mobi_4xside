from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings loaded from environment (and optional .env file).
    """

    # Database (NO secrets in code defaults)
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432, ge=1, le=65535)
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: SecretStr = Field(default=SecretStr(""))
    DB_NAME: str = Field(default="db")

    DB_POOL_SIZE: int = Field(default=10, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=200)

    # Files sandbox root (all svg/json must be inside this directory)
    BASE_FILES_PATH: Path = Field(default=Path("/var/lib/orchestra/BD"))

    # Limits
    SEARCH_LIMIT: int = Field(default=40, ge=1, le=200)

    # API (keep backwards-compat: APP_HOST/APP_PORT also accepted)
    HOST: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST", "APP_HOST"))
    PORT: int = Field(default=8000, ge=1, le=65535, validation_alias=AliasChoices("PORT", "APP_PORT"))
    LOG_LEVEL: str = Field(default="INFO")
    API_PREFIX: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @property
    def database_url(self) -> str:
        pwd = self.DB_PASSWORD.get_secret_value()
        if pwd:
            return f"postgresql+asyncpg://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return f"postgresql+asyncpg://{self.DB_USER}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def files_root(self) -> Path:
        return self.BASE_FILES_PATH.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
