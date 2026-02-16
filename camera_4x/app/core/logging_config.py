from __future__ import annotations

import logging
import logging.config
import sys

from app.core.config import get_settings

def setup_logging():
    """
    Configure application logging.
    """
    settings = get_settings()
    level_name = (settings.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": sys.stdout,
                }
            },
            "root": {"level": level, "handlers": ["stdout"]},
            "loggers": {
                # Reduce noise in container logs
                "uvicorn.access": {"level": "WARNING"},
            },
        }
    )
