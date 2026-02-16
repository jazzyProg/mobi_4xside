from typing import Optional
from app.services.camera_manager import CameraManager
import logging

logger = logging.getLogger(__name__)

_camera_manager: Optional[CameraManager] = None

def get_camera_manager() -> CameraManager:
    """
    Singleton Camera Manager с поддержкой cleanup при shutdown
    """
    global _camera_manager
    if _camera_manager is None:
        logger.info("Initializing Camera Manager...")
        _camera_manager = CameraManager()
    return _camera_manager


def cleanup_camera_manager():
    """
    Полная очистка Camera Manager

    Вызывается при:
    - Shutdown приложения (lifespan)
    - Signal handlers (SIGTERM, SIGINT)
    """
    global _camera_manager
    if _camera_manager is not None:
        logger.info("Cleaning up Camera Manager...")
        try:
            # Очистить SHM и память
            _camera_manager.cleanup()
            logger.info("✓ Camera Manager cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup Camera Manager: {e}")
        finally:
            _camera_manager = None
