"""
Глобальное состояние QC Service
"""
from typing import Optional
import threading


class QCEngineState:
    """Глобальное состояние сервиса"""
    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.shm_client = None  # Optional[CameraSHMClient]
        self.last_processed_id = -1
        self.stats = {
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "last_status": "idle"
        }


# Singleton instance
state = QCEngineState()
