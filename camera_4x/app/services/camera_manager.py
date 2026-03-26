# app/services/camera_manager.py
import uuid
import logging
from datetime import datetime
from typing import Optional, List
from threading import Lock, Thread
from collections import deque
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.config import get_settings
settings = get_settings()
from app.core.models import (
    CameraState,
    SessionInfo,
    FrameMetadata,
    CapturedFrame,
    StorageLocation
)
from app.services.frame_storage import FrameStorage
from app.services.shm_manager import ShmManager
from app.services.disk_archiver import DiskArchiver
from app.camera.worker import CameraWorker

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.state = CameraState.IDLE
        self.lock = Lock()

        self.frame_storage = FrameStorage(
            max_size_mb=settings.frame_buffer_limit_mb,
            max_items=1000  # опционально
        )

        self.shm_manager = ShmManager(
            name=settings.shm_name,
            slot_size=settings.shm_slot_size,
            slot_count=settings.shm_slot_count,
            create=settings.shm_create_mode
        )

        self.disk_archiver = DiskArchiver(
            base_path=settings.disk_storage_path,
            max_size_gb=settings.max_disk_storage_gb,
            auto_cleanup=settings.enable_auto_cleanup
        )

        self.worker_thread: Optional[CameraWorker] = None
        self.current_session: Optional[SessionInfo] = None
        self.last_error: Optional[str] = None

        logger.info("CameraManager initialized successfully")

    def start_capture(self, session_id: Optional[str] = None) -> dict:
        with self.lock:
            if self.state != CameraState.IDLE:
                raise RuntimeError(f"Cannot start capture in state {self.state.value}")

            # ===== НОВОЕ: Очистка перед началом новой сессии =====
            logger.info("Cleaning up before starting new capture session...")

            try:
                # Очистить SHM
                logger.info("Clearing SHM buffer...")
                self.shm_manager.clear_all_slots()
                logger.info("✓ SHM cleared")
            except Exception as e:
                logger.error(f"Failed to clear SHM: {e}")

            try:
                # Очистить RAM
                logger.info("Clearing RAM frame buffer...")
                self.frame_storage.clear()
                logger.info("✓ RAM cleared")
            except Exception as e:
                logger.error(f"Failed to clear RAM: {e}")

            try:
                # Очистить диск (удалить все файлы кадров)
                logger.info("Clearing disk storage...")
                self.disk_archiver.clear_all()
                logger.info("✓ Disk cleared")
            except Exception as e:
                logger.error(f"Failed to clear disk: {e}")

            try:
                self.state = CameraState.STARTING
                self.last_error = None

                if not session_id:
                    session_id = str(uuid.uuid4())

                self.current_session = SessionInfo(
                    session_id=session_id,
                    start_time=datetime.now()
                )

                self.worker_thread = CameraWorker(
                    settings=settings,
                    on_frame=self._on_frame_captured,
                    on_error=self._on_worker_error,
                    on_trigger=self._on_camera_trigger
                )
                self.worker_thread.start()

                self.state = CameraState.RUNNING
                logger.info(f"Capture started: session {session_id}")

                return {
                    "status": "started",
                    "session_id": session_id,
                    "message": "Camera capture started successfully"
                }

            except Exception as e:
                self.state = CameraState.ERROR
                self.last_error = str(e)
                logger.error(f"Failed to start capture: {e}", exc_info=True)
                raise

    def stop_capture(self) -> dict:
        with self.lock:
            if self.state not in (CameraState.RUNNING, CameraState.STARTING):
                return {
                    "status": "already_stopped",
                    "message": "Capture was not running",
                    "frames_captured": 0
                }

            self.state = CameraState.STOPPING
            frames_captured = 0

            if self.worker_thread:
                self.worker_thread.stop()

            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5.0)

                if self.worker_thread.is_alive():
                    logger.error("Worker thread did not stop gracefully within timeout")

            if self.current_session:
                self.current_session.end_time = datetime.now()
                frames_captured = self.current_session.frames_count

            self.state = CameraState.IDLE
            logger.info(f"Capture stopped, {frames_captured} frames captured")

            return {
                "status": "stopped",
                "message": "Capture stopped successfully",
                "frames_captured": frames_captured
            }

    def _emit_camera_trigger_signal(self) -> None:
        if not settings.quality_check_enabled:
            return

        try:
            url = f"{settings.signals_api_url.rstrip('/')}/signal/camera-trigger"
            query = urlencode({"duration": settings.camera_trigger_pulse_sec})
            req = Request(
                url=f"{url}?{query}",
                method="POST",
                data=b"",
            )
            with urlopen(req, timeout=settings.signal_timeout_sec):
                return
        except Exception as e:
            logger.debug("Camera trigger signal failed: %s", e)

    def _on_camera_trigger(self) -> None:
        # Non-blocking fire-and-forget to avoid slowing down frame acquisition loop.
        if not settings.quality_check_enabled:
            return

        Thread(target=self._emit_camera_trigger_signal, daemon=True).start()

    def _on_frame_captured(self, frame: CapturedFrame):
        try:
            shm_slot = self.shm_manager.push_frame(
                frame.data,
                metadata={
                    'frame_id': frame.metadata.frame_id,
                    'timestamp': frame.metadata.timestamp,
                    'width': frame.metadata.width,
                    'height': frame.metadata.height,
                    'channels': 3,
                    'pixel_format': 0,
                }
            )

            frame.metadata.shm_slot = shm_slot
            frame.metadata.storage_location = StorageLocation.SHM

            added_to_ram = self.frame_storage.add_frame(frame.metadata)

            if not added_to_ram:
                disk_path = self.disk_archiver.save_frame(frame)
                frame.metadata.disk_path = disk_path
                frame.metadata.storage_location = StorageLocation.DISK

                if self.current_session:
                    self.current_session.frames_on_disk += 1

                logger.debug(f"Frame {frame.metadata.frame_id} saved to disk (RAM full)")
            else:
                if self.current_session:
                    self.current_session.frames_in_memory += 1

            if self.current_session:
                self.current_session.frames_count += 1

        except ValueError as e:
            if "Frame too large" in str(e):
                logger.error(
                    "Frame %s exceeds SHM slot size (%s bytes). "
                    "Increase SHM_SLOT_SIZE (camera-api + qc-service) or reduce frame payload.",
                    frame.metadata.frame_id,
                    settings.shm_slot_size,
                )
            logger.error(f"Error processing frame {frame.metadata.frame_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing frame {frame.metadata.frame_id}: {e}", exc_info=True)

    def _on_worker_error(self, error: Exception):
        with self.lock:
            self.state = CameraState.ERROR
            self.last_error = str(error)
            logger.error(f"Worker thread error: {error}", exc_info=True)

    def get_latest_frames(self, count: int = 1) -> List[FrameMetadata]:
        return self.frame_storage.get_latest(count)

    def get_oldest_frame(self) -> Optional[FrameMetadata]:
        """
        Получить самый старый кадр в очереди (для FIFO обработки)

        Сначала проверяет SHM, затем RAM, затем disk
        """
        with self.lock:
            frames = self.frame_storage.get_all_frames()

            if not frames:
                return None

            # Сортируем по frame_id (возрастание) - самый старый первым
            frames.sort(key=lambda f: f.frame_id)

            # Возвращаем самый старый
            return frames[0]

    def get_frame_by_id(self, frame_id: int) -> Optional[FrameMetadata]:
        return self.frame_storage.get_by_frame_id(frame_id)

    def load_frame_data(self, metadata: FrameMetadata) -> Optional[bytes]:
        if metadata.storage_location == StorageLocation.SHM and metadata.shm_slot is not None:
            data, _ = self.shm_manager.read_slot(metadata.shm_slot)
            return data
        elif metadata.storage_location == StorageLocation.DISK and metadata.disk_path:
            return self.disk_archiver.load_frame(metadata.disk_path)
        return None

    def get_status(self) -> dict:
        storage_stats = self.frame_storage.get_stats()
        disk_stats = self.disk_archiver.get_stats()
        shm_stats = self.shm_manager.get_stats()

        session_data = {
            "session_id": None,
            "start_time": None,
            "end_time": None,
            "frames_total": 0,
            "frames_in_memory": 0,
            "frames_on_disk": 0,
        }

        if self.current_session:
            session_data = {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time.isoformat(),
                "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                "frames_total": self.current_session.frames_count,
                "frames_in_memory": self.current_session.frames_in_memory,
                "frames_on_disk": self.current_session.frames_on_disk,
            }

        return {
            "state": self.state.value,
            "session": session_data,
            "storage": {
                "ram": storage_stats,
                "disk": disk_stats,
                "shm": shm_stats,
            },
            "error": self.last_error
        }

    def delete_frame(self, frame_id: int) -> bool:
        """
        Удалить обработанный кадр и освободить ресурсы

        ИСПРАВЛЕНО: После удаления проверяет диск и загружает следующий кадр в SHM
        """
        with self.lock:
            metadata = self.frame_storage.get_frame_by_id(frame_id)

            if not metadata:
                logger.debug(f"Frame {frame_id} not found (already deleted)")
                return False

            # Запомним, был ли кадр в SHM
            was_in_shm = (metadata.storage_location == StorageLocation.SHM)
            freed_slot = metadata.shm_slot

            # 1. Удалить из storage (RAM)
            removed = self.frame_storage.remove_frame(frame_id)

            # 2. Удалить с диска (если там был)
            if metadata.storage_location == StorageLocation.DISK and metadata.disk_path:
                try:
                    disk_path = Path(metadata.disk_path)
                    if disk_path.exists():
                        disk_path.unlink()
                        logger.debug(f"Deleted frame {frame_id} from disk: {disk_path}")
                except Exception as e:
                    logger.error(f"Failed to delete frame {frame_id} from disk: {e}")

            # 3. Освободить SHM слот (если кадр был в SHM)
            if was_in_shm and freed_slot is not None:
                try:
                    # Очистить слот
                    self.shm_manager.clear_slot(freed_slot)
                    logger.debug(f"Cleared SHM slot {freed_slot} after deleting frame {frame_id}")
                except Exception as e:
                    logger.error(f"Failed to clear SHM slot {freed_slot}: {e}")

            # ===== НОВАЯ ЛОГИКА: Загрузка кадра с диска в освободившийся SHM слот =====
            if was_in_shm and freed_slot is not None:
                try:
                    # Найти самый старый кадр на диске
                    oldest_disk_frame = self._get_oldest_disk_frame()

                    if oldest_disk_frame:
                        logger.info(f"Loading frame {oldest_disk_frame.frame_id} from disk to SHM slot {freed_slot}")

                        # Загрузить данные с диска
                        disk_path = Path(oldest_disk_frame.disk_path)
                        if disk_path.exists():
                            frame_data = disk_path.read_bytes()

                            # Записать в освободившийся SHM слот
                            self.shm_manager.write_frame(
                                slot_idx=freed_slot,
                                frame_data=frame_data,
                                metadata={
                                    'frame_id': oldest_disk_frame.frame_id,
                                    'timestamp': oldest_disk_frame.timestamp,
                                    'width': oldest_disk_frame.width,
                                    'height': oldest_disk_frame.height,
                                    'channels': 3,  # Для JPEG обычно 3
                                    'pixel_format': 1,  # JPEG
                                    'data_size': len(frame_data)
                                }
                            )

                            # Обновить метаданные в RAM
                            oldest_disk_frame.storage_location = StorageLocation.SHM
                            oldest_disk_frame.shm_slot = freed_slot
                            oldest_disk_frame.disk_path = None

                            # Удалить файл с диска
                            disk_path.unlink()

                            logger.info(f"✓ Frame {oldest_disk_frame.frame_id} moved from disk to SHM slot {freed_slot}")
                        else:
                            logger.warning(f"Disk file not found: {disk_path}")

                except Exception as e:
                    logger.error(f"Failed to load frame from disk to SHM: {e}", exc_info=True)

            logger.debug(f"Frame {frame_id} deleted successfully")
            return removed

    def _get_oldest_disk_frame(self) -> Optional[FrameMetadata]:
        """
        Получить самый старый кадр, который находится на диске

        Вызывается БЕЗ lock (предполагается, что вызывающий метод уже держит lock)
        """
        all_frames = self.frame_storage.get_all_frames()

        # Фильтруем только кадры на диске
        disk_frames = [f for f in all_frames if f.storage_location == StorageLocation.DISK]

        if not disk_frames:
            return None

        # Сортируем по frame_id (возрастание) - самый старый первым
        disk_frames.sort(key=lambda f: f.frame_id)

        return disk_frames[0]


    def cleanup(self, clear_disk: bool = False):
        """
        Полная очистка Camera Manager

        Args:
            clear_disk: Если True - удалить файлы с диска
        """
        logger.info("=" * 60)
        logger.info("🧹 Cleaning up CameraManager...")
        logger.info("=" * 60)

        # 1. Остановить захват если активен
        try:
            if self.state in (CameraState.RUNNING, CameraState.STARTING):
                logger.info("Stopping capture...")
                self.stop_capture()
        except Exception as e:
            logger.error(f"Error during stop_capture in cleanup: {e}")

        # 2. Очистить RAM frame buffer
        try:
            logger.info("Clearing frame buffer...")
            self.frame_storage.clear()
            logger.info("✓ Frame buffer cleared")
        except Exception as e:
            logger.error(f"Error clearing frame buffer: {e}")

        # 3. Очистить все SHM слоты
        try:
            logger.info("Clearing SHM slots...")
            self.shm_manager.clear_all_slots()
            logger.info("✓ SHM slots cleared")
        except Exception as e:
            logger.error(f"Error clearing SHM slots: {e}")

        # 4. Удалить файлы с диска (опционально)
        if clear_disk:
            try:
                logger.info("Deleting disk files...")
                deleted = self.disk_archiver.clear_session_files()
                logger.info(f"✓ Deleted {deleted} files from disk")
            except Exception as e:
                logger.error(f"Error deleting disk files: {e}")

        # 5. Закрыть SHM
        try:
            logger.info("Closing SHM...")
            self.shm_manager.close()
            logger.info("✓ SHM closed")
        except Exception as e:
            logger.error(f"Error closing SHM: {e}")

        # 6. Сбросить состояние
        self.current_session = None
        self.last_error = None

        logger.info("=" * 60)
        logger.info("✅ Cleanup complete")
        logger.info("=" * 60)
