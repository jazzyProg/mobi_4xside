# app/services/frame_storage.py
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from app.core.models import FrameMetadata, StorageLocation

logger = logging.getLogger(__name__)


class FrameStorage:
    """
    In-memory хранилище метаданных кадров с ограничением по размеру

    Хранит только метаданные, сами данные в SHM или на диске
    """

    def __init__(self, max_size_mb: int = 100, max_items: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items

        self.frames: Dict[int, FrameMetadata] = {}  # frame_id -> metadata
        self.current_size_bytes = 0
        self.lock = threading.RLock()

        logger.info(f"FrameStorage: max {max_size_mb}MB in RAM, max {max_items} items")

    def add_frame(self, metadata: FrameMetadata) -> bool:
        """
        Добавить метаданные кадра

        Returns:
            True если добавлен, False если лимит превышен
        """
        with self.lock:
            # Проверка лимита по количеству
            if len(self.frames) >= self.max_items:
                logger.warning(f"Frame storage full: {len(self.frames)} items")
                return False

            # Проверка лимита по размеру
            if self.current_size_bytes + metadata.size_bytes > self.max_size_bytes:
                logger.warning(
                    f"Frame storage size limit exceeded: "
                    f"{self.current_size_bytes / 1024 / 1024:.1f}MB / "
                    f"{self.max_size_bytes / 1024 / 1024:.1f}MB"
                )
                return False

            self.frames[metadata.frame_id] = metadata
            self.current_size_bytes += metadata.size_bytes

            logger.debug(
                f"Added frame {metadata.frame_id} to storage "
                f"({len(self.frames)} items, "
                f"{self.current_size_bytes / 1024 / 1024:.1f}MB)"
            )

            return True

    def get_frame_by_id(self, frame_id: int) -> Optional[FrameMetadata]:
        """Получить метаданные кадра по ID"""
        with self.lock:
            return self.frames.get(frame_id)

    def get_latest_frames(self, count: int) -> List[FrameMetadata]:
        """
        Получить последние N кадров (отсортированы по frame_id убывание)
        """
        with self.lock:
            sorted_frames = sorted(
                self.frames.values(),
                key=lambda f: f.frame_id,
                reverse=True
            )
            return sorted_frames[:count]

    def get_all_frames(self) -> List[FrameMetadata]:
        """
        Получить все кадры из хранилища (для поиска самого старого/нового)

        Returns:
            List[FrameMetadata]: Список всех кадров в памяти
        """
        with self.lock:
            # Возвращаем копию списка, чтобы избежать проблем с многопоточностью
            return list(self.frames.values())

    def remove_frame(self, frame_id: int) -> bool:
        """
        Удалить метаданные кадра из хранилища

        Returns:
            True если удален, False если не найден
        """
        with self.lock:
            metadata = self.frames.pop(frame_id, None)

            if metadata:
                self.current_size_bytes -= metadata.size_bytes
                logger.debug(
                    f"Removed frame {frame_id} from storage "
                    f"({len(self.frames)} items remain, "
                    f"{self.current_size_bytes / 1024 / 1024:.1f}MB)"
                )
                return True

            return False

    def update_frame_location(
        self,
        frame_id: int,
        location: StorageLocation,
        **kwargs
    ) -> bool:
        """
        Обновить местоположение кадра (например, после переноса на диск)

        Args:
            frame_id: ID кадра
            location: Новое местоположение
            **kwargs: Дополнительные поля для обновления (disk_path, shm_slot и т.д.)
        """
        with self.lock:
            metadata = self.frames.get(frame_id)

            if not metadata:
                return False

            metadata.storage_location = location

            # Обновить дополнительные поля
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            logger.debug(f"Updated frame {frame_id} location to {location}")
            return True

    def get_stats(self) -> dict:
        """Получить статистику хранилища"""
        with self.lock:
            return {
                "frames_count": len(self.frames),
                "size_bytes": self.current_size_bytes,
                "size_mb": round(self.current_size_bytes / 1024 / 1024, 2),
                "max_size_mb": round(self.max_size_bytes / 1024 / 1024, 2),
                "usage_percent": round(
                    (self.current_size_bytes / self.max_size_bytes) * 100, 1
                ) if self.max_size_bytes > 0 else 0,
                "max_items": self.max_items
            }

    def clear(self):
        """Очистить все кадры из хранилища"""
        with self.lock:
            count = len(self.frames)
            self.frames.clear()
            self.current_size_bytes = 0
            logger.info(f"Cleared frame storage ({count} frames removed)")

    def get_latest(self, count: int = 1):
        return self.get_latest_frames(count)
