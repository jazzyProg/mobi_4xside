# app/services/shm_reader.py
import struct
import mmap
import posix_ipc
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ShmSlotMetadata:
    frame_id: int
    timestamp: float
    width: int
    height: int
    channels: int
    pixel_format: int
    data_size: int

    STRUCT_FORMAT = "=IddIIIII"
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)


class CameraSHMReader:
    """
    Reader для SHM созданного Camera API (posix_ipc)
    Совместим со структурой ShmManager из Camera Service
    """
    HEADER_SIZE = 64

    def __init__(self, name: str, slot_size: int, slot_count: int):
        self.name = name
        self.slot_size = slot_size
        self.slot_count = slot_count

        self.metadata_size = ShmSlotMetadata.STRUCT_SIZE
        self.full_slot_size = self.metadata_size + self.slot_size
        self.total_size = self.HEADER_SIZE + (self.full_slot_size * slot_count)

        self._shm = None
        self._mmap = None
        self._connect()

    def _connect(self):
        """Подключиться к существующему SHM сегменту (только чтение)"""
        try:
            # Открываем существующий сегмент (без O_CREAT)
            self._shm = posix_ipc.SharedMemory(
                self.name,
                flags=posix_ipc.O_RDWR,  # Только открытие, без создания
                mode=0o666
            )

            self._mmap = mmap.mmap(
                self._shm.fd,
                self.total_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ  # Только чтение
            )

            logger.info(f"✓ Connected to SHM: {self.name}")

        except posix_ipc.ExistentialError:
            logger.error(f"SHM '{self.name}' not found. Camera service not running?")
            self._shm = None
            self._mmap = None
            raise RuntimeError(f"SHM not found: {self.name}")
        except Exception as e:
            logger.error(f"Failed to connect to SHM: {e}")
            self._shm = None
            self._mmap = None
            raise

    def read_slot(self, slot_idx: int) -> Tuple[Optional[bytes], Optional[dict]]:
        """
        Читает данные из указанного слота

        Returns:
            (frame_data, metadata) или (None, None) при ошибке
        """
        if self._mmap is None:
            raise RuntimeError("Not connected to SHM")

        # Проверка на закрытый mmap
        try:
            if self._mmap.closed:
                logger.warning("SHM mmap is closed")
                return None, None
        except (ValueError, AttributeError):
            logger.warning("SHM mmap is invalid or closed")
            return None, None

        if slot_idx >= self.slot_count:
            raise ValueError(f"Invalid slot index: {slot_idx}")

        slot_offset = self.HEADER_SIZE + (slot_idx * self.full_slot_size)

        try:
            # Читаем метаданные
            meta_bytes = self._mmap[slot_offset:slot_offset + self.metadata_size]
            unpacked = struct.unpack(ShmSlotMetadata.STRUCT_FORMAT, meta_bytes)

            metadata = {
                'frame_id': unpacked[0],
                'timestamp': unpacked[1],
                'width': unpacked[3],
                'height': unpacked[4],
                'channels': unpacked[5],
                'pixel_format': unpacked[6],
                'data_size': unpacked[7]
            }

            # Читаем данные кадра
            data_offset = slot_offset + self.metadata_size
            frame_data = bytes(self._mmap[data_offset:data_offset + metadata['data_size']])

            return frame_data, metadata

        except Exception as e:
            logger.error(f"Failed to read slot {slot_idx}: {e}")
            return None, None


    def get_write_index(self) -> int:
        """Получить текущий write index из заголовка"""
        if self._mmap is None:
            return 0
        try:
            return struct.unpack_from("=Q", self._mmap, 0)[0]
        except Exception as e:
            logger.error(f"Failed to read write index: {e}")
            return 0

    def close(self):
        """Закрыть соединение с SHM"""
        # Безопасная проверка mmap (может быть уже закрыт)
        try:
            if self._mmap and not self._mmap.closed:
                self._mmap.close()
                logger.info(f"✓ Closed mmap for SHM: {self.name}")
        except (ValueError, AttributeError) as e:
            logger.debug(f"mmap already closed or invalid: {e}")
        except Exception as e:
            logger.error(f"Failed to close mmap: {e}")

        # Закрыть file descriptor
        if self._shm:
            try:
                self._shm.close_fd()
                logger.info(f"✓ Disconnected from SHM: {self.name}")
            except Exception as e:
                logger.error(f"Failed to close SHM: {e}")

        # Обнулить ссылки
        self._mmap = None
        self._shm = None
