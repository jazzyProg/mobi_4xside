import struct
import mmap
import posix_ipc
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ShmSlotMetadata:
    frame_id: int
    timestamp: float
    camera_timestamp: float
    width: int
    height: int
    channels: int
    pixel_format: int
    data_size: int

    STRUCT_FORMAT = "=IddIIIIII"
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

class ShmManager:
    HEADER_SIZE = 64

    def __init__(self, name: str, slot_size: int, slot_count: int, create: bool = True):
        self.name = name
        self.slot_size = slot_size
        self.slot_count = slot_count
        self.create_mode = create

        self.metadata_size = ShmSlotMetadata.STRUCT_SIZE
        self.full_slot_size = self.metadata_size + self.slot_size
        self.total_size = self.HEADER_SIZE + (self.full_slot_size * slot_count)

        self._shm = None
        self._mmap = None
        self._write_index = 0
        self._init_shm()

        logger.info(f"SHM initialized: {name}, {slot_count} slots x {slot_size/1024/1024:.1f}MB")

    def _init_shm(self):
        flags = posix_ipc.O_CREAT if self.create_mode else 0
        flags |= posix_ipc.O_RDWR

        try:
            self._shm = posix_ipc.SharedMemory(
                self.name,
                flags=flags,
                mode=0o666,
                size=self.total_size if self.create_mode else 0
            )

            self._mmap = mmap.mmap(
                self._shm.fd,
                self.total_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )

            if self.create_mode:
                self._mmap[0:self.HEADER_SIZE] = b'\x00' * self.HEADER_SIZE
                struct.pack_into("=Q", self._mmap, 0, 0)
                logger.info(f"Created new SHM segment: {self.name}")

        except posix_ipc.ExistentialError as e:
            logger.error(f"SHM '{self.name}' not found. Camera service must run first.")
            raise RuntimeError(f"SHM not found: {self.name}") from e
        except Exception as e:
            logger.error(f"Failed to initialize SHM: {e}")
            raise

    def push_frame(self, frame_data: bytes, metadata: dict) -> int:
        if len(frame_data) > self.slot_size:
            raise ValueError(f"Frame too large: {len(frame_data)} > {self.slot_size}")

        slot_idx = self._write_index % self.slot_count
        slot_offset = self.HEADER_SIZE + (slot_idx * self.full_slot_size)

        meta = ShmSlotMetadata(
            frame_id=metadata.get("frame_id", 0),
            timestamp=metadata.get("timestamp", 0.0),
            camera_timestamp=float(metadata.get("camera_timestamp") or 0.0),
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            channels=metadata.get("channels", 1),
            pixel_format=metadata.get("pixel_format", 0),
            data_size=len(frame_data),
        )

        camera_ts = metadata.get("camera_timestamp", 0)

        packed_meta = struct.pack(
            ShmSlotMetadata.STRUCT_FORMAT,
            meta.frame_id,
            meta.timestamp,          # host timestamp
            float(camera_ts),        # или camera timestamp (если хочешь d)
            meta.width,
            meta.height,
            meta.channels,
            meta.pixel_format,
            meta.data_size,
            0                        # reserved (или crc/flags)
        )

        data_offset = slot_offset + self.metadata_size
        self._mmap[data_offset:data_offset + len(frame_data)] = frame_data
        self._mmap[slot_offset:slot_offset + self.metadata_size] = packed_meta

        self._write_index += 1
        struct.pack_into("=Q", self._mmap, 0, self._write_index)

        return slot_idx

    def get_write_index(self) -> int:
        try:
            return struct.unpack_from("=Q", self._mmap, 0)[0]
        except Exception as e:
            logger.error(f"Failed to read write index: {e}")
            return 0

    def read_slot(self, slot_idx: int) -> Tuple[Optional[bytes], Optional[dict]]:
        if slot_idx >= self.slot_count:
            raise ValueError(f"Invalid slot index: {slot_idx}")

        slot_offset = self.HEADER_SIZE + (slot_idx * self.full_slot_size)

        try:
            meta_bytes = self._mmap[slot_offset:slot_offset + self.metadata_size]
            unpacked = struct.unpack(ShmSlotMetadata.STRUCT_FORMAT, meta_bytes)

            metadata = {
                'frame_id': unpacked[0],
                'timestamp': unpacked[1],
                'camera_timestamp': unpacked[2],
                'width': unpacked[3],
                'height': unpacked[4],
                'channels': unpacked[5],
                'pixel_format': unpacked[6],
                'data_size': unpacked[7]
            }

            data_offset = slot_offset + self.metadata_size
            frame_data = bytes(self._mmap[data_offset:data_offset + metadata['data_size']])

            return frame_data, metadata

        except Exception as e:
            logger.error(f"Failed to read slot {slot_idx}: {e}")
            return None, None

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "slot_count": self.slot_count,
            "slot_size_mb": round(self.slot_size / 1024 / 1024, 2),
            "total_size_mb": round(self.total_size / 1024 / 1024, 2),
            "write_index": self.get_write_index(),
        }

    def clear_all_slots(self):
        """
        Очистить все слоты в SHM (обнулить данные)

        Вызывается при stop_capture для освобождения памяти
        """
        logger.info(f"Clearing all {self.slot_count} SHM slots...")
        try:
            # Обнулить все слоты (кроме header)
            slots_start = self.HEADER_SIZE
            slots_end = self.HEADER_SIZE + (self.full_slot_size * self.slot_count)

            # Записать нули во все слоты
            self._mmap[slots_start:slots_end] = b'\x00' * (slots_end - slots_start)

            # Сбросить write index
            self._write_index = 0
            struct.pack_into("=Q", self._mmap, 0, 0)

            logger.info("✓ All SHM slots cleared")
        except Exception as e:
            logger.error(f"Failed to clear SHM slots: {e}")

    def close(self):
        """Закрыть SHM (без удаления - для reader mode)"""
        if self._mmap:
            try:
                self._mmap.close()
            except Exception as e:
                logger.error(f"Failed to close mmap: {e}")

        if self._shm:
            try:
                self._shm.close_fd()
                # Удаляем SHM только если создавали его
                if self.create_mode:
                    posix_ipc.unlink_shared_memory(self.name)
                    logger.info(f"SHM unlinked: {self.name}")
            except Exception as e:
                logger.error(f"Failed to close/unlink SHM: {e}")

    def clear_slot(self, slot_idx: int):
        """
        Очистить конкретный SHM слот (заполнить нулями)

        Используется при удалении кадра для освобождения слота
        """
        if not self._mmap:
            return

        if slot_idx >= self.slot_count:
            raise ValueError(f"Invalid slot index: {slot_idx}")

        slot_offset = self.HEADER_SIZE + (slot_idx * self.full_slot_size)

        # Очищаем метаданные и данные (заполняем нулями)
        zero_data = b'\x00' * self.full_slot_size
        self._mmap[slot_offset:slot_offset + self.full_slot_size] = zero_data

        logger.debug(f"Cleared SHM slot {slot_idx}")
