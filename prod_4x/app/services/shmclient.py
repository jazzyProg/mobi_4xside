import logging
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

class CameraSHMClient:
    def __init__(self, shm_name: str, slot_size: int, num_slots: int):
        self.shm_name = shm_name
        self.slot_size = slot_size
        self.num_slots = num_slots
        self._shm = None

        # Пытаемся подключиться сразу при инициализации
        self.connect()

    def connect(self):
        """Подключается к существующему SHM сегменту."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            # logger.info(f"[SHM] Connected to segment {self.shm_name}")
        except FileNotFoundError:
            # Это нормально, если камера еще не запущена
            # logger.warning(f"[SHM] Segment {self.shm_name} not found. Camera not running?")
            self._shm = None
        except Exception as e:
            logger.error(f"[SHM] Connection failed: {e}")
            self._shm = None

    def read_slot(self, slot_index: int, data_size: int) -> bytes:
        """Читает байты JPEG из указанного слота."""
        # Если соединения нет, пробуем переподключиться
        if self._shm is None:
            self.connect()
            if self._shm is None:
                raise RuntimeError("No connection to Shared Memory (Camera not running?)")

        if slot_index >= self.num_slots:
            raise ValueError(f"Slot index {slot_index} out of range (max {self.num_slots-1})")

        if data_size > self.slot_size:
            raise ValueError(f"Data size {data_size} exceeds slot size {self.slot_size}")

        offset = slot_index * self.slot_size

        # Считываем данные из буфера памяти
        try:
            # Читаем срез байтов напрямую
            return bytes(self._shm.buf[offset : offset + data_size])
        except Exception as e:
            # Если сегмент был удален камерой (рестарт), буфер станет невалидным
            # Пробуем закрыть и переподключиться один раз
            self.close()
            self.connect()
            if self._shm:
                return bytes(self._shm.buf[offset : offset + data_size])
            raise e

    def close(self):
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
