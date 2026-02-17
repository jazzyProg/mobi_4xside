# app/camera/worker.py
import time
import threading
import logging
from typing import Callable, Optional
from ctypes import *
import numpy as np
import cv2

from app.core.models import CapturedFrame, FrameMetadata, StorageLocation

logger = logging.getLogger(__name__)

try:
    from MvCameraControl_class import *
    SDK_AVAILABLE = True
    logger.info("MVS SDK loaded successfully")
except ImportError:
    logger.warning("MvCameraControl_class not found. Will use mock camera for testing.")
    SDK_AVAILABLE = False
    MvCamera = None

class CameraWorker(threading.Thread):
    def __init__(
        self,
        settings,
        on_frame: Callable[[CapturedFrame], None],
        on_error: Callable[[Exception], None],
        on_trigger: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=True, name="CameraWorker")
        self.settings = settings
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_trigger = on_trigger

        self.stop_event = threading.Event()
        self.frame_id = 0
        self.cam = None
        self.payload_size = 0
        self.session_id = ""

    def run(self):
        try:
            if SDK_AVAILABLE:
                self._run_real_camera()
            else:
                self._run_mock_camera()
        except Exception as e:
            logger.error(f"Worker thread exception: {e}", exc_info=True)
            self.on_error(e)
        finally:
            self._disconnect_camera()

    def _run_real_camera(self):
        """Главный цикл с реальной камерой - polling mode"""
        with self._sdk_context():
            self._connect_camera()
            self._start_capture()

            # Polling mode - активно запрашиваем кадры
            while not self.stop_event.is_set():
                stOutFrame = MV_FRAME_OUT()
                memset(byref(stOutFrame), 0, sizeof(stOutFrame))

                # Запрашиваем кадр с таймаутом 1 секунда
                ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
                if ret == 0:
                    # Кадр получен успешно
                    self._process_frame_buffer(stOutFrame)
                elif ret != 0x80000007:  # 0x80000007 = timeout, это норма
                    logger.debug(f"GetImageBuffer error: 0x{ret:x}")

    def _run_mock_camera(self):
        """Mock режим для тестирования"""
        logger.info("Running in MOCK mode (no real camera)")
        while not self.stop_event.is_set():
            if self.on_trigger:
                self.on_trigger()
            frame = self._create_mock_frame()
            if frame:
                self.on_frame(frame)
            time.sleep(0.033)  # ~30 FPS

    def _process_frame_buffer(self, frame_out):
        """Обработка кадра от SDK"""
        try:
            if not frame_out.pBufAddr or frame_out.stFrameInfo.nFrameLen == 0:
                return

            info = frame_out.stFrameInfo
            p_data = frame_out.pBufAddr
            data_len = info.nFrameLen
            width = info.nWidth
            height = info.nHeight

            if self.on_trigger:
                self.on_trigger()

            # Копируем данные из C-памяти
            raw_data = string_at(p_data, data_len)

            # Конвертируем в NumPy (монохром 8-бит)
            np_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))

            # Кодируем в JPEG
            success, jpeg_buffer = cv2.imencode('.jpg', np_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                logger.error("Failed to encode frame to JPEG")
                return

            jpeg_data = jpeg_buffer.tobytes()

            self.frame_id += 1
            timestamp = time.time()

            # Создаем метаданные БЕЗ выдуманных полей
            metadata = FrameMetadata(
                frame_id=self.frame_id,
                session_id=self.session_id,
                timestamp=timestamp,
                width=width,
                height=height,
                pixel_format="jpg",  # НЕ "jpeg"!
                size_bytes=len(jpeg_data),
                storage_location=StorageLocation.SHM,
                shm_slot=None,  # Заполнит camera_manager
                disk_path=None,
                camera_timestamp=None,  # НЕ выдумываем
                exposure_time=None,     # НЕ выдумываем
                gain=None               # НЕ выдумываем
            )

            frame = CapturedFrame(metadata=metadata, data=jpeg_data)
            self.on_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame {self.frame_id}: {e}", exc_info=True)
        finally:
            if self.cam:
                self.cam.MV_CC_FreeImageBuffer(frame_out)

    def _sdk_context(self):
        """Context manager для SDK"""
        class _Ctx:
            def __enter__(_):
                ret = MvCamera.MV_CC_Initialize()
                if ret != 0:
                    logger.warning(f"MV_CC_Initialize returned {ret}")
                return _

            def __exit__(_, exc_type, exc, tb):
                MvCamera.MV_CC_Finalize()

        return _Ctx()

    def _connect_camera(self):
        """Подключение к камере по IP (твой метод)"""
        if not SDK_AVAILABLE or MvCamera is None:
            return

        self.cam = MvCamera()

        # Создаем структуры для GigE
        stDevInfo = MV_CC_DEVICE_INFO()
        memset(byref(stDevInfo), 0, sizeof(stDevInfo))

        stGigEDev = MV_GIGE_DEVICE_INFO()
        memset(byref(stGigEDev), 0, sizeof(stGigEDev))

        # IP адреса
        stGigEDev.nCurrentIp = self._ip_to_int(self.settings.camera_ip)
        stGigEDev.nNetExport = self._ip_to_int(self.settings.net_ip)

        stDevInfo.nTLayerType = MV_GIGE_DEVICE
        stDevInfo.SpecialInfo.stGigEInfo = stGigEDev

        # Создаем handle
        ret = self.cam.MV_CC_CreateHandle(stDevInfo)
        if ret != 0:
            raise RuntimeError(f"CreateHandle failed: 0x{ret:x}")

        # Открываем в эксклюзивном режиме
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"OpenDevice failed: 0x{ret:x}")

        logger.info(f"Camera connected: {self.settings.camera_ip}")

        # Оптимальный размер пакета
        nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            logger.info(f"Packet size: {nPacketSize}")

        # ВСЕ ТВОИ НАСТРОЙКИ ТРИГГЕРОВ
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        logger.info("TriggerMode: ON")

        self.cam.MV_CC_SetEnumValueByString("TriggerSelector", "FrameBurstStart")
        self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
        logger.info("FrameBurstStart trigger: Line0")

        self.cam.MV_CC_SetEnumValueByString("TriggerSelector", "LineStart")
        self.cam.MV_CC_SetEnumValueByString("TriggerSource", "EncoderConverter")
        logger.info("LineStart trigger: EncoderConverter")

        # Payload size (ПРАВИЛЬНО)
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(stParam))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret == 0:
            self.payload_size = stParam.nCurValue
            logger.info(f"Payload size: {self.payload_size} bytes ({self.payload_size/1024/1024:.1f} MB)")
        else:
            logger.warning(f"Failed to get payload size: 0x{ret:x}")

    def _start_capture(self):
        """Запуск захвата"""
        if not SDK_AVAILABLE or self.cam is None:
            return

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing failed: 0x{ret:x}")

        logger.info("Camera grabbing started (Polling Mode)")

    def _disconnect_camera(self):
        """Отключение камеры"""
        if not SDK_AVAILABLE or self.cam is None:
            return

        try:
            self.cam.MV_CC_StopGrabbing()
            logger.info("Camera grabbing stopped")
        except Exception as e:
            logger.error(f"Error stopping: {e}")

        try:
            self.cam.MV_CC_CloseDevice()
            logger.info("Camera closed")
        except Exception as e:
            logger.error(f"Error closing: {e}")

        try:
            self.cam.MV_CC_DestroyHandle()
            logger.info("Camera handle destroyed")
        except Exception as e:
            logger.error(f"Error destroying: {e}")

    def _create_mock_frame(self) -> CapturedFrame:
        """Mock кадр"""
        self.frame_id += 1

        # ТВОЙ РАЗМЕР: 8192x8192
        width, height = 8192, 8192
        img = np.random.randint(0, 255, (height, width), dtype=np.uint8)

        success, jpeg_buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpeg_data = jpeg_buffer.tobytes()

        timestamp = time.time()

        metadata = FrameMetadata(
            frame_id=self.frame_id,
            session_id="mock_session",
            timestamp=timestamp,
            width=width,
            height=height,
            pixel_format="jpg",
            size_bytes=len(jpeg_data),
            storage_location=StorageLocation.SHM,
            shm_slot=None,
            disk_path=None,
            camera_timestamp=None,  # НЕ выдумываем
            exposure_time=None,     # НЕ выдумываем
            gain=None               # НЕ выдумываем
        )

        return CapturedFrame(metadata=metadata, data=jpeg_data)

    @staticmethod
    def _ip_to_int(ip: str) -> int:
        """IP в int"""
        if not ip:
            return 0
        try:
            a, b, c, d = map(int, ip.split("."))
            return (a << 24) | (b << 16) | (c << 8) | d
        except (ValueError, AttributeError):
            return 0

    def stop(self):
        """Остановка"""
        logger.info("Stopping camera worker...")
        self.stop_event.set()
