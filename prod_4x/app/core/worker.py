"""app.core.worker

Background processing loop for reading frames from Camera service (via SHM) and
running QC pipeline.

This module intentionally stays synchronous because it runs in a dedicated thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.core.api_client import ExternalServicesClient, get_api_client
from app.core.state import state
from app.services.shm_reader import CameraSHMReader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatConfig:
    interval_sec: float = 3600.0
    duration_sec: float = 1.0


class PeriodicHeartbeat:
    """Small helper around `threading.Timer` to avoid timer leaks."""

    def __init__(self, client: ExternalServicesClient, cfg: HeartbeatConfig):
        self._client = client
        self._cfg = cfg
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._timer is not None:
                return
            self._schedule_next()

    def stop(self) -> None:
        with self._lock:
            if self._timer is None:
                return
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._cfg.interval_sec, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        if state.stop_event.is_set():
            return

        try:
            log.debug("Sending periodic heartbeat")
            self._client.signal_heartbeat(duration=self._cfg.duration_sec)
        except Exception as e:
            log.warning("Heartbeat failed: %s", e)

        with self._lock:
            if self._timer is None:
                return
            self._schedule_next()


def _resolve_camera_shm_name(settings: Settings, client: ExternalServicesClient) -> str:
    """Resolve SHM name from Camera /status with fallback to config."""
    try:
        status = client.get_camera_status()
        shm_info = (status.get("storage") or {}).get("shm") or {}
        shm_name = shm_info.get("name")
        if shm_name:
            log.info("Got SHM name from Camera API: %s", shm_name)
            return str(shm_name)
        log.warning("SHM name not found in Camera status response, using fallback")
    except Exception as e:
        log.warning("Failed to fetch Camera SHM name: %s. Using fallback.", e)

    return settings.CAMERA_SHM_NAME


def _ensure_shm_reader(settings: Settings, shm_name: str) -> CameraSHMReader:
    if state.shm_client is None:
        log.info("Initializing SHM reader: %s", shm_name)
        state.shm_client = CameraSHMReader(
            name=shm_name,
            slot_size=settings.SHM_SLOT_SIZE,
            slot_count=settings.SHM_NUM_SLOTS,
        )
    return state.shm_client


def _cleanup_shm_reader() -> None:
    if not state.shm_client:
        return
    try:
        state.shm_client.close()
    except Exception as e:
        log.error("Error closing SHM client: %s", e)
    finally:
        state.shm_client = None


def qc_loop(settings: Settings | None = None) -> None:
    """Main background loop.

    Reads *oldest* frame metadata (FIFO), then reads bytes from SHM slot and runs
    QC pipeline. After processing, the frame is deleted in Camera service to free SHM slot.
    """
    settings = settings or get_settings()
    log.info("QC Loop started")

    api_client = get_api_client()

    from app.core import pipeline
    from app.core.accumulator import increment_counters, reset_alarm, reset_counters_on_pass

    shm_name = _resolve_camera_shm_name(settings, api_client)
    heartbeat = PeriodicHeartbeat(api_client, HeartbeatConfig())
    heartbeat.start()

    while not state.stop_event.is_set():
        try:
            # 1) Fetch oldest frame metadata (FIFO)
            try:
                frame_meta = api_client.get_oldest_frame_metadata()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    time.sleep(0.5)
                    continue
                log.warning("Unexpected status from camera API: %s", e.response.status_code)
                time.sleep(0.5)
                continue
            except Exception:
                time.sleep(1.0)
                continue

            fid = frame_meta.get("frame_id")
            if fid is None:
                log.debug("Skipping: frame_id is None")
                time.sleep(0.1)
                continue

            # Duplicate guard
            if fid <= state.last_processed_id:
                log.debug("Skipping frame %s: already processed (last_processed_id=%s)", fid, state.last_processed_id)
                time.sleep(0.05)
                continue

            storage_loc = frame_meta.get("storage_location")
            if storage_loc != "shm":
                log.warning("Skipping frame %s: storage_location=%s (expecting 'shm')", fid, storage_loc)
                time.sleep(0.1)
                continue

            shm_slot = frame_meta.get("shm_slot")
            if shm_slot is None:
                log.warning("Skipping frame %s: shm_slot is None (storage_location=%s)", fid, storage_loc)
                time.sleep(0.1)
                continue

            log.info("Attempting to read frame %s from SHM slot %s", fid, shm_slot)

            # 2) Read bytes from SHM
            reader = _ensure_shm_reader(settings, shm_name)
            frame_data, _slot_md = reader.read_slot(int(shm_slot))

            if not frame_data:
                log.warning(
                    "Skipping frame %s: SHM slot %s returned empty data, deleting frame",
                    fid, shm_slot
                )
                # УДАЛЯЕМ фрейм чтобы не зацикливаться
                try:
                    api_client.delete_camera_frame(int(fid))
                    log.info("Frame %s deleted from queue", fid)
                except Exception as e:
                    log.error("Failed to delete frame %s: %s", fid, e)

                time.sleep(0.1)
                continue

            # ДИАГНОСТИКА
            log.info(
                f"Frame {fid}: read {len(frame_data)} bytes, "
                f"first 32 bytes: {frame_data[:32].hex(' ')}"
            )

            # ============ ИСПРАВЛЕНИЕ: Удаление padding в начале ============
            # Проверяем если начинается с нулей - ищем настоящее начало JPEG
            if frame_data[:2] != b'\xff\xd8':
                # Ищем маркер JPEG SOI (Start Of Image)
                jpeg_start = frame_data.find(b'\xff\xd8')
                if jpeg_start > 0 and jpeg_start < 100:  # Padding не больше 100 байт
                    log.warning(
                        f"Frame {fid}: JPEG starts at offset {jpeg_start}, "
                        f"stripping {jpeg_start} padding bytes"
                    )
                    frame_data = frame_data[jpeg_start:]
                    log.info(f"Frame {fid}: after strip, first 16 bytes: {frame_data[:16].hex(' ')}")
                else:
                    log.error(
                        f"Frame {fid}: Invalid data - no JPEG header found! "
                        f"First 32 bytes: {frame_data[:32].hex(' ')}"
                    )
                    # Удаляем фрейм
                    try:
                        api_client.delete_camera_frame(int(fid))
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue

            # Теперь frame_data точно начинается с ff d8
            # ============ КОНЕЦ ИСПРАВЛЕНИЯ ============

            # 3) Process frame
            state.stats["last_status"] = "processing"
            log.info("Processing frame %s (oldest in queue)", fid)

            qc_ok, _vision_json, report = pipeline.process_single(
                frame_data,  # ← Теперь правильные данные без padding!
                modelpath=settings.MODEL_PATH,
                quarantinedir=settings.QUARANTINE_DIR,
                stemname=f"cam_{fid}",
                session_id=f"auto_{fid}",
            )

            state.last_processed_id = int(fid)
            state.stats["processed"] += 1

            # 4) Update stats + signals + accumulator
            if qc_ok:
                state.stats["passed"] += 1
                state.stats["last_status"] = "PASS"
                try:
                    reset_counters_on_pass()
                    reset_alarm()
                except Exception as e:
                    log.error("Accumulator reset failed: %s", e)

                try:
                    api_client.signal_success(duration=3.0)
                except Exception as e:
                    log.warning("Failed to send success signal: %s", e)
            else:
                state.stats["failed"] += 1
                state.stats["last_status"] = "FAIL"
                try:
                    increment_counters(report)
                except Exception as e:
                    log.error("Accumulator increment failed: %s", e)

                try:
                    api_client.signal_fail()
                except Exception as e:
                    log.warning("Failed to send fail signal: %s", e)

            # 5) Delete frame to free slot
            try:
                api_client.delete_camera_frame(int(fid))
            except Exception as e:
                log.error("Failed to delete frame %s: %s", fid, e)
                time.sleep(1.0)

        except Exception as e:
            log.error("Loop error: %s", e, exc_info=True)
            time.sleep(1.0)

    heartbeat.stop()
    _cleanup_shm_reader()
    log.info("QC Loop stopped")
