"""
Система накопления ошибок и alarm сигналов
"""
import threading
import logging
import json
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.core.api_client import get_api_client
from app.core.state import state

logger = logging.getLogger(__name__)

# ============================================
# Пороги накопления для alarm
# ============================================
THRESHOLDS = {
    "diameter": 5,            # 5 ошибок по диаметру → alarm
    "holes": 5,               # 5 ошибок по отверстиям → alarm
    "rectangles": 5,          # 5 ошибок по прямоугольникам → alarm
    "general": 500000,         # 5 общих ошибок → alarm
}

# ============================================
# Global state для накопления
# ============================================
class AccumulatorState:
    def __init__(self):
        self.counters: Dict[str, int] = {
            "diameter": 0,
            "holes": 0,
            "rectangles": 0,
            "general": 0,
        }
        self.last_increment: Dict[str, bool] = {
            "diameter": False,
            "holes": False,
            "rectangles": False,
            "general": False,
        }
        self.lock = threading.Lock()
        self.alarm_active = False
        self.shift_key = ""
        self.sample_frames: Dict[str, list[dict[str, Any]]] = {
            "diameter": [],
            "holes": [],
            "rectangles": [],
        }
        self.artifact_saved_for_streak: Dict[str, bool] = {
            "diameter": False,
            "holes": False,
            "rectangles": False,
        }

accumulator = AccumulatorState()


def _build_shift_key(now: datetime) -> str:
    day_start = time(7, 0, 0)
    night_start = time(19, 0, 0)
    suffix = "day" if day_start <= now.time() < night_start else "night"
    return f"{now.strftime('%Y%m%d')}_{suffix}"


def _reset_shift_if_needed() -> None:
    now = datetime.now(ZoneInfo(settings.TIMEZONE))
    shift_key = _build_shift_key(now)
    if accumulator.shift_key == shift_key:
        return
    logger.info("ACCUM: shift changed %s -> %s, resetting accumulated counters", accumulator.shift_key, shift_key)
    for key in accumulator.counters:
        accumulator.counters[key] = 0
        accumulator.last_increment[key] = False
    accumulator.alarm_active = False
    accumulator.shift_key = shift_key
    for key in accumulator.sample_frames:
        accumulator.sample_frames[key] = []
        accumulator.artifact_saved_for_streak[key] = False


def _save_accumulated_failure_artifact(error_type: str, count: int, samples: list[dict[str, Any]]) -> None:
    """Persist single aggregated artifact folder: 5 photos + JSON reason."""
    if error_type == "general" or not samples:
        return

    try:
        api_client = get_api_client()
        active_product = api_client.get_active_product()
    except Exception:
        active_product = {}

    now = datetime.now(ZoneInfo(settings.TIMEZONE))
    product_name = str(active_product.get("product_name", "unknown"))
    folder_name = f"{product_name}_{now.strftime('%Y%m%d_%H%M%S')}_{error_type}_accumulated"

    out_dir = Path(settings.BRAK_BASE_DIR) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples[:5], start=1):
        frame_id = sample.get("frame_id", "unknown")
        img_data: bytes = sample.get("frame_data", b"")
        (out_dir / f"detail_{idx:02d}_frame_{frame_id}.jpg").write_bytes(img_data)

    payload = {
        "product_name": product_name,
        "reason": error_type,
        "accumulated_count": count,
        "timestamp": now.isoformat(),
        "samples_count": min(len(samples), 5),
        "sample_frames": [sample.get("frame_id") for sample in samples[:5]],
    }
    (out_dir / "defect_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_qc_report(report: dict) -> list[str]:
    """
    Определить какие типы ошибок произошли из QC отчета

    Returns:
        Список ключей для инкремента (например: ['diameter', 'holes'])
    """
    errors = []

    # Проверяем отчет на конкретные ошибки
    if not report.get('ok', True):
        # Проверка ошибок диаметра
        if 'diameters_ok' in report and not report['diameters_ok']:
            errors.append('diameter')

        # Проверка ошибок отверстий
        if 'holes_ok' in report and not report['holes_ok']:
            errors.append('holes')

        # Проверка ошибок прямоугольников
        if 'rectangles_ok' in report and not report['rectangles_ok']:
            errors.append('rectangles')

        # Общая ошибка (всегда добавляем при любом FAIL)
        errors.append('general')

    return errors


def increment_counters(report: dict, frame_data: bytes | None = None, frame_id: int | None = None) -> None:
    """
    Инкрементировать счетчики на основе QC отчета при FAIL
    """
    errors = parse_qc_report(report)

    if not errors:
        return

    with accumulator.lock:
        _reset_shift_if_needed()
        for key in errors:
            accumulator.counters[key] += 1
            accumulator.last_increment[key] = True

            logger.info(f"ACCUM: {key} incremented to {accumulator.counters[key]}/{THRESHOLDS.get(key, 999)}")
            if key in accumulator.sample_frames and frame_data:
                accumulator.sample_frames[key].append(
                    {
                        "frame_id": frame_id,
                        "frame_data": frame_data,
                        "timestamp": datetime.now(ZoneInfo(settings.TIMEZONE)).isoformat(),
                    }
                )
                if len(accumulator.sample_frames[key]) > 5:
                    accumulator.sample_frames[key] = accumulator.sample_frames[key][-5:]

            # Проверка превышения порога
            threshold = THRESHOLDS.get(key)
            if threshold and accumulator.counters[key] >= threshold:
                if key in accumulator.sample_frames and not accumulator.artifact_saved_for_streak[key]:
                    _save_accumulated_failure_artifact(
                        key,
                        accumulator.counters[key],
                        accumulator.sample_frames[key],
                    )
                    accumulator.artifact_saved_for_streak[key] = True
                trigger_alarm(key, accumulator.counters[key])
        state.stats["shift"]["accumulated_failures_total"] += 1


def reset_counters_on_pass() -> None:
    """
    Сбросить счетчики которые были инкрементированы перед последним PASS
    """
    with accumulator.lock:
        _reset_shift_if_needed()
        for key in list(accumulator.counters.keys()):
            if accumulator.last_increment.get(key, False):
                logger.info(f"ACCUM: reset {key} after PASS (was {accumulator.counters[key]})")
                accumulator.counters[key] = 0
                accumulator.last_increment[key] = False
                if key in accumulator.sample_frames:
                    accumulator.sample_frames[key] = []
                    accumulator.artifact_saved_for_streak[key] = False


def trigger_alarm(error_type: str, count: int) -> None:
    """
    Отправить alarm сигнал при превышении порога
    """
    if accumulator.alarm_active:
        return  # Уже активен

    accumulator.alarm_active = True
    logger.warning(f"⚠️ ALARM TRIGGERED: {error_type} reached {count} (threshold: {THRESHOLDS.get(error_type)})")

    try:
        api_client = get_api_client()
        api_client.signal_alarm()
    except Exception as e:
        logger.error(f"Failed to send alarm signal: {e}")


def reset_alarm() -> None:
    """Сбросить состояние alarm"""
    with accumulator.lock:
        _reset_shift_if_needed()
        if accumulator.alarm_active:
            logger.info("ALARM reset")
            accumulator.alarm_active = False


def get_accumulator_stats() -> dict:
    """Получить текущее состояние счетчиков"""
    with accumulator.lock:
        _reset_shift_if_needed()
        return {
            "counters": accumulator.counters.copy(),
            "thresholds": THRESHOLDS.copy(),
            "alarm_active": accumulator.alarm_active,
            "shift_key": accumulator.shift_key,
        }
