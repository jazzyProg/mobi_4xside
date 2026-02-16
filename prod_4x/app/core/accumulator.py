"""
Система накопления ошибок и alarm сигналов
"""
import threading
import logging
from typing import Dict
from app.core.api_client import get_api_client

logger = logging.getLogger(__name__)

# ============================================
# Пороги накопления для alarm
# ============================================
THRESHOLDS = {
    "diameter": 30000,        # 3 ошибки по диаметру → alarm
    "holes": 30000,           # 3 ошибки по отверстиям → alarm
    "rectangles": 200000,      # 2 ошибки по прямоугольникам → alarm
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

accumulator = AccumulatorState()


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


def increment_counters(report: dict) -> None:
    """
    Инкрементировать счетчики на основе QC отчета при FAIL
    """
    errors = parse_qc_report(report)

    if not errors:
        return

    with accumulator.lock:
        for key in errors:
            accumulator.counters[key] += 1
            accumulator.last_increment[key] = True

            logger.info(f"ACCUM: {key} incremented to {accumulator.counters[key]}/{THRESHOLDS.get(key, 999)}")

            # Проверка превышения порога
            threshold = THRESHOLDS.get(key)
            if threshold and accumulator.counters[key] >= threshold:
                trigger_alarm(key, accumulator.counters[key])


def reset_counters_on_pass() -> None:
    """
    Сбросить счетчики которые были инкрементированы перед последним PASS
    """
    with accumulator.lock:
        for key in list(accumulator.counters.keys()):
            if accumulator.last_increment.get(key, False):
                logger.info(f"ACCUM: reset {key} after PASS (was {accumulator.counters[key]})")
                accumulator.counters[key] = 0
                accumulator.last_increment[key] = False


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
        if accumulator.alarm_active:
            logger.info("ALARM reset")
            accumulator.alarm_active = False


def get_accumulator_stats() -> dict:
    """Получить текущее состояние счетчиков"""
    with accumulator.lock:
        return {
            "counters": accumulator.counters.copy(),
            "thresholds": THRESHOLDS.copy(),
            "alarm_active": accumulator.alarm_active
        }
