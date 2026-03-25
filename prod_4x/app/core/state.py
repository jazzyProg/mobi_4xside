"""Глобальное состояние QC Service."""

from __future__ import annotations

import threading
from datetime import datetime, time
from typing import Optional

from zoneinfo import ZoneInfo

from app.config import settings


def _current_shift_key(now: datetime) -> str:
    """Return shift key in format `YYYYMMDD_day|night` for configured timezone."""
    day_start = time(7, 0, 0)
    night_start = time(19, 0, 0)
    suffix = "day" if day_start <= now.time() < night_start else "night"
    return f"{now.strftime('%Y%m%d')}_{suffix}"


class QCEngineState:
    """Глобальное состояние сервиса"""
    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.shm_client = None  # Optional[CameraSHMClient]
        self.last_processed_id: Optional[int] = None
        self.stats = {
            "processed": 0,  # kept for backward compatibility (equals current detail checked)
            "passed": 0,     # kept for backward compatibility (equals current detail passed)
            "failed": 0,     # kept for backward compatibility (equals current detail failed)
            "last_status": "idle",
            "current_detail": {
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "pass_percent": 0.0,
                "fail_percent": 0.0,
            },
            "shift": {
                "shift_key": "",
                "checked_total": 0,
                "passed_total": 0,
                "failed_total": 0,
                "checked_details_total": 0,
                "accumulated_failures_total": 0,
            },
        }
        self._refresh_shift(force=True)

    def _refresh_shift(self, force: bool = False) -> None:
        tz_now = datetime.now(ZoneInfo(settings.TIMEZONE))
        shift_key = _current_shift_key(tz_now)
        if force or self.stats["shift"]["shift_key"] != shift_key:
            self.stats["shift"] = {
                "shift_key": shift_key,
                "checked_total": 0,
                "passed_total": 0,
                "failed_total": 0,
                "checked_details_total": 0,
                "accumulated_failures_total": 0,
            }

    def start_new_detail(self) -> None:
        self._refresh_shift()
        current_detail = self.stats["current_detail"]
        current_detail["checked"] = 0
        current_detail["passed"] = 0
        current_detail["failed"] = 0
        current_detail["pass_percent"] = 0.0
        current_detail["fail_percent"] = 0.0
        self.stats["processed"] = 0
        self.stats["passed"] = 0
        self.stats["failed"] = 0

    def register_detail_result(self, qc_ok: bool) -> None:
        self._refresh_shift()
        current_detail = self.stats["current_detail"]
        shift = self.stats["shift"]
        current_detail["checked"] += 1
        shift["checked_total"] += 1
        shift["checked_details_total"] += 1

        if qc_ok:
            current_detail["passed"] += 1
            shift["passed_total"] += 1
        else:
            current_detail["failed"] += 1
            shift["failed_total"] += 1

        checked = current_detail["checked"]
        current_detail["pass_percent"] = round((current_detail["passed"] / checked) * 100, 2)
        current_detail["fail_percent"] = round((current_detail["failed"] / checked) * 100, 2)

        self.stats["processed"] = current_detail["checked"]
        self.stats["passed"] = current_detail["passed"]
        self.stats["failed"] = current_detail["failed"]


# Singleton instance
state = QCEngineState()
