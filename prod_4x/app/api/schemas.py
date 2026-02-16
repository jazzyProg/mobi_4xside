"""
Pydantic модели для API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class CheckRequest(BaseModel):
    """Запрос на проверку качества"""
    session_id: str = Field(..., description="ID сессии проверки")
    use_latest_frame: bool = Field(True, description="Использовать последний кадр с камеры")
    frame_id: Optional[int] = Field(None, description="Конкретный frame_id (если use_latest_frame=False)")


class CheckResponse(BaseModel):
    """Ответ проверки качества"""
    ok: bool = Field(..., description="Результат проверки (True=PASS, False=FAIL)")
    session_id: str = Field(..., description="ID сессии")
    report: Dict[str, Any] = Field(..., description="Детальный отчет проверки")
    processing_time_ms: Optional[float] = Field(None, description="Время обработки в миллисекундах")


class StatusResponse(BaseModel):
    """Статус сервиса"""
    running: bool = Field(..., description="Работает ли background loop")
    current_frame_id: int = Field(..., description="ID последнего обработанного кадра")
    stats: Dict[str, Any] = Field(..., description="Статистика обработки")


class HealthResponse(BaseModel):
    """Health check"""
    status: str = Field(..., description="healthy/unhealthy")
    service: str = Field(..., description="Имя сервиса")
    version: str = Field(..., description="Версия API")


class ControlResponse(BaseModel):
    """Ответ на control команды"""
    msg: str = Field(..., description="Сообщение о результате")
    status: str = Field(..., description="Текущий статус (running/idle)")
