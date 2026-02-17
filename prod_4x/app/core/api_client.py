"""app.core.api_client

Synchronous HTTP client for integration with other microservices.

The QC pipeline runs in a background thread, therefore a sync client is appropriate here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ExternalServicesClient:
    """Client for Products, Signals and Camera microservices."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

        self.products_url = self._settings.PRODUCTS_API_URL
        self.signals_url = self._settings.SIGNALS_API_URL
        self.camera_url = self._settings.CAMERA_API_URL

        self._client = httpx.Client(
            timeout=self._settings.API_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def _signals_enabled(self) -> bool:
        return bool(self._settings.QUALITY_CHECK_ENABLED)

    # --------------------------------------------
    # Products API
    # --------------------------------------------
    def get_active_product(self) -> Dict[str, Any]:
        """GET /products/active_detail."""
        try:
            logger.info("Fetching active product from Products API")
            resp = self._client.get(f"{self.products_url}/products/active_detail")
            resp.raise_for_status()
            product = resp.json()
            logger.info("Got product: %s", product.get("product_name", "unknown"))
            return product
        except httpx.HTTPError as e:
            logger.error("Failed to get active product: %s", e)
            raise RuntimeError(f"Products API unavailable: {e}") from e

    # --------------------------------------------
    # Signals API
    # --------------------------------------------
    def signal_success(self, duration: float = 3.0) -> None:
        """POST /signal/success."""
        if not self._signals_enabled():
            logger.info("QUALITY_CHECK_ENABLED=false, skip SUCCESS signal")
            return

        try:
            logger.info("Sending SUCCESS signal (duration=%ss)", duration)
            resp = self._client.post(
                f"{self.signals_url}/signal/success",
                params={"duration": duration},
                timeout=self._settings.SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to send success signal: %s", e)

    def signal_fail(self) -> None:
        """POST /signal/fail."""
        if not self._signals_enabled():
            logger.info("QUALITY_CHECK_ENABLED=false, skip FAIL signal")
            return

        try:
            logger.info("Sending FAIL signal")
            resp = self._client.post(
                f"{self.signals_url}/signal/fail",
                timeout=self._settings.SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to send fail signal: %s", e)

    def signal_alarm(self) -> None:
        """POST /signal/alarm."""
        if not self._signals_enabled():
            logger.info("QUALITY_CHECK_ENABLED=false, skip ALARM signal")
            return

        try:
            logger.info("Sending ALARM signal")
            resp = self._client.post(
                f"{self.signals_url}/signal/alarm",
                timeout=self._settings.SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to send alarm signal: %s", e)

    def signal_heartbeat(self, duration: float = 1.0) -> None:
        """POST /signal/heartbeat."""
        if not self._signals_enabled():
            logger.debug("QUALITY_CHECK_ENABLED=false, skip HEARTBEAT signal")
            return

        try:
            resp = self._client.post(
                f"{self.signals_url}/signal/heartbeat",
                params={"duration": duration},
                timeout=self._settings.SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Heartbeat failed: %s", e)

    def signal_camera_trigger(self, duration: float | None = None) -> None:
        """POST /signal/camera-trigger."""
        if not self._signals_enabled():
            logger.info("QUALITY_CHECK_ENABLED=false, skip CAMERA_TRIGGER signal")
            return

        pulse_duration = duration if duration is not None else self._settings.CAMERA_TRIGGER_PULSE_SEC

        try:
            logger.info("Sending CAMERA_TRIGGER signal (duration=%ss)", pulse_duration)
            resp = self._client.post(
                f"{self.signals_url}/signal/camera-trigger",
                params={"duration": pulse_duration},
                timeout=self._settings.SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to send camera trigger signal: %s", e)

    # --------------------------------------------
    # Camera API
    # --------------------------------------------
    def start_capture(self, session_id: str) -> Dict[str, Any]:
        """POST /capture/start."""
        try:
            logger.info("Starting camera capture for session %s", session_id)
            resp = self._client.post(
                f"{self.camera_url}/capture/start",
                json={"session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error("Failed to start capture: %s", e)
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def stop_capture(self) -> Dict[str, Any]:
        """POST /capture/stop."""
        try:
            logger.info("Stopping camera capture")
            resp = self._client.post(f"{self.camera_url}/capture/stop")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error("Failed to stop capture: %s", e)
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def get_latest_frames_metadata(self, count: int = 1) -> list[dict]:
        """GET /frames/latest/meta."""
        try:
            resp = self._client.get(
                f"{self.camera_url}/frames/latest/meta",
                params={"count": count},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error("Failed to get frames metadata: %s", e)
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def get_oldest_frame_metadata(self) -> dict:
        """GET /frames/oldest/meta."""
        try:
            resp = self._client.get(
                f"{self.camera_url}/frames/oldest/meta",
                timeout=0.5,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            # Preserve old semantics: caller checks 404 to mean "no frames".
            raise
        except httpx.HTTPError as e:
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def get_camera_status(self) -> Dict[str, Any]:
        """GET /status."""
        try:
            resp = self._client.get(f"{self.camera_url}/status", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def delete_camera_frame(self, frame_id: int) -> Dict[str, Any]:
        """DELETE /frames/{frame_id}."""
        try:
            logger.debug("Deleting frame %s from camera storage", frame_id)
            resp = self._client.delete(
                f"{self.camera_url}/frames/{frame_id}",
                timeout=self._settings.API_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning("Failed to delete frame %s: %s", frame_id, e)
            raise RuntimeError(f"Camera API unavailable: {e}") from e

    def close(self) -> None:
        """Close underlying HTTP connections."""
        self._client.close()


_client: Optional[ExternalServicesClient] = None


def get_api_client() -> ExternalServicesClient:
    """Return singleton client instance (FastAPI dependency compatible)."""
    global _client
    if _client is None:
        _client = ExternalServicesClient()
    return _client
