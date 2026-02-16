"""Backward-compatible router module.

Historically, the QC service exposed a single `app.api.routes` module.
For better maintainability, routers are now split into `app.api.endpoints`.
This module re-exports the same `router` object.
"""

from __future__ import annotations

from app.api.endpoints import router  # noqa: F401
