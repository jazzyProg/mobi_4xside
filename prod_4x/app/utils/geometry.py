from __future__ import annotations

ALPHA_X = 0.10930
ALPHA_Y = 0.09820
DIAM_MIN = 2.7
DIAM_MAX = 46.0
GRID = 0.5


def snap_to_grid(v: float) -> float:
    """Round value to the nearest GRID step."""
    return round(v / GRID) * GRID

