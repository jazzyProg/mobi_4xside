#!/usr/bin/env python3
"""filter.py – fast cleaning of YOLO‑Seg label polygons (v2‑fix)

Changes vs. v2‑beta (bug‑fix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **FIX** boolean‑mask shape mismatch when a polygon had exactly one point
  remaining after the border filter. The mask is now flattened explicitly and
  validated against the number of rows before indexing (`arr[mask_flat]`).
* Added guard for empty *edges* (degenerate hull with <2 vertices).
* More robust fallback when SciPy’s `ConvexHull` raises `QhullError` and
  *shapely* is missing – the function now simply returns the original lines.
"""
from __future__ import annotations

from typing import List, Tuple, Iterable
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull, QhullError

try:                                    # optional run‑time dependency
    from shapely.geometry import LinearRing  # type: ignore
except ModuleNotFoundError:              # pragma: no cover
    LinearRing = None  # type: ignore

##############################################################################
# 1. Remove consecutive duplicate points                                     #
##############################################################################

def remove_duplicate_points_in_line(line: str, eps: float = 1e-6) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None

    cls, raw = parts[0], parts[1:]
    if len(raw) < 6:
        return None

    uniq: List[Tuple[float, float]] = []
    for i in range(0, len(raw), 2):
        pt = float(raw[i]), float(raw[i + 1])
        if not uniq or abs(pt[0] - uniq[-1][0]) > eps or abs(pt[1] - uniq[-1][1]) > eps:
            uniq.append(pt)
    if len(uniq) < 3:
        return None

    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in uniq)

##############################################################################
# 2. Vectorised border filter for all class‑0 polygons                        #
##############################################################################

_RE_SPLIT = re.compile(r"\s+")

def _hull_distance(points: np.ndarray, hull_pts: np.ndarray) -> np.ndarray:
    """Return *min* perpendicular distance of every point to hull edges."""
    if len(hull_pts) < 2:               # degenerate – all points identical
        return np.zeros(len(points))

    # build edges
    edges = np.stack([hull_pts, np.roll(hull_pts, -1, axis=0)], axis=1)  # (m,2,2)
    vec = edges[:, 1] - edges[:, 0]
    norm = np.stack([vec[:, 1], -vec[:, 0]], axis=1)
    norm_len = np.linalg.norm(norm, axis=1, keepdims=True)
    nonzero = norm_len[:, 0] > 0
    norm[nonzero] /= norm_len[nonzero]

    d = np.abs(((points[:, None, :] - edges[None, :, 0, :]) * norm[None, :, :]).sum(-1))
    return d.min(axis=1)


def filter_class0_bulk(
    lines: Iterable[str], *, thickness: float = 3.0, eps: float = 1e-9
) -> List[str]:
    """Fast border filter: keeps only points lying on/near the *global* convex
    hull of all class‑0 polygons. Returns new list of label lines."""
    lines = list(lines)
    if not lines:
        return []

    idx_c0: List[int] = []
    polys_c0: List[np.ndarray] = []
    parsed: List[tuple[str, np.ndarray]] = []

    for i, ln in enumerate(lines):
        parts = _RE_SPLIT.split(ln.strip())
        if len(parts) < 7:
            parsed.append(("", np.empty((0, 2))))
            continue
        cls, raw = parts[0], parts[1:]
        arr = np.asarray(raw, float).reshape(-1, 2)
        parsed.append((cls, arr))
        if cls == "0" and arr.shape[0] >= 3:
            idx_c0.append(i)
            polys_c0.append(arr)

    if not polys_c0:
        return lines

    all_pts = np.concatenate(polys_c0, axis=0)
    try:
        hull_pts = all_pts[ConvexHull(all_pts).vertices]
    except QhullError:
        if LinearRing is None:
            return lines  # graceful fallback
        hull_pts = np.asarray(LinearRing(all_pts).coords, float)

    dist_all = _hull_distance(all_pts, hull_pts)

    out = lines.copy()
    offset = 0
    for idx, poly in zip(idx_c0, polys_c0):
        n = poly.shape[0]
        mask_flat = (dist_all[offset : offset + n] <= thickness + eps).ravel()
        offset += n
        if mask_flat.sum() >= 3:
            kept = poly[mask_flat, :]
            out[idx] = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in kept)
        else:
            out[idx] = None
    return [l for l in out if l]

##############################################################################
# 3. Per‑line compatibility wrapper                                          #
##############################################################################

def keep_border_points_in_line(line: str, *, thickness: float = 3.0, eps: float = 1e-9) -> str | None:
    res = filter_class0_bulk([line], thickness=thickness, eps=eps)
    return res[0] if res else None

##############################################################################
# 4. CLI helper                                                              #
##############################################################################

def main(file_path: str | Path, *, thickness: float = 3.0):
    p = Path(file_path)
    raw = p.read_text("utf-8").splitlines()

    deduped = [remove_duplicate_points_in_line(l) for l in raw]
    deduped = [l for l in deduped if l]

    cleaned = filter_class0_bulk(deduped, thickness=thickness)
    p.write_text("\n".join(cleaned), encoding="utf-8")


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser("Clean label polygons (dup + border)")
    ap.add_argument("path", help="labels.txt")
    ap.add_argument("--thickness", type=float, default=3.0)
    ns = ap.parse_args()
    try:
        main(ns.path, thickness=ns.thickness)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
