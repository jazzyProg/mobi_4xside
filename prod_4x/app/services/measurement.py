#!/usr/bin/env python3
"""
measure5.py — inline-метрология прямоугольной детали и её круглых отверстий

Основано на вашей версии v3.3-obb, но с правками стабильности:

FIX #1: Стабилизация локальных осей x_norm / y_norm в _fit_rectangle():
        вместо опоры на одну точку (mid_bottom) берём усреднённый вектор
        верхней и нижней стороны (v_top, v_bottom). Это снижает джиттер,
        из-за которого у отверстий в углу “прыгают” nearest_x/nearest_y.

FIX #2: Проверка “отверстие внутри прямоугольника” перед расчётом nearest_x/y:
        если dist_left/right/bot/top уходит в заметный минус, отверстие
        пропускается (как артефакт/ошибка осей/ошибка полигона).

(Подписи nearest_x/nearest_y остаются как у вас: слева/снизу отрицательные,
справа/сверху положительные.)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# ───────── калибровка ─────────
ALPHA_X, ALPHA_Y = 0.10930, 0.09820  # мм / px
DIAM_MIN, DIAM_MAX = 2.0, 40.0
MAX_AXIS_RATIO = float(os.getenv("HOLE_OVALITY_MAX_RATIO", "1.15"))
MIN_CIRCULARITY = float(os.getenv("HOLE_MIN_CIRCULARITY", "0.72"))
NESTED_HOLE_MARGIN_MM = float(os.getenv("NESTED_HOLE_MARGIN_MM", "0.20"))
CONCENTRIC_CENTER_TOL_MM = float(os.getenv("HOLE_CONCENTRIC_CENTER_TOL_MM", "0.35"))
CONCENTRIC_DIA_RATIO_MAX = float(os.getenv("HOLE_CONCENTRIC_DIA_RATIO_MAX", "1.45"))
OVERLAP_DUP_RATIO_MIN = float(os.getenv("HOLE_OVERLAP_DUP_RATIO_MIN", "0.70"))
ALLOW_SHORT_HOLE_FALLBACK = os.getenv("HOLE_ALLOW_SHORT_CONTOUR_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
TAU_MM, K_RANSAC, MIN_FRACTION = 0.25, 150, 0.15
GRID = 0.5  # Шаг сетки CAD
NEAR_STEP_MM = 0.10

# ───── новые пороги валидации (мм) ─────
# Если отверстие "вне" прямоугольника сильнее, чем на MARGIN_MM (+ его радиус) → отбрасываем.
OUTSIDE_MARGIN_MM = 1.0
# Если отверстие совсем близко к краю (< EDGE_WARN_MM) — можно логировать, но не отбрасывать.
EDGE_WARN_MM = 0.2

# Цвета (BGR)
COLORS_HOLES = [(0, 255, 0), (0, 200, 255), (255, 128, 0), (200, 0, 255)]
COLOR_RECT, COLOR_DIAG, COLOR_ANGLE = (0, 255, 0), (255, 128, 0), (0, 0, 255)
COLOR_CENTER, COLOR_INFO = (255, 0, 255), (255, 0, 255)

# ───────── TXT → полигоны ─────────
def _parse_polys(txt: os.PathLike) -> Dict[int, List[np.ndarray]]:
    cls: Dict[int, List[np.ndarray]] = {}
    with open(txt, "r", encoding="utf-8") as f:
        for ln in f:
            pr = ln.split()
            if not pr:
                continue
            c = int(float(pr[0]))
            arr = np.asarray(pr[1:], float)
            if arr.size % 2:
                raise ValueError("Нечётное число координат")
            cls.setdefault(c, []).append(arr.reshape(-1, 2).astype(np.float32))
    return cls

# ───────── мелкие визуальные утилиты ─────────
def _draw_dashed(img, p1, p2, color, thickness=1, dash=6, gap=4):
    p1, p2 = map(np.asarray, (p1, p2))
    d = p2 - p1
    L = float(np.hypot(*d))
    if L == 0:
        return
    v = d / L
    n = int(L // (dash + gap))
    for i in range(n + 1):
        s = p1 + v * (i * (dash + gap))
        e = s + v * min(dash, L - i * (dash + gap))
        cv2.line(
            img,
            tuple(s.astype(int)),
            tuple(e.astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )

def _mid_text(img, p1, p2, text):
    m = ((p1 + p2) / 2).astype(int)
    cv2.putText(
        img,
        text,
        (m[0] + 5, m[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_RECT,
        1,
        cv2.LINE_AA,
    )


def _hole_far_outside_board(*, dist_left: float, dist_right: float, dist_bot: float, dist_top: float, radius_mm: float) -> bool:
    """Return True when the hole is confidently outside board bounds.

    We allow minor OBB fitting jitter for edge/corner holes. A hole is rejected
    only if its center is farther than (radius + OUTSIDE_MARGIN_MM) from any side.
    """
    outside_threshold = -(OUTSIDE_MARGIN_MM + max(radius_mm, 0.0))
    return (
        dist_left < outside_threshold
        or dist_right < outside_threshold
        or dist_bot < outside_threshold
        or dist_top < outside_threshold
    )



def _hole_geom_from_poly(pts_px: np.ndarray):
    """Return hole geometry from contour.

    By default, short contours (<5 points) are ignored because they frequently
    produce false nested/oval holes. Legacy fallback to minEnclosingCircle can be
    re-enabled with HOLE_ALLOW_SHORT_CONTOUR_FALLBACK=1.
    """
    if len(pts_px) >= 5:
        center_px, axes_px, ang = _ellipse_props_px(pts_px)
        return center_px, axes_px, ang, False

    if len(pts_px) >= 3 and ALLOW_SHORT_HOLE_FALLBACK:
        (cx, cy), r = cv2.minEnclosingCircle(pts_px.astype(np.float32))
        d = 2.0 * float(r)
        center_px = np.array([cx, cy], dtype=np.float32)
        axes_px = (d, d)
        return center_px, axes_px, 0.0, True

    return None

def _ellipse_props_px(pts_px: np.ndarray):
    (cx, cy), (maj, minr), ang = cv2.fitEllipse(pts_px)
    return np.array([cx, cy], np.float32), (float(maj), float(minr)), float(ang)

def _diameter_mm(length_px: float, angle_deg: float) -> float:
    theta = math.radians(angle_deg)
    dx = 0.5 * length_px * math.cos(theta)
    dy = 0.5 * length_px * math.sin(theta)
    return 2.0 * math.hypot(dx * ALPHA_X, dy * ALPHA_Y)


def _is_oval_hole(maj_mm: float, min_mm: float, *, max_axis_ratio: float = MAX_AXIS_RATIO) -> bool:
    """Return True if fitted hole geometry is too oval to be considered a round hole."""
    if min_mm <= 0:
        return True
    return (max(maj_mm, min_mm) / min_mm) > max_axis_ratio


def _contour_circularity(pts_px: np.ndarray) -> float:
    """Return contour circularity in [0, 1] where 1 is a perfect circle."""
    area = float(cv2.contourArea(pts_px.astype(np.float32)))
    perimeter = float(cv2.arcLength(pts_px.astype(np.float32), True))
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def _is_nested_hole(*, inner_center_mm: np.ndarray, inner_dia_mm: float, outer_center_mm: np.ndarray, outer_dia_mm: float,
                    margin_mm: float = NESTED_HOLE_MARGIN_MM) -> bool:
    """Return True if `inner` is geometrically contained by `outer` (with configurable tolerance)."""
    center_distance = float(math.hypot(*(inner_center_mm - outer_center_mm)))
    inner_radius = max(float(inner_dia_mm), 0.0) / 2.0
    outer_radius = max(float(outer_dia_mm), 0.0) / 2.0
    return center_distance + inner_radius <= outer_radius + max(margin_mm, 0.0)


def _is_concentric_duplicate(*, center_a_mm: np.ndarray, dia_a_mm: float, center_b_mm: np.ndarray, dia_b_mm: float,
                             center_tol_mm: float = CONCENTRIC_CENTER_TOL_MM,
                             max_dia_ratio: float = CONCENTRIC_DIA_RATIO_MAX) -> bool:
    """Return True for near-concentric duplicate detections of the same hole."""
    center_distance = float(math.hypot(*(center_a_mm - center_b_mm)))
    if center_distance > max(center_tol_mm, 0.0):
        return False

    d_small = max(min(float(dia_a_mm), float(dia_b_mm)), 1e-6)
    d_large = max(float(dia_a_mm), float(dia_b_mm))
    return (d_large / d_small) <= max(max_dia_ratio, 1.0)


def _circle_intersection_area(radius_a: float, radius_b: float, center_distance: float) -> float:
    """Return area of two-circle intersection."""
    ra = max(radius_a, 0.0)
    rb = max(radius_b, 0.0)
    d = max(center_distance, 0.0)

    if ra <= 0.0 or rb <= 0.0:
        return 0.0
    if d >= ra + rb:
        return 0.0
    if d <= abs(ra - rb):
        return math.pi * min(ra, rb) ** 2

    alpha = math.acos(np.clip((d * d + ra * ra - rb * rb) / (2.0 * d * ra), -1.0, 1.0))
    beta = math.acos(np.clip((d * d + rb * rb - ra * ra) / (2.0 * d * rb), -1.0, 1.0))
    return (
        ra * ra * alpha
        + rb * rb * beta
        - 0.5 * math.sqrt(max(0.0, (-d + ra + rb) * (d + ra - rb) * (d - ra + rb) * (d + ra + rb)))
    )


def _is_overlap_duplicate(*, center_a_mm: np.ndarray, dia_a_mm: float, center_b_mm: np.ndarray, dia_b_mm: float,
                          overlap_ratio_min: float = OVERLAP_DUP_RATIO_MIN) -> bool:
    """Return True when circles overlap too much to be treated as separate holes."""
    center_distance = float(math.hypot(*(center_a_mm - center_b_mm)))
    ra = max(float(dia_a_mm), 0.0) / 2.0
    rb = max(float(dia_b_mm), 0.0) / 2.0
    intersection = _circle_intersection_area(ra, rb, center_distance)
    if intersection <= 0.0:
        return False

    min_area = math.pi * (min(ra, rb) ** 2)
    if min_area <= 0.0:
        return False

    overlap_ratio = intersection / min_area
    return overlap_ratio >= max(min(overlap_ratio_min, 1.0), 0.0)

# ───────── прямоугольник через OBB ─────────
def _fit_rectangle(poly_px: np.ndarray, img):
    """
    Строит ориентированный прямоугольник детали (OBB),
    задаёт устойчивую локальную систему координат детали
    и возвращает геометрию, инвариантную к повороту.
    """

    # 1) Oriented Bounding Box (px)
    cnt = poly_px.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box_px = cv2.boxPoints(rect).astype(np.float32)

    # 2) Устойчивый порядок TL,TR,BR,BL (по пикселям)
    idx = np.argsort(box_px[:, 1])  # сортировка по Y
    top = idx[:2]
    bot = idx[2:]

    top = top[np.argsort(box_px[top, 0])]  # TL, TR
    bot = bot[np.argsort(box_px[bot, 0])]  # BL, BR

    TL_px, TR_px = box_px[top[0]], box_px[top[1]]
    BL_px, BR_px = box_px[bot[0]], box_px[bot[1]]

    corners_px = np.vstack([TL_px, TR_px, BR_px, BL_px])

    # 3) Перевод в мм
    corners_mm = np.column_stack([corners_px[:, 0] * ALPHA_X, corners_px[:, 1] * ALPHA_Y])
    TL_mm, TR_mm, BR_mm, BL_mm = corners_mm

    # 4) Центр детали
    (cx, cy) = rect[0]
    center_px = np.array([cx, cy], dtype=float)
    center_mm = np.array([cx * ALPHA_X, cy * ALPHA_Y], dtype=float)

    # 5) Локальные оси детали (FIX: устойчивее)
    # y_norm — "слева направо" (ширина) через усреднение верхней и нижней стороны.
    v_bottom = BR_mm - BL_mm
    v_top = TR_mm - TL_mm
    v_horiz = 0.5 * (v_bottom + v_top)

    hv = float(np.linalg.norm(v_horiz))
    if hv < 1e-9:
        # fallback на верхнюю сторону, если вдруг выродилось
        v_horiz = v_top
        hv = float(np.linalg.norm(v_horiz))
    if hv < 1e-9:
        # совсем выродилось — аварийный fallback
        v_horiz = np.array([1.0, 0.0], dtype=float)
        hv = 1.0

    y_norm = v_horiz / hv

    # x_norm — перпендикуляр к y_norm, направленный "снизу вверх"
    x_norm = np.array([-y_norm[1], y_norm[0]], dtype=float)

    # Убедиться, что x_norm смотрит от низа к центру (вверх)
    mid_bottom_mm = 0.5 * (BL_mm + BR_mm)
    if (center_mm - mid_bottom_mm) @ x_norm < 0:
        x_norm *= -1

    # 6) Полуразмеры детали
    half_w = 0.5 * float(np.linalg.norm(TR_mm - TL_mm))  # ширина
    half_h = 0.5 * float(np.linalg.norm(TL_mm - BL_mm))  # высота

    # 7) Геометрия для JSON
    sides_mm = [
        float(np.linalg.norm(TL_mm - TR_mm)),
        float(np.linalg.norm(TR_mm - BR_mm)),
        float(np.linalg.norm(BR_mm - BL_mm)),
        float(np.linalg.norm(BL_mm - TL_mm)),
    ]
    diags_mm = [
        float(np.linalg.norm(TL_mm - BR_mm)),
        float(np.linalg.norm(TR_mm - BL_mm)),
    ]

    rect_out = dict(
        sides_mm=[round(v, 3) for v in sides_mm],
        diagonals_mm=[round(v, 3) for v in diags_mm],
        corners_mm=corners_mm.round(3).tolist(),
        center_px=[round(float(center_px[0]), 3), round(float(center_px[1]), 3)],
    )

    # 8) Визуализация
    _draw_rectangle(img, corners_px, sides_mm, diags_mm, center_px)

    # 9) Геометрия для дальнейших расчётов
    geo = dict(
        center_mm=center_mm,
        center_px=center_px,
        x_norm=x_norm,
        y_norm=y_norm,
        half_w=half_w,
        half_h=half_h,
        corners_px=corners_px.astype(np.float32),
    )

    return rect_out, geo

def _calc_angles(cmm):
    angs = []
    for i in range(4):
        a = cmm[i - 1] - cmm[i]
        b = cmm[(i + 1) % 4] - cmm[i]
        angs.append(
            round(
                math.degrees(
                    math.acos(
                        np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
                    )
                ),
                2,
            )
        )
    return angs

def _draw_rectangle(img, pts_px, sides_mm, diags_mm, center_px):
    pi = pts_px.astype(int)
    for i in range(4):
        p1, p2 = pi[i], pi[(i + 1) % 4]
        cv2.line(img, tuple(p1), tuple(p2), COLOR_RECT, 2, cv2.LINE_AA)
        _mid_text(img, p1, p2, f"{sides_mm[i]:.2f} мм")
    cv2.line(img, tuple(pi[0]), tuple(pi[2]), COLOR_DIAG, 1, cv2.LINE_AA)
    cv2.line(img, tuple(pi[1]), tuple(pi[3]), COLOR_DIAG, 1, cv2.LINE_AA)
    c = center_px.astype(int)
    cv2.circle(img, tuple(c), 4, COLOR_CENTER, -1, cv2.LINE_AA)

def _draw_ellipse(img, center_px, axes_px, ang_deg, label, color):
    c_int = tuple(map(int, map(round, center_px)))
    a_int = tuple(map(int, map(lambda v: round(v / 2), axes_px)))
    cv2.ellipse(img, c_int, a_int, ang_deg, 0, 360, color, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (c_int[0] + 10, c_int[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )

def _signed_nearest_mm(point_mm: np.ndarray, lines):
    """
    Корректно вычисляет ближайшее расстояние до ЛЕВОГО/ПРАВОГО и
    ВЕРХНЕГО/НИЖНЕГО края с устойчивым знаком.

    nearest_x < 0 → ближе к левому краю
    nearest_x > 0 → ближе к правому краю
    nearest_y < 0 → ближе к нижнему краю
    nearest_y > 0 → ближе к верхнему краю
    """
    (n_left, d_left) = lines["left"]
    (n_right, d_right) = lines["right"]
    (n_top, d_top) = lines["top"]
    (n_bottom, d_bot) = lines["bottom"]

    dist_left = abs(n_left @ point_mm + d_left)
    dist_right = abs(n_right @ point_mm + d_right)
    dist_top = abs(n_top @ point_mm + d_top)
    dist_bot = abs(n_bottom @ point_mm + d_bot)

    if dist_left <= dist_right:
        nearest_x = -dist_left
    else:
        nearest_x = dist_right

    if dist_bot <= dist_top:
        nearest_y = -dist_bot
    else:
        nearest_y = dist_top

    return float(nearest_x), float(nearest_y)

def _inside_poly(center_px, poly):
    return (
        cv2.pointPolygonTest(poly.reshape(-1, 1, 2), tuple(map(float, center_px)), False)
        >= 0
    )

def _snap(v: float) -> float:
    return round(v / GRID) * GRID

# ───────── основной поток ─────────
def measure_board(
    txt: os.PathLike,
    img_path: os.PathLike,
    out: str | os.PathLike = "annotated.png",
    *,
    verbose: bool = False,
):
    cls = _parse_polys(txt)
    if 0 not in cls:
        raise RuntimeError("Контур детали (class 0) не найден")

    rect_polys = cls.get(0, []) + cls.get(3, [])
    rect_poly = max(rect_polys, key=lambda p: p.shape[0]) if rect_polys else None
    hole_polys = cls.get(1, [])

    if rect_poly is None:
        raise RuntimeError("Полигоны детали не найдены")

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    rect, geo = _fit_rectangle(rect_poly, img)

    # ───────── отверстия ─────────
    candidates = []
    hole_geom_fn = globals().get("_hole_geom_from_poly")
    for pts in hole_polys:
        if callable(hole_geom_fn):
            geom = hole_geom_fn(pts)
        else:
            # merge-safe fallback: even if helper was dropped by conflict,
            # keep short/partial contours in processing.
            if len(pts) >= 5:
                center_px, axes_px, ang = _ellipse_props_px(pts)
                geom = (center_px, axes_px, ang, False)
            elif len(pts) >= 3:
                (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
                d = 2.0 * float(r)
                center_px = np.array([cx, cy], dtype=np.float32)
                geom = (center_px, (d, d), 0.0, True)
            else:
                geom = None

        if geom is None:
            if verbose and len(pts) >= 3 and not ALLOW_SHORT_HOLE_FALLBACK:
                print(f"[WARN] short contour ignored: points={len(pts)} (set HOLE_ALLOW_SHORT_CONTOUR_FALLBACK=1 to restore old behavior)")
            continue

        center_px, axes_px, ang, used_circle_fallback = geom
        if verbose and used_circle_fallback:
            print(f"[WARN] short hole contour fallback: points={len(pts)}")
        maj_mm = _diameter_mm(axes_px[0], ang)
        min_mm = _diameter_mm(axes_px[1], ang + 90)
        dia = (maj_mm + min_mm) / 2.0

        if not (DIAM_MIN <= max(maj_mm, min_mm) <= DIAM_MAX):
            continue
        if (not used_circle_fallback) and _is_oval_hole(maj_mm, min_mm):
            if verbose:
                ratio = max(maj_mm, min_mm) / min_mm if min_mm > 0 else float("inf")
                print(
                    "[WARN] oval hole ignored:",
                    f"major={maj_mm:.3f} minor={min_mm:.3f} ratio={ratio:.3f} lim={MAX_AXIS_RATIO:.3f}",
                )
            continue

        circularity = _contour_circularity(pts)
        if (not used_circle_fallback) and circularity < MIN_CIRCULARITY:
            if verbose:
                print(
                    "[WARN] low-circularity hole ignored:",
                    f"circ={circularity:.3f} lim={MIN_CIRCULARITY:.3f}",
                )
            continue

        # if not _inside_poly(center_px, rect_poly): continue
        c_mm = np.array([center_px[0] * ALPHA_X, center_px[1] * ALPHA_Y], dtype=float)
        rel = c_mm - geo["center_mm"]

        # локальная СК детали (как у вас)
        posX = float(rel @ geo["y_norm"])
        posY = float(rel @ geo["x_norm"])

        local_x = posX
        local_y = posY

        # расстояния до сторон В ЛОКАЛЬНОЙ СИСТЕМЕ
        dist_left = local_x + geo["half_w"]
        dist_right = geo["half_w"] - local_x
        dist_bot = local_y + geo["half_h"]
        dist_top = geo["half_h"] - local_y

        # FIX: отверстие рядом с кромкой/в углу не должно отбрасываться из-за джиттера OBB.
        # Отбрасываем только если центр отверстия ушёл далеко наружу даже с учётом радиуса.
        radius_mm = dia / 2.0
        if _hole_far_outside_board(
            dist_left=dist_left,
            dist_right=dist_right,
            dist_bot=dist_bot,
            dist_top=dist_top,
            radius_mm=radius_mm,
        ):
            if verbose:
                outside_threshold = -(OUTSIDE_MARGIN_MM + radius_mm)
                print(
                    "[WARN] hole outside OBB:",
                    f"L={dist_left:.3f} R={dist_right:.3f} B={dist_bot:.3f} T={dist_top:.3f} thr={outside_threshold:.3f}",
                )
            continue

        if verbose and (
            dist_left < EDGE_WARN_MM
            or dist_right < EDGE_WARN_MM
            or dist_bot < EDGE_WARN_MM
            or dist_top < EDGE_WARN_MM
        ):
            print(
                "[WARN] hole near edge:",
                f"L={dist_left:.3f} R={dist_right:.3f} B={dist_bot:.3f} T={dist_top:.3f}",
            )

        # signed nearest
        nearest_x = -dist_left if dist_left <= dist_right else dist_right
        nearest_y = -dist_bot if dist_bot <= dist_top else dist_top

        candidates.append(
            dict(
                center_px=center_px,
                axes_px=axes_px,
                angle_deg=ang,
                maj_mm=maj_mm,
                min_mm=min_mm,
                diameter_mm=dia,
                center_mm=c_mm,
                posX=posX,
                posY=posY,
                nearest_x=nearest_x,
                nearest_y=nearest_y,
                distance=float(math.hypot(posX, posY)),
            )
        )

    # Фильтр вложенных отверстий: оставляем только наибольшее из пересекающихся вложенных.
    # Сортировка по убыванию диаметра гарантирует приоритет большего отверстия.
    candidates.sort(key=lambda h: -h["diameter_mm"])
    accepted = []

    for h in candidates:
        nested_or_dup = False
        for a in accepted:
            is_nested = _is_nested_hole(
                inner_center_mm=h["center_mm"],
                inner_dia_mm=h["diameter_mm"],
                outer_center_mm=a["center_mm"],
                outer_dia_mm=a["diameter_mm"],
            )
            is_dup = _is_concentric_duplicate(
                center_a_mm=h["center_mm"],
                dia_a_mm=h["diameter_mm"],
                center_b_mm=a["center_mm"],
                dia_b_mm=a["diameter_mm"],
            )
            is_overlap_dup = _is_overlap_duplicate(
                center_a_mm=h["center_mm"],
                dia_a_mm=h["diameter_mm"],
                center_b_mm=a["center_mm"],
                dia_b_mm=a["diameter_mm"],
            )
            if is_nested or is_dup or is_overlap_dup:
                nested_or_dup = True
                if verbose:
                    reason = "nested" if is_nested else "concentric-duplicate" if is_dup else "high-overlap-duplicate"
                    print(
                        f"[WARN] {reason} hole ignored:",
                        f"dia={h['diameter_mm']:.3f} pos=({h['posX']:.3f}, {h['posY']:.3f})",
                    )
                break

        if nested_or_dup:
            continue

        accepted.append(h)

    # сортировка под CAD
    accepted.sort(key=lambda h: (_snap(h["posX"]), _snap(h["posY"])))

    holes = []
    for idx, h in enumerate(accepted, 1):
        color = COLORS_HOLES[(idx - 1) % len(COLORS_HOLES)]
        _draw_ellipse(
            img,
            h["center_px"],
            h["axes_px"],
            h["angle_deg"],
            f"#{idx}: Ø{h['diameter_mm']:.2f} мм",
            color,
        )
        _draw_dashed(img, h["center_px"], geo["center_px"], color)

        holes.append(
            dict(
                hole_id=idx,
                center_px=[
                    round(float(h["center_px"][0]), 3),
                    round(float(h["center_px"][1]), 3),
                ],
                posX=f"{h['posX']:.3f}",
                posY=f"{h['posY']:.3f}",
                diameter=f"{h['diameter_mm']:.3f}",
                nearest_x=f"{h['nearest_x']:.3f}",
                nearest_y=f"{h['nearest_y']:.3f}",
                distance=f"{h['distance']:.3f}",
            )
        )

    cv2.putText(
        img,
        f"holes: {len(holes)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_INFO,
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out), img)

    js_path = Path(out).with_suffix(".json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                rectangle=rect,
                center_detail_px=rect["center_px"],
                holes_count=len(holes),
                holes=holes,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )
    return rect, holes, str(js_path)

# ───────── CLI ─────────
def main():
    ap = argparse.ArgumentParser(description="measure_board v3.3-obb (fixed axes + inside check)")
    ap.add_argument("coords_txt")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="annotated.png")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        rect, holes, js_path = measure_board(args.coords_txt, args.image, args.out, verbose=args.verbose)
        print(json.dumps({"rectangle": rect, "holes": holes}, ensure_ascii=False, indent=2))
        print("JSON saved to", js_path)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
