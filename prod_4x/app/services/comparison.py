#!/usr/bin/env python3
# compare.py  –  v1.4  (stable sorting with Hungarian algorithm)

from __future__ import annotations
import argparse
import csv
import json
import math
import textwrap
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ─── compare.py ─────────────────────────────────────────────────────────
GRID = 0.5  # Шаг сетки CAD

def _snap(v: float) -> float:
    """Округляет значение до ближайшего кратного GRID."""
    return round(v / GRID) * GRID

def dims(d: dict) -> tuple[float, float]:
    if "length" in d and "width" in d:          # CAD
        a, b = map(float, (d["length"], d["width"]))
    else:                                       # Vision
        s = list(map(float, d["rectangle"]["sides_mm"]))
        a, b = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    return tuple(sorted((a, b)))                # (short, long)

def load_holes(d: dict, is_vision: bool, invert_y: bool = False, invert_x: bool = False) -> list[dict]:  # MOD: added invert_x
    """Загружает данные об отверстиях, с опциональной инверсией Y и/или X для Vision."""
    holes = []
    for h in d["holes"]:
        x = float(h["posX"])
        y = float(h["posY"])
        if is_vision:
            if invert_x:
                x = -x
            if invert_y:
                y = -y
        nx = float(h["nearest_x"])
        ny = float(h["nearest_y"])
        if is_vision:  # NEW: invert signs for nearest if flipping axes
            if invert_x:
                nx = -nx  # assuming signed in CAD, flip sign on invert_x
            if invert_y:
                ny = -ny
        holes.append(dict(
            id=h["hole_id"],
            x=x,
            y=y,
            nx=nx,
            ny=ny,
            dia=float(h["diameter"]),
        ))
    return holes

# ------ КОСТ‑ФУНКЦИЯ + матрица ------------------------------------
def _match_cost(a: dict, b: dict) -> float:
    """Сто́имость сопоставления CAD‑отверстия *a* и Vision‑отверстия *b*."""
    w_xy, w_dia, w_near = 1.0, 15.0, 0.7
    d_xy = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    d_dia = abs(a["dia"] - b["dia"])
    d_near = (abs(abs(a["nx"]) - abs(b["nx"])) + abs(abs(a["ny"]) - abs(b["ny"])))
    return w_xy * d_xy + w_dia * d_dia + w_near * d_near

def _cost_matrix(A, B):
    n = len(A)
    m = len(B)
    C = np.empty((n, m))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            C[i, j] = _match_cost(a, b)
    return C

# ────────── plot ───────────────────────────────────────────────────────────
def make_plot(Hc: Sequence[dict], Hv: Sequence[dict],
              rows: Sequence[int], cols: Sequence[int],
              plot_path: Path, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    # CAD – blue
    ax.scatter([h["x"] for h in Hc],
               [h["y"] for h in Hc],
               s=20, c="blue", label="CAD (эталон)")
    # Vision – red
    ax.scatter([h["x"] for h in Hv],
               [h["y"] for h in Hv],
               s=20, c="red", label="Vision (факт)")

    # Соединяем пары и подписываем Δ
    for i, j in zip(rows, cols):
        a, b = Hc[i], Hv[j]
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                c="grey", lw=0.5, alpha=0.6)
        mid_x = (a["x"] + b["x"]) / 2
        mid_y = (a["y"] + b["y"]) / 2
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        ax.text(mid_x, mid_y,
                f"{dx:+.1f}/{dy:+.1f}",
                fontsize=6, color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("posX, мм")
    ax.set_ylabel("posY, мм")
    ax.set_aspect("equal")
    ax.set_title("Отклонения отверстий (CAD vs Vision)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ────────── csv helper ─────────────────────────────────────────────────────
def save_deltas_csv(
    Hc: Sequence[dict],
    Hv: Sequence[dict],
    rows: Sequence[int],
    cols: Sequence[int],
    csv_path: Path,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "hole_id", "dx_mm", "dy_mm", "dØ_mm", "d_nearX_mm", "d_nearY_mm",
            "posX_cad_mm", "posY_cad_mm", "posX_vis_mm", "posY_vis_mm",
        ])
        for i, j in zip(rows, cols):
            a, b = Hc[i], Hv[j]
            wr.writerow([
                a["id"],
                b["x"] - a["x"],
                b["y"] - a["y"],
                b["dia"] - a["dia"],
                abs(b["nx"]) - abs(a["nx"]),
                abs(b["ny"]) - abs(a["ny"]),
                a["x"], a["y"],
                b["x"], b["y"],
            ])

# ────────── mirror check ───────────────────────────────────────────────────
def check_mirror(Hc, Hv, rows, cols, use_inverted_x: bool = False):  # MOD: pass use_inverted_x
    """Проверяет, может ли деталь быть зеркальной."""
    mirror_count = 0
    for i, j in zip(rows, cols):
        a, b = Hc[i], Hv[j]
        if (a["nx"] * b["nx"] < 0) or (a["ny"] * b["ny"] < 0):
            mirror_count += 1
    if mirror_count > len(rows) / 2 or use_inverted_x:  # NEW: if invert_x chosen, consider mirrored
        return True
    return False

# ────────── compare ────────────────────────────────────────────────────────
def compare(
    p1: Path,
    p2: Path,
    out_p: Path,
    plot_file: Path | None = None,
    csv_file: Path | None = None,
    dpi: int = 150,
    force_invert_y: bool = False,
):
    d1, d2 = (json.loads(p.read_text('utf8')) for p in (p1, p2))

    is_v1 = "rectangle" in d1 and "length" not in d1
    is_v2 = "rectangle" in d2 and "length" not in d2
    if is_v1 == is_v2:
        raise RuntimeError(
            "Нужен один CAD-файл (length/width) и один Vision-файл (rectangle)."
        )

    CAD, VSN = (d1, d2) if not is_v1 else (d2, d1)
    Hc = load_holes(CAD, is_vision=False)

    # NEW: 4 variants for Vision
    Hv_normal = load_holes(VSN, is_vision=True, invert_y=False, invert_x=False)
    Hv_inverted_y = load_holes(VSN, is_vision=True, invert_y=True, invert_x=False)
    Hv_inverted_x = load_holes(VSN, is_vision=True, invert_y=False, invert_x=True)
    Hv_inverted_both = load_holes(VSN, is_vision=True, invert_y=True, invert_x=True)

    # Compute costs for all 4
    costs = {}
    costs['normal'] = (_cost_matrix(Hc, Hv_normal), Hv_normal)
    costs['invert_y'] = (_cost_matrix(Hc, Hv_inverted_y), Hv_inverted_y)
    costs['invert_x'] = (_cost_matrix(Hc, Hv_inverted_x), Hv_inverted_x)
    costs['both'] = (_cost_matrix(Hc, Hv_inverted_both), Hv_inverted_both)

    # Choose the variant with minimal total cost
    min_variant = min(costs, key=lambda k: costs[k][0][linear_sum_assignment(costs[k][0])[0], linear_sum_assignment(costs[k][0])[1]].sum())
    cost_matrix, Hv = costs[min_variant]
    rows, cols = linear_sum_assignment(cost_matrix)
    matched = len(rows)
    if matched < len(Hc):
        raise RuntimeError(f"Недостающие отверстия в Vision: {len(Hc) - matched}")
    extra = len(Hv) - matched
    if extra > 1:
        raise RuntimeError(f"Слишком много лишних отверстий в Vision: {extra}")
    total_cost = cost_matrix[rows, cols].sum()

    use_inverted_y = 'invert_y' in min_variant or 'both' in min_variant
    use_inverted_x = 'invert_x' in min_variant or 'both' in min_variant

    if force_invert_y:
        use_inverted_y = True
        # Override if forced, but respect x
        if use_inverted_x:
            Hv = Hv_inverted_both
        else:
            Hv = Hv_inverted_y
        rows, cols = linear_sum_assignment(_cost_matrix(Hc, Hv))

    # Размеры
    sc, lc = dims(CAD)
    sv, lv = dims(VSN)
    size_ok = abs(sc - sv) < 1e-3 and abs(lc - lv) < 1e-3

    # Отчёт
    out = []
    out.append("=== СРАВНЕНИЕ ФАЙЛОВ ===\n")
    if use_inverted_y:
        out.append("Примечание: использована инверсия оси Y для Vision-данных\n")
    if use_inverted_x:  # NEW
        out.append("Примечание: использована инверсия оси X для Vision-данных\n")
    if check_mirror(Hc, Hv, rows, cols, use_inverted_x):  # MOD: pass use_inverted_x
        out.append("Внимание: возможно, деталь зеркальная!\n")
    if extra > 0:
        out.append(f"Внимание: обнаружены лишние отверстия в Vision ({extra})! Они игнорируются.\n")
    out.append(f"Размеры (без поворота): {sc}×{lc}  vs  {sv}×{lv}  → {'ДА' if size_ok else 'НЕТ'}\n")
    out.append(f"Отверстий: {len(Hc)} (совпало {matched}, лишних {extra})\n")

    hdr = ("ID CAD/Vsn | ΔX | ΔY | ΔØ | ΔnX | ΔnY | "
           "posX posY (CAD) | posX posY (Vsn)")
    out.append(hdr)
    out.append('-' * len(hdr))

    for i, j in sorted(zip(rows, cols), key=lambda p: Hc[p[0]]["id"]):
        a, b = Hc[i], Hv[j]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        dd = b["dia"] - a["dia"]
        dnx = abs(b["nx"]) - abs(a["nx"])
        dny = abs(b["ny"]) - abs(a["ny"])
        out.append(
            f"{a['id']:>3}/{b['id']:<3} | "
            f"{dx:+6.2f} | {dy:+7.2f} | {dd:+5.3f} | "
            f"{dnx:+6.2f} | {dny:+6.2f} | "
            f"{a['x']:+7.1f} {a['y']:+7.1f} | {b['x']:+7.1f} {b['y']:+7.1f}"
        )

    out_p.write_text("\n".join(out), encoding="utf-8")
    print("Отчёт →", out_p)

    if plot_file:
        make_plot(Hc, Hv, rows, cols, plot_file, dpi=dpi)
        print("График →", plot_file)

    if csv_file:
        save_deltas_csv(Hc, Hv, rows, cols, csv_file)

# ────────── CLI ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""
            Сравнение CAD-эталона и Vision-замера.
            Исправляет инверсию координат у Vision и умеет строить график отклонений.

            Пример:
              python compare.py cad.json vision.json \\
                     -o diff.txt --plot deviations.png --dpi 200 --force-invert-y
        """))
    ap.add_argument("file1", type=Path, help="CAD или Vision файл")
    ap.add_argument("file2", type=Path, help="второй файл другого типа")
    ap.add_argument("-o", "--output", type=Path, default=Path("result.txt"),
                    help="TXT-отчёт (по умолчанию result.txt)")
    ap.add_argument("--plot", type=Path, help="PNG/SVG файл с графиком")
    ap.add_argument("--dpi", type=int, default=150, help="dpi картинки (по умолчанию 150)")
    ap.add_argument("--force-invert-y", action="store_true",
                    help="Принудительно инвертировать posY для Vision")
    args = ap.parse_args()

    compare(args.file1, args.file2, args.output, args.plot, dpi=args.dpi,
            force_invert_y=args.force_invert_y)

if __name__ == "__main__":
    main()
